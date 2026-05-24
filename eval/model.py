"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    time_context_feats: torch.Tensor  # [B, 2], Beijing hour and weekday ids
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    seq_time_deltas: dict   # {domain: tensor [B, L]} log1p(dt/3600)
    seq_time_gaps: dict     # {domain: tensor [B, L]} log1p(adjacent gap minutes)
    seq_time_calendars: dict  # {domain: tensor [B, L, 4]} Beijing hour/weekday sin-cos


def _parse_fid_list(value: Optional[Union[str, List[int], Tuple[int, ...]]]) -> List[int]:
    """Parse a CLI-style comma-separated fid list into integers."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [int(x.strip()) for x in value.split(',') if x.strip()]
    return [int(x) for x in value]


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class HashEmbedding(nn.Module):
    """Multi-hash embedding fallback for skipped high-cardinality features."""

    HASH_PRIMES = [999983, 999979, 999961]

    def __init__(
        self,
        num_buckets: int = 100000,
        emb_dim: int = 64,
        num_hashes: int = 2,
    ) -> None:
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.num_hashes = int(num_hashes)
        self.embs = nn.ModuleList([
            nn.Embedding(self.num_buckets, emb_dim, padding_idx=0)
            for _ in range(self.num_hashes)
        ])

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pad_mask = ids == 0
        result = self.embs[0](ids.abs() % (self.num_buckets - 1) + 1)
        for i in range(1, self.num_hashes):
            h = (ids.abs() * self.HASH_PRIMES[i - 1]) % (self.num_buckets - 1) + 1
            result = result + self.embs[i](h)
        result = result / self.num_hashes
        return result.masked_fill(pad_mask.unsqueeze(-1), 0.0)


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: additive float mask (Lq, Lk), -inf means do not attend
            # Convert to bool: positions that are not -inf are True
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # Per-token FFN (shared parameters) — used by both 'full' and 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model

        global_info_dim = (num_ns + 1) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


class TimeContextEncoder(nn.Module):
    """Aggregates per-domain time-bucket embeddings into one time context."""

    def __init__(
        self,
        dim: int,
        hidden_mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(dim * hidden_mult))
        self.dim = dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        seq_time_buckets_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
        time_embedding: nn.Embedding,
    ) -> torch.Tensor:
        """Return z_t with shape (B, D).

        seq_masks_list uses the project convention: True means padding.
        Padding bucket 0 and empty domains do not contribute to the mean.
        """
        if not seq_time_buckets_list:
            raise ValueError("TimeContextEncoder requires at least one sequence domain")

        domain_summaries = []
        domain_valids = []
        for bucket_ids, padding_mask in zip(seq_time_buckets_list, seq_masks_list):
            if bucket_ids.shape != padding_mask.shape:
                raise RuntimeError(
                    f"time bucket/mask shape mismatch: buckets={tuple(bucket_ids.shape)}, "
                    f"mask={tuple(padding_mask.shape)}")
            time_emb = time_embedding(bucket_ids.long())  # (B, L, D)
            valid = (~padding_mask) & (bucket_ids.long() != 0)
            valid_f = valid.unsqueeze(-1).to(time_emb.dtype)
            count = valid_f.sum(dim=1).clamp(min=1.0)
            summary = (time_emb * valid_f).sum(dim=1) / count
            domain_summaries.append(summary)
            domain_valids.append(valid.any(dim=1))

        stacked = torch.stack(domain_summaries, dim=1)  # (B, S, D)
        valid_domains = torch.stack(domain_valids, dim=1)  # (B, S)
        valid_f = valid_domains.unsqueeze(-1).to(stacked.dtype)
        domain_count = valid_f.sum(dim=1).clamp(min=1.0)
        z_raw = (stacked * valid_f).sum(dim=1) / domain_count
        return self.net(z_raw)


class TemporalQueryGenerator(nn.Module):
    """Instance-conditioned query generator with optional time context."""

    def __init__(
        self,
        d_model: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        alpha: float = 0.1,
        use_base_query: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.alpha = float(alpha)
        self.use_base_query = bool(use_base_query)

        global_dim = (2 + num_sequences) * d_model
        hidden_dim = max(1, d_model * hidden_mult)
        self.norm = nn.LayerNorm(global_dim)
        self.mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sequences * num_queries * d_model),
        )
        # Start close to a learned base-query-only model for stable ablations.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        if self.use_base_query:
            self.base_query = nn.Parameter(
                torch.empty(num_sequences, num_queries, d_model))
            nn.init.normal_(self.base_query, mean=0.0, std=0.02)
        else:
            self.register_parameter('base_query', None)

    @staticmethod
    def _masked_mean(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if x.shape[:2] != padding_mask.shape:
            raise RuntimeError(
                f"seq token/mask shape mismatch: tokens={tuple(x.shape)}, "
                f"mask={tuple(padding_mask.shape)}")
        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
        count = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / count

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        z_t: Optional[torch.Tensor] = None,
    ) -> list:
        """Generate q_tokens_list compatible with MultiSeqHyFormerBlock."""
        B = ns_tokens.shape[0]
        if len(seq_tokens_list) != self.num_sequences:
            raise RuntimeError(
                f"TemporalQueryGenerator expected {self.num_sequences} domains, "
                f"got {len(seq_tokens_list)}")

        s_ns = ns_tokens.mean(dim=1)  # (B, D)
        seq_summaries = [
            self._masked_mean(seq_tokens_list[i], seq_padding_masks[i])
            for i in range(self.num_sequences)
        ]
        if z_t is None:
            z_t = torch.zeros_like(s_ns)
        elif z_t.shape != s_ns.shape:
            raise RuntimeError(
                f"z_t shape mismatch: expected {tuple(s_ns.shape)}, got {tuple(z_t.shape)}")

        g = torch.cat([s_ns] + seq_summaries + [z_t], dim=-1)
        q_delta = self.mlp(self.norm(g))
        q_delta = q_delta.view(
            B, self.num_sequences, self.num_queries, self.d_model)

        if self.base_query is None:
            q = self.alpha * q_delta
        else:
            q = self.base_query.unsqueeze(0) + self.alpha * q_delta
        return [q[:, i, :, :] for i in range(self.num_sequences)]


class TokenSE(nn.Module):
    """Squeeze-excitation recalibration for query tokens using Q+NS context."""

    def __init__(
        self,
        dim: int,
        hidden_mult: float = 0.25,
        dropout: float = 0.0,
        gamma_init: float = 0.01,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(dim * hidden_mult))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, all_q: torch.Tensor, curr_ns: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([all_q, curr_ns], dim=1)  # (B, T, D)
        se = tokens.mean(dim=1)
        gate = torch.sigmoid(self.net(se)).unsqueeze(1)
        return all_q + self.gamma * all_q * gate


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════



class SeqTimeFiLM(nn.Module):
    """Continuous sequence-time modulation without heavy time embeddings."""

    def __init__(
        self,
        d_model: int,
        time_feat_dim: int = 12,
        hidden_mult: int = 2,
        dropout: float = 0.02,
        gate_init: float = -2.5,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, d_model * hidden_mult)
        self.net = nn.Sequential(
            nn.LayerNorm(time_feat_dim),
            nn.Linear(time_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        seq_tokens: torch.Tensor,
        time_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.net(time_features.to(seq_tokens.dtype)).chunk(2, dim=-1)
        valid = (~padding_mask).unsqueeze(-1).to(seq_tokens.dtype)
        g = torch.sigmoid(self.gate)
        return seq_tokens * (1.0 + g * gamma * valid) + g * beta * valid


class DualQueryTargetTimeGenerator(nn.Module):
    """Generate semantic long-interest and target-time queries per domain."""

    def __init__(
        self,
        d_model: int,
        num_sequences: int,
        profile_dim: int = 19,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_queries = 2
        self.num_sequences = num_sequences
        self.alpha = float(alpha)
        hidden_dim = max(1, d_model * hidden_mult)
        self.base_query = nn.Parameter(torch.empty(num_sequences, 2, d_model))
        nn.init.normal_(self.base_query, mean=0.0, std=0.02)
        self.long_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(3 * d_model),
                nn.Linear(3 * d_model, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
            )
            for _ in range(num_sequences)
        ])
        self.target_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(5 * d_model + profile_dim),
                nn.Linear(5 * d_model + profile_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
            )
            for _ in range(num_sequences)
        ])
        for mlp in list(self.long_mlps) + list(self.target_mlps):
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    @staticmethod
    def _masked_mean(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
        count = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / count

    @staticmethod
    def _fixed_recent_pool(
        seq_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        time_delta: torch.Tensor,
    ) -> torch.Tensor:
        scores = -0.12 * time_delta.to(seq_tokens.dtype)
        scores = scores.masked_fill(padding_mask, -1e9)
        all_pad = padding_mask.all(dim=1, keepdim=True)
        weights = F.softmax(scores, dim=1).masked_fill(all_pad, 0.0)
        return (seq_tokens * weights.unsqueeze(-1)).sum(dim=1)

    def forward(
        self,
        item_repr: torch.Tensor,
        ns_tokens: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
        seq_time_deltas_list: List[torch.Tensor],
        seq_time_profiles_list: List[torch.Tensor],
        z_t: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        B = ns_tokens.shape[0]
        ns_summary = ns_tokens.mean(dim=1)
        if z_t is None:
            z_t = torch.zeros_like(ns_summary)
        out = []
        for i in range(self.num_sequences):
            seq_mean = self._masked_mean(seq_tokens_list[i], seq_masks_list[i])
            recent_pool = self._fixed_recent_pool(
                seq_tokens_list[i], seq_masks_list[i], seq_time_deltas_list[i])
            seq_recent = 0.6 * seq_mean + 0.4 * recent_pool
            q_long_delta = self.long_mlps[i](torch.cat([ns_summary, seq_mean, z_t], dim=-1))
            q_target_delta = self.target_mlps[i](torch.cat([
                item_repr, ns_summary, seq_mean, seq_recent, z_t,
                seq_time_profiles_list[i].to(seq_mean.dtype),
            ], dim=-1))
            base = self.base_query[i].unsqueeze(0).expand(B, -1, -1)
            q_long = base[:, 0, :] + self.alpha * q_long_delta
            q_target = base[:, 1, :] + self.alpha * q_target_delta
            out.append(torch.stack([q_long, q_target], dim=1))
        return out


class TargetTimeDINReader(nn.Module):
    """DIN-style reader that only augments the target-time query."""

    def __init__(
        self,
        d_model: int,
        num_sequences: int,
        time_feat_dim: int = 12,
        hidden_mult: int = 3,
        dropout: float = 0.02,
        query_gate_init: float = -2.5,
        output_gate_init: float = -2.8,
        history_dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        hidden_dim = max(1, d_model * hidden_mult)
        self.score_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(5 * d_model + time_feat_dim),
                nn.Linear(5 * d_model + time_feat_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_sequences)
        ])
        self.query_gates = nn.Parameter(torch.full((num_sequences,), float(query_gate_init)))
        self.history_dropout = nn.Dropout(history_dropout)
        self.output_gate = nn.Parameter(torch.tensor(float(output_gate_init)))
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        nn.init.zeros_(self.output_mlp[-1].weight)
        nn.init.zeros_(self.output_mlp[-1].bias)

    def forward(
        self,
        q_tokens_list: List[torch.Tensor],
        item_repr: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
        seq_time_features_list: List[torch.Tensor],
        seq_time_deltas_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        new_qs = []
        din_contexts = []
        for i, q in enumerate(q_tokens_list):
            target_q = q[:, 1, :]
            seq_tokens = seq_tokens_list[i]
            if self.training:
                seq_tokens = self.history_dropout(seq_tokens)
            B, L, D = seq_tokens.shape
            q_exp = target_q.unsqueeze(1).expand(-1, L, -1)
            item_exp = item_repr.unsqueeze(1).expand(-1, L, -1)
            score_in = torch.cat([
                q_exp, seq_tokens, q_exp - seq_tokens, q_exp * seq_tokens,
                item_exp, seq_time_features_list[i].to(seq_tokens.dtype),
            ], dim=-1)
            scores = self.score_mlps[i](score_in).squeeze(-1)
            scores = scores - 0.03 * seq_time_deltas_list[i].to(scores.dtype)
            scores = scores.masked_fill(seq_masks_list[i], -1e9)
            all_pad = seq_masks_list[i].all(dim=1, keepdim=True)
            weights = F.softmax(scores, dim=1).masked_fill(all_pad, 0.0)
            ctx = (seq_tokens * weights.unsqueeze(-1)).sum(dim=1)
            q_new = q.clone()
            q_new[:, 1, :] = q_new[:, 1, :] + torch.sigmoid(self.query_gates[i]) * ctx
            new_qs.append(q_new)
            din_contexts.append(ctx)
        din_pool = torch.stack(din_contexts, dim=1).mean(dim=1)
        return new_qs, din_pool

    def fuse_output(
        self,
        output: torch.Tensor,
        din_pool: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.output_mlp(torch.cat([output, din_pool, item_repr], dim=-1))
        return output + torch.sigmoid(self.output_gate) * delta


class DCNNSCross(nn.Module):
    """DCN-v2 cross network over NS tokens with token-aware gates."""

    def __init__(
        self,
        num_ns: int,
        d_model: int,
        num_layers: int = 2,
        low_rank: int = 64,
        dropout: float = 0.05,
        token_gate_inits: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ns = num_ns
        self.d_model = d_model
        self.num_layers = int(num_layers)
        in_dim = num_ns * d_model
        rank = max(1, int(low_rank))
        self.U = nn.ParameterList([
            nn.Parameter(torch.empty(in_dim, rank))
            for _ in range(self.num_layers)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(torch.empty(rank, in_dim))
            for _ in range(self.num_layers)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(in_dim))
            for _ in range(self.num_layers)
        ])
        for u, v in zip(self.U, self.V):
            nn.init.xavier_normal_(u)
            nn.init.xavier_normal_(v)
        self.drop = nn.Dropout(dropout)
        if token_gate_inits is None:
            token_gate_inits = [-3.5] * num_ns
        if len(token_gate_inits) != num_ns:
            raise ValueError(
                f"token_gate_inits length {len(token_gate_inits)} != num_ns {num_ns}")
        self.token_gate = nn.Parameter(torch.tensor(token_gate_inits, dtype=torch.float32).view(1, num_ns, 1))
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, ns_tokens: torch.Tensor) -> torch.Tensor:
        B = ns_tokens.shape[0]
        x0 = ns_tokens.reshape(B, -1)
        xl = x0
        for i in range(self.num_layers):
            vx = xl @ self.V[i].t()
            uvx = vx @ self.U[i].t()
            xl = x0 * (uvx + self.bias[i]) + xl
            xl = self.drop(xl)
        cross = self.out_norm(xl.view(B, self.num_ns, self.d_model))
        gate = torch.sigmoid(self.token_gate.to(ns_tokens.dtype))
        return gate * cross + (1.0 - gate) * ns_tokens


class NSOutputFusionResidual(nn.Module):
    """Residual output fusion from final NS tokens."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.02,
        gate_init: float = -3.2,
    ) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.mlp = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, output: torch.Tensor, ns_tokens: torch.Tensor) -> torch.Tensor:
        ns_mean = ns_tokens.mean(dim=1)
        delta = self.mlp(torch.cat([output, ns_mean], dim=-1))
        return output + torch.sigmoid(self.gate) * delta


class StrongTimeResidual(nn.Module):
    """Small logits-side residual conditioned on final representation and time context."""

    def __init__(
        self,
        d_model: int,
        action_num: int,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        gamma_init: float = 0.03,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(d_model * hidden_mult))
        self.gamma_time = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_num),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, output: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        z_t = z_t.to(dtype=output.dtype, device=output.device)
        time_input = torch.cat([output, z_t, output * z_t, (output - z_t).abs()], dim=-1)
        return self.gamma_time.to(output.dtype) * self.mlp(time_input)


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════

class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full',
        use_gamma_writeback: bool = False,
        gamma_q_init: float = 0.05,
        gamma_ns_init: float = 0.03,
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns
        self.use_gamma_writeback = bool(use_gamma_writeback)
        if self.use_gamma_writeback:
            gq = min(max(float(gamma_q_init), 1e-4), 1.0 - 1e-4)
            gns = min(max(float(gamma_ns_init), 1e-4), 1.0 - 1e-4)
            self.gamma_q_logit = nn.Parameter(
                torch.tensor(math.log(gq / (1.0 - gq)), dtype=torch.float32))
            self.gamma_ns_logit = nn.Parameter(
                torch.tensor(math.log(gns / (1.0 - gns)), dtype=torch.float32))

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS. Optional gamma write-back
        # regularizes only the RankMixer write-back, preserving sequence decoding.
        next_q_list = []
        offset = 0
        if self.use_gamma_writeback:
            gamma_q = torch.sigmoid(self.gamma_q_logit).to(boosted.dtype)
            gamma_ns = torch.sigmoid(self.gamma_ns_logit).to(boosted.dtype)
        for i in range(S):
            raw_q = boosted[:, offset:offset + Nq, :]
            if self.use_gamma_writeback:
                next_q_list.append(decoded_qs[i] + gamma_q * (raw_q - decoded_qs[i]))
            else:
                next_q_list.append(raw_q)
            offset += Nq
        raw_ns = boosted[:, offset:, :]
        if self.use_gamma_writeback:
            next_ns = ns_tokens + gamma_ns * (raw_ns - ns_tokens)
        else:
            next_ns = raw_ns

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0,
                 hash_bucket_size: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold
        self.hash_bucket_size = int(hash_bucket_size)

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        self._hash_embs = nn.ModuleDict()
        if self.hash_bucket_size > 0:
            for i, (vs, _, _) in enumerate(feature_specs):
                if self._emb_index[i] == -1 and int(vs) > 0:
                    self._hash_embs[str(i)] = HashEmbedding(self.hash_bucket_size, emb_dim)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    hash_key = str(fid_idx)
                    if hash_key in self._hash_embs:
                        emb_layer = self._hash_embs[hash_key]
                        if length == 1:
                            fid_emb = emb_layer(int_feats[:, offset].long())
                        else:
                            vals = int_feats[:, offset:offset + length].long()
                            emb_all = emb_layer(vals)
                            mask = (vals != 0).float().unsqueeze(-1)
                            count = mask.sum(dim=1).clamp(min=1)
                            fid_emb = (emb_all * mask).sum(dim=1) / count
                    else:
                        fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
        hash_bucket_size: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold
        self.hash_bucket_size = int(hash_bucket_size)

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        self._hash_embs = nn.ModuleDict()
        if self.hash_bucket_size > 0:
            for i, (vs, _, _) in enumerate(feature_specs):
                if self._emb_index[i] == -1 and int(vs) > 0:
                    self._hash_embs[str(i)] = HashEmbedding(self.hash_bucket_size, emb_dim)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    hash_key = str(fid_idx)
                    if hash_key in self._hash_embs:
                        emb_layer = self._hash_embs[hash_key]
                        if length == 1:
                            fid_emb = emb_layer(int_feats[:, offset].long())
                        else:
                            vals = int_feats[:, offset:offset + length].long()
                            emb_all = emb_layer(vals)
                            mask = (vals != 0).float().unsqueeze(-1)
                            count = mask.sum(dim=1).clamp(min=1)
                            fid_emb = (emb_all * mask).sum(dim=1) / count
                    else:
                        fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class QuantileTrendEncoder(nn.Module):
    """Encodes rank/quantile vectors as one dense NS token."""

    def __init__(
        self,
        d_model: int,
        in_channels: int = 3,
        vector_dim: int = 10,
        mid_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.vector_dim = vector_dim
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=mid_dim,
            kernel_size=3,
            padding=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=mid_dim,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, 3, 10), rank/quantile vectors for fids 89/90/91.
        """
        if x.dim() != 3 or x.shape[1] != self.in_channels or x.shape[2] != self.vector_dim:
            raise ValueError(
                f"QuantileTrendEncoder expects shape (B, {self.in_channels}, "
                f"{self.vector_dim}), got {tuple(x.shape)}")
        h = self.conv1(x.float())
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.pool(h).squeeze(-1)
        return self.out_norm(h)


class DenseGroupProjector(nn.Module):
    """Projects user_dense features into semantically grouped NS tokens.

    Default layout follows the competition baseline notes:
    - emb group: fid 61 + fid 87, L2-normalized then Linear+LN.
    - stat group: fid 62-66, log1p+clamp counts plus Beijing hour/weekday embeddings.
    - quantile group: fid 89-91 rank vectors encoded by QuantileTrendEncoder.
    """

    def __init__(
        self,
        user_dense_feature_specs: List[Tuple[int, int, int]],
        d_model: int,
        dense_emb_group_fids: Optional[Union[str, List[int], Tuple[int, ...]]] = None,
        dense_stat_group_fids: Optional[Union[str, List[int], Tuple[int, ...]]] = None,
        dense_quantile_group_fids: Optional[Union[str, List[int], Tuple[int, ...]]] = None,
        stat_clamp_value: float = 18_000_000.0,
        use_quantile_trend_encoder: bool = True,
        version: str = 'v7',
        use_time_context_features: bool = True,
        time_context_emb_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_specs = [
            (int(fid), int(offset), int(length))
            for fid, offset, length in user_dense_feature_specs
            if int(length) > 0
        ]
        self.stat_clamp_value = float(stat_clamp_value)
        self.use_quantile_trend_encoder = bool(use_quantile_trend_encoder)
        self.version = version
        self.use_time_context_features = bool(use_time_context_features)
        self.time_context_emb_dim = int(time_context_emb_dim)

        emb_fids = set(_parse_fid_list(
            '61,87' if dense_emb_group_fids is None else dense_emb_group_fids))
        stat_fids = set(_parse_fid_list(
            '62,63,64,65,66' if dense_stat_group_fids is None else dense_stat_group_fids))
        quantile_fids = set(_parse_fid_list(
            '89,90,91' if dense_quantile_group_fids is None else dense_quantile_group_fids))

        self.emb_group_specs = self._filter_specs(emb_fids, 'emb')
        self.stat_group_specs = self._filter_specs(stat_fids, 'stat')
        self.quantile_group_specs = self._filter_specs(quantile_fids, 'quantile')

        emb_indices = self._indices_from_specs(self.emb_group_specs)
        stat_indices = self._indices_from_specs(self.stat_group_specs)
        quantile_indices = self._indices_from_specs(self.quantile_group_specs)
        self.register_buffer('emb_group_indices', torch.tensor(emb_indices, dtype=torch.long), persistent=False)
        self.register_buffer('stat_group_indices', torch.tensor(stat_indices, dtype=torch.long), persistent=False)
        self.register_buffer('quantile_group_indices', torch.tensor(quantile_indices, dtype=torch.long), persistent=False)

        self.has_emb_group = self.emb_group_indices.numel() > 0
        if self.has_emb_group:
            self.emb_proj = nn.Sequential(
                nn.Linear(int(self.emb_group_indices.numel()), d_model),
                nn.LayerNorm(d_model),
            )

        if self.use_time_context_features:
            self.hour_embedding = nn.Embedding(24, self.time_context_emb_dim)
            self.weekday_embedding = nn.Embedding(7, self.time_context_emb_dim)
        time_context_dim = 2 * self.time_context_emb_dim if self.use_time_context_features else 0

        self.has_stat_group = self.stat_group_indices.numel() > 0 or self.use_time_context_features
        if self.has_stat_group:
            self.stat_proj = nn.Sequential(
                nn.Linear(int(self.stat_group_indices.numel()) + time_context_dim, d_model),
                nn.LayerNorm(d_model),
            )

        quantile_lengths = [length for _, _, length in self.quantile_group_specs]
        self.has_quantile_group = bool(quantile_lengths)
        self.quantile_equal_length = (
            self.has_quantile_group and len(set(quantile_lengths)) == 1
        )
        if self.has_quantile_group and self.use_quantile_trend_encoder:
            if len(self.quantile_group_specs) != 3 or set(quantile_lengths) != {10}:
                raise ValueError(
                    "v7 QuantileTrendEncoder expects exactly fids 89/90/91 "
                    f"with length 10 each, got specs={self.quantile_group_specs}")
            self.quantile_encoder = QuantileTrendEncoder(
                d_model=d_model,
                in_channels=3,
                vector_dim=10,
                dropout=dropout,
            )
        elif self.has_quantile_group:
            self.quantile_proj = nn.Sequential(
                nn.Linear(int(self.quantile_group_indices.numel()), d_model),
                nn.LayerNorm(d_model),
            )

        self.emb_token_index = 0 if self.has_emb_group else -1
        self.stat_token_index = int(self.has_emb_group) if self.has_stat_group else -1
        self.quantile_token_index = (
            int(self.has_emb_group) + int(self.has_stat_group)
            if self.has_quantile_group else -1
        )
        self.num_tokens = (
            int(self.has_emb_group)
            + int(self.has_stat_group)
            + int(self.has_quantile_group)
        )

        logging.info(
            "DenseGroupProjector: emb_fids=%s dim=%d, stat_fids=%s dim=%d, "
            "quantile_fids=%s dim=%d, time_context=%s, tokens=%d",
            [fid for fid, _, _ in self.emb_group_specs],
            int(self.emb_group_indices.numel()),
            [fid for fid, _, _ in self.stat_group_specs],
            int(self.stat_group_indices.numel()),
            [fid for fid, _, _ in self.quantile_group_specs],
            int(self.quantile_group_indices.numel()),
            self.use_time_context_features,
            self.num_tokens,
        )

    def _filter_specs(self, fids: set, group_name: str) -> List[Tuple[int, int, int]]:
        specs = [spec for spec in self.feature_specs if spec[0] in fids]
        found = {fid for fid, _, _ in specs}
        missing = sorted(fids - found)
        if missing:
            raise ValueError(
                f"DenseGroupProjector {self.version} missing user_dense fid(s) "
                f"for {group_name} group: {missing}. Available fids: "
                f"{[fid for fid, _, _ in self.feature_specs]}")
        return specs

    @staticmethod
    def _indices_from_specs(specs: List[Tuple[int, int, int]]) -> List[int]:
        indices: List[int] = []
        for _, offset, length in specs:
            indices.extend(range(offset, offset + length))
        return indices

    def _select(self, dense_feats: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return dense_feats.new_zeros(dense_feats.shape[0], 0)
        return dense_feats.index_select(1, indices.to(dense_feats.device)).float()

    def _select_l2_normalized_specs(
        self,
        dense_feats: torch.Tensor,
        specs: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        parts = []
        for _, offset, length in specs:
            part = dense_feats[:, offset:offset + length].float()
            parts.append(F.normalize(part, p=2, dim=-1, eps=1e-6))
        return torch.cat(parts, dim=-1)

    def _embed_time_context(self, time_context_feats: torch.Tensor) -> torch.Tensor:
        if not self.use_time_context_features:
            return time_context_feats.new_zeros(time_context_feats.shape[0], 0).float()
        hour = time_context_feats[:, 0].long().clamp(0, 23)
        weekday = time_context_feats[:, 1].long().clamp(0, 6)
        return torch.cat([
            self.hour_embedding(hour),
            self.weekday_embedding(weekday),
        ], dim=-1)

    def _select_quantile_vectors(self, dense_feats: torch.Tensor) -> torch.Tensor:
        vectors = []
        for _, offset, length in self.quantile_group_specs:
            vectors.append(dense_feats[:, offset:offset + length].float())
        return torch.stack(vectors, dim=1)

    def forward(
        self,
        dense_feats: torch.Tensor,
        time_context_feats: torch.Tensor,
    ) -> torch.Tensor:
        tokens = []
        if self.has_emb_group:
            emb_x = self._select_l2_normalized_specs(
                dense_feats, self.emb_group_specs)
            tokens.append(F.silu(self.emb_proj(emb_x)).unsqueeze(1))

        if self.has_stat_group:
            stat_x = self._select(dense_feats, self.stat_group_indices)
            if stat_x.numel() > 0:
                stat_x = stat_x.clamp(min=0.0, max=self.stat_clamp_value)
                stat_x = torch.log1p(stat_x) / math.log1p(self.stat_clamp_value)
            time_x = self._embed_time_context(time_context_feats)
            stat_x = torch.cat([stat_x, time_x], dim=-1)
            tokens.append(F.silu(self.stat_proj(stat_x)).unsqueeze(1))

        if self.has_quantile_group:
            if self.use_quantile_trend_encoder:
                quantile_x = self._select_quantile_vectors(dense_feats)
                quantile_tok = self.quantile_encoder(quantile_x)
            else:
                quantile_x = self._select(dense_feats, self.quantile_group_indices)
                quantile_tok = self.quantile_proj(quantile_x)
            tokens.append(F.silu(quantile_tok).unsqueeze(1))

        if not tokens:
            return dense_feats.new_zeros(dense_feats.shape[0], 0, 0)
        out = torch.cat(tokens, dim=1)
        if out.shape[1] != self.num_tokens:
            raise RuntimeError(
                f"DenseGroupProjector token count mismatch: expected "
                f"{self.num_tokens}, got {out.shape[1]}")
        return out


class SparseDensePairResidual(nn.Module):
    """Gated residual for paired user sparse ids and dense values."""

    def __init__(
        self,
        pair_specs: List[Tuple[int, int, int, int, int]],
        emb_dim: int,
        d_model: int,
        gate_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.pair_specs = pair_specs
        self.embs = nn.ModuleList([
            nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0)
            for _, vs, _, _, _ in pair_specs
        ])
        self.proj = nn.Sequential(
            nn.Linear(max(1, len(pair_specs)) * emb_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        user_int_feats: torch.Tensor,
        user_dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        if not self.pair_specs:
            return user_dense_feats.new_zeros(user_dense_feats.shape[0], 0)
        pooled = []
        for emb, (_, _, int_offset, dense_offset, length) in zip(self.embs, self.pair_specs):
            ids = user_int_feats[:, int_offset:int_offset + length].long()
            values = user_dense_feats[:, dense_offset:dense_offset + length].float()
            weights = torch.log1p(values.clamp_min(0.0))
            weights = weights / weights.detach().amax(dim=1, keepdim=True).clamp(min=1.0)
            mask = (ids != 0).to(values.dtype)
            emb_all = emb(ids)
            count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled.append((emb_all * weights.unsqueeze(-1) * mask.unsqueeze(-1)).sum(dim=1) / count)
        return torch.sigmoid(self.gate) * F.silu(self.proj(torch.cat(pooled, dim=-1)))


class ItemAwareQueryAdapter(nn.Module):
    """Small candidate-item adapter added to baseline query tokens."""

    def __init__(
        self,
        d_model: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        gate_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        hidden_dim = d_model * hidden_mult
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(3 * d_model),
                nn.Linear(3 * d_model, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_queries * d_model),
            )
            for _ in range(num_sequences)
        ])
        for mlp in self.mlps:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
        self.gates = nn.Parameter(torch.full((num_sequences,), float(gate_init)))

    @staticmethod
    def _masked_mean(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
        count = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / count

    def forward(
        self,
        q_tokens_list: List[torch.Tensor],
        item_repr: torch.Tensor,
        ns_tokens: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        ns_summary = ns_tokens.mean(dim=1)
        out = []
        for i, q in enumerate(q_tokens_list):
            seq_mean = self._masked_mean(seq_tokens_list[i], seq_masks_list[i])
            delta = self.mlps[i](torch.cat([item_repr, ns_summary, seq_mean], dim=-1))
            delta = delta.view(q.shape[0], self.num_queries, q.shape[-1])
            out.append(q + torch.sigmoid(self.gates[i]) * delta)
        return out


class RecencyQueryAdapter(nn.Module):
    """Light recency pooling adapter; time only nudges query tokens."""

    def __init__(
        self,
        d_model: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        gate_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        hidden_dim = d_model * hidden_mult
        self.alphas = nn.Parameter(torch.full((num_sequences,), 0.2))
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(3 * d_model),
                nn.Linear(3 * d_model, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_queries * d_model),
            )
            for _ in range(num_sequences)
        ])
        for mlp in self.mlps:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
        self.gates = nn.Parameter(torch.full((num_sequences,), float(gate_init)))

    @staticmethod
    def _masked_mean(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
        count = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / count

    def forward(
        self,
        q_tokens_list: List[torch.Tensor],
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
        seq_time_deltas_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        out = []
        for i, q in enumerate(q_tokens_list):
            seq_tokens = seq_tokens_list[i]
            mask = seq_masks_list[i]
            time_delta = seq_time_deltas_list[i].to(seq_tokens.dtype)
            seq_mean = self._masked_mean(seq_tokens, mask)
            scores = -F.softplus(self.alphas[i]) * time_delta
            scores = scores.masked_fill(mask, -1e9)
            all_pad = mask.all(dim=1, keepdim=True)
            weights = F.softmax(scores, dim=1).masked_fill(all_pad, 0.0)
            seq_recent = (seq_tokens * weights.unsqueeze(-1)).sum(dim=1)
            delta_in = torch.cat([seq_mean, seq_recent, seq_recent - seq_mean], dim=-1)
            delta = self.mlps[i](delta_in).view(q.shape[0], self.num_queries, q.shape[-1])
            out.append(q + torch.sigmoid(self.gates[i]) * delta)
        return out


class NSCrossAdapter(nn.Module):
    """Low-rank gated residual over flattened NS tokens."""

    def __init__(
        self,
        num_ns: int,
        d_model: int,
        low_rank: int = 64,
        dropout: float = 0.05,
        gate_init: float = -4.0,
    ) -> None:
        super().__init__()
        in_dim = num_ns * d_model
        rank = max(1, int(low_rank))
        self.norm = nn.LayerNorm(in_dim)
        self.down = nn.Linear(in_dim, rank)
        self.up = nn.Linear(rank, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.num_ns = num_ns
        self.d_model = d_model

    def forward(self, ns_tokens: torch.Tensor) -> torch.Tensor:
        B = ns_tokens.shape[0]
        flat = ns_tokens.reshape(B, -1)
        delta = self.down(self.norm(flat))
        delta = F.silu(delta)
        delta = self.dropout(delta)
        delta = self.up(delta).view(B, self.num_ns, self.d_model)
        return ns_tokens + torch.sigmoid(self.gate) * self.out_norm(delta)


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        user_dense_feature_specs: Optional[List[Tuple[int, int, int]]] = None,
        user_int_feature_fids: Optional[List[int]] = None,
        ue_dense_fids: Optional[Union[str, List[int], Tuple[int, ...]]] = None,
        pair_dense_fids: Optional[Union[str, List[int], Tuple[int, ...]]] = None,
        dense_emb_group_fids: Optional[Union[str, List[int], Tuple[int, ...]]] = None,
        dense_stat_group_fids: Optional[Union[str, List[int], Tuple[int, ...]]] = None,
        dense_quantile_group_fids: Optional[Union[str, List[int], Tuple[int, ...]]] = None,
        dense_stat_clamp_value: float = 18_000_000.0,
        use_dense_group_projector: bool = True,
        use_quantile_trend_encoder: bool = True,
        dense_group_projector_version: str = 'v7',
        use_time_context_features: bool = True,
        time_context_emb_dim: int = 16,
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        use_time_context_encoder: bool = False,
        time_context_hidden_mult: int = 2,
        time_context_dropout: float = 0.0,
        query_generator_type: str = 'vanilla',
        temporal_query_alpha: float = 0.1,
        temporal_query_use_base_query: bool = True,
        use_token_se: bool = False,
        token_se_gamma_init: float = 0.01,
        token_se_hidden_mult: float = 0.25,
        token_se_dropout: float = 0.0,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        hash_bucket_size: int = 0,
        use_item_query_adapter: bool = False,
        item_query_gate_init: float = -4.0,
        use_recency_query_adapter: bool = False,
        recency_query_gate_init: float = -4.0,
        recency_query_dropout: float = 0.0,
        use_stat_pair_residual: bool = False,
        stat_pair_gate_init: float = -4.0,
        user_sparse_dense_pair_specs: Optional[List[Tuple[int, int, int, int, int]]] = None,
        use_ns_cross_adapter: bool = False,
        ns_cross_low_rank: int = 64,
        ns_cross_dropout: float = 0.05,
        ns_cross_gate_init: float = -4.0,
        use_seq_time_film: bool = False,
        seq_time_film_gate_init: float = -2.5,
        seq_time_dropout: float = 0.02,
        use_target_time_din: bool = False,
        din_hidden_mult: int = 3,
        din_dropout: float = 0.02,
        din_query_gate_init: float = -2.5,
        din_output_gate_init: float = -2.8,
        din_history_dropout: float = 0.08,
        use_dcn_ns_cross: bool = False,
        dcn_ns_cross_layers: int = 2,
        dcn_ns_cross_low_rank: int = 64,
        dcn_ns_cross_dropout: float = 0.05,
        use_ns_output_fusion_residual: bool = False,
        ns_output_fusion_gate_init: float = -3.2,
        ns_output_fusion_dropout: float = 0.02,
        use_strong_time_residual: bool = False,
        time_residual_gamma_init: float = 0.03,
        use_gamma_writeback: bool = False,
        gamma_q_init: float = 0.05,
        gamma_ns_init: float = 0.03,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.hash_bucket_size = int(hash_bucket_size)
        self.use_item_query_adapter = bool(use_item_query_adapter)
        self.use_recency_query_adapter = bool(use_recency_query_adapter)
        self.use_stat_pair_residual = bool(use_stat_pair_residual)
        self.use_ns_cross_adapter = bool(use_ns_cross_adapter)
        self.use_seq_time_film = bool(use_seq_time_film)
        self.use_target_time_din = bool(use_target_time_din)
        self.use_dcn_ns_cross = bool(use_dcn_ns_cross)
        self.use_ns_output_fusion_residual = bool(use_ns_output_fusion_residual)
        self.use_time_context_encoder = bool(use_time_context_encoder)
        self.query_generator_type = query_generator_type
        self.use_token_se = bool(use_token_se)
        self.use_strong_time_residual = bool(use_strong_time_residual)
        self.use_gamma_writeback = bool(use_gamma_writeback)
        if self.use_strong_time_residual and (
            not self.use_time_context_encoder
            or self.query_generator_type not in ('temporal', 'dual_target_time')
        ):
            raise ValueError(
                "use_strong_time_residual requires --use_time_context_encoder "
                "with temporal or dual_target_time query generator")

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
                hash_bucket_size=hash_bucket_size,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
                hash_bucket_size=hash_bucket_size,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                hash_bucket_size=hash_bucket_size,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                hash_bucket_size=hash_bucket_size,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection. v7 uses semantic dense groups; v6
        # compatibility falls back to the old single Linear+LN token.
        if user_dense_feature_specs is None:
            user_dense_feature_specs = (
                [(0, 0, user_dense_dim)] if user_dense_dim > 0 else []
            )
        self.use_dense_group_projector = bool(use_dense_group_projector)
        self.use_time_context_features = bool(use_time_context_features)
        if self.use_dense_group_projector:
            if dense_group_projector_version != 'v7':
                raise ValueError(
                    f"Unsupported dense_group_projector_version={dense_group_projector_version!r}; "
                    "only 'v7' is currently implemented.")
            if dense_emb_group_fids is None:
                dense_emb_group_fids = ue_dense_fids or '61,87'
            self.user_dense_projector = DenseGroupProjector(
                user_dense_feature_specs=user_dense_feature_specs,
                d_model=d_model,
                dense_emb_group_fids=dense_emb_group_fids,
                dense_stat_group_fids=dense_stat_group_fids,
                dense_quantile_group_fids=dense_quantile_group_fids,
                stat_clamp_value=dense_stat_clamp_value,
                use_quantile_trend_encoder=use_quantile_trend_encoder,
                version=dense_group_projector_version,
                use_time_context_features=use_time_context_features,
                time_context_emb_dim=time_context_emb_dim,
                dropout=dropout_rate,
            )
            self.user_dense_token_count = self.user_dense_projector.num_tokens
        else:
            self.user_dense_token_count = 1 if user_dense_dim > 0 else 0
            if user_dense_dim > 0:
                self.user_dense_proj = nn.Sequential(
                    nn.Linear(user_dense_dim, d_model),
                    nn.LayerNorm(d_model),
                )
        self.has_user_dense = self.user_dense_token_count > 0

        # Item dense token is always present in the v7 NS layout. When
        # item_dense_dim == 0, Linear(0, d_model) behaves as a learned bias token.
        self.has_item_dense = True
        self.item_dense_proj = nn.Sequential(
            nn.Linear(item_dense_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Total NS token count
        self.num_ns = (num_user_ns + self.user_dense_token_count
                       + num_item_ns + (1 if self.has_item_dense else 0))

        self.stat_pair_residual = None
        if self.use_stat_pair_residual:
            if not self.use_dense_group_projector:
                raise ValueError("use_stat_pair_residual requires DenseGroupProjector")
            if self.user_dense_projector.stat_token_index < 0:
                raise ValueError("use_stat_pair_residual requires a stat dense token")
            if not user_sparse_dense_pair_specs:
                raise ValueError("use_stat_pair_residual requires paired fid62-66 specs")
            self.stat_pair_residual = SparseDensePairResidual(
                pair_specs=user_sparse_dense_pair_specs,
                emb_dim=emb_dim,
                d_model=d_model,
                gate_init=stat_pair_gate_init,
            )

        self.item_repr_mlp = None
        self.needs_item_repr = (
            self.use_item_query_adapter
            or self.query_generator_type == 'dual_target_time'
            or self.use_target_time_din
        )
        if self.needs_item_repr:
            self.item_repr_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            if self.use_item_query_adapter:
                self.item_query_adapter = ItemAwareQueryAdapter(
                    d_model=d_model,
                    num_queries=num_queries,
                    num_sequences=self.num_sequences,
                    hidden_mult=2,
                    dropout=dropout_rate,
                    gate_init=item_query_gate_init,
                )

        if self.use_recency_query_adapter:
            self.recency_query_adapter = RecencyQueryAdapter(
                d_model=d_model,
                num_queries=num_queries,
                num_sequences=self.num_sequences,
                hidden_mult=2,
                dropout=recency_query_dropout,
                gate_init=recency_query_gate_init,
            )

        if self.use_ns_cross_adapter:
            self.ns_cross_adapter = NSCrossAdapter(
                num_ns=self.num_ns,
                d_model=d_model,
                low_rank=ns_cross_low_rank,
                dropout=ns_cross_dropout,
                gate_init=ns_cross_gate_init,
            )

        if self.use_dcn_ns_cross:
            token_gate_inits = (
                [-2.8] * num_user_ns
                + [-4.2] * self.user_dense_token_count
                + [-2.8] * num_item_ns
                + ([-3.2] if self.has_item_dense else [])
            )
            self.dcn_ns_cross = DCNNSCross(
                num_ns=self.num_ns,
                d_model=d_model,
                num_layers=dcn_ns_cross_layers,
                low_rank=dcn_ns_cross_low_rank,
                dropout=dcn_ns_cross_dropout,
                token_gate_inits=token_gate_inits,
            )

        if self.use_seq_time_film:
            self.seq_time_film = SeqTimeFiLM(
                d_model=d_model,
                time_feat_dim=12,
                dropout=seq_time_dropout,
                gate_init=seq_time_film_gate_init,
            )

        if self.use_target_time_din:
            self.target_time_din = TargetTimeDINReader(
                d_model=d_model,
                num_sequences=self.num_sequences,
                time_feat_dim=12,
                hidden_mult=din_hidden_mult,
                dropout=din_dropout,
                query_gate_init=din_query_gate_init,
                output_gate_init=din_output_gate_init,
                history_dropout=din_history_dropout,
            )

        if self.use_ns_output_fusion_residual:
            self.ns_output_fusion = NSOutputFusionResidual(
                d_model=d_model,
                dropout=ns_output_fusion_dropout,
                gate_init=ns_output_fusion_gate_init,
            )

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        self._seq_hash_embs = nn.ModuleDict()
        if hash_bucket_size > 0:
            for domain in self.seq_domains:
                hash_embs = nn.ModuleDict()
                for i, vs in enumerate(self._seq_vocab_sizes[domain]):
                    if self._seq_emb_index[domain][i] == -1 and int(vs) > 0:
                        hash_embs[str(i)] = HashEmbedding(hash_bucket_size, emb_dim)
                if len(hash_embs) > 0:
                    self._seq_hash_embs[domain] = hash_embs

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        if self.use_time_context_encoder:
            if num_time_buckets <= 0:
                raise ValueError(
                    "use_time_context_encoder=True requires num_time_buckets > 0")
            self.time_context_encoder = TimeContextEncoder(
                dim=d_model,
                hidden_mult=time_context_hidden_mult,
                dropout=time_context_dropout,
            )

        # ================== HyFormer Components ==================
        # Query generator. "vanilla" preserves the original baseline path.
        if query_generator_type == 'vanilla':
            self.query_generator = MultiSeqQueryGenerator(
                d_model=d_model,
                num_ns=self.num_ns,
                num_queries=num_queries,
                num_sequences=self.num_sequences,
                hidden_mult=hidden_mult,
            )
        elif query_generator_type == 'temporal':
            self.query_generator = TemporalQueryGenerator(
                d_model=d_model,
                num_queries=num_queries,
                num_sequences=self.num_sequences,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                alpha=temporal_query_alpha,
                use_base_query=temporal_query_use_base_query,
            )
        elif query_generator_type == 'dual_target_time':
            if num_queries != 2:
                raise ValueError("dual_target_time requires --num_queries 2")
            if not self.needs_item_repr:
                raise RuntimeError("dual_target_time requires item representation")
            self.query_generator = DualQueryTargetTimeGenerator(
                d_model=d_model,
                num_sequences=self.num_sequences,
                profile_dim=19,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                alpha=temporal_query_alpha,
            )
        else:
            raise ValueError(
                f"Unknown query_generator_type={query_generator_type!r}; "
                "expected 'vanilla', 'temporal', or 'dual_target_time'.")

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
                use_gamma_writeback=self.use_gamma_writeback,
                gamma_q_init=gamma_q_init,
                gamma_ns_init=gamma_ns_init,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        if self.use_token_se:
            self.token_se = TokenSE(
                dim=d_model,
                hidden_mult=token_se_hidden_mult,
                dropout=token_se_dropout,
                gamma_init=token_se_gamma_init,
            )

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        if self.use_strong_time_residual:
            self.time_residual = StrongTimeResidual(
                d_model=d_model,
                action_num=action_num,
                hidden_mult=2,
                dropout=dropout_rate,
                gamma_init=time_residual_gamma_init,
            )

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for domain_hash_embs in getattr(self, '_seq_hash_embs', {}).values():
            for he in domain_hash_embs.values():
                for emb in he.embs:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
            for he in tokenizer._hash_embs.values():
                for emb in he.embs:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0

        if self.stat_pair_residual is not None:
            for emb in self.stat_pair_residual.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for domain_hash_embs in getattr(self, '_seq_hash_embs', {}).values():
            for he in domain_hash_embs.values():
                for emb in he.embs:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1
            for he in tokenizer._hash_embs.values():
                for emb in he.embs:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1

        if self.stat_pair_residual is not None:
            for emb, (_, vs, _, _, _) in zip(
                self.stat_pair_residual.embs,
                self.stat_pair_residual.pair_specs,
            ):
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1
        if self.use_time_context_features:
            skip_count += 2

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]


    def _make_ns_tokens_and_item_repr(
        self,
        inputs: ModelInput,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        item_parts = [item_ns]
        if self.has_user_dense:
            if self.use_dense_group_projector:
                user_dense_tokens = self.user_dense_projector(
                    inputs.user_dense_feats, inputs.time_context_feats)
                if self.stat_pair_residual is not None:
                    stat_idx = self.user_dense_projector.stat_token_index
                    user_dense_tokens = user_dense_tokens.clone()
                    user_dense_tokens[:, stat_idx, :] = (
                        user_dense_tokens[:, stat_idx, :]
                        + self.stat_pair_residual(
                            inputs.user_int_feats, inputs.user_dense_feats)
                    )
                if user_dense_tokens.shape[1] > 0:
                    ns_parts.append(user_dense_tokens)
            else:
                user_dense_tok = F.silu(
                    self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)
                ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            item_parts.append(item_dense_tok)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)
        if ns_tokens.shape[1] != self.num_ns:
            raise RuntimeError(
                f"ns_tokens token count mismatch: expected {self.num_ns}, "
                f"got {ns_tokens.shape[1]}")
        if self.use_ns_cross_adapter:
            ns_tokens = self.ns_cross_adapter(ns_tokens)
        if self.use_dcn_ns_cross:
            ns_tokens = self.dcn_ns_cross(ns_tokens)

        item_repr = None
        if self.item_repr_mlp is not None:
            item_tokens = torch.cat(item_parts, dim=1)
            item_repr = self.item_repr_mlp(torch.cat([
                item_tokens.mean(dim=1),
                item_tokens.max(dim=1).values,
            ], dim=-1))
        return ns_tokens, item_repr

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        hash_embs: Optional[nn.ModuleDict] = None,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                hash_key = str(i)
                if hash_embs is not None and hash_key in hash_embs:
                    e = hash_embs[hash_key](seq[:, i, :])
                    if is_id[i] and self.training:
                        e = self.seq_id_emb_dropout(e)
                    emb_list.append(e)
                else:
                    emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the multi-sequence block stack with dropout and output projection."""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
            )

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        if self.use_token_se:
            all_q = self.token_se(all_q, curr_ns)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output, curr_ns

    def _make_seq_time_features(
        self,
        time_delta: torch.Tensor,
        time_gap: torch.Tensor,
        time_calendar: torch.Tensor,
        time_context_feats: torch.Tensor,
    ) -> torch.Tensor:
        B, L = time_delta.shape
        device = time_delta.device
        dtype = time_delta.dtype
        hour = time_context_feats[:, 0].to(device=device, dtype=dtype).clamp(0, 23)
        weekday = time_context_feats[:, 1].to(device=device, dtype=dtype).clamp(0, 6)
        curr = torch.stack([
            torch.sin(2.0 * math.pi * hour / 24.0),
            torch.cos(2.0 * math.pi * hour / 24.0),
            torch.sin(2.0 * math.pi * weekday / 7.0),
            torch.cos(2.0 * math.pi * weekday / 7.0),
        ], dim=-1).unsqueeze(1).expand(-1, L, -1)
        cal = time_calendar.to(device=device, dtype=dtype)
        hour_align = (cal[..., 0:1] * curr[..., 0:1] + cal[..., 1:2] * curr[..., 1:2])
        weekday_align = (cal[..., 2:3] * curr[..., 2:3] + cal[..., 3:4] * curr[..., 3:4])
        return torch.cat([
            time_delta.unsqueeze(-1).to(dtype),
            time_gap.unsqueeze(-1).to(dtype),
            cal,
            curr,
            hour_align,
            weekday_align,
        ], dim=-1)

    def _make_seq_time_profile(
        self,
        time_delta: torch.Tensor,
        time_gap: torch.Tensor,
        time_calendar: torch.Tensor,
        time_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        dtype = time_delta.dtype
        valid = (~padding_mask).to(dtype)
        count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        age_hours = torch.expm1(time_delta.clamp_min(0.0))
        gap_minutes = torch.expm1(time_gap.clamp_min(0.0))
        mean_age = torch.log1p((age_hours * valid).sum(dim=1, keepdim=True) / count)
        masked_age = age_hours.masked_fill(padding_mask, 1e9)
        min_age = torch.log1p(masked_age.min(dim=1, keepdim=True).values.clamp(max=1e8))
        min_age = torch.where(valid.sum(dim=1, keepdim=True) > 0, min_age, torch.zeros_like(min_age))
        ratios = [
            ((age_hours < h).to(dtype) * valid).sum(dim=1, keepdim=True) / count
            for h in (1.0, 6.0, 24.0, 168.0)
        ]
        mean_gap = torch.log1p((gap_minutes * valid).sum(dim=1, keepdim=True) / count)
        gap_ratio = ((gap_minutes < 30.0).to(dtype) * valid).sum(dim=1, keepdim=True) / count
        hist_cal = (time_calendar.to(dtype) * valid.unsqueeze(-1)).sum(dim=1) / count
        curr_cal = time_features[:, :, 6:10].mean(dim=1)
        align = (time_features[:, :, 10:12] * valid.unsqueeze(-1)).sum(dim=1) / count
        log_count = torch.log1p(valid.sum(dim=1, keepdim=True))
        return torch.cat([
            mean_age, min_age, *ratios, mean_gap, gap_ratio,
            hist_cal, curr_cal, align, log_count,
        ], dim=-1)

    def _apply_strong_time_residual(
        self,
        logits: torch.Tensor,
        output: torch.Tensor,
        z_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_strong_time_residual:
            return logits
        if z_t is None:
            raise RuntimeError(
                "StrongTimeResidual is enabled but z_t is None; "
                "enable --use_time_context_encoder with temporal/dual_target_time")
        return logits + self.time_residual(output, z_t)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        # 1. NS tokens and optional item representation.
        ns_tokens, item_repr = self._make_ns_tokens_and_item_repr(inputs)

        # 2. Embed each sequence domain (dynamic)
        seq_tokens_list = []
        seq_masks_list = []
        seq_time_buckets_list = []
        seq_time_deltas_list = []
        seq_time_gaps_list = []
        seq_time_features_list = []
        seq_time_profiles_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                hash_embs=self._seq_hash_embs[domain] if domain in self._seq_hash_embs else None)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            time_gap = inputs.seq_time_gaps[domain]
            time_cal = inputs.seq_time_calendars[domain]
            time_feat = self._make_seq_time_features(
                inputs.seq_time_deltas[domain], time_gap, time_cal, inputs.time_context_feats)
            if self.use_seq_time_film:
                tokens = self.seq_time_film(tokens, time_feat, mask)
            seq_tokens_list.append(tokens)
            seq_masks_list.append(mask)
            seq_time_buckets_list.append(inputs.seq_time_buckets[domain])
            seq_time_deltas_list.append(inputs.seq_time_deltas[domain])
            seq_time_gaps_list.append(time_gap)
            seq_time_features_list.append(time_feat)
            seq_time_profiles_list.append(self._make_seq_time_profile(
                inputs.seq_time_deltas[domain], time_gap, time_cal, time_feat, mask))

        z_t = None
        if self.use_time_context_encoder and self.query_generator_type in ('temporal', 'dual_target_time'):
            z_t = self.time_context_encoder(
                seq_time_buckets_list,
                seq_masks_list,
                self.time_embedding,
            )

        # 3. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        din_pool = None
        if self.query_generator_type == 'dual_target_time':
            if item_repr is None:
                raise RuntimeError("dual_target_time requires item_repr")
            q_tokens_list = self.query_generator(
                item_repr, ns_tokens, seq_tokens_list, seq_masks_list,
                seq_time_deltas_list, seq_time_profiles_list, z_t=z_t)
        elif self.query_generator_type == 'temporal':
            q_tokens_list = self.query_generator(
                ns_tokens, seq_tokens_list, seq_masks_list, z_t=z_t)
        else:
            q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)
        if self.use_item_query_adapter and item_repr is not None and self.query_generator_type != 'dual_target_time':
            q_tokens_list = self.item_query_adapter(
                q_tokens_list, item_repr, ns_tokens, seq_tokens_list, seq_masks_list)
        if self.use_recency_query_adapter and self.query_generator_type != 'dual_target_time':
            q_tokens_list = self.recency_query_adapter(
                q_tokens_list, seq_tokens_list, seq_masks_list, seq_time_deltas_list)
        if self.use_target_time_din:
            if item_repr is None:
                raise RuntimeError("use_target_time_din requires item_repr")
            q_tokens_list, din_pool = self.target_time_din(
                q_tokens_list, item_repr, seq_tokens_list, seq_masks_list,
                seq_time_features_list, seq_time_deltas_list)

        # 4. Dropout + MultiSeqHyFormerBlock stack + output projection
        output, final_ns = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training
        )
        if self.use_target_time_din and din_pool is not None:
            output = self.target_time_din.fuse_output(output, din_pool, item_repr)
        if self.use_ns_output_fusion_residual:
            output = self.ns_output_fusion(output, final_ns)

        # 5. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        logits = self._apply_strong_time_residual(logits, output, z_t)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        # Reuses forward logic but without dropout
        ns_tokens, item_repr = self._make_ns_tokens_and_item_repr(inputs)

        seq_tokens_list = []
        seq_masks_list = []
        seq_time_buckets_list = []
        seq_time_deltas_list = []
        seq_time_features_list = []
        seq_time_profiles_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                hash_embs=self._seq_hash_embs[domain] if domain in self._seq_hash_embs else None)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            time_gap = inputs.seq_time_gaps[domain]
            time_cal = inputs.seq_time_calendars[domain]
            time_feat = self._make_seq_time_features(
                inputs.seq_time_deltas[domain], time_gap, time_cal, inputs.time_context_feats)
            if self.use_seq_time_film:
                tokens = self.seq_time_film(tokens, time_feat, mask)
            seq_tokens_list.append(tokens)
            seq_masks_list.append(mask)
            seq_time_buckets_list.append(inputs.seq_time_buckets[domain])
            seq_time_deltas_list.append(inputs.seq_time_deltas[domain])
            seq_time_features_list.append(time_feat)
            seq_time_profiles_list.append(self._make_seq_time_profile(
                inputs.seq_time_deltas[domain], time_gap, time_cal, time_feat, mask))

        z_t = None
        if self.use_time_context_encoder and self.query_generator_type in ('temporal', 'dual_target_time'):
            z_t = self.time_context_encoder(
                seq_time_buckets_list,
                seq_masks_list,
                self.time_embedding,
            )

        din_pool = None
        if self.query_generator_type == 'dual_target_time':
            if item_repr is None:
                raise RuntimeError("dual_target_time requires item_repr")
            q_tokens_list = self.query_generator(
                item_repr, ns_tokens, seq_tokens_list, seq_masks_list,
                seq_time_deltas_list, seq_time_profiles_list, z_t=z_t)
        elif self.query_generator_type == 'temporal':
            q_tokens_list = self.query_generator(
                ns_tokens, seq_tokens_list, seq_masks_list, z_t=z_t)
        else:
            q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)
        if self.use_item_query_adapter and item_repr is not None and self.query_generator_type != 'dual_target_time':
            q_tokens_list = self.item_query_adapter(
                q_tokens_list, item_repr, ns_tokens, seq_tokens_list, seq_masks_list)
        if self.use_recency_query_adapter and self.query_generator_type != 'dual_target_time':
            q_tokens_list = self.recency_query_adapter(
                q_tokens_list, seq_tokens_list, seq_masks_list, seq_time_deltas_list)
        if self.use_target_time_din:
            if item_repr is None:
                raise RuntimeError("use_target_time_din requires item_repr")
            q_tokens_list, din_pool = self.target_time_din(
                q_tokens_list, item_repr, seq_tokens_list, seq_masks_list,
                seq_time_features_list, seq_time_deltas_list)

        output, final_ns = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=False
        )
        if self.use_target_time_din and din_pool is not None:
            output = self.target_time_din.fuse_output(output, din_pool, item_repr)
        if self.use_ns_output_fusion_residual:
            output = self.ns_output_fusion(output, final_ns)

        logits = self.clsfier(output)
        logits = self._apply_strong_time_residual(logits, output, z_t)
        return logits, output
