"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


USER_SPARSE_DENSE_PAIR_FIDS = [62, 63, 64, 65, 66]


def build_user_sparse_dense_pair_specs(
    user_int_schema: FeatureSchema,
    user_dense_schema: FeatureSchema,
    user_int_vocab_sizes: List[int],
    fids: List[int] = USER_SPARSE_DENSE_PAIR_FIDS,
) -> List[Tuple[int, int, int, int, int]]:
    specs: List[Tuple[int, int, int, int, int]] = []
    for fid in fids:
        try:
            int_offset, int_length = user_int_schema.get_offset_length(fid)
            dense_offset, dense_length = user_dense_schema.get_offset_length(fid)
        except KeyError:
            continue
        if int_length != dense_length:
            raise ValueError(
                f"Paired user fid {fid} has mismatched lengths: "
                f"user_int={int_length}, user_dense={dense_length}")
        vs = max(user_int_vocab_sizes[int_offset:int_offset + int_length])
        specs.append((fid, vs, int_offset, dense_offset, int_length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce',
                        choices=['bce', 'focal', 'bce_pairwise'],
                        help='Loss type: bce, focal, or BCE plus batch pairwise ranking loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='AdamW weight decay for dense parameters')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Linear warmup steps for dense optimizer')
    parser.add_argument('--lr_schedule', type=str, default='none', choices=['none', 'cosine'],
                        help='Dense LR schedule')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA decay for model weights; 0 disables EMA')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Tiny label smoothing epsilon')
    parser.add_argument('--pairwise_lambda', type=float, default=0.03,
                        help='Weight for bce_pairwise loss')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')
    parser.add_argument('--use_time_context_encoder', action='store_true', default=False,
                        help='Enable TAAC v7.6 TimeContextEncoder over sequence time buckets.')
    parser.add_argument('--time_context_hidden_mult', type=int, default=2,
                        help='Hidden multiplier for TimeContextEncoder MLP.')
    parser.add_argument('--time_context_dropout', type=float, default=0.0,
                        help='Dropout inside TimeContextEncoder MLP.')
    parser.add_argument('--query_generator_type', type=str, default='vanilla',
                        choices=['vanilla', 'temporal', 'dual_target_time'],
                        help='Query generator type: vanilla preserves baseline; '
                             'temporal uses NS/sequence summaries plus optional z_t.')
    parser.add_argument('--temporal_query_alpha', type=float, default=0.1,
                        help='Residual scale for TemporalQueryGenerator q_delta.')
    parser.add_argument('--temporal_query_use_base_query', action='store_true', default=True,
                        help='Use learned base query in TemporalQueryGenerator (default on).')
    parser.add_argument('--no_temporal_query_base_query',
                        dest='temporal_query_use_base_query',
                        action='store_false',
                        help='Disable learned base query in TemporalQueryGenerator.')
    parser.add_argument('--use_token_se', action='store_true', default=False,
                        help='Enable TokenSE recalibration before output_proj.')
    parser.add_argument('--token_se_gamma_init', type=float, default=0.01,
                        help='Initial learnable gamma for TokenSE residual.')
    parser.add_argument('--token_se_hidden_mult', type=float, default=0.25,
                        help='Hidden dim multiplier for TokenSE MLP.')
    parser.add_argument('--token_se_dropout', type=float, default=0.0,
                        help='Dropout inside TokenSE MLP.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')
    parser.add_argument('--ue_dense_fids', type=str, default='61,87',
                        help='Backward-compatible alias for --dense_emb_group_fids.')
    parser.add_argument('--pair_dense_fids', type=str, default='',
                        help='Deprecated; DenseGroupProjector no longer uses dense/int pair gates.')
    parser.add_argument('--dense_emb_group_fids', type=str, default='61,87',
                        help='Comma-separated user_dense fids for the embedding group '
                             '(L2-normalized then projected to one NS token).')
    parser.add_argument('--dense_stat_group_fids', type=str, default='62,63,64,65,66',
                        help='Comma-separated user_dense fids for the stat/count group '
                             '(log1p+clamp then projected to one NS token).')
    parser.add_argument('--dense_quantile_group_fids', type=str, default='89,90,91',
                        help='Comma-separated user_dense fids for the quantile/rank group '
                             '(encoded by QuantileTrendEncoder into one NS token).')
    parser.add_argument('--dense_stat_clamp_value', type=float, default=18000000.0,
                        help='Upper clamp for stat/count dense values before log1p.')
    parser.add_argument('--use_dense_group_projector', action='store_true', default=True,
                        help='Enable TAAC v7 DenseGroupProjector for user_dense_feats '
                             '(default on).')
    parser.add_argument('--no_dense_groups', dest='use_dense_group_projector',
                        action='store_false',
                        help='Disable DenseGroupProjector and fall back to the v6 '
                             'single Linear+LN user_dense token.')
    parser.add_argument('--use_quantile_trend_encoder', action='store_true', default=True,
                        help='Use Conv1d QuantileTrendEncoder for dense quantile group '
                             '(default on).')
    parser.add_argument('--no_quantile_trend_encoder', dest='use_quantile_trend_encoder',
                        action='store_false',
                        help='Project quantile group with a single Linear+LN instead.')
    parser.add_argument('--dense_group_projector_version', type=str, default='v7',
                        choices=['v7'],
                        help='DenseGroupProjector implementation version.')
    parser.add_argument('--no_semantic_seq', action='store_true', default=False,
                        help='Compatibility flag only; semantic sequence changes are not '
                             'implemented in this baseline.')
    parser.add_argument('--no_domain_gate', action='store_true', default=False,
                        help='Compatibility flag only; domain gate changes are not '
                             'implemented in this baseline.')
    parser.add_argument('--use_time_context_features', action='store_true', default=True,
                        help='Enable Beijing-time hour/weekday embeddings in the '
                             'user dense branch (default on).')
    parser.add_argument('--no_time_context_features', dest='use_time_context_features',
                        action='store_false',
                        help='Disable Beijing-time hour/weekday embeddings.')
    parser.add_argument('--time_context_emb_dim', type=int, default=16,
                        help='Embedding dimension for each time context field '
                             '(hour and weekday).')

    parser.add_argument('--hash_bucket_size', type=int, default=0,
                        help='Bucket count for hash fallback on skipped high-card features.')
    parser.add_argument('--use_item_query_adapter', action='store_true', default=False,
                        help='Enable gated candidate-item adapter on query tokens.')
    parser.add_argument('--item_query_gate_init', type=float, default=-4.0,
                        help='Initial logit gate for item query adapter.')
    parser.add_argument('--use_recency_query_adapter', action='store_true', default=False,
                        help='Enable light recency pooling adapter on query tokens.')
    parser.add_argument('--recency_query_gate_init', type=float, default=-4.0,
                        help='Initial logit gate for recency query adapter.')
    parser.add_argument('--recency_query_dropout', type=float, default=0.0,
                        help='Dropout inside recency query adapter.')
    parser.add_argument('--use_stat_pair_residual', action='store_true', default=False,
                        help='Enable gated sparse-dense residual on the 62-66 dense stat token.')
    parser.add_argument('--stat_pair_gate_init', type=float, default=-4.0,
                        help='Initial logit gate for stat pair residual.')
    parser.add_argument('--use_ns_cross_adapter', action='store_true', default=False,
                        help='Enable low-rank gated NS token cross adapter.')
    parser.add_argument('--ns_cross_low_rank', type=int, default=64,
                        help='Low-rank dimension for NS cross adapter.')
    parser.add_argument('--ns_cross_dropout', type=float, default=0.05,
                        help='Dropout inside NS cross adapter.')
    parser.add_argument('--ns_cross_gate_init', type=float, default=-4.0,
                        help='Initial logit gate for NS cross adapter.')
    parser.add_argument('--use_seq_time_film', action='store_true', default=False,
                        help='Enable continuous TimeFiLM modulation on sequence tokens.')
    parser.add_argument('--seq_time_film_gate_init', type=float, default=-2.5,
                        help='Initial logit gate for SeqTimeFiLM.')
    parser.add_argument('--seq_time_dropout', type=float, default=0.02,
                        help='Dropout inside sequence time modules.')
    parser.add_argument('--use_target_time_din', action='store_true', default=False,
                        help='Enable target-time DIN reader on the target query.')
    parser.add_argument('--din_hidden_mult', type=int, default=3,
                        help='Hidden multiplier for TargetTimeDINReader.')
    parser.add_argument('--din_dropout', type=float, default=0.02,
                        help='Dropout inside TargetTimeDINReader.')
    parser.add_argument('--din_query_gate_init', type=float, default=-2.5,
                        help='Initial logit gate for DIN query residual.')
    parser.add_argument('--din_output_gate_init', type=float, default=-2.8,
                        help='Initial logit gate for DIN output residual.')
    parser.add_argument('--din_history_dropout', type=float, default=0.08,
                        help='Dropout on history tokens inside DIN reader.')
    parser.add_argument('--use_dcn_ns_cross', action='store_true', default=False,
                        help='Enable DCN-v2 cross network over NS tokens.')
    parser.add_argument('--dcn_ns_cross_layers', type=int, default=2,
                        help='Number of DCN-v2 NS cross layers.')
    parser.add_argument('--dcn_ns_cross_low_rank', type=int, default=64,
                        help='Low-rank dimension for DCN-v2 NS cross.')
    parser.add_argument('--dcn_ns_cross_dropout', type=float, default=0.05,
                        help='Dropout inside DCN-v2 NS cross.')
    parser.add_argument('--use_ns_output_fusion_residual', action='store_true', default=False,
                        help='Enable gated residual fusion from final NS tokens.')
    parser.add_argument('--ns_output_fusion_gate_init', type=float, default=-3.2,
                        help='Initial logit gate for NS output fusion residual.')
    parser.add_argument('--ns_output_fusion_dropout', type=float, default=0.02,
                        help='Dropout inside NS output fusion residual.')
    parser.add_argument('--use_strong_time_residual', action='store_true', default=False,
                        help='Enable logits-side residual conditioned on output and z_t.')
    parser.add_argument('--time_residual_gamma_init', type=float, default=0.03,
                        help='Initial scalar gamma for StrongTimeResidual logits branch.')
    parser.add_argument('--use_gamma_writeback', action='store_true', default=False,
                        help='Enable gated residual write-back after RankMixer in HyFormer blocks.')
    parser.add_argument('--gamma_q_init', type=float, default=0.05,
                        help='Initial write-back blend for query tokens when gamma write-back is enabled.')
    parser.add_argument('--gamma_ns_init', type=float, default=0.03,
                        help='Initial write-back blend for NS tokens when gamma write-back is enabled.')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)
    user_dense_feature_specs = [
        (fid, offset, length)
        for fid, offset, length in pcvr_dataset.user_dense_schema.entries
    ]
    user_int_feature_fids = [
        fid for fid, _, _ in pcvr_dataset.user_int_schema.entries
    ]

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "user_dense_feature_specs": user_dense_feature_specs,
        "user_int_feature_fids": user_int_feature_fids,
        "ue_dense_fids": args.ue_dense_fids,
        "pair_dense_fids": args.pair_dense_fids,
        "dense_emb_group_fids": args.dense_emb_group_fids,
        "dense_stat_group_fids": args.dense_stat_group_fids,
        "dense_quantile_group_fids": args.dense_quantile_group_fids,
        "dense_stat_clamp_value": args.dense_stat_clamp_value,
        "use_dense_group_projector": args.use_dense_group_projector,
        "use_quantile_trend_encoder": args.use_quantile_trend_encoder,
        "dense_group_projector_version": args.dense_group_projector_version,
        "use_time_context_features": args.use_time_context_features,
        "time_context_emb_dim": args.time_context_emb_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "use_time_context_encoder": args.use_time_context_encoder,
        "time_context_hidden_mult": args.time_context_hidden_mult,
        "time_context_dropout": args.time_context_dropout,
        "query_generator_type": args.query_generator_type,
        "temporal_query_alpha": args.temporal_query_alpha,
        "temporal_query_use_base_query": args.temporal_query_use_base_query,
        "use_token_se": args.use_token_se,
        "token_se_gamma_init": args.token_se_gamma_init,
        "token_se_hidden_mult": args.token_se_hidden_mult,
        "token_se_dropout": args.token_se_dropout,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "hash_bucket_size": args.hash_bucket_size,
        "use_item_query_adapter": args.use_item_query_adapter,
        "item_query_gate_init": args.item_query_gate_init,
        "use_recency_query_adapter": args.use_recency_query_adapter,
        "recency_query_gate_init": args.recency_query_gate_init,
        "recency_query_dropout": args.recency_query_dropout,
        "use_stat_pair_residual": args.use_stat_pair_residual,
        "stat_pair_gate_init": args.stat_pair_gate_init,
        "user_sparse_dense_pair_specs": build_user_sparse_dense_pair_specs(
            pcvr_dataset.user_int_schema,
            pcvr_dataset.user_dense_schema,
            pcvr_dataset.user_int_vocab_sizes,
        ),
        "use_ns_cross_adapter": args.use_ns_cross_adapter,
        "ns_cross_low_rank": args.ns_cross_low_rank,
        "ns_cross_dropout": args.ns_cross_dropout,
        "ns_cross_gate_init": args.ns_cross_gate_init,
        "use_seq_time_film": args.use_seq_time_film,
        "seq_time_film_gate_init": args.seq_time_film_gate_init,
        "seq_time_dropout": args.seq_time_dropout,
        "use_target_time_din": args.use_target_time_din,
        "din_hidden_mult": args.din_hidden_mult,
        "din_dropout": args.din_dropout,
        "din_query_gate_init": args.din_query_gate_init,
        "din_output_gate_init": args.din_output_gate_init,
        "din_history_dropout": args.din_history_dropout,
        "use_dcn_ns_cross": args.use_dcn_ns_cross,
        "dcn_ns_cross_layers": args.dcn_ns_cross_layers,
        "dcn_ns_cross_low_rank": args.dcn_ns_cross_low_rank,
        "dcn_ns_cross_dropout": args.dcn_ns_cross_dropout,
        "use_ns_output_fusion_residual": args.use_ns_output_fusion_residual,
        "ns_output_fusion_gate_init": args.ns_output_fusion_gate_init,
        "ns_output_fusion_dropout": args.ns_output_fusion_dropout,
        "use_strong_time_residual": args.use_strong_time_residual,
        "time_residual_gamma_init": args.time_residual_gamma_init,
        "use_gamma_writeback": args.use_gamma_writeback,
        "gamma_q_init": args.gamma_q_init,
        "gamma_ns_init": args.gamma_ns_init,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        ema_decay=args.ema_decay,
        label_smoothing=args.label_smoothing,
        pairwise_lambda=args.pairwise_lambda,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
