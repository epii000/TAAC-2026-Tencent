#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
echo "[algo8286] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[algo8286] train.py=${SCRIPT_DIR}/train.py"
if ! grep -q -- "--hash_bucket_size" "${SCRIPT_DIR}/train.py"; then
    echo "[algo8286][ERROR] ${SCRIPT_DIR}/train.py does not contain new algo8286++ arguments." >&2
    echo "[algo8286][ERROR] You are likely running a stale copy or the wrong directory." >&2
    exit 2
fi

# ---- Active config: Ratio-only probe, valid_ratio=0.05 ----
# GPU0 / data-window probe, only changes valid_ratio from the 0.831082 setup:
#   CUDA_VISIBLE_DEVICES=0 bash train/run.sh --seed 42
#
# GPU1 / strong-DIN probe, keeps valid_ratio=0.1 and only strengthens DIN:
#   CUDA_VISIBLE_DEVICES=1 bash train/run.sh \
#     --valid_ratio 0.1 \
#     --seed 42 \
#     --seq_time_film_gate_init -2.0 \
#     --din_query_gate_init -1.8 \
#     --din_output_gate_init -2.2 \
#     --din_history_dropout 0.04 \
#     --ns_output_fusion_gate_init -2.6
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 2 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --dense_emb_group_fids "61,87" \
    --dense_stat_group_fids "62,63,64,65,66" \
    --dense_quantile_group_fids "89,90,91" \
    --emb_skip_threshold 1000000 \
    --hash_bucket_size 100000 \
    --query_generator_type dual_target_time \
    --use_time_context_encoder \
    --use_token_se \
    --temporal_query_alpha 0.1 \
    --use_seq_time_film \
    --seq_time_film_gate_init -1.6 \
    --seq_time_dropout 0.01 \
    --use_target_time_din \
    --din_hidden_mult 4 \
    --din_dropout 0.01 \
    --din_query_gate_init -1.25 \
    --din_output_gate_init -1.65 \
    --din_history_dropout 0.02 \
    --use_dcn_ns_cross \
    --dcn_ns_cross_layers 2 \
    --dcn_ns_cross_low_rank 64 \
    --dcn_ns_cross_dropout 0.04 \
    --use_ns_output_fusion_residual \
    --ns_output_fusion_gate_init -2.25 \
    --ns_output_fusion_dropout 0.01 \
    --use_strong_time_residual \
    --time_residual_gamma_init 0.04 \
    --use_gamma_writeback \
    --gamma_q_init 0.65 \
    --gamma_ns_init 0.45 \
    --num_workers 8 \
    "$@"


# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With DenseGroupProjector: user_ns=2 + dense=3 + item_ns=2 + item_dense=1 => num_ns=8,
# so num_queries=2 and 4 sequence domains gives T=16, which divides d_model=64.
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"

# ---- TAAC v7.6 ablation configs on top of the best baseline ----
# Prefer passing these flags to run.sh, for example:
#   bash train/run.sh --use_token_se
#   bash train/run.sh --query_generator_type temporal
#   bash train/run.sh --query_generator_type temporal --use_time_context_encoder
#   bash train/run.sh --query_generator_type temporal --use_time_context_encoder --use_token_se
#
# token_se:
#   --use_token_se
#
# temporal_q_nozt:
#   --query_generator_type temporal --temporal_query_alpha 0.1
#
# temporal_q_zt:
#   --query_generator_type temporal --use_time_context_encoder --temporal_query_alpha 0.1
#
# temporal_q_zt_tokense:
#   --query_generator_type temporal --use_time_context_encoder --use_token_se --temporal_query_alpha 0.1
