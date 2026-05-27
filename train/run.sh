#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "[algo8286] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[algo8286] train.py=${SCRIPT_DIR}/train.py"
if ! grep -q -- "--use_strong_time_residual" "${SCRIPT_DIR}/train.py"; then
    echo "[algo8286][ERROR] ${SCRIPT_DIR}/train.py does not contain final residual arguments." >&2
    echo "[algo8286][ERROR] You are likely running a stale copy or the wrong directory." >&2
    exit 2
fi

# ---- Experiment B: 0.831 base + aggressive DIN + Strong Time Residual + gamma write-back ----
# Higher-ceiling shot: more aggressive target-history branch plus RankMixer write-back regularization.
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
    --valid_ratio 0.05 \
    --num_workers 8 \
    "$@"
å
