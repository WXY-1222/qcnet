#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/eval_qcnet_interaction.sh \
#     /data/sdb/bitwxy/interaction_data/interaction_digir_all_12loc_h8_f12.pkl \
#     /path/to/checkpoint.ckpt

DATA_PKL="${1:-/data/sdb/bitwxy/interaction_data/interaction_digir_all_12loc_h8_f12.pkl}"
CKPT_PATH="${2:?Please provide checkpoint path}"

python val.py \
  --model QCNet \
  --ckpt_path "${CKPT_PATH}" \
  --interaction_data_path "${DATA_PKL}" \
  --split val \
  --batch_size 4 \
  --devices 1 \
  --accelerator gpu \
  --num_workers 8
