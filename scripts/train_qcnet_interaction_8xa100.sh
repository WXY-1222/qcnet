#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_qcnet_interaction_8xa100.sh \
#     /data/sdb/bitwxy/interaction_data/interaction_digir_all_12loc_h8_f12.pkl

DATA_PKL="${1:-/data/sdb/bitwxy/interaction_data/interaction_digir_all_12loc_h8_f12.pkl}"

python train_qcnet.py \
  --dataset interaction_digir \
  --interaction_data_path "${DATA_PKL}" \
  --train_batch_size 4 \
  --val_batch_size 4 \
  --test_batch_size 4 \
  --devices 8 \
  --accelerator gpu \
  --num_workers 16 \
  --max_epochs 64 \
  --num_historical_steps 8 \
  --num_future_steps 12 \
  --num_recurrent_steps 3 \
  --pl2pl_radius 80 \
  --time_span 8 \
  --pl2a_radius 50 \
  --a2a_radius 50 \
  --num_t2m_steps 8 \
  --pl2m_radius 80 \
  --a2m_radius 80
