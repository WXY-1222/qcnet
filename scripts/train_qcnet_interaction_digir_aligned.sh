#!/usr/bin/env bash
set -euo pipefail

# DIGIR-aligned QCNet baseline config
# Maps DIGIR COMMON_ARGS into QCNet arguments as closely as possible.
#
# Usage:
#   bash scripts/train_qcnet_interaction_digir_aligned.sh
#
# Optional env overrides:
#   DATA_ROOT=/data/sdb/bitwxy/interaction_data
#   SAVE_ROOT=/data/sdb/bitwxy/interaction_runs/gate_compare
#   DATA_FILE=interaction_digir_all_12loc_h8_f12.pkl

DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
SAVE_ROOT="${SAVE_ROOT:-/data/sdb/bitwxy/interaction_runs/gate_compare}"
DATA_FILE="${DATA_FILE:-interaction_digir_all_12loc_h8_f12.pkl}"
DATA_PKL="${DATA_ROOT}/${DATA_FILE}"

python train_qcnet.py \
  --dataset interaction_digir \
  --interaction_data_path "${DATA_PKL}" \
  --save_root "${SAVE_ROOT}" \
  --seed 42 \
  --batch_by_location \
  --max_epochs 20 \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --test_batch_size 8 \
  --num_workers 4 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --train_max_samples 5000 \
  --eval_batches 0 \
  --eval_k 5 \
  --devices 8 \
  --accelerator gpu \
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
