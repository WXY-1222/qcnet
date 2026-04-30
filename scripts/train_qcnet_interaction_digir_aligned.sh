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
#   DATA_FILE=interaction_digir_all_12loc_h8_f12_s5.pkl

DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
SAVE_ROOT="${SAVE_ROOT:-/data/sdb/bitwxy/interaction_runs/gate_compare}"
DATA_FILE="${DATA_FILE:-interaction_digir_all_12loc_h8_f12_s5.pkl}"
DATA_PKL="${DATA_ROOT}/${DATA_FILE}"
NUM_HISTORICAL_STEPS="${NUM_HISTORICAL_STEPS:-8}"
NUM_FUTURE_STEPS="${NUM_FUTURE_STEPS:-12}"
NUM_MODES="${NUM_MODES:-6}"
EVAL_K="${EVAL_K:-6}"

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
  --eval_batches 0 \
  --num_modes "${NUM_MODES}" \
  --eval_k "${EVAL_K}" \
  --monitor_metric val_minADE \
  --monitor_mode min \
  --devices 8 \
  --accelerator gpu \
  --num_historical_steps "${NUM_HISTORICAL_STEPS}" \
  --num_future_steps "${NUM_FUTURE_STEPS}" \
  --num_recurrent_steps 3 \
  --pl2pl_radius 80 \
  --time_span 8 \
  --pl2a_radius 50 \
  --a2a_radius 50 \
  --num_t2m_steps 8 \
  --pl2m_radius 80 \
  --a2m_radius 80
