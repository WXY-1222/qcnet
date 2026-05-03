#!/usr/bin/env bash
set -euo pipefail

# QC-DIGIR Topology-SSM probe:
# QCNet proposals/refinement + homotopy/KG corridor binding + Bi-Mamba residual/scoring.

DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
SAVE_ROOT="${SAVE_ROOT:-/data/sdb/bitwxy/qcnet_data/qcnet_topossm_h8_f12_s5_k6_4gpu_20260503}"
DATA_FILE="${DATA_FILE:-interaction_digir_all_12loc_h8_f12_s5.pkl}"
DATA_PKL="${DATA_ROOT}/${DATA_FILE}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
export CUDA_VISIBLE_DEVICES
ulimit -n 65535 2>/dev/null || true

"${PYTHON_BIN}" train_qcnet.py \
  --dataset interaction_digir \
  --interaction_data_path "${DATA_PKL}" \
  --save_root "${SAVE_ROOT}" \
  --seed 42 \
  --batch_by_location \
  --max_epochs "${MAX_EPOCHS:-20}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-16}" \
  --val_batch_size "${VAL_BATCH_SIZE:-16}" \
  --test_batch_size "${TEST_BATCH_SIZE:-16}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --pin_memory false \
  --persistent_workers false \
  --lr "${LR:-1e-4}" \
  --weight_decay 1e-4 \
  --eval_batches 0 \
  --num_modes 6 \
  --eval_k 6 \
  --monitor_metric val_minADE \
  --monitor_mode min \
  --devices 4 \
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
  --a2m_radius 80 \
  --enable_topo_ssm_refiner \
  --topo_refine_weight "${TOPO_REFINE_WEIGHT:-0.10}" \
  --topo_score_weight "${TOPO_SCORE_WEIGHT:-0.10}" \
  --topo_ssm_layers "${TOPO_SSM_LAYERS:-1}" \
  --topo_mamba_d_state "${TOPO_MAMBA_D_STATE:-16}" \
  --topo_mamba_d_conv "${TOPO_MAMBA_D_CONV:-4}" \
  --topo_mamba_expand "${TOPO_MAMBA_EXPAND:-2}"
