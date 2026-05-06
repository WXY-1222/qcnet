#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/bitwxy/qcnet}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
DATA_PATH="${DATA_PATH:-/data/sdb/bitwxy/interaction_data/interaction_digir_all_12loc_h8_f12_s5.pkl}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
GPU_ID="${GPU_ID:-4}"
RUN_TAG="${RUN_TAG:-20260506_paperA_clean_eval}"

PAPERA_CKPT="${PAPERA_CKPT:-${BASE_OUT}/qcnet_topossm_decoder_paperA_mode_endpoint_noscore_seed77_h8_f12_s5_k6_4gpu_20260506_paperA_mode_endpoint_noscore/lightning_logs/version_0/checkpoints/epoch=1-step=430.ckpt}"
SAFETYFT_CKPT="${SAFETYFT_CKPT:-${BASE_OUT}/qcnet_topossm_safetyft_h8_f12_s5_k6_4gpu_20260503/lightning_logs/version_0/checkpoints/epoch=7-step=1288.ckpt}"
QCNET_CKPT="${QCNET_CKPT:-${BASE_OUT}/qcnet_h8_f12_s5_k6_4gpu_repro_20260503/lightning_logs/version_0/checkpoints/epoch=14-step=2415.ckpt}"

OUT_ROOT="${OUT_ROOT:-${BASE_OUT}/paperA_clean_eval_${RUN_TAG}}"
SAFETY_OUT="${OUT_ROOT}/safety_table"
DIAG_OUT="${OUT_ROOT}/proposal_diagnostics"
COVERAGE_OUT="${OUT_ROOT}/coverage_decomposition"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"
cd "${REPO}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    log "Missing required file: ${path}"
    exit 2
  fi
}

if ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep | \
   grep -q "/data/sdb/bitwxy/qcnet_data"; then
  log "QCNet training is active; refusing to overlap eval with training."
  ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep || true
  exit 3
fi

if pgrep -af "eval_topossm_|eval_qcnet_val_metrics.py" >/dev/null 2>&1; then
  log "Another eval job is already active; refusing to start a second one."
  pgrep -af "eval_topossm_|eval_qcnet_val_metrics.py" || true
  exit 4
fi

require_file "${PYTHON_BIN}"
require_file "${DATA_PATH}"
require_file "${PAPERA_CKPT}"
require_file "${SAFETYFT_CKPT}"
require_file "${QCNET_CKPT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

log "Repo=${REPO}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "OUT_ROOT=${OUT_ROOT}"
log "paperA=${PAPERA_CKPT}"
log "safetyft=${SAFETYFT_CKPT}"
log "qcnet=${QCNET_CKPT}"

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for TopoSSM eval")
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
PY

log "Running safety/topology table"
"${PYTHON_BIN}" scripts/eval_topossm_safety_table.py \
  --data "${DATA_PATH}" \
  --out_dir "${SAFETY_OUT}" \
  --device cuda:0 \
  --batch_size 16 \
  --num_workers 4 \
  --model "paperA=${PAPERA_CKPT}" \
  --model "safetyft=${SAFETYFT_CKPT}" \
  --model "qcnet=${QCNET_CKPT}"

log "Running proposal diagnostics"
"${PYTHON_BIN}" scripts/eval_topossm_proposal_diagnostics.py \
  --data "${DATA_PATH}" \
  --student_ckpt "${PAPERA_CKPT}" \
  --teacher_ckpt "${SAFETYFT_CKPT}" \
  --out_dir "${DIAG_OUT}" \
  --device cuda:0 \
  --batch_size 16 \
  --num_workers 4

log "Running coverage decomposition"
"${PYTHON_BIN}" scripts/eval_topossm_coverage_decomposition.py \
  --data "${DATA_PATH}" \
  --student_ckpt "${PAPERA_CKPT}" \
  --teacher_ckpt "${SAFETYFT_CKPT}" \
  --out_dir "${COVERAGE_OUT}" \
  --device cuda:0 \
  --batch_size 32 \
  --num_workers 4

log "PaperA clean eval finished"
log "Safety table: ${SAFETY_OUT}"
log "Proposal diagnostics: ${DIAG_OUT}"
log "Coverage decomposition: ${COVERAGE_OUT}"
