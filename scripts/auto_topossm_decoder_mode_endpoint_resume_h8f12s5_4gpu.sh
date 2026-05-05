#!/usr/bin/env bash
set -euo pipefail

# True resume from the best mode-specific endpoint TopoSSM checkpoint.
# This keeps the optimizer/scheduler/callback state from the checkpoint instead
# of starting a fresh fine-tuning optimizer.

REPO="${REPO:-/home/bitwxy/qcnet}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
DATA_FILE="${DATA_FILE:-interaction_digir_all_12loc_h8_f12_s5.pkl}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
TARGET_ADE="${TARGET_ADE:-0.0525}"
BASELINE_ADE="${BASELINE_ADE:-0.0532527268}"
STOP_LATEST_ADE="${STOP_LATEST_ADE:-0.061}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-180}"

SAVE_ROOT="${SAVE_ROOT:-${BASE_OUT}/qcnet_topossm_decoder_mode_endpoint_h8_f12_s5_k6_4gpu_20260505}"
CKPT_PATH="${CKPT_PATH:-${SAVE_ROOT}/lightning_logs/version_0/checkpoints/epoch=4-step=1075.ckpt}"
TEACHER_CKPT="${TEACHER_CKPT:-${BASE_OUT}/qcnet_topossm_safetyft_h8_f12_s5_k6_4gpu_20260503/lightning_logs/version_0/checkpoints/epoch=7-step=1288.ckpt}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/qcnet_topossm_decoder_mode_endpoint_resume_h8_f12_s5_k6_4gpu_20260505.log}"
LIGHTNING_ROOT="${SAVE_ROOT}/lightning_logs"

mkdir -p "${LOG_DIR}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
ulimit -n 65535 2>/dev/null || true

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

metric_json() {
  local lightning_root="$1"
  "${PYTHON_BIN}" - "${lightning_root}" <<'PY'
import glob
import json
import os
import sys
import torch

lightning_root = sys.argv[1]
paths = sorted(glob.glob(os.path.join(lightning_root, "version_*", "checkpoints", "*.ckpt")), key=os.path.getmtime)
out = {"latest_epoch": None, "latest_ade": None, "best_ade": None, "best_path": None, "num_ckpts": len(paths)}
if paths:
    ckpt = torch.load(paths[-1], map_location="cpu")
    out["latest_epoch"] = ckpt.get("epoch")
    for value in ckpt.get("callbacks", {}).values():
        if isinstance(value, dict) and value.get("monitor") == "val_minADE":
            cur = value.get("current_score")
            best = value.get("best_model_score")
            out["latest_ade"] = float(cur) if cur is not None else None
            out["best_ade"] = float(best) if best is not None else None
            out["best_path"] = value.get("best_model_path")
            break
print(json.dumps(out, sort_keys=True))
PY
}

json_value() {
  local json="$1"
  local key="$2"
  "${PYTHON_BIN}" - "${json}" "${key}" <<'PY'
import json
import sys
value = json.loads(sys.argv[1]).get(sys.argv[2])
print("" if value is None else value)
PY
}

float_le() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)
PY
}

float_lt() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)
PY
}

stop_train() {
  local pids
  pids="$(pgrep -f "train_qcnet.py.*${SAVE_ROOT}" || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  log "Stopping train processes: ${pids//$'\n'/ }"
  kill -TERM ${pids} 2>/dev/null || true
  sleep 12
  pids="$(pgrep -f "train_qcnet.py.*${SAVE_ROOT}" || true)"
  if [[ -n "${pids}" ]]; then
    log "Force stopping remaining train processes: ${pids//$'\n'/ }"
    kill -9 ${pids} 2>/dev/null || true
  fi
}

if ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep | \
   grep -q "/data/sdb/bitwxy/qcnet_data"; then
  log "Another QCNet training process is active; refusing to launch a concurrent run."
  ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep || true
  exit 3
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  log "Missing checkpoint: ${CKPT_PATH}"
  exit 4
fi

log "TopoSSM mode-endpoint true-resume supervisor started"
log "Repo=${REPO} GPUs=${GPU_IDS} target=${TARGET_ADE} baseline=${BASELINE_ADE}"
log "Resume ckpt=${CKPT_PATH}"
log "Teacher=${TEACHER_CKPT}"
log "SaveRoot=${SAVE_ROOT}"
log "RunLog=${RUN_LOG}"

"${PYTHON_BIN}" train_qcnet.py \
  --dataset interaction_digir \
  --interaction_data_path "${DATA_ROOT}/${DATA_FILE}" \
  --save_root "${SAVE_ROOT}" \
  --ckpt_path "${CKPT_PATH}" \
  --distill_teacher_checkpoint "${TEACHER_CKPT}" \
  --seed 49 \
  --batch_by_location \
  --max_epochs 9 \
  --train_batch_size 12 \
  --val_batch_size 12 \
  --test_batch_size 12 \
  --num_workers 4 \
  --pin_memory false \
  --persistent_workers false \
  --lr 3e-5 \
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
  --decoder_type topossm \
  --topo_proposal_type mode_endpoint \
  --topo_mode_endpoint_scale 0.08 \
  --topo_ssm_layers 2 \
  --topo_mamba_d_state 16 \
  --topo_mamba_d_conv 4 \
  --topo_mamba_expand 2 \
  --topo_corridor_loss_weight 0.02 \
  --topo_score_loss_weight 0.02 \
  --topo_score_temperature 0.20 \
  --distill_propose_weight 0.01 \
  --distill_refine_weight 0.03 \
  --distill_score_weight 0.03 \
  --distill_temperature 1.5 \
  --distill_warmup_epochs 1 \
  --freeze_encoder \
  > "${RUN_LOG}" 2>&1 &

pid="$!"
log "Train PID ${pid}"

while kill -0 "${pid}" 2>/dev/null; do
  sleep "${MONITOR_INTERVAL_SEC}"
  metrics="$(metric_json "${LIGHTNING_ROOT}")"
  latest_epoch="$(json_value "${metrics}" latest_epoch)"
  latest_ade="$(json_value "${metrics}" latest_ade)"
  best_ade="$(json_value "${metrics}" best_ade)"
  best_path="$(json_value "${metrics}" best_path)"
  log "metrics latest_epoch=${latest_epoch:-na} latest_ade=${latest_ade:-na} best_ade=${best_ade:-na} best_path=${best_path:-na}"

  if [[ -n "${best_ade}" ]] && float_le "${best_ade}" "${TARGET_ADE}"; then
    log "Reached target best_ade=${best_ade} <= ${TARGET_ADE}; stopping run."
    stop_train
    exit 0
  fi
  if [[ -n "${latest_epoch}" && -n "${best_ade}" ]] && (( latest_epoch >= 7 )); then
    if ! float_lt "${best_ade}" "${BASELINE_ADE}"; then
      log "No improvement by epoch ${latest_epoch}: best_ade=${best_ade} >= baseline=${BASELINE_ADE}; stopping run."
      stop_train
      exit 2
    fi
  fi
  if [[ -n "${latest_epoch}" && -n "${latest_ade}" ]] && (( latest_epoch >= 6 )); then
    if ! float_le "${latest_ade}" "${STOP_LATEST_ADE}"; then
      log "Unstable latest_ade=${latest_ade} > ${STOP_LATEST_ADE}; stopping run."
      stop_train
      exit 2
    fi
  fi
done

wait "${pid}" || true
metrics="$(metric_json "${LIGHTNING_ROOT}")"
log "Training process exited. metrics=${metrics}"
best_ade="$(json_value "${metrics}" best_ade)"
if [[ -n "${best_ade}" ]] && float_le "${best_ade}" "${TARGET_ADE}"; then
  exit 0
fi
exit 1
