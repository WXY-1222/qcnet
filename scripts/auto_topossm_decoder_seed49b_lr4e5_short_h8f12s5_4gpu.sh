#!/usr/bin/env bash
set -euo pipefail

# Short continuation from Seed49B best.
# Goal: test whether a slightly larger LR can improve the full TopoSSM decoder
# without increasing topology loss strength.

REPO="${REPO:-/home/bitwxy/qcnet}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
DATA_FILE="${DATA_FILE:-interaction_digir_all_12loc_h8_f12_s5.pkl}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
TARGET_ADE="${TARGET_ADE:-0.0525}"
MAX_EPOCHS="${MAX_EPOCHS:-6}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-180}"

INIT_CKPT="${INIT_CKPT:-${BASE_OUT}/qcnet_topossm_decoder_B10_light_distill_seed49_h8_f12_s5_k6_4gpu_20260504/lightning_logs/version_0/checkpoints/epoch=5-step=1290.ckpt}"
TEACHER_CKPT="${TEACHER_CKPT:-${BASE_OUT}/qcnet_topossm_safetyft_h8_f12_s5_k6_4gpu_20260503/lightning_logs/version_0/checkpoints/epoch=7-step=1288.ckpt}"
SAVE_ROOT="${SAVE_ROOT:-${BASE_OUT}/qcnet_topossm_decoder_seed49b_lr4e5_short_h8_f12_s5_k6_4gpu_20260504}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/qcnet_topossm_decoder_seed49b_lr4e5_short_h8_f12_s5_k6_4gpu_20260504.log}"
CKPT_DIR="${SAVE_ROOT}/lightning_logs/version_0/checkpoints"

LR="${LR:-4e-5}"

mkdir -p "${LOG_DIR}" "${SAVE_ROOT}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
ulimit -n 65535 2>/dev/null || true

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

metric_json() {
  local ckpt_dir="$1"
  "${PYTHON_BIN}" - "${ckpt_dir}" <<'PY'
import glob
import json
import os
import sys
import torch

ckpt_dir = sys.argv[1]
paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime)
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

kill_run() {
  local pids
  pids="$(pgrep -f "train_qcnet.py.*${SAVE_ROOT}" || true)"
  if [[ -n "${pids}" ]]; then
    log "Stopping run processes: ${pids//$'\n'/ }"
    kill -TERM ${pids} 2>/dev/null || true
    sleep 10
    pids="$(pgrep -f "train_qcnet.py.*${SAVE_ROOT}" || true)"
    if [[ -n "${pids}" ]]; then
      kill -9 ${pids} 2>/dev/null || true
    fi
  fi
}

if ps -eo cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep | grep -q "bitwxy/qcnet"; then
  log "Another bitwxy QCNet training process is active; refusing to launch a concurrent run."
  ps -eo pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep | grep "bitwxy/qcnet" || true
  exit 3
fi

rm -rf "${SAVE_ROOT}/lightning_logs/version_0"

log "Seed49B short LR sweep started"
log "Repo=${REPO} GPUs=${GPU_IDS} target=${TARGET_ADE} lr=${LR}"
log "init=${INIT_CKPT}"
log "teacher=${TEACHER_CKPT}"
log "save_root=${SAVE_ROOT}"
log "run_log=${RUN_LOG}"

"${PYTHON_BIN}" train_qcnet.py \
  --dataset interaction_digir \
  --interaction_data_path "${DATA_ROOT}/${DATA_FILE}" \
  --save_root "${SAVE_ROOT}" \
  --init_from_checkpoint "${INIT_CKPT}" \
  --distill_teacher_checkpoint "${TEACHER_CKPT}" \
  --seed 49 \
  --batch_by_location \
  --max_epochs "${MAX_EPOCHS}" \
  --train_batch_size 12 \
  --val_batch_size 12 \
  --test_batch_size 12 \
  --num_workers 4 \
  --pin_memory false \
  --persistent_workers false \
  --lr "${LR}" \
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
  --topo_proposal_type goal_mlp \
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
  > "${RUN_LOG}" 2>&1 &
pid="$!"
log "PID ${pid}"

while kill -0 "${pid}" 2>/dev/null; do
  sleep "${MONITOR_INTERVAL_SEC}"
  metrics="$(metric_json "${CKPT_DIR}")"
  latest_epoch="$(json_value "${metrics}" latest_epoch)"
  latest_ade="$(json_value "${metrics}" latest_ade)"
  best_ade="$(json_value "${metrics}" best_ade)"
  best_path="$(json_value "${metrics}" best_path)"
  log "latest_epoch=${latest_epoch:-na} latest_ade=${latest_ade:-na} best_ade=${best_ade:-na} best_path=${best_path:-na}"

  if [[ -n "${best_ade}" ]] && float_le "${best_ade}" "${TARGET_ADE}"; then
    log "Reached target best_ade=${best_ade} <= ${TARGET_ADE}; stopping early."
    kill_run
    exit 0
  fi
  if [[ -n "${latest_epoch}" && -n "${best_ade}" ]] && (( latest_epoch >= 4 )); then
    if ! float_le "${best_ade}" 0.056; then
      log "Behind schedule at epoch ${latest_epoch}: best_ade=${best_ade} > 0.056; stopping."
      kill_run
      exit 2
    fi
  fi
done

wait "${pid}" || true
metrics="$(metric_json "${CKPT_DIR}")"
best_ade="$(json_value "${metrics}" best_ade)"
log "Finished metrics=${metrics}"
if [[ -n "${best_ade}" ]] && float_le "${best_ade}" "${TARGET_ADE}"; then
  exit 0
fi
exit 1
