#!/usr/bin/env bash
set -euo pipefail

# PaperA-v5 warm-start smoke test:
# start from the best clean PaperA checkpoint, swap in the new
# mode_endpoint_polyline_readout proposal, and short-run to verify whether
# the structural change has immediate benefit.

REPO="${REPO:-/home/bitwxy/qcnet}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
DATA_FILE="${DATA_FILE:-interaction_digir_all_12loc_h8_f12_s5.pkl}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
SEED="${SEED:-17}"
RUN_TAG="${RUN_TAG:-20260506_paperA_v5_warmstart_seed17}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-180}"

INIT_CKPT="${INIT_CKPT:-${BASE_OUT}/paperA_fresh_h8_f12_s5_k6_4gpu_seed17_20260506_paperA_fresh_seed17_100ep_probe/lightning_logs/version_0/checkpoints/epoch=30-step=4991.ckpt}"
TOPO_MODE_ENDPOINT_SCALE="${TOPO_MODE_ENDPOINT_SCALE:-0.16}"
TOPO_GOAL_RESIDUAL_SCALE="${TOPO_GOAL_RESIDUAL_SCALE:-0.20}"
TOPO_GOAL_ANCHOR_BLEND="${TOPO_GOAL_ANCHOR_BLEND:-0.60}"
TOPO_POLYLINE_CONTROL_SCALE="${TOPO_POLYLINE_CONTROL_SCALE:-0.10}"
TOPO_CORRIDOR_LOSS_WEIGHT="${TOPO_CORRIDOR_LOSS_WEIGHT:-0.02}"
TOPO_SCORE_LOSS_WEIGHT="${TOPO_SCORE_LOSS_WEIGHT:-0.0}"
TOPO_SCORE_TEMPERATURE="${TOPO_SCORE_TEMPERATURE:-0.20}"

MAX_EPOCHS="${MAX_EPOCHS:-4}"
LR="${LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TARGET_ADE="${TARGET_ADE:-0.0605}"
STOP_IF_BEST_GT_E2="${STOP_IF_BEST_GT_E2:-0.0665}"
STOP_IF_BEST_GT_E3="${STOP_IF_BEST_GT_E3:-0.0645}"

SAVE_ROOT="${SAVE_ROOT:-${BASE_OUT}/paperA_v5_warmstart_h8_f12_s5_k6_4gpu_seed${SEED}_${RUN_TAG}}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/paperA_v5_warmstart_h8_f12_s5_k6_4gpu_seed${SEED}_${RUN_TAG}.log}"
CKPT_DIR="${SAVE_ROOT}/lightning_logs/version_0/checkpoints"

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

float_gt() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
}

if ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep | \
   grep -q "/data/sdb/bitwxy/qcnet_data"; then
  log "Another QCNet training process is active; refusing to launch a concurrent run."
  ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep || true
  exit 3
fi

if [[ -e "${SAVE_ROOT}/lightning_logs/version_0" ]]; then
  log "Refusing to overwrite existing run: ${SAVE_ROOT}/lightning_logs/version_0"
  exit 2
fi

if [[ ! -f "${INIT_CKPT}" ]]; then
  log "Missing init checkpoint: ${INIT_CKPT}"
  exit 4
fi

log "PaperA-v5 warm-start smoke test started"
log "Repo=${REPO} GPUs=${GPU_IDS} seed=${SEED}"
log "init_ckpt=${INIT_CKPT}"
log "save_root=${SAVE_ROOT}"
log "run_log=${RUN_LOG}"
log "config=proposal=mode_endpoint_polyline_readout endpoint_scale=${TOPO_MODE_ENDPOINT_SCALE} goal_residual_scale=${TOPO_GOAL_RESIDUAL_SCALE} anchor_blend=${TOPO_GOAL_ANCHOR_BLEND} polyline_scale=${TOPO_POLYLINE_CONTROL_SCALE} lr=${LR}"

"${PYTHON_BIN}" train_qcnet.py \
  --dataset interaction_digir \
  --interaction_data_path "${DATA_ROOT}/${DATA_FILE}" \
  --save_root "${SAVE_ROOT}" \
  --init_from_checkpoint "${INIT_CKPT}" \
  --seed "${SEED}" \
  --batch_by_location \
  --max_epochs "${MAX_EPOCHS}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --test_batch_size "${TEST_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --pin_memory true \
  --persistent_workers true \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
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
  --topo_proposal_type mode_endpoint_polyline_readout \
  --topo_ssm_layers 2 \
  --topo_mamba_d_state 16 \
  --topo_mamba_d_conv 4 \
  --topo_mamba_expand 2 \
  --topo_mode_endpoint_scale "${TOPO_MODE_ENDPOINT_SCALE}" \
  --topo_goal_residual_scale "${TOPO_GOAL_RESIDUAL_SCALE}" \
  --topo_goal_anchor_blend "${TOPO_GOAL_ANCHOR_BLEND}" \
  --topo_polyline_control_scale "${TOPO_POLYLINE_CONTROL_SCALE}" \
  --topo_corridor_loss_weight "${TOPO_CORRIDOR_LOSS_WEIGHT}" \
  --topo_score_loss_weight "${TOPO_SCORE_LOSS_WEIGHT}" \
  --topo_score_temperature "${TOPO_SCORE_TEMPERATURE}" \
  > "${RUN_LOG}" 2>&1 &

pid="$!"
log "train PID ${pid}"

while kill -0 "${pid}" 2>/dev/null; do
  sleep "${MONITOR_INTERVAL_SEC}"
  metrics="$(metric_json "${CKPT_DIR}")"
  latest_epoch="$(json_value "${metrics}" latest_epoch)"
  latest_ade="$(json_value "${metrics}" latest_ade)"
  best_ade="$(json_value "${metrics}" best_ade)"
  best_path="$(json_value "${metrics}" best_path)"
  log "latest_epoch=${latest_epoch:-na} latest_ade=${latest_ade:-na} best_ade=${best_ade:-na} best_path=${best_path:-na}"
  if [[ -n "${best_ade}" ]] && float_le "${best_ade}" "${TARGET_ADE}"; then
    log "Positive warm-start signal: best_ade=${best_ade} <= ${TARGET_ADE}"
  fi
  if [[ -n "${best_ade}" && -n "${latest_epoch}" ]]; then
    if [[ "${latest_epoch}" -ge 2 ]] && float_gt "${best_ade}" "${STOP_IF_BEST_GT_E2}"; then
      log "Early stop: warm-start best_ade=${best_ade} is still above ${STOP_IF_BEST_GT_E2} at epoch ${latest_epoch}"
      kill "${pid}" 2>/dev/null || true
      break
    fi
    if [[ "${latest_epoch}" -ge 3 ]] && float_gt "${best_ade}" "${STOP_IF_BEST_GT_E3}"; then
      log "Early stop: warm-start best_ade=${best_ade} is still above ${STOP_IF_BEST_GT_E3} at epoch ${latest_epoch}"
      kill "${pid}" 2>/dev/null || true
      break
    fi
  fi
done

wait "${pid}" || true
metrics="$(metric_json "${CKPT_DIR}")"
log "Finished metrics=${metrics}"
