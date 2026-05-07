#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/bitwxy/qcnet}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
DATA_FILE="${DATA_FILE:-interaction_digir_all_12loc_h8_f12_s5.pkl}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
SEED="${SEED:-17}"
RUN_TAG="${RUN_TAG:-20260507_paperA_v10_soft_route_slot_fresh20_seed17}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-180}"
STALL_EPOCH_SEC="${STALL_EPOCH_SEC:-900}"
STALL_LOG_SEC="${STALL_LOG_SEC:-900}"

TOPO_MODE_ENDPOINT_SCALE="${TOPO_MODE_ENDPOINT_SCALE:-0.16}"
TOPO_GOAL_ANCHOR_BLEND="${TOPO_GOAL_ANCHOR_BLEND:-0.65}"
TOPO_POLYLINE_CONTROL_SCALE="${TOPO_POLYLINE_CONTROL_SCALE:-0.05}"
TOPO_ROUTE_SLOT_LONGITUDINAL_SCALE="${TOPO_ROUTE_SLOT_LONGITUDINAL_SCALE:-0.12}"
TOPO_ROUTE_SLOT_LATERAL_SCALE="${TOPO_ROUTE_SLOT_LATERAL_SCALE:-0.06}"
TOPO_ROUTE_SLOT_TOPK="${TOPO_ROUTE_SLOT_TOPK:-12}"
TOPO_ROUTE_SLOT_SOFT_TEMPERATURE="${TOPO_ROUTE_SLOT_SOFT_TEMPERATURE:-0.35}"
TOPO_CORRIDOR_LOSS_WEIGHT="${TOPO_CORRIDOR_LOSS_WEIGHT:-0.02}"
TOPO_SCORE_LOSS_WEIGHT="${TOPO_SCORE_LOSS_WEIGHT:-0.0}"
TOPO_SCORE_TEMPERATURE="${TOPO_SCORE_TEMPERATURE:-0.20}"

MAX_EPOCHS="${MAX_EPOCHS:-20}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TARGET_ADE="${TARGET_ADE:-0.058}"
STOP_IF_BEST_GT_E4="${STOP_IF_BEST_GT_E4:-0.130}"
STOP_IF_BEST_GT_E8="${STOP_IF_BEST_GT_E8:-0.095}"

SAVE_ROOT="${SAVE_ROOT:-${BASE_OUT}/paperA_v10_soft_route_slot_fresh20_h8_f12_s5_k6_4gpu_seed${SEED}_${RUN_TAG}}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/paperA_v10_soft_route_slot_fresh20_h8_f12_s5_k6_4gpu_seed${SEED}_${RUN_TAG}.log}"
CKPT_DIR="${SAVE_ROOT}/lightning_logs/version_0/checkpoints"

mkdir -p "${LOG_DIR}" "${SAVE_ROOT}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
ulimit -n 65535 2>/dev/null || true

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

file_mtime_or_zero() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    stat -c %Y "${path}"
  else
    echo 0
  fi
}

kill_run_tree() {
  local root_pid="$1"
  local reason="$2"
  log "Stopping run tree for PID ${root_pid}: ${reason}"
  kill -TERM -- "-${root_pid}" 2>/dev/null || true
  sleep 5
  kill -KILL -- "-${root_pid}" 2>/dev/null || true
}

metric_json() {
  local ckpt_dir="$1"
  "${PYTHON_BIN}" - "${ckpt_dir}" <<'PY'
import glob, json, os, sys, torch
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
import json, sys
value = json.loads(sys.argv[1]).get(sys.argv[2])
print("" if value is None else value)
PY
}

float_le() { "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)
PY
}

float_gt() { "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
}

monitor_run() {
  local pid="$1"
  local ckpt_dir="$2"
  local run_log="$3"
  local last_epoch_seen=""
  local last_epoch_change_ts
  last_epoch_change_ts="$(date +%s)"

  while true; do
    sleep "${MONITOR_INTERVAL_SEC}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      log "Monitor noticed PID ${pid} has exited"
      break
    fi

    local now_ts metrics latest_epoch latest_ade best_ade best_path log_mtime log_silence_sec epoch_stall_sec
    now_ts="$(date +%s)"
    if ! metrics="$(metric_json "${ckpt_dir}")"; then
      log "Monitor warning: failed to read checkpoint metrics from ${ckpt_dir}"
      continue
    fi
    latest_epoch="$(json_value "${metrics}" latest_epoch)"
    latest_ade="$(json_value "${metrics}" latest_ade)"
    best_ade="$(json_value "${metrics}" best_ade)"
    best_path="$(json_value "${metrics}" best_path)"
    log "latest_epoch=${latest_epoch:-na} latest_ade=${latest_ade:-na} best_ade=${best_ade:-na} best_path=${best_path:-na}"

    if [[ -n "${latest_epoch}" && "${latest_epoch}" != "${last_epoch_seen}" ]]; then
      last_epoch_seen="${latest_epoch}"
      last_epoch_change_ts="${now_ts}"
    fi
    if [[ -n "${best_ade}" ]] && float_le "${best_ade}" "${TARGET_ADE}"; then
      log "Target reached: best_ade=${best_ade} <= ${TARGET_ADE}"
    fi

    log_mtime="$(file_mtime_or_zero "${run_log}")"
    log_silence_sec="$(( now_ts - log_mtime ))"
    epoch_stall_sec="$(( now_ts - last_epoch_change_ts ))"
    if (( log_silence_sec >= STALL_LOG_SEC )); then
      kill_run_tree "${pid}" "run log silent for ${log_silence_sec}s"
      break
    fi
    if [[ -n "${last_epoch_seen}" ]] && (( epoch_stall_sec >= STALL_EPOCH_SEC )); then
      kill_run_tree "${pid}" "latest_epoch=${last_epoch_seen} stalled for ${epoch_stall_sec}s"
      break
    fi
    if [[ -n "${latest_epoch}" && -n "${best_ade}" ]]; then
      if (( latest_epoch >= 4 )) && float_gt "${best_ade}" "${STOP_IF_BEST_GT_E4}"; then
        log "Early stop: best_ade=${best_ade} is still above ${STOP_IF_BEST_GT_E4} at epoch ${latest_epoch}"
        kill_run_tree "${pid}" "failed epoch-4 threshold"
        break
      fi
      if (( latest_epoch >= 8 )) && float_gt "${best_ade}" "${STOP_IF_BEST_GT_E8}"; then
        log "Early stop: best_ade=${best_ade} is still above ${STOP_IF_BEST_GT_E8} at epoch ${latest_epoch}"
        kill_run_tree "${pid}" "failed epoch-8 threshold"
        break
      fi
    fi
  done
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

log "PaperA-v10 soft-route-slot fresh 20-epoch run started"
log "Repo=${REPO} GPUs=${GPU_IDS} seed=${SEED}"
log "save_root=${SAVE_ROOT}"
log "run_log=${RUN_LOG}"
log "config=proposal=soft_route_slot_polyline endpoint_scale=${TOPO_MODE_ENDPOINT_SCALE} slot_long=${TOPO_ROUTE_SLOT_LONGITUDINAL_SCALE} slot_lat=${TOPO_ROUTE_SLOT_LATERAL_SCALE} topk=${TOPO_ROUTE_SLOT_TOPK} temp=${TOPO_ROUTE_SLOT_SOFT_TEMPERATURE} lr=${LR}"

setsid "${PYTHON_BIN}" train_qcnet.py \
  --dataset interaction_digir \
  --interaction_data_path "${DATA_ROOT}/${DATA_FILE}" \
  --save_root "${SAVE_ROOT}" \
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
  --topo_proposal_type soft_route_slot_polyline \
  --topo_ssm_layers 2 \
  --topo_mamba_d_state 16 \
  --topo_mamba_d_conv 4 \
  --topo_mamba_expand 2 \
  --topo_mode_endpoint_scale "${TOPO_MODE_ENDPOINT_SCALE}" \
  --topo_goal_anchor_blend "${TOPO_GOAL_ANCHOR_BLEND}" \
  --topo_polyline_control_scale "${TOPO_POLYLINE_CONTROL_SCALE}" \
  --topo_route_slot_longitudinal_scale "${TOPO_ROUTE_SLOT_LONGITUDINAL_SCALE}" \
  --topo_route_slot_lateral_scale "${TOPO_ROUTE_SLOT_LATERAL_SCALE}" \
  --topo_route_slot_topk "${TOPO_ROUTE_SLOT_TOPK}" \
  --topo_route_slot_soft_temperature "${TOPO_ROUTE_SLOT_SOFT_TEMPERATURE}" \
  --topo_corridor_loss_weight "${TOPO_CORRIDOR_LOSS_WEIGHT}" \
  --topo_score_loss_weight "${TOPO_SCORE_LOSS_WEIGHT}" \
  --topo_score_temperature "${TOPO_SCORE_TEMPERATURE}" \
  > "${RUN_LOG}" 2>&1 &

pid="$!"
log "train PID ${pid}"
monitor_run "${pid}" "${CKPT_DIR}" "${RUN_LOG}" &
monitor_pid="$!"

wait "${pid}" || true
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
metrics="$(metric_json "${CKPT_DIR}")"
log "Finished metrics=${metrics}"
