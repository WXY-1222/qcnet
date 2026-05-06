#!/usr/bin/env bash
set -euo pipefail

# Fair QCNet seed-stability sweep using the same 4-GPU recipe as
# qcnet_h8_f12_s5_k6_4gpu_repro_20260503.

REPO="${REPO:-/home/bitwxy/qcnet}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
DATA_PATH="${DATA_PATH:-/data/sdb/bitwxy/interaction_data/interaction_digir_all_12loc_h8_f12_s5.pkl}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
SEEDS="${SEEDS:-17 23 42 77}"
RUN_TAG="${RUN_TAG:-20260506_qcnet_seed_stability}"
BASELINE_ADE="${BASELINE_ADE:-0.04878547}"
TARGET_ADE="${TARGET_ADE:-0.0465}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-120}"

mkdir -p "${LOG_DIR}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
ulimit -n 65535 2>/dev/null || true

BEST_OVERALL_ADE=""
BEST_OVERALL_PATH=""
BEST_OVERALL_SEED=""

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

update_best_overall() {
  local seed="$1"
  local best_ade="$2"
  local best_path="$3"
  if [[ -z "${best_ade}" || -z "${best_path}" ]]; then
    return 0
  fi
  if [[ -z "${BEST_OVERALL_ADE}" ]] || float_le "${best_ade}" "${BEST_OVERALL_ADE}"; then
    BEST_OVERALL_ADE="${best_ade}"
    BEST_OVERALL_PATH="${best_path}"
    BEST_OVERALL_SEED="${seed}"
  fi
}

run_seed() {
  local seed="$1"
  local save_root="${BASE_OUT}/qcnet_h8_f12_s5_k6_4gpu_seed${seed}_${RUN_TAG}"
  local run_log="${LOG_DIR}/qcnet_h8_f12_s5_k6_4gpu_seed${seed}_${RUN_TAG}.log"
  local ckpt_dir="${save_root}/lightning_logs/version_0/checkpoints"

  if [[ -e "${save_root}/lightning_logs/version_0" ]]; then
    log "Refusing to overwrite existing run: ${save_root}/lightning_logs/version_0"
    return 2
  fi
  mkdir -p "${save_root}"

  log "===== START qcnet seed=${seed} ====="
  log "save_root=${save_root}"
  log "run_log=${run_log}"

  "${PYTHON_BIN}" train_qcnet.py \
    --dataset interaction_digir \
    --interaction_data_path "${DATA_PATH}" \
    --save_root "${save_root}" \
    --seed "${seed}" \
    --batch_by_location \
    --max_epochs "${MAX_EPOCHS}" \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --test_batch_size 16 \
    --num_workers 4 \
    --pin_memory true \
    --persistent_workers true \
    --lr 1e-4 \
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
    > "${run_log}" 2>&1 &

  local pid="$!"
  log "qcnet seed=${seed} train PID ${pid}"

  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${MONITOR_INTERVAL_SEC}"
    local metrics latest_epoch latest_ade best_ade best_path
    metrics="$(metric_json "${ckpt_dir}")"
    latest_epoch="$(json_value "${metrics}" latest_epoch)"
    latest_ade="$(json_value "${metrics}" latest_ade)"
    best_ade="$(json_value "${metrics}" best_ade)"
    best_path="$(json_value "${metrics}" best_path)"
    log "qcnet seed=${seed} latest_epoch=${latest_epoch:-na} latest_ade=${latest_ade:-na} best_ade=${best_ade:-na} best_path=${best_path:-na}"
    update_best_overall "${seed}" "${best_ade}" "${best_path}"
  done

  wait "${pid}" || true
  local metrics best_ade best_path
  metrics="$(metric_json "${ckpt_dir}")"
  best_ade="$(json_value "${metrics}" best_ade)"
  best_path="$(json_value "${metrics}" best_path)"
  update_best_overall "${seed}" "${best_ade}" "${best_path}"
  log "qcnet seed=${seed} finished. metrics=${metrics}"
  return 0
}

if ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep | \
   grep -q "/data/sdb/bitwxy/qcnet_data"; then
  log "Another QCNet training process is active; refusing to launch a concurrent run."
  ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep || true
  exit 3
fi

log "QCNet seed-stability sweep started"
log "Repo=${REPO} GPUs=${GPU_IDS} seeds=${SEEDS} baseline=${BASELINE_ADE} target=${TARGET_ADE}"

for seed in ${SEEDS}; do
  run_seed "${seed}" || true
done

log "QCNet seed-stability sweep finished."
log "Best overall: seed=${BEST_OVERALL_SEED:-na} ade=${BEST_OVERALL_ADE:-na} path=${BEST_OVERALL_PATH:-na}"
if [[ -n "${BEST_OVERALL_ADE}" ]] && float_le "${BEST_OVERALL_ADE}" "${TARGET_ADE}"; then
  log "At least one QCNet seed reached target ${TARGET_ADE}."
  exit 0
fi
exit 0
