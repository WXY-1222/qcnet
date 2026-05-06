#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO="${REPO:-/home/bitwxy/qcnet}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
RUN_TAG="${RUN_TAG:-20260507_paperA_clean_scale_seed23_sweep}"
SEED="${SEED:-23}"
SCALES="${SCALES:-0.12 0.16 0.20}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-60}"
BEST_OVERALL_ADE=""
BEST_OVERALL_SCALE=""
BEST_OVERALL_PATH=""

mkdir -p "${LOG_DIR}"
cd "${REPO}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUP_LOG}"
}

metric_best() {
  local ckpt_dir="$1"
  "${PYTHON_BIN}" - "${ckpt_dir}" <<'PY'
import glob
import json
import os
import sys
import torch

ckpt_dir = sys.argv[1]
paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime)
out = {"best": None, "path": None}
if paths:
    ckpt = torch.load(paths[-1], map_location="cpu")
    for value in ckpt.get("callbacks", {}).values():
        if isinstance(value, dict) and value.get("monitor") == "val_minADE":
            score = value.get("best_model_score")
            out["best"] = float(score) if score is not None else None
            out["path"] = value.get("best_model_path")
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

wait_until_idle() {
  while true; do
    local active
    active="$(ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E 'train_qcnet.py|torchrun' | grep -v grep | grep '/data/sdb/bitwxy/qcnet_data' || true)"
    if [[ -z "${active}" ]]; then
      return 0
    fi
    log "Waiting for active QCNet job(s) to finish:"
    echo "${active}" | tee -a "${SUP_LOG}"
    sleep "${WAIT_INTERVAL_SEC}"
  done
}

update_best() {
  local scale="$1"
  local best="$2"
  local path="$3"
  if [[ -z "${best}" || -z "${path}" ]]; then
    return 0
  fi
  if [[ -z "${BEST_OVERALL_ADE}" ]] || float_le "${best}" "${BEST_OVERALL_ADE}"; then
    BEST_OVERALL_ADE="${best}"
    BEST_OVERALL_SCALE="${scale}"
    BEST_OVERALL_PATH="${path}"
  fi
}

SUP_LOG="${LOG_DIR}/auto_paperA_clean_scale_seed23_supervisor_h8f12s5_4gpu_${RUN_TAG}.log"

log "PaperA clean scale seed23 supervisor started"
log "seed=${SEED} scales=${SCALES}"
wait_until_idle

for scale in ${SCALES}; do
  scale_tag="${scale//./p}"
  save_root="${BASE_OUT}/paperA_clean_scale${scale_tag}_h8_f12_s5_k6_4gpu_seed${SEED}_${RUN_TAG}"
  ckpt_dir="${save_root}/lightning_logs/version_0/checkpoints"
  log "===== START scale=${scale} ====="
  RUN_TAG="${RUN_TAG}" SEED="${SEED}" TOPO_MODE_ENDPOINT_SCALE="${scale}" \
    bash "${SCRIPT_DIR}/auto_paperA_clean_scale_probe_h8f12s5_4gpu.sh" >> "${SUP_LOG}" 2>&1 || true
  result="$(metric_best "${ckpt_dir}")"
  best="$(json_value "${result}" best)"
  path="$(json_value "${result}" path)"
  log "scale=${scale} best_ade=${best:-na} best_path=${path:-na}"
  update_best "${scale}" "${best}" "${path}"
done

log "PaperA clean scale seed23 supervisor finished"
log "Best overall: scale=${BEST_OVERALL_SCALE:-na} ade=${BEST_OVERALL_ADE:-na} path=${BEST_OVERALL_PATH:-na}"
