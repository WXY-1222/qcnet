#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/bitwxy/qcnet}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
SUP_TAG="${SUP_TAG:-20260506_paperA_v6_lite_gated}"

WARM_SCRIPT="${WARM_SCRIPT:-${REPO}/scripts/auto_paperA_v6_lite_warmstart_h8f12s5_4gpu.sh}"
FRESH_SCRIPT="${FRESH_SCRIPT:-${REPO}/scripts/auto_paperA_v6_lite_fresh20_h8f12s5_4gpu.sh}"
WARM_CKPT_DIR="${WARM_CKPT_DIR:-${BASE_OUT}/paperA_v6_lite_warmstart_h8_f12_s5_k6_4gpu_seed17_20260506_paperA_v6_lite_warmstart_seed17/lightning_logs/version_0/checkpoints}"
PROMOTE_THRESHOLD="${PROMOTE_THRESHOLD:-0.0630}"
SUP_LOG="${SUP_LOG:-${LOG_DIR}/auto_paperA_v6_lite_gated_supervisor_h8f12s5_4gpu_${SUP_TAG}.log}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUP_LOG}"
}

metric_best() {
  "${PYTHON_BIN}" - "${WARM_CKPT_DIR}" <<'PY'
import glob, json, os, sys, torch
ckpt_dir = sys.argv[1]
paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime)
best = None
if paths:
    ckpt = torch.load(paths[-1], map_location="cpu")
    for value in ckpt.get("callbacks", {}).values():
        if isinstance(value, dict) and value.get("monitor") == "val_minADE":
            score = value.get("best_model_score")
            best = float(score) if score is not None else None
            break
print("" if best is None else best)
PY
}

float_le() { "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)
PY
}

log "PaperA-v6 lite gated supervisor started"
log "warm_script=${WARM_SCRIPT}"
log "fresh_script=${FRESH_SCRIPT}"
log "promote_threshold=${PROMOTE_THRESHOLD}"

if [[ ! -x "${WARM_SCRIPT}" ]]; then
  log "Warm script missing or not executable: ${WARM_SCRIPT}"
  exit 2
fi
if [[ ! -x "${FRESH_SCRIPT}" ]]; then
  log "Fresh script missing or not executable: ${FRESH_SCRIPT}"
  exit 2
fi

log "===== START warm-start ====="
"${WARM_SCRIPT}" >> "${SUP_LOG}" 2>&1 || true
best_ade="$(metric_best)"
log "warm_best_ade=${best_ade:-na}"

if [[ -z "${best_ade}" ]]; then
  log "No warm-start checkpoint found; not promoting to fresh run"
  exit 3
fi

if float_le "${best_ade}" "${PROMOTE_THRESHOLD}"; then
  log "Warm-start passed gate; launching fresh 20-epoch run"
  "${FRESH_SCRIPT}" >> "${SUP_LOG}" 2>&1 || true
  log "Fresh run exited"
else
  log "Warm-start failed gate (${best_ade} > ${PROMOTE_THRESHOLD}); skip fresh run"
fi

log "PaperA-v6 lite gated supervisor finished"
