#!/usr/bin/env bash
set -euo pipefail

# Sequential supervisor:
# 1. run PaperA-v5 warm-start smoke test
# 2. regardless of warm-start outcome, launch the clean fresh 20-epoch main run

REPO="${REPO:-/home/bitwxy/qcnet}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
SUP_TAG="${SUP_TAG:-20260506_paperA_v5_warm_then_fresh}"

WARM_SCRIPT="${WARM_SCRIPT:-${REPO}/scripts/auto_paperA_v5_warmstart_h8f12s5_4gpu.sh}"
FRESH_SCRIPT="${FRESH_SCRIPT:-${REPO}/scripts/auto_paperA_v5_fresh20_h8f12s5_4gpu.sh}"
SUP_LOG="${SUP_LOG:-${LOG_DIR}/auto_paperA_v5_warm_then_fresh_h8f12s5_4gpu_${SUP_TAG}.log}"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUP_LOG}"
}

log "PaperA-v5 sequential supervisor started"
log "warm_script=${WARM_SCRIPT}"
log "fresh_script=${FRESH_SCRIPT}"

if [[ ! -x "${WARM_SCRIPT}" ]]; then
  log "Warm-start script is missing or not executable: ${WARM_SCRIPT}"
  exit 2
fi
if [[ ! -x "${FRESH_SCRIPT}" ]]; then
  log "Fresh script is missing or not executable: ${FRESH_SCRIPT}"
  exit 2
fi

log "===== START warm-start smoke test ====="
if "${WARM_SCRIPT}" >> "${SUP_LOG}" 2>&1; then
  log "Warm-start script exited normally"
else
  log "Warm-start script exited non-zero; continuing to fresh main run"
fi

log "===== START fresh 20-epoch main run ====="
if "${FRESH_SCRIPT}" >> "${SUP_LOG}" 2>&1; then
  log "Fresh script exited normally"
else
  log "Fresh script exited non-zero"
  exit 3
fi

log "PaperA-v5 sequential supervisor finished"
