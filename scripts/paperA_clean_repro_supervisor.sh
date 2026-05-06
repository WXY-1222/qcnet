#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO="${REPO:-/home/bitwxy/qcnet}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-60}"
RUN_TAG="${RUN_TAG:-20260506_paperA_mode_endpoint_noscore_extra_seeds_clean}"
SEEDS="${SEEDS:-17 23}"
BASELINE_ADE="${BASELINE_ADE:-0.04288549}"
TARGET_ADE="${TARGET_ADE:-0.0425}"

mkdir -p "${LOG_DIR}"
cd "${REPO}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_until_idle() {
  while true; do
    local active
    active="$(ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E 'train_qcnet.py|torchrun|eval_topossm_|eval_qcnet_val_metrics.py' | grep -v grep || true)"
    if [[ -z "${active}" ]]; then
      return 0
    fi
    log "Waiting for active QCNet job(s) to finish before extra-seed repro:"
    echo "${active}"
    sleep "${WAIT_INTERVAL_SEC}"
  done
}

log "PaperA clean repro supervisor started"
log "seeds=${SEEDS} baseline=${BASELINE_ADE} target=${TARGET_ADE}"
wait_until_idle

export RUN_TAG
export SEEDS
export BASELINE_ADE
export TARGET_ADE

"${SCRIPT_DIR}/auto_topossm_decoder_paper_ablation_A_mode_endpoint_noscore_h8f12s5_4gpu.sh"

log "PaperA clean repro supervisor finished"
