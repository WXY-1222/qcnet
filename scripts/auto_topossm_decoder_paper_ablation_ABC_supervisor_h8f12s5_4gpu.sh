#!/usr/bin/env bash
set -euo pipefail

# Sequential paper-phase structural ablation supervisor:
# A = mode_endpoint without topo score loss
# B = current full mode_endpoint baseline
# C = corridor-conditioned mode_endpoint proposal

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_stage() {
  local stage="$1"
  local cmd="$2"
  log "===== START ${stage} ====="
  bash -lc "${cmd}" >> "${LOG_DIR}/auto_topossm_decoder_paper_ablation_ABC_supervisor_h8f12s5_4gpu_20260506.log" 2>&1 || true
  log "===== END ${stage} ====="
}

log "Paper ablation ABC supervisor started"
run_stage "A_mode_endpoint_noscore" "${SCRIPT_DIR}/auto_topossm_decoder_paper_ablation_A_mode_endpoint_noscore_h8f12s5_4gpu.sh"
run_stage "B_mode_endpoint_full" "${SCRIPT_DIR}/auto_topossm_decoder_paper_ablation_B_mode_endpoint_full_h8f12s5_4gpu.sh"
run_stage "C_corridor_mode_endpoint" "${SCRIPT_DIR}/auto_topossm_decoder_paper_ablation_C_corridor_mode_endpoint_h8f12s5_4gpu.sh"
log "Paper ablation ABC supervisor finished"
