#!/usr/bin/env bash
set -euo pipefail

# C: push corridor / homotopy information into proposal selection itself.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROPOSAL_TYPE="corridor_mode_endpoint"
export PROPOSAL_TAG="paperC_corridor_mode_endpoint"
export TOPO_MODE_ENDPOINT_SCALE="${TOPO_MODE_ENDPOINT_SCALE:-0.16}"
export TOPO_CORRIDOR_LOSS_WEIGHT="${TOPO_CORRIDOR_LOSS_WEIGHT:-0.02}"
export TOPO_SCORE_LOSS_WEIGHT="${TOPO_SCORE_LOSS_WEIGHT:-0.02}"
export RUN_TAG="${RUN_TAG:-20260506_paperC_corridor_mode_endpoint}"
export SEEDS="${SEEDS:-42 77}"
export BASELINE_ADE="${BASELINE_ADE:-0.04384515}"
export TARGET_ADE="${TARGET_ADE:-0.0433}"

"${SCRIPT_DIR}/auto_topossm_decoder_paper_ablation_seed_sweep_h8f12s5_4gpu.sh"
