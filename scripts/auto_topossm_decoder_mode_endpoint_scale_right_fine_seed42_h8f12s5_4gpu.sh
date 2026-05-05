#!/usr/bin/env bash
set -euo pipefail

# Fine right-side scale sweep after the independently confirmed scale=0.13 result.
# It intentionally wraps the existing refine supervisor instead of modifying it, so
# previous scripts and run directories remain untouched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SCALES="${SCALES:-0.135 0.14 0.145}"
export TARGET_ADE="${TARGET_ADE:-0.0470}"
export BASELINE_ADE="${BASELINE_ADE:-0.04781242}"
export SWITCH_IF_AFTER_EPOCH="${SWITCH_IF_AFTER_EPOCH:-3}"
export SWITCH_ADE="${SWITCH_ADE:-0.0486}"
export STOP_LATEST_ADE="${STOP_LATEST_ADE:-0.0565}"
export RUN_TAG="${RUN_TAG:-20260505_right_fine}"

exec "${SCRIPT_DIR}/auto_topossm_decoder_mode_endpoint_scale_refine_seed42_h8f12s5_4gpu.sh"
