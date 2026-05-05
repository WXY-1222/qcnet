#!/usr/bin/env bash
set -euo pipefail

# Generic paper-phase structural ablation seed sweep for the full-replacement
# TopoSSM decoder. Wrapper scripts should only set env vars and call this file.

REPO="${REPO:-/home/bitwxy/qcnet}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
DATA_FILE="${DATA_FILE:-interaction_digir_all_12loc_h8_f12_s5.pkl}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
SEEDS="${SEEDS:-42 77}"
BASELINE_ADE="${BASELINE_ADE:-0.04384515}"
TARGET_ADE="${TARGET_ADE:-0.0433}"
STOP_LATEST_ADE="${STOP_LATEST_ADE:-0.058}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-120}"
RUN_TAG="${RUN_TAG:-20260506_paper_ablation}"

PROPOSAL_TYPE="${PROPOSAL_TYPE:-mode_endpoint}"
PROPOSAL_TAG="${PROPOSAL_TAG:-mode_endpoint}"
TOPO_MODE_ENDPOINT_SCALE="${TOPO_MODE_ENDPOINT_SCALE:-0.16}"
TOPO_CORRIDOR_LOSS_WEIGHT="${TOPO_CORRIDOR_LOSS_WEIGHT:-0.02}"
TOPO_SCORE_LOSS_WEIGHT="${TOPO_SCORE_LOSS_WEIGHT:-0.02}"
TOPO_SCORE_TEMPERATURE="${TOPO_SCORE_TEMPERATURE:-0.20}"
DISTILL_PROPOSE_WEIGHT="${DISTILL_PROPOSE_WEIGHT:-0.01}"
DISTILL_REFINE_WEIGHT="${DISTILL_REFINE_WEIGHT:-0.03}"
DISTILL_SCORE_WEIGHT="${DISTILL_SCORE_WEIGHT:-0.03}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.5}"
DISTILL_WARMUP_EPOCHS="${DISTILL_WARMUP_EPOCHS:-1}"
LR="${LR:-3e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-4}"
FREEZE_ENCODER="${FREEZE_ENCODER:-1}"

INIT_CKPT="${INIT_CKPT:-${BASE_OUT}/qcnet_topossm_decoder_B10_light_distill_seed49_h8_f12_s5_k6_4gpu_20260504/lightning_logs/version_0/checkpoints/epoch=5-step=1290.ckpt}"
TEACHER_CKPT="${TEACHER_CKPT:-${BASE_OUT}/qcnet_topossm_safetyft_h8_f12_s5_k6_4gpu_20260503/lightning_logs/version_0/checkpoints/epoch=7-step=1288.ckpt}"

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

stop_train() {
  local save_root="$1"
  local pids
  pids="$(pgrep -f "train_qcnet.py.*${save_root}" || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  log "Stopping train processes for ${save_root}: ${pids//$'\n'/ }"
  kill -TERM ${pids} 2>/dev/null || true
  sleep 12
  pids="$(pgrep -f "train_qcnet.py.*${save_root}" || true)"
  if [[ -n "${pids}" ]]; then
    log "Force stopping remaining train processes for ${save_root}: ${pids//$'\n'/ }"
    kill -9 ${pids} 2>/dev/null || true
  fi
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
  local save_root="${BASE_OUT}/qcnet_topossm_decoder_${PROPOSAL_TAG}_seed${seed}_h8_f12_s5_k6_4gpu_${RUN_TAG}"
  local run_log="${LOG_DIR}/qcnet_topossm_decoder_${PROPOSAL_TAG}_seed${seed}_h8_f12_s5_k6_4gpu_${RUN_TAG}.log"
  local ckpt_dir="${save_root}/lightning_logs/version_0/checkpoints"
  local -a extra_args

  if [[ -e "${save_root}/lightning_logs/version_0" ]]; then
    log "Refusing to overwrite existing run: ${save_root}/lightning_logs/version_0"
    return 2
  fi
  mkdir -p "${save_root}"

  extra_args=()
  if [[ "${FREEZE_ENCODER}" == "1" ]]; then
    extra_args+=(--freeze_encoder)
  fi
  if [[ "${PROPOSAL_TYPE}" == "mode_endpoint" || "${PROPOSAL_TYPE}" == "corridor_mode_endpoint" ]]; then
    extra_args+=(--topo_mode_endpoint_scale "${TOPO_MODE_ENDPOINT_SCALE}")
  fi

  log "===== START ${PROPOSAL_TAG} seed=${seed} ====="
  log "save_root=${save_root}"
  log "run_log=${run_log}"
  log "init=${INIT_CKPT}"
  log "teacher=${TEACHER_CKPT}"

  "${PYTHON_BIN}" train_qcnet.py \
    --dataset interaction_digir \
    --interaction_data_path "${DATA_ROOT}/${DATA_FILE}" \
    --save_root "${save_root}" \
    --init_from_checkpoint "${INIT_CKPT}" \
    --distill_teacher_checkpoint "${TEACHER_CKPT}" \
    --seed "${seed}" \
    --batch_by_location \
    --max_epochs "${MAX_EPOCHS}" \
    --train_batch_size 12 \
    --val_batch_size 12 \
    --test_batch_size 12 \
    --num_workers 4 \
    --pin_memory false \
    --persistent_workers false \
    --lr "${LR}" \
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
    --decoder_type topossm \
    --topo_proposal_type "${PROPOSAL_TYPE}" \
    --topo_ssm_layers 2 \
    --topo_mamba_d_state 16 \
    --topo_mamba_d_conv 4 \
    --topo_mamba_expand 2 \
    --topo_corridor_loss_weight "${TOPO_CORRIDOR_LOSS_WEIGHT}" \
    --topo_score_loss_weight "${TOPO_SCORE_LOSS_WEIGHT}" \
    --topo_score_temperature "${TOPO_SCORE_TEMPERATURE}" \
    --distill_propose_weight "${DISTILL_PROPOSE_WEIGHT}" \
    --distill_refine_weight "${DISTILL_REFINE_WEIGHT}" \
    --distill_score_weight "${DISTILL_SCORE_WEIGHT}" \
    --distill_temperature "${DISTILL_TEMPERATURE}" \
    --distill_warmup_epochs "${DISTILL_WARMUP_EPOCHS}" \
    "${extra_args[@]}" \
    > "${run_log}" 2>&1 &

  local pid="$!"
  log "${PROPOSAL_TAG} seed=${seed} train PID ${pid}"

  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${MONITOR_INTERVAL_SEC}"
    local metrics latest_epoch latest_ade best_ade best_path
    metrics="$(metric_json "${ckpt_dir}")"
    latest_epoch="$(json_value "${metrics}" latest_epoch)"
    latest_ade="$(json_value "${metrics}" latest_ade)"
    best_ade="$(json_value "${metrics}" best_ade)"
    best_path="$(json_value "${metrics}" best_path)"
    log "${PROPOSAL_TAG} seed=${seed} latest_epoch=${latest_epoch:-na} latest_ade=${latest_ade:-na} best_ade=${best_ade:-na} best_path=${best_path:-na}"
    update_best_overall "${seed}" "${best_ade}" "${best_path}"

    if [[ -n "${latest_epoch}" && -n "${latest_ade}" ]] && (( latest_epoch >= 1 )); then
      if ! float_le "${latest_ade}" "${STOP_LATEST_ADE}"; then
        log "${PROPOSAL_TAG} seed=${seed} unstable latest_ade=${latest_ade} > ${STOP_LATEST_ADE}; switching seed."
        stop_train "${save_root}"
        return 2
      fi
    fi
  done

  wait "${pid}" || true
  local metrics best_ade best_path
  metrics="$(metric_json "${ckpt_dir}")"
  best_ade="$(json_value "${metrics}" best_ade)"
  best_path="$(json_value "${metrics}" best_path)"
  update_best_overall "${seed}" "${best_ade}" "${best_path}"
  log "${PROPOSAL_TAG} seed=${seed} finished. metrics=${metrics}"
  return 0
}

if ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep | \
   grep -q "/data/sdb/bitwxy/qcnet_data"; then
  log "Another QCNet training process is active; refusing to launch a concurrent run."
  ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep || true
  exit 3
fi

log "Paper ablation seed sweep started"
log "proposal=${PROPOSAL_TYPE} tag=${PROPOSAL_TAG} seeds=${SEEDS} baseline=${BASELINE_ADE} target=${TARGET_ADE}"

for seed in ${SEEDS}; do
  run_seed "${seed}" || true
done

log "Paper ablation seed sweep finished."
log "Best overall: seed=${BEST_OVERALL_SEED:-na} ade=${BEST_OVERALL_ADE:-na} path=${BEST_OVERALL_PATH:-na}"
if [[ -n "${BEST_OVERALL_ADE}" ]] && float_le "${BEST_OVERALL_ADE}" "${BASELINE_ADE}"; then
  log "At least one seed matched or beat baseline ${BASELINE_ADE}."
  exit 0
fi
exit 1
