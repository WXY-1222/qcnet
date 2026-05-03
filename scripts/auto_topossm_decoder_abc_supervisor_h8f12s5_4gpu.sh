#!/usr/bin/env bash
set -euo pipefail

# Sequential TopoSSM decoder follow-up experiments for h8_f12_s5.
# A: pure supervised low-LR fine-tune from the heavy-distill epoch5 best.
# B: very light teacher distillation from the same checkpoint.
# C: clean TopoSSM-decoder checkpoint plus light teacher distillation.

REPO="${REPO:-/home/bitwxy/qcnet}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
DATA_FILE="${DATA_FILE:-interaction_digir_all_12loc_h8_f12_s5.pkl}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
TARGET_ADE="${TARGET_ADE:-0.055}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-180}"

HEAVY_E5_CKPT="${HEAVY_E5_CKPT:-${BASE_OUT}/qcnet_topossm_decoder_distill_h8_f12_s5_k6_4gpu_20260503/lightning_logs/version_0/checkpoints/epoch=5-step=1290.ckpt}"
CLEAN_DECODER_CKPT="${CLEAN_DECODER_CKPT:-${BASE_OUT}/qcnet_topossm_decoder_h8_f12_s5_k6_4gpu_20260503/lightning_logs/version_0/checkpoints/epoch=23-step=3864.ckpt}"
TEACHER_CKPT="${TEACHER_CKPT:-${BASE_OUT}/qcnet_topossm_safetyft_h8_f12_s5_k6_4gpu_20260503/lightning_logs/version_0/checkpoints/epoch=7-step=1288.ckpt}"

mkdir -p "${LOG_DIR}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
ulimit -n 65535 2>/dev/null || true

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
out = {
    "latest_epoch": None,
    "latest_ade": None,
    "best_ade": None,
    "best_path": None,
    "num_ckpts": len(paths),
}
if paths:
    latest = paths[-1]
    ckpt = torch.load(latest, map_location="cpu")
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
data = json.loads(sys.argv[1])
value = data.get(sys.argv[2])
print("" if value is None else value)
PY
}

float_le() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
a = float(sys.argv[1])
b = float(sys.argv[2])
raise SystemExit(0 if a <= b else 1)
PY
}

kill_run() {
  local save_root="$1"
  local pids
  pids="$(pgrep -f "train_qcnet.py.*${save_root}" || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  log "Stopping run processes for ${save_root}: ${pids//$'\n'/ }"
  kill -TERM ${pids} 2>/dev/null || true
  sleep 12
  pids="$(pgrep -f "train_qcnet.py.*${save_root}" || true)"
  if [[ -n "${pids}" ]]; then
    log "Force stopping remaining processes for ${save_root}: ${pids//$'\n'/ }"
    kill -9 ${pids} 2>/dev/null || true
    sleep 5
  fi
}

run_variant() {
  local letter="$1"
  local name="$2"
  local save_root="$3"
  local run_log="$4"
  local init_ckpt="$5"
  local max_epochs="$6"
  local lr="$7"
  local seed="$8"
  local distill_propose="$9"
  local distill_refine="${10}"
  local distill_score="${11}"
  local distill_warmup="${12}"
  local switch_epoch="${13}"
  local switch_threshold="${14}"
  local early_bad_threshold="${15}"

  local ckpt_dir="${save_root}/lightning_logs/version_0/checkpoints"
  rm -rf "${save_root}/lightning_logs/version_0"
  mkdir -p "${save_root}"

  log "===== START ${letter} ${name} ====="
  log "save_root=${save_root}"
  log "init=${init_ckpt} lr=${lr} max_epochs=${max_epochs} distill=(${distill_propose},${distill_refine},${distill_score})"
  log "run_log=${run_log}"

  local args=(
    train_qcnet.py
    --dataset interaction_digir
    --interaction_data_path "${DATA_ROOT}/${DATA_FILE}"
    --save_root "${save_root}"
    --init_from_checkpoint "${init_ckpt}"
    --seed "${seed}"
    --batch_by_location
    --max_epochs "${max_epochs}"
    --train_batch_size 12
    --val_batch_size 12
    --test_batch_size 12
    --num_workers 4
    --pin_memory false
    --persistent_workers false
    --lr "${lr}"
    --weight_decay 1e-4
    --eval_batches 0
    --num_modes 6
    --eval_k 6
    --monitor_metric val_minADE
    --monitor_mode min
    --devices 4
    --accelerator gpu
    --num_historical_steps 8
    --num_future_steps 12
    --num_recurrent_steps 3
    --pl2pl_radius 80
    --time_span 8
    --pl2a_radius 50
    --a2a_radius 50
    --num_t2m_steps 8
    --pl2m_radius 80
    --a2m_radius 80
    --decoder_type topossm
    --topo_ssm_layers 2
    --topo_mamba_d_state 16
    --topo_mamba_d_conv 4
    --topo_mamba_expand 2
    --topo_corridor_loss_weight 0.02
    --topo_score_loss_weight 0.02
    --topo_score_temperature 0.20
  )

  if [[ "${distill_propose}" != "0" || "${distill_refine}" != "0" || "${distill_score}" != "0" ]]; then
    args+=(
      --distill_teacher_checkpoint "${TEACHER_CKPT}"
      --distill_propose_weight "${distill_propose}"
      --distill_refine_weight "${distill_refine}"
      --distill_score_weight "${distill_score}"
      --distill_temperature 1.5
      --distill_warmup_epochs "${distill_warmup}"
    )
  fi

  "${PYTHON_BIN}" "${args[@]}" > "${run_log}" 2>&1 &
  local pid="$!"
  log "${letter} PID ${pid}"

  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${MONITOR_INTERVAL_SEC}"
    local metrics
    metrics="$(metric_json "${ckpt_dir}")"
    local latest_epoch latest_ade best_ade best_path
    latest_epoch="$(json_value "${metrics}" latest_epoch)"
    latest_ade="$(json_value "${metrics}" latest_ade)"
    best_ade="$(json_value "${metrics}" best_ade)"
    best_path="$(json_value "${metrics}" best_path)"
    log "${letter} latest_epoch=${latest_epoch:-na} latest_ade=${latest_ade:-na} best_ade=${best_ade:-na} best_path=${best_path:-na}"

    if [[ -n "${best_ade}" ]] && float_le "${best_ade}" "${TARGET_ADE}"; then
      log "${letter} reached target best_ade=${best_ade} <= ${TARGET_ADE}; stopping supervisor."
      kill_run "${save_root}"
      return 0
    fi

    if [[ -n "${latest_epoch}" && -n "${best_ade}" ]]; then
      if (( latest_epoch >= switch_epoch )); then
        if ! float_le "${best_ade}" "${switch_threshold}"; then
          log "${letter} behind at epoch ${latest_epoch}: best_ade=${best_ade} > ${switch_threshold}; switching."
          kill_run "${save_root}"
          return 2
        fi
      fi
      if (( latest_epoch >= 2 )); then
        if [[ -n "${latest_ade}" ]] && ! float_le "${latest_ade}" "${early_bad_threshold}"; then
          log "${letter} latest_ade=${latest_ade} > early_bad_threshold=${early_bad_threshold}; switching."
          kill_run "${save_root}"
          return 2
        fi
      fi
    fi
  done

  wait "${pid}" || true
  local metrics
  metrics="$(metric_json "${ckpt_dir}")"
  local best_ade
  best_ade="$(json_value "${metrics}" best_ade)"
  log "${letter} finished. metrics=${metrics}"
  if [[ -n "${best_ade}" ]] && float_le "${best_ade}" "${TARGET_ADE}"; then
    log "${letter} reached target after finish."
    return 0
  fi
  return 2
}

log "TopoSSM decoder ABC supervisor started"
log "Repo=${REPO} BaseOut=${BASE_OUT} GPUs=${GPU_IDS} target=${TARGET_ADE}"

run_variant \
  A \
  topassm_dec_A_supervised_low_lr \
  "${BASE_OUT}/qcnet_topossm_decoder_abc_A_supervised_h8_f12_s5_k6_4gpu_20260503" \
  "${LOG_DIR}/qcnet_topossm_decoder_abc_A_supervised_h8_f12_s5_k6_4gpu_20260503.log" \
  "${HEAVY_E5_CKPT}" \
  16 \
  2e-5 \
  45 \
  0 \
  0 \
  0 \
  0 \
  5 \
  0.0585 \
  0.075 || status=$?
status="${status:-0}"
if [[ "${status}" == "0" ]]; then
  exit 0
fi
unset status

run_variant \
  B \
  topassm_dec_B_light_distill \
  "${BASE_OUT}/qcnet_topossm_decoder_abc_B_light_distill_h8_f12_s5_k6_4gpu_20260503" \
  "${LOG_DIR}/qcnet_topossm_decoder_abc_B_light_distill_h8_f12_s5_k6_4gpu_20260503.log" \
  "${HEAVY_E5_CKPT}" \
  16 \
  2e-5 \
  46 \
  0.01 \
  0.03 \
  0.03 \
  1 \
  4 \
  0.0585 \
  0.070 || status=$?
status="${status:-0}"
if [[ "${status}" == "0" ]]; then
  exit 0
fi
unset status

run_variant \
  C \
  topassm_dec_C_clean_light_distill \
  "${BASE_OUT}/qcnet_topossm_decoder_abc_C_clean_light_distill_h8_f12_s5_k6_4gpu_20260503" \
  "${LOG_DIR}/qcnet_topossm_decoder_abc_C_clean_light_distill_h8_f12_s5_k6_4gpu_20260503.log" \
  "${CLEAN_DECODER_CKPT}" \
  20 \
  5e-5 \
  47 \
  0.03 \
  0.06 \
  0.03 \
  1 \
  5 \
  0.060 \
  0.075 || status=$?
status="${status:-0}"
if [[ "${status}" == "0" ]]; then
  exit 0
fi

log "All ABC variants finished or switched without reaching target ${TARGET_ADE}. Check logs under ${LOG_DIR}."
exit 1
