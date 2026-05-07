#!/usr/bin/env bash
set -euo pipefail

# Independent eval + rerun supervisor for the clean no-distill PaperA P07 result.
# This validates the checkpoint once with eval_qcnet_val_metrics.py, then replays
# the exact P07 training config from the same parent checkpoint multiple times.

REPO="${REPO:-/home/bitwxy/qcnet}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
DATA_FILE="${DATA_FILE:-interaction_digir_all_12loc_h8_f12_s5.pkl}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_GPU_ID="${EVAL_GPU_ID:-4}"
SEED="${SEED:-23}"
RUN_TAG="${RUN_TAG:-20260508_paperA_p07_eval_repro_v1}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-90}"
STALL_EPOCH_SEC="${STALL_EPOCH_SEC:-1200}"
STALL_LOG_SEC="${STALL_LOG_SEC:-1200}"
RERUNS="${RERUNS:-R01 R02 R03}"

PARENT_CKPT="${PARENT_CKPT:-${BASE_OUT}/paperA_seed23_lowlr_cont_h8_f12_s5_k6_4gpu_20260507_paperA_seed23_lowlr_cont/lightning_logs/version_0/checkpoints/epoch=7-step=1288.ckpt}"
P07_CKPT="${P07_CKPT:-${BASE_OUT}/paperA_param_sweep_h8_f12_s5_k6_4gpu_20260507_paperA_param_sweep_from0531_v1/P07_parent_lr1p5e-5_mlr5e-6_wd1e-4_s0p16_c0p00_fz0/lightning_logs/version_0/checkpoints/epoch=2-step=483.ckpt}"

OUT_ROOT="${OUT_ROOT:-${BASE_OUT}/paperA_p07_eval_repro_h8_f12_s5_k6_4gpu_${RUN_TAG}}"
SUMMARY_CSV="${SUMMARY_CSV:-${LOG_DIR}/paperA_p07_eval_repro_h8_f12_s5_k6_4gpu_${RUN_TAG}.csv}"
EVAL_OUT="${OUT_ROOT}/independent_eval"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}" "${EVAL_OUT}"
cd "${REPO}"
ulimit -n 65535 2>/dev/null || true

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
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

float_gt() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
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

kill_run_tree() {
  local root_pid="$1"
  local reason="$2"
  log "Stopping run tree for PID ${root_pid}: ${reason}"
  kill -TERM -- "-${root_pid}" 2>/dev/null || true
  sleep 5
  kill -KILL -- "-${root_pid}" 2>/dev/null || true
}

append_summary() {
  local candidate="$1" lr="$2" mamba_lr="$3" weight_decay="$4" scale="$5" corridor_loss="$6" metrics="$7" status="$8"
  local latest_epoch latest_ade best_ade best_path
  latest_epoch="$(json_value "${metrics}" latest_epoch)"
  latest_ade="$(json_value "${metrics}" latest_ade)"
  best_ade="$(json_value "${metrics}" best_ade)"
  best_path="$(json_value "${metrics}" best_path)"
  if [[ ! -f "${SUMMARY_CSV}" ]]; then
    echo "candidate,seed,lr,mamba_lr,weight_decay,scale,corridor_loss,status,latest_epoch,latest_ade,best_ade,best_path" > "${SUMMARY_CSV}"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${candidate}" "${SEED}" "${lr}" "${mamba_lr}" "${weight_decay}" "${scale}" "${corridor_loss}" \
    "${status}" "${latest_epoch:-}" "${latest_ade:-}" "${best_ade:-}" "${best_path:-}" >> "${SUMMARY_CSV}"
}

ensure_idle() {
  if ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun|eval_qcnet_val_metrics.py" | grep -v grep | \
     grep -q "/data/sdb/bitwxy/qcnet_data"; then
    log "Another QCNet train/eval process is active; refusing to launch."
    ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun|eval_qcnet_val_metrics.py" | grep -v grep || true
    exit 3
  fi
}

run_eval() {
  local eval_log="${LOG_DIR}/paperA_p07_independent_eval_${RUN_TAG}.log"
  log "Running independent eval for P07"
  export CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}"
  "${PYTHON_BIN}" scripts/eval_qcnet_val_metrics.py \
    --model "paperA_p07=${P07_CKPT}" \
    --out_dir "${EVAL_OUT}" \
    --interaction_data_path "${DATA_ROOT}/${DATA_FILE}" \
    --split val \
    --batch_size 16 \
    --num_workers 4 \
    --pin_memory \
    --persistent_workers \
    --accelerator gpu \
    --devices 1 \
    > "${eval_log}" 2>&1
  log "Independent eval finished: ${EVAL_OUT}"
  log "Eval log: ${eval_log}"
}

run_rerun() {
  local candidate="$1"
  local lr="1.5e-5" mamba_lr="5e-6" weight_decay="1e-4" scale="0.16" corridor_loss="0.00"
  local run_name="${candidate}_parent_lr1p5e-5_mlr5e-6_wd1e-4_s0p16_c0p00"
  local save_root="${OUT_ROOT}/${run_name}"
  local run_log="${LOG_DIR}/paperA_p07_repro_${RUN_TAG}_${run_name}.log"
  local ckpt_dir="${save_root}/lightning_logs/version_0/checkpoints"

  if [[ -e "${save_root}/lightning_logs/version_0" ]]; then
    log "Skipping ${candidate}: existing run ${save_root}"
    local metrics
    metrics="$(metric_json "${ckpt_dir}")"
    append_summary "${candidate}" "${lr}" "${mamba_lr}" "${weight_decay}" "${scale}" "${corridor_loss}" "${metrics}" "existing"
    return 0
  fi

  log "Starting ${candidate}: exact P07 replay from parent"
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  setsid "${PYTHON_BIN}" train_qcnet.py \
    --dataset interaction_digir \
    --interaction_data_path "${DATA_ROOT}/${DATA_FILE}" \
    --save_root "${save_root}" \
    --init_from_checkpoint "${PARENT_CKPT}" \
    --seed "${SEED}" \
    --batch_by_location \
    --max_epochs 4 \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --test_batch_size 16 \
    --num_workers 4 \
    --pin_memory true \
    --persistent_workers true \
    --lr "${lr}" \
    --weight_decay "${weight_decay}" \
    --mamba_lr "${mamba_lr}" \
    --mamba_weight_decay 0.0 \
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
    --topo_proposal_type mode_endpoint \
    --topo_ssm_layers 2 \
    --topo_mamba_d_state 16 \
    --topo_mamba_d_conv 4 \
    --topo_mamba_expand 2 \
    --topo_mode_endpoint_scale "${scale}" \
    --topo_corridor_loss_weight "${corridor_loss}" \
    --topo_score_loss_weight 0.0 \
    --topo_score_temperature 0.20 \
    > "${run_log}" 2>&1 &
  local pid="$!"
  local last_epoch_seen="" last_epoch_change_ts now_ts log_mtime log_silence_sec epoch_stall_sec status="finished"
  last_epoch_change_ts="$(date +%s)"

  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${MONITOR_INTERVAL_SEC}"
    now_ts="$(date +%s)"
    local metrics latest_epoch best_ade best_path latest_ade
    metrics="$(metric_json "${ckpt_dir}")" || metrics="{}"
    latest_epoch="$(json_value "${metrics}" latest_epoch)"
    latest_ade="$(json_value "${metrics}" latest_ade)"
    best_ade="$(json_value "${metrics}" best_ade)"
    best_path="$(json_value "${metrics}" best_path)"
    log "${candidate}: latest_epoch=${latest_epoch:-na} latest_ade=${latest_ade:-na} best_ade=${best_ade:-na} best_path=${best_path:-na}"
    if [[ -n "${latest_epoch}" && "${latest_epoch}" != "${last_epoch_seen}" ]]; then
      last_epoch_seen="${latest_epoch}"
      last_epoch_change_ts="${now_ts}"
    fi
    log_mtime=0
    [[ -e "${run_log}" ]] && log_mtime="$(stat -c %Y "${run_log}")"
    log_silence_sec="$(( now_ts - log_mtime ))"
    epoch_stall_sec="$(( now_ts - last_epoch_change_ts ))"
    if (( log_silence_sec >= STALL_LOG_SEC )); then
      status="stopped_log_silent"
      kill_run_tree "${pid}" "${candidate} run log silent for ${log_silence_sec}s"
      break
    fi
    if [[ -n "${last_epoch_seen}" ]] && (( epoch_stall_sec >= STALL_EPOCH_SEC )); then
      status="stopped_epoch_stall"
      kill_run_tree "${pid}" "${candidate} latest_epoch=${last_epoch_seen} stalled for ${epoch_stall_sec}s"
      break
    fi
    if [[ -n "${latest_epoch}" && -n "${best_ade}" ]]; then
      if (( latest_epoch >= 3 )) && float_gt "${best_ade}" "0.0585"; then
        status="early_bad_e3"
        kill_run_tree "${pid}" "${candidate} best_ade=${best_ade} > 0.0585 at epoch ${latest_epoch}"
        break
      fi
    fi
  done
  wait "${pid}" || true

  local final_metrics
  final_metrics="$(metric_json "${ckpt_dir}")"
  append_summary "${candidate}" "${lr}" "${mamba_lr}" "${weight_decay}" "${scale}" "${corridor_loss}" "${final_metrics}" "${status}"
  log "${candidate}: finished status=${status} metrics=${final_metrics}"
}

ensure_idle
if [[ ! -f "${PARENT_CKPT}" || ! -f "${P07_CKPT}" ]]; then
  log "Missing checkpoint. PARENT_CKPT=${PARENT_CKPT} P07_CKPT=${P07_CKPT}"
  exit 4
fi

log "PaperA P07 eval/repro supervisor started"
log "parent_ckpt=${PARENT_CKPT}"
log "p07_ckpt=${P07_CKPT}"
log "out_root=${OUT_ROOT}"
log "summary_csv=${SUMMARY_CSV}"

run_eval
for candidate in ${RERUNS}; do
  run_rerun "${candidate}"
done

log "PaperA P07 eval/repro supervisor finished. Summary:"
cat "${SUMMARY_CSV}" 2>/dev/null || true
