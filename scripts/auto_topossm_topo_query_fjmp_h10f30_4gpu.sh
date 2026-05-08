#!/usr/bin/env bash
set -euo pipefail

# Full-replacement TopoSSM with topology-query proposal on FJMP h10/f30.
# Clean diagnostic run: no teacher proposal, no distillation, no warm-start.

REPO="${REPO:-/home/bitwxy/qcnet}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
DATA_FILE="${DATA_FILE:-interaction_fjmp_digir_h10_f30.pkl}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
SEED="${SEED:-23}"
RUN_TAG="${RUN_TAG:-20260508_topossm_topo_query_fjmp_h10f30_fastval64_v1}"

MAX_EPOCHS="${MAX_EPOCHS:-4}"
TRAIN_BS="${TRAIN_BS:-32}"
EVAL_BS="${EVAL_BS:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCHES="${EVAL_BATCHES:-64}"
PIN_MEMORY="${PIN_MEMORY:-false}"
LR="${LR:-8e-5}"
MAMBA_LR="${MAMBA_LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAMBA_WEIGHT_DECAY="${MAMBA_WEIGHT_DECAY:-0.0}"
MODE_ENDPOINT_SCALE="${MODE_ENDPOINT_SCALE:-4.0}"
CORRIDOR_LOSS="${CORRIDOR_LOSS:-0.00}"
SCORE_LOSS="${SCORE_LOSS:-0.0}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-60}"

RUN_NAME="topossm_topo_query_fjmp_h10_f30_k6_4gpu_seed${SEED}_${RUN_TAG}"
SAVE_ROOT="${SAVE_ROOT:-${BASE_OUT}/${RUN_NAME}}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
CONSOLE_LOG="${CONSOLE_LOG:-${LOG_DIR}/auto_${RUN_NAME}_console.log}"
SUMMARY_CSV="${SUMMARY_CSV:-${LOG_DIR}/${RUN_NAME}_summary.csv}"

mkdir -p "${LOG_DIR}" "${SAVE_ROOT}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
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

metric_json() {
  local ckpt_dir="$1"
  "${PYTHON_BIN}" - "${ckpt_dir}" <<'PY'
import glob
import json
import os
import sys
import torch

paths = sorted(glob.glob(os.path.join(sys.argv[1], "*.ckpt")), key=os.path.getmtime)
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

write_summary() {
  local status="$1"
  local metrics="$2"
  local latest_epoch latest_ade best_ade best_path
  latest_epoch="$(json_value "${metrics}" latest_epoch)"
  latest_ade="$(json_value "${metrics}" latest_ade)"
  best_ade="$(json_value "${metrics}" best_ade)"
  best_path="$(json_value "${metrics}" best_path)"
  echo "status,seed,proposal_type,lr,mamba_lr,weight_decay,scale,corridor_loss,score_loss,max_epochs,train_bs,eval_bs,eval_batches,latest_epoch,latest_ade,best_ade,best_path" > "${SUMMARY_CSV}"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${status}" "${SEED}" "topo_query" "${LR}" "${MAMBA_LR}" "${WEIGHT_DECAY}" \
    "${MODE_ENDPOINT_SCALE}" "${CORRIDOR_LOSS}" "${SCORE_LOSS}" "${MAX_EPOCHS}" \
    "${TRAIN_BS}" "${EVAL_BS}" "${EVAL_BATCHES}" \
    "${latest_epoch:-}" "${latest_ade:-}" "${best_ade:-}" "${best_path:-}" >> "${SUMMARY_CSV}"
}

if ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep | \
   grep -q "/data/sdb/bitwxy/qcnet_data"; then
  log "Another bitwxy QCNet training process is active; refusing to launch."
  ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep || true
  exit 3
fi

if [[ -e "${SAVE_ROOT}/lightning_logs/version_0" ]]; then
  log "Existing run detected at ${SAVE_ROOT}; refusing to overwrite."
  metrics="$(metric_json "${SAVE_ROOT}/lightning_logs/version_0/checkpoints")"
  write_summary "existing" "${metrics}"
  exit 0
fi

log "TopoSSM topo-query FJMP h10/f30 run started"
log "repo=${REPO}"
log "data=${DATA_ROOT}/${DATA_FILE}"
log "save_root=${SAVE_ROOT}"
log "run_log=${RUN_LOG}"
log "summary_csv=${SUMMARY_CSV}"
log "config=seed=${SEED} proposal=topo_query bs=${TRAIN_BS}/${EVAL_BS} eval_batches=${EVAL_BATCHES} lr=${LR} mamba_lr=${MAMBA_LR} scale=${MODE_ENDPOINT_SCALE}"

setsid "${PYTHON_BIN}" train_qcnet.py \
  --dataset interaction_digir \
  --interaction_data_path "${DATA_ROOT}/${DATA_FILE}" \
  --save_root "${SAVE_ROOT}" \
  --seed "${SEED}" \
  --batch_by_location \
  --max_epochs "${MAX_EPOCHS}" \
  --train_batch_size "${TRAIN_BS}" \
  --val_batch_size "${EVAL_BS}" \
  --test_batch_size "${EVAL_BS}" \
  --num_workers "${NUM_WORKERS}" \
  --pin_memory "${PIN_MEMORY}" \
  --persistent_workers true \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --mamba_lr "${MAMBA_LR}" \
  --mamba_weight_decay "${MAMBA_WEIGHT_DECAY}" \
  --eval_batches "${EVAL_BATCHES}" \
  --num_modes 6 \
  --eval_k 6 \
  --monitor_metric val_minADE \
  --monitor_mode min \
  --devices 4 \
  --accelerator gpu \
  --num_historical_steps 10 \
  --num_future_steps 30 \
  --num_recurrent_steps 3 \
  --pl2pl_radius 80 \
  --time_span 10 \
  --pl2a_radius 50 \
  --a2a_radius 50 \
  --num_t2m_steps 10 \
  --pl2m_radius 80 \
  --a2m_radius 80 \
  --decoder_type topossm \
  --topo_proposal_type topo_query \
  --topo_ssm_layers 2 \
  --topo_mamba_d_state 16 \
  --topo_mamba_d_conv 4 \
  --topo_mamba_expand 2 \
  --topo_mode_endpoint_scale "${MODE_ENDPOINT_SCALE}" \
  --topo_corridor_loss_weight "${CORRIDOR_LOSS}" \
  --topo_score_loss_weight "${SCORE_LOSS}" \
  --topo_score_temperature 0.20 \
  > "${RUN_LOG}" 2>&1 &

pid="$!"
status="finished"
ckpt_dir="${SAVE_ROOT}/lightning_logs/version_0/checkpoints"
log "train PID ${pid}"

while kill -0 "${pid}" 2>/dev/null; do
  sleep "${MONITOR_INTERVAL_SEC}"
  metrics="$(metric_json "${ckpt_dir}")" || metrics="{}"
  latest_epoch="$(json_value "${metrics}" latest_epoch)"
  latest_ade="$(json_value "${metrics}" latest_ade)"
  best_ade="$(json_value "${metrics}" best_ade)"
  best_path="$(json_value "${metrics}" best_path)"
  log "latest_epoch=${latest_epoch:-na} latest_ade=${latest_ade:-na} best_ade=${best_ade:-na} best_path=${best_path:-na}"

  if [[ -n "${latest_epoch}" && -n "${best_ade}" ]]; then
    if [[ "${latest_epoch}" -ge 0 ]]; then
      if "${PYTHON_BIN}" - "${best_ade}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 5.2 else 1)
PY
      then
        status="stopped_epoch0_bad"
        kill_run_tree "${pid}" "best_ade=${best_ade} is still above 5.2 after epoch ${latest_epoch}"
        break
      fi
    fi
  fi
done
wait "${pid}" || {
  if [[ "${status}" == "finished" ]]; then
    status="failed"
  fi
}

final_metrics="$(metric_json "${ckpt_dir}")"
write_summary "${status}" "${final_metrics}"
log "Finished status=${status} metrics=${final_metrics}"
