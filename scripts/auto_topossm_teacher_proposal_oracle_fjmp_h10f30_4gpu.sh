#!/usr/bin/env bash
set -euo pipefail

# Diagnostic only: full-replacement TopoSSM with externally supplied teacher proposals.
# No distillation losses are enabled; the teacher only supplies anchor trajectories.

REPO="${REPO:-/home/bitwxy/qcnet}"
PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
DATA_FILE="${DATA_FILE:-interaction_fjmp_digir_h10_f30.pkl}"
BASE_OUT="${BASE_OUT:-/data/sdb/bitwxy/qcnet_data}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
SEED="${SEED:-23}"
RUN_TAG="${RUN_TAG:-20260508_topossm_teacher_proposal_oracle_fjmp_h10f30_fastval64_v1}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-/data/sdb/bitwxy/qcnet_data/qcnet_topossm_refiner_fjmp_h10_f30_k6_4gpu_seed23_20260508_qcnet_topossm_refiner_fjmp_h10f30_fastval64_v1/lightning_logs/version_0/checkpoints/epoch=2-step=1119.ckpt}"
TEACHER_PROPOSAL_SOURCE="${TEACHER_PROPOSAL_SOURCE:-refine}"

MAX_EPOCHS="${MAX_EPOCHS:-4}"
TRAIN_BS="${TRAIN_BS:-32}"
EVAL_BS="${EVAL_BS:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PIN_MEMORY="${PIN_MEMORY:-false}"
EVAL_BATCHES="${EVAL_BATCHES:-64}"
TOPO_PROPOSAL_TYPE="${TOPO_PROPOSAL_TYPE:-goal_mlp}"
LR="${LR:-8e-5}"
MAMBA_LR="${MAMBA_LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAMBA_WEIGHT_DECAY="${MAMBA_WEIGHT_DECAY:-0.0}"
CORRIDOR_LOSS="${CORRIDOR_LOSS:-0.00}"
SCORE_LOSS="${SCORE_LOSS:-0.0}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-60}"

RUN_NAME="topossm_teacher_proposal_oracle_fjmp_h10_f30_k6_4gpu_seed${SEED}_${RUN_TAG}"
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

write_summary() {
  local status="$1"
  local metrics="$2"
  local latest_epoch latest_ade best_ade best_path
  latest_epoch="$(json_value "${metrics}" latest_epoch)"
  latest_ade="$(json_value "${metrics}" latest_ade)"
  best_ade="$(json_value "${metrics}" best_ade)"
  best_path="$(json_value "${metrics}" best_path)"
  echo "status,seed,proposal_type,teacher_source,teacher_checkpoint,lr,mamba_lr,max_epochs,train_bs,eval_bs,eval_batches,latest_epoch,latest_ade,best_ade,best_path" > "${SUMMARY_CSV}"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${status}" "${SEED}" "${TOPO_PROPOSAL_TYPE}" "${TEACHER_PROPOSAL_SOURCE}" "${TEACHER_CHECKPOINT}" \
    "${LR}" "${MAMBA_LR}" "${MAX_EPOCHS}" "${TRAIN_BS}" "${EVAL_BS}" "${EVAL_BATCHES}" \
    "${latest_epoch:-}" "${latest_ade:-}" "${best_ade:-}" "${best_path:-}" >> "${SUMMARY_CSV}"
}

if ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep | \
   grep -q "/data/sdb/bitwxy/qcnet_data"; then
  log "Another bitwxy QCNet training process is active; refusing to launch."
  ps -u bitwxy -o pid,ppid,stat,etime,cmd | grep -E "train_qcnet.py|torchrun" | grep -v grep || true
  exit 3
fi

if [[ ! -f "${DATA_ROOT}/${DATA_FILE}" ]]; then
  log "Missing data file: ${DATA_ROOT}/${DATA_FILE}"
  exit 4
fi
if [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
  log "Missing teacher checkpoint: ${TEACHER_CHECKPOINT}"
  exit 5
fi
if [[ -e "${SAVE_ROOT}/lightning_logs/version_0" ]]; then
  log "Existing run detected at ${SAVE_ROOT}; refusing to overwrite."
  metrics="$(metric_json "${SAVE_ROOT}/lightning_logs/version_0/checkpoints")"
  write_summary "existing" "${metrics}"
  exit 0
fi

log "TopoSSM teacher-proposal oracle diagnostic started"
log "repo=${REPO}"
log "data=${DATA_ROOT}/${DATA_FILE}"
log "teacher=${TEACHER_CHECKPOINT}"
log "teacher_source=${TEACHER_PROPOSAL_SOURCE}"
log "save_root=${SAVE_ROOT}"
log "run_log=${RUN_LOG}"
log "summary_csv=${SUMMARY_CSV}"

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
  --topo_proposal_type "${TOPO_PROPOSAL_TYPE}" \
  --topo_ssm_layers 2 \
  --topo_mamba_d_state 16 \
  --topo_mamba_d_conv 4 \
  --topo_mamba_expand 2 \
  --topo_corridor_loss_weight "${CORRIDOR_LOSS}" \
  --topo_score_loss_weight "${SCORE_LOSS}" \
  --topo_score_temperature 0.20 \
  --distill_teacher_checkpoint "${TEACHER_CHECKPOINT}" \
  --use_teacher_proposals \
  --teacher_proposal_source "${TEACHER_PROPOSAL_SOURCE}" \
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
done
wait "${pid}" || status="failed"

final_metrics="$(metric_json "${ckpt_dir}")"
write_summary "${status}" "${final_metrics}"
log "Finished status=${status} metrics=${final_metrics}"
