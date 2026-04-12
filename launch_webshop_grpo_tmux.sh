#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PORT="${ENV_PORT:-8013}"
ENV_ADDR="http://127.0.0.1:${ENV_PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/models/Qwen2.5-3B-Instruct}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ANONYMOUS="${WANDB_ANONYMOUS:-never}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-webshop}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
SAVE_FREQ="${SAVE_FREQ:-200}"
ENABLE_WMC="${ENABLE_WMC:-0}"
WMC_COEFF="${WMC_COEFF:-1e-4}"
ENABLE_ERC="${ENABLE_ERC:-0}"
ERC_MU_BASE="${ERC_MU_BASE:-1.0}"
ERC_MU_EXP="${ERC_MU_EXP:-2.0}"
ERC_ETA_WM="${ERC_ETA_WM:-3.0}"
ERC_LAMBDA_WM="${ERC_LAMBDA_WM:-1.0}"
ERC_CLIPPING_TYPE="${ERC_CLIPPING_TYPE:-global}"
ERC_CLIPPING_METHOD="${ERC_CLIPPING_METHOD:-mask}"
ERC_MOMENTUM="${ERC_MOMENTUM:-0.9}"
REMOVE_PREVIOUS_CKPT_IN_SAVE="${REMOVE_PREVIOUS_CKPT_IN_SAVE:-0}"
MAX_LOCAL_CKPT_TO_KEEP="${MAX_LOCAL_CKPT_TO_KEEP:-10}"
BASE_MODEL_NAME="$(basename "${MODEL_PATH:-${ROOT}/models/Qwen2.5-3B-Instruct}")"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-webshop_grpo_${BASE_MODEL_NAME}_${RUN_TS}}"

ENV_SESSION="${ENV_SESSION:-webshop_env_${ENV_PORT}_${RUN_TS}}"
TRAIN_SESSION="${TRAIN_SESSION:-webshop_grpo_${RUN_TS}}"
ENV_LOG="${ROOT}/runlogs/${ENV_SESSION}.log"
TRAIN_LOG="${ROOT}/runlogs/${EXP_NAME}/train.log"

mkdir -p "${ROOT}/runlogs/${EXP_NAME}"

python3 "${ROOT}/scripts/prepare_webshop_grpo_splits.py"

if tmux has-session -t "${ENV_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${ENV_SESSION}"
  exit 1
fi
if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${TRAIN_SESSION}"
  exit 1
fi

tmux new-session -d -s "${ENV_SESSION}" \
  "cd ${ROOT} && HOST=127.0.0.1 PORT=${ENV_PORT} LOG_PATH=${ENV_LOG} bash ${ROOT}/scripts/run_webshop_env_service.sh"

for _ in $(seq 1 120); do
  if curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null; then
  echo "WebShop service did not become healthy on ${ENV_ADDR}"
  exit 1
fi

WARMUP_ID="$(curl --noproxy '*' --max-time 1800 -sS -X POST "${ENV_ADDR}/create")"
curl --noproxy '*' -sS -X POST "${ENV_ADDR}/close" \
  -H 'Content-Type: application/json' \
  -d "{\"env_idx\": ${WARMUP_ID}}" >/dev/null || true

tmux new-session -d -s "${TRAIN_SESSION}" \
  "cd ${ROOT} && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ENV_ADDR=${ENV_ADDR} MODEL_PATH=${MODEL_PATH} WANDB_MODE=${WANDB_MODE} WANDB_ANONYMOUS=${WANDB_ANONYMOUS} PROJECT_NAME=${PROJECT_NAME} ROLLOUT_N=${ROLLOUT_N} TOTAL_EPOCHS=${TOTAL_EPOCHS} SAVE_FREQ=${SAVE_FREQ} ENABLE_WMC=${ENABLE_WMC} WMC_COEFF=${WMC_COEFF} ENABLE_ERC=${ENABLE_ERC} ERC_MU_BASE=${ERC_MU_BASE} ERC_MU_EXP=${ERC_MU_EXP} ERC_ETA_WM=${ERC_ETA_WM} ERC_LAMBDA_WM=${ERC_LAMBDA_WM} ERC_CLIPPING_TYPE=${ERC_CLIPPING_TYPE} ERC_CLIPPING_METHOD=${ERC_CLIPPING_METHOD} ERC_MOMENTUM=${ERC_MOMENTUM} REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE} MAX_LOCAL_CKPT_TO_KEEP=${MAX_LOCAL_CKPT_TO_KEEP} EXP_NAME=${EXP_NAME} LOG_PATH=${TRAIN_LOG} bash ${ROOT}/scripts/run_webshop_grpo_train.sh"

echo "Environment tmux session: ${ENV_SESSION}"
echo "Training tmux session: ${TRAIN_SESSION}"
echo "WebShop service: ${ENV_ADDR}"
echo "Environment log: ${ENV_LOG}"
echo "Training log: ${TRAIN_LOG}"
echo "Warmup env id: ${WARMUP_ID}"
