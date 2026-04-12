#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_CODE_DIR="${ROOT}/AgentGym-RL"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"
WEBSHOP_ENV="${WEBSHOP_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-webshop}"

MODEL_PATH="${MODEL_PATH:-${ROOT}/models/Qwen2.5-3B-Instruct}"
TASK_NAME="webshop"

ENV_HOST="${ENV_HOST:-127.0.0.1}"
ENV_PORT="${ENV_PORT:-8011}"
ENV_ADDR="http://${ENV_HOST}:${ENV_PORT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_GPUS < 1 )); then
  echo "CUDA_VISIBLE_DEVICES is empty."
  exit 1
fi

WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-webshop}"

KL_COEF="${KL_COEF:-0.001}"
POLICY_LR="${POLICY_LR:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
MAX_ROUNDS="${MAX_ROUNDS:-15}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-256}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"
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

MODE_TAG="grpo"
if [[ "${ENABLE_WMC}" == "1" && "${ENABLE_ERC}" == "1" ]]; then
  MODE_TAG="grpo_wmc_erc"
elif [[ "${ENABLE_WMC}" == "1" ]]; then
  MODE_TAG="grpo_wmc"
elif [[ "${ENABLE_ERC}" == "1" ]]; then
  MODE_TAG="grpo_erc"
fi

BASE_MODEL_NAME="$(basename "${MODEL_PATH}")"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${TASK_NAME}_${MODE_TAG}_${BASE_MODEL_NAME}_${RUN_TS}}"

RUN_DIR="${RUN_DIR:-${ROOT}/runlogs/${EXP_NAME}}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/${EXP_NAME}}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${RUN_DIR}/rollout_logs}"
TRAIN_LOG="${RUN_DIR}/train.log"
ENV_LOG="${RUN_DIR}/webshop_env.log"
ENV_PID_FILE="${RUN_DIR}/webshop_env.pid"
TRAIN_PID_FILE="${RUN_DIR}/train.pid"
WARMUP_ID_FILE="${RUN_DIR}/webshop_warmup_env_id.txt"

TRAIN_FILE="${TRAIN_FILE:-${ROOT}/AgentItemId/train/webshop_train.json}"

mkdir -p "${RUN_DIR}" "${CKPT_DIR}" "${ROLLOUT_LOG_DIR}"

REAL_TRAIN_BATCH_SIZE=$(( TRAIN_BATCH_SIZE * ROLLOUT_N ))
if (( REAL_TRAIN_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "train_batch_size * rollout_n must be divisible by number of visible GPUs."
  echo "train_batch_size=${TRAIN_BATCH_SIZE}, rollout_n=${ROLLOUT_N}, num_gpus=${NUM_GPUS}"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model path does not look valid: ${MODEL_PATH}"
  exit 1
fi

source "${CONDA_SH}"

conda_activate_safe() {
  set +u
  conda activate "$1"
  set -u
}

conda_deactivate_safe() {
  set +u
  conda deactivate || true
  set -u
}

conda_activate_safe "${TRAIN_ENV}"
python "${ROOT}/scripts/prepare_webshop_grpo_splits.py"
conda_deactivate_safe

export WEBSHOP_DATASET_SIZE="all"
export WEBSHOP_GOAL_SOURCE="human"
export WEBSHOP_HUMAN_GOAL_MODE="official"
export WEBSHOP_GOAL_SPLIT="train"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

if lsof -iTCP:"${ENV_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port ${ENV_PORT} is already listening. Reusing existing WebShop service."
else
  conda_activate_safe "${WEBSHOP_ENV}"
  nohup env \
    -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    NO_PROXY="${NO_PROXY}" \
    no_proxy="${no_proxy}" \
    WEBSHOP_DATASET_SIZE="${WEBSHOP_DATASET_SIZE}" \
    WEBSHOP_GOAL_SOURCE="${WEBSHOP_GOAL_SOURCE}" \
    WEBSHOP_HUMAN_GOAL_MODE="${WEBSHOP_HUMAN_GOAL_MODE}" \
    WEBSHOP_GOAL_SPLIT="${WEBSHOP_GOAL_SPLIT}" \
    webshop --host "${ENV_HOST}" --port "${ENV_PORT}" \
    > "${ENV_LOG}" 2>&1 &
  ENV_PID=$!
  printf '%s\n' "${ENV_PID}" > "${ENV_PID_FILE}"
  conda_deactivate_safe
fi

for _ in $(seq 1 60); do
  if curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null; then
  echo "WebShop service did not become healthy: ${ENV_ADDR}"
  exit 1
fi

echo "Warming up full WebShop data..."
WARMUP_ID="$(curl --noproxy '*' --max-time 1800 -sS -X POST "${ENV_ADDR}/create")"
printf '%s\n' "${WARMUP_ID}" > "${WARMUP_ID_FILE}"
curl --noproxy '*' -sS -X POST "${ENV_ADDR}/close" \
  -H 'Content-Type: application/json' \
  -d "{\"env_idx\": ${WARMUP_ID}}" >/dev/null || true

WMC_COEFF_VALUE="0.0"
if [[ "${ENABLE_WMC}" == "1" ]]; then
  WMC_COEFF_VALUE="${WMC_COEFF}"
fi

ERC_ENABLE_VALUE="False"
if [[ "${ENABLE_ERC}" == "1" ]]; then
  ERC_ENABLE_VALUE="True"
fi

conda_activate_safe "${TRAIN_ENV}"
cd "${TRAIN_CODE_DIR}"

nohup env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  NO_PROXY="${NO_PROXY}" \
  no_proxy="${no_proxy}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  VLLM_USE_MODELSCOPE=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  HYDRA_FULL_ERROR=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_MODE="${WANDB_MODE}" \
  python -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds="${MAX_ROUNDS}" \
    data.train_file="${TRAIN_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.agentgym.task_name="${TASK_NAME}" \
    actor_rollout_ref.agentgym.env_addr="${ENV_ADDR}" \
    actor_rollout_ref.agentgym.timeout=2400 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.world_model_coeff="${WMC_COEFF_VALUE}" \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.actor.optim.lr="${POLICY_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_tokens="${MAX_TOKENS_PER_TURN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.rollout_log_dir="${ROLLOUT_LOG_DIR}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    wmc_erc.enable="${ERC_ENABLE_VALUE}" \
    wmc_erc.mu_base="${ERC_MU_BASE}" \
    wmc_erc.mu_exp="${ERC_MU_EXP}" \
    wmc_erc.eta_wm="${ERC_ETA_WM}" \
    wmc_erc.lambda_wm="${ERC_LAMBDA_WM}" \
    wmc_erc.clipping_type="${ERC_CLIPPING_TYPE}" \
    wmc_erc.clipping_method="${ERC_CLIPPING_METHOD}" \
    wmc_erc.momentum="${ERC_MOMENTUM}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" > "${TRAIN_PID_FILE}"

echo "WebShop GRPO has started in the background."
echo "Experiment: ${EXP_NAME}"
echo "WebShop service: ${ENV_ADDR}"
echo "Train file: ${TRAIN_FILE}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Run log dir: ${RUN_DIR}"
echo "Env log: ${ENV_LOG}"
echo "Train log: ${TRAIN_LOG}"
if [[ -f "${ENV_PID_FILE}" ]]; then
  echo "Env PID: $(cat "${ENV_PID_FILE}")"
fi
echo "Train PID: $(cat "${TRAIN_PID_FILE}")"
echo "Warmup env id: $(cat "${WARMUP_ID_FILE}")"
echo "Follow training log with:"
echo "tail -f ${TRAIN_LOG}"
