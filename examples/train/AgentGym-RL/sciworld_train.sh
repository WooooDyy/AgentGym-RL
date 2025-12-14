set -x

exp_name="14b_n8"
pure_agent_model_name="Qwen2.5-14B-Instruct"


# export CUDA_LAUNCH_BLOCKING="0"
export VLLM_USE_MODELSCOPE="0"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_ATTENTION_BACKEND="XFORMERS"
# export VLLM_HOST_IP="localhost"
export RAY_DEDUP_LOGS=0
export VLLM_LOGGING_LEVEL=INFO
# export OMP_NUM_THREADS=8
export HOST_IP=$(hostname -I | awk '{print $1}')
export VLLM_HOST_IP="${HOST_IP}"
export MASTER_ADDR="${HOST_IP}"
# export VLLM_TRACE_FUNCTION=0
# export RAY_memory_monitor_refresh_ms=0
# export RAY_record_ref_creation_sites=1
export RAY_LOG_TO_DRIVER="true"
export VLLM_USE_V1=0  # Force V0 engine to avoid Ray placement group bug
# export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VERL_PPO_LOGGING_LEVEL="INFO"
export RAY_BACKEND_LOG_LEVEL="debug"
# export VLLM_GC_DEBUG=1

# export CUDA_VISIBLE_DEVICES="0,1"
# gpu_memory_utilization=0.7
export gpu_memory_utilization=0.5 # 0.5 works.
# gpu_memory_utilization=0.40 # 0.7
export NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr -cd , | wc -c); ((NUM_GPUS++))
# export train_batch_size="${NUM_GPUS}" # NOTE: needs to match num gpus # 4 works
export BATCH_SIZE_PER_GPU=2 # 1 works
export train_batch_size=$((${NUM_GPUS} * ${BATCH_SIZE_PER_GPU}))
export tensor_model_parallel_size=1 # NOTE: needs to match num gpus
# export tensor_model_parallel_size="${NUM_GPUS}" # NOTE: needs to match num gpus
export n_gpus_per_node="${NUM_GPUS}" # NOTE: needs to match num gpus
# pure_agent_model_name="Qwen2.5-1.5B-Instruct"
# pure_agent_model_name="Qwen2.5-0.5B-Instruct"

# export TMPDIR="${HOME}/tmp"
# export RAY_TMPDIR="${HOME}/tmp"
# export TMPDIR="/scratch/${USER}/tmp"
# export RAY_TMPDIR="/scratch/${USER}/tmp"
# export TRITON_HOME="/scratch/${USER}"

task_name="sciworld"

# source activate
# conda activate agentgym-rl
export VLLM_ATTENTION_BACKEND="XFORMERS"
# export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_BASE_URL="https://api.wandb.ai"

export env_server_url="http://localhost:${PORT}"

# start training
# wandb login xxx


agent_model_path="${HOME}/agentgym_rl/models/${pure_agent_model_name}"

kl_coef=0.001
policy_learning_rate=1e-6
# rollout_sample_num=8
rollout_sample_num=8 # 1 works
# train_batch_size=16
# ppo_mini_batch_size=8
# ppo_mini_batch_size_per_gpu=1
ppo_mini_batch_size="${NUM_GPUS}" # 4 works
ppo_micro_batch_size_per_gpu=1
ppo_inner_epochs=1
# XDG_DATA_DIRS
# export BATCH_SIZE=$((${NUM_GPUS} * ${BATCH_SIZE_PER_GPU}))

# kl_coef=0.001
# policy_learning_rate=1e-6
# rollout_sample_num=8
# train_batch_size=16
# ppo_mini_batch_size=8
# ppo_micro_batch_size_per_gpu=1
# ppo_inner_epochs=1

total_epoches=10

model_save_dir="AgentGym-RL/saves"
mkdir -p ${model_save_dir}
model_save_path=${model_save_dir}/${exp_name}

mkdir -p ${model_save_path}
# rm -rf ${RAY_TMPDIR}/ray
# rm -rf ${HOME}/.cache/{vllm,nvidia} ${HOME}/.triton

# HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online python3 -m verl.agent_trainer.main_ppo  \
# HYDRA_FULL_ERROR=1 WANDB_MODE=online python -m verl.agent_trainer.main_ppo  \
HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online ${PYTHON_BIN} -m verl.agent_trainer.main_ppo  \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=20 \
    data.train_file=AgentGym-RL/AgentItemId/${task_name}_train.json \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_tokens=200 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${tensor_model_parallel_size}" \
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs} \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.default_local_dir=${model_save_path} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.project_name=ag \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=25 \
    trainer.total_epochs=${total_epoches}

status=$?
exit $status
