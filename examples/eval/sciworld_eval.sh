set -x
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

task_name="sciworld"

cd AgentGym-RL
source activate
conda activate agentgym-rl
export VLLM_ATTENTION_BACKEND=XFORMERS

env_server_url="http://localhost:36001"

sample_num=1
max_rounds=30

# ckpt_path="global_step_150/actor"
ckpt_path="${HOME}/agentgym_rl/AgentGym-RL/saves/3b_n8/global_step_125/actor"
model_path=${ckpt_path}/huggingface

cd AgentGym-RL/scripts
python model_merger.py --local_dir ${ckpt_path}

HYDRA_FULL_ERROR=1 python -m verl.agent_trainer.main_generation  \
    data.path=AgentEval/${task_name} \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.n_samples=${sample_num} \
    data.batch_size=32 \
    agentgym.task_name=${task_name} \
    agentgym.env_addr=${env_server_url} \
    agentgym.max_rounds=${max_rounds} \
    agentgym.timeout=500 \
    model.path=${model_path} \
    rollout.gpu_memory_utilization=0.95 \
    rollout.temperature=1 \
    rollout.max_model_len=32768 \
    rollout.max_tokens=200 \
    rollout.tensor_model_parallel_size=1 \
    rollout.rollout_log_dir=executer_logs
status=$?
exit $status

# bash examples/eval/AgentGym-RL/sciworld_eval.sh |& tee log_eval_512gb_80gb_3b_n8.log
