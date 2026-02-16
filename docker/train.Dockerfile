# Training Dockerfile for AgentGym-RL
# Provides complete environment for RL training with verl and agentenv
#
# Usage:
#   docker build -f docker/train.Dockerfile -t agentgym-rl/train:latest .
#
# Run training:
#   docker run --gpus all -v $(pwd):/workspace agentgym-rl/train:latest \
#     python -m verl.agent_trainer.main_ppo ...

FROM agentgym-rl/base:latest

# Install verl (AgentGym-RL training framework)
COPY AgentGym-RL/requirements.txt /tmp/verl-requirements.txt
RUN pip install -r /tmp/verl-requirements.txt

COPY AgentGym-RL /workspace/AgentGym-RL
RUN pip install -e /workspace/AgentGym-RL

# Install agentenv (core environment client)
COPY AgentGym/agentenv /workspace/agentenv
RUN pip install -e /workspace/agentenv

# Install latest transformers for compatibility
RUN pip install transformers==4.51.3

# Environment variables for training
ENV VLLM_USE_MODELSCOPE=0
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV VLLM_ATTENTION_BACKEND=XFORMERS
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create directories for models and outputs
RUN mkdir -p /workspace/models /workspace/saves /workspace/data

WORKDIR /workspace

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import verl; print('OK')" || exit 1

CMD ["/bin/bash"]
