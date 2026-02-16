# Scripts Dockerfile for AgentGym-RL
# Provides environment for utility scripts: model merging, formatting, etc.
#
# Usage:
#   docker build -f docker/scripts.Dockerfile -t agentgym-rl/scripts:latest .
#
# Run model merger:
#   docker run --gpus all -v $(pwd):/workspace agentgym-rl/scripts:latest \
#     python scripts/model_merger.py --local_dir /workspace/checkpoints/step_100
#
# Run formatter:
#   docker run -v $(pwd):/workspace agentgym-rl/scripts:latest \
#     bash scripts/format.sh

FROM agentgym-rl/base:latest

# Install dependencies for scripts
RUN pip install \
    transformers \
    huggingface_hub \
    yapf \
    safetensors \
    accelerate

# Copy scripts
COPY AgentGym-RL/scripts /workspace/scripts
RUN chmod +x /workspace/scripts/*.sh

# Copy verl for model loading utilities
COPY AgentGym-RL /workspace/AgentGym-RL
RUN pip install -e /workspace/AgentGym-RL

WORKDIR /workspace

# Entrypoint allows flexible command execution
ENTRYPOINT []
CMD ["/bin/bash"]
