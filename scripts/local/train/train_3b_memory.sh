#!/bin/bash
# Local Training Script for ScienceWorld with Memory
# Usage: bash scripts/local/train_3b_memory.sh

set -ex  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
export MODEL_SIZE="3b"
export USE_MEMORY=true  # Set to false to disable memory

echo "=========================================="
echo "ScienceWorld Training (Local)"
echo "Model Size: ${MODEL_SIZE}"
echo "Memory Enabled: ${USE_MEMORY}"
echo "Start time: $(date)"
echo "=========================================="

# ============================================================================
# Setup Paths
# ============================================================================
export DIRPATH_PROJECT="${HOME}/agentgym_rl"
# cd "${DIRPATH_PROJECT}"


# source "${DIRPATH_PROJECT}/scripts/local/train/train_shared.sh"
source "${DIRPATH_PROJECT}/scripts/local/train/train_shared.sh"

# ============================================================================
# Conda Environment
# ============================================================================
# echo "Activating conda environment..."
# source ~/miniforge3/etc/profile.d/conda.sh  # Adjust if your conda is elsewhere
# conda activate agog  # Or your environment name



echo "Python: $(which python)"
echo "Python version: $(python --version)"



# Wait for server to be ready
sleep 10
echo "Server ready!"

# ============================================================================
# Cleanup function
# ============================================================================
cleanup() {
    echo ""
    echo "Cleaning up..."
    echo "Killing server (PID: ${SERVER_PID})..."
    kill ${SERVER_PID} 2>/dev/null || true
    echo "Done!"
}
trap cleanup EXIT INT TERM

# ============================================================================
# Run Training
# ============================================================================
echo ""
echo "Starting training..."
echo "=========================================="

# Override environment variables for sciworld_train.sh
export env_server_url="http://localhost:${PORT}"

# Run training script with memory configuration
cd AgentGym-RL

if [ "${USE_MEMORY}" = true ]; then
    echo "Memory bank: ENABLED"
    HYDRA_FULL_ERROR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    WANDB_MODE=online \
    bash "${DIRPATH_PROJECT}/examples/train/AgentGym-RL/sciworld_train.sh" \
        actor_rollout_ref.rollout.memory.enabled=true \
        actor_rollout_ref.rollout.memory.k=3 \
        actor_rollout_ref.rollout.memory.min_reward=0.5 \
        actor_rollout_ref.rollout.memory.save_path="outputs/memory_bank/sciworld_${MODEL_SIZE}" 2>&1 | tee "../logs/train_${MODEL_SIZE}_memory_$(date +%Y%m%d_%H%M%S).log"
else
    echo "Memory bank: DISABLED"
    HYDRA_FULL_ERROR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    WANDB_MODE=online \
    bash ../examples/train/AgentGym-RL/sciworld_train.sh \
        actor_rollout_ref.rollout.memory.enabled=false 2>&1 | tee "../logs/train_${MODEL_SIZE}_nomem_$(date +%Y%m%d_%H%M%S).log"
fi

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training finished with exit code: ${TRAIN_EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${TRAIN_EXIT_CODE}
