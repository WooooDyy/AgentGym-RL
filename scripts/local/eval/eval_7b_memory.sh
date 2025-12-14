#!/bin/bash
# Local Evaluation Script for ScienceWorld with Memory (7B model)
# Usage: bash scripts/local/eval_7b_memory.sh [checkpoint_path]

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
export MODEL_SIZE="7b"
export USE_MEMORY=true  # Set to false to disable memory

# Use provided checkpoint or default
if [ -n "$1" ]; then
    export ckpt_path="$1"
else
    export ckpt_path="${HOME}/agentgym_rl/AgentGym-RL/saves/${MODEL_SIZE}_n8/global_step_125/actor"
fi

echo "=========================================="
echo "ScienceWorld Evaluation (Local)"
echo "Model Size: ${MODEL_SIZE}"
echo "Checkpoint: ${ckpt_path}"
echo "Memory Enabled: ${USE_MEMORY}"
echo "Start time: $(date)"
echo "=========================================="

# Check if checkpoint exists
if [ ! -d "${ckpt_path}" ]; then
    echo "Error: Checkpoint not found at ${ckpt_path}"
    exit 1
fi

# ============================================================================
# Setup Paths
# ============================================================================
export DIRPATH_PROJECT="${HOME}/agentgym_rl"
cd "${DIRPATH_PROJECT}"

# ============================================================================
# Conda Environment
# ============================================================================
echo "Activating conda environment..."
source ~/miniforge3/etc/profile.d/conda.sh  # Adjust if your conda is elsewhere
conda activate agog  # Or your environment name

export PATH="${HOME}/miniforge3/bin/:${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${DIRPATH_PROJECT}/AgentGym-RL:${DIRPATH_PROJECT}/AgentGym/agentenv:${DIRPATH_PROJECT}/AgentGym/agentenv-sciworld:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
export TMPDIR="${HOME}/tmp"
mkdir -p "${TMPDIR}"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# ============================================================================
# Start ScienceWorld Server
# ============================================================================
# Find available port
export PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Starting sciworld server on port ${PORT}..."
sciworld --host localhost --port "${PORT}" > /dev/null 2>&1 &
export SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

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
# Run Evaluation
# ============================================================================
echo ""
echo "Starting evaluation..."
echo "=========================================="

# Override environment variables
export env_server_url="http://localhost:${PORT}"

# Check if eval script exists
if [ ! -f "examples/eval/sciworld_eval.sh" ]; then
    echo "Error: examples/eval/sciworld_eval.sh not found"
    exit 1
fi

# Run evaluation script with memory configuration
cd AgentGym-RL

if [ "${USE_MEMORY}" = true ]; then
    echo "Memory bank: ENABLED"
    # Note: Load memory from training for eval
    bash ../examples/eval/sciworld_eval.sh \
        rollout.memory.enabled=true \
        rollout.memory.k=3 \
        rollout.memory.save_path="outputs/memory_bank/sciworld_${MODEL_SIZE}" 2>&1 | tee "../logs/eval_${MODEL_SIZE}_memory_$(date +%Y%m%d_%H%M%S).log"
else
    echo "Memory bank: DISABLED"
    bash ../examples/eval/sciworld_eval.sh \
        rollout.memory.enabled=false 2>&1 | tee "../logs/eval_${MODEL_SIZE}_nomem_$(date +%Y%m%d_%H%M%S).log"
fi

EVAL_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Evaluation finished with exit code: ${EVAL_EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${EVAL_EXIT_CODE}
