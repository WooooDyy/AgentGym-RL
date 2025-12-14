#!/bin/bash

# Check for required environment variables
if [ -z "${MODEL_SIZE}" ]; then
    echo "Error: MODEL_SIZE environment variable required"
    exit 1
fi

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model Size: ${MODEL_SIZE}"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

module load gcc/14
source "${HOME}"

export DIRPATH_PROJECT="${HOME}/agentgym_rl"

cd "${DIRPATH_PROJECT}"

# Activate conda environment
conda activate agog

echo "CONDA_PREFIX=${CONDA_PREFIX}"

export PATH="${HOME}/miniforge3/bin/:${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${DIRPATH_PROJECT}/AgentGym-RL:${DIRPATH_PROJECT}/AgentGym/agentenv:${DIRPATH_PROJECT}/AgentGym/agentenv-sciworld:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
export TMPDIR="/scratch/${USER}/tmp"
export RAY_TMPDIR="/scratch/${USER}/tmp"
export TRITON_HOME="/scratch/${USER}"


echo "ll PATH: $(ls -alt ${PATH})"
echo "which python: $(which python)"

# Start sciworld env server in background
export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
echo "Starting sciworld server on port ${PORT}..."
sciworld --host localhost --port "${PORT}" &
export SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Wait for server to be ready
sleep 5
echo "Server should be ready..."
