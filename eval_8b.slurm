#!/bin/bash
#SBATCH --job-name=eval_3b_n8
#SBATCH --account=hartvigsen_lab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pct4et@virginia.edu
#SBATCH --exclude=udc-an28-1,udc-an28-7,udc-an40-1


export ckpt_path="${HOME}/agentgym_rl/AgentGym-RL/saves/3b_n8/global_step_75/actor"



# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

module load gcc/14
source "${HOME}"
# module --latest load miniforge

export DIRPATH_PROJECT="${HOME}/agentgym_rl"

cd "${DIRPATH_PROJECT}"


# Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh  # adjust path if needed
conda init bash
conda activate agog

echo "CONDA_PREFIX=${CONDA_PREFIX}"

export CONDA_PREFIX="${HOME}/envs/agog"

export PATH="${HOME}/miniforge3/bin/:${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${DIRPATH_PROJECT}/AgentGym-RL:${DIRPATH_PROJECT}/AgentGym/agentenv:${DIRPATH_PROJECT}/AgentGym/agentenv-sciworld:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export PYTHON_BIN="${CONDA_PREFIX}/bin/python"

echo "ll PATH: $(ll ${PATH})"
echo "which python: $(which python)"
# "${HOME}/miniforge3/bin/conda" activate /home/pct4et/envs/agog
# /home/zhanwen/miniconda3/condabin/conda
# Start sciworld env server in background
echo "Starting sciworld server on port 36001..."
sciworld --host localhost --port 36001 &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Wait for server to be ready
sleep 10
echo "Server should be ready, starting eval..."

# Run eval
# cd /home/pct4et/agentgym_rl/AgentGym-RL
# bash examples/eval/sciworld_eval.sh 2>&1 | tee log_eval_512gb_80gb_3b_n8_slurm.log
bash examples/eval/sciworld_eval.sh 2>&1
EVAL_EXIT_CODE=$?

# Cleanup: kill the server
echo "Eval finished with exit code: $EVAL_EXIT_CODE"
echo "Killing server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null

echo "End time: $(date)"
exit $EVAL_EXIT_CODE
