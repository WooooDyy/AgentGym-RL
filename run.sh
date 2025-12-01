# run.sh

conda create -n ag python numpy pandas ninja cmake setuptools_scm # triton # triton messes up the nvcc bin to 12.8
# conda create -n ag python=13 numpy pandas ninja psutil gcc=14.* gxx=14.* cmake setuptools_scm # triton # triton messes up the nvcc bin to 12.8
conda create -n ag python=13 numpy pandas ninja psutil cmake setuptools_scm # triton # triton messes up the nvcc bin to 12.8
conda activate ag
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install torch torchvision # for 12.8

screen -S flash_attn
conda activate ag
export TMPDIR="/home/pct4et/tmp"
module load gcc/14
MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache
pip install transformers datasets wandb safetensors tensordict ray codetiming omegaconf hydra-core
pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
pip install -e AgentGym-RL
pip install -e AgentGym/agentenv
pip install -e AgentGym/agentenv-sciworld

screen -S sw
# sciworld --host 0.0.0.0 --port 36001
sciworld --host localhost --port 36001
# exit screen with Ctrl+A D


screen -S ag
RAY_DEBUG=legacy bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_train.log
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install --no-build-isolation .

HF_XET_HIGH_PERFORMANCE=1 hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
HF_XET_HIGH_PERFORMANCE=1 hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct
HF_XET_HIGH_PERFORMANCE=1 hf download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
HF_XET_HIGH_PERFORMANCE=1 hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
HF_XET_HIGH_PERFORMANCE=1 hf download Qwen/Qwen2.5-14B-Instruct --local-dir models/Qwen2.5-14B-Instruct
RAY_DEBUG=legacy bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_train.log



# og:
echo "Preparing environment for agentgym-rl..."
conda create -n agog python==3.10 -y
conda activate agog
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# install flash-atten
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
pip install $FLASH_ATTENTION_NAME
rm -f $FLASH_ATTENTION_NAME
# for RL
pip install -e AgentGym-RL
pip install -e AgentGym/agentenv
pip install -e AgentGym/agentenv-sciworld
pip install vllm==0.6.3 transformers==4.51.3 tokenizers

# Use scratch for models
export DIRPATH_SAVES_SCRATCH="/scratch/${USER}/agentgym_rl_AgentGym-RL_saves"
mkdir -p "${DIRPATH_SAVES_SCRATCH}"
ln -sf "${DIRPATH_SAVES_SCRATCH}" "${HOME}/agentgym_rl/AgentGym-RL/saves"


# Use scratch for saves
export DIRPATH_MODELS_SCRATCH="/scratch/${USER}/agentgym_rl_models"
mkdir -p "${DIRPATH_MODELS_SCRATCH}"
ln -sf "${DIRPATH_MODELS_SCRATCH}" "${HOME}/agentgym_rl/models"

# bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_512gb_80gb.log
# bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_512gb_80gb.log
# RAY_DEBUG=legacy bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_512gb_80gb_3b_n8.log
bash examples/train/AgentGym-RL/sciworld_train.sh |& tee log_512gb_80gb_14b_n8.log

scp -r AgentGym-RL/AgentItemId riv:~/agentgym_rl/AgentGym-RL

# Lastly, don't forget to modify
# vim /home/pct4et/envs/agog/lib/python3.10/site-packages/vllm/version.py
# to be
# try:
#     __version__ = '0.6.3'
#     __version_tuple__ = (0, 6, 3)
# except Exception as e:
#     import warnings

#     warnings.warn(f"Failed to read commit hash:\n{e}",
#                   RuntimeWarning,
#                   stacklevel=2)

#     __version__ = "dev"
#     __version_tuple__ = (0, 0, __version__)


conda remove gcc gxx libstdcxx-ng # nope. Need to module load gcc/14
pip install click==8.0.1 # click 8.3* breaks ray
