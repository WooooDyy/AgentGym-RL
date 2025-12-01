---
task_categories:
- reinforcement-learning
license: cc-by-nc-4.0
language:
- en
tags:
- llm-agents
- decision-making
- multi-turn
- web-navigation
- deep-search
- text-based-games
- embodied-tasks
- scientific-tasks
---

# AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning

This repository contains the RL dataset and benchmark presented in the paper [AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning](https://huggingface.co/papers/2509.08755).

-   **Paper**: [https://huggingface.co/papers/2509.08755](https://huggingface.co/papers/2509.08755)
-   **Project Page**: [https://agentgym-rl.github.io/](https://agentgym-rl.github.io/)
-   **Code**: [https://github.com/WooooDyy/AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL)

## Abstract

Developing autonomous LLM agents capable of making a series of intelligent decisions to solve complex, real-world tasks is a fast-evolving frontier. Like human cognitive development, agents are expected to acquire knowledge and skills through exploration and interaction with the environment. Despite advances, the community still lacks a unified, interactive reinforcement learning (RL) framework that can effectively train such agents from scratch -- without relying on supervised fine-tuning (SFT) -- across diverse and realistic environments. To bridge this gap, we introduce AgentGym-RL, a new framework to train LLM agents for multi-turn interactive decision-making through RL. The framework features a modular and decoupled architecture, ensuring high flexibility and extensibility. It encompasses a wide variety of real-world scenarios, and supports mainstream RL algorithms. Furthermore, we propose ScalingInter-RL, a training approach designed for exploration-exploitation balance and stable RL optimization. In early stages, it emphasizes exploitation by restricting the number of interactions, and gradually shifts towards exploration with larger horizons to encourage diverse problem-solving strategies. In this way, the agent develops more diverse behaviors and is less prone to collapse under long horizons. We perform extensive experiments to validate the stability and effectiveness of both the AgentGym-RL framework and the ScalingInter-RL approach. Our agents match or surpass commercial models on 27 tasks across diverse environments. We offer key insights and will open-source the complete AgentGym-RL framework -- including code and datasets -- to empower the research community in developing the next generation of intelligent agents.

## Environments and Scenarios

The AgentGym-RL framework and this dataset support training and evaluation across a variety of real-world scenarios:

*   **Web Navigation**: Includes tasks from **WebArena**, a realistic and reproducible web environment containing 4 distinct domains prevalent on the internet: online shopping, discussion forums, collaborative development, and business content management.
*   **Deep Search**: Building upon **Search-R1**, this RAG-based environment enables LLMs to interact with search engines and solve multi-turn retrieval and reasoning tasks.
*   **Digital Games**: Includes **TextCraft**, a text-based crafting game environment in which agents complete tasks via natural language interactions and task-based planning.
*   **Embodied Tasks**: Includes **BabyAI** which provides a controllable grid world with text instructions for embodied reasoning in simulated environments.
*   **Scientific Tasks**: Includes **SciWorld** which offers a scientific exploration simulator where agents conduct scientific experiments through text-driven reasoning cycles.

## Sample Usage

This section provides a quick guide to setting up the environment, preparing the data (this dataset!), and running training and evaluation with the AgentGym-RL framework.

### Environment Setup

We recommend using CUDA 12.4, PyTorch 2.4, and Python 3.10. First, install the requirements using the following command:
```sh
echo "Preparing environment for agentgym-rl..."
conda create -n agentgym-rl python==3.10 -y
conda activate agentgym-rl
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# install flash-atten
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
pip3 install $FLASH_ATTENTION_NAME
rm -f $FLASH_ATTENTION_NAME
# for RL
cd AgentGym-RL
pip3 install -e .
# for agentgym
echo "Preparing environment for agentenv..."
cd AgentGym/agentenv
pip3 install -e .
pip3 install transformers==4.51.3
```

### Data Preparation

Download the AgentGym-RL-Data-ID dataset from Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID
```

### Training

For RL training:

**1. Environment Setup**

Make sure you have the required environments set up (see [Environment Setup section](#environment-setup) above).

**2. Launch the environment server**

Please launch the environment server by referring to the `README.md` of [AgentGym](https://github.com/WooooDyy/AgentGym/tree/640f8bca6901a6a6d540ff61522b813988da47c4).

**3. Training**

You can see the training example scripts for each task in the [examples/train](https://github.com/WooooDyy/AgentGym-RL/tree/main/examples/train) directory for AgentGym-RL and ScalingInter-RL. For instance, to launch AgentGym-RL training, set:

```sh
algorithm.rounds_ctrl.type=fixed \
algorithm.rounds_ctrl.rounds=15 \
```

You can see [examples/train/AgentGym-RL/webarena_train.sh](https://github.com/WooooDyy/AgentGym-RL/blob/main/examples/train/AgentGym-RL/webarena_train.sh) as an example.

To launch the ScalingInter-RL training, set:

```sh
algorithm.rounds_ctrl.type=scaling_inter_stepwise\
algorithm.rounds_ctrl.steps_scaling_inter=100 \
algorithm.rounds_ctrl.rounds=[10,20,30] \
```

You can see [examples/train/ScalingInter-RL/webarena_train.sh](https://github.com/WooooDyy/AgentGym-RL/blob/main/examples/train/ScalingInter-RL/webarena_train.sh) as an example.

### Evaluation

**1. Environment Setup**

Make sure you have the required environments set up (see [Environment Setup section](#environment-setup) above).

**2. Launch the environment server**

Please launch the environment server by referring to the `README.md` of [AgentGym](https://github.com/WooooDyy/AgentGym/tree/640f8bca6901a6a6d540ff61522b813988da47c4).

**3. Evaluation**

You can see the evaluation example scripts for each task in the `examples/eval` directory. To run the evaluation, you can see `examples/eval/webarena_eval.sh` as an example:

```sh
bash webarena_eval.sh
```

## Citation

Please cite the following paper if you find AgentGym-RL helpful!

```bibtex
@misc{xi2025agentgymrltrainingllmagents,
      title={AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning}, 
      author={Zhiheng Xi and Jixuan Huang and Chenyang Liao and Baodai Huang and Honglin Guo and Jiaqi Liu and Rui Zheng and Junjie Ye and Jiazheng Zhang and Wenxiang Chen and Wei He and Yiwen Ding and Guanyu Li and Zehui Chen and Zhengyin Du and Xuesong Yao and Yufei Xu and Jiecao Chen and Tao Gui and Zuxuan Wu and Qi Zhang and Xuanjing Huang and Yu-Gang Jiang},
      year={2025},
      eprint={2509.08755},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.08755}, 
}
```

## License

This dataset is licensed under the CC-BY-NC-4.0 License.