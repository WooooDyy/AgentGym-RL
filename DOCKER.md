# Docker Virtual Environments for AgentGym-RL

This document describes the Docker-based virtual environment setup for reproducible training, evaluation, and utility operations in AgentGym-RL.

## Overview

The Docker infrastructure provides:

- **Reproducible Environments**: Consistent CUDA 12.4 + PyTorch 2.4 + Python 3.10 across all machines
- **Plug-and-Play Scripts**: Run model merging, formatting, and other utilities without local setup
- **Isolated Training**: GPU-enabled containers for RL training without dependency conflicts
- **Environment Servers**: Containerized environment servers for SearchQA, BabyAI, SciWorld, etc.

## Quick Start

### Prerequisites

- Docker 24.0+ with Docker Compose v2
- NVIDIA Container Toolkit (for GPU support)
- At least 32GB disk space for images

### Build All Images

```bash
make docker-build
```

This builds:

1. `agentgym-rl/base:latest` - Base image with CUDA, PyTorch, flash-attention
2. `agentgym-rl/train:latest` - Training environment with verl and agentenv
3. `agentgym-rl/scripts:latest` - Utilities for model merging and formatting
4. `agentgym/eval:latest` - Lightweight evaluation runner

### Start Training Shell

```bash
make docker-train-shell
```

Inside the container:

```bash
# Run training
python -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=5 \
    ...
```

## Docker Images

### Base Image (`docker/base.Dockerfile`)

Foundation image with:

- CUDA 12.4.1 (devel)
- Python 3.10
- PyTorch 2.4.0
- flash-attention 2.7.3

Build independently:

```bash
make docker-build-base
```

### Training Image (`docker/train.Dockerfile`)

Extends base with:

- verl (AgentGym-RL training framework)
- agentenv (environment client)
- All training dependencies

Build:

```bash
make docker-build-train
```

Usage:

```bash
# Interactive shell
make docker-train-shell

# Or via docker compose
docker compose --profile train run --rm train /bin/bash
```

### Scripts Image (`docker/scripts.Dockerfile`)

Extends base with:

- transformers
- huggingface_hub
- yapf (formatter)
- Model loading utilities

Build:

```bash
make docker-build-scripts
```

## Common Operations

### Model Merging

Merge FSDP checkpoints to HuggingFace format:

```bash
# Single checkpoint
make docker-merge LOCAL_DIR=saves/global_step_100/actor

# With custom output directory
make docker-merge LOCAL_DIR=saves/global_step_100/actor SAVE_DIR=models/merged

# Upload to HuggingFace
make docker-merge LOCAL_DIR=saves/global_step_100/actor HF_UPLOAD_PATH=username/model-name
```

### Code Formatting

Format AgentGym-RL code with yapf:

```bash
make docker-format
```

### Environment Servers

Start an environment server:

```bash
# SearchQA (default)
make docker-env

# Other environments
make docker-env ENV=babyai
make docker-env ENV=sciworld
```

### Evaluation

Run evaluation against a running environment server:

```bash
make docker-eval ENV=searchqa
```

## Volume Mounts

The Docker setup uses the following volume mounts:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./models` | `/workspace/models` | Pre-trained models |
| `./saves` | `/workspace/saves` | Training checkpoints |
| `./data` | `/workspace/data` | Training data |
| `./AgentItemId` | `/workspace/AgentItemId` | Training item IDs |
| `./AgentEval` | `/workspace/AgentEval` | Evaluation data |

## Environment Variables

Set these in `.env` or export before running:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `searchqa` | Environment name |
| `ENV_PORT` | `36001` | Environment server port |
| `MODEL` | `gpt-4o-mini` | Model for evaluation |
| `MAX_ROUND` | `10` | Max interaction rounds |
| `LOCAL_DIR` | `saves/checkpoint` | Checkpoint path for merging |
| `OPENAI_API_KEY` | - | Required for evaluation |
| `WANDB_API_KEY` | - | Optional for training logging |
| `WANDB_MODE` | `offline` | WandB mode |

## Docker Compose Profiles

The `docker-compose.yml` uses profiles to organize services:

| Profile | Services | Command |
|---------|----------|---------|
| `build` | base | `docker compose --profile build up base` |
| `train` | train | `docker compose --profile train up -d` |
| `scripts` | scripts | `docker compose --profile scripts up -d` |
| `model-merger` | model-merger | `docker compose --profile model-merger run --rm model-merger` |
| `formatter` | formatter | `docker compose --profile formatter run --rm formatter` |
| `env` | env-server | `docker compose --profile env up -d` |
| `eval` | eval-runner | `docker compose --profile eval up` |

## Troubleshooting

### GPU Not Detected

Ensure NVIDIA Container Toolkit is installed:

```bash
nvidia-smi  # Should work
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi  # Should work
```

### Out of Memory

Increase shared memory for training:

```bash
docker compose --profile train run --rm --shm-size=32g train /bin/bash
```

### Build Failures

Clean and rebuild:

```bash
make docker-clean
make docker-build
```

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Run as current user
docker compose --profile train run --rm --user $(id -u):$(id -g) train /bin/bash
```

## Development Workflow

### Typical Training Session

```bash
# 1. Build images (first time only)
make docker-build

# 2. Start environment server
make docker-env ENV=searchqa

# 3. Enter training container
make docker-train-shell

# 4. Inside container: run training
HYDRA_FULL_ERROR=1 python -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=5 \
    data.train_file=AgentItemId/searchqa_train.json \
    actor_rollout_ref.agentgym.task_name=searchqa \
    actor_rollout_ref.agentgym.env_addr=http://host.docker.internal:36001 \
    actor_rollout_ref.model.path=/workspace/models/Qwen2.5-7B-Instruct \
    trainer.default_local_dir=/workspace/saves/experiment1 \
    trainer.total_epochs=20

# 5. Merge checkpoint to HuggingFace format
make docker-merge LOCAL_DIR=saves/experiment1/global_step_100/actor
```

### CI/CD Integration

For automated pipelines:

```yaml
# GitHub Actions example
jobs:
  train:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - name: Build images
        run: make docker-build
      - name: Run training
        run: |
          docker compose --profile train run --rm train \
            python -m verl.agent_trainer.main_ppo ...
```

## File Structure

```text
AgentGym-RL/
├── docker/
│   ├── base.Dockerfile      # CUDA + PyTorch base
│   ├── train.Dockerfile     # Training environment
│   └── scripts.Dockerfile   # Utilities environment
├── docker-compose.yml       # Service orchestration
├── Dockerfile.eval          # Evaluation runner
├── .dockerignore            # Build context exclusions
├── Makefile                 # Convenient targets
└── DOCKER.md                # This documentation
```
