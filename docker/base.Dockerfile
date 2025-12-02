# Base Dockerfile for AgentGym-RL
# Provides CUDA 12.4 + PyTorch 2.4 + Python 3.10 foundation
#
# Usage:
#   docker build -f docker/base.Dockerfile -t agentgym-rl/base:latest .
#
# This image is used as the foundation for training and scripts images.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    software-properties-common \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.4 with CUDA 12.4
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install flash-attention (prebuilt wheel for CUDA 12 + PyTorch 2.4)
ARG FLASH_ATTENTION_VERSION=2.7.3
RUN pip install flash-attn==${FLASH_ATTENTION_VERSION} --no-build-isolation || \
    (wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTENTION_VERSION}/flash_attn-${FLASH_ATTENTION_VERSION}+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install flash_attn-${FLASH_ATTENTION_VERSION}+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    rm -f flash_attn-*.whl)

# Set working directory
WORKDIR /workspace

# Environment variables for optimal CUDA operation
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Default command
CMD ["/bin/bash"]
