# Dockerfile for RL Training on SFT Qwen-14B
# Optimized for NVIDIA H200 with fast local model caching

FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache
ENV HF_HUB_CACHE=/workspace/hf_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Set working directory
WORKDIR /workspace

# Install Python dependencies (base packages already in NGC container)
RUN pip install --no-cache-dir \
    transformers>=4.45.0 \
    datasets>=2.18.0 \
    accelerate>=0.27.0 \
    trl>=0.12.0 \
    peft>=0.10.0 \
    wandb \
    tqdm \
    huggingface_hub

# Install flash-attn (pre-compiled for CUDA)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Create cache directories
RUN mkdir -p /workspace/hf_cache /workspace/models /workspace/outputs

# Copy project files (done at runtime via volume mount for flexibility)
# COPY . /workspace/project

# Default command
CMD ["bash"]





