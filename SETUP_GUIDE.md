# Complete Setup Guide

## TL;DR - Minimal Path (Recommended for POC)

### Option A: One-Command Setup (Recommended for NFS machines)

```bash
# Just run this - handles fast cache, model download, and training
chmod +x run_training.sh
./run_training.sh
```

### Option B: Manual Setup

```bash
# 1. Setup fast local cache (CRITICAL for NFS machines!)
source scripts/setup_fast_cache.sh

# 2. Download model to fast storage
huggingface-cli download billxbf/Nano-Raccoon-Preview-1104

# 3. Install deps
pip install torch transformers datasets accelerate trl peft wandb tqdm

# 4. Run RL training (uses MBPP dataset, no synthetic data needed)
python scripts/train_rl_poc.py

# 5. Evaluate
python scripts/evaluate_models.py --models ./Nano-Raccoon-Preview-1104 ./nano-raccoon-rl
```

### Option C: Docker (Most Reproducible)

```bash
# Build and run in container with fast local cache
docker compose run --rm train
```

**That's it.** No API keys, no serving models, no synthetic data generation needed for the POC.

---

## NFS Machine Speed Optimization

**Problem**: NFS is slow for large model files (~28GB).

**Solution**: Use local fast storage (`/tmp`, `/scratch`, or `/local`) for model cache.

```bash
# Automatic setup - detects best local storage
source scripts/setup_fast_cache.sh

# This sets:
# - HF_HOME=/tmp/hf_cache_$USER  (or /scratch if available)
# - Model downloads go to fast local storage
# - Training loads from local disk, not NFS
```

**Speed comparison**:
| Storage | Model Load Time | Notes |
|---------|-----------------|-------|
| NFS | 10-30 minutes | Slow, network bottleneck |
| Local SSD/scratch | 1-2 minutes | Much faster |
| Local /tmp | 1-2 minutes | Fast but cleared on reboot |

---

## Understanding the Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WHAT BINFENG ALREADY DID                         │
│  MiniMax-M2 (200B) ──► Generate SFT Data ──► Train Qwen-14B (SFT)  │
│                                                                     │
│  Result: Nano-Raccoon-Preview-1104 (your starting point)           │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     WHAT YOU'RE DOING (POC)                         │
│  Nano-Raccoon (SFT) ──► GRPO with Code Rewards ──► Nano-Raccoon-RL │
│                                                                     │
│  Dataset: MBPP (built-in, no generation needed)                    │
│  Reward: Code execution + test passing                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Option A: Just Run RL (Simplest - Recommended)

**What you need**: 2x H200, the SFT checkpoint, MBPP dataset (auto-downloaded)

**What you DON'T need**: MiniMax-M2, API keys, synthetic data

```bash
# Install
pip install -r requirements.txt

# Train
python scripts/train_rl_poc.py

# Evaluate
python scripts/evaluate_models.py --models ./Nano-Raccoon-Preview-1104 ./nano-raccoon-rl
```

---

## Option B: Generate More Synthetic Data with NVIDIA API

**Why**: If you want more diverse training examples beyond MBPP.

**Cost**: Free tier available at build.nvidia.com

### Step 1: Get NVIDIA API Key

1. Go to https://build.nvidia.com/
2. Sign in / Create account
3. Click any model → "Get API Key"
4. Copy the key (starts with `nvapi-`)

### Step 2: Set Environment Variable

```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
```

### Step 3: Generate Data

```bash
# Using Qwen3-235B (very capable)
python scripts/create_synthetic_data.py \
    --backend nvidia \
    --model qwen/qwen3-235b-a22b-instruct \
    --num-samples 100 \
    --output data/synthetic_nvidia.jsonl

# Or using Llama 3.3-70B
python scripts/create_synthetic_data.py \
    --backend nvidia \
    --model meta/llama-3.3-70b-instruct \
    --num-samples 100
```

### Available NVIDIA NIM Models

| Model | Size | Notes |
|-------|------|-------|
| `qwen/qwen3-235b-a22b-instruct` | 235B | Very capable, MoE |
| `meta/llama-3.3-70b-instruct` | 70B | Great for coding |
| `nvidia/llama-3.1-nemotron-70b-instruct` | 70B | NVIDIA fine-tuned |
| `mistralai/mistral-large-2-instruct-2411` | Large | Strong reasoning |

---

## Option C: Serve MiniMax-M2 Locally (Advanced)

**Why**: If you want to use the exact same teacher model, or add KL penalty during RL.

**Hardware needed**: 8x A100-80GB or 8x H100 (MiniMax-M2 is ~200B MoE)

**Recommendation**: Skip this for POC. Use NVIDIA API instead.

### If you really want to serve MiniMax-M2:

```bash
# Download (WARNING: ~400GB)
huggingface-cli download MiniMaxAI/MiniMax-M2

# Serve with vLLM (needs 8 GPUs with 80GB each)
vllm serve MiniMaxAI/MiniMax-M2 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000

# Generate data from local server
python scripts/create_synthetic_data.py \
    --backend vllm \
    --vllm-url http://localhost:8000/v1 \
    --model MiniMaxAI/MiniMax-M2
```

---

## Option D: Serve Your SFT Model for Inference

**Why**: To test the model before/after RL training.

```bash
# Serve with vLLM (fits on 1x H200 easily)
vllm serve ./Nano-Raccoon-Preview-1104 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000

# Or after RL training:
vllm serve ./nano-raccoon-rl \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

Then query it:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./Nano-Raccoon-Preview-1104",
    "messages": [{"role": "user", "content": "Write a Python function to reverse a string"}]
  }'
```

---

## Hardware Requirements Summary

| Task | Hardware | Time |
|------|----------|------|
| **RL Training (POC)** | 2x H200 | 4-8 hours |
| Serve Nano-Raccoon for inference | 1x H200 (or even 1x A100-40GB with quantization) | Instant |
| Serve MiniMax-M2 | 8x A100-80GB or 8x H100 | ~30 min to load |
| NVIDIA API for synthetic data | Just your laptop | Minutes |

---

## Exact Steps for Your 2x H200 Setup

### Day 1: Run the POC

```bash
# SSH into your H200 machine
ssh user@your-h200-machine

# Clone/copy the project
cd /path/to/deep-learning-proj

# Install dependencies
pip install torch transformers datasets accelerate trl peft wandb tqdm flash-attn

# Verify model is downloaded
ls -la Nano-Raccoon-Preview-1104/

# Start training (runs overnight)
nohup python scripts/train_rl_poc.py > training.log 2>&1 &

# Monitor
tail -f training.log
# Or watch wandb dashboard
```

### Day 2: Evaluate

```bash
# Compare SFT vs RL
python scripts/evaluate_models.py \
    --models ./Nano-Raccoon-Preview-1104 ./nano-raccoon-rl \
    --output ./eval_results/comparison.json

# Optionally serve the RL model
vllm serve ./nano-raccoon-rl --trust-remote-code --port 8000
```

---

## Common Issues

### "CUDA out of memory"

```python
# In train_rl_poc.py, reduce batch size:
BATCH_SIZE = 2  # Instead of 4
```

### "Dataset not found"

```bash
# MBPP downloads automatically, but if issues:
pip install datasets --upgrade
```

### Model download stuck

```bash
# Cancel git clone, use huggingface-cli instead:
huggingface-cli download billxbf/Nano-Raccoon-Preview-1104 \
    --local-dir ./Nano-Raccoon-Preview-1104
```

### "flash_attn not found"

```bash
pip install flash-attn --no-build-isolation
```

---

## Docker Setup (For Reproducibility)

If you want a containerized environment:

### Build the Container

```bash
cd /home/nfs/kyzheng/deep-learning-proj
docker build -t rl-training .
```

### Run Training in Container

```bash
# Interactive shell
docker run --gpus all -it --rm \
    -v $(pwd):/workspace/project \
    -v /tmp/hf_cache:/workspace/hf_cache \
    -e HF_HOME=/workspace/hf_cache \
    --shm-size=16g \
    rl-training bash

# Inside container:
cd /workspace/project
python scripts/train_rl_poc.py
```

### Or Use Docker Compose

```bash
# Training (runs in background)
docker compose up train

# Interactive shell
docker compose run --rm shell
```

**Docker Benefits**:
- Consistent environment across machines
- Pre-installed flash-attn and CUDA
- Easy to reproduce on any machine with NVIDIA GPUs

---

## Quick Reference Commands

```bash
# Check GPU availability
nvidia-smi

# Monitor training
tail -f outputs/training.log

# Check model cache size
du -sh $HF_HOME

# Kill background training
pkill -f train_rl_poc.py

# Clean cache (if needed)
rm -rf /tmp/hf_cache_$USER
```
