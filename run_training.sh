#!/bin/bash
# One-command training script for NFS machines
# Handles fast cache setup, model download, and training

set -e

echo "=============================================="
echo "  RL Training on SFT Qwen-14B (Nano-Raccoon)"
echo "=============================================="
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Step 1: Setup fast local cache
echo "[1/4] Setting up fast local cache..."
source scripts/setup_fast_cache.sh

# Step 2: Install dependencies (if not in container)
echo ""
echo "[2/4] Checking dependencies..."
if ! python -c "import trl" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q torch transformers datasets accelerate trl peft wandb tqdm huggingface_hub
    # Try flash-attn (may fail if no CUDA headers)
    pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn not installed (optional)"
fi

# Step 3: Download model if not cached
echo ""
echo "[3/4] Checking model cache..."
if ! python -c "from huggingface_hub import snapshot_download; snapshot_download('billxbf/Nano-Raccoon-Preview-1104', local_files_only=True)" 2>/dev/null; then
    echo "Model not cached. Downloading to fast local storage..."
    echo "This will take 5-15 minutes..."
    huggingface-cli download billxbf/Nano-Raccoon-Preview-1104 --cache-dir "$HF_HOME"
else
    echo "Model already cached!"
fi

# Step 4: Run training
echo ""
echo "[4/4] Starting RL training..."
echo "Logs will be saved to: ./outputs/training.log"
echo ""

mkdir -p outputs

# Run with nohup so it continues if SSH disconnects
nohup python scripts/train_rl_poc.py > outputs/training.log 2>&1 &
TRAIN_PID=$!

echo "Training started in background (PID: $TRAIN_PID)"
echo ""
echo "Monitor with:"
echo "  tail -f outputs/training.log"
echo ""
echo "Or check GPU usage:"
echo "  nvidia-smi -l 1"
echo ""
echo "To stop training:"
echo "  kill $TRAIN_PID"





