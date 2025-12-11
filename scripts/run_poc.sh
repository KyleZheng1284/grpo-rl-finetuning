#!/bin/bash
# Quick start script for the RL POC

set -e

echo "=== RL POC Training Pipeline ==="

# Step 1: Install dependencies
echo "[1/3] Installing dependencies..."
pip install -q -r requirements.txt

# Step 2: Run RL training
echo "[2/3] Starting RL training..."
echo "This will take a few hours depending on your GPU."
echo "Monitor with: tail -f wandb/latest-run/logs/debug.log"

python scripts/train_rl_poc.py

# Step 3: Evaluate
echo "[3/3] Running evaluation..."
python scripts/evaluate_models.py \
    --models ./Nano-Raccoon-Preview-1104 ./nano-raccoon-rl \
    --output ./eval_results/comparison.json

echo "=== Done! ==="
echo "Results saved to ./eval_results/comparison.json"

