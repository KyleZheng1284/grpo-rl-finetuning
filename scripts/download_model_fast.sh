#!/bin/bash
# Download model to fast local storage with progress tracking
# This avoids slow NFS and downloads directly to local SSD/scratch

set -e

MODEL_ID="${1:-billxbf/Nano-Raccoon-Preview-1104}"
echo "=== Fast Model Download ==="
echo "Model: $MODEL_ID"

# Source the cache setup
source "$(dirname "$0")/setup_fast_cache.sh"

echo ""
echo "Downloading to: $HF_HOME"
echo "This may take 5-15 minutes depending on network speed..."
echo ""

# Download with resume support and progress
huggingface-cli download "$MODEL_ID" \
    --cache-dir "$HF_HOME" \
    --resume-download

echo ""
echo "=== Download Complete ==="
echo ""
echo "Model cached at: $HF_HOME"
echo "Cache size: $(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
echo ""
echo "You can now run training with:"
echo "  python scripts/train_rl_poc.py"





