#!/bin/bash
# Setup script to use fast local storage instead of slow NFS
# Run this BEFORE training to speed up model loading significantly

set -e

echo "=== Fast Model Cache Setup for NFS Machines ==="

# Detect available fast local storage
FAST_STORAGE=""

# Priority order: /raid (local NVMe), /scratch, /local, /dev/shm, /tmp
if [ -d "/raid" ] && [ -w "/raid" ]; then
    FAST_STORAGE="/raid/${USER}/hf_cache"
    echo "Found /raid (local NVMe) - using for model cache (FAST!)"
elif [ -d "/scratch" ] && [ -w "/scratch" ]; then
    FAST_STORAGE="/scratch/${USER}/hf_cache"
    echo "Found /scratch - using for model cache"
elif [ -d "/local" ] && [ -w "/local" ]; then
    FAST_STORAGE="/local/${USER}/hf_cache"
    echo "Found /local - using for model cache"
elif [ -d "/dev/shm" ] && [ "$(df -BG /dev/shm | tail -1 | awk '{print $4}' | tr -d 'G')" -gt 40 ]; then
    FAST_STORAGE="/dev/shm/${USER}/hf_cache"
    echo "Using /dev/shm (ramdisk) for model cache (VERY FAST but cleared on reboot)"
elif [ -d "/tmp" ] && [ "$(df -BG /tmp | tail -1 | awk '{print $4}' | tr -d 'G')" -gt 40 ]; then
    FAST_STORAGE="/tmp/hf_cache_${USER}"
    echo "Using /tmp for model cache (WARNING: cleared on reboot)"
else
    echo "WARNING: No fast local storage found with enough space, using NFS (will be slow)"
    FAST_STORAGE="${HOME}/.cache/huggingface"
fi

# Create cache directory
mkdir -p "$FAST_STORAGE"

# Set environment variables
export HF_HOME="$FAST_STORAGE"
export HF_HUB_CACHE="$FAST_STORAGE"
export TRANSFORMERS_CACHE="$FAST_STORAGE"
export HUGGINGFACE_HUB_CACHE="$FAST_STORAGE"

echo ""
echo "Cache location: $FAST_STORAGE"
echo ""
echo "Add these to your .bashrc or run before training:"
echo "----------------------------------------"
echo "export HF_HOME=\"$FAST_STORAGE\""
echo "export HF_HUB_CACHE=\"$FAST_STORAGE\""
echo "export TRANSFORMERS_CACHE=\"$FAST_STORAGE\""
echo "export HUGGINGFACE_HUB_CACHE=\"$FAST_STORAGE\""
echo "----------------------------------------"

# Check if model already cached
MODEL_DIR="$FAST_STORAGE/hub/models--billxbf--Nano-Raccoon-Preview-1104"
if [ -d "$MODEL_DIR" ]; then
    echo ""
    echo "Model already cached at: $MODEL_DIR"
    echo "Size: $(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)"
else
    echo ""
    echo "Model not cached yet. Download with:"
    echo "  huggingface-cli download billxbf/Nano-Raccoon-Preview-1104"
    echo ""
    echo "Or let the training script download it automatically."
fi

echo ""
echo "=== Setup Complete ==="

