# Quick Start - RL Training on SFT Model

## What You Have
- **Model**: Nano-Raccoon (Qwen-14B SFT'd on MiniMax-M2 distillation) - already downloaded to `/raid/kyzheng/hf_cache`
- **Hardware**: 2x NVIDIA H200 NVL (143GB each = 286GB total)

## What You're Training
Adding **GRPO (Reinforcement Learning)** on top of the SFT model using code execution rewards.

## Datasets (All FREE, No API Keys Needed)

| Dataset | Examples | Tests Included | Source |
|---------|----------|----------------|--------|
| MBPP | 374 | Yes (asserts) | HuggingFace |
| HumanEval | 164 | Yes (unit tests) | OpenAI |
| APPS | ~2000 | Yes (I/O pairs) | CodeParrot |
| CodeContests | ~1000 | Yes (public tests) | DeepMind |
| code_search_net | ~2000 | No (heuristic) | GitHub |
| **TOTAL** | **~5500** | Most have tests | Auto-download |

## Run Training (One Command)

```bash
cd /home/nfs/kyzheng/deep-learning-proj

docker run --gpus all --ipc=host --shm-size=16g --rm \
  -v /raid/kyzheng/hf_cache:/hf_cache \
  -v $(pwd):/workspace \
  -e HF_HOME=/hf_cache \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.01-py3 \
  bash -c "pip install -q --upgrade typing_extensions && \
           pip install -q transformers datasets accelerate trl peft tqdm huggingface_hub && \
           python scripts/train_rl_poc.py"
```

## Expected Output

1. Downloads 5 datasets from HuggingFace (~5500 examples)
2. Loads Nano-Raccoon model from cache
3. Runs GRPO training for 3000 steps
4. Saves RL-trained model to `./nano-raccoon-rl/`

## Training Time Estimate

- ~4-8 hours for 3000 steps on 2x H200
- Checkpoints saved every 500 steps

## Optional: Generate More Synthetic Data

Only if you want MORE training examples beyond the 5500 included:

```bash
# Get free API key from https://build.nvidia.com/
export NVIDIA_API_KEY="nvapi-your-key"

# Generate 100 additional examples
python scripts/create_synthetic_data.py \
    --backend nvidia \
    --model qwen/qwen3-235b-a22b-instruct \
    --num-samples 100
```

## Evaluate After Training

```bash
docker run --gpus all --ipc=host --rm \
  -v /raid/kyzheng/hf_cache:/hf_cache \
  -v $(pwd):/workspace \
  -e HF_HOME=/hf_cache \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.01-py3 \
  bash -c "pip install -q transformers datasets accelerate tqdm huggingface_hub && \
           python scripts/evaluate_models.py --models billxbf/Nano-Raccoon-Preview-1104 ./nano-raccoon-rl"
```





