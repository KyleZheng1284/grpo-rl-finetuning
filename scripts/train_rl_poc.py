"""
RL POC: GRPO on top of SFT checkpoint with coding tasks

Hardware: 2x H200 (160GB total) - can do full fine-tuning!

DATASETS (all FREE, no API keys needed, auto-download from HuggingFace):
- MBPP: 374 basic Python problems with assert tests
- HumanEval: 164 OpenAI problems with unit tests  
- APPS: ~2000 introductory problems with I/O test cases
- CodeContests: ~1000 competitive programming with public tests
- code_search_net: ~2000 function implementations (heuristic scoring)

Total: ~5500 diverse coding problems with verifiable test cases

OPTIONAL (requires NVIDIA_API_KEY):
- Generate additional synthetic data with NVIDIA NIM models
- See: python scripts/create_synthetic_data.py --help

REWARD: Code execution - if tests pass, reward=1.0 (verifiable outcomes)
"""

import os
import torch
import re
import json
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import warnings

# Disable wandb if not configured (avoids import issues)
os.environ.setdefault("WANDB_DISABLED", "true")

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG - Optimized for 2x H200
# ============================================================================

# Model path - will try HuggingFace cache first, then local
MODEL_PATH = os.environ.get("MODEL_PATH", "billxbf/Nano-Raccoon-Preview-1104")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./nano-raccoon-rl")

# With 2x H200 you can do full fine-tuning, no LoRA needed!
USE_LORA = False

# Training hyperparameters - aggressive for H200s
BATCH_SIZE = 4  # Per device
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-6
MAX_STEPS = 500  # Reduced for faster POC (~8-10 hours)
NUM_GENERATIONS = 4  # Samples per prompt for GRPO ranking
MAX_COMPLETION_LENGTH = 1024
MAX_PROMPT_LENGTH = 512

# Dataset size limit (set to None for full dataset)
MAX_DATASET_SIZE = None  # Use all examples with test cases (~580)

# Code extraction method: "regex" (fast, free) or "llm" (accurate, needs NVIDIA_API_KEY)
CODE_EXTRACTION_METHOD = os.environ.get("CODE_EXTRACTION", "regex")

# LLM extraction setup (only used if CODE_EXTRACTION_METHOD="llm")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
LLM_EXTRACTION_MODEL = "meta/llama-3.1-8b-instruct"  # Fast, cheap model for extraction


def extract_code_with_llm(response: str) -> str:
    """Use NVIDIA API to extract just the Python code from a response."""
    if not NVIDIA_API_KEY:
        print("[WARNING] NVIDIA_API_KEY not set, falling back to regex extraction")
        return extract_code_from_response(response)
    
    try:
        import requests
        
        extraction_prompt = f"""Extract ONLY the Python function code from the following response. 
Output ONLY the raw Python code, nothing else. No markdown, no explanations.

Response:
{response[:2000]}

Python code:"""
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": LLM_EXTRACTION_MODEL,
            "messages": [{"role": "user", "content": extraction_prompt}],
            "max_tokens": 500,
            "temperature": 0.1
        }
        
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if resp.status_code == 200:
            code = resp.json()["choices"][0]["message"]["content"].strip()
            # Clean up any remaining markdown
            if code.startswith("```"):
                code = re.sub(r"```(?:python)?\n?", "", code)
                code = code.rstrip("`")
            return code.strip()
    except Exception as e:
        print(f"[LLM extraction failed: {e}] Falling back to regex")
    
    return extract_code_from_response(response)


# ============================================================================
# CODE EXECUTION REWARD - The key for coding tasks
# ============================================================================

def safe_execute(code: str, test_code: str, timeout: float = 5.0) -> dict:
    """
    Safely execute code and run test cases.
    Returns dict with success status and details.
    """
    import signal
    
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out")
    
    result = {
        "executed": False,
        "tests_passed": False,
        "error": None,
        "output": "",
    }
    
    # Combine code with test
    full_code = f"{code}\n\n{test_code}"
    
    # Set up execution environment
    exec_globals = {
        "__builtins__": __builtins__,
        "print": print,
        "len": len,
        "range": range,
        "list": list,
        "dict": dict,
        "set": set,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "sum": sum,
        "max": max,
        "min": min,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "abs": abs,
        "all": all,
        "any": any,
        "reversed": reversed,
    }
    
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    try:
        # Set timeout (Unix only)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(full_code, exec_globals)
            result["executed"] = True
            result["tests_passed"] = True
            result["output"] = stdout_capture.getvalue()
        except AssertionError as e:
            result["executed"] = True
            result["tests_passed"] = False
            result["error"] = f"Test failed: {e}"
        except TimeoutError:
            result["error"] = "Timeout"
        except Exception as e:
            result["executed"] = False
            result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
    except Exception as e:
        result["error"] = f"Setup error: {e}"
    
    return result


def extract_code_from_response(response: str) -> str:
    """Extract Python code from markdown code blocks or raw response."""
    
    # Method 1: Try to find code block (```python ... ```)
    code_pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(code_pattern, response, re.DOTALL)
    if matches:
        # Return the longest code block (most likely the actual solution)
        return max(matches, key=len).strip()
    
    # Method 2: Find function definition and extract just the function
    # Look for "def function_name(" and extract until we hit non-code
    func_start_pattern = r"(def\s+\w+\s*\([^)]*\)\s*(?:->.*?)?:)"
    func_match = re.search(func_start_pattern, response)
    
    if func_match:
        start_idx = func_match.start()
        lines = response[start_idx:].split('\n')
        
        code_lines = []
        in_function = False
        base_indent = None
        
        for line in lines:
            stripped = line.lstrip()
            
            # Start of function
            if stripped.startswith('def ') and not in_function:
                in_function = True
                base_indent = len(line) - len(stripped)
                code_lines.append(line)
                continue
            
            if in_function:
                # Empty lines are okay
                if not stripped:
                    code_lines.append('')
                    continue
                
                current_indent = len(line) - len(stripped)
                
                # Check if this line is part of the function (indented more than base)
                # or a continuation of function body
                if current_indent > base_indent or stripped.startswith('#'):
                    # Skip lines that are clearly not code
                    if any(stripped.startswith(x) for x in ['Input', 'Output', 'Example', 'Test', 'Wait', 'Let', 'Yes', 'No', 'The ', 'So ', 'Okay']):
                        break
                    # Skip lines with special unicode characters
                    if any(c in stripped for c in ['→', '←', '↔', '•', '✓', '✗']):
                        break
                    code_lines.append(line)
                else:
                    # New function or end of current function
                    if stripped.startswith('def '):
                        # Another function - include it too
                        code_lines.append(line)
                        base_indent = current_indent
                    else:
                        # End of function
                        break
        
        if code_lines:
            extracted = '\n'.join(code_lines).strip()
            # Final cleanup - remove any trailing non-code
            clean_lines = []
            for line in extracted.split('\n'):
                # Stop if we hit obvious non-code
                if re.match(r'^[A-Z][a-z].*[.!?]$', line.strip()):  # English sentence
                    break
                clean_lines.append(line)
            return '\n'.join(clean_lines).strip()
    
    # Method 3: Last resort - find any "def" line and take a few lines after
    simple_match = re.search(r'(def\s+\w+[^\n]*\n(?:[ \t]+[^\n]*\n?)*)', response)
    if simple_match:
        return simple_match.group(1).strip()
    
    # Nothing found - return empty to trigger heuristic scoring
    return ""


# Global dict to store prompt -> test mappings (set during training init)
PROMPT_TO_TESTS = {}

# Track reward statistics
REWARD_STATS = {"total": 0, "passed": 0, "partial": 0, "failed": 0, "heuristic": 0, "sum": 0.0}

# Debug: store some examples for logging
DEBUG_EXAMPLES = []
DEBUG_LOG_INTERVAL = 16  # Log detailed examples every N completions

def reward_function(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Reward function for coding tasks with test execution.
    
    Reward structure (following DeepSeek R1 / OpenAI approach):
    - Tests pass: +1.0 (main signal)
    - Code executes but tests fail: +0.2 (partial credit)
    - Syntax error / doesn't run: -0.5
    - Bonus for clean code: up to +0.2
    """
    global PROMPT_TO_TESTS, REWARD_STATS, DEBUG_EXAMPLES
    rewards = []
    
    for completion, prompt in zip(completions, prompts):
        score = 0.0
        status = "heuristic"
        error_msg = ""
        
        # Extract code from response (regex or LLM based on config)
        if CODE_EXTRACTION_METHOD == "llm":
            code = extract_code_with_llm(completion)
        else:
            code = extract_code_from_response(completion)
        
        # Look up test cases for this prompt
        test_code = PROMPT_TO_TESTS.get(prompt, "")
        
        if test_code:
            # Execute and test
            result = safe_execute(code, test_code)
            
            if result["tests_passed"]:
                score = 1.0  # Full reward for passing tests
                status = "PASS"
                REWARD_STATS["passed"] += 1
            elif result["executed"]:
                score = 0.2  # Partial credit for running code
                status = "PARTIAL"
                REWARD_STATS["partial"] += 1
                error_msg = result.get("error", "")
            else:
                score = -0.5  # Penalty for broken code
                status = "FAIL"
                REWARD_STATS["failed"] += 1
                error_msg = result.get("error", "")
        else:
            # Fallback: heuristic scoring when no tests available
            score = heuristic_code_score(code, completion)
            status = "HEURISTIC"
            REWARD_STATS["heuristic"] += 1
        
        # Bonus for code quality
        score += code_quality_bonus(code, completion)
        
        # Clamp to reasonable range
        score = max(-1.0, min(1.5, score))
        rewards.append(score)
        
        REWARD_STATS["total"] += 1
        REWARD_STATS["sum"] += score
        
        # Store debug example
        if len(DEBUG_EXAMPLES) < 100:  # Keep last 100
            DEBUG_EXAMPLES.append({
                "prompt": prompt[:200],
                "completion": completion[:500],
                "code": code[:300],
                "test": test_code[:200] if test_code else "NO_TEST",
                "status": status,
                "score": score,
                "error": error_msg[:100] if error_msg else ""
            })
    
    # Log detailed examples periodically
    if REWARD_STATS["total"] % DEBUG_LOG_INTERVAL == 0 and DEBUG_EXAMPLES:
        print("\n" + "="*80)
        print(f"[DEBUG] Example at total={REWARD_STATS['total']}")
        print("="*80)
        ex = DEBUG_EXAMPLES[-1]
        print(f"PROMPT: {ex['prompt']}...")
        print(f"\nMODEL OUTPUT: {ex['completion'][:300]}...")
        print(f"\nEXTRACTED CODE:\n{ex['code']}")
        print(f"\nTEST CASES: {ex['test']}")
        print(f"\nSTATUS: {ex['status']} | SCORE: {ex['score']:.2f}")
        if ex['error']:
            print(f"ERROR: {ex['error']}")
        print("="*80)
    
    # Log reward stats every 64 completions
    if REWARD_STATS["total"] % 64 == 0:
        total_tested = REWARD_STATS["passed"] + REWARD_STATS["partial"] + REWARD_STATS["failed"]
        pass_rate = REWARD_STATS["passed"] / max(1, total_tested) if total_tested > 0 else 0
        avg = REWARD_STATS["sum"] / max(1, REWARD_STATS["total"])
        print(f"\n[Rewards] Total={REWARD_STATS['total']} | Avg={avg:.3f}")
        print(f"  Tested: Pass={REWARD_STATS['passed']} Partial={REWARD_STATS['partial']} Fail={REWARD_STATS['failed']} | PassRate={pass_rate:.1%}")
        print(f"  Heuristic (no tests): {REWARD_STATS['heuristic']}")
    
    return rewards


def heuristic_code_score(code: str, full_response: str) -> float:
    """Fallback scoring when test cases aren't available."""
    score = 0.0
    
    # Has function definition
    if "def " in code:
        score += 0.3
    
    # Reasonable length
    if 50 < len(code) < 2000:
        score += 0.2
    
    # Has return statement
    if "return " in code:
        score += 0.2
    
    # Try to compile (syntax check)
    try:
        compile(code, "<string>", "exec")
        score += 0.3
    except SyntaxError:
        score -= 0.5
    
    return score


def code_quality_bonus(code: str, full_response: str) -> float:
    """Small bonus for clean, well-formatted code."""
    bonus = 0.0
    
    # Has docstring
    if '"""' in code or "'''" in code:
        bonus += 0.05
    
    # Reasonable line lengths (not one giant line)
    lines = code.split("\n")
    if lines and max(len(l) for l in lines) < 120:
        bonus += 0.05
    
    # Not excessively verbose
    if len(full_response) < 3000:
        bonus += 0.05
    
    # Uses type hints (modern Python)
    if "->" in code or ": str" in code or ": int" in code or ": list" in code:
        bonus += 0.05
    
    return bonus


# ============================================================================
# DATASET PREPARATION - Comprehensive coding data for real learning
# ============================================================================

def extract_function_name(test_code: str) -> str:
    """Extract expected function name from test cases."""
    match = re.search(r'assert\s+(\w+)\s*\(', test_code)
    return match.group(1) if match else None


def load_mbpp():
    """MBPP: ~600 basic Python problems with tests (all splits)."""
    print("  Loading MBPP (all splits)...")
    try:
        # Load all available splits
        all_data = []
        for split in ["train", "test", "validation"]:
            try:
                ds = load_dataset("mbpp", "sanitized", split=split)
                all_data.append(ds)
            except:
                pass
        
        if not all_data:
            raise ValueError("No MBPP splits loaded")
        
        dataset = concatenate_datasets(all_data)
        
        def format_mbpp(example):
            test_code = "\n".join(example["test_list"])
            # Extract the expected function name from tests
            func_name = extract_function_name(test_code)
            
            # CRITICAL: Tell model the EXACT function name to use!
            if func_name:
                prompt = f"""Solve this problem by writing a Python function named `{func_name}`.

Problem: {example['prompt']}

IMPORTANT: 
- The function MUST be named `{func_name}`
- Output ONLY the Python code inside a markdown code block
- No explanations

```python
def {func_name}(...):
    # Your code here
```"""
            else:
                prompt = f"""Solve this problem by writing a Python function.

Problem: {example['prompt']}

IMPORTANT: Output ONLY the Python code inside a markdown code block. No explanations.

```python
# Your code here
```"""
            return {"prompt": prompt, "test_list": test_code, "source": "mbpp", "difficulty": "easy"}
        
        dataset = dataset.map(format_mbpp)
        print(f"    -> {len(dataset)} examples")
        return dataset.select_columns(["prompt", "test_list", "source", "difficulty"])
    except Exception as e:
        print(f"    -> Failed: {e}")
        return None


def load_humaneval():
    """HumanEval: 164 OpenAI coding problems with tests."""
    print("  Loading HumanEval...")
    try:
        dataset = load_dataset("openai/openai_humaneval", split="test")
        
        def format_humaneval(example):
            # HumanEval has the function signature in 'prompt'
            # STRICT FORMAT: Tell model to output ONLY code
            prompt = f"""Complete this Python function. Output ONLY the code, no explanations.

{example['prompt']}

```python
# Complete the function above
```"""
            # HumanEval uses 'test' field
            test_code = example.get("test", "")
            return {"prompt": prompt, "test_list": test_code, "source": "humaneval", "difficulty": "medium"}
        
        dataset = dataset.map(format_humaneval)
        print(f"    -> {len(dataset)} examples")
        return dataset.select_columns(["prompt", "test_list", "source", "difficulty"])
    except Exception as e:
        print(f"    -> Failed: {e}")
        return None


def load_code_alpaca():
    """Code Alpaca: ~20k code instruction-following examples."""
    print("  Loading Code Alpaca...")
    try:
        dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        # Take a subset for balance
        if len(dataset) > 2000:
            dataset = dataset.shuffle(seed=42).select(range(2000))
        
        def format_code_alpaca(example):
            instruction = example.get("instruction", "")
            inp = example.get("input", "")
            
            if inp:
                prompt = f"""{instruction}

Input: {inp}

Write a Python solution."""
            else:
                prompt = f"""{instruction}

Write a Python solution."""
            
            # No tests, use heuristic scoring
            return {"prompt": prompt, "test_list": "", "source": "code_alpaca", "difficulty": "medium"}
        
        dataset = dataset.map(format_code_alpaca)
        print(f"    -> {len(dataset)} examples")
        return dataset.select_columns(["prompt", "test_list", "source", "difficulty"])
    except Exception as e:
        print(f"    -> Failed: {e}")
        return None


def load_python_code_instructions():
    """Python code instructions dataset."""
    print("  Loading Python code instructions...")
    try:
        dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        if len(dataset) > 2000:
            dataset = dataset.shuffle(seed=42).select(range(2000))
        
        def format_python_instructions(example):
            instruction = example.get("instruction", "")
            inp = example.get("input", "")
            
            if inp:
                prompt = f"""{instruction}

Input: {inp}

Write the Python code."""
            else:
                prompt = f"""{instruction}

Write the Python code."""
            
            return {"prompt": prompt, "test_list": "", "source": "python_instructions", "difficulty": "medium"}
        
        dataset = dataset.map(format_python_instructions)
        print(f"    -> {len(dataset)} examples")
        return dataset.select_columns(["prompt", "test_list", "source", "difficulty"])
    except Exception as e:
        print(f"    -> Failed: {e}")
        return None


def load_evol_instruct():
    """EvolInstruct Code: evolved code instructions."""
    print("  Loading EvolInstruct Code...")
    try:
        dataset = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
        # Filter for Python and limit size
        dataset = dataset.filter(lambda x: "python" in x.get("output", "").lower()[:500])
        if len(dataset) > 2000:
            dataset = dataset.shuffle(seed=42).select(range(2000))
        
        def format_evol(example):
            prompt = example.get("instruction", "")
            if not prompt:
                return None
            
            prompt = f"""{prompt}

Write the Python solution."""
            
            return {"prompt": prompt, "test_list": "", "source": "evol_instruct", "difficulty": "medium"}
        
        dataset = dataset.map(format_evol)
        dataset = dataset.filter(lambda x: x is not None and x.get("prompt"))
        print(f"    -> {len(dataset)} examples")
        return dataset.select_columns(["prompt", "test_list", "source", "difficulty"])
    except Exception as e:
        print(f"    -> Failed: {e}")
        return None


def load_synthetic_data():
    """Load any synthetic data generated via create_synthetic_data.py"""
    print("  Checking for synthetic data...")
    synthetic_path = "./data/synthetic_coding.jsonl"
    if not os.path.exists(synthetic_path):
        print("    -> No synthetic data found (optional)")
        return None
    
    try:
        examples = []
        with open(synthetic_path, "r") as f:
            for line in f:
                data = json.loads(line)
                examples.append({
                    "prompt": data.get("prompt", ""),
                    "test_list": data.get("test_cases", ""),
                    "source": "synthetic",
                    "difficulty": data.get("difficulty", "medium"),
                })
        
        if examples:
            dataset = Dataset.from_list(examples)
            print(f"    -> {len(dataset)} synthetic examples")
            return dataset
    except Exception as e:
        print(f"    -> Failed to load: {e}")
    return None


def prepare_comprehensive_dataset():
    """
    Load all available coding datasets for robust RL training.
    Total: ~5000+ diverse examples from multiple sources.
    """
    print("\n" + "=" * 60)
    print("Loading comprehensive coding dataset")
    print("=" * 60)
    
    # IMPORTANT: For RL with code execution rewards, we NEED test cases!
    # Only use datasets that have executable tests
    datasets_to_load = [
        load_mbpp,                    # ~420 basic problems WITH TESTS
        load_humaneval,               # 164 OpenAI problems WITH TESTS  
        # load_code_alpaca,           # NO TESTS - heuristic only (disabled)
        # load_python_code_instructions, # NO TESTS - heuristic only (disabled)
        # load_evol_instruct,         # NO TESTS - heuristic only (disabled)
        load_synthetic_data,          # User-generated (optional)
    ]
    
    all_datasets = []
    for loader in datasets_to_load:
        try:
            ds = loader()
            if ds is not None and len(ds) > 0:
                all_datasets.append(ds)
        except Exception as e:
            print(f"  Warning: {loader.__name__} failed: {e}")
    
    if not all_datasets:
        raise ValueError("No datasets loaded! Check your internet connection.")
    
    # Concatenate all datasets
    combined = concatenate_datasets(all_datasets)
    combined = combined.shuffle(seed=42)
    
    # Limit dataset size if specified
    if MAX_DATASET_SIZE and len(combined) > MAX_DATASET_SIZE:
        print(f"\nLimiting dataset from {len(combined)} to {MAX_DATASET_SIZE} examples")
        combined = combined.select(range(MAX_DATASET_SIZE))
    
    # Print summary
    print("\n" + "-" * 40)
    print("Dataset Summary:")
    sources = {}
    for ex in combined:
        src = ex.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
    print(f"  TOTAL: {len(combined)}")
    print("-" * 40 + "\n")
    
    return combined


def prepare_coding_dataset():
    """Main entry point - loads comprehensive dataset."""
    return prepare_comprehensive_dataset()


def prepare_mixed_dataset():
    """Coding + general tasks for maintaining broad capabilities."""
    print("Loading mixed dataset...")
    
    # Get comprehensive coding data
    coding = prepare_comprehensive_dataset()
    
    # Add some general instruction data
    try:
        general = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
        
        def format_general(example):
            if example.get("input"):
                prompt = f"{example['instruction']}\n\nInput: {example['input']}"
            else:
                prompt = example["instruction"]
            return {
                "prompt": prompt,
                "test_list": "",
                "source": "alpaca",
                "difficulty": "general",
            }
        
        general = general.map(format_general)
        general = general.select_columns(["prompt", "test_list", "source", "difficulty"])
        
        combined = concatenate_datasets([coding, general])
        combined = combined.shuffle(seed=42)
        print(f"Mixed dataset: {len(combined)} total examples")
        return combined
    except Exception as e:
        print(f"Failed to add general data: {e}")
        return coding


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 60)
    print("RL Training POC - GRPO with Code Execution Reward")
    print("Hardware: 2x H200")
    print("=" * 60)
    
    # Load model
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Will use both H200s
        trust_remote_code=True,
        attn_implementation="sdpa",  # PyTorch native scaled dot-product attention (works on H200s)
    )
    
    # Optional: Apply LoRA (not needed with H200s but saves some memory)
    if USE_LORA:
        from peft import LoraConfig, get_peft_model
        print("Applying LoRA...")
        lora_config = LoraConfig(
            r=32,  # Higher rank for better capacity
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load dataset
    # Option 1: Pure coding (recommended for POC)
    dataset = prepare_coding_dataset()
    
    # Option 2: Mixed (uncomment if you want general + coding)
    # dataset = prepare_mixed_dataset()
    
    # GRPO Config - optimized for 2x H200
    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        temperature=0.8,  # Sampling temperature for generations
        report_to="none",  # Set to "wandb" if you have it configured
        bf16=True,
        gradient_checkpointing=True,  # Save memory
        dataloader_num_workers=4,
    )
    
    # Store test cases globally so reward function can access them
    # TRL's GRPO passes prompts to reward function, we match by prompt text
    global PROMPT_TO_TESTS
    PROMPT_TO_TESTS = {ex["prompt"]: ex.get("test_list", "") for ex in dataset}
    
    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function,
    )
    
    print("\n" + "=" * 60)
    print("Starting RL training...")
    print(f"  Steps: {MAX_STEPS}")
    print(f"  Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Generations per prompt: {NUM_GENERATIONS}")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\nDone! Run evaluation with:")
    print(f"  python scripts/evaluate_models.py --models {MODEL_PATH} {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
