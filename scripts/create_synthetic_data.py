"""
OPTIONAL: Generate additional synthetic coding data.

NOTE: You can SKIP THIS entirely! The training script already includes:
  - MBPP (374 problems)
  - HumanEval (164 problems)
  - APPS (2000 problems)
  - CodeContests (1000 problems)
  - code_search_net (2000 problems)
  Total: ~5500 FREE examples with test cases (no API key needed)

Use this script ONLY if you want MORE diverse training data.

BACKENDS:
1. NVIDIA NIM API (recommended)
   - Free tier at https://build.nvidia.com/
   - export NVIDIA_API_KEY="nvapi-xxx"
   - Models: qwen/qwen3-235b-a22b-instruct, meta/llama-3.3-70b-instruct

2. Local vLLM (if serving your SFT model)
   - No API key needed
   - Use your own Nano-Raccoon model to generate more examples
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# ============================================================================
# CODING TASK TEMPLATES
# ============================================================================

CODING_TASKS = [
    # Easy
    {
        "prompt": "Write a Python function `reverse_string(s)` that reverses a string.",
        "test": 'assert reverse_string("hello") == "olleh"\nassert reverse_string("") == ""',
        "difficulty": "easy",
    },
    {
        "prompt": "Write a Python function `is_palindrome(s)` that checks if a string is a palindrome (ignoring case and spaces).",
        "test": 'assert is_palindrome("racecar") == True\nassert is_palindrome("A man a plan a canal Panama".replace(" ", "").lower()) == True',
        "difficulty": "easy",
    },
    {
        "prompt": "Write a Python function `factorial(n)` that returns the factorial of n using recursion.",
        "test": 'assert factorial(5) == 120\nassert factorial(0) == 1\nassert factorial(1) == 1',
        "difficulty": "easy",
    },
    {
        "prompt": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1).",
        "test": 'assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55',
        "difficulty": "easy",
    },
    {
        "prompt": "Write a Python function `find_max(lst)` that finds the maximum element in a list without using built-in max().",
        "test": 'assert find_max([1, 5, 3, 9, 2]) == 9\nassert find_max([-1, -5, -3]) == -1',
        "difficulty": "easy",
    },
    {
        "prompt": "Write a Python function `count_vowels(s)` that counts the number of vowels (a,e,i,o,u) in a string.",
        "test": 'assert count_vowels("hello world") == 3\nassert count_vowels("xyz") == 0',
        "difficulty": "easy",
    },
    {
        "prompt": "Write a Python function `is_prime(n)` that checks if a positive integer is prime.",
        "test": 'assert is_prime(17) == True\nassert is_prime(4) == False\nassert is_prime(2) == True\nassert is_prime(1) == False',
        "difficulty": "easy",
    },
    {
        "prompt": "Write a Python function `sum_digits(n)` that returns the sum of digits of a non-negative integer.",
        "test": 'assert sum_digits(123) == 6\nassert sum_digits(9999) == 36\nassert sum_digits(0) == 0',
        "difficulty": "easy",
    },
    {
        "prompt": "Write a Python function `remove_duplicates(lst)` that removes duplicates from a list while preserving order.",
        "test": 'assert remove_duplicates([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]',
        "difficulty": "easy",
    },
    {
        "prompt": "Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list.",
        "test": 'assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted([], [1, 2]) == [1, 2]',
        "difficulty": "easy",
    },
    # Medium
    {
        "prompt": "Write a Python function `binary_search(arr, target)` that performs binary search and returns the index of target, or -1 if not found.",
        "test": 'assert binary_search([1, 2, 3, 4, 5], 3) == 2\nassert binary_search([1, 2, 3], 4) == -1\nassert binary_search([], 1) == -1',
        "difficulty": "medium",
    },
    {
        "prompt": "Write a Python function `valid_parentheses(s)` that checks if a string of parentheses ()[]{{}} is valid.",
        "test": 'assert valid_parentheses("()[]{}") == True\nassert valid_parentheses("([)]") == False\nassert valid_parentheses("([])") == True',
        "difficulty": "medium",
    },
    {
        "prompt": "Write a Python function `longest_common_prefix(strs)` that finds the longest common prefix among a list of strings.",
        "test": 'assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"\nassert longest_common_prefix(["dog", "cat"]) == ""',
        "difficulty": "medium",
    },
    {
        "prompt": "Write a Python function `two_sum(nums, target)` that returns indices of two numbers that add up to target.",
        "test": 'result = two_sum([2, 7, 11, 15], 9)\nassert set(result) == {0, 1}',
        "difficulty": "medium",
    },
    {
        "prompt": "Write a Python function `flatten(nested_list)` that flattens a nested list of arbitrary depth.",
        "test": 'assert flatten([[1, [2, 3]], [4, [5, [6]]]]) == [1, 2, 3, 4, 5, 6]\nassert flatten([1, 2, 3]) == [1, 2, 3]',
        "difficulty": "medium",
    },
    {
        "prompt": "Write a Python function `anagrams(s1, s2)` that checks if two strings are anagrams of each other.",
        "test": 'assert anagrams("listen", "silent") == True\nassert anagrams("hello", "world") == False',
        "difficulty": "medium",
    },
    {
        "prompt": "Write a Python function `rotate_array(arr, k)` that rotates an array to the right by k steps.",
        "test": 'assert rotate_array([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]',
        "difficulty": "medium",
    },
    {
        "prompt": "Write a Python function `find_duplicates(lst)` that finds all duplicates in a list.",
        "test": 'assert sorted(find_duplicates([1, 2, 3, 2, 1, 4])) == [1, 2]',
        "difficulty": "medium",
    },
]

SYSTEM_PROMPT = """You are an expert Python programmer. Write clean, correct, and efficient Python code.
Always include a docstring explaining what the function does.
Only output the function definition, no explanations before or after."""


# ============================================================================
# NVIDIA NIM API (Recommended)
# ============================================================================

def generate_with_nvidia(
    tasks: list[dict],
    model: str = "qwen/qwen3-235b-a22b-instruct",  # Or other NIM models
    output_file: str = "data/synthetic_nvidia.jsonl",
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    Generate synthetic data using NVIDIA NIM API.
    
    Available models on build.nvidia.com:
    - qwen/qwen3-235b-a22b-instruct (very capable)
    - meta/llama-3.3-70b-instruct
    - mistralai/mistral-large-2-instruct-2411
    - nvidia/llama-3.1-nemotron-70b-instruct
    
    Get API key from: https://build.nvidia.com/
    Set: export NVIDIA_API_KEY=nvapi-xxx
    """
    import requests
    
    api_key = api_key or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY not set. Get one free at https://build.nvidia.com/"
        )
    
    # NVIDIA NIM API endpoint
    base_url = "https://integrate.api.nvidia.com/v1"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    for task in tqdm(tasks, desc=f"Generating with {model}"):
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task["prompt"]},
                ],
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            
            data = response.json()
            completion = data["choices"][0]["message"]["content"]
            
            results.append({
                "prompt": task["prompt"],
                "completion": completion,
                "test": task.get("test", ""),
                "difficulty": task.get("difficulty", "unknown"),
                "model": model,
            })
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Save results
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"Saved {len(results)} examples to {output_file}")
    return results


# ============================================================================
# OpenAI API (Alternative)
# ============================================================================

def generate_with_openai(
    tasks: list[dict],
    model: str = "gpt-4o-mini",
    output_file: str = "data/synthetic_openai.jsonl",
    api_key: Optional[str] = None,
) -> list[dict]:
    """Generate using OpenAI API."""
    import openai
    
    client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    for task in tqdm(tasks, desc=f"Generating with {model}"):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task["prompt"]},
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            
            completion = response.choices[0].message.content
            
            results.append({
                "prompt": task["prompt"],
                "completion": completion,
                "test": task.get("test", ""),
                "difficulty": task.get("difficulty", "unknown"),
                "model": model,
            })
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"Saved {len(results)} examples to {output_file}")
    return results


# ============================================================================
# Local vLLM (For MiniMax-M2 or other local models)
# ============================================================================

def generate_with_vllm_server(
    tasks: list[dict],
    server_url: str = "http://localhost:8000/v1",
    model: str = "MiniMaxAI/MiniMax-M2",  # Model name as served
    output_file: str = "data/synthetic_local.jsonl",
) -> list[dict]:
    """
    Generate using a vLLM server (OpenAI-compatible API).
    
    First start the server:
        vllm serve MiniMaxAI/MiniMax-M2 --tensor-parallel-size 8 --port 8000
    
    Then run this script.
    """
    import requests
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    for task in tqdm(tasks, desc=f"Generating with local {model}"):
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task["prompt"]},
                ],
                "max_tokens": 1024,
                "temperature": 0.7,
            }
            
            response = requests.post(
                f"{server_url}/chat/completions",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            
            data = response.json()
            completion = data["choices"][0]["message"]["content"]
            
            results.append({
                "prompt": task["prompt"],
                "completion": completion,
                "test": task.get("test", ""),
                "difficulty": task.get("difficulty", "unknown"),
                "model": model,
            })
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"Saved {len(results)} examples to {output_file}")
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument(
        "--backend", 
        choices=["nvidia", "openai", "vllm"], 
        default="nvidia",
        help="Which API to use"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Model name (defaults vary by backend)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/synthetic_trajectories.jsonl"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=50,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server URL (for --backend vllm)"
    )
    args = parser.parse_args()
    
    # Prepare tasks
    if args.num_samples <= len(CODING_TASKS):
        tasks = random.sample(CODING_TASKS, args.num_samples)
    else:
        # Repeat tasks
        tasks = (CODING_TASKS * (args.num_samples // len(CODING_TASKS) + 1))[:args.num_samples]
    
    print(f"Generating {len(tasks)} examples using {args.backend}...")
    
    if args.backend == "nvidia":
        model = args.model or "qwen/qwen3-235b-a22b-instruct"
        generate_with_nvidia(tasks, model=model, output_file=args.output)
        
    elif args.backend == "openai":
        model = args.model or "gpt-4o-mini"
        generate_with_openai(tasks, model=model, output_file=args.output)
        
    elif args.backend == "vllm":
        model = args.model or "MiniMaxAI/MiniMax-M2"
        generate_with_vllm_server(
            tasks, 
            server_url=args.vllm_url,
            model=model, 
            output_file=args.output
        )


if __name__ == "__main__":
    main()
