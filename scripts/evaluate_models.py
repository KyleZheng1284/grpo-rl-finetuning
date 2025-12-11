"""
Evaluation script for comparing SFT vs RL models.

Tests on the SAME datasets used for training:
1. MBPP (coding with execution tests) - 420 examples
2. HumanEval (coding with unit tests) - 164 examples

Produces detailed logs showing:
- Each problem
- Model's generated code
- Test cases
- Pass/Fail result
- Error messages if failed
"""

import os
import torch
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============================================================================
# LOGGING SETUP
# ============================================================================

class Logger:
    """Log to both console and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "w")
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


# ============================================================================
# CODE EXECUTION (same as training)
# ============================================================================

def safe_execute(code: str, test_code: str, timeout: float = 5.0) -> dict:
    """Safely execute code and run test cases."""
    import signal
    from io import StringIO
    from contextlib import redirect_stdout, redirect_stderr
    
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Timeout")
    
    result = {"passed": False, "error": None, "executed": False}
    
    full_code = f"{code}\n\n{test_code}"
    
    exec_globals = {
        "__builtins__": __builtins__,
        "print": print,
        "len": len, "range": range, "list": list, "dict": dict,
        "set": set, "str": str, "int": int, "float": float,
        "bool": bool, "sum": sum, "max": max, "min": min,
        "sorted": sorted, "enumerate": enumerate, "zip": zip,
        "map": map, "filter": filter, "abs": abs, "all": all,
        "any": any, "reversed": reversed, "tuple": tuple,
    }
    
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                exec(full_code, exec_globals)
            result["passed"] = True
            result["executed"] = True
        except AssertionError as e:
            result["executed"] = True
            result["error"] = f"AssertionError: {e}"
        except TimeoutError:
            result["error"] = "Timeout (>5s)"
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except Exception as e:
        result["error"] = f"Setup error: {e}"
    
    return result


def extract_code(response: str) -> str:
    """Extract Python code from response (same logic as training)."""
    # Method 1: Find code block
    code_pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(code_pattern, response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    
    # Method 2: Find function definition
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
            
            if stripped.startswith('def ') and not in_function:
                in_function = True
                base_indent = len(line) - len(stripped)
                code_lines.append(line)
                continue
            
            if in_function:
                if not stripped:
                    code_lines.append('')
                    continue
                
                current_indent = len(line) - len(stripped)
                
                if current_indent > base_indent or stripped.startswith('#'):
                    if any(stripped.startswith(x) for x in ['Input', 'Output', 'Example', 'Test', 'Wait', 'Let', 'Yes', 'No', 'The ', 'So ', 'Okay']):
                        break
                    code_lines.append(line)
                else:
                    if stripped.startswith('def '):
                        code_lines.append(line)
                        base_indent = current_indent
                    else:
                        break
        
        if code_lines:
            return '\n'.join(code_lines).strip()
    
    # Method 3: Last resort
    simple_match = re.search(r'(def\s+\w+[^\n]*\n(?:[ \t]+[^\n]*\n?)*)', response)
    if simple_match:
        return simple_match.group(1).strip()
    
    return ""


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path: str):
    """Load model and tokenizer."""
    print(f"\nLoading model: {model_path}")
    print("-" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_tokens: int = 1024) -> str:
    """Generate a response."""
    messages = [{"role": "user", "content": prompt}]
    
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except:
        text = f"User: {prompt}\nAssistant:"
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    return response


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def extract_function_name_from_tests(test_code: str) -> str:
    """Extract the expected function name from test cases."""
    # Handle: assert math.isclose(function_name(...), ...)
    match = re.search(r'assert\s+math\.isclose\s*\(\s*(\w+)\s*\(', test_code)
    if match:
        return match.group(1)
    
    # Handle: assert function_name(...) == ...
    match = re.search(r'assert\s+(\w+)\s*\(', test_code)
    if match and match.group(1) not in ['len', 'str', 'int', 'float', 'list', 'set', 'dict', 'tuple', 'sorted', 'sum', 'max', 'min', 'abs', 'all', 'any', 'round']:
        return match.group(1)
    
    # Handle: assert (function_name(...)) == ...
    match = re.search(r'assert\s+\(\s*(\w+)\s*\(', test_code)
    if match:
        return match.group(1)
    
    return None


def extract_function_name_from_code(code: str) -> str:
    """Extract the function name from generated code."""
    match = re.search(r'def\s+(\w+)\s*\(', code)
    if match:
        return match.group(1)
    return None


def rename_function_in_code(code: str, old_name: str, new_name: str) -> str:
    """Rename a function in the code to match expected name."""
    if not old_name or not new_name or old_name == new_name:
        return code
    # Replace function definition
    code = re.sub(rf'\bdef\s+{old_name}\s*\(', f'def {new_name}(', code)
    # Replace recursive calls to the function
    code = re.sub(rf'\b{old_name}\s*\(', f'{new_name}(', code)
    return code


def eval_mbpp(model, tokenizer, verbose: bool = True) -> dict:
    """Evaluate on MBPP - same dataset used in training."""
    print("\n" + "=" * 80)
    print("MBPP EVALUATION (Same dataset used in RL training)")
    print("=" * 80)
    
    # Load all splits like we did in training
    all_data = []
    for split in ["train", "test", "validation"]:
        try:
            ds = load_dataset("mbpp", "sanitized", split=split)
            all_data.extend(list(ds))
        except:
            pass
    
    print(f"Total MBPP problems: {len(all_data)}")
    
    passed = 0
    executed = 0
    total = 0
    results = []
    
    for i, example in enumerate(tqdm(all_data, desc="MBPP")):
        test_code = "\n".join(example["test_list"])
        
        # Extract expected function name from tests FIRST
        expected_func_name = extract_function_name_from_tests(test_code)
        
        # Get the first test case as an example
        first_test = example["test_list"][0] if example["test_list"] else ""
        
        if expected_func_name:
            prompt = f"""Write a Python function named `{expected_func_name}` to solve this problem.

Problem: {example['prompt']}

Example test (your code will be tested with assertions like this):
{first_test}

HOW YOUR CODE WILL BE EVALUATED:
- Your code will be executed using Python's exec()
- Then the test assertions will run against your function
- If any assertion fails or throws an error, you fail

REQUIREMENTS:
1. Name your function EXACTLY `{expected_func_name}` (must match the test)
2. Your code must be SELF-CONTAINED (define all helper functions you use)
3. You may import standard libraries (math, re, collections, etc.) if needed
4. Output ONLY executable Python code in a markdown block - NO explanations

```python
def {expected_func_name}(...):
    ...
```"""
        else:
            prompt = f"""Write a Python function to solve this problem.

Problem: {example['prompt']}

HOW YOUR CODE WILL BE EVALUATED:
- Your code will be executed using Python's exec()
- Then test assertions will run against your function
- If any assertion fails or throws an error, you fail

REQUIREMENTS:
1. Your code must be SELF-CONTAINED (define all helper functions you use)
2. You may import standard libraries if needed
3. Output ONLY executable Python code in a markdown block - NO explanations

```python
def solution(...):
    ...
```"""
        
        response = generate(model, tokenizer, prompt)
        code = extract_code(response)
        
        # POST-PROCESS: Rename function if still mismatched
        generated_func_name = extract_function_name_from_code(code)
        
        original_code = code  # Save for logging
        if expected_func_name and generated_func_name and expected_func_name != generated_func_name:
            code = rename_function_in_code(code, generated_func_name, expected_func_name)
        
        result = safe_execute(code, test_code)
        
        if result["passed"]:
            passed += 1
        if result["executed"]:
            executed += 1
        total += 1
        
        # Detailed logging
        if verbose and (i < 10 or not result["passed"]):  # Log first 10 and all failures
            print(f"\n{'='*80}")
            print(f"MBPP #{i+1} | {'PASS' if result['passed'] else 'FAIL'}")
            print(f"{'='*80}")
            print(f"PROBLEM: {example['prompt'][:200]}...")
            if generated_func_name and expected_func_name and generated_func_name != expected_func_name:
                print(f"\nFUNCTION RENAME: {generated_func_name} -> {expected_func_name}")
            print(f"\nGENERATED CODE (after rename):")
            print(f"{code[:500]}" if code else "[NO CODE EXTRACTED]")
            print(f"\nTEST CASES:")
            print(f"{test_code[:300]}...")
            if result["error"]:
                print(f"\nERROR: {result['error']}")
            print("-" * 80)
        
        results.append({
            "id": i,
            "prompt": example["prompt"],
            "code": code[:1000],
            "tests": test_code[:500],
            "passed": result["passed"],
            "executed": result["executed"],
            "error": result.get("error"),
        })
    
    accuracy = passed / total if total > 0 else 0
    exec_rate = executed / total if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"MBPP RESULTS")
    print(f"{'='*80}")
    print(f"Pass Rate:      {passed}/{total} = {accuracy:.1%}")
    print(f"Execution Rate: {executed}/{total} = {exec_rate:.1%}")
    print(f"{'='*80}")
    
    return {
        "benchmark": "MBPP",
        "passed": passed,
        "executed": executed,
        "total": total,
        "accuracy": accuracy,
        "execution_rate": exec_rate,
        "results": results,
    }


def eval_humaneval(model, tokenizer, verbose: bool = True) -> dict:
    """Evaluate on HumanEval - same dataset used in training."""
    print("\n" + "=" * 80)
    print("HUMANEVAL EVALUATION (Same dataset used in RL training)")
    print("=" * 80)
    
    dataset = load_dataset("openai/openai_humaneval", split="test")
    print(f"Total HumanEval problems: {len(dataset)}")
    
    passed = 0
    executed = 0
    total = 0
    results = []
    
    for i, example in enumerate(tqdm(dataset, desc="HumanEval")):
        # Use SAME prompt format as training
        prompt = f"""Complete this Python function. Output ONLY the code, no explanations.

{example['prompt']}

```python
# Complete the function above
```"""
        
        response = generate(model, tokenizer, prompt)
        code = extract_code(response)
        
        # HumanEval test format
        test_code = example.get("test", "")
        
        # Need to combine function + tests properly for HumanEval
        full_code = code
        if test_code:
            result = safe_execute(full_code, test_code)
        else:
            result = {"passed": False, "error": "No test cases", "executed": False}
        
        if result["passed"]:
            passed += 1
        if result.get("executed"):
            executed += 1
        total += 1
        
        # Detailed logging
        if verbose and (i < 5 or not result["passed"]):
            print(f"\n{'='*80}")
            print(f"HumanEval #{i+1} | {'PASS' if result['passed'] else 'FAIL'}")
            print(f"{'='*80}")
            print(f"PROMPT: {example['prompt'][:300]}...")
            print(f"\nGENERATED CODE:")
            print(f"{code[:500]}" if code else "[NO CODE EXTRACTED]")
            if result["error"]:
                print(f"\nERROR: {result['error']}")
            print("-" * 80)
        
        results.append({
            "id": example.get("task_id", i),
            "prompt": example["prompt"][:500],
            "code": code[:1000],
            "passed": result["passed"],
            "executed": result.get("executed", False),
            "error": result.get("error"),
        })
    
    accuracy = passed / total if total > 0 else 0
    exec_rate = executed / total if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"HUMANEVAL RESULTS")
    print(f"{'='*80}")
    print(f"Pass Rate:      {passed}/{total} = {accuracy:.1%}")
    print(f"Execution Rate: {executed}/{total} = {exec_rate:.1%}")
    print(f"{'='*80}")
    
    return {
        "benchmark": "HumanEval",
        "passed": passed,
        "executed": executed,
        "total": total,
        "accuracy": accuracy,
        "execution_rate": exec_rate,
        "results": results,
    }


def eval_cruxeval(model, tokenizer, num_samples: int = 250, verbose: bool = True) -> dict:
    """Evaluate on CRUXEval - code execution reasoning from Meta."""
    print("\n" + "=" * 80)
    print("CRUXEVAL EVALUATION (Code Execution Reasoning - Meta)")
    print("=" * 80)
    
    try:
        # CRUXEval - code execution reasoning
        dataset = load_dataset("cruxeval-org/cruxeval", split="test")
        print(f"  Loaded CRUXEval: {len(dataset)} problems")
        
        # Randomly sample
        import random
        random.seed(42)
        all_data = list(dataset)
        if num_samples < len(all_data):
            all_data = random.sample(all_data, num_samples)
        
        print(f"Total CRUXEval problems (randomly sampled): {len(all_data)}")
    except Exception as e:
        print(f"Failed to load CRUXEval: {e}")
        return {"benchmark": "CRUXEval", "error": str(e)}
    
    passed = 0
    executed = 0
    total = 0
    results = []
    
    for i, example in enumerate(tqdm(all_data, desc="CRUXEval")):
        # CRUXEval format: code, input, output
        code_snippet = example.get("code", "")
        test_input = example.get("input", "")
        expected_output = example.get("output", "")
        
        if not code_snippet or not test_input:
            continue
        
        prompt = f"""What is the exact output of this Python code when executed?

```python
{code_snippet}

result = f({test_input})
print(result)
```

IMPORTANT: Reply with ONLY the raw output value. Do NOT explain, do NOT use <think> tags, do NOT reason. Just the value.
Example format: aph?d
Example format: ['a', 'b']
Example format: {{'key': 1}}"""
        
        response = generate(model, tokenizer, prompt, max_tokens=256)
        
        # Extract the predicted output - handle <think>...</think> reasoning tags
        predicted = ""
        
        # First, strip out <think>...</think> reasoning blocks
        clean_response = response
        if '<think>' in clean_response:
            # Remove everything between <think> and </think> (including tags)
            import re
            clean_response = re.sub(r'<think>.*?</think>', '', clean_response, flags=re.DOTALL)
            # If no </think> found, remove from <think> onwards
            if '<think>' in clean_response:
                clean_response = clean_response.split('<think>')[0]
        
        # Now extract the actual answer from cleaned response
        lines = [l.strip() for l in clean_response.strip().split("\n") if l.strip()]
        if lines:
            # Skip lines that start with thinking patterns
            for line in lines:
                if not any(line.lower().startswith(x) for x in ['wait', 'let', 'so ', 'the ', 'first', 'i ', 'okay', 'hmm', 'output:']):
                    predicted = line.strip('`"\' ')
                    break
            # Check for "OUTPUT: value" format
            if not predicted:
                for line in lines:
                    if line.lower().startswith('output:'):
                        predicted = line.split(':', 1)[1].strip().strip('`"\' ')
                        break
            if not predicted and lines:
                predicted = lines[-1].strip('`"\' ')  # Take last line as fallback
        
        # Clean up predictions
        predicted = predicted.strip('`"\' ')
        expected_clean = str(expected_output).strip('`"\' ')
        
        test_passed = False
        error_msg = None
        
        # Check if prediction matches expected
        if predicted == expected_clean:
            test_passed = True
            passed += 1
        else:
            # Try numeric comparison for floats
            try:
                if abs(float(predicted) - float(expected_clean)) < 1e-6:
                    test_passed = True
                    passed += 1
            except:
                pass
            
            if not test_passed:
                error_msg = f"Expected '{expected_clean[:50]}', got '{predicted[:50]}'"
        
        executed += 1
        total += 1
        
        if verbose and (i < 5 or not test_passed):
            print(f"\n{'='*80}")
            print(f"CRUXEval #{i+1} | {'PASS' if test_passed else 'FAIL'}")
            print(f"{'='*80}")
            print(f"CODE:\n{code_snippet[:300]}...")
            print(f"\nINPUT: {test_input}")
            print(f"EXPECTED: {expected_clean}")
            print(f"PREDICTED: {predicted}")
            if error_msg:
                print(f"\nERROR: {error_msg}")
            print("-" * 80)
        
        results.append({
            "id": i,
            "passed": test_passed,
            "predicted": predicted[:100],
            "expected": expected_clean[:100],
            "error": error_msg,
        })
    
    accuracy = passed / total if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"CRUXEVAL RESULTS")
    print(f"{'='*80}")
    print(f"Pass Rate: {passed}/{total} = {accuracy:.1%}")
    print(f"{'='*80}")
    
    return {
        "benchmark": "CRUXEval",
        "passed": passed,
        "executed": executed,
        "total": total,
        "accuracy": accuracy,
        "results": results,
    }


def eval_bigcodebench(model, tokenizer, num_samples: int = 250, verbose: bool = True) -> dict:
    """Evaluate on BigCodeBench - diverse coding tasks."""
    print("\n" + "=" * 80)
    print("BIGCODEBENCH EVALUATION (Diverse Coding Tasks)")
    print("=" * 80)
    
    try:
        # Load BigCodeBench
        dataset = load_dataset("bigcode/bigcodebench", split="v0.1.2")
        print(f"  Loaded BigCodeBench: {len(dataset)} problems")
        
        # Randomly sample
        import random
        random.seed(42)
        all_data = list(dataset)
        if num_samples < len(all_data):
            all_data = random.sample(all_data, num_samples)
        
        print(f"Total BigCodeBench problems (randomly sampled): {len(all_data)}")
    except Exception as e:
        print(f"Failed to load BigCodeBench: {e}")
        return {"benchmark": "BigCodeBench", "error": str(e)}
    
    passed = 0
    executed = 0
    total = 0
    results = []
    
    for i, example in enumerate(tqdm(all_data, desc="BigCodeBench")):
        # BigCodeBench has 'instruct_prompt' and 'test' fields
        problem = example.get("instruct_prompt", example.get("complete_prompt", ""))
        test_code = example.get("test", "")
        
        if not problem:
            continue
        
        # Extract function name from test if possible
        func_name = extract_function_name_from_tests(test_code) if test_code else None
        
        if func_name:
            prompt = f"""Write a Python function named `{func_name}` to solve this task.

Task:
{problem[:1500]}

HOW YOUR CODE WILL BE EVALUATED:
- Your code will be executed using Python's exec()
- Then test assertions will run against your function
- If any assertion fails or throws an error, you fail

REQUIREMENTS:
1. Name your function EXACTLY `{func_name}`
2. Your code must be SELF-CONTAINED (define all helper functions)
3. You may import standard libraries if needed
4. Output ONLY executable Python code in a markdown block

```python
def {func_name}(...):
    ...
```"""
        else:
            prompt = f"""Write Python code to solve this task.

Task:
{problem[:1500]}

REQUIREMENTS:
1. Your code must be SELF-CONTAINED
2. You may import standard libraries if needed
3. Output ONLY executable Python code in a markdown block

```python
# Your solution
```"""
        
        response = generate(model, tokenizer, prompt, max_tokens=1024)
        code = extract_code(response)
        
        # Test the code
        if code and test_code:
            # Rename function if needed
            generated_func_name = extract_function_name_from_code(code)
            if func_name and generated_func_name and func_name != generated_func_name:
                code = rename_function_in_code(code, generated_func_name, func_name)
            
            result = safe_execute(code, test_code, timeout=10.0)
        else:
            result = {"passed": False, "error": "No code or test", "executed": False}
        
        if result["passed"]:
            passed += 1
        if result.get("executed"):
            executed += 1
        total += 1
        
        if verbose and (i < 5 or not result["passed"]):
            print(f"\n{'='*80}")
            print(f"BigCodeBench #{i+1} | {'PASS' if result['passed'] else 'FAIL'}")
            print(f"{'='*80}")
            print(f"TASK: {problem[:200]}...")
            print(f"\nCODE: {code[:400] if code else '[NO CODE]'}...")
            if result.get("error"):
                print(f"\nERROR: {result['error']}")
            print("-" * 80)
        
        results.append({
            "id": i,
            "passed": result["passed"],
            "executed": result.get("executed", False),
            "error": result.get("error"),
        })
    
    accuracy = passed / total if total > 0 else 0
    exec_rate = executed / total if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"BIGCODEBENCH RESULTS")
    print(f"{'='*80}")
    print(f"Pass Rate:      {passed}/{total} = {accuracy:.1%}")
    print(f"Execution Rate: {executed}/{total} = {exec_rate:.1%}")
    print(f"{'='*80}")
    
    return {
        "benchmark": "BigCodeBench",
        "passed": passed,
        "executed": executed,
        "total": total,
        "accuracy": accuracy,
        "execution_rate": exec_rate,
        "results": results,
    }


# ============================================================================
# COMPARISON
# ============================================================================

def compare_models(model_paths: list, output_dir: str = "./eval_results"):
    """Compare multiple models with detailed logging."""
    
    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{output_dir}/eval_log_{timestamp}.txt"
    
    print(f"\n{'#'*80}")
    print(f"# MODEL EVALUATION")
    print(f"# Log file: {log_file}")
    print(f"# Models: {model_paths}")
    print(f"{'#'*80}\n")
    
    # Redirect stdout to also write to log file
    sys.stdout = Logger(log_file)
    
    all_results = {}
    
    for path in model_paths:
        print(f"\n{'#'*80}")
        print(f"# EVALUATING: {path}")
        print(f"{'#'*80}")
        
        try:
            model, tokenizer = load_model(path)
            
            # Only run CRUXEval - code execution reasoning
            results = {
                "model": path,
                "cruxeval": eval_cruxeval(model, tokenizer, num_samples=250, verbose=True),
            }
            
            all_results[path] = results
            
            # Free memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR loading {path}: {e}")
            all_results[path] = {"error": str(e)}
    
    # Final Summary
    print("\n" + "#" * 80)
    print("# FINAL COMPARISON SUMMARY")
    print("#" * 80)
    print(f"\n{'Model':<40} {'CRUXEval (250)':<18}")
    print("-" * 60)
    
    for path, results in all_results.items():
        if "error" in results:
            print(f"{Path(path).name:<40} {'ERROR':<18}")
        else:
            name = Path(path).name[:38]
            crux = f"{results['cruxeval']['accuracy']:.1%}" if 'cruxeval' in results and 'accuracy' in results['cruxeval'] else "N/A"
            print(f"{name:<40} {crux:<18}")
    
    print("-" * 80)
    
    # Save detailed results to JSON
    json_file = f"{output_dir}/results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {json_file}")
    print(f"Full log saved to: {log_file}")
    
    return all_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["billxbf/Nano-Raccoon-Preview-1104"],
        help="Model paths to evaluate"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./eval_results",
        help="Directory to save results"
    )
    args = parser.parse_args()
    
    compare_models(args.models, args.output_dir)
