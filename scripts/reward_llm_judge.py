"""
LLM-as-Judge Reward Function

Use a smaller/faster model to score completions.
This is more realistic than rule-based rewards but still cheap.

Options:
1. Local small model (Phi-3, Llama-3-8B)
2. API call (OpenAI, Anthropic) - more expensive but better
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

# Global judge model (loaded once)
_judge_model = None
_judge_tokenizer = None


JUDGE_PROMPT = """You are evaluating an AI assistant's response. Score from 0-10.

Criteria:
- Correctness: Is the answer factually correct?
- Helpfulness: Does it address the user's question?
- Clarity: Is it well-structured and easy to understand?
- Conciseness: Is it appropriately detailed without being verbose?

User prompt: {prompt}

Assistant response: {response}

Provide ONLY a single number 0-10 as your score, nothing else."""


def load_judge_model(model_name: str = "microsoft/phi-3-mini-4k-instruct"):
    """Load a small model as judge. Phi-3 is good balance of speed/quality."""
    global _judge_model, _judge_tokenizer
    
    if _judge_model is None:
        print(f"Loading judge model: {model_name}")
        _judge_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _judge_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        _judge_model.eval()
    
    return _judge_model, _judge_tokenizer


def score_with_judge(
    prompt: str, 
    response: str,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> float:
    """Score a single response using the judge model."""
    
    if model is None or tokenizer is None:
        model, tokenizer = load_judge_model()
    
    judge_input = JUDGE_PROMPT.format(prompt=prompt, response=response)
    
    inputs = tokenizer(judge_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Parse score
    try:
        score = float(generated.strip().split()[0])
        score = max(0, min(10, score))  # Clamp to 0-10
        return score / 10.0  # Normalize to 0-1
    except (ValueError, IndexError):
        return 0.5  # Default if parsing fails


def reward_function_llm_judge(completions: list[str], prompts: list[str]) -> list[float]:
    """
    Batch reward function using LLM judge.
    Drop-in replacement for the rule-based reward.
    """
    model, tokenizer = load_judge_model()
    
    rewards = []
    for completion, prompt in zip(completions, prompts):
        score = score_with_judge(prompt, completion, model, tokenizer)
        rewards.append(score)
    
    return rewards


# ============================================================================
# API-based judge (if you have API access)
# ============================================================================

def reward_function_api_judge(
    completions: list[str], 
    prompts: list[str],
    model: str = "gpt-4o-mini"  # Cheap and fast
) -> list[float]:
    """
    Use OpenAI API as judge. More expensive but better quality.
    Set OPENAI_API_KEY environment variable.
    """
    import openai
    import os
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    rewards = []
    for completion, prompt in zip(completions, prompts):
        judge_prompt = JUDGE_PROMPT.format(prompt=prompt, response=completion)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=5,
                temperature=0,
            )
            score_text = response.choices[0].message.content.strip()
            score = float(score_text.split()[0])
            score = max(0, min(10, score)) / 10.0
        except Exception as e:
            print(f"API error: {e}")
            score = 0.5
            
        rewards.append(score)
    
    return rewards

