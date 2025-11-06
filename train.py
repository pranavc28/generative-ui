import os
import json
import re
import numpy as np
import tinker
from tinker import types
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

DATASET_NAME = "cfahlgren1/react-code-instructions"
NUM_EXAMPLES = 5  # Increased for better generalization -> freshbeer was here
BASE_MODEL = "Qwen/Qwen3-30B-A3B"
LEARNING_RATE = 1e-5
NUM_PPO_EPOCHS = 3
NUM_SAMPLES_PER_PROMPT = 4
MAX_GENERATION_TOKENS = 28000  # Qwen3-30B-A3B native context window (can handle full React components)
PPO_CLIP_EPSILON = 0.2
VALUE_CLIP_EPSILON = 0.2
GAE_LAMBDA = 0.95
ENTROPY_COEFF = 0.01
OUTPUT_DIR = "outputs"
CHECKPOINT_NAME = "react-code-ppo-qwen3-30b-a3b-v2"

# Reward Component Weights (adjust based on priorities)
REWARD_BASE = 1.0               # Base reward for any generation
REWARD_LENGTH_WEIGHT = 0.1      # Weight for length penalty (lower = less important)
REWARD_JSX_BONUS = 0.5          # Bonus for having JSX structure
REWARD_TAILWIND_WEIGHT = 0.4    # Weight for TailwindCSS similarity (0.4 = moderate importance)
REWARD_STRUCTURE_WEIGHT = 1.0   # Weight for JSX structure (1.0 = high importance)
REWARD_VALIDITY_WEIGHT = 2.0    # Weight for code validity (2.0 = very high importance - penalize errors heavily)

def format_react_example(example, idx, tokenizer=None):
    messages = example.get('messages', [])
    
    system_prompt = messages[0]['content'] if len(messages) > 0 else ''
    user_message = messages[1]['content'] if len(messages) > 1 else ''
    assistant_response = messages[2]['content'] if len(messages) > 2 else ''
    
    if tokenizer:
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        full_prompt = tokenizer.apply_chat_template(
            chat_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        full_prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
    
    print(f"\n{'='*70}")
    print(f"EXAMPLE {idx}")
    print(f"{'='*70}")
    print(f"SYSTEM PROMPT:\n{system_prompt}...\n")
    print(f"USER REQUEST:\n{user_message}\n")
    print(f"ASSISTANT RESPONSE:\n{assistant_response}\n")
    print(f"FULL RESPONSE LENGTH: {len(assistant_response)} chars")
    print(f"{'='*70}")
    
    return {
        "system_prompt": system_prompt,
        "user_message": user_message,
        "reference_response": assistant_response,
        "full_prompt": full_prompt
    }

def load_data(tokenizer=None):
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    print(f"Dataset loaded. Total examples: {len(dataset)}")
    print(f"Selecting first {NUM_EXAMPLES} examples\n")
    
    selected = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))
    return [format_react_example(ex, i, tokenizer) for i, ex in enumerate(selected)]

def check_code_validity(code):
    """
    Check for common code errors and return a validity score.
    Returns a score between -1.0 (very invalid) and 0.0 (valid).
    Checks for:
    1. Basic syntax errors (unmatched braces, brackets, parentheses)
    2. Undefined variables (common React/TS patterns)
    3. Missing imports for React
    4. Function/component structure issues
    """
    validity_score = 0.0
    penalties = []
    
    # Check 1: Balanced braces, brackets, and parentheses
    try:
        brace_count = code.count('{') - code.count('}')
        bracket_count = code.count('[') - code.count(']')
        paren_count = code.count('(') - code.count(')')
        
        if abs(brace_count) > 0:
            validity_score -= 0.3
            penalties.append(f"Unmatched braces: {brace_count}")
        if abs(bracket_count) > 0:
            validity_score -= 0.2
            penalties.append(f"Unmatched brackets: {bracket_count}")
        if abs(paren_count) > 0:
            validity_score -= 0.2
            penalties.append(f"Unmatched parentheses: {paren_count}")
    except:
        validity_score -= 0.1
    
    # Check 2: Common undefined variable patterns
    # Look for variables used but not defined (basic heuristic)
    try:
        # Extract variable assignments (const, let, var, function parameters)
        defined_vars = set()
        
        # Find variable declarations
        const_vars = re.findall(r'(?:const|let|var)\s+(\w+)', code)
        defined_vars.update(const_vars)
        
        # Find function declarations
        func_vars = re.findall(r'function\s+(\w+)', code)
        defined_vars.update(func_vars)
        
        # Find arrow function assignments
        arrow_vars = re.findall(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:\(|async)', code)
        defined_vars.update(arrow_vars)
        
        # Find function parameters (simplified)
        params = re.findall(r'(?:function\s+\w+|=>)\s*\(([^)]*)\)', code)
        for param_list in params:
            param_names = re.findall(r'(\w+)(?:\s*:|,|$)', param_list)
            defined_vars.update(param_names)
        
        # Check for common React hooks and variables that should exist
        common_react = {'useState', 'useEffect', 'useCallback', 'useMemo', 'useRef', 'useContext', 
                        'React', 'props', 'children', 'className', 'style', 'key', 'ref'}
        defined_vars.update(common_react)
        
        # Find variable usages (simplified - look for standalone words that are likely variables)
        # This is a heuristic and won't catch everything
        used_vars = re.findall(r'\b([a-z][a-zA-Z0-9]*)\b', code)
        used_vars = set([v for v in used_vars if not v in ['const', 'let', 'var', 'function', 'return', 
                                                             'if', 'else', 'for', 'while', 'switch', 
                                                             'case', 'break', 'continue', 'true', 'false',
                                                             'null', 'undefined', 'this', 'class', 'export',
                                                             'import', 'from', 'default', 'async', 'await',
                                                             'try', 'catch', 'finally', 'throw', 'new',
                                                             'typeof', 'instanceof', 'in', 'of', 'delete']])
        
        # Check for potentially undefined variables
        potentially_undefined = used_vars - defined_vars
        
        # Filter out common valid identifiers
        valid_identifiers = {'console', 'window', 'document', 'Array', 'Object', 'String', 
                           'Number', 'Boolean', 'Math', 'Date', 'JSON', 'Promise',
                           'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
                           'px', 'em', 'rem', 'vh', 'vw'}  # CSS units
        
        potentially_undefined = potentially_undefined - valid_identifiers
        
        # Penalize if there are many undefined variables (more than 5 could be a problem)
        if len(potentially_undefined) > 5:
            validity_score -= 0.3
            penalties.append(f"Potentially undefined variables: {len(potentially_undefined)}")
    except Exception as e:
        validity_score -= 0.05
        penalties.append(f"Variable analysis error: {str(e)}")
    
    # Check 3: Missing React import (for TSX/JSX code)
    if '<' in code and '>' in code:  # Likely JSX
        if 'import' not in code.lower() or 'react' not in code.lower():
            # Missing import statement is less critical in some contexts
            validity_score -= 0.1
            penalties.append("Missing React import")
    
    # Check 4: Component structure (should have at least a return or export)
    has_return = 'return' in code.lower()
    has_export = 'export' in code.lower()
    
    if not has_return and not has_export:
        validity_score -= 0.2
        penalties.append("Missing return or export statement")
    
    # Check 5: Syntax error indicators (unclosed strings, common mistakes)
    try:
        # Count quotes (should be even)
        single_quotes = code.count("'") - code.count("\\'")
        double_quotes = code.count('"') - code.count('\\"')
        backticks = code.count('`')
        
        if single_quotes % 2 != 0:
            validity_score -= 0.2
            penalties.append("Unmatched single quotes")
        if double_quotes % 2 != 0:
            validity_score -= 0.2
            penalties.append("Unmatched double quotes")
        if backticks % 2 != 0:
            validity_score -= 0.2
            penalties.append("Unmatched backticks")
    except:
        validity_score -= 0.05
    
    # Check 6: Common React/TypeScript errors
    # Using useState without destructuring
    if 'useState(' in code and 'const [' not in code and 'const {' not in code:
        # This might indicate incorrect useState usage
        validity_score -= 0.1
        penalties.append("Possible incorrect useState usage")
    
    # Clamp score to -1.0 minimum
    validity_score = max(validity_score, -1.0)
    
    return validity_score, penalties

def compute_code_reward(generated_code, reference_code):
    """
    Compute reward based on four metrics:
    1. JSX presence and length penalty
    2. TailwindCSS class similarity 
    3. JSX structure similarity using simple AST parsing
    4. Code validity (syntax, undefined variables, etc.)
    """
    gen_len = len(generated_code)
    ref_len = len(reference_code)
    
    # Metric 1: JSX presence and length penalty
    has_jsx = "return" in generated_code.lower() and ("<" in generated_code or "/>" in generated_code)
    length_penalty = -abs(gen_len - ref_len) / max(ref_len, 1) * REWARD_LENGTH_WEIGHT
    jsx_bonus = REWARD_JSX_BONUS if has_jsx else 0.0
    
    # Metric 2: TailwindCSS similarity using cosine similarity
    tailwind_reward = 0.0
    try:
        # Extract className attributes from both codes
        gen_classes = re.findall(r'className=["\']([^"\']+)["\']', generated_code)
        ref_classes = re.findall(r'className=["\']([^"\']+)["\']', reference_code)
        
        if gen_classes and ref_classes:
            # Flatten all classes into single strings
            gen_tailwind = ' '.join(gen_classes)
            ref_tailwind = ' '.join(ref_classes)
            
            # Use CountVectorizer for simple token-based similarity
            vectorizer = CountVectorizer()
            try:
                vectors = vectorizer.fit_transform([ref_tailwind, gen_tailwind])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                # Convert similarity (0-1) to reward/penalty scaled by weight
                tailwind_reward = (similarity - 0.5) * REWARD_TAILWIND_WEIGHT * 2
            except:
                tailwind_reward = -0.2 * REWARD_TAILWIND_WEIGHT  # Penalty if vectorization fails
        elif ref_classes and not gen_classes:
            tailwind_reward = -0.5 * REWARD_TAILWIND_WEIGHT  # Penalty for missing TailwindCSS when expected
        elif not ref_classes and not gen_classes:
            tailwind_reward = 0.0  # Neutral if neither has Tailwind
    except:
        tailwind_reward = 0.0
    
    # Metric 3: JSX structure similarity using simple tree-based comparison
    jsx_structure_reward = 0.0
    try:
        # Extract JSX tags (simplified AST approach without tree-sitter)
        gen_tags = re.findall(r'<(\w+)[\s>]', generated_code)
        ref_tags = re.findall(r'<(\w+)[\s>]', reference_code)
        
        if gen_tags and ref_tags:
            # Create tag frequency distributions
            gen_tag_str = ' '.join(gen_tags)
            ref_tag_str = ' '.join(ref_tags)
            
            vectorizer = CountVectorizer()
            try:
                vectors = vectorizer.fit_transform([ref_tag_str, gen_tag_str])
                tag_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                
                # Also check tag count similarity
                tag_count_ratio = min(len(gen_tags), len(ref_tags)) / max(len(gen_tags), len(ref_tags), 1)
                
                # Combined structure reward (70% tag similarity + 30% count ratio)
                structure_score = (tag_similarity * 0.7 + tag_count_ratio * 0.3)
                jsx_structure_reward = (structure_score - 0.5) * REWARD_STRUCTURE_WEIGHT * 2
            except:
                jsx_structure_reward = -0.2 * REWARD_STRUCTURE_WEIGHT
        elif ref_tags and not gen_tags:
            jsx_structure_reward = -0.7 * REWARD_STRUCTURE_WEIGHT  # Strong penalty for missing JSX structure
        elif not ref_tags and not gen_tags:
            jsx_structure_reward = 0.0
    except:
        jsx_structure_reward = 0.0
    
    # Metric 4: Code validity (syntax errors, undefined variables, etc.)
    validity_reward = 0.0
    penalties = []
    try:
        validity_score, penalties = check_code_validity(generated_code)
        # validity_score is between -1.0 (very invalid) and 0.0 (valid)
        # Scale it by the weight - invalid code gets heavily penalized
        validity_reward = validity_score * REWARD_VALIDITY_WEIGHT
    except Exception as e:
        validity_reward = -0.5 * REWARD_VALIDITY_WEIGHT  # Penalty for analysis failure
        penalties.append(f"Validity check error: {str(e)}")
    
    # Combine all rewards
    total_reward = REWARD_BASE + length_penalty + jsx_bonus + tailwind_reward + jsx_structure_reward + validity_reward
    
    return total_reward

def sample_trajectories(sampling_client, tokenizer, prompts):
    trajectories = []
    
    params = types.SamplingParams(
        max_tokens=MAX_GENERATION_TOKENS, 
        temperature=0.7, 
        top_p=0.9,
        stop=[]
    )
    
    for prompt_text in prompts:
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_input = types.ModelInput.from_ints(prompt_tokens)
        
        future = sampling_client.sample(
            prompt=prompt_input, 
            sampling_params=params, 
            num_samples=NUM_SAMPLES_PER_PROMPT
        )
        result = future.result()
        
        for seq in result.sequences:
            generated_tokens = seq.tokens
            if seq.logprobs is None:
                print("WARNING: No logprobs returned from sampling. Using zeros as placeholder.")
                logprobs = [0.0] * len(generated_tokens)
            else:
                logprobs = seq.logprobs
            
            trajectories.append({
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "logprobs": logprobs,
                "prompt_text": prompt_text
            })
    
    return trajectories

def process_trajectories_for_ppo(trajectories, data, tokenizer):
    processed_data = []
    reward_stats = {"total": [], "count": 0, "validity_issues": []}
    
    for traj in trajectories:
        generated_text = tokenizer.decode(traj["generated_tokens"])
        
        ref_response = ""
        for ex in data:
            if ex["full_prompt"] == traj["prompt_text"]:
                ref_response = ex["reference_response"]
                break
        
        reward = compute_code_reward(generated_text, ref_response)
        reward_stats["total"].append(reward)
        reward_stats["count"] += 1
        
        # Check validity and track issues for logging
        try:
            validity_score, penalties = check_code_validity(generated_text)
            if penalties:
                reward_stats["validity_issues"].append({
                    "score": validity_score,
                    "penalties": penalties,
                    "preview": generated_text[:100] + "..."
                })
        except:
            pass
        
        all_tokens = traj["prompt_tokens"] + traj["generated_tokens"]
        target_tokens = all_tokens[1:]
        input_tokens = all_tokens[:-1]

        old_logprobs = [0.0] * len(traj["prompt_tokens"]) + traj["logprobs"]
        old_logprobs = old_logprobs[1:]

        # Apply full reward to each generated token, zero to prompt tokens
        prompt_length = len(traj["prompt_tokens"]) - 1
        advantages = [0.0] * prompt_length + [reward] * len(traj["generated_tokens"])
        
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "logprobs": old_logprobs,
                "advantages": advantages
            }
        )
        processed_data.append(datum)
    
    # Log reward statistics
    if reward_stats["total"]:
        avg_reward = np.mean(reward_stats["total"])
        min_reward = np.min(reward_stats["total"])
        max_reward = np.max(reward_stats["total"])
        print(f"Reward Stats - Avg: {avg_reward:.4f}, Min: {min_reward:.4f}, Max: {max_reward:.4f}")
        
        # Log validity issues
        if reward_stats["validity_issues"]:
            print(f"\n{'='*70}")
            print(f"CODE VALIDITY ISSUES DETECTED: {len(reward_stats['validity_issues'])} samples")
            print(f"{'='*70}")
            for i, issue in enumerate(reward_stats["validity_issues"][:3]):  # Show first 3
                print(f"\nIssue {i+1}:")
                print(f"  Validity Score: {issue['score']:.3f}")
                print(f"  Penalties: {', '.join(issue['penalties'])}")
                print(f"  Code Preview: {issue['preview']}")
            if len(reward_stats["validity_issues"]) > 3:
                print(f"\n  ... and {len(reward_stats['validity_issues']) - 3} more issues")
            print(f"{'='*70}\n")
    
    return processed_data

def train_ppo():
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
    
    tokenizer = training_client.get_tokenizer()

    data = load_data(tokenizer)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PPO TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"PPO Epochs: {NUM_PPO_EPOCHS}")
    print(f"Samples per Prompt: {NUM_SAMPLES_PER_PROMPT}")
    print(f"Clip Epsilon: {PPO_CLIP_EPSILON}")
    print(f"Total Examples: {len(data)}")
    print(f"{'='*70}\n")
    
    for epoch in range(NUM_PPO_EPOCHS):
        print(f"\n{'='*70}")
        print(f"PPO EPOCH {epoch + 1}/{NUM_PPO_EPOCHS}")
        print(f"{'='*70}")
        
        sampling_client = training_client.save_weights_and_get_sampling_client(name=f"temp_epoch_{epoch}")
        
        prompts = [ex["full_prompt"] for ex in data]
        print(f"Sampling {NUM_SAMPLES_PER_PROMPT} trajectories per prompt...")
        trajectories = sample_trajectories(sampling_client, tokenizer, prompts)
        print(f"Generated {len(trajectories)} total trajectories")
        
        print(f"Processing trajectories for PPO update...")
        processed_examples = process_trajectories_for_ppo(trajectories, data, tokenizer)
        
        fwdbwd_future = training_client.forward_backward(processed_examples, "ppo")
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=LEARNING_RATE)
        )
        
        fwdbwd_result = fwdbwd_future.result()
        
        logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
        avg_logprob = np.mean(logprobs)
        
        print(f"Epoch {epoch + 1} - Avg LogProb: {avg_logprob:.4f}")
    
    sampling_client = training_client.save_weights_and_get_sampling_client(name=CHECKPOINT_NAME)
    
    print(f"\nModel saved as {CHECKPOINT_NAME}")
    
    return sampling_client, tokenizer, data

def evaluate(sampling_client, tokenizer, data):
    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}")
    results = []
    
    params = types.SamplingParams(
        max_tokens=MAX_GENERATION_TOKENS, 
        temperature=0.0, 
        stop=[]  # Match training: no stop sequences
    )
    
    for idx, example in enumerate(data):
        prompt_text = example["full_prompt"]
        expected_response = example["reference_response"]
        user_message = example["user_message"]
        
        print(f"\n--- Evaluating Example {idx} ---")
        print(f"User Request: {user_message}...")
        
        prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))
        future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
        result = future.result()
        
        predicted = tokenizer.decode(result.sequences[0].tokens).strip()
        
        has_code = "function" in predicted or "const" in predicted or "return" in predicted
        
        results.append({
            "task_id": f"example_{idx}",
            "user_message": user_message,
            "system_prompt": example["system_prompt"],
            "expected_response": expected_response,
            "predicted_response": predicted,
            "has_code": has_code
        })
        
        print(f"Expected length: {len(expected_response)} chars")
        print(f"Generated length: {len(predicted)} chars")
        print(f"Has code structure: {has_code}")
        print(f"Generated code preview:\n{predicted}...")
    
    with open(f"{OUTPUT_DIR}/eval_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    code_rate = sum(r["has_code"] for r in results) / len(results) if results else 0.0
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total examples: {len(results)}")
    print(f"Examples with code structure: {sum(r['has_code'] for r in results)}")
    print(f"Code generation rate: {code_rate:.2%}")
    print(f"Results saved to: {OUTPUT_DIR}/eval_results.jsonl")
    
    return results

if __name__ == "__main__":
    sampling_client, tokenizer, data = train_ppo()
    evaluate(sampling_client, tokenizer, data)

