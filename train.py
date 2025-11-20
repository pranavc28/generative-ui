"""
GRPO (Group Relative Policy Optimization) Training with Async Sampling (Tinker API)

GRPO samples multiple responses per prompt and uses group-relative advantages:
1. Sample N responses for each prompt (NUM_SAMPLES_PER_PROMPT)
2. Compute rewards for all responses
3. Compute advantages as: reward - group_mean (mean-centered within each group)
4. This provides a self-baseline: good responses get positive advantages,
   bad responses get negative advantages, all relative to their group

Async implementation:
- sample_async() launches all sampling requests asynchronously
- forward_backward_async() and optim_step_async() overlap computation
- Evaluation also uses async sampling for faster inference

This provides better GPU utilization and faster training iterations.
"""
import os
import json
import re
import asyncio
import time
import numpy as np
import tinker
from tinker import types
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

DATASET_NAME = "cfahlgren1/react-code-instructions"
NUM_EXAMPLES = 600  # INCREASED: 20 was too small, need more diverse training data
NUM_EVAL_EXAMPLES = 10  # Evaluate on subset for faster feedback (600 would take too long)
MAX_PROMPT_TOKENS = 16000  # Filter: Keep prompts under 16k tokens (leave 16k for generation)
BASE_MODEL = "Qwen/Qwen3-30B-A3B"
LEARNING_RATE = 1e-5
NUM_GRPO_EPOCHS = 5  # INCREASED: More epochs for better convergence
NUM_SAMPLES_PER_PROMPT = 4  # GRPO: Multiple samples per prompt for group-relative advantages
MAX_GENERATION_TOKENS = 16000  # Reduced from 20k: Model has 32k context, need room for long prompts
GENERATION_STOP_SEQUENCES = ["</code>", "```\n\n", "\n\n\n\n"]  # Stop sequences to detect completion
GRPO_CLIP_EPSILON = 0.2  # Clip ratio for policy gradient
ENTROPY_COEFF = 0.01
OUTPUT_DIR = "outputs"
CHECKPOINT_NAME = "react-code-grpo-qwen3-30b-a3b-v1"

REWARD_BASE = 1.0
REWARD_COMPLETENESS_WEIGHT = 15.0
REWARD_VALIDITY_WEIGHT = 6.0
REWARD_QUOTE_WEIGHT = 4.0
REWARD_DYNAMIC_REACT_WEIGHT = 5.0
REWARD_LENGTH_PENALTY_WEIGHT = 0.1

def format_react_example(example, idx, tokenizer=None):
    messages = example.get('messages', [])
    
    system_prompt = messages[0]['content'] if len(messages) > 0 else ''
    user_message = messages[1]['content'] if len(messages) > 1 else ''
    assistant_response = messages[2]['content'] if len(messages) > 2 else ''
    
    code_correctness_instructions = """

CRITICAL CODE QUALITY REQUIREMENTS:
- Ensure ALL variables are properly declared before use (const, let, var)
- Initialize ALL state variables and hooks correctly
- Match ALL opening and closing braces, brackets, and parentheses
- Use proper React hooks syntax (e.g., const [state, setState] = useState(initialValue))
- Ensure all function parameters are defined
- Import React if using JSX
- Close all strings, template literals, and JSX tags properly
- Define all functions before calling them
- Use proper TypeScript types and interfaces
- Export the default component correctly

BEFORE RESPONDING, VERIFY:
✓ All braces, brackets, and parentheses are balanced
✓ All JSX tags are properly closed
✓ All variables are defined before use
✓ The component is complete with proper export
✓ No syntax errors exist

Generate COMPLETE, SYNTACTICALLY CORRECT, and FULLY FUNCTIONAL code. Do not use undefined variables or leave any code incomplete. ALWAYS close all code blocks properly."""
    
    enhanced_system_prompt = system_prompt + code_correctness_instructions
    
    if tokenizer:
        chat_messages = [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": user_message}
        ]
        full_prompt = tokenizer.apply_chat_template(
            chat_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        full_prompt = f"{enhanced_system_prompt}\n\nUser: {user_message}\n\nAssistant:"
    
    return {
        "system_prompt": enhanced_system_prompt,
        "user_message": user_message,
        "reference_response": assistant_response,
        "full_prompt": full_prompt
    }

def load_data(tokenizer=None):
    dataset = load_dataset(DATASET_NAME, split="train")
    formatted_examples = []
    skipped = 0
    
    for i, ex in enumerate(dataset):
        formatted = format_react_example(ex, i, tokenizer)
        
        if tokenizer:
            prompt_length = len(tokenizer.encode(formatted["full_prompt"]))
            if prompt_length > MAX_PROMPT_TOKENS:
                skipped += 1
                continue
        
        formatted_examples.append(formatted)
        if len(formatted_examples) >= NUM_EXAMPLES:
            break
    
    print(f"Loaded {len(formatted_examples)} examples (skipped {skipped} with prompts > {MAX_PROMPT_TOKENS} tokens)")
    return formatted_examples

def extract_valid_identifiers_from_reference(reference_code):
    valid_ids = set()
    
    try:
        import_patterns = [
            r'import\s+(\w+)\s+from',
            r'import\s+\*\s+as\s+(\w+)\s+from',
            r'import\s+{([^}]+)}\s+from',
            r'import\s+(\w+)\s*,\s*{([^}]+)}\s+from',
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, reference_code)
            for match in matches:
                if isinstance(match, tuple):
                    for item in match:
                        # Split by comma for named imports
                        for name in item.split(','):
                            # Handle 'as' aliases
                            if ' as ' in name:
                                name = name.split(' as ')[-1]
                            clean_name = name.strip()
                            if clean_name:
                                valid_ids.add(clean_name)
                else:
                    valid_ids.add(match.strip())
        
        type_imports = re.findall(r'import\s+type\s+{([^}]+)}\s+from', reference_code)
        for imports_str in type_imports:
            for name in imports_str.split(','):
                if ' as ' in name:
                    name = name.split(' as ')[-1]
                clean_name = name.strip()
                if clean_name:
                    valid_ids.add(clean_name)
        
        interfaces = re.findall(r'(?:interface|type)\s+(\w+)', reference_code)
        valid_ids.update(interfaces)
        
        const_declarations = re.findall(r'(?:const|let|var)\s+(\w+)', reference_code)
        valid_ids.update(const_declarations)
        
    except Exception as e:
        print(f"Warning: Error extracting identifiers from reference: {e}")
    
    return valid_ids

def check_code_validity(code, reference_code=None):
    validity_score = 0.0
    penalties = []
    
    is_truncated = False
    truncation_indicators = [
        code.count('{') > code.count('}'),
        code.count('[') > code.count(']'),
        code.rstrip().endswith((',', '(', '[', '{', '<')),
        not code.rstrip().endswith(('}', ';', '>', ')', '`', '"', "'")),
    ]
    
    if sum(truncation_indicators) >= 2:
        is_truncated = True
        validity_score -= 1.5
        penalties.append("Code is truncated/incomplete - CRITICAL ERROR")
    
    reference_identifiers = set()
    if reference_code:
        reference_identifiers = extract_valid_identifiers_from_reference(reference_code)
    
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
    
    try:
        defined_vars = set()
        const_vars = re.findall(r'(?:const|let|var)\s+(\w+)', code)
        defined_vars.update(const_vars)
        
        func_vars = re.findall(r'function\s+(\w+)', code)
        defined_vars.update(func_vars)
        
        arrow_vars = re.findall(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:\(|async)', code)
        defined_vars.update(arrow_vars)
        
        params = re.findall(r'(?:function\s+\w+|=>)\s*\(([^)]*)\)', code)
        for param_list in params:
            param_names = re.findall(r'(\w+)(?:\s*:|,|$)', param_list)
            defined_vars.update(param_names)
        
        common_react = {'useState', 'useEffect', 'useCallback', 'useMemo', 'useRef', 'useContext', 
                        'React', 'props', 'children', 'className', 'style', 'key', 'ref'}
        defined_vars.update(common_react)
        defined_vars.update(reference_identifiers)
        
        used_vars = re.findall(r'\b([a-z][a-zA-Z0-9]*)\b', code)
        used_vars = set([v for v in used_vars if not v in ['const', 'let', 'var', 'function', 'return', 
                                                             'if', 'else', 'for', 'while', 'switch', 
                                                             'case', 'break', 'continue', 'true', 'false',
                                                             'null', 'undefined', 'this', 'class', 'export',
                                                             'import', 'from', 'default', 'async', 'await',
                                                             'try', 'catch', 'finally', 'throw', 'new',
                                                             'typeof', 'instanceof', 'in', 'of', 'delete']])
        
        potentially_undefined = used_vars - defined_vars
        
        valid_identifiers = {'console', 'window', 'document', 'Array', 'Object', 'String', 
                           'Number', 'Boolean', 'Math', 'Date', 'JSON', 'Promise',
                           'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
                           'px', 'em', 'rem', 'vh', 'vw', 'FC', 'ReactNode', 'ReactElement'}
        
        potentially_undefined = potentially_undefined - valid_identifiers
        
        if len(potentially_undefined) > 5:
            validity_score -= 0.3
            penalties.append(f"Potentially undefined variables: {len(potentially_undefined)}")
    except Exception as e:
        validity_score -= 0.05
        penalties.append(f"Variable analysis error: {str(e)}")
    
    if '<' in code and '>' in code:
        has_imports = 'import' in code.lower()
        reference_has_imports = reference_code and 'import' in reference_code.lower()
        
        if not has_imports and reference_has_imports:
            validity_score -= 0.05
            penalties.append("Missing imports (present in reference)")
    
    has_return = 'return' in code.lower()
    has_export = 'export' in code.lower()
    
    if not has_return and not has_export:
        validity_score -= 0.2
        penalties.append("Missing return or export statement")
    
    try:
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
    
    if 'useState(' in code and 'const [' not in code and 'const {' not in code:
        validity_score -= 0.1
        penalties.append("Possible incorrect useState usage")
    
    validity_score = max(validity_score, -1.0)
    
    return validity_score, penalties

def compute_dynamic_react_reward(code):
    """Reward dynamic/interactive React features."""
    dynamic_score = 0.0
    features_found = []
    
    if 'useState' in code:
        dynamic_score += 0.25
        features_found.append("useState")
    if 'useReducer' in code:
        dynamic_score += 0.15
        features_found.append("useReducer")
    if 'useContext' in code:
        dynamic_score += 0.10
        features_found.append("useContext")
    
    if 'useEffect' in code:
        dynamic_score += 0.15
        features_found.append("useEffect")
    if 'useLayoutEffect' in code:
        dynamic_score += 0.10
        features_found.append("useLayoutEffect")
    
    event_handlers = ['onClick', 'onChange', 'onSubmit', 'onFocus', 'onBlur',
                      'onMouseEnter', 'onMouseLeave', 'onKeyDown', 'onKeyUp',
                      'onInput', 'onSelect', 'onScroll']
    found_handlers = [handler for handler in event_handlers if handler in code]
    if found_handlers:
        handler_score = min(len(found_handlers) * 0.05, 0.15)
        dynamic_score += handler_score
        features_found.append(f"event_handlers({len(found_handlers)})")
    
    has_conditional = ('?' in code and ':' in code) or '&&' in code
    if has_conditional:
        dynamic_score += 0.10
        features_found.append("conditional_rendering")
    
    if re.search(r'\.map\s*\(', code):
        dynamic_score += 0.10
        features_found.append("map")
    if re.search(r'\.filter\s*\(', code):
        dynamic_score += 0.05
        features_found.append("filter")
    
    if 'useRef' in code or 'createRef' in code:
        dynamic_score += 0.05
        features_found.append("refs")
    
    if 'useMemo' in code or 'useCallback' in code or 'React.memo' in code or 'memo(' in code:
        dynamic_score += 0.05
        features_found.append("memoization")
    
    dynamic_score = min(dynamic_score, 1.0)
    return dynamic_score, features_found

def compute_code_reward(generated_code, reference_code):
    gen_len = len(generated_code)
    ref_len = len(reference_code)
    
    completeness_reward = 0.0
    truncation_indicators = [
        generated_code.count('{') != generated_code.count('}'),
        generated_code.count('[') != generated_code.count(']'),
        generated_code.rstrip().endswith((',', '(', '[', '{', '<')),
        not generated_code.rstrip().endswith(('}', ';', '>', ')', '`', '"', "'")),
    ]
    
    if sum(truncation_indicators) >= 2:
        completeness_reward = -1.0 * REWARD_COMPLETENESS_WEIGHT
    else:
        completeness_reward = 0.5 * REWARD_COMPLETENESS_WEIGHT
    
    validity_reward = 0.0
    if generated_code.count('{') == generated_code.count('}'):
        validity_reward += 0.3 * REWARD_VALIDITY_WEIGHT
    else:
        validity_reward -= 0.5 * REWARD_VALIDITY_WEIGHT
    
    if generated_code.count('[') == generated_code.count(']'):
        validity_reward += 0.15 * REWARD_VALIDITY_WEIGHT
    else:
        validity_reward -= 0.25 * REWARD_VALIDITY_WEIGHT
    
    if generated_code.count('(') == generated_code.count(')'):
        validity_reward += 0.15 * REWARD_VALIDITY_WEIGHT
    else:
        validity_reward -= 0.25 * REWARD_VALIDITY_WEIGHT
    
    if 'return' in generated_code.lower():
        validity_reward += 0.2 * REWARD_VALIDITY_WEIGHT
    
    quote_reward = 0.0
    single_quotes = generated_code.count("'") - generated_code.count("\\'")
    if single_quotes % 2 == 0:
        quote_reward += 0.4 * REWARD_QUOTE_WEIGHT
    else:
        quote_reward -= 0.6 * REWARD_QUOTE_WEIGHT
    
    double_quotes = generated_code.count('"') - generated_code.count('\\"')
    if double_quotes % 2 == 0:
        quote_reward += 0.3 * REWARD_QUOTE_WEIGHT
    else:
        quote_reward -= 0.5 * REWARD_QUOTE_WEIGHT
    
    backticks = generated_code.count('`')
    if backticks % 2 == 0:
        quote_reward += 0.3 * REWARD_QUOTE_WEIGHT
    else:
        quote_reward -= 0.5 * REWARD_QUOTE_WEIGHT
    
    dynamic_score, dynamic_features = compute_dynamic_react_reward(generated_code)
    dynamic_reward = dynamic_score * REWARD_DYNAMIC_REACT_WEIGHT
    
    length_penalty = -abs(gen_len - ref_len) / max(ref_len, 1) * REWARD_LENGTH_PENALTY_WEIGHT
    
    total_reward = REWARD_BASE + completeness_reward + validity_reward + quote_reward + dynamic_reward + length_penalty
    
    return total_reward

async def sample_trajectories_async(sampling_client, tokenizer, prompts, data):
    params = types.SamplingParams(
        max_tokens=MAX_GENERATION_TOKENS, 
        temperature=0.7, 
        top_p=0.9,
        stop=GENERATION_STOP_SEQUENCES
    )
    
    contexts = []
    coroutines = []
    prompt_lengths = []
    
    for prompt_text in prompts:
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_len = len(prompt_tokens)
        prompt_lengths.append(prompt_len)
        
        if prompt_len + MAX_GENERATION_TOKENS > 32768:
            print(f"WARNING: Skipping prompt with {prompt_len} tokens (would exceed context window)")
            continue
        
        prompt_input = types.ModelInput.from_ints(prompt_tokens)
        
        ref_response = ""
        for ex in data:
            if ex["full_prompt"] == prompt_text:
                ref_response = ex["reference_response"]
                break
        
        coro = sampling_client.sample_async(
            prompt=prompt_input, 
            sampling_params=params, 
            num_samples=NUM_SAMPLES_PER_PROMPT
        )
        
        coroutines.append(coro)
        contexts.append({
            "prompt_tokens": prompt_tokens,
            "prompt_text": prompt_text,
            "ref_response": ref_response
        })
    
    if prompt_lengths:
        print(f"      Prompt lengths - Min: {min(prompt_lengths)}, Max: {max(prompt_lengths)}, Avg: {sum(prompt_lengths)/len(prompt_lengths):.0f}")
    
    print(f"      🚀 Launching {len(coroutines)} concurrent sampling requests...")
    sample_start = time.time()
    results = await asyncio.gather(*coroutines)
    sample_time = time.time() - sample_start
    print(f"      ✅ All samples completed in {sample_time:.1f}s ({sample_time/len(coroutines):.2f}s per prompt)")
    
    # GRPO Step 1: Collect all samples and rewards, grouped by prompt
    groups = []  # Each element: {"ctx": ctx, "samples": [{"tokens": ..., "logprobs": ..., "reward": ...}]}
    reward_stats = {"total": [], "count": 0}
    
    for idx, (result, ctx) in enumerate(zip(results, contexts)):
        group_samples = []
        for seq in result.sequences:
            generated_tokens = seq.tokens
            if seq.logprobs is None:
                print("WARNING: No logprobs returned from sampling. Using zeros as placeholder.")
                logprobs = [0.0] * len(generated_tokens)
            else:
                logprobs = seq.logprobs
            
            generated_text = tokenizer.decode(generated_tokens)
            reward = compute_code_reward(generated_text, ctx["ref_response"])
            
            reward_stats["total"].append(reward)
            reward_stats["count"] += 1
            
            group_samples.append({
                "generated_tokens": generated_tokens,
                "logprobs": logprobs,
                "reward": reward,
                "generated_text": generated_text
            })
        
        groups.append({"ctx": ctx, "samples": group_samples})
    
    # GRPO Step 2: Compute group-relative advantages
    # For each prompt group, compute advantages as: reward - group_mean
    processed_data = []
    advantage_stats = {"advantages": [], "raw_rewards": []}
    
    for group in groups:
        ctx = group["ctx"]
        samples = group["samples"]
        
        # Compute group baseline (mean of group rewards)
        group_rewards = [s["reward"] for s in samples]
        baseline = sum(group_rewards) / len(group_rewards)  # Group mean as baseline
        
        # Compute advantages: reward - baseline
        for sample in samples:
            raw_reward = sample["reward"]
            advantage = raw_reward - baseline
            
            advantage_stats["advantages"].append(advantage)
            advantage_stats["raw_rewards"].append(raw_reward)
            
            # Create Datum with group-relative advantages
            all_tokens = ctx["prompt_tokens"] + sample["generated_tokens"]
            target_tokens = all_tokens[1:]
            input_tokens = all_tokens[:-1]
            
            old_logprobs = [0.0] * len(ctx["prompt_tokens"]) + sample["logprobs"]
            old_logprobs = old_logprobs[1:]
            
            prompt_length = len(ctx["prompt_tokens"])
            gen_length = len(sample["generated_tokens"])
            # Create per-token advantage array: 0 for prompt, advantage for generation
            advantages = [0.0] * (prompt_length - 1) + [advantage] * gen_length
            
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "logprobs": old_logprobs,
                    "advantages": advantages
                }
            )
            processed_data.append(datum)
    
    # Print statistics
    if reward_stats["total"]:
        avg_reward = np.mean(reward_stats["total"])
        min_reward = np.min(reward_stats["total"])
        max_reward = np.max(reward_stats["total"])
        std_reward = np.std(reward_stats["total"])
        print(f"      Rewards (raw) - Avg: {avg_reward:.3f} ± {std_reward:.3f}, Range: [{min_reward:.3f}, {max_reward:.3f}]")
    
    if advantage_stats["advantages"]:
        avg_adv = np.mean(advantage_stats["advantages"])
        std_adv = np.std(advantage_stats["advantages"])
        min_adv = np.min(advantage_stats["advantages"])
        max_adv = np.max(advantage_stats["advantages"])
        print(f"      Advantages (group-relative) - Avg: {avg_adv:.3f} ± {std_adv:.3f}, Range: [{min_adv:.3f}, {max_adv:.3f}]")
    
    return processed_data


async def train_grpo():
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(base_model=BASE_MODEL)
    
    tokenizer = training_client.get_tokenizer()

    data = load_data(tokenizer)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"GRPO TRAINING: {BASE_MODEL}")
    print(f"Training Examples: {len(data)} | Eval Examples: {NUM_EVAL_EXAMPLES}")
    print(f"Epochs: {NUM_GRPO_EPOCHS} | LR: {LEARNING_RATE}")
    print(f"Samples/Prompt: {NUM_SAMPLES_PER_PROMPT} | Max Tokens: {MAX_GENERATION_TOKENS}")
    print(f"Total samples per epoch: {len(data) * NUM_SAMPLES_PER_PROMPT}")
    print(f"{'='*70}\n")
    
    for epoch in range(NUM_GRPO_EPOCHS):
        epoch_start = time.time()
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{NUM_GRPO_EPOCHS}")
        print(f"{'='*70}")
        
        t1 = time.time()
        print(f"[1/4] Saving weights...")
        sampling_client = training_client.save_weights_and_get_sampling_client(name=f"temp_epoch_{epoch}")
        print(f"      ⏱️  {time.time() - t1:.1f}s")
        
        t2 = time.time()
        prompts = [ex["full_prompt"] for ex in data]
        print(f"[2/4] Sampling {len(prompts)} prompts × {NUM_SAMPLES_PER_PROMPT} samples = {len(prompts) * NUM_SAMPLES_PER_PROMPT} total...")
        processed_examples = await sample_trajectories_async(sampling_client, tokenizer, prompts, data)
        print(f"      ⏱️  {time.time() - t2:.1f}s")
        
        t3 = time.time()
        print(f"[3/4] Running forward/backward and optimizer step...")
        fwdbwd_future = await training_client.forward_backward_async(processed_examples, "grpo")
        optim_future = await training_client.optim_step_async(types.AdamParams(learning_rate=LEARNING_RATE))
        fwdbwd_result, optim_result = await asyncio.gather(
            fwdbwd_future.result_async(),
            optim_future.result_async()
        )
        print(f"      ⏱️  {time.time() - t3:.1f}s")
        
        logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
        avg_logprob = np.mean(logprobs)
        epoch_time = time.time() - epoch_start
        print(f"[4/4] Epoch Complete - Avg LogProb: {avg_logprob:.4f} | Total: {epoch_time:.1f}s")
    
    sampling_client = training_client.save_weights_and_get_sampling_client(name=CHECKPOINT_NAME)
    
    print(f"\nModel saved as {CHECKPOINT_NAME}")
    
    return sampling_client, tokenizer, data

async def evaluate(sampling_client, tokenizer, data):
    eval_data = data[:NUM_EVAL_EXAMPLES]
    
    print(f"\n{'='*70}")
    print(f"EVALUATION - {len(eval_data)} examples (sampled from {len(data)} total)")
    print(f"{'='*70}")
    results = []
    
    params = types.SamplingParams(
        max_tokens=MAX_GENERATION_TOKENS, 
        temperature=0.0, 
        stop=GENERATION_STOP_SEQUENCES
    )
    
    coroutines = []
    contexts = []
    
    for idx, example in enumerate(eval_data):
        prompt_text = example["full_prompt"]
        prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))
        
        coro = sampling_client.sample_async(prompt=prompt, sampling_params=params, num_samples=1)
        coroutines.append(coro)
        contexts.append({
            "idx": idx,
            "example": example
        })
    
    eval_results = await asyncio.gather(*coroutines)
    
    for eval_result, ctx in zip(eval_results, contexts):
        idx = ctx["idx"]
        example = ctx["example"]
        expected_response = example["reference_response"]
        user_message = example["user_message"]
        
        predicted = tokenizer.decode(eval_result.sequences[0].tokens).strip()
        
        has_code = "function" in predicted or "const" in predicted or "return" in predicted
        
        results.append({
            "task_id": f"example_{idx}",
            "user_message": user_message,
            "system_prompt": example["system_prompt"],
            "expected_response": expected_response,
            "predicted_response": predicted,
            "has_code": has_code
        })
    
    with open(f"{OUTPUT_DIR}/eval_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    code_rate = sum(r["has_code"] for r in results) / len(results) if results else 0.0
    print(f"Evaluation Complete - Code generation rate: {code_rate:.1%} ({sum(r['has_code'] for r in results)}/{len(results)})")
    print(f"Results saved to: {OUTPUT_DIR}/eval_results.jsonl")
    
    return results

async def main():
    sampling_client, tokenizer, data = await train_grpo()
    await evaluate(sampling_client, tokenizer, data)

if __name__ == "__main__":
    asyncio.run(main())

