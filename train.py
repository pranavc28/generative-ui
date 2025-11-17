"""
PPO Training with Asynchronous Grouped Policy Training (Tinker API)

This implementation uses async sampling and processing for improved efficiency:
1. sample_async() launches all sampling requests asynchronously
2. Trajectories are processed and rewards computed as samples complete
3. forward_backward_async() and optim_step_async() overlap computation
4. Evaluation also uses async sampling for faster inference

This approach provides:
- Immediate feedback as samples complete
- Better GPU utilization through overlapping compute
- Faster training iterations compared to synchronous batching
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
NUM_EXAMPLES = 20  # Start small for testing, scale to 60+ for full training
BASE_MODEL = "Qwen/Qwen3-30B-A3B"
LEARNING_RATE = 1e-5
NUM_PPO_EPOCHS = 3
NUM_SAMPLES_PER_PROMPT = 2  # Start with 2 for speed, increase to 4+ for better training
MAX_GENERATION_TOKENS = 12288  # Increased to 12k: eval showed 40% truncation at 8192. Components avg 9k tokens.
GENERATION_STOP_SEQUENCES = ["</code>", "```\n\n", "\n\n\n\n"]  # Stop sequences to detect completion
PPO_CLIP_EPSILON = 0.2
VALUE_CLIP_EPSILON = 0.2
GAE_LAMBDA = 0.95
ENTROPY_COEFF = 0.01
OUTPUT_DIR = "outputs"
CHECKPOINT_NAME = "react-code-ppo-qwen3-30b-a3b-v4"

# Simplified Reward System - Focus on Core Quality
# 
# Philosophy: Keep it simple! PPO learns best with clear, strong signals.
# Start with fundamentals, add complexity later if needed.
#
# Phase 1: Get basic code generation working
REWARD_BASE = 1.0                      # Base reward
REWARD_COMPLETENESS_WEIGHT = 10.0      # CRITICAL: Code must be complete (not truncated) - DOUBLED!
REWARD_VALIDITY_WEIGHT = 4.0           # IMPORTANT: Basic syntax validity (balanced braces) - DOUBLED!
REWARD_LENGTH_PENALTY_WEIGHT = 0.1     # MINOR: Encourage reasonable length
#
# Eval results (C grade): 60% complete, 55% balanced braces
# → Increased weights to provide stronger learning signals
#
# Removed (for now): TailwindCSS similarity, JSX structure analysis, 
# undefined variable detection. Add these back in Phase 2 if needed.

def format_react_example(example, idx, tokenizer=None):
    messages = example.get('messages', [])
    
    system_prompt = messages[0]['content'] if len(messages) > 0 else ''
    user_message = messages[1]['content'] if len(messages) > 1 else ''
    assistant_response = messages[2]['content'] if len(messages) > 2 else ''
    
    # Add critical code correctness instructions to the system prompt
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
    selected = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))
    return [format_react_example(ex, i, tokenizer) for i, ex in enumerate(selected)]

def extract_valid_identifiers_from_reference(reference_code):
    """
    Extract valid identifiers (imports, constants) from reference code.
    This helps avoid penalizing the generated code for using identifiers 
    that are imported/defined in the reference.
    """
    valid_ids = set()
    
    try:
        # Extract all imports (named and default)
        # Match: import X from 'Y' or import { A, B } from 'Y' or import * as X from 'Y'
        import_patterns = [
            r'import\s+(\w+)\s+from',  # default imports
            r'import\s+\*\s+as\s+(\w+)\s+from',  # namespace imports
            r'import\s+{([^}]+)}\s+from',  # named imports
            r'import\s+(\w+)\s*,\s*{([^}]+)}\s+from',  # mixed imports
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
        
        # Extract type imports (TypeScript)
        type_imports = re.findall(r'import\s+type\s+{([^}]+)}\s+from', reference_code)
        for imports_str in type_imports:
            for name in imports_str.split(','):
                if ' as ' in name:
                    name = name.split(' as ')[-1]
                clean_name = name.strip()
                if clean_name:
                    valid_ids.add(clean_name)
        
        # Extract interface and type definitions
        interfaces = re.findall(r'(?:interface|type)\s+(\w+)', reference_code)
        valid_ids.update(interfaces)
        
        # Extract const/let/var declarations that might be used as constants
        const_declarations = re.findall(r'(?:const|let|var)\s+(\w+)', reference_code)
        valid_ids.update(const_declarations)
        
    except Exception as e:
        print(f"Warning: Error extracting identifiers from reference: {e}")
    
    return valid_ids

def check_code_validity(code, reference_code=None):
    """
    Check for common code errors and return a validity score.
    Returns a score between -1.0 (very invalid) and 0.0 (valid).
    Checks for:
    1. Basic syntax errors (unmatched braces, brackets, parentheses)
    2. Undefined variables (common React/TS patterns)
    3. Missing imports for React
    4. Function/component structure issues
    5. Truncated/incomplete code
    """
    validity_score = 0.0
    penalties = []
    
    # Check for truncated/incomplete code (CRITICAL for quality)
    is_truncated = False
    truncation_indicators = [
        code.count('{') > code.count('}'),  # More opening than closing braces
        code.count('[') > code.count(']'),  # Unbalanced brackets
        code.rstrip().endswith((',', '(', '[', '{', '<')),  # Ends with opening token
        not code.rstrip().endswith(('}', ';', '>', ')', '`', '"', "'")),  # Doesn't end properly
    ]
    
    if sum(truncation_indicators) >= 2:  # Multiple indicators suggest truncation
        is_truncated = True
        validity_score -= 1.5  # HEAVY penalty for truncation
        penalties.append("Code is truncated/incomplete - CRITICAL ERROR")
    
    # Extract valid identifiers from reference code if provided
    reference_identifiers = set()
    if reference_code:
        reference_identifiers = extract_valid_identifiers_from_reference(reference_code)
    
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
        
        # Add identifiers from reference code (imports, constants, etc.)
        defined_vars.update(reference_identifiers)
        
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
                           'px', 'em', 'rem', 'vh', 'vw', 'FC', 'ReactNode', 'ReactElement'}  # CSS units + React types
        
        potentially_undefined = potentially_undefined - valid_identifiers
        
        # Penalize if there are many undefined variables (more than 5 could be a problem)
        if len(potentially_undefined) > 5:
            validity_score -= 0.3
            penalties.append(f"Potentially undefined variables: {len(potentially_undefined)}")
    except Exception as e:
        validity_score -= 0.05
        penalties.append(f"Variable analysis error: {str(e)}")
    
    # Check 3: Missing React import (for TSX/JSX code)
    # Only penalize if reference doesn't have imports either (be lenient about imports)
    if '<' in code and '>' in code:  # Likely JSX
        has_imports = 'import' in code.lower()
        reference_has_imports = reference_code and 'import' in reference_code.lower()
        
        # Only penalize if generated has no imports but reference does
        if not has_imports and reference_has_imports:
            validity_score -= 0.05  # Reduced penalty
            penalties.append("Missing imports (present in reference)")
    
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
    SIMPLIFIED REWARD FUNCTION - Focus on Core Quality:
    1. Code completeness (not truncated) - CRITICAL
    2. Basic validity (balanced braces) - IMPORTANT  
    3. Reasonable length - MINOR
    """
    gen_len = len(generated_code)
    ref_len = len(reference_code)
    
    # CRITICAL: Check if code is complete (not truncated)
    completeness_reward = 0.0
    truncation_indicators = [
        generated_code.count('{') != generated_code.count('}'),
        generated_code.count('[') != generated_code.count(']'),
        generated_code.rstrip().endswith((',', '(', '[', '{', '<')),
        not generated_code.rstrip().endswith(('}', ';', '>', ')', '`', '"', "'")),
    ]
    
    if sum(truncation_indicators) >= 2:
        # Code is truncated - HEAVY penalty
        completeness_reward = -1.0 * REWARD_COMPLETENESS_WEIGHT
    else:
        # Code is complete - REWARD this!
        completeness_reward = 0.5 * REWARD_COMPLETENESS_WEIGHT
    
    # IMPORTANT: Basic validity checks
    validity_reward = 0.0
    
    # Check balanced braces
    if generated_code.count('{') == generated_code.count('}'):
        validity_reward += 0.3 * REWARD_VALIDITY_WEIGHT
    else:
        validity_reward -= 0.5 * REWARD_VALIDITY_WEIGHT
    
    # Check has return statement (basic React component requirement)
    if 'return' in generated_code.lower():
        validity_reward += 0.2 * REWARD_VALIDITY_WEIGHT
    
    # MINOR: Length penalty (don't deviate too much from reference)
    length_penalty = -abs(gen_len - ref_len) / max(ref_len, 1) * REWARD_LENGTH_PENALTY_WEIGHT
    
    # Combine all rewards
    total_reward = REWARD_BASE + completeness_reward + validity_reward + length_penalty
    
    return total_reward

async def sample_trajectories_async(sampling_client, tokenizer, prompts, data):
    """
    Asynchronous sampling that processes trajectories as they complete.
    Returns processed data ready for PPO update.
    """
    params = types.SamplingParams(
        max_tokens=MAX_GENERATION_TOKENS, 
        temperature=0.7, 
        top_p=0.9,
        stop=GENERATION_STOP_SEQUENCES
    )
    
    # Prepare all sampling requests with context
    contexts = []
    coroutines = []
    
    for prompt_text in prompts:
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_input = types.ModelInput.from_ints(prompt_tokens)
        
        # Find reference response for reward computation
        ref_response = ""
        for ex in data:
            if ex["full_prompt"] == prompt_text:
                ref_response = ex["reference_response"]
                break
        
        # sample_async returns a coroutine that needs to be awaited
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
    
    # Launch ALL sampling requests concurrently using asyncio.gather
    print(f"      🚀 Launching {len(coroutines)} concurrent sampling requests...")
    sample_start = time.time()
    results = await asyncio.gather(*coroutines)
    sample_time = time.time() - sample_start
    print(f"      ✅ All samples completed in {sample_time:.1f}s ({sample_time/len(coroutines):.2f}s per prompt)")
    
    # Process all results
    processed_data = []
    reward_stats = {"total": [], "count": 0}
    
    for idx, (result, ctx) in enumerate(zip(results, contexts)):
        # Process each sample in the batch
        for seq in result.sequences:
            generated_tokens = seq.tokens
            if seq.logprobs is None:
                print("WARNING: No logprobs returned from sampling. Using zeros as placeholder.")
                logprobs = [0.0] * len(generated_tokens)
            else:
                logprobs = seq.logprobs
            
            # Decode and compute reward immediately
            generated_text = tokenizer.decode(generated_tokens)
            reward = compute_code_reward(generated_text, ctx["ref_response"])
            
            reward_stats["total"].append(reward)
            reward_stats["count"] += 1
            
            # Create PPO datum
            all_tokens = ctx["prompt_tokens"] + generated_tokens
            target_tokens = all_tokens[1:]
            input_tokens = all_tokens[:-1]
            
            old_logprobs = [0.0] * len(ctx["prompt_tokens"]) + logprobs
            old_logprobs = old_logprobs[1:]
            
            # Apply full reward to each generated token, zero to prompt tokens
            prompt_length = len(ctx["prompt_tokens"]) - 1
            advantages = [0.0] * prompt_length + [reward] * len(generated_tokens)
            
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
        std_reward = np.std(reward_stats["total"])
        print(f"      Rewards - Avg: {avg_reward:.3f} ± {std_reward:.3f}, Range: [{min_reward:.3f}, {max_reward:.3f}]")
    
    return processed_data


async def train_ppo():
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(base_model=BASE_MODEL)
    
    tokenizer = training_client.get_tokenizer()

    data = load_data(tokenizer)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PPO TRAINING: {BASE_MODEL}")
    print(f"Examples: {len(data)} | Epochs: {NUM_PPO_EPOCHS} | LR: {LEARNING_RATE}")
    print(f"Samples/Prompt: {NUM_SAMPLES_PER_PROMPT} | Max Tokens: {MAX_GENERATION_TOKENS}")
    print(f"Total samples per epoch: {len(data) * NUM_SAMPLES_PER_PROMPT}")
    print(f"{'='*70}\n")
    
    for epoch in range(NUM_PPO_EPOCHS):
        epoch_start = time.time()
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{NUM_PPO_EPOCHS}")
        print(f"{'='*70}")
        
        # Stage 1: Save weights and create sampling client
        t1 = time.time()
        print(f"[1/4] Saving weights...")
        sampling_client = training_client.save_weights_and_get_sampling_client(name=f"temp_epoch_{epoch}")
        print(f"      ⏱️  {time.time() - t1:.1f}s")
        
        # Stage 2: Async sampling
        t2 = time.time()
        prompts = [ex["full_prompt"] for ex in data]
        print(f"[2/4] Sampling {len(prompts)} prompts × {NUM_SAMPLES_PER_PROMPT} samples = {len(prompts) * NUM_SAMPLES_PER_PROMPT} total...")
        processed_examples = await sample_trajectories_async(sampling_client, tokenizer, prompts, data)
        print(f"      ⏱️  {time.time() - t2:.1f}s")
        
        # Stage 3: Training step
        t3 = time.time()
        print(f"[3/4] Running forward/backward and optimizer step...")
        # Submit both requests (first await)
        fwdbwd_future = await training_client.forward_backward_async(processed_examples, "ppo")
        optim_future = await training_client.optim_step_async(types.AdamParams(learning_rate=LEARNING_RATE))
        # Wait for results to complete (second await)
        fwdbwd_result, optim_result = await asyncio.gather(
            fwdbwd_future.result_async(),
            optim_future.result_async()
        )
        print(f"      ⏱️  {time.time() - t3:.1f}s")
        
        # Stage 4: Log metrics
        logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
        avg_logprob = np.mean(logprobs)
        epoch_time = time.time() - epoch_start
        print(f"[4/4] Epoch Complete - Avg LogProb: {avg_logprob:.4f} | Total: {epoch_time:.1f}s")
    
    sampling_client = training_client.save_weights_and_get_sampling_client(name=CHECKPOINT_NAME)
    
    print(f"\nModel saved as {CHECKPOINT_NAME}")
    
    return sampling_client, tokenizer, data

async def evaluate(sampling_client, tokenizer, data):
    print(f"\n{'='*70}")
    print(f"EVALUATION - {len(data)} examples")
    print(f"{'='*70}")
    results = []
    
    params = types.SamplingParams(
        max_tokens=MAX_GENERATION_TOKENS, 
        temperature=0.0, 
        stop=GENERATION_STOP_SEQUENCES
    )
    
    # Prepare all evaluation samples
    coroutines = []
    contexts = []
    
    for idx, example in enumerate(data):
        prompt_text = example["full_prompt"]
        prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))
        
        coro = sampling_client.sample_async(prompt=prompt, sampling_params=params, num_samples=1)
        coroutines.append(coro)
        contexts.append({
            "idx": idx,
            "example": example
        })
    
    # Launch ALL evaluation samples concurrently
    eval_results = await asyncio.gather(*coroutines)
    
    # Process all results
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
    """Main async entry point for training and evaluation."""
    sampling_client, tokenizer, data = await train_ppo()
    await evaluate(sampling_client, tokenizer, data)

if __name__ == "__main__":
    asyncio.run(main())

