import os
import json
import numpy as np
import tinker
from tinker import types
from datasets import load_dataset

DATASET_NAME = "cfahlgren1/react-code-instructions"
NUM_EXAMPLES = 5
BASE_MODEL = "meta-llama/Llama-3.2-1B"
LEARNING_RATE = 1e-5
NUM_PPO_EPOCHS = 3
NUM_SAMPLES_PER_PROMPT = 4
MAX_GENERATION_TOKENS = 512
PPO_CLIP_EPSILON = 0.2
VALUE_CLIP_EPSILON = 0.2
GAE_LAMBDA = 0.95
ENTROPY_COEFF = 0.01
OUTPUT_DIR = "outputs"
CHECKPOINT_NAME = "react-code-ppo"

def format_react_example(example, idx):
    messages = example.get('messages', [])
    
    system_prompt = messages[0]['content'] if len(messages) > 0 else ''
    user_message = messages[1]['content'] if len(messages) > 1 else ''
    assistant_response = messages[2]['content'] if len(messages) > 2 else ''
    
    full_prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
    
    print(f"\n{'='*70}")
    print(f"EXAMPLE {idx}")
    print(f"{'='*70}")
    print(f"SYSTEM PROMPT (first 200 chars):\n{system_prompt[:200]}...\n")
    print(f"USER REQUEST:\n{user_message}\n")
    print(f"ASSISTANT RESPONSE (first 300 chars):\n{assistant_response[:300]}...\n")
    print(f"FULL RESPONSE LENGTH: {len(assistant_response)} chars")
    print(f"{'='*70}")
    
    return {
        "system_prompt": system_prompt,
        "user_message": user_message,
        "reference_response": assistant_response,
        "full_prompt": full_prompt
    }

def load_data():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    print(f"Dataset loaded. Total examples: {len(dataset)}")
    print(f"Selecting first {NUM_EXAMPLES} examples\n")
    
    selected = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))
    return [format_react_example(ex, i) for i, ex in enumerate(selected)]

def compute_code_reward(generated_code, reference_code):
    gen_len = len(generated_code)
    ref_len = len(reference_code)
    
    has_jsx = "return" in generated_code.lower() and ("<" in generated_code or "/>" in generated_code)
    length_penalty = -abs(gen_len - ref_len) / max(ref_len, 1) * 0.1
    jsx_bonus = 0.5 if has_jsx else 0.0
    base_reward = 1.0
    
    return base_reward + length_penalty + jsx_bonus

def sample_trajectories(sampling_client, tokenizer, prompts):
    trajectories = []
    
    params = types.SamplingParams(
        max_tokens=MAX_GENERATION_TOKENS, 
        temperature=0.7, 
        top_p=0.9,
        stop=["###", "\n\n\n"]
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
    
    for traj in trajectories:
        generated_text = tokenizer.decode(traj["generated_tokens"])
        
        ref_response = ""
        for ex in data:
            if ex["full_prompt"] == traj["prompt_text"]:
                ref_response = ex["reference_response"]
                break
        
        reward = compute_code_reward(generated_text, ref_response)
        
        all_tokens = traj["prompt_tokens"] + traj["generated_tokens"]
        target_tokens = all_tokens[1:]
        input_tokens = all_tokens[:-1]
        
        old_logprobs = [0.0] * len(traj["prompt_tokens"]) + traj["logprobs"]
        old_logprobs = old_logprobs[1:]
        
        advantages = [reward / len(traj["generated_tokens"])] * len(old_logprobs)
        
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "logprobs": old_logprobs,
                "advantages": advantages
            }
        )
        processed_data.append(datum)
    
    return processed_data

def train_ppo():
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
    
    tokenizer = training_client.get_tokenizer()
    
    data = load_data()
    
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
        optim_result = optim_future.result()
        
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
        stop=["###", "\n\n\n"]
    )
    
    for idx, example in enumerate(data):
        prompt_text = example["full_prompt"]
        expected_response = example["reference_response"]
        user_message = example["user_message"]
        
        print(f"\n--- Evaluating Example {idx} ---")
        print(f"User Request: {user_message[:100]}...")
        
        prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))
        future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
        result = future.result()
        
        predicted = tokenizer.decode(result.sequences[0].tokens).strip()
        
        has_code = "function" in predicted or "const" in predicted or "return" in predicted
        
        results.append({
            "task_id": f"example_{idx}",
            "user_message": user_message,
            "system_prompt": example["system_prompt"][:500],
            "expected_response": expected_response,
            "predicted_response": predicted,
            "has_code": has_code
        })
        
        print(f"Expected length: {len(expected_response)} chars")
        print(f"Generated length: {len(predicted)} chars")
        print(f"Has code structure: {has_code}")
        print(f"Generated code preview:\n{predicted[:300]}...")
    
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

