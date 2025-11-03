import os
import json
import yaml
import numpy as np
import tinker
from tinker import types
from datasets import load_dataset

CONFIG_PATH = "tinker.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def format_toolbench_example(example, idx):
    system = example.get('system', '')
    chat = example.get('chat', '')
    
    user_msg = chat.split('\n\n\nASSISTANT:')[0] if '\n\n\nASSISTANT:' in chat else chat.split('\n\n\nA:')[0]
    assistant_response = chat.split('\n\n\nASSISTANT:')[1].split('<|endoftext|>')[0].strip() if '\n\n\nASSISTANT:' in chat else ""
    
    eval_prompt = f"{system}\n\n{user_msg}\n\n\nASSISTANT:"
    
    print(f"\n{'='*70}")
    print(f"EXAMPLE {idx}")
    print(f"{'='*70}")
    print(f"SYSTEM:\n{system}\n")
    print(f"USER MESSAGE:\n{user_msg}\n")
    print(f"EXPECTED RESPONSE:\n{assistant_response}\n")
    print(f"EVAL PROMPT (ends with 'ASSISTANT:'):\n{eval_prompt[-100:]}...\n")
    print(f"{'='*70}")
    
    return {
        "input": system, 
        "output": chat,
        "user_message": user_msg,
        "assistant_response": assistant_response,
        "eval_input": eval_prompt
    }

def load_data(config):
    dataset_name = config['data']['dataset']
    split = config['data']['split']
    num_examples = config['data']['num_examples']
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    print(f"Dataset loaded. Total examples: {len(dataset)}")
    print(f"Selecting first {num_examples} examples\n")
    
    selected = dataset.select(range(min(num_examples, len(dataset))))
    return [format_toolbench_example(ex, i) for i, ex in enumerate(selected)]

def process_example(example, tokenizer, idx=None):
    prompt = example['input']
    completion = f" {example['output']}\n\n"
    
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)
    
    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights
    
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]
    
    if idx is not None:
        print(f"  Example {idx}: {len(prompt_tokens)} prompt tokens + {len(completion_tokens)} completion tokens = {len(tokens)} total")
    
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )

def train_ppo(config):
    service_client = tinker.ServiceClient()
    model_name = config['model']['base_model']
    training_client = service_client.create_lora_training_client(base_model=model_name)
    
    tokenizer = training_client.get_tokenizer()
    
    data = load_data(config)
    
    print(f"\nTokenizing {len(data)} examples...")
    processed_examples = [process_example(ex, tokenizer, idx=i) for i, ex in enumerate(data)]
    
    ppo_config = config['ppo']
    output_dir = config['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTraining with lr={ppo_config['learning_rate']}")
    print(f"Total examples: {len(data)}")
    
    num_epochs = ppo_config['epochs_per_update']
    
    for epoch in range(num_epochs):
        fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=ppo_config['learning_rate'])
        )
        
        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()
        
        logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([ex.loss_fn_inputs['weights'].tolist() for ex in processed_examples])
        loss = -np.dot(logprobs, weights) / weights.sum()
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.4f}")
    
    checkpoint_name = config['checkpoint']['save_name']
    sampling_client = training_client.save_weights_and_get_sampling_client(name=checkpoint_name)
    
    print(f"\nModel saved as {checkpoint_name}")
    
    return sampling_client, tokenizer, data

def evaluate(sampling_client, tokenizer, data):
    print("\n=== Evaluation ===")
    results = []
    
    params = types.SamplingParams(max_tokens=512, temperature=0.0, stop=["<|endoftext|>", "\n\n\n"])
    
    for idx, example in enumerate(data):
        eval_prompt = example["eval_input"]
        expected_response = example["assistant_response"]
        user_msg = example["user_message"]
        
        print(f"\n--- Evaluating Example {idx} ---")
        print(f"Prompt ends with: ...{eval_prompt[-50:]}")
        
        prompt = types.ModelInput.from_ints(tokenizer.encode(eval_prompt))
        future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
        result = future.result()
        
        predicted = tokenizer.decode(result.sequences[0].tokens).strip()
        
        match = predicted == expected_response
        
        results.append({
            "task_id": f"example_{idx}",
            "user_message": user_msg,
            "expected_response": expected_response,
            "predicted_response": predicted,
            "match": match
        })
        
        print(f"Expected: {expected_response[:100]}...")
        print(f"Predicted: {predicted[:100]}...")
        print(f"Match: {match}")
    
    config = load_config()
    output_dir = config['data']['output_dir']
    
    with open(f"{output_dir}/eval_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    accuracy = sum(r["match"] for r in results) / len(results) if results else 0.0
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total examples: {len(results)}")
    print(f"Exact matches: {sum(r['match'] for r in results)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results saved to: {output_dir}/eval_results.jsonl")
    
    return results

if __name__ == "__main__":
    config = load_config()
    sampling_client, tokenizer, data = train_ppo(config)
    evaluate(sampling_client, tokenizer, data)

