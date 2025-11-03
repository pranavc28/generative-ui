import yaml
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
    print(f"TRAINING INPUT:\n{system}\n")
    print(f"TRAINING OUTPUT:\n{chat}\n")
    print(f"--- EVALUATION ---")
    print(f"EVAL PROMPT (model sees this):\n{eval_prompt}\n")
    print(f"EXPECTED COMPLETION:\n{assistant_response}\n")
    print(f"{'='*70}")
    
    return {
        "input": system, 
        "output": chat,
        "user_message": user_msg,
        "assistant_response": assistant_response,
        "eval_input": eval_prompt
    }

config = load_config()
dataset_name = config['data']['dataset']
split = config['data']['split']
num_examples = config['data']['num_examples']

print(f"Loading dataset: {dataset_name}")
print(f"Split: {split}")
print(f"Number of examples: {num_examples}\n")

try:
    dataset = load_dataset(dataset_name, split=split)
    print(f"✅ Successfully loaded {dataset_name}")
except Exception as e:
    print(f"❌ Failed to load {dataset_name}: {e}")
    print("\nTrying alternative: glaiveai/glaive-function-calling-v2")
    dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    print(f"✅ Successfully loaded alternative dataset")

print(f"Total examples in dataset: {len(dataset)}\n")

selected = dataset.select(range(min(num_examples, len(dataset))))
formatted_examples = [format_toolbench_example(ex, i) for i, ex in enumerate(selected)]

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total examples processed: {len(formatted_examples)}")
print(f"Average input length: {sum(len(e['input']) for e in formatted_examples) / len(formatted_examples):.0f} chars")
print(f"Average output length: {sum(len(e['output']) for e in formatted_examples) / len(formatted_examples):.0f} chars")

