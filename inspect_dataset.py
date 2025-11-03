from datasets import load_dataset

DATASET_NAME = "cfahlgren1/react-code-instructions"
NUM_EXAMPLES = 5

def format_react_example(example, idx):
    messages = example.get('messages', [])
    
    system_prompt = messages[0]['content'] if len(messages) > 0 else ''
    user_message = messages[1]['content'] if len(messages) > 1 else ''
    assistant_response = messages[2]['content'] if len(messages) > 2 else ''
    
    full_prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
    
    print(f"\n{'='*70}")
    print(f"EXAMPLE {idx}")
    print(f"{'='*70}")
    print(f"SYSTEM PROMPT (first 300 chars):\n{system_prompt[:300]}...\n")
    print(f"USER REQUEST:\n{user_message}\n")
    print(f"ASSISTANT RESPONSE (first 500 chars):\n{assistant_response[:500]}...\n")
    print(f"FULL RESPONSE LENGTH: {len(assistant_response)} chars\n")
    print(f"FULL PROMPT FOR MODEL (first 400 chars):\n{full_prompt[:400]}...\n")
    print(f"{'='*70}")
    
    return {
        "system_prompt": system_prompt,
        "user_message": user_message,
        "reference_response": assistant_response,
        "full_prompt": full_prompt
    }

print(f"Loading dataset: {DATASET_NAME}")
print(f"Number of examples to inspect: {NUM_EXAMPLES}\n")

try:
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"✅ Successfully loaded {DATASET_NAME}")
except Exception as e:
    print(f"❌ Failed to load {DATASET_NAME}: {e}")
    exit(1)

print(f"Total examples in dataset: {len(dataset)}\n")

selected = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))
formatted_examples = [format_react_example(ex, i) for i, ex in enumerate(selected)]

print(f"\n{'='*70}")
print(f"DATASET SUMMARY")
print(f"{'='*70}")
print(f"Total examples processed: {len(formatted_examples)}")
print(f"Average system prompt length: {sum(len(e['system_prompt']) for e in formatted_examples) / len(formatted_examples):.0f} chars")
print(f"Average user message length: {sum(len(e['user_message']) for e in formatted_examples) / len(formatted_examples):.0f} chars")
print(f"Average response length: {sum(len(e['reference_response']) for e in formatted_examples) / len(formatted_examples):.0f} chars")
print(f"Average full prompt length: {sum(len(e['full_prompt']) for e in formatted_examples) / len(formatted_examples):.0f} chars")

code_keywords = ['function', 'const', 'return', 'import', 'export', 'tsx', 'jsx']
has_code_structure = sum(1 for ex in formatted_examples if any(kw in ex['reference_response'].lower() for kw in code_keywords))
print(f"Examples with code structure: {has_code_structure}/{len(formatted_examples)}")
print(f"{'='*70}")

