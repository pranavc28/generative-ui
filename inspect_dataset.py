from datasets import load_dataset

DATASET_NAME = "cfahlgren1/react-code-instructions"
NUM_EXAMPLES = 5

def format_react_example(example, idx):
    instruction = example.get('instruction', '')
    response = example.get('response', '')
    
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    print(f"\n{'='*70}")
    print(f"EXAMPLE {idx}")
    print(f"{'='*70}")
    print(f"INSTRUCTION:\n{instruction}\n")
    print(f"REFERENCE RESPONSE (first 500 chars):\n{response[:500]}...\n")
    print(f"FULL RESPONSE LENGTH: {len(response)} chars\n")
    print(f"PROMPT FOR MODEL:\n{prompt}\n")
    print(f"{'='*70}")
    
    return {
        "instruction": instruction,
        "reference_response": response,
        "prompt": prompt
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
print(f"Average instruction length: {sum(len(e['instruction']) for e in formatted_examples) / len(formatted_examples):.0f} chars")
print(f"Average response length: {sum(len(e['reference_response']) for e in formatted_examples) / len(formatted_examples):.0f} chars")

code_keywords = ['function', 'const', 'return', 'import', 'export']
has_code_structure = sum(1 for ex in formatted_examples if any(kw in ex['reference_response'] for kw in code_keywords))
print(f"Examples with code structure: {has_code_structure}/{len(formatted_examples)}")
print(f"{'='*70}")

