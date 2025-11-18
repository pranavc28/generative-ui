"""
Sample from Qwen model using user messages from extracted_results.jsonl

This script:
1. Reads user messages from extracted_results.jsonl
2. Samples completions from the Qwen/Qwen3-30B-A3B model
3. Outputs results to sampled_outputs.jsonl
"""
import json
import asyncio
import tinker
from tinker import types
from typing import List, Dict, Any
from transformers import AutoTokenizer


# Model configuration (matching train.py)
BASE_MODEL = "Qwen/Qwen3-30B-A3B"
INPUT_FILE = "outputs/extracted_results.jsonl"
OUTPUT_FILE = "outputs/sampled_outputs.jsonl"


async def sample_from_model(
    sampling_client: tinker.SamplingClient,
    tokenizer: Any,
    user_message: str,
    task_id: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Sample a completion from the model for a given user message.
    
    Args:
        sampling_client: Tinker sampling client
        tokenizer: Tokenizer for the model
        user_message: User's input message
        task_id: Task identifier
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Dictionary with task_id, user_message, and generated response
    """
    # Add critical code correctness instructions (from train.py)
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
    
    # Format the prompt with enhanced system message
    system_message = "You are an expert React, TypeScript, and TailwindCSS developer." + code_correctness_instructions
    prompt_text = f"<|im_start|>system\n{system_message}\n<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    
    # Encode the prompt
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt = types.ModelInput.from_ints(prompt_tokens)
    
    # Set up sampling parameters
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    # Sample from the model
    try:
        # Submit the sampling request and await the response
        response = await sampling_client.sample_async(
            prompt=prompt,
            sampling_params=sampling_params,
            num_samples=1
        )
        
        # Decode the generated tokens
        generated_tokens = response.sequences[0].tokens
        generated_text = tokenizer.decode(generated_tokens)
        
        return {
            "task_id": task_id,
            "user_message": user_message,
            "generated_response": generated_text,
            "stop_reason": response.sequences[0].stop_reason,
            "num_tokens": len(generated_tokens)
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "user_message": user_message,
            "error": str(e)
        }


async def main():
    """Main function to process all examples from extracted_results.jsonl"""
    
    print(f"Loading examples from {INPUT_FILE}...")
    
    # Read the input file
    examples: List[Dict[str, Any]] = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append({
                "task_id": data["task_id"],
                "user_message": data["user_message"]
            })
    
    print(f"Found {len(examples)} examples")
    
    # Load tokenizer
    print(f"\nLoading tokenizer for {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Create Tinker service client
    print(f"Initializing Tinker service client...")
    service_client = tinker.ServiceClient()
    
    # Create sampling client (synchronous method)
    sampling_client = service_client.create_sampling_client(
        base_model=BASE_MODEL
    )
    
    print(f"Sampling from {BASE_MODEL}...")
    print(f"Output will be saved to {OUTPUT_FILE}\n")
    
    # Process each example
    results = []
    for i, example in enumerate(examples):
        print(f"Processing example {i+1}/{len(examples)}: {example['task_id']}")
        
        result = await sample_from_model(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            user_message=example["user_message"],
            task_id=example["task_id"],
            max_tokens=16000,
            temperature=0.7
        )
        
        results.append(result)
        
        # Show preview of generated text
        if "error" in result:
            print(f"  ERROR: {result['error']}\n")
        else:
            preview = result["generated_response"][:100].replace('\n', ' ')
            print(f"  Generated {result['num_tokens']} tokens: {preview}...\n")
    
    # Write results to output file
    print(f"\nWriting results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Done! Processed {len(results)} examples.")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

