import tinker
from tinker import types

MODEL_PATH = "tinker://react-code-ppo"
SYSTEM_PROMPT = """
You are an expert React, TypeScript, and TailwindCSS developer with a keen eye for modern, aesthetically pleasing design.

Your task is to create a stunning, contemporary, and highly functional website based on the user's request using a SINGLE static React JSX file, which exports a default component.
"""

def sample_react_code(user_request: str, max_tokens: int = 512, temperature: float = 0.7):
    service_client = tinker.ServiceClient()
    
    sampling_client = service_client.create_sampling_client(model_path=MODEL_PATH)
    
    tokenizer = sampling_client.get_tokenizer()
    
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_request}\n\nAssistant:"
    
    prompt_tokens = tokenizer.encode(full_prompt)
    prompt_input = types.ModelInput.from_ints(prompt_tokens)
    
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        stop=["User:", "\n\n\n"]
    )
    
    print(f"Generating React code for: {user_request}")
    print("="*70)
    
    future = sampling_client.sample(prompt=prompt_input, sampling_params=params, num_samples=1)
    result = future.result()
    
    generated_code = tokenizer.decode(result.sequences[0].tokens)
    
    print(generated_code)
    print("="*70)
    
    return generated_code


if __name__ == "__main__":
    examples = [
        "create a todo list app",
        "build a calculator",
        "make a weather dashboard"
    ]
    
    for example in examples:
        code = sample_react_code(example, max_tokens=800, temperature=0.7)
        print("\n")

