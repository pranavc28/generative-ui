================================================================================
TINKER PPO FINE-TUNING PIPELINE - QUICK START GUIDE
================================================================================

SETUP:
1. Ensure TINKER_API_KEY is set:
   export TINKER_API_KEY='your_api_key_here'

2. Install dependencies:
   pip install -r requirements.txt

TESTING THE PIPELINE:

Step 1: Test dataset loading (no API calls)
   python test_data_load.py
   
   This will:
   - Load the cfahlgren1/react-code-instructions dataset
   - Display the first 5 examples
   - Show instruction and response previews
   - Verify React code structure

Step 2: Inspect dataset in detail (no API calls)
   python inspect_dataset.py
   
   This will:
   - Show full formatting of each example
   - Display prompts that will be sent to the model
   - Show statistics about the dataset

Step 3: Run PPO training (requires Tinker API)
   python train.py
   
   This will:
   - Load 5 React code examples
   - Print all input data with instructions and reference responses
   - Run 3 epochs of PPO training
   - Sample 4 trajectories per prompt each epoch (60 total samples)
   - Compute rewards for generated code
   - Update the policy using PPO loss
   - Evaluate on the training examples
   - Save results to outputs/eval_results.jsonl

Step 4: View detailed metrics (after training)
   python evaluation.py
   
   This will:
   - Load evaluation results
   - Compute code generation metrics
   - Show detailed results for each example

FULL PIPELINE (Steps 3 + 4 combined):
   python run_pipeline.py

CONFIGURATION:
All settings are constants at the top of train.py:
- DATASET_NAME: "cfahlgren1/react-code-instructions"
- NUM_EXAMPLES: 5
- BASE_MODEL: "meta-llama/Llama-3.2-1B"
- LEARNING_RATE: 1e-5
- NUM_PPO_EPOCHS: 3
- NUM_SAMPLES_PER_PROMPT: 4
- MAX_GENERATION_TOKENS: 512
- PPO_CLIP_EPSILON: 0.2

To modify settings, edit these constants in train.py.

WHAT YOU'LL SEE:
- Each training example printed with full instruction and reference response
- PPO training progress for each epoch
- Trajectory sampling and reward computation
- Final evaluation results with code generation metrics

OUTPUT FILES:
- outputs/eval_results.jsonl: Evaluation results in JSONL format

================================================================================

