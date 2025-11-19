================================================================================
TINKER PPO FINE-TUNING PIPELINE - QUICK START GUIDE
================================================================================

SETUP:
1. Ensure TINKER_API_KEY is set:
   export TINKER_API_KEY='your_api_key_here'

2. Install dependencies:
   pip install -r requirements.txt

TRAINING:

Run PPO training (requires Tinker API):
   python train.py
   
   This will:
   - Load training examples from the configured dataset
   - Print all input data with instructions and reference responses
   - Run PPO training epochs
   - Sample trajectories per prompt each epoch
   - Compute rewards for generated outputs
   - Update the policy using PPO loss
   - Evaluate on the training examples
   - Save results to outputs/eval_results.jsonl

CONFIGURATION:
All settings are constants at the top of train.py:
- DATASET_NAME: Dataset to use for training
- NUM_EXAMPLES: Number of examples to train on
- BASE_MODEL: Base model identifier
- LEARNING_RATE: Learning rate for optimization
- NUM_PPO_EPOCHS: Number of PPO training epochs
- NUM_SAMPLES_PER_PROMPT: Number of samples per prompt
- MAX_GENERATION_TOKENS: Maximum tokens to generate
- PPO_CLIP_EPSILON: PPO clipping epsilon parameter

To modify settings, edit these constants in train.py.

WHAT YOU'LL SEE:
- Each training example printed with full instruction and reference response
- PPO training progress for each epoch
- Trajectory sampling and reward computation
- Final evaluation results with generation metrics

OUTPUT FILES:
- outputs/eval_results.jsonl: Evaluation results in JSONL format
- outputs/sampled_outputs.jsonl: Sampled model outputs
- outputs/extracted_results.jsonl: Extracted evaluation metrics

================================================================================

