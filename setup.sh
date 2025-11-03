#!/bin/bash

echo "Installing Tinker Fine-Tuning Pipeline..."

pip install -q tinker datasets transformers torch pyyaml numpy

echo ""
echo "Setup complete!"
echo ""
echo "To run the pipeline:"
echo "1. Set your API key: export TINKER_API_KEY='your_key_here'"
echo "2. Run: python run_pipeline.py"
echo ""
echo "Or run steps individually:"
echo "  - Training (PPO): python train.py"
echo "  - Evaluation: python evaluation.py"

