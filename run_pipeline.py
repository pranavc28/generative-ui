import os
import subprocess
import sys

TINKER_API_KEY = os.environ.get("TINKER_API_KEY")

if not TINKER_API_KEY:
    print("Error: TINKER_API_KEY environment variable not set")
    print("Set it with: export TINKER_API_KEY='your_api_key_here'")
    sys.exit(1)

print("=" * 60)
print("TINKER FINE-TUNING PIPELINE")
print("=" * 60)

print("\n[1/2] Training and evaluating model...")
result = subprocess.run([sys.executable, "train.py"], capture_output=False)
if result.returncode != 0:
    print("Training failed!")
    sys.exit(1)

print("\n[2/2] Displaying detailed evaluation metrics...")
result = subprocess.run([sys.executable, "evaluation.py"], capture_output=False)
if result.returncode != 0:
    print("Detailed evaluation display failed!")
    sys.exit(1)

print("\n" + "=" * 60)
print("Pipeline completed successfully!")
print("=" * 60)

