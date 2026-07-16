import os
import json
import time
import subprocess
import numpy as np
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE / "reports"
OUTPUT_DIR = BASE / "reports" / "variance_metrics.json"

SEEDS = [42, 1337, 2026]
f1_scores = []

# Modify the seed in train_qwen_lora.py safely
train_script_path = BASE / "training" / "train_qwen_lora.py"
with open(train_script_path, "r") as f:
    original_train_code = f.read()

for seed in SEEDS:
    print(f"\n{'='*50}\nRunning Qwen LoRA Fine-Tuning with SEED = {seed}\n{'='*50}")
    
    # Inject seed
    modified_code = original_train_code.replace("SEED = 42", f"SEED = {seed}")
    
    # Also we should probably change the output dir so it doesn't just overwrite, or it's fine to overwrite.
    # We will just overwrite for the variance test.
    with open(train_script_path, "w") as f:
        f.write(modified_code)
        
    try:
        # Run training
        subprocess.run([sys.executable, str(train_script_path)], check=True)
        
        print(f"Merging LoRA adapter for SEED = {seed}...")
        subprocess.run([sys.executable, str(BASE / "training" / "merge_lora.py")], check=True)
        
        # Run evaluation directly instead of the full pipeline to save time, or run the pipeline?
        # The user's exact benchmark is in compare_extraction.py or run_batch_inference.py
        # Actually, running run_batch_inference.py and then compare_extraction.py yields the F1.
        print(f"Running inference for SEED = {seed}...")
        subprocess.run([sys.executable, str(BASE / "inference" / "run_batch_inference.py")], check=True)
        
        print(f"Running evaluation for SEED = {seed}...")
        subprocess.run([sys.executable, str(BASE / "evaluation" / "compare_extraction.py")], check=True)
        
        # Read the resulting F1 from comparison_metrics.csv
        with open(REPORTS_DIR / "comparison_metrics.csv", "r") as f:
            lines = f.readlines()
            # Last line should be Pipeline
            pipeline_line = lines[-1].split(",")
            pipeline_f1 = float(pipeline_line[3]) # Index 3 is F1 usually, check schema: model,precision,recall,f1,...
            f1_scores.append(pipeline_f1)
            print(f"SEED {seed} -> F1 Score: {pipeline_f1}")
            
    finally:
        # Restore original code
        with open(train_script_path, "w") as f:
            f.write(original_train_code)

mean_f1 = float(np.mean(f1_scores))
variance_f1 = float(np.var(f1_scores))

metrics = {
    "runs": len(SEEDS),
    "seeds_used": SEEDS,
    "f1_scores": f1_scores,
    "mean_f1": mean_f1,
    "variance_f1": variance_f1
}

with open(OUTPUT_DIR, "w") as f:
    json.dump(metrics, f, indent=2)

print("\nVARIANCE METRICS ACQUIRED:")
print(json.dumps(metrics, indent=2))
