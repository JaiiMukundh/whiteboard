import json
import subprocess
import time
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE / "reports"
MODELS_TO_TEST = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
]
TRAIN_SCRIPT = BASE / "training" / "train_qwen_lora.py"
BATCH_INFERENCE = BASE / "inference" / "run_batch_inference.py"
COMPARE_SCRIPT = BASE / "evaluation" / "compare_extraction.py"

with open(TRAIN_SCRIPT, "r") as f:
    orig_train_code = f.read()

with open(BATCH_INFERENCE, "r") as f:
    orig_inference_code = f.read()

def run_model(model_name):
    print(f"\n{'*'*60}\nEvaluating Model: {model_name}\n{'*'*60}")
    safe_name = model_name.split("/")[-1]
    
    # 1. Update train_qwen_lora.py
    # Change BASE_MODEL and OUTPUT_DIR
    mod_train = orig_train_code.replace('"Qwen/Qwen2.5-1.5B-Instruct"', f'"{model_name}"')
    mod_train = mod_train.replace('"qwen_lora"', f'"{safe_name}_lora"')
    with open(TRAIN_SCRIPT, "w") as f:
        f.write(mod_train)
        
    # 2. Update run_batch_inference.py
    # Change BASE_MODEL, ADAPTER_PATH, and output files
    mod_inf = orig_inference_code.replace('"Qwen/Qwen2.5-1.5B-Instruct"', f'"{model_name}"')
    mod_inf = mod_inf.replace('"qwen_lora"', f'"{safe_name}_lora"')
    mod_inf = mod_inf.replace('"baseline_structured_outputs.jsonl"', f'"{safe_name}_baseline.jsonl"')
    mod_inf = mod_inf.replace('"structured_outputs.jsonl"', f'"{safe_name}_pipeline.jsonl"')
    
    # Skip baseline inference
    mod_inf = mod_inf.replace('print(">>> Running Baseline Inference...")', 'print(">>> Skipping Baseline Inference..."); import sys; sys.exit(0)')
    
    with open(BATCH_INFERENCE, "w") as f:
        f.write(mod_inf)
        
    try:
        lora_out = BASE / "models" / f"{safe_name}_lora"
        if not (lora_out / "adapter_config.json").exists():
            print(f"[{safe_name}] Starting LoRA Training (~45m)...")
            subprocess.run(["python", str(TRAIN_SCRIPT)], check=True)
        else:
            print(f"[{safe_name}] LoRA adapter already exists. Skipping training!")
            
        print(f"[{safe_name}] Merging LoRA adapter into base model weights...")
        subprocess.run(["python", str(BASE / "training" / "merge_adapters.py"), model_name, str(lora_out)], check=True)
        
        print(f"[{safe_name}] Running Batch Inference...")
        subprocess.run(["python", str(BATCH_INFERENCE)], check=True)
        
    finally:
        with open(TRAIN_SCRIPT, "w") as f:
            f.write(orig_train_code)
        with open(BATCH_INFERENCE, "w") as f:
            f.write(orig_inference_code)

for model in MODELS_TO_TEST:
    run_model(model)
    
print("\nAll model comparisons have finished running.")
