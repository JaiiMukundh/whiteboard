from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = BASE / "models" / "qwen_lora"
OUTPUT_DIR = ADAPTER_DIR / "merged"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
merged = model.merge_and_unload()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
merged.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)
