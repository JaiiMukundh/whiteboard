import sys
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

BASE = Path(__file__).resolve().parents[1]
BASE_MODEL = str(BASE / "models" / "qwen_lora" / "merged")

print("Loading base model configuration and weights (CPU)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)
    
# Calculate base model params
total_base_params = sum(p.numel() for p in model.parameters())

# Apply LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

peft_model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total_peft_params = sum(p.numel() for p in peft_model.parameters())

# Print statistics
print("=" * 60)
print("MODEL PARAMETER STATISTICS & ARCHITECTURE")
print("=" * 60)
print(f"Base Model (Qwen-2.5-1.5B): {BASE_MODEL}")
print(f"Base Model Parameters:     {total_base_params:,}")
print(f"Trainable LoRA Parameters:  {trainable_params:,}")
print(f"Total Parameters with LoRA: {total_peft_params:,}")
print(f"LoRA % of Base Model:      {trainable_params / total_base_params * 100:.4f}%")
print("-" * 60)
print("Adapter Architecture:")
print(f"  - Rank (r):             16")
print(f"  - Alpha (lora_alpha):   32")
print(f"  - Dropout:              0.1")
print(f"  - Target Modules:       {lora_config.target_modules}")
print("=" * 60)

