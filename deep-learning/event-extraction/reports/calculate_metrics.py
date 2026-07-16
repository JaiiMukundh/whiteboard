import json
import os
import re
import sys
import time
import csv
from pathlib import Path
import numpy as np
import torch

# Setup paths
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
RAW_FILE = DATA_DIR / "raw" / "splittable_redo.jsonl"
DISTILBERT_DIR = DATA_DIR / "distilbert"
QWEN_DIR = DATA_DIR / "qwen"
REPORTS_DIR = BASE / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
CSV_FILE = BASE / "reports" / "comparison_metrics.csv"

print("=" * 80)
print("EXPERIENCE REPORT METRICS GENERATION SUITE")
print("=" * 80)

# ==============================================================================
# 1. READ CSV METRICS
# ==============================================================================
print("\n[1/4] Loading actual extraction metrics...")
f1_metrics = {}
if CSV_FILE.exists():
    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model"]
            f1_metrics[model_name] = {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "schema_validity": float(row["schema_validity"])
            }
else:
    raise FileNotFoundError(f"Missing {CSV_FILE}")

with open(REPORTS_DIR / "extraction_metrics.json", "w") as f:
    json.dump(f1_metrics, f, indent=2)

# ==============================================================================
# 2. DATASET QUANTIFICATION
# ==============================================================================
print("\n[2/4] Quantifying Datasets...")
raw_samples = []
if RAW_FILE.exists():
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_samples.append(json.loads(line))

total_raw = len(raw_samples)
pos_raw = sum(1 for s in raw_samples if s.get("has_disruption", 0) == 1)
neg_raw = total_raw - pos_raw

raw_class_dist = {}
for s in raw_samples:
    if s.get("has_disruption", 0) == 1 and "event_data" in s and s["event_data"]:
        etype = s["event_data"].get("event_type", "Unknown")
        raw_class_dist[etype] = raw_class_dist.get(etype, 0) + 1

def count_splits(split_name):
    dist_file = DISTILBERT_DIR / f"distilbert_{split_name}.jsonl"
    qwen_file = QWEN_DIR / f"qwen_{split_name}.jsonl"
    
    dist_len, positives, negatives = 0, 0, 0
    class_dist = {}
    
    if dist_file.exists():
        with open(dist_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dist_len += 1
                    if json.loads(line).get("label") == 1:
                        positives += 1
                    else:
                        negatives += 1
                        
    if qwen_file.exists():
        with open(qwen_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    event = json.loads(line).get("event_data")
                    if event:
                        etype = event.get("event_type", "Unknown")
                        class_dist[etype] = class_dist.get(etype, 0) + 1
    return dist_len, positives, negatives, class_dist

train_len, train_pos, train_neg, train_dist = count_splits("train")
val_len, val_pos, val_neg, val_dist = count_splits("val")
test_len, test_pos, test_neg, test_dist = count_splits("test")

dataset_metrics = {
    "raw_total": total_raw,
    "raw_positives": pos_raw,
    "raw_negatives": neg_raw,
    "raw_class_distribution": raw_class_dist,
    "splits": {
        "train": {"total": train_len, "positives": train_pos, "negatives": train_neg, "class_distribution": train_dist},
        "val": {"total": val_len, "positives": val_pos, "negatives": val_neg, "class_distribution": val_dist},
        "test": {"total": test_len, "positives": test_pos, "negatives": test_neg, "class_distribution": test_dist}
    }
}
with open(REPORTS_DIR / "dataset_metrics.json", "w") as f:
    json.dump(dataset_metrics, f, indent=2)

# ==============================================================================
# 3. BENCHMARKS & PERPLEXITY
# ==============================================================================
print("\n[3/4] Running Load, Inference, and Perplexity Benchmarks...")
import psutil
def get_current_ram():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

ram_start = get_current_ram()
t0 = time.time()
base_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
t_base_load = time.time() - t0
ram_base = get_current_ram() - ram_start

t0 = time.time()
triage_tok = AutoTokenizer.from_pretrained(BASE / "models" / "distilbert")
triage_model = AutoModelForSequenceClassification.from_pretrained(BASE / "models" / "distilbert")
t_triage_load = time.time() - t0

gen_text = "The history of science is the study of the development of science, including both the natural and social sciences."
sc_text = "A massive natural disaster disrupted semiconductor fabrication plants, halting silicon wafer manufacturing and creating severe bottlenecks in the automotive supply chain."
device = next(base_model.parameters()).device

def get_perplexity(model, text):
    inputs = base_tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        loss = model(inputs["input_ids"], labels=inputs["input_ids"]).loss
    return np.exp(loss.item())

# Base Model Perplexity
ppl_base_gen = get_perplexity(base_model, gen_text)
ppl_base_sc = get_perplexity(base_model, sc_text)

# Base Inference Latency
test_prompt = f"Extract supply chain event matching schema:\nText: {sc_text}\nJSON Output:"
inputs_inf = base_tok(test_prompt, return_tensors="pt").to(device)
t0 = time.time()
with torch.no_grad():
    gen_base = base_model.generate(**inputs_inf, max_new_tokens=256, do_sample=False, pad_token_id=base_tok.eos_token_id)
lat_base = time.time() - t0
toks_base = len(gen_base[0]) - len(inputs_inf["input_ids"][0])

# Load PEFT adapter and merge for accurate latency
from peft import PeftModel
t0 = time.time()
peft_model = PeftModel.from_pretrained(base_model, BASE / "models" / "qwen_lora")
t_peft_load = time.time() - t0
ram_peft = get_current_ram() - (ram_start + ram_base)
peak_ram = get_current_ram()

peft_model = peft_model.merge_and_unload()

# LoRA Model Perplexity
ppl_lora_gen = get_perplexity(peft_model, gen_text)
ppl_lora_sc = get_perplexity(peft_model, sc_text)

# LoRA Inference Latency (First Pass)
t0 = time.time()
with torch.no_grad():
    gen_first = peft_model.generate(**inputs_inf, max_new_tokens=256, do_sample=False, pad_token_id=base_tok.eos_token_id)
lat_first = time.time() - t0
toks_first = len(gen_first[0]) - len(inputs_inf["input_ids"][0])

# Simulate Correction Loop Pass
prompt_correction = test_prompt + "\nCorrecting hallucinations..."
inputs_corr = base_tok(prompt_correction, return_tensors="pt").to(device)
t0 = time.time()
with torch.no_grad():
    gen_corr = peft_model.generate(**inputs_corr, max_new_tokens=256, do_sample=False, pad_token_id=base_tok.eos_token_id)
lat_corr = time.time() - t0

lat_total = lat_first + lat_corr
toks_total = toks_first + (len(gen_corr[0]) - len(inputs_corr["input_ids"][0]))

perplexity_metrics = {
    "base_model": {"general_perplexity": ppl_base_gen, "supply_chain_perplexity": ppl_base_sc},
    "lora_model": {"general_perplexity": ppl_lora_gen, "supply_chain_perplexity": ppl_lora_sc}
}
with open(REPORTS_DIR / "perplexity_metrics.json", "w") as f:
    json.dump(perplexity_metrics, f, indent=2)

system_metrics = {
    "loading": {
        "base_model_loading_seconds": t_base_load,
        "adapter_loading_seconds": t_peft_load,
        "triage_loading_seconds": t_triage_load,
        "base_model_ram_mb": ram_base,
        "adapter_ram_mb": ram_peft,
        "peak_total_ram_mb": peak_ram
    },
    "inference": {
        "baseline": {"latency_seconds": lat_base, "tokens_generated": int(toks_base), "tokens_per_second": toks_base/lat_base},
        "pipeline_lora": {
            "first_pass_latency": lat_first,
            "correction_pass_latency": lat_corr,
            "total_latency_seconds": lat_total,
            "tokens_generated": int(toks_total),
            "effective_tokens_per_second": toks_total/lat_total
        }
    }
}
with open(REPORTS_DIR / "system_performance_metrics.json", "w") as f:
    json.dump(system_metrics, f, indent=2)

# ==============================================================================
# 4. LoRA WEIGHT NORMS
# ==============================================================================
print("\n[4/4] Calculating LoRA Adapter Norms...")
from safetensors.torch import load_file
adapter_file = BASE / "models" / "qwen_lora" / "adapter_model.safetensors"
weights = load_file(adapter_file)

scaling = 32 / 16 # alpha / rank
pattern_attn = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(\w+)\.lora_A\.weight")
pattern_mlp = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.mlp\.(\w+)\.lora_A\.weight")

module_norms = {m: [] for m in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
layers_data = {}

for key in weights.keys():
    match_attn = pattern_attn.match(key)
    match_mlp = pattern_mlp.match(key)
    layer_idx = None
    module_name = None
    is_attn = True
    
    if match_attn:
        layer_idx, module_name = int(match_attn.group(1)), match_attn.group(2)
    elif match_mlp:
        layer_idx, module_name = int(match_mlp.group(1)), match_mlp.group(2)
        is_attn = False
        
    if layer_idx is not None:
        p_path = f"base_model.model.model.layers.{layer_idx}." + ("self_attn" if is_attn else "mlp") + f".{module_name}"
        key_A, key_B = f"{p_path}.lora_A.weight", f"{p_path}.lora_B.weight"
        
        if key_A in weights and key_B in weights:
            w_A, w_B = weights[key_A].to(torch.float32), weights[key_B].to(torch.float32)
            delta_W = torch.matmul(w_B, w_A) * scaling
            n_delta = torch.norm(delta_W).item()
            
            if layer_idx not in layers_data: layers_data[layer_idx] = []
            layers_data[layer_idx].append(n_delta)
            module_norms[module_name].append(n_delta)

mean_module_norms = {k: float(np.mean(v)) if v else 0.0 for k, v in module_norms.items()}
layer_total_norms = {layer: float(np.mean(norms)) for layer, norms in layers_data.items()}
sorted_layers = sorted(layer_total_norms.items(), key=lambda x: x[1], reverse=True)

lora_metrics = {
    "mean_norms_per_module": mean_module_norms,
    "sorted_layers_by_magnitude": sorted_layers,
    "layer_total_norms": layer_total_norms
}
with open(REPORTS_DIR / "lora_weight_metrics.json", "w") as f:
    json.dump(lora_metrics, f, indent=2)

print("\n" + "=" * 80)
print("ALL METRICS GENERATED AND SAVED TO JSON FILES SUCCESSFULLY!")
print(f"Saved reports under: {REPORTS_DIR}")
print("=" * 80)
