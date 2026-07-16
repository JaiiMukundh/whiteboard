import json
import re
from pathlib import Path
from collections import Counter
from datetime import datetime, date

import matplotlib.pyplot as plt
import pandas as pd
from jsonschema import Draft7Validator
import numpy as np

BASE = Path(__file__).resolve().parents[1]
SCHEMA_FILE = BASE / "schemas" / "extraction_schema.json"
PIPELINE_FILE = BASE / "reports" /"structured_outputs.jsonl"
BASELINE_FILE = BASE / "reports" /"baseline_structured_outputs.jsonl"
TEST_FILE = BASE / "data" / "qwen" / "qwen_test.jsonl"
PLOT_FILE = BASE / "reports" / "comparison_metrics.png"
CSV_FILE = BASE / "reports" /"comparison_metrics.csv"
MODEL_LABELS = {
    "pipeline": "Qwen2.5-1.5B (LoRA)",
    "tinyllama": "TinyLlama-1.1B (LoRA)",
    "smollm2": "SmolLM2-1.7B (LoRA)",
    "baseline": "Baseline (Qwen Instruct)"
}
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "schema_validity": "Schema Validity",
}

schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
validator = Draft7Validator(schema)


# Date parsing helper function (kept exactly as requested)
def parse_to_date(val):
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str or val_str.lower() == "null":
        return None
    if "t" in val_str.lower():
        parts = re.split(r"[tT]", val_str)
        val_str = parts[0]
    
    try:
        dt = datetime.strptime(val_str, "%Y-%m-%d")
        return dt.date()
    except ValueError:
        pass

    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", val_str)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    match_ym = re.search(r"(\d{4})-(\d{2})", val_str)
    if match_ym:
        try:
            return date(int(match_ym.group(1)), int(match_ym.group(2)), 1)
        except ValueError:
            pass

    return val_str


# Token-level F1 helper function
def compute_token_f1(pred_val, gold_val):
    if pred_val is None and gold_val is None:
        return 1.0
    if pred_val is None or gold_val is None:
        return 0.0

    if isinstance(pred_val, list):
        pred_val = " ".join([str(item) for item in pred_val])
    if isinstance(gold_val, list):
        gold_val = " ".join([str(item) for item in gold_val])

    pred_str = str(pred_val).strip().lower()
    gold_str = str(gold_val).strip().lower()

    if not pred_str and not gold_str:
        return 1.0
    if not pred_str or not gold_str:
        return 0.0

    pred_tokens = re.findall(r'\w+', pred_str)
    gold_tokens = re.findall(r'\w+', gold_str)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    intersection = sum((pred_counter & gold_counter).values())
    if intersection == 0:
        return 0.0

    precision = intersection / len(pred_tokens)
    recall = intersection / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


# Exact match helper function
def compute_exact_match(pred_val, gold_val):
    if pred_val is None and gold_val is None:
        return 1.0
    if pred_val is None or gold_val is None:
        return 0.0
    
    pred_str = str(pred_val).strip().lower()
    gold_str = str(gold_val).strip().lower()
    return 1.0 if pred_str == gold_str else 0.0


# Normalize a prediction dictionary
def normalize_prediction(pred):
    if pred is None:
        return None
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            return None
    if isinstance(pred, list):
        pred = pred[0] if pred else None
    if isinstance(pred, dict):
        pred = dict(pred)
        pred.pop("event_id", None)
        pred.pop("triage_label", None)
        pred.pop("chunk_text", None)
        return pred
    return None


# Extract all keys and values from flat representation
def get_flat_dict(obj):
    if obj is None:
        return {}
    flat = {}
    for field in ["event_type", "source_timestamp", "text_evidence"]:
        flat[field] = obj.get(field)
    
    args = obj.get("arguments", {})
    if isinstance(args, dict):
        for k, v in args.items():
            flat[k] = v
    return flat


# Helper to compute metrics on key subsets
def evaluate_fields_subset(pred_flat, gold_flat, scores, keys):
    pred_keys = [k for k in pred_flat if k in keys]
    pred_scores = [scores[k] for k in pred_keys] if pred_keys else [1.0]
    precision = sum(pred_scores) / len(pred_scores)

    gold_keys = [k for k in gold_flat if k in keys]
    gold_scores = [scores[k] for k in gold_keys] if gold_keys else [1.0]
    recall = sum(gold_scores) / len(gold_keys) if gold_keys else [1.0]

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


# Evaluate sample
def evaluate_sample(pred_dict, gold_dict):
    """
    Returns (precision, recall, f1, top_level_f1, other_fields_f1) for this sample
    using the exact scoring rules from the training/validation script.
    """
    if not pred_dict and not gold_dict:
        return 1.0, 1.0, 1.0, 1.0, 1.0
    if not pred_dict:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if not gold_dict:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    pred_flat = get_flat_dict(pred_dict)
    gold_flat = get_flat_dict(gold_dict)

    scores = {}
    all_keys = set(pred_flat.keys()) | set(gold_flat.keys())
    
    for key in all_keys:
        p_val = pred_flat.get(key)
        g_val = gold_flat.get(key)
        
        if key == "event_type":
            scores[key] = compute_exact_match(p_val, g_val)
        elif key == "source_timestamp":
            p_date = parse_to_date(p_val)
            g_date = parse_to_date(g_val)
            if p_date is None and g_date is None:
                scores[key] = 1.0
            elif p_date is None or g_date is None:
                scores[key] = 0.0
            else:
                scores[key] = 1.0 if p_date == g_date else 0.0
        else:
            # Everything else (text_evidence and arguments) gets evaluated with fuzzy token-level F1
            scores[key] = compute_token_f1(p_val, g_val)

    # 1. Overall Metrics
    pred_keys = list(pred_flat.keys())
    pred_scores = [scores[k] for k in pred_keys] if pred_keys else [1.0]
    precision = sum(pred_scores) / len(pred_scores)

    gold_keys = list(gold_flat.keys())
    gold_scores = [scores[k] for k in gold_keys] if gold_keys else [1.0]
    recall = sum(gold_scores) / len(gold_scores)

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    # 2. Top-Level Metrics (event_type, source_timestamp, text_evidence)
    top_keys = {"event_type", "source_timestamp", "text_evidence"}
    _, _, top_f1 = evaluate_fields_subset(pred_flat, gold_flat, scores, top_keys)

    # 3. Other Fields Metrics (all other argument fields)
    other_keys = all_keys - top_keys
    _, _, other_f1 = evaluate_fields_subset(pred_flat, gold_flat, scores, other_keys)

    return precision, recall, f1, top_f1, other_f1


# Load test rows
test_rows = []
with open(TEST_FILE, "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            test_rows.append(json.loads(line))

# Load predictions
results = []
files_to_eval = [
    ("pipeline", PIPELINE_FILE),
    ("tinyllama", BASE / "reports" / "TinyLlama-1.1B-Chat-v1.0_pipeline.jsonl"),
    ("smollm2", BASE / "reports" / "SmolLM2-1.7B-Instruct_pipeline.jsonl")
]
if BASELINE_FILE.exists():
    files_to_eval.append(("baseline", BASELINE_FILE))

for name, path in files_to_eval:
    with open(path, "r", encoding="utf-8") as handle:
        pred_rows = [json.loads(line) for line in handle if line.strip()]
        
    for index, predicted in enumerate(pred_rows):
        gold_row = test_rows[index] if index < len(test_rows) else None
        gold_event = gold_row.get("event_data") if gold_row else None
        
        norm_pred = normalize_prediction(predicted)
        valid = False
        if norm_pred is not None:
            valid = not any(validator.iter_errors(norm_pred))
            
        p, r, f1, top_f1, other_f1 = evaluate_sample(norm_pred, gold_event)
        
        results.append({
            "model": MODEL_LABELS[name],
            "schema_valid": int(valid),
            "precision": p,
            "recall": r,
            "f1": f1,
            "top_level_f1": top_f1,
            "other_fields_f1": other_f1
        })

# Compute summary statistics
df = pd.DataFrame(results)
summary = df.groupby("model").agg(
    precision=("precision", "mean"),
    recall=("recall", "mean"),
    f1=("f1", "mean"),
    top_level_f1=("top_level_f1", "mean"),
    other_fields_f1=("other_fields_f1", "mean"),
    schema_validity=("schema_valid", "mean"),
)

# Save to CSV
summary[["precision", "recall", "f1", "top_level_f1", "other_fields_f1", "schema_validity"]].to_csv(CSV_FILE)

# Plotting using matplotlib.pyplot directly
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

models = summary.index.tolist()
x = np.arange(len(models))

# 1. Overall Extraction Quality Chart (Precision, Recall, F1)
width_3 = 0.25
rects1 = axes[0].bar(x - width_3, summary["precision"], width_3, label="Precision", color="#2a9d8f")
rects2 = axes[0].bar(x, summary["recall"], width_3, label="Recall", color="#e9c46a")
rects3 = axes[0].bar(x + width_3, summary["f1"], width_3, label="F1", color="#457b9d")

axes[0].set_title("Overall Extraction Quality", fontsize=11, fontweight="bold", pad=10)
axes[0].set_ylabel("Score", fontsize=10)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, fontsize=10)
axes[0].set_ylim(0, 1.1)
axes[0].legend(loc="upper right", frameon=True)
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

# Add value labels for Subplot 1
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        axes[0].annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

# 2. F1 Breakdown Chart (Top-Level F1, Other Fields F1)
width_2 = 0.3
rects_tl = axes[1].bar(x - width_2/2, summary["top_level_f1"], width_2, label="Top-Level F1", color="#1d3557")
rects_of = axes[1].bar(x + width_2/2, summary["other_fields_f1"], width_2, label="Other Fields F1", color="#e63946")

axes[1].set_title("F1 Performance Breakdown", fontsize=11, fontweight="bold", pad=10)
axes[1].set_ylabel("Score", fontsize=10)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, fontsize=10)
axes[1].set_ylim(0, 1.1)
axes[1].legend(loc="upper right", frameon=True)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

# Add value labels for Subplot 2
for rects in [rects_tl, rects_of]:
    for rect in rects:
        height = rect.get_height()
        axes[1].annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

# 3. Schema Validity Chart
rects_valid = axes[2].bar(x, summary["schema_validity"], width_2 * 1.5, color="#8d99ae")
axes[2].set_title("Schema Validity Rate", fontsize=11, fontweight="bold", pad=10)
axes[2].set_ylabel("Rate", fontsize=10)
axes[2].set_xticks(x)
axes[2].set_xticklabels(models, fontsize=10)
axes[2].set_ylim(0, 1.1)
axes[2].grid(axis="y", linestyle="--", alpha=0.5)

# Add value labels for Subplot 3
for rect in rects_valid:
    height = rect.get_height()
    axes[2].annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Remove top and right borders from all plots
for axis in axes:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

fig.suptitle("Baseline vs Pipeline Performance Comparison", fontsize=14, fontweight="bold")
plt.savefig(PLOT_FILE, dpi=200, bbox_inches="tight")
plt.close(fig)

print(summary[["precision", "recall", "f1", "top_level_f1", "other_fields_f1", "schema_validity"]])
print("Saved:", CSV_FILE)
print("Saved:", PLOT_FILE)