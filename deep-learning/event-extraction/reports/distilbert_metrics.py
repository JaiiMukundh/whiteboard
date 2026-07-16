import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Paths
BASE_DIR = Path("/home/jaiimukundha/Desktop/Triage_Pipeline_Final/Triage_Pipeline")
MODEL_DIR = BASE_DIR / "models" / "distilbert"
OUTPUT_FILE = BASE_DIR / "reports" / "triage_metrics.json"

def evaluate_split(model, tokenizer, device, split_name, file_path):
    print(f"\nEvaluating {split_name} split...")
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                
    total = len(data)
    if total == 0:
        return None
        
    print(f"Loaded {total} samples for {split_name}.")

    tp, fp, tn, fn = 0, 0, 0, 0

    for i, item in enumerate(data):
        text = item["text"]
        true_label = int(item["label"])
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        inputs.pop("token_type_ids", None)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            pred_label = int(torch.argmax(logits, dim=-1).item())
        
        if pred_label == 1 and true_label == 1:
            tp += 1
        elif pred_label == 1 and true_label == 0:
            fp += 1
        elif pred_label == 0 and true_label == 1:
            fn += 1
        elif pred_label == 0 and true_label == 0:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    print(f"[{split_name.upper()}] Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    return {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn
        }
    }

def calculate_all_metrics():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    splits = {
        "train": BASE_DIR / "data" / "distilbert" / "distilbert_train.jsonl",
        "val": BASE_DIR / "data" / "distilbert" / "distilbert_val.jsonl",
        "test": BASE_DIR / "data" / "distilbert" / "distilbert_test.jsonl",
    }
    
    all_results = {}
    for split_name, file_path in splits.items():
        if file_path.exists():
            res = evaluate_split(model, tokenizer, device, split_name, file_path)
            if res:
                all_results[split_name] = res

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {OUTPUT_FILE}")

if __name__ == "__main__":
    calculate_all_metrics()
