import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models" / "distilbert"
TEST_FILE = BASE / "data" / "distilbert" / "distilbert_test.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# Force CPU so it doesn't interrupt the GPU jobs
device = torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

texts = []
y_true = []
with open(TEST_FILE, "r") as f:
    for line in f:
        data = json.loads(line)
        texts.append(data["text"])
        y_true.append(data["label"])

# Batch inference
batch_size = 16
y_pred = []

model.eval()
with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"DistilBERT Triage Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
