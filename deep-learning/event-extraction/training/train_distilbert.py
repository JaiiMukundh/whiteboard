import json
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


BASE = Path(__file__).resolve().parents[1]
MODEL_NAME = "distilbert-base-uncased"
TRAIN_FILE = BASE / "data" / "distilbert" / "distilbert_train.jsonl"
VAL_FILE = BASE / "data" / "distilbert" / "distilbert_val.jsonl"
TEST_FILE = BASE / "data" / "distilbert" / "distilbert_test.jsonl"
OUTPUT_DIR = BASE / "models" / "distilbert"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-4
SEED = 42
DROPOUT = 0.2


torch.manual_seed(SEED)
np.random.seed(SEED)

train_rows = []
with open(TRAIN_FILE, "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            train_rows.append(json.loads(line))

val_rows = []
with open(VAL_FILE, "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            val_rows.append(json.loads(line))

test_rows = []
with open(TEST_FILE, "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            test_rows.append(json.loads(line))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token

config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
config.dropout = DROPOUT
config.attention_dropout = DROPOUT
if hasattr(config, "seq_classif_dropout"):
    config.seq_classif_dropout = DROPOUT

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
accelerator = Accelerator()
model, optimizer = accelerator.prepare(model, optimizer)


for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    batch_count = 0
    progress = tqdm(range(0, len(train_rows), BATCH_SIZE), desc="distilbert epoch %d" % (epoch + 1), leave=False)
    for start in progress:
        batch_rows = train_rows[start : start + BATCH_SIZE]
        texts = [row["text"] for row in batch_rows]
        labels = torch.tensor([row["label"] for row in batch_rows], dtype=torch.long, device=accelerator.device)
        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {key: value.to(accelerator.device) for key, value in enc.items()}
        outputs = model(**enc, labels=labels)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        batch_count += 1
        progress.set_postfix(loss=loss.item())

    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for start in range(0, len(val_rows), BATCH_SIZE):
            batch_rows = val_rows[start : start + BATCH_SIZE]
            texts = [row["text"] for row in batch_rows]
            labels = torch.tensor([row["label"] for row in batch_rows], dtype=torch.long, device=accelerator.device)
            enc = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {key: value.to(accelerator.device) for key, value in enc.items()}
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1)
            val_preds.extend(preds.detach().cpu().tolist())
            val_labels.extend(labels.detach().cpu().tolist())

    val_preds = np.asarray(val_preds)
    val_labels = np.asarray(val_labels)
    val_acc = float((val_preds == val_labels).mean()) if len(val_labels) else 0.0
    tp = int(((val_preds == 1) & (val_labels == 1)).sum())
    fp = int(((val_preds == 1) & (val_labels == 0)).sum())
    fn = int(((val_preds == 0) & (val_labels == 1)).sum())
    val_precision = tp / (tp + fp) if tp + fp else 0.0
    val_recall = tp / (tp + fn) if tp + fn else 0.0
    val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if val_precision + val_recall else 0.0
    print("epoch", epoch + 1, "loss", train_loss / max(1, batch_count))
    print("val acc", val_acc, "precision", val_precision, "recall", val_recall, "f1", val_f1)

test_preds = []
test_labels = []
model.eval()
with torch.no_grad():
    for start in range(0, len(test_rows), BATCH_SIZE):
        batch_rows = test_rows[start : start + BATCH_SIZE]
        texts = [row["text"] for row in batch_rows]
        labels = torch.tensor([row["label"] for row in batch_rows], dtype=torch.long, device=accelerator.device)
        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {key: value.to(accelerator.device) for key, value in enc.items()}
        logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1)
        test_preds.extend(preds.detach().cpu().tolist())
        test_labels.extend(labels.detach().cpu().tolist())

test_preds = np.asarray(test_preds)
test_labels = np.asarray(test_labels)
test_acc = float((test_preds == test_labels).mean()) if len(test_labels) else 0.0
tp = int(((test_preds == 1) & (test_labels == 1)).sum())
fp = int(((test_preds == 1) & (test_labels == 0)).sum())
fn = int(((test_preds == 0) & (test_labels == 1)).sum())
test_precision = tp / (tp + fp) if tp + fp else 0.0
test_recall = tp / (tp + fn) if tp + fn else 0.0
test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall) if test_precision + test_recall else 0.0
print("test acc", test_acc, "precision", test_precision, "recall", test_recall, "f1", test_f1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
