"""
split_dataset.py
----------------
Reads the single annotated source file (data/raw/splittable_redo.jsonl) and
produces train/val/test splits exactly matching the original dataset distributions.

DistilBERT splits  → {text, label}
  - 125 positives, 71 negatives (Total: 196)
  - Train (137): 83 pos, 54 neg
  - Val (29): 24 pos, 5 neg
  - Test (30): 18 pos, 12 neg

Qwen2.5 splits     → {text, event_data}
  - 175 positives exactly, perfectly stratified by event_type.
  - Train (115): 23-24 per class
  - Val (30): 6 per class
  - Test (30): 6 per class

Output directory is set to 'splittage_trial' to preserve original datasets.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
RAW_INPUT = BASE / "data" / "raw" / "splittable_redo.jsonl"

# TEST MODE: Outputting to splittage_trial so we DO NOT touch original datasets
OUTPUT_BASE = BASE / "splittage_trial"
QWEN_DIR = OUTPUT_BASE / "qwen"
DISTILBERT_DIR = OUTPUT_BASE / "distilbert"

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Validate input
# ---------------------------------------------------------------------------
if not RAW_INPUT.exists():
    raise SystemExit(f"[ERROR] Raw input file not found: {RAW_INPUT}")

# ---------------------------------------------------------------------------
# Load and validate source records
# ---------------------------------------------------------------------------
positives = []
negatives = []

with open(RAW_INPUT, "r", encoding="utf-8") as fh:
    for lineno, raw_line in enumerate(fh, start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping malformed JSON on line {lineno}: {exc}")
            continue

        text = record.get("text", "").strip()
        if not text:
            print(f"[WARN] Skipping empty text on line {lineno}")
            continue

        has_disruption = int(record.get("has_disruption", 0))
        event_data = record.get("event_data")

        entry = {
            "text": text,
            "label": has_disruption,
            "event_data": event_data,
        }

        if has_disruption == 1:
            positives.append(entry)
        else:
            negatives.append(entry)

print(f"[INFO] Loaded {len(positives)} positive and {len(negatives)} negative records from raw data.")

# ---------------------------------------------------------------------------
# Prepare Splitting (Deterministic sorting + seed)
# ---------------------------------------------------------------------------
# Sort to guarantee exact same random selections across different OS environments
positives.sort(key=lambda x: x["text"])
negatives.sort(key=lambda x: x["text"])

random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Qwen Splits (Stratified by event_type, Exactly 175 total rows)
# ---------------------------------------------------------------------------
# Desired distribution to match README:
# FacilityHalt (36):       train=24, val=6, test=6
# ShipmentDelay (36):      train=24, val=6, test=6
# SupplierInsolvency (35): train=23, val=6, test=6
# TariffChange (35):       train=23, val=6, test=6
# ForceMajeure (33):       train=21, val=6, test=6

qwen_target_counts = {
    "FacilityHalt": (24, 6, 6),
    "ShipmentDelay": (24, 6, 6),
    "SupplierInsolvency": (23, 6, 6),
    "TariffChange": (23, 6, 6),
    "ForceMajeure": (21, 6, 6),
}

by_event_type = defaultdict(list)
for p in positives:
    if isinstance(p["event_data"], dict):
        by_event_type[p["event_data"].get("event_type")].append(p)

for k in by_event_type:
    random.shuffle(by_event_type[k])

qwen_train, qwen_val, qwen_test = [], [], []

for evt_type, (t_n, v_n, te_n) in qwen_target_counts.items():
    group = by_event_type[evt_type]
    qwen_train.extend(group[:t_n])
    qwen_val.extend(group[t_n : t_n + v_n])
    qwen_test.extend(group[t_n + v_n : t_n + v_n + te_n])

random.shuffle(qwen_train)
random.shuffle(qwen_val)
random.shuffle(qwen_test)

# ---------------------------------------------------------------------------
# 2. DistilBERT Splits (Subsampled to 125 pos, 71 neg to maintain 64% ratio)
# ---------------------------------------------------------------------------
# Positives: train=83, val=24, test=18  (Total 125)
# Negatives: train=54, val=5, test=12   (Total 71)

# Shuffle the master pools first
random.shuffle(positives)
random.shuffle(negatives)

dist_pos_train = positives[:83]
dist_pos_val = positives[83:83+24]
dist_pos_test = positives[83+24:83+24+18]

dist_neg_train = negatives[:54]
dist_neg_val = negatives[54:54+5]
dist_neg_test = negatives[54+5:54+5+12]

distilbert_train = dist_pos_train + dist_neg_train
distilbert_val = dist_pos_val + dist_neg_val
distilbert_test = dist_pos_test + dist_neg_test

random.shuffle(distilbert_train)
random.shuffle(distilbert_val)
random.shuffle(distilbert_test)

# ---------------------------------------------------------------------------
# Create output directories
# ---------------------------------------------------------------------------
QWEN_DIR.mkdir(parents=True, exist_ok=True)
DISTILBERT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Write splits
# ---------------------------------------------------------------------------
def write_qwen(split, name):
    path = QWEN_DIR / f"qwen_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in split:
            record = {"text": item["text"], "event_data": item["event_data"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def write_distilbert(split, name):
    path = DISTILBERT_DIR / f"distilbert_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in split:
            record = {"text": item["text"], "label": item["label"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

write_qwen(qwen_train, "train")
write_qwen(qwen_val, "val")
write_qwen(qwen_test, "test")

write_distilbert(distilbert_train, "train")
write_distilbert(distilbert_val, "val")
write_distilbert(distilbert_test, "test")

print(f"[INFO] Split sizes — ")
print(f"  DistilBERT train: {len(distilbert_train)} | val: {len(distilbert_val)} | test: {len(distilbert_test)}")
print(f"  Qwen train:       {len(qwen_train)} | val: {len(qwen_val)} | test: {len(qwen_test)}")
print(f"\n[DONE] Output safely written to Trial Directory: {OUTPUT_BASE}")
print(f"  (Original datasets in data/ were NOT touched)")
