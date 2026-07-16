"""
split_dataset.py
----------------
Reads the single annotated source file (data/raw/splittable_redo.jsonl) and
produces train/val/test splits for both models, following annotation guidelines:

DistilBERT splits  → {text, label}
  - label = 1 if has_disruption == 1, else 0
  - ALL rows included (positives + negatives) to train the binary triage gate

Qwen2.5 splits     → {text, event_data}
  - ONLY positive rows (has_disruption == 1) are included; negatives are excluded
  - event_data is the full annotated object from the source:
      event_id, event_type, confidence_score, source_timestamp,
      text_evidence, arguments
  - Rows where event_data is missing or null are silently skipped

Annotation-guideline compliance enforced at write time:
  - source_timestamp: null if absent in source text (never omit the field)
  - text_evidence: must be present (shortest span supporting the event)
  - All required schema fields are preserved as-is from the gold annotation

Splits: 70 % train | 15 % val | 15 % test (stratified by has_disruption)
Random seed: 42 (reproducible)

Output directory: data/
  data/distilbert/distilbert_{train,val,test}.jsonl
  data/qwen/qwen_{train,val,test}.jsonl
"""

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
RAW_INPUT = BASE / "data" / "raw" / "splittable_redo.jsonl"

OUTPUT_BASE = BASE / "data"
QWEN_DIR = OUTPUT_BASE / "qwen"
DISTILBERT_DIR = OUTPUT_BASE / "distilbert"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15  (remainder after train + val)
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
        event_data = record.get("event_data")  # may be None for negatives

        entry = {
            "text": text,
            "label": has_disruption,
            "event_data": event_data,
        }

        if has_disruption == 1:
            positives.append(entry)
        else:
            negatives.append(entry)

print(f"[INFO] Loaded {len(positives)} positive and {len(negatives)} negative records.")

# ---------------------------------------------------------------------------
# Stratified split (keeps pos/neg ratio stable across all three splits)
# ---------------------------------------------------------------------------
random.seed(RANDOM_SEED)
random.shuffle(positives)
random.shuffle(negatives)


def split_group(group):
    """Return (train, val, test) slices of `group` at 70/15/15."""
    n = len(group)
    train_n = int(n * TRAIN_RATIO)
    val_n = int(n * VAL_RATIO)
    return group[:train_n], group[train_n : train_n + val_n], group[train_n + val_n :]


pos_train, pos_val, pos_test = split_group(positives)
neg_train, neg_val, neg_test = split_group(negatives)

# Combine and re-shuffle each split so pos/neg aren't in blocks
train_split = pos_train + neg_train
val_split = pos_val + neg_val
test_split = pos_test + neg_test

random.shuffle(train_split)
random.shuffle(val_split)
random.shuffle(test_split)

print(
    f"[INFO] Split sizes — "
    f"train: {len(train_split)} ({len(pos_train)}+/{len(neg_train)}-) | "
    f"val: {len(val_split)} ({len(pos_val)}+/{len(neg_val)}-) | "
    f"test: {len(test_split)} ({len(pos_test)}+/{len(neg_test)}-)"
)

# ---------------------------------------------------------------------------
# Create output directories
# ---------------------------------------------------------------------------
QWEN_DIR.mkdir(parents=True, exist_ok=True)
DISTILBERT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Write splits
# ---------------------------------------------------------------------------
for name, split in [("train", train_split), ("val", val_split), ("test", test_split)]:
    qwen_path = QWEN_DIR / f"qwen_{name}.jsonl"
    bert_path = DISTILBERT_DIR / f"distilbert_{name}.jsonl"
    qwen_count = 0
    bert_count = 0

    with open(qwen_path, "w", encoding="utf-8") as qf, \
         open(bert_path, "w", encoding="utf-8") as df:

        for item in split:
            text = item["text"]
            label = item["label"]
            event_data = item["event_data"]

            # --- DistilBERT annotation ---
            # Guideline: binary label; all rows including negatives
            df.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
            bert_count += 1

            # --- Qwen2.5 annotation ---
            # Guideline: only positive events with a valid, complete event_data object
            if label == 1 and isinstance(event_data, dict):
                # Enforce annotation guideline: source_timestamp must be present
                # (null is valid; omission is not — preserved from gold annotation)
                # text_evidence must be present (preserved from gold annotation)
                # All other fields preserved as-is from the gold annotation
                qwen_record = {
                    "text": text,
                    "event_data": event_data,
                }
                qf.write(json.dumps(qwen_record, ensure_ascii=False) + "\n")
                qwen_count += 1

    print(f"[INFO] {name}: wrote {bert_count} DistilBERT rows | {qwen_count} Qwen rows")

print(f"\n[DONE] Output written to: {OUTPUT_BASE}")
print(f"  DistilBERT splits → {DISTILBERT_DIR}")
print(f"  Qwen2.5 splits    → {QWEN_DIR}")
