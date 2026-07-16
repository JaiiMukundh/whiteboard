import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Configuration
BASE = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_FILE = BASE / "data" / "qwen" / "qwen_train.jsonl"
VAL_FILE = BASE / "data" / "qwen" / "qwen_val.jsonl"
OUTPUT_DIR = BASE / "models" / "qwen_lora"
BEST_OUTPUT_DIR = OUTPUT_DIR / "best"

MAX_LENGTH = 1024
BATCH_SIZE = 1
EPOCHS = 3
LR = 1e-4
SEED = 42

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

torch.manual_seed(SEED)
np.random.seed(SEED)


# Restored exactly as originally provided
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


def compute_token_f1(pred_val, gold_val):
    if pred_val is None and gold_val is None:
        return 1.0
    if pred_val is None or gold_val is None:
        return 0.0

    pred_str = " ".join(map(str, pred_val)) if isinstance(pred_val, list) else str(pred_val)
    gold_str = " ".join(map(str, gold_val)) if isinstance(gold_val, list) else str(gold_val)

    pred_str, gold_str = pred_str.strip().lower(), gold_str.strip().lower()
    if pred_str == gold_str:
        return 1.0
    if not pred_str or not gold_str:
        return 0.0

    pred_tokens = re.findall(r"\w+", pred_str)
    gold_tokens = re.findall(r"\w+", gold_str)
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    pred_counter, gold_counter = Counter(pred_tokens), Counter(gold_tokens)
    intersection = sum((pred_counter & gold_counter).values())
    if intersection == 0:
        return 0.0

    precision = intersection / len(pred_tokens)
    recall = intersection / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(pred_val, gold_val):
    if pred_val is None and gold_val is None:
        return 1.0
    if pred_val is None or gold_val is None:
        return 0.0
    return 1.0 if str(pred_val).strip().lower() == str(gold_val).strip().lower() else 0.0


def normalize_prediction(pred):
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            return None
    if isinstance(pred, list) and pred:
        pred = pred[0]
    if isinstance(pred, dict):
        return {k: v for k, v in pred.items() if k not in {"event_id", "triage_label", "chunk_text"}}
    return None


def get_flat_dict(obj):
    if not isinstance(obj, dict):
        return {}
    flat = {k: obj.get(k) for k in ["event_type", "source_timestamp", "text_evidence"]}
    args = obj.get("arguments", {})
    if isinstance(args, dict):
        flat.update(args)
    return flat


def compute_field_score(key, pred_val, gold_val, fuzzy=False):
    if pred_val is None and gold_val is None:
        return 1.0
    if pred_val is None or gold_val is None:
        return 0.0
    if fuzzy:
        return compute_token_f1(pred_val, gold_val)
    if key == "event_type":
        return compute_exact_match(pred_val, gold_val)
    if key == "source_timestamp":
        return 1.0 if parse_to_date(pred_val) == parse_to_date(gold_val) else 0.0
    return compute_token_f1(pred_val, gold_val)


def evaluate_fields(pred_dict, gold_dict, keys, fuzzy=False):
    if not pred_dict and not gold_dict:
        return 1.0, 1.0, 1.0
    if not pred_dict or not gold_dict:
        return 0.0, 0.0, 0.0

    pred_flat = get_flat_dict(pred_dict)
    gold_flat = get_flat_dict(gold_dict)
    eval_keys = (set(pred_flat.keys()) | set(gold_flat.keys())) & keys

    scores = {k: compute_field_score(k, pred_flat.get(k), gold_flat.get(k), fuzzy) for k in eval_keys}

    pred_keys = [k for k in pred_flat if k in keys]
    precision = sum(scores[k] for k in pred_keys) / len(pred_keys) if pred_keys else 1.0

    gold_keys = [k for k in gold_flat if k in keys]
    recall = sum(scores[k] for k in gold_keys) / len(gold_keys) if gold_keys else 1.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def evaluate_top_level(pred_dict, gold_dict):
    return evaluate_fields(pred_dict, gold_dict, {"event_type", "source_timestamp", "text_evidence"}, fuzzy=False)


def evaluate_fuzzy_other_fields(pred_dict, gold_dict):
    pred_flat = get_flat_dict(pred_dict)
    gold_flat = get_flat_dict(gold_dict)
    other_keys = (set(pred_flat.keys()) | set(gold_flat.keys())) - {"event_type", "source_timestamp", "text_evidence"}
    return evaluate_fields(pred_dict, gold_dict, other_keys, fuzzy=True)


# Accelerator Initialization
accelerator = Accelerator()
device = accelerator.device
print(f"Accelerate utilizing device: {device}")


# Load Data
def load_valid_rows(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("event_data") is not None:
                    rows.append(row)
    return rows


train_rows = load_valid_rows(TRAIN_FILE)
val_rows = load_valid_rows(VAL_FILE)

# Model and Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
    device_map={"": device},
)
base_model.config.use_cache = False

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=TARGET_MODULES,
)
model = get_peft_model(base_model, lora)
model.enable_input_require_grads()

generation_config = GenerationConfig(
    max_new_tokens=256,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model.generation_config = generation_config

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model, optimizer = accelerator.prepare(model, optimizer)

SYSTEM_PROMPT = (
    "You are an expert supply chain disruption event extractor. "
    "Extract a single event matching the provided JSON schema.\n\n"
    "Guidelines:\n"
    "- Extract only information that is explicitly stated in the source text.\n"
    "- Do not infer, estimate, or hallucinate any facts.\n"
    "- If a value is not explicitly mentioned, return null for that field.\n"
    "- Extract the date of the event and format it as an ISO 8601 string (e.g., YYYY-MM-DDT00:00:00Z) in the source_timestamp field. If only a year is mentioned, use the first day of that year (e.g., YYYY-01-01T00:00:00Z). If only a month and year are mentioned, use the first day of that month (e.g., YYYY-MM-01T00:00:00Z). If no date/time is mentioned, return null.\n"
    "- Preserve the original meaning of the text.\n"
    "- Use the smallest span of text that directly supports the extracted event as the text_evidence.\n"
    "- Classify an event as SupplierInsolvency ONLY if there is explicit mention of legal or financial failure, such as bankruptcy, Chapter 11, liquidation, or receivership. Do not classify temporary operational shutdowns, resource depletion (e.g., running out of fuel), or physical facility halts as SupplierInsolvency."
)

# Encode training dataset
train_encoded = []
for row in train_rows:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["text"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    messages.append({"role": "assistant", "content": json.dumps(row["event_data"], ensure_ascii=False)})
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids[:MAX_LENGTH]

    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
    labels = labels[: len(full_ids)]
    train_encoded.append(
        {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    )

# Pre-format validation evaluation templates to avoid redundant in-loop template applications
val_prompts = []
for row in val_rows:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["text"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    val_prompts.append((prompt, row["event_data"]))


def save_checkpoint(target_dir, metrics=None, filename="metrics.json"):
    if accelerator.is_main_process:
        target_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)
        if metrics:
            (target_dir / filename).write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")


best_top_level_f1 = -float("inf")

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    train_loss = 0.0
    batch_count = 0

    progress_bar = tqdm(range(0, len(train_encoded), BATCH_SIZE), desc=f"Train Epoch {epoch + 1}")

    for start_idx in progress_bar:
        batch = train_encoded[start_idx : start_idx + BATCH_SIZE]
        if not batch:
            continue

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(device)
        labels = torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch], batch_first=True, padding_value=-100
        ).to(device)

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()

        train_loss += loss.item()
        batch_count += 1
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = train_loss / max(1, batch_count)

    # --- EVALUATION PHASE ---
    model.eval()
    val_top_f1_scores = []
    val_other_f1_scores = []

    with torch.inference_mode():
        for prompt, gold_event in tqdm(val_prompts, desc=f"Eval Epoch {epoch + 1}", leave=False):
            prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            prompt_ids = prompt_inputs["input_ids"].to(device)
            attention_mask = prompt_inputs["attention_mask"].to(device)

            generated_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )

            generated_text = tokenizer.decode(generated_ids[0][prompt_ids.shape[1] :], skip_special_tokens=True)
            predicted_event = normalize_prediction(generated_text)

            _, _, top_f1 = evaluate_top_level(predicted_event, gold_event)
            _, _, other_f1 = evaluate_fuzzy_other_fields(predicted_event, gold_event)

            val_top_f1_scores.append(top_f1)
            val_other_f1_scores.append(other_f1)

    avg_top_f1 = float(np.mean(val_top_f1_scores)) if val_top_f1_scores else 0.0
    avg_other_f1 = float(np.mean(val_other_f1_scores)) if val_other_f1_scores else 0.0

    epoch_metrics = {
        "epoch": epoch + 1,
        "top_level_f1": avg_top_f1,
        "other_fields_f1": avg_other_f1,
        "train_loss": avg_train_loss,
    }

    save_checkpoint(OUTPUT_DIR, metrics=epoch_metrics, filename=f"epoch_{epoch + 1}_metrics.json")

    if avg_top_f1 > best_top_level_f1:
        best_top_level_f1 = avg_top_f1
        best_metrics = {"epoch": epoch + 1, "top_level_f1": avg_top_f1, "other_fields_f1": avg_other_f1}
        save_checkpoint(BEST_OUTPUT_DIR, metrics=best_metrics, filename="best_metrics.json")
        print(
            f"New best checkpoint saved to {BEST_OUTPUT_DIR} "
            f"(epoch {epoch + 1}, top_level_f1={avg_top_f1:.4f}, other_fields_f1={avg_other_f1:.4f})"
        )

    print(
        f"\n[Epoch {epoch + 1} Summary] "
        f"Train Loss: {avg_train_loss:.4f} | Top Level F1: {avg_top_f1:.4f} | "
        f"Other Fields F1: {avg_other_f1:.4f}"
    )

# Save final checkpoint
save_checkpoint(OUTPUT_DIR)
print(f"Process complete. Final model files saved in {OUTPUT_DIR}")
print(f"Best checkpoint saved in {BEST_OUTPUT_DIR} with top_level_f1={best_top_level_f1:.4f}")