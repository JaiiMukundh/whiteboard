import json
import sys
from pathlib import Path
import uuid

import torch
from outlines import from_transformers, json_schema
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM


BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models" / "distilbert"
QWEN_DIR = BASE / "models" / "qwen_lora" / "merged"
SCHEMA_FILE = BASE / "schemas" / "extraction_schema.json"

def is_numeric_or_percentage(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if isinstance(value, str):
        cleaned = value.replace("%", "").replace("$", "").replace(",", "").replace(".", "").strip()
        return cleaned.isdigit()
    return False


def filter_hallucinations(output, chunk):
    if not isinstance(output, dict):
        return output
    
    import string
    chunk_lower = chunk.lower()
    for char in string.punctuation:
        chunk_lower = chunk_lower.replace(char, " ")
    
    def is_present(value):
        if value is None:
            return True
        if is_numeric_or_percentage(value):
            return True
        val_str = str(value).strip().lower()
        if not val_str or val_str == "null":
            return True
        
        # Abstract placeholders should be filtered to null
        if val_str.startswith("[") and val_str.endswith("]"):
            return False
            
        val_words = [w.strip(string.punctuation) for w in val_str.split()]
        val_words = [w for w in val_words if len(w) > 2]
        if not val_words:
            return True
        return any(w in chunk_lower for w in val_words)

    args = output.get("arguments", {})
    if isinstance(args, dict):
        for k, v in list(args.items()):
            if k in ["tariff_action", "disruption_type", "legal_action"]:
                continue
            if v is not None and not is_present(v):
                args[k] = None

    for field in ["text_evidence"]:
        val = output.get(field)
        if val is not None and not is_present(val):
            output[field] = None
            
    return output


def get_hallucinated_fields(output, chunk):
    if not isinstance(output, dict):
        return []
    
    import string
    chunk_lower = chunk.lower()
    for char in string.punctuation:
        chunk_lower = chunk_lower.replace(char, " ")
    
    def is_present(value):
        if value is None:
            return True
        if is_numeric_or_percentage(value):
            return True
        val_str = str(value).strip().lower()
        if not val_str or val_str == "null":
            return True
        
        # Abstract placeholders are hallucinations
        if val_str.startswith("[") and val_str.endswith("]"):
            return False
            
        val_words = [w.strip(string.punctuation) for w in val_str.split()]
        val_words = [w for w in val_words if len(w) > 2]
        if not val_words:
            return True
        return any(w in chunk_lower for w in val_words)

    hallucinated = []
    
    # Check arguments
    args = output.get("arguments", {})
    if isinstance(args, dict):
        for k, v in args.items():
            if k in ["tariff_action", "disruption_type", "legal_action"]:
                continue
            if v is not None and not is_present(v):
                hallucinated.append(k)
                
    # Check top-level fields
    for field in ["text_evidence"]:
        val = output.get(field)
        if val is not None and not is_present(val):
            hallucinated.append(field)
            
    return hallucinated


if len(sys.argv) < 2:
    raise SystemExit("usage: python inference/triage_pipeline.py \"text_to_extract\"")

args = sys.argv[1:]
text = " ".join(args)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# 1. Text Chunking (260 words per chunk to prevent multi-event confusion)
words = text.split()
chunks = [" ".join(words[i:i + 260]) for i in range(0, len(words), 260)]

# If no text chunks were parsed, exit
if not chunks:
    print("null")
    raise SystemExit(0)

# Load Qwen model and resources only if we have chunks to process
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True, fix_mistral_regex=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_DIR,
    trust_remote_code=True,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
generator = from_transformers(qwen_model, qwen_tokenizer)
schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))

# 2. Sequential Extraction Loop
extracted_events = []
for chunk in chunks:
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        logits = model(**inputs).logits
        label = int(torch.argmax(logits, dim=-1).item())

    if label == 1:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert supply chain disruption event extractor. "
                    "Extract a single event matching the provided JSON schema.\n\n"
                    "Guidelines:\n"
                    "- Extract only information that is explicitly stated in the source text. NEVER use external world knowledge to fill in blank fields. If a value is not explicitly written in the text, you MUST return null.\n"
                    "- Do not copy any names or placeholders from the examples in the prompt unless they appear word-for-word in the source text.\n"
                    "- Do not infer, estimate, or hallucinate any facts.\n"
                    "- If a value is not explicitly mentioned, return null for that field.\n"
                    "- IF EXPLICIT TIME OR MONTH OR DATE OR EXACT NUMBER FOR YEAR IS NOT PRESENT set the source_timestamp to null.\n"
                    "- Extract the date of the event and format it as an ISO 8601 string.\n"
                    "- Preserve the original meaning of the text.\n"
                    "- Use the smallest span of text that directly supports the extracted event as the text_evidence.\n"
                    "- Classify an event as SupplierInsolvency ONLY if there is explicit mention of legal or financial failure, such as declaring bankruptcy, filing for Chapter 11 reorganization, insolvency, liquidation, defaulting on debts, receivership, or experiencing severe liquidity crises leading to restructuring.\n\n"
                    "Example 1:\n"
                    "Text: In other news, global logistics giant [LOGISTICS_COMPANY] announced a 3-day halt of operations at [PORT_LOCATION] due to a severe labor strike starting on [SOURCE_TIMESTAMP].\n"
                    "Output: {\"event_type\": \"ShipmentDelay\", \"source_timestamp\": \"[SOURCE_TIMESTAMP]\", \"text_evidence\": \"halt of operations at [PORT_LOCATION] due to a severe labor strike starting on 2023-11-05\", \"arguments\": {\"operator\": \"[LOGISTICS_COMPANY]\", \"facility_location\": \"[PORT_LOCATION]\", \"disruption_type\": \"Labor_Dispute\", \"start_date\": \"2023-11-05T00:00:00Z\", \"expected_restart_date\": null}}\n\n"
                    "Example 2:\n"
                    "Text: The CEO mentioned they are launching a new product line next week.\n"
                    "Output: {\"event_type\": \"NoEvent\"}"
                ),
            },
            {
                "role": "user",
                "content": chunk,
            },
        ]

        if hasattr(qwen_tokenizer, "apply_chat_template"):
            prompt = qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                "You are an expert supply chain disruption event extractor. "
                "Extract a single event matching the provided JSON schema.\n\n"
                "Text:\n"
                + chunk.strip()
                + "\n"
            )

        output = generator(
            prompt,
            json_schema(schema),
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=qwen_tokenizer.eos_token_id,
        )

        if isinstance(output, str):
            output = json.loads(output)
        if isinstance(output, list):
            output = output[0]

        # Check for hallucinations
        hallucinated_fields = get_hallucinated_fields(output, chunk)

        # Self-correction loop: if hallucinations are detected, pass back to model for a second pass
        if hallucinated_fields:
            correction_instruction = (
                f"In your previous response, the following fields contain hallucinations "
                f"because they are not present in the source text: {', '.join(hallucinated_fields)}. "
                f"Please regenerate the JSON payload and set these hallucinated fields to null, "
                f"or correct them using ONLY the facts explicitly mentioned in the source text."
            )

            correction_messages = messages + [
                {
                    "role": "assistant",
                    "content": json.dumps(output)
                },
                {
                    "role": "user",
                    "content": correction_instruction
                }
            ]

            if hasattr(qwen_tokenizer, "apply_chat_template"):
                prompt = qwen_tokenizer.apply_chat_template(
                    correction_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                output = generator(
                    prompt,
                    json_schema(schema),
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=qwen_tokenizer.eos_token_id,
                )

                if isinstance(output, str):
                    output = json.loads(output)
                if isinstance(output, list):
                    output = output[0]

        # Hard filter fallback to guarantee compliance
        output = filter_hallucinations(output, chunk)
        output["event_id"] = "EVT-" + uuid.uuid4().hex[:8].upper()
        extracted_events.append(output)
    else:
        extracted_events.append({
            "event_type": "NoEvent",
            "event_id": "EVT-" + uuid.uuid4().hex[:8].upper()
        })

# Print output
if not extracted_events:
    print("null")
elif len(extracted_events) == 1:
    print(json.dumps(extracted_events[0], ensure_ascii=False, indent=2))
else:
    print(json.dumps(extracted_events, ensure_ascii=False, indent=2))