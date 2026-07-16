import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = "Qwen/Qwen2.5-1.5B-Instruct"
SCHEMA_FILE = BASE / "schemas" / "extraction_schema.json"

if len(sys.argv) < 2:
    raise SystemExit("usage: python inference/extract_baseline_test.py \"text_to_extract\"")

args = sys.argv[1:]
text = " ".join(args)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
schema_str = json.dumps(schema, indent=2)

# 1. Text Chunking (Linear, no helper functions)
paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
chunks = []
current_chunk = []
current_len = 0
for para in paragraphs:
    para_len = len(tokenizer.tokenize(para))
    if current_len + para_len > 1500 and current_chunk:
        chunks.append("\n\n".join(current_chunk))
        current_chunk = [para]
        current_len = para_len
    else:
        current_chunk.append(para)
        current_len += para_len
if current_chunk:
    chunks.append("\n\n".join(current_chunk))

# If no text chunks were parsed, exit
if not chunks:
    print("null")
    raise SystemExit(0)

# 2. Sequential Extraction Loop
extracted_events = []
for chunk in chunks:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert supply chain disruption event extractor. "
                "Extract a single event matching the provided JSON schema. "
                "Respond ONLY with a JSON object. Do not wrap the JSON in markdown code blocks or add any explanatory text.\n\n"
                "Guidelines:\n"
                "- Extract only information that is explicitly stated in the source text.\n"
                "- Do not infer, estimate, or hallucinate any facts.\n"
                "- If a value is not explicitly mentioned, return null for that field.\n"
                "- Extract the date of the event and format it as an ISO 8601 string (e.g., YYYY-MM-DDT00:00:00Z) in the source_timestamp field. If only a year is mentioned, use the first day of that year (e.g., YYYY-01-01T00:00:00Z). If only a month and year are mentioned, use the first day of that month (e.g., YYYY-MM-01T00:00:00Z). If no date/time is mentioned, return null.\n"
                "- Preserve the original meaning of the text.\n"
                "- Use the smallest span of text that directly supports the extracted event as the text_evidence.\n"
                "- Classify an event as SupplierInsolvency ONLY if there is explicit mention of legal or financial failure, such as bankruptcy, Chapter 11, liquidation, or receivership. Do not classify temporary operational shutdowns, resource depletion (e.g., running out of fuel), or physical facility halts as SupplierInsolvency.\n\n"
                f"JSON Schema:\n{schema_str}"
            ),
        },
        {
            "role": "user",
            "content": chunk,
        },
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (
            "You are an expert supply chain disruption event extractor. "
            "Extract a single event matching the provided JSON schema. "
            "Respond ONLY with a JSON object.\n\n"
            f"JSON Schema:\n{schema_str}\n\n"
            "Text:\n"
            + chunk.strip()
            + "\n"
        )

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    with torch.no_grad():
        generated = model.generate(
            **{key: value.to(device) for key, value in inputs.items()},
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    import re
    output = None
    try:
        output = json.loads(completion)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(\{.*?\})\s*```", completion, re.DOTALL)
        if match:
            try:
                output = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        if output is None:
            match = re.search(r"(\{.*\})", completion, re.DOTALL)
            if match:
                try:
                    output = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

    if output is not None:
        extracted_events.append(output)

# Print output
if not extracted_events:
    print("null")
elif len(extracted_events) == 1:
    print(json.dumps(extracted_events[0], ensure_ascii=False, indent=2))
else:
    print(json.dumps(extracted_events, ensure_ascii=False, indent=2))
