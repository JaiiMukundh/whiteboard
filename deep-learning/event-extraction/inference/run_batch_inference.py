import json
import re
import sys
import uuid
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

# Set paths
BASE = Path(__file__).resolve().parents[1]
TEST_FILE = BASE / "data" / "qwen" / "qwen_test.jsonl"
REPORTS_DIR = BASE / "reports"
PIPELINE_OUT = REPORTS_DIR / "structured_outputs.jsonl"
BASELINE_OUT = REPORTS_DIR / "baseline_structured_outputs.jsonl"
SCHEMA_FILE = BASE / "schemas" / "extraction_schema.json"

# Models paths
DISTILBERT_DIR = BASE / "models" / "distilbert"
QWEN_MERGED_DIR = BASE / "models" / "qwen_lora" / "merged"
BASELINE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Replicate pipeline helpers
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

def run_pipeline(test_rows):
    print(">>> Initializing Pipeline Models...")
    from outlines import from_transformers, json_schema
    
    # Load DistilBERT Triage Classifier
    triage_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    triage_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triage_model.to(device)
    triage_model.eval()
    
    # Load Qwen Merged Model
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MERGED_DIR, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MERGED_DIR,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    generator = from_transformers(qwen_model, qwen_tokenizer)
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    
    print(">>> Running Pipeline Inference...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(PIPELINE_OUT, "w", encoding="utf-8") as out_file:
        for idx, row in enumerate(tqdm(test_rows, desc="Pipeline inference")):
            text = row["text"]
            
            # Chunking (260 words per chunk to prevent multi-event confusion)
            words = text.split()
            chunks = [" ".join(words[i:i + 260]) for i in range(0, len(words), 260)]
            
            if not chunks:
                out_file.write("null\n")
                continue
            
            extracted_events = []
            for chunk in chunks:
                inputs = triage_tokenizer(chunk, return_tensors="pt", truncation=True)
                inputs.pop("token_type_ids", None)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = triage_model(**inputs).logits
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
                                "Output: {\"event_type\": \"FacilityHalt\", \"source_timestamp\": \"[SOURCE_TIMESTAMP]\", \"text_evidence\": \"halt of operations at [PORT_LOCATION] due to a severe labor strike starting on 2023-11-05\", \"arguments\": {\"operator\": \"[LOGISTICS_COMPANY]\", \"facility_location\": \"[PORT_LOCATION]\", \"disruption_type\": \"Labor_Dispute\", \"start_date\": \"2023-11-05T00:00:00Z\", \"expected_restart_date\": null}}\n\n"
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
            
            # Save a single event dictionary per test line (selecting first real event over NoEvent)
            real_events = [e for e in extracted_events if e.get("event_type") != "NoEvent"]
            if real_events:
                out_file.write(json.dumps(real_events[0], ensure_ascii=False) + "\n")
            elif extracted_events:
                out_file.write(json.dumps(extracted_events[0], ensure_ascii=False) + "\n")
            else:
                out_file.write("null\n")
    print(f">>> Pipeline outputs saved to {PIPELINE_OUT}")

def run_baseline(test_rows):
    print(">>> Initializing Baseline Model...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASELINE_MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    schema_str = json.dumps(schema, indent=2)
    device = next(model.parameters()).device
    
    print(">>> Running Baseline Inference...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(BASELINE_OUT, "w", encoding="utf-8") as out_file:
        for idx, row in enumerate(tqdm(test_rows, desc="Baseline inference")):
            text = row["text"]
            
            # Baseline chunking logic from extract_baseline_test.py
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
                
            if not chunks:
                out_file.write("null\n")
                continue
                
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
                
                # Parse JSON
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
            
            # Save a single event dictionary per test line (selecting first real event over NoEvent)
            real_events = [e for e in extracted_events if e.get("event_type") != "NoEvent"]
            if real_events:
                out_file.write(json.dumps(real_events[0], ensure_ascii=False) + "\n")
            elif extracted_events:
                out_file.write(json.dumps(extracted_events[0], ensure_ascii=False) + "\n")
            else:
                out_file.write("null\n")
    print(f">>> Baseline outputs saved to {BASELINE_OUT}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference on qwen_test.jsonl")
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["pipeline", "baseline", "both"],
        help="Which model(s) to run batch inference for (default: both)",
    )
    args = parser.parse_args()
    
    # Load test rows
    if not TEST_FILE.exists():
        print(f"Error: test file {TEST_FILE} not found.")
        sys.exit(1)
        
    test_rows = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_rows.append(json.loads(line))
                
    print(f"Loaded {len(test_rows)} test cases from {TEST_FILE}")
    
    if args.model in ["pipeline", "both"]:
        run_pipeline(test_rows)
        
    if args.model in ["baseline", "both"]:
        run_baseline(test_rows)
        
    print("Done running batch inference.")
