import json
import sys
from pathlib import Path
from jsonschema import Draft7Validator

BASE = Path(__file__).resolve().parents[1]
SCHEMA_FILE = BASE / "schemas" / "extraction_schema.json"
TEST_FILE = BASE / "data" / "qwen" / "qwen_test.jsonl"
PIPELINE_FILE = BASE / "reports" /"structured_outputs.jsonl"
BASELINE_FILE = BASE / "reports" /"baseline_structured_outputs.jsonl"

schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
validator = Draft7Validator(schema)

# Read test inputs
test_rows = []
with open(TEST_FILE, "r", encoding="utf-8") as h:
    for line in h:
        if line.strip():
            test_rows.append(json.loads(line))

# Read pipeline outputs
if not PIPELINE_FILE.exists():
    print(f"Error: Fine-tuned pipeline output file not found at {PIPELINE_FILE}. Please run triage_pipeline.py first.")
    sys.exit(1)
    
with open(PIPELINE_FILE, "r", encoding="utf-8") as h:
    pipeline_rows = [json.loads(line) for line in h if line.strip()]

# Read baseline outputs
if not BASELINE_FILE.exists():
    print(f"Error: Baseline output file not found at {BASELINE_FILE}. Please run extract_baseline_test.py first.")
    sys.exit(1)

with open(BASELINE_FILE, "r", encoding="utf-8") as h:
    baseline_rows = [json.loads(line) for line in h if line.strip()]

print("=" * 80)
print("DETAILED SCHEMA VALIDATION & COMPARISON REPORT")
print("=" * 80)

pipeline_passed = 0
baseline_passed = 0

comparison_details = []

for idx, gold_row in enumerate(test_rows):
    text = gold_row["text"]
    
    # Get predictions
    pipeline_pred = pipeline_rows[idx] if idx < len(pipeline_rows) else None
    baseline_pred = baseline_rows[idx] if idx < len(baseline_rows) else None

    # Clean up pipeline_pred for validation (it has event_id injected, which isn't in the schema)
    pipeline_val_input = None
    if pipeline_pred is not None and isinstance(pipeline_pred, dict):
        pipeline_val_input = dict(pipeline_pred)
        pipeline_val_input.pop("event_id", None)

    # Validate pipeline
    pipeline_errors = []
    if pipeline_val_input is not None:
        errors = list(validator.iter_errors(pipeline_val_input))
        if not errors:
            pipeline_passed += 1
        else:
            pipeline_errors = [e.message for e in errors]
    else:
        pipeline_errors = ["No prediction generated (null)"]

    # Validate baseline
    baseline_errors = []
    if baseline_pred is not None:
        errors = list(validator.iter_errors(baseline_pred))
        if not errors:
            baseline_passed += 1
        else:
            baseline_errors = [e.message for e in errors]
    else:
        baseline_errors = ["No prediction generated (null)"]

    comparison_details.append({
        "idx": idx,
        "text": text,
        "pipeline": {
            "pred": pipeline_pred,
            "valid": len(pipeline_errors) == 0,
            "errors": pipeline_errors
        },
        "baseline": {
            "pred": baseline_pred,
            "valid": len(baseline_errors) == 0,
            "errors": baseline_errors
        }
    })

# Print Summary Table
print(f"\nQuantitative Validation Summary:")
print(f"{'Model':<35} | {'Total Rows':<12} | {'Passed Validation':<18} | {'Pass Rate':<10}")
print("-" * 85)
print(f"{'Pipeline (Fine-Tuned + Outlines)':<35} | {len(test_rows):<12} | {pipeline_passed:<18} | {pipeline_passed/len(test_rows)*100:.1f}%")
print(f"{'Baseline (Qwen Instruct - Raw)':<35} | {len(test_rows):<12} | {baseline_passed:<18} | {baseline_passed/len(test_rows)*100:.1f}%")
print("-" * 85)

# Print a few prominent comparison examples
print("\nQualitative Validation Examples:")
print("=" * 80)

# We want to show examples where the baseline failed but pipeline passed
examples_shown = 0
for item in comparison_details:
    if not item["baseline"]["valid"] and item["pipeline"]["valid"]:
        examples_shown += 1
        print(f"\nExample {examples_shown}:")
        print(f"Input Text: {item['text'][:150]}...")
        print("-" * 40)
        print("Baseline Prediction (FAILED VALIDATION):")
        print(json.dumps(item["baseline"]["pred"], indent=2, ensure_ascii=False))
        print(f"Validation Errors: {item['baseline']['errors']}")
        print("-" * 40)
        print("Pipeline Prediction (PASSED VALIDATION):")
        print(json.dumps(item["pipeline"]["pred"], indent=2, ensure_ascii=False))
        print("=" * 80)
        if examples_shown >= 3:
            break

