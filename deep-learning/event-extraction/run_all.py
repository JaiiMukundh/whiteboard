"""
run_all.py
----------
End-to-end pipeline runner. Executes every stage in dependency order:

  Stage 0  – Data preparation
  Stage 1  – Model training (DistilBERT triage + Qwen2.5-1.5B LoRA)
  Stage 2  – LoRA adapter merge
  Stage 3  – Inference (pipeline batch + zero-shot baseline)
  Stage 4  – Comparison model training + batch inference
  Stage 5  – Variance / reproducibility testing
  Stage 6  – Evaluation (extraction F1, schema validation, DistilBERT eval,
                          LoRA parameter stats, forgetting assessment)
  Stage 7  – Experience report metric generation + heatmap plots

Each script is run as a subprocess. If any step fails, execution stops
immediately with a non-zero exit code and prints the failing script name.

Usage:
    python run_all.py               # run every stage
    python run_all.py --from STAGE  # resume from a given stage number (0-7)
    python run_all.py --only STAGE  # run a single stage only
"""

import subprocess
import sys
import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Stage definitions
# Each stage is a list of (script_path, extra_args, description) tuples.
# Scripts within the same stage have no inter-dependencies and run in order.
# ---------------------------------------------------------------------------
STAGES = {
    0: {
        "name": "Data Preparation",
        "description": "Split unified raw master into train/val/test for both models",
        "scripts": [
            (BASE / "training" / "split_dataset.py", [],
             "Split splittable_redo.jsonl -> data/"),
        ],
    },
    1: {
        "name": "Model Training",
        "description": "Fine-tune DistilBERT triage classifier and Qwen2.5-1.5B LoRA extractor",
        "scripts": [
            (BASE / "training" / "train_distilbert.py", [],
             "Train DistilBERT binary triage classifier -> models/distilbert/"),
            (BASE / "training" / "train_qwen_lora.py", [],
             "Train Qwen2.5-1.5B LoRA adapter -> models/qwen_lora/"),
        ],
    },
    2: {
        "name": "LoRA Adapter Merge",
        "description": "Merge LoRA adapter weights into base model for faster inference",
        "scripts": [
            (BASE / "training" / "merge_lora.py", [],
             "Merge LoRA adapter -> models/qwen_lora/merged/"),
        ],
    },
    3: {
        "name": "Inference",
        "description": "Run pipeline batch inference and zero-shot baseline on the test set",
        "scripts": [
            (BASE / "inference" / "run_batch_inference.py", [],
             "Pipeline batch inference on qwen_test.jsonl -> reports/structured_outputs.jsonl"),
        ],
    },
    4: {
        "name": "Comparison Model Training and Inference",
        "description": "Train and evaluate TinyLlama-1.1B and SmolLM2-1.7B for benchmarking",
        "scripts": [
            (BASE / "training" / "run_model_comparisons.py", [],
             "Train + infer TinyLlama and SmolLM2 -> reports/comparison_metrics.csv"),
        ],
    },
    5: {
        "name": "Variance Testing",
        "description": "Re-train Qwen with seeds 42/1337/2026 to measure F1 variance",
        "scripts": [
            (BASE / "training" / "run_variance_tests.py", [],
             "3-seed variance run -> reports/variance_metrics.json"),
        ],
    },
    6: {
        "name": "Evaluation",
        "description": "F1/precision/recall, schema validity, DistilBERT metrics, LoRA stats, forgetting",
        "scripts": [
            (BASE / "evaluation" / "compare_extraction.py", [],
             "Extraction F1 / Precision / Recall -> reports/comparison_metrics.csv"),
            (BASE / "evaluation" / "compare_validation.py", [],
             "JSON schema validity check on all model outputs"),
            (BASE / "evaluation" / "evaluate_distilbert.py", [],
             "DistilBERT triage accuracy / F1 on test set"),
            (BASE / "evaluation" / "calculate_parameters.py", [],
             "LoRA parameter count and adapter size statistics"),
            (BASE / "evaluation" / "evaluate_forgetting.py", [],
             "Catastrophic forgetting assessment on general-purpose tasks"),
        ],
    },
    7: {
        "name": "Report Metrics and Visualizations",
        "description": "Aggregate metrics for the industry report and generate heatmaps",
        "scripts": [
            (BASE / "reports" / "calculate_metrics.py", [],
             "Compute all JSON metric files -> reports/*.json"),
            (BASE / "reports" / "generate_heatmaps.py", [],
             "Generate LoRA weight norm heatmaps -> reports/lora_heatmaps.png"),
        ],
    },
}


def run_stage(stage_id: int) -> None:
    stage = STAGES[stage_id]
    print(f"\n{'=' * 80}")
    print(f"  STAGE {stage_id}: {stage['name'].upper()}")
    print(f"  {stage['description']}")
    print(f"{'=' * 80}")

    for script_path, args, description in stage["scripts"]:
        print(f"\n>>> {script_path.relative_to(BASE)}  |  {description}")
        print("-" * 80)
        if not script_path.exists():
            print(f"[SKIP] Script not found: {script_path}")
            continue
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)] + args,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"\n[FAILED] {script_path.name} exited with code {exc.returncode}")
            sys.exit(exc.returncode)

    print(f"\n[OK] Stage {stage_id} complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline runner for the Supply Chain Event Extraction project."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from",
        dest="from_stage",
        type=int,
        metavar="STAGE",
        help="Resume execution from this stage number (0-7, inclusive).",
    )
    group.add_argument(
        "--only",
        dest="only_stage",
        type=int,
        metavar="STAGE",
        help="Run only this stage number (0-7).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  SUPPLY CHAIN EVENT EXTRACTION PIPELINE  --  FULL RUN")
    print("=" * 80)
    print(f"\nBase directory: {BASE}\n")
    print("Execution order:")
    for sid, stage in STAGES.items():
        print(f"  Stage {sid}: {stage['name']}")
    print()

    if args.only_stage is not None:
        if args.only_stage not in STAGES:
            print(f"[ERROR] Unknown stage: {args.only_stage}. Valid range: 0-{max(STAGES)}.")
            sys.exit(1)
        run_stage(args.only_stage)
    else:
        start = args.from_stage if args.from_stage is not None else 0
        if start not in STAGES:
            print(f"[ERROR] Unknown start stage: {start}. Valid range: 0-{max(STAGES)}.")
            sys.exit(1)
        for stage_id in sorted(STAGES.keys()):
            if stage_id < start:
                print(f"[SKIP] Stage {stage_id}: {STAGES[stage_id]['name']} (skipped by --from {start})")
                continue
            run_stage(stage_id)

    print("\n" + "=" * 80)
    print("  PIPELINE RUN COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
