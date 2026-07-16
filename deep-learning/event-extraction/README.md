# Supply Chain Event Extraction Pipeline

> **Internship Project · Industry Experience Report Reference Implementation**

A resource-efficient, two-stage pipeline for detecting and extracting structured supply chain disruption events from unstructured text, using small language models (SLMs) with LoRA fine-tuning.

---

## Overview

Enterprise organizations monitor large volumes of news and operational documents to detect supply chain risks. Sending every document to a large generative LLM is computationally expensive. This pipeline solves that with a **two-stage triage architecture**:

1. **Stage 1 — Triage (DistilBERT, 66M params):** Fast binary classifier that filters out non-event documents in ~15 ms per chunk on a T4 GPU (with a CPU loading time of ~71 ms).
2. **Stage 2 — Extraction (Qwen2.5-1.5B + LoRA + Outlines):** Structured JSON event extraction, only triggered for documents that pass triage (~14.3 s per chunk for the extraction pass on CPU).

### Supported Event Types

| Event Type | Captures |
|---|---|
| `FacilityHalt` | Physical production stoppages (fires, strikes, disasters, cyberattacks) |
| `ShipmentDelay` | Transit and carrier delays |
| `SupplierInsolvency` | Bankruptcy, liquidation, and financial restructuring |
| `TariffChange` | Customs duty and trade policy changes |
| `ForceMajeure` | Legal force majeure declarations |

---

## Repository Structure

```
Triage_Pipeline/
│
├── run_all.py                     # End-to-end pipeline runner (stages 0-7)
├── requirements.txt               # Python dependencies
├── wiki_links.txt                 # References for event taxonomy validation
│
├── data/
│   ├── distilbert/          # Binary triage classification datasets
│   │   ├── distilbert_base.jsonl   (196 rows: 125 pos / 71 neg)
│   │   ├── distilbert_train.jsonl  (137 rows: 83 pos / 54 neg)
│   │   ├── distilbert_val.jsonl    (29 rows)
│   │   └── distilbert_test.jsonl   (30 rows)
│   │
│   ├── qwen/                # Structured extraction datasets (positives only)
│   │   ├── qwen_base.jsonl         (175 rows, balanced ~35/class)
│   │   ├── qwen_train.jsonl        (115 rows)
│   │   ├── qwen_val.jsonl          (30 rows, 6/class)
│   │   └── qwen_test.jsonl         (30 rows, 6/class)
│   │
│   ├── raw/
│   │   └── splittable_redo.jsonl   # Master unified dataset (351 rows: 280 pos / 71 neg)
│
├── schemas/
│   └── extraction_schema.json   # JSON Schema governing all extractions
│
├── training/
│   ├── split_dataset.py           # Reads splittable_redo.jsonl → outputs to data/
│   ├── train_distilbert.py        # Fine-tunes DistilBERT for binary triage
│   ├── train_qwen_lora.py         # Fine-tunes Qwen2.5-1.5B with LoRA
│   ├── merge_lora.py              # Merges LoRA adapter into base model
│   ├── run_model_comparisons.py   # Trains/evaluates comparison models (SmolLM2, TinyLlama)
│   └── run_variance_tests.py      # Re-trains Qwen with seeds 42/1337/2026 for variance assessment
│
├── models/
│   ├── distilbert/          # Fine-tuned DistilBERT weights (gitignored: .safetensors)
│   ├── qwen_lora/           # LoRA adapter + merged model (gitignored: .safetensors)
│   ├── SmolLM2-1.7B-Instruct_lora/  # Comparison model (gitignored: .safetensors)
│   └── TinyLlama-1.1B-Chat-v1.0_lora/  # Comparison model (gitignored: .safetensors)
│
├── inference/
│   ├── extract_baseline_test.py   # Baseline extraction testing
│   ├── run_batch_inference.py     # Pipeline batch inference runner
│   └── triage_pipeline.py         # Single-sample inference pipeline
│
├── evaluation/
│   ├── compare_extraction.py      # F1 / Precision / Recall / Schema Validity for extractions
│   ├── compare_validation.py      # JSON schema validation check on all outputs
│   ├── evaluate_distilbert.py     # DistilBERT classifier test-set evaluation
│   ├── calculate_parameters.py    # LoRA parameter count and adapter size statistics
│   └── evaluate_forgetting.py     # Catastrophic forgetting assessment on general tasks
│
├── reports/
│   ├── structured_outputs.jsonl        # Pipeline predictions (30 test samples)
│   ├── baseline_structured_outputs.jsonl  # Zero-shot baseline predictions
│   ├── SmolLM2-1.7B-Instruct_pipeline.jsonl
│   ├── TinyLlama-1.1B-Chat-v1.0_pipeline.jsonl
│   ├── comparison_metrics.csv          # Aggregated model comparison results
│   ├── comparison_metrics.png          # Results bar chart
│   ├── lora_heatmaps.png               # Layer-wise LoRA weight norm heatmap
│   ├── generate_heatmaps.py            # Generates LoRA heatmaps from adapter
│   ├── distilbert_metrics.py           # Calculates precision/recall/F1 for DistilBERT
│   ├── calculate_metrics.py            # Generates comparative evaluation JSON files
│   ├── triage_metrics.json             # DistilBERT evaluation results across splits
│   ├── extraction_metrics.json         # Comparative extraction metrics
│   ├── system_performance_metrics.json  # Latencies, load times, and RAM metrics
│   ├── lora_weight_metrics.json        # PEFT matrix weight change norms
│   ├── perplexity_metrics.json         # Perplexity scores for domain drift
│   ├── dataset_metrics.json            # Dataset split statistics
│   └── variance_metrics.json           # F1 variance metrics across seeds (42/1337/2026)
```

> **Note:** Model weight files (`.safetensors`) are excluded from this repository due to size. Train them following the instructions below, or download from the companion release.

---

## Dataset

The unified master dataset is `data/raw/splittable_redo.jsonl`. It is the single source of truth that feeds both model training stages.

| Stage | Total | Positive | Negative | Positive Rate |
|---|---|---|---|---|
| Raw master (`splittable_redo.jsonl`) | **351** | **280** | **71** | 79.8% |
| DistilBERT base | 196 | 125 | 71 | 63.8% |
| DistilBERT train | 137 | 83 | 54 | 60.6% |
| DistilBERT val | 29 | 24 | 5 | 82.8% |
| DistilBERT test | 30 | 18 | 12 | 60.0% |
| Qwen base | 175 | 175 | — | 100% |
| Qwen train | 115 | 115 | — | 100% |
| Qwen val | 30 | 30 | — | 100% |
| Qwen test | 30 | 30 | — | 100% |

**Qwen base event type distribution** (near-uniform, 35 per class):

| Event Type | Count | % |
|---|---|---|
| FacilityHalt | 36 | 20.6% |
| ShipmentDelay | 36 | 20.6% |
| SupplierInsolvency | 35 | 20.0% |
| TariffChange | 35 | 20.0% |
| ForceMajeure | 33 | 18.9% |

---

## Quickstart

### 1. Install Dependencies
> **Requirement:** Python 3.11 is required.

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
You can run the entire pipeline (data preparation, training, merging, inference, comparative model runs, variance runs, evaluation scripts, and report metrics generation) with a single command:

```bash
python run_all.py
```

#### Pipeline Stages
The pipeline execution is divided into the following numbered stages:
* **Stage 0: Data Preparation** — Runs `split_dataset.py` to partition `splittable_redo.jsonl` into stratified splits.
* **Stage 1: Model Training** — Trains the DistilBERT triage gate classifier and the Qwen2.5-1.5B LoRA extractor.
* **Stage 2: LoRA Adapter Merge** — Merges the trained Qwen LoRA adapter into the base model.
* **Stage 3: Inference** — Runs batch inference over the test set using the merged pipeline.
* **Stage 4: Comparison Models** — Trains and runs batch inference on TinyLlama and SmolLM2 for comparison.
* **Stage 5: Variance Testing** — Retrains the Qwen LoRA pipeline with alternative seeds (1337, 2026).
* **Stage 6: Evaluation** — Runs all evaluation scripts (extraction F1, schema validity, DistilBERT metrics, forgetting).
* **Stage 7: Reports and Visualizations** — Computes comparison JSON metrics and plots LoRA adapter weight change heatmaps.

#### Resuming or Running Specific Stages
* Run a single stage only:
  ```bash
  python run_all.py --only 6
  ```
* Resume pipeline execution from a specific stage:
  ```bash
  python run_all.py --from 3
  ```

### 3. Step-by-Step Training & Run Commands
If you prefer running individual scripts manually instead of using `run_all.py`, follow these commands:

```bash
# Step A: Prepare dataset splits
python training/split_dataset.py

# Step B: Train triage classifier
python training/train_distilbert.py

# Step C: Train extraction model (Qwen LoRA)
python training/train_qwen_lora.py

# Step D: Merge the adapter weights
python training/merge_lora.py
```

### 4. Single-Sample Inference
Test the end-to-end pipeline (Triage + Extraction) on a single string snippet:
```bash
python inference/triage_pipeline.py "A major port strike has halted container shipments at Rotterdam for 3 days."
```

**Output:**
```json
{
  "event_type": "ShipmentDelay",
  "source_timestamp": null,
  "text_evidence": "halted container shipments at Rotterdam for 3 days",
  "arguments": {
    "carrier": null,
    "origin": "Rotterdam",
    "destination": null,
    "delay_duration_days": 3,
    "reason": "major port strike"
  },
  "event_id": "EVT-XXXXXXXX"
}
```

### 5. Evaluate Individual Modules
```bash
# Evaluate Extraction (F1 / Recall / Precision)
python evaluation/compare_extraction.py

# Evaluate Triage Classifier (DistilBERT)
python evaluation/evaluate_distilbert.py
```

---

## Key Results

### 1. Stage 1: DistilBERT Triage Classifier Performance
The binary classifier is optimized to prevent false negatives (missed disruption events). It achieves the following performance metrics on the test set:

| Split | Accuracy | Precision | Recall | F1-Score | True Positives | False Positives | True Negatives | False Negatives |
|---|---|---|---|---|---|---|---|---|
| **Test Set** | **90.00%** | **85.71%** | **100.00%** | **92.31%** | 18 | 3 | 9 | 0 |
| **Val Set** | 96.55% | 100.00% | 95.83% | 97.87% | 23 | 0 | 5 | 1 |
| **Train Set** | 100.00% | 100.00% | 100.00% | 100.00% | 83 | 0 | 54 | 0 |

* **100.00% Recall at the Gate:** The triage classifier guarantees that no actual supply chain events are dropped.
* **85.71% Precision:** Ensures that completely irrelevant text chunks are filtered out effectively before downstream generative processing.

### 2. Stage 2: Event Extraction and Argument Fill (Test Set)

| Model / Configuration | Precision | Recall | F1-Score | Top-Level F1 | Arguments F1 | Schema Validity |
|---|---|---|---|---|---|---|
| Baseline (Qwen Zero-Shot) | 46.96% | 47.65% | 46.81% | 58.40% | 39.28% | 23.33% |
| **Qwen2.5-1.5B (LoRA)** | **72.51%** | **67.61%** | **69.24%** | **83.97%** | **59.23%** | **100.00%** |
| SmolLM2-1.7B (LoRA) | 66.20% | 57.01% | 60.21% | 73.19% | 50.06% | 100.00% |
| TinyLlama-1.1B (LoRA) | 53.28% | 50.28% | 50.58% | 52.15% | 49.01% | 100.00% |

- **47.9% relative F1 improvement** over zero-shot baseline (46.81% → 69.24%)
- **100% schema validity** via Outlines constrained decoding (vs. 23.33% baseline)
- **~92–99% cost reduction** in production (only ~20% of general incoming documents trigger generative extraction, with the other 80% successfully routed away by DistilBERT)
- **Efficient Memory Footprint:** The entire pipeline runs within **~1.44 GB RAM** (peak) on standard CPU during inference, and routes triage checks in ~15 ms on a T4 GPU (with a CPU loading time of ~71 ms).

---

## LoRA Configuration

| Parameter | Value |
|---|---|
| Base Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Rank (r) | 16 |
| Alpha (α) | 32 |
| Target Modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable Parameters | 18.46M (1.196% of base) |
| Optimizer Memory | ~147.7 MB (vs ~12.3 GB full fine-tuning) |
| Adapter File Size | 73.9 MB |

---

## Training Progression (Qwen2.5-1.5B LoRA)

| Epoch | Train Loss | Top-Level F1 (Val) | Arguments F1 (Val) |
|---|---|---|---|
| 1 | 0.1923 | 34.21% | 36.38% |
| 2 | 0.0492 | 52.05% | 37.56% |
| 3 | 0.0114 | 76.94% | 68.35% |

---

## Citation

> Paper under preparation. BibTeX entry will be added upon publication.

---

## License

This project is for research and educational purposes. See [LICENSE](LICENSE) for details.
