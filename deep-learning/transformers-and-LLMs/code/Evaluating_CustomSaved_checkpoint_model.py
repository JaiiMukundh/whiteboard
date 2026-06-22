import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from evaluate import load
import wandb

wandb.init(project="bert-mrpc-split-pipeline", name="evaluation-run")
wandb.define_metric("eval_step")
wandb.define_metric("val/*", step_metric="eval_step")

# 1. Point to your saved model directory
saved_model_path = "./my_finetuned_bert_mrpc"

print(f"Loading saved model and tokenizer from '{saved_model_path}'...")
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)

# 2. Load and preprocess only the validation split of the dataset
raw_datasets = load_dataset("nyu-mll/glue", "mrpc")

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer)

# 3. Create only the evaluation DataLoader (keeps batches on CPU by default)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# 4. Initialize the evaluation metric
metric = load("glue", "mrpc")

# Move the model to CPU (this remains robust and avoids the XPU driver bug)
model.to("cpu")
model.eval()

print("Running evaluation on CPU (this will take less than 5 seconds)...")
total_eval_loss = 0

for step, batch in enumerate(eval_dataloader):
    # Ensure all batch tensors are on the CPU
    batch = {k: v.to("cpu") for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
    
    loss = outputs.loss.item() if outputs.loss is not None else 0.0
    total_eval_loss += loss

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    # Calculate evaluation accuracy for this specific batch
    batch_accuracy = (predictions == batch["labels"]).float().mean().item()
    
    # 3. Log batch-level step curves for validation
    wandb.log({
        "val/batch_loss": loss,
        "val/batch_accuracy": batch_accuracy,
        "eval_step": step
    })

    # Safely evaluate using CPU NumPy arrays
    metric.add_batch(
        predictions=predictions.numpy(), 
        references=batch["labels"].numpy()
    )

print("\nEvaluation Results:")
print(metric.compute())

wandb.finish()