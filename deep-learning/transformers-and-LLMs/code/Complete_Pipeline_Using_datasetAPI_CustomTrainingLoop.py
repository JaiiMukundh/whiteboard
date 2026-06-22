import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from evaluate import load
from accelerate import Accelerator
import wandb

# Preprocessing GLUE MRPC dataset
raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer)

# LOAD THE DATA
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=64, collate_fn=data_collator
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=True, batch_size=64, collate_fn=data_collator
)

# INITIALIZE ACCELERATOR
accelerator = Accelerator()

if accelerator.is_main_process:
    wandb.init(
        project="bert-mrpc-split-pipeline",
        name="training-run",
        config={
            "checkpoint": checkpoint,
            "epochs": 3,
            "batch_size": 64,
            "learning_rate": 5e-5
        }
    )
    # Configure WandB to superpose training and validation metrics on the same X-axis
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")


# MODEL/ARCHITECTURE
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
metric = load("glue", "mrpc")

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# PREPARE EVERYTHING VIA ACCELERATOR
# Prepared both dataloaders so Accelerate knows about them
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

print(f"Accelerate is using device: {accelerator.device}")
progress_bar = tqdm(range(num_training_steps))

# TRAINING LOOP
global_step = 0

for epoch in range(num_epochs):
    for batch in train_dataloader:
        model.train()  # Ensure model is in training mode
        
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        # Track training accuracy and explicitly move tensor calculation to CPU
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        train_accuracy = (predictions == batch["labels"]).float().mean().cpu().item()
        
        # ====================================================================
        # RUN VALIDATION EVALUATION (Runs after every single training batch)
        # ====================================================================
        model.eval()  # Switch model to evaluation mode
        val_loss = 0.0
        val_predictions = []
        val_references = []
        
        for val_batch in eval_dataloader:
            # Explicitly move validation batch tensors to the XPU/GPU target device
            val_batch = {k: v.to(accelerator.device) for k, v in val_batch.items()}
            
            with torch.no_grad():
                outputs = model(**val_batch)
                
            val_loss += outputs.loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Gather metrics across devices and cleanly cast to CPU numpy arrays
            gathered_predictions, gathered_references = accelerator.gather_for_metrics(
                (predictions, val_batch["labels"])
            )
            val_predictions.extend(gathered_predictions.cpu().numpy())
            val_references.extend(gathered_references.cpu().numpy())
        
        # Compute average metrics for this step
        avg_val_loss = val_loss / len(eval_dataloader)
        eval_metrics = metric.compute(predictions=val_predictions, references=val_references)
        
        # LOG BOTH TRAINING & VALIDATION IN A SINGLE CALL
        # This aligns the timeline and superposes the graphs automatically
        if accelerator.is_main_process:
            wandb.log({
                "train/loss": loss.item(),
                "train/accuracy": train_accuracy,
                "val/loss": avg_val_loss,
                "val/accuracy": eval_metrics["accuracy"],
                "val/f1": eval_metrics["f1"],
                "epoch": epoch + (global_step % len(train_dataloader)) / len(train_dataloader),
                "global_step": global_step
            })
            
        global_step += 1

# ===================================================
# SAVE THE MODEL HERE
# ===================================================
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("./my_finetuned_bert_mrpc")
tokenizer.save_pretrained("./my_finetuned_bert_mrpc")

if accelerator.is_main_process:
    wandb.finish()