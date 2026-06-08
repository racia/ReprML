import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    RobertaForQuestionAnswering,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset

from task.utils import extract_text
from task.utils import preprocess

# -----------------------------
# 1. Load Dataset (either GLUE (SST-2) or SQuAD)
# -----------------------------
def build_dataloaders(tokenizer, dataset_name, task_name=None, preprocess_fn=None, num_samples: int = None):
        
    dataset = load_dataset(path=dataset_name, name=task_name)  
    if num_samples != None and num_samples > 0:
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    else:
        sampler = RandomSampler(dataset, replacement=False, num_samples=dataset["train"].num_rows)
    print(sampler, "with sampler size:", len(sampler))
    
    if "squad" in dataset_name:
        preprocess_fn = preprocess
        dataset = dataset.map(preprocess_fn, batched=True, remove_columns=dataset["train"].column_names)
        dataset.set_format(type="torch")

        train_loader = DataLoader(dataset["train"], batch_size=8, shuffle=False, sampler=sampler) # TODO: Try 16 for a100 architecture
        val_loader = DataLoader(dataset["validation"], batch_size=16)
    else:
        def tokenize(batch):
            return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)
        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=False, sampler=sampler)
        val_loader = DataLoader(dataset["validation"], batch_size=32)
    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    return train_loader, val_loader

# -----------------------------
# 2. Model
# -----------------------------
def build_model(model_name):
    if "roberta" in model_name:
        model = RobertaForQuestionAnswering.from_pretrained(
            model_name,
            num_labels=2
        )
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    elif "distilbert" in model_name:
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    return model, tokenizer

# -----------------------------
# 4. Training Loop
# -----------------------------
def train_epoch(task, model, train_loader, device, optimizer, scheduler):
    model.train()
    total, correct = 0, 0

    if task == "sst2":
         for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    elif task == "plain_text":
        for batch in train_loader:
            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            # logits = outputs.logits

            loss.backward()
            optimizer.step()
            scheduler.step()

            # preds = logits.argmax(dim=1)
            # correct += (preds == labels).sum().item()
            # total += labels.size(0)

    return correct / total if task == "sst2" else None

# -----------------------------
# 5. Evaluation Loop (GLUE)
# -----------------------------
def evaluate_glue(model, val_loader, device):    
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total, preds.cpu().numpy(), labels.cpu().numpy(), outputs.logits.cpu().numpy()

# -----------------------------
# 5. Evaluation Loop (SQuAD)
# -----------------------------
def evaluate_squad(model, val_loader, device):
    model.eval()
    total_loss = 0
    pred_texts = []
    start_positions = []
    end_positions = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.start_logits.argmax(dim=1), outputs.end_logits.argmax(dim=1)
            # print("Preds:", preds)
            start, end = batch["start_positions"], batch["end_positions"]
            # pred_text = extract_text(batch, start.cpu().numpy(), end.cpu().numpy()) TODO: Fix by not omitting context
            # pred_texts.extend(pred_text)
            start_positions.extend(start.cpu().numpy())
            end_positions.extend(end.cpu().numpy())
            total_loss += outputs.loss.item()
    return total_loss / len(val_loader), start_positions, end_positions


# -----------------------------
# 6. Run Training
# -----------------------------
def train(task, model, num_epochs, train_loader, val_loader, device):
    # -----------------------------
    # 3. Optimizer + Scheduler
    # -----------------------------
    optimizer = AdamW(model.parameters(), lr=2e-5 if "distilbert" in model.__class__.__name__.lower() else 3e-5)
    num_training_steps = num_epochs * len(train_loader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0 if "distilbert" in model.__class__.__name__.lower() else int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    val_acc, val_loss, preds, labels, logits, start_positions, end_positions = None, None, None, None, None, None, None
    for epoch in range(num_epochs):
        train_acc = train_epoch(task, model, train_loader, device, optimizer, scheduler)
        if task == "sst2":
            val_acc, preds, labels, logits = evaluate_glue(model, val_loader, device)
        elif task == "plain_text":
            # print(f"Keys in val_loader batch: {next(iter(val_loader)).keys()}")  # Debugging line to check batch keys")
            val_loss, start_positions, end_positions = evaluate_squad(model, val_loader, device)
        print(f"Epoch {epoch+1}:", f"train_acc={train_acc:.4f}" if train_acc is not None else "train_acc=0.0000", f"val_loss={val_loss:.4f}" if val_loss is not None else "val_acc={val_acc:.4f}")
    # Return final evaluation results after training
    return val_loss, val_acc, preds, labels, logits, start_positions, end_positions