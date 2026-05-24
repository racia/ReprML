import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForQuestionAnswering,
    get_linear_schedule_with_warmup
)

# -----------------------------
# 1. Load SQuAD
# -----------------------------
dataset = load_dataset("squad")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# -----------------------------
# 2. Preprocessing
# -----------------------------
def preprocess(batch):

    batch["question"] = [q.lstrip() for q in batch["question"]] # Strip leading whitespace from questions to avoid tokenization issues

    tokenized = tokenizer(
        batch["question"],
        batch["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sample_idx = sample_mapping[i]
        answer = batch["answers"][sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = tokenized.sequence_ids(i)

        # Find start/end token positions
        token_start = 0
        while token_start < len(offsets) and sequence_ids[token_start] != 1:
            token_start += 1

        token_end = len(offsets) - 1
        while token_end >= 0 and sequence_ids[token_end] != 1:
            token_end -= 1

        if not (offsets[token_start][0] <= start_char and offsets[token_end][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # Find exact token positions
            s = token_start
            while s <= token_end and offsets[s][0] <= start_char:
                s += 1
            start_positions.append(s - 1)

            e = token_end
            while e >= token_start and offsets[e][1] >= end_char:
                e -= 1
            end_positions.append(e + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

if __name__ == "__main__":

    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
    dataset.set_format(type="torch")

    train_loader = DataLoader(dataset["train"], batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=16)

    # -----------------------------
    # 3. Model
    # -----------------------------
    model = RobertaForQuestionAnswering.from_pretrained("roberta-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # -----------------------------
    # 4. Optimizer + Scheduler
    # -----------------------------
    optimizer = AdamW(model.parameters(), lr=3e-5)
    num_epochs = 2
    num_training_steps = num_epochs * len(train_loader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # -----------------------------
    # 5. Training Loop
    # -----------------------------
    def train_epoch():
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

    # -----------------------------
    # 6. Evaluation Loop (simple)
    # -----------------------------
    def evaluate():
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                total_loss += outputs.loss.item()
        return total_loss / len(val_loader)

    # -----------------------------
    # 7. Run Training
    # -----------------------------
    for epoch in range(num_epochs):
        train_epoch()
        val_loss = evaluate()
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}")
