import numpy as np
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForQuestionAnswering,
    get_linear_schedule_with_warmup
)

def extract_text(batch, start: np.ndarray, end: np.ndarray):
    return [
        batch["context"][i][start[i]:end[i]]
        for i in range(len(start))
    ]

def squad_churn_text(runA, runB):
    textA = runA["pred_text"]
    textB = runB["pred_text"]
    return np.mean([a != b for a, b in zip(textA, textB)])

def squad_churn_spans(runA, runB):
    assert len(runA["start_positions"]) == len(runB["start_positions"]), "Runs must have the same number of examples for churn calculation"
    startA = runA["start_positions"]
    endA = runA["end_positions"]
    startB = runB["start_positions"]
    endB = runB["end_positions"]

    disagree = [(sA != sB) or (eA != eB) for sA, eA, sB, eB in zip(startA, endA, startB, endB)]
    
    return np.mean(disagree)

def preprocess(batch):
    # -----------------------------
    # 1. Load SQuAD
    # -----------------------------
    
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


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