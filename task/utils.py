import numpy as np

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
    assert runA["start_positions"].shape == runB["start_positions"].shape
    startA = runA["start_positions"]
    endA = runA["end_positions"]
    startB = runB["start_positions"]
    endB = runB["end_positions"]

    disagree = [(sA != sB) or (eA != eB) for sA, eA, sB, eB in zip(startA, endA, startB, endB)]
    
    return np.mean(disagree)