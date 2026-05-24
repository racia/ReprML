import argparse
# from pyexpat import model
from train_model import build_model, build_dataloaders, train, evaluate
import torch
import random
import numpy as np
from train_squad import preprocess
import tqdm

# Load model from checkpoint and run inference on test set

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0, preds.cpu().numpy(), labels.cpu().numpy(), logits.cpu().numpy()


if __name__ == "__main__":
