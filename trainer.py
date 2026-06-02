import argparse
from xml.parsers.expat import model
# from pyexpat import model
from task.utils import squad_churn_spans, squad_churn_text
from train_model import build_model, build_dataloaders, train, evaluate_glue, evaluate_squad
# from test_model import evaluate
import torch
import random
import numpy as np
from train_squad import preprocess
import tqdm
import logging

logging.basicConfig(filename="experiment.log", level=logging.INFO)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_experiment(seed, model, train_loader, val_loader):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    val_acc, preds, labels, logits, start_positions, end_positions = train(task=args.task_name, model=model, train_loader=train_loader, val_loader=val_loader, device=device)  # your training loop
    # Save model checkpoint after training
    # torch.save(model.state_dict(), f"model_checkpoint_seed_{seed}.pt")
    print(f"Start positions: {start_positions}, End positions: {end_positions}")
    return val_acc, preds, labels, logits, start_positions, end_positions


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Model name (e.g., 'roberta-base' or 'distilbert-base-uncased')")
    parser.add_argument("--dataset_name", type=str, default="glue", help="Dataset name (e.g., 'glue' or 'squad')")
    parser.add_argument("--task_name", type=str, default="sst2", help="Task name (e.g., 'sst2' for GLUE or 'plain_text' for SQuAD)")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of random seeds to run experiments with")
    args = parser.parse_args()

    # Log the experiment configuration
    print("Experiment Configuration:")
    print(f"Model Name: {args.model_name}")
    print(f"Dataset Name: {args.dataset_name}")
    print(f"Task Name: {args.task_name}")
    print(f"Number of Seeds: {args.num_seeds}")


    print("Loading model and dataset...")
    model, tokenizer = build_model(args.model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loader, val_loader = build_dataloaders(tokenizer, args.dataset_name, args.task_name if args.dataset_name == "glue" else None, preprocess_fn=preprocess if args.dataset_name == "squad" else None)

    results = []
    for seed in tqdm.tqdm(range(args.num_seeds)):
        print(f"Running experiment with seed {seed}")
        acc, preds, labels, logits, start_positions, end_positions = run_experiment(seed, model, train_loader, val_loader)
        results.append({
            "seed": seed,
            "acc": acc,
            "preds": preds,
            "labels": labels,
            "logits": logits,
            "start_positions": start_positions,
            "end_positions": end_positions
        })
    
    print(f"Mean false error rate per seed: {[np.mean(r["preds"]!=r["labels"]) for r in results]} (in case of existing preds, labels)")
    # print([r["labels"] for r in results])

    num_results = len(results)

    accuracies = [r["acc"] for r in results]
    if all(accuracies):
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        print("Mean accuracy:", mean_acc)
        print("Stddev:", std_acc)

    def compute_churn(preds_a, preds_b):
        preds_a = np.array(preds_a)
        preds_b = np.array(preds_b)
        return np.mean(preds_a != preds_b)


    churn_matrix = np.zeros((num_results, num_results))

    # if all(~np.isnan(np.array(list(results[i].values())) for i in range(num_results))):
    if args.dataset_name == "glue":
        for i in range(num_results):
            for j in range(num_results):
                print(f"Comparing run {i} and run {j} for normal churn...")
                churn_matrix[i, j] = compute_churn( # Pairwise churn between all runs
                    results[i]["preds"],
                    results[j]["preds"]
                )
        mean_churn = churn_matrix[np.triu_indices(num_results, k=1)].mean() # Mean over all independent runs
        print("Mean churn:", mean_churn)

    # if all(results[i]["preds"] is not None for i in range(num_results)):
    elif args.dataset_name == "squad":
        for i in range(num_results):
            for j in range(num_results):
                print(f"Comparing run {i} and run {j} for span churn...")
                churn_matrix[i, j] = squad_churn_spans( # Pairwise churn between all runs
                    results[i],
                    results[j]
                )
        mean_churn = churn_matrix[np.triu_indices(num_results, k=1)].mean() # Mean over all independent runs
        print("Mean text churn:", mean_churn)


    def compute_l2(logits_a, logits_b):
        logits_a = np.array(logits_a)
        logits_b = np.array(logits_b)
        diffs = logits_a - logits_b
        return np.mean(np.linalg.norm(diffs, axis=1))

    l2_matrix = np.zeros((num_results, num_results))

    if all(results[i]["logits"] for i in range(num_results)):
        for i in range(num_results):
            for j in range(num_results):
                l2_matrix[i, j] = compute_l2(
                    results[i]["logits"],
                    results[j]["logits"]
                )

        mean_l2 = l2_matrix[np.triu_indices(num_results, k=1)].mean()
        print("Mean L2 divergence:", mean_l2)


    def compute_confusion_groups(preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)

        tp = np.where((preds == 1) & (labels == 1))[0]
        tn = np.where((preds == 0) & (labels == 0))[0]
        fp = np.where((preds == 1) & (labels == 0))[0]
        fn = np.where((preds == 0) & (labels == 1))[0]

        return tp, tn, fp, fn

    for r in results:
        if all((r["preds"], r["labels"])):
            tp, tn, fp, fn = compute_confusion_groups(r["preds"], r["labels"])
            r["tp_idx"] = tp
            r["tn_idx"] = tn
            r["fp_idx"] = fp
            r["fn_idx"] = fn

    if all(results[i]["fp_idx"] for i in range(num_results)):
        fp_counts = [len(r["fp_idx"]) for r in results]
        fp_std = np.std(fp_counts)
        print("FP stddev:", fp_std)

        num_samples = len(results[0]["labels"])
        fp_frequency = np.zeros(num_samples)

        for r in results:
            fp_frequency[r["fp_idx"]] += 1

        # Samples that frequently flip into FP
        unstable_fp_samples = np.where(fp_frequency > 5)[0]
        print("Unstable FP samples:", unstable_fp_samples) # These are samples that frequently flip into FP across different runs, indicating they may be inherently ambiguous or difficult for the model to classify consistently.

    def subgroup_churn(results, subgroup="fp_idx"):
        churns = []
        for i in range(num_results):
            for j in range(i+1, num_results):
                a = results[i][subgroup]
                b = results[j][subgroup]
                # Jaccard distance
                churn = 1 - len(set(a) & set(b)) / len(set(a) | set(b)) if len(set(a) | set(b)) > 0 else 0
                churns.append(churn)
        return np.mean(churns)

    if all((results[i]["fp_idx"], results[i]["fn_idx"]) for i in range(num_results)):
        print("FP churn:", subgroup_churn(results, "fp_idx"))
        print("FN churn:", subgroup_churn(results, "fn_idx"))

    # Collate all results into a DataFrame for easier analysis
    import pandas as pd
    df = pd.DataFrame({
        "seed": [r["seed"] for r in results],
        "acc": [r["acc"] for r in results],
        "fp_count": [len(r["fp_idx"]) if "fp_idx" in r else np.nan for r in results],
        "fn_count": [len(r["fn_idx"]) if "fn_idx" in r else np.nan for r in results],
        "fp_churn": [subgroup_churn(results, "fp_idx") if "fp_idx" in r else np.nan for r in results],
        "fn_churn": [subgroup_churn(results, "fn_idx") if "fn_idx" in r else np.nan for r in results],
        # Add more columns as needed for analysis
    })

    # Store per seed results as JSON or CSV for further analysis
    import json
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
