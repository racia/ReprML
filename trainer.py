import argparse
from pathlib import Path

from omegaconf import OmegaConf
from task.utils import squad_churn_spans, squad_churn_text
from train_model import build_model, build_dataloaders, train, evaluate_glue, evaluate_squad
# from test_model import evaluate
import torch
import random
import numpy as np
import tqdm
import logging
import json

logging.basicConfig(filename="experiment.log", level=logging.INFO)

def set_seed(seed):
    '''Fixes all random seeds for weights initialization, data shuffling, random sampling (batch ordering)
    Residue noise may come from execution order of floating point arithmetic and parallel cuDNN kernels'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_torch_determ_alg(mode: bool = True):
    '''Sets deterministic torch algoriths (impl noise)'''
    print(f"Setting torch determ. algorithms: {mode}")
    torch.use_deterministic_algorithms(mode) # True

def set_cudnn_kernels(mode: bool = True):
    '''Sets cuDNN deterministic kernels  (impl noise)'''
    print(f"Setting cudnn deterministic kernels: {mode}")
    torch.backends.cudnn.deterministic = mode # True
    torch.backends.cudnn.benchmark = not mode # False

def set_float32():
    '''Sets float32 fixed precision instead of switching between types (impl noise)'''
    print(f"Setting fixed precision to float32")
    torch_dtype = torch.float32


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value

def run_experiment(seed, model, task, epochs, train_loader, val_loader, alg_noise: bool = True):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    val_loss, val_acc, preds, labels, logits, start_positions, end_positions = train(task=task, model=model, num_epochs=epochs, train_loader=train_loader, val_loader=val_loader, device=device)  # your training loop
    # Save model checkpoint after training
    # torch.save(model.state_dict(), f"model_checkpoint_seed_{seed}.pt")
    if start_positions.size > 0 and end_positions.size > 0:
        pass
        # assert len(start_positions) == len(end_positions), f"Length mismatch between start and end positions"
    return val_loss, val_acc, preds, labels, logits, start_positions, end_positions


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="train-glue", help="Config file path for either squad oder glue training run")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Model name (e.g., 'roberta-base' or 'distilbert-base-uncased')")
    parser.add_argument("--dataset_name", type=str, default="glue", help="Dataset name (e.g., 'glue' or 'squad')")
    parser.add_argument("--task_name", type=str, default="sst2", help="Task name (e.g., 'sst2' for GLUE or 'plain_text' for SQuAD)")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of random seeds to run experiments with")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to use for each experiment")
    # parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    args = parser.parse_args()

    # Set project path
    prj_path = Path.cwd()
    
    # Load config file from project path
    conf_path = Path(prj_path, "config")
    conf_name = args.config
    conf = OmegaConf.load(Path(conf_path, conf_name))

    model_name = conf.model.name if conf else args.dataset_name
    dataset_name = conf.task.dataset_name if conf else args.dataset_name
    task_name = conf.task.task_name if conf else args.task_name
    num_seeds = conf.run.seeds if conf else args.num_seeds
    num_epochs = conf.train.epochs if conf else args.num_epochs
    num_samples = conf.train.samples if conf else args.num_samples

    # Set results dir
    results_path = Path(prj_path / "results" / conf.results.output_path if conf else dataset_name)
    Path.mkdir(Path(results_path), parents=True, exist_ok=True)

    # Set algo./impl. noise control configs
    impl_noise = conf.noise.algo.control if conf else False
    algo_noise = conf.noise.impl.control if conf else False
    both = (impl_noise and algo_noise) or conf.run.deterministic
    no_control = not(impl_noise or algo_noise)
    print(f"Deteceted investigating one of noise types:\nimpl:{impl_noise}, alg:{algo_noise}, both:{both}, no_control:{no_control}")
    
    if not (both or no_control):
        res_noise_path = Path(results_path, "alg" if algo_noise else "imp")
    elif both:
        res_noise_path = Path(results_path, "both")
    else:
        res_noise_path = Path(results_path, "no_crl")
    print(f"Creating noise results path as: ", res_noise_path)
    Path.mkdir(Path(res_noise_path), parents=True, exist_ok=True)


    if not no_control or both: # What individual noise types to fix
        if algo_noise:
            impl_noise_types = conf.noise.impl.types
        if impl_noise:
            algo_noise_types = conf.noise.algo.types # i.e. torch.determ.algo.

    # Log the experiment configuration
    print("Experiment Configuration:")
    print(f"Model Name: {model_name}")
    print(f"Dataset Name: {dataset_name}")
    print(f"Task Name: {task_name}")
    print(f"Number of Seeds: {num_seeds}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Number of Samples: {num_samples}")
    # print(f"Batch Size: {batch_size}")
    # print(f"Num Epochs: {epochs}")

    print("Loading model and dataset...")
    model, tokenizer = build_model(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loader, val_loader = build_dataloaders(
        tokenizer, 
        "nyu-mll/glue" if "glue" in dataset_name else "rajpurkar/squad", 
        task_name if dataset_name == "glue" else None, 
        num_samples=num_samples,
        )
    train_loader_size = train_loader.dataset.shape[0]
    val_loader_size = val_loader.dataset.shape[0]
    print(f"Running training on {train_loader_size} samples")

    results = []
    for seed in tqdm.tqdm(range(num_seeds)):

        # Set all implementation-level seeds, ensuring determinism
        set_torch_determ_alg(mode=algo_noise or both)
        set_cudnn_kernels(mode=algo_noise or both)
        if (algo_noise or both):
            set_float32() 

        # Set all algorithm-level noise (weights init., random sampling, data shuffling..)
        seed = conf.run.seed if (impl_noise or both) else seed # Fixed impl or both controlled
        print(f"Running experiment with seed {seed}")
        Path.mkdir(Path(res_noise_path, f"seed_{seed}"), parents=True, exist_ok=True) if (impl_noise or both) else None
        print(f"Creating results directory for seed {seed} at: ", res_noise_path)
        
        val_loss, val_acc, preds, labels, logits, start_positions, end_positions = run_experiment(seed, model, task_name, num_epochs, train_loader, val_loader)
        
        results.append({
            "seed": seed,
            "val_acc": val_acc if val_acc else None,
            "val_loss": val_loss if val_loss else None,
            "preds": preds, # TODO: Case None 
            "labels": labels,
            "logits": logits,
            "start_positions": start_positions,
            "end_positions": end_positions
        })
    assert (not isinstance(vals, np.ndarray) for res in results for keys, vals in res.items())
    print(f"Mean false error rate per seed: {[np.mean(np.array(r['preds'])!=np.array(r['labels'])) for r in results]} (in case of existing preds, labels)")
    # print([r["labels"] for r in results])

    num_results = len(results)

    accuracies = [r["val_acc"] for r in results]

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print("Mean accuracy:", mean_acc)
    print("Stddev:", std_acc)

    def compute_churn(preds_a, preds_b):
        preds_a = np.array(preds_a)
        preds_b = np.array(preds_b)
        return np.mean(preds_a != preds_b)


    churn_matrix = np.zeros((num_results, num_results))

    if dataset_name == "glue":
        for i in range(num_results):
            for j in range(num_results):
                print(f"Comparing run {i} and run {j} for normal churn...")
                churn_matrix[i, j] = compute_churn( # Pairwise churn between all runs
                    results[i]["preds"],
                    results[j]["preds"]
                )
        mean_churn = churn_matrix[np.triu_indices(num_results, k=1)].mean() # Mean over all independent runs
        print("Mean churn:", mean_churn)
    elif dataset_name == "squad":
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
    try:
        for i in range(num_results):
            for j in range(num_results):
                l2_matrix[i, j] = compute_l2(
                    results[i]["logits"],
                    results[j]["logits"]
                )
        mean_l2 = l2_matrix[np.triu_indices(num_results, k=1)].mean()
        print("Mean L2 divergence:", mean_l2)
    except TypeError:
        print(f"L2-divergence couldn't be computed (likeliky due to missing logits) for run with seed {seed}, skipping..")


    def compute_confusion_groups(preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)

        tp = np.where((preds == 1) & (labels == 1))[0]
        tn = np.where((preds == 0) & (labels == 0))[0]
        fp = np.where((preds == 1) & (labels == 0))[0]
        fn = np.where((preds == 0) & (labels == 1))[0]

        return tp, tn, fp, fn

    for r in results:
        # if np.all((r["preds"], r["labels"])):
        try:
            tp, tn, fp, fn = compute_confusion_groups(r["preds"], r["labels"])
            r["tp_idx"] = tp
            r["tn_idx"] = tn
            r["fp_idx"] = fp
            r["fn_idx"] = fn
            print(f"Preds and labels found for run")#: {r}")
        except Exception as e:
            print(str(e))
    fp_counts = [len(r["fp_idx"]) for r in results] # Check for pred/ĺabs not necessary since 0 otherwise 
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
                try:
                    a = results[i][subgroup]
                    b = results[j][subgroup]
                    # Jaccard distance
                    churn = 1 - len(set(a) & set(b)) / len(set(a) | set(b)) if len(set(a) | set(b)) > 0 else 0
                    churns.append(churn)
                except KeyError:
                    print(f"No {subgroup} entries found for run: ")#{r}, skipping..")
        return np.mean(churns)

    try: # Nans should be manageable
        # fp_idx_churn = subgroup_churn(results, "fp_idx")
        # fn_idx_churn = subgroup_churn(results, "fn_idx")
        print("FP churn:", subgroup_churn(results, "fp_idx"))
        print("FN churn:", subgroup_churn(results, "fn_idx"))
    except TypeError as e:
        print(str(e))
        fp_idx_churn = np.nan
        fn_idx_churn = np.nan

    # Collate all results into a DataFrame for easier analysis
    import pandas as pd
    df = pd.DataFrame({
        "seed": [r["seed"] for r in results],
        "val_acc": [r["val_acc"] for r in results],
        "val_loss": [r["val_loss"] for r in results],
        "fp_count": [len(r["fp_idx"]) for r in results],
        "fn_count": [len(r["fn_idx"]) for r in results],
        "fp_churn": subgroup_churn(results, "fp_idx"),
        "fn_churn": subgroup_churn(results, "fn_idx"),
        # Add more columns as needed for analysis
    })

    results_serializable = []
    for r in results:
        results_serializable.append(to_jsonable(r))

    # Store per seed results as JSON or CSV for further analysis
    with open(Path(res_noise_path, f"seed_{results[0]['seed']}" if (impl_noise or both) else "", f"experiment_results-{dataset_name}.json"), "w") as f:
        json.dump(
            {"conf": to_jsonable(OmegaConf.to_container(conf, resolve=True)),
            "results": results_serializable
            },
            f, indent=4)
