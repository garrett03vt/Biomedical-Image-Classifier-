# src/evaluation/run_all_noise_evaluations.py
# Runs the existing noise_evaluation_2d.py and noise_evaluation_3d.py
# pipelines across ALL MedMNIST datasets and produces a unified comparison
# CSV + heatmap suitable for the report.
#
# Output:
#   noise_results/all_noise_results.csv       — long-format results table
#   noise_results/auc_degradation_heatmap.png — AUC drop under each noise type

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _THIS_DIR)  # so we can import sibling modules in evaluation/

from train2d import DATASETS_2D
from train3d import DATASETS_3D
from utils import load_dataset, dataset_to_arrays
from cnn import CNN2D, CNN3D, prepare_tensors_2d, prepare_tensors_3d
from noise import add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score


OUTPUT_DIR = "noise_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_labels(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1)
    return y


def _is_multi_label(y):
    y = np.asarray(y)
    return y.ndim == 2 and y.shape[1] > 1


def _compute_auc(y_true, y_probs, multi_label):
    y_true = np.asarray(y_true)
    if multi_label:
        return roc_auc_score(y_true, y_probs, average="macro")
    y_true = _normalize_labels(y_true)
    if len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, y_probs[:, 1])
    return roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")


def _compute_acc(y_true, y_pred, multi_label):
    if multi_label:
        return float(((y_pred == np.asarray(y_true)).mean(axis=0)).mean())
    return float(accuracy_score(_normalize_labels(y_true), y_pred))


def _train_quick(X_train, y_train, is_3d_data, multi_label, epochs):
    """Train a small CNN. Returns the trained model."""
    y_train = _normalize_labels(y_train)
    num_classes = y_train.shape[1] if multi_label else int(np.max(y_train)) + 1

    if is_3d_data:
        in_channels = 1 if X_train.ndim == 4 else (
            X_train.shape[1] if X_train.shape[1] in (1, 3) else X_train.shape[-1]
        )
        model = CNN3D(in_channels, num_classes).to(DEVICE)
        train_ds = prepare_tensors_3d(X_train, y_train, multi_label=multi_label)
        batch_size = 16
    else:
        in_channels = 1 if X_train.ndim == 3 else X_train.shape[-1]
        model = CNN2D(in_channels, num_classes).to(DEVICE)
        train_ds = prepare_tensors_2d(X_train, y_train, multi_label=multi_label, augment=True)
        batch_size = 64

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

    return model, in_channels, num_classes


def _evaluate(model, X_eval, y_eval, is_3d_data, multi_label):
    if is_3d_data:
        ds = prepare_tensors_3d(X_eval, y_eval, multi_label=multi_label)
        batch_size = 16
    else:
        ds = prepare_tensors_2d(X_eval, y_eval, multi_label=multi_label, augment=False)
        batch_size = 64
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    model.eval()
    all_probs, all_preds = [], []
    with torch.inference_mode():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            out = model(X_batch)
            if multi_label:
                probs = torch.sigmoid(out).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            else:
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)

    y_probs = np.concatenate(all_probs, axis=0)
    y_preds = np.concatenate(all_preds, axis=0)
    return _compute_auc(y_eval, y_probs, multi_label), _compute_acc(y_eval, y_preds, multi_label)


def run_one(data_flag, is_3d_data, epochs):
    print(f"\n--- {data_flag} ---")
    train_ds, val_ds, _ = load_dataset(data_flag)
    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
    X_val,   y_val   = dataset_to_arrays(val_ds,   "val",   data_flag)
    multi_label = _is_multi_label(y_train)

    model, _, _ = _train_quick(X_train, y_train, is_3d_data, multi_label, epochs)

    results = {}
    auc, acc = _evaluate(model, X_val, y_val, is_3d_data, multi_label)
    results["clean"] = (auc, acc)
    print(f"  clean: AUC={auc:.4f} ACC={acc:.4f}")

    for noise_name, fn, kw in [
        ("gaussian_0.05", add_gaussian_noise, {"std": 0.05}),
        ("gaussian_0.10", add_gaussian_noise, {"std": 0.10}),
        ("gaussian_0.20", add_gaussian_noise, {"std": 0.20}),
        ("salt_pepper_0.05", add_salt_pepper_noise, {"amount": 0.05}),
        ("speckle_0.10", add_speckle_noise, {"std": 0.10}),
    ]:
        X_noisy = fn(X_val, **kw)
        auc, acc = _evaluate(model, X_noisy, y_val, is_3d_data, multi_label)
        results[noise_name] = (auc, acc)
        print(f"  {noise_name}: AUC={auc:.4f} ACC={acc:.4f}")

    return results


def save_long_csv(all_results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Type", "Condition", "AUC", "Accuracy"])
        for ds, info in all_results.items():
            for cond, (auc, acc) in info["results"].items():
                w.writerow([ds, info["type"], cond, f"{auc:.4f}", f"{acc:.4f}"])
    print(f"\nWrote {path}")


def plot_degradation_heatmap(all_results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    datasets = sorted(all_results.keys(), key=lambda d: (all_results[d]["type"], d))
    conditions = ["clean", "gaussian_0.05", "gaussian_0.10", "gaussian_0.20",
                  "salt_pepper_0.05", "speckle_0.10"]

    matrix = np.zeros((len(datasets), len(conditions)))
    for i, ds in enumerate(datasets):
        clean_auc = all_results[ds]["results"]["clean"][0]
        for j, cond in enumerate(conditions):
            auc = all_results[ds]["results"][cond][0]
            # Drop relative to clean (positive = degradation)
            matrix[i, j] = clean_auc - auc

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.5),
                                    max(6, len(datasets) * 0.4)))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0,
                   vmax=max(0.05, matrix.max()))
    ax.figure.colorbar(im, ax=ax, label="AUC drop vs. clean")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_title("AUC Degradation Under Noise (clean − noisy)",
                 fontsize=12, fontweight="bold")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.03f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10,
                        help="Quick training epochs (default 10 — for the "
                             "comparison only, not for final reported results)")
    parser.add_argument("--mode", choices=["all", "2d", "3d"], default="all")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of datasets")
    args = parser.parse_args()

    flags_2d = list(DATASETS_2D.keys()) if args.mode in ("all", "2d") else []
    flags_3d = list(DATASETS_3D.keys()) if args.mode in ("all", "3d") else []
    if args.datasets:
        flags_2d = [f for f in flags_2d if f in args.datasets]
        flags_3d = [f for f in flags_3d if f in args.datasets]

    print(f"Noise evaluation across {len(flags_2d)} 2D + {len(flags_3d)} 3D "
          f"datasets, {args.epochs} quick epochs each")
    print(f"Device: {DEVICE}")

    all_results = {}
    for flag in flags_2d:
        try:
            all_results[flag] = {
                "type": "2D",
                "results": run_one(flag, is_3d_data=False, epochs=args.epochs),
            }
        except Exception as e:
            print(f"  [{flag}] ERROR: {e}")

    for flag in flags_3d:
        try:
            all_results[flag] = {
                "type": "3D",
                "results": run_one(flag, is_3d_data=True, epochs=args.epochs),
            }
        except Exception as e:
            print(f"  [{flag}] ERROR: {e}")

    if all_results:
        save_long_csv(all_results, os.path.join(OUTPUT_DIR, "all_noise_results.csv"))
        plot_degradation_heatmap(all_results,
                                 os.path.join(OUTPUT_DIR, "auc_degradation_heatmap.png"))


if __name__ == "__main__":
    main()
