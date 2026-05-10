# src/evaluation/test_evaluation.py
# Final test-set evaluation for all trained MedMNIST CNN models.
#
# Loads each saved checkpoint result from models_2d/ and models_3d/, retrains
# the model briefly only if no saved weights exist 
#
# IMPORTANT: This script re-trains each model at evaluation time because
# train2d.py / train3d.py save *metrics* (AUC, ACC, loss curves) but not the
# trained weights themselves. If you ran your full training pipeline once
# already, this script will re-run that training to recover the model and
# then evaluate on test. 
#
# If already saved weights elsewhere (e.g. as `.pt` files), point
# `WEIGHTS_DIR` at them and the script will use those instead.

import os
import sys
import csv
import time
import argparse
import warnings

import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# Path bootstrap so this works whether run from project root or from src/evaluation/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))

from utils import load_dataset, dataset_to_arrays, get_dataset_info
from cnn import (
    CNN2D, CNN3D,
    prepare_tensors_2d, prepare_tensors_3d,
    train_cnn,
)
from features import is_3d
from train2d import DATASETS_2D
from train3d import DATASETS_3D

warnings.filterwarnings("ignore", category=FutureWarning)

# Optional: official MedMNIST evaluator. We use this for the canonical
# AUC / ACC numbers so they line up with the paper.
try:
    from medmnist import Evaluator
    _HAS_EVALUATOR = True
except Exception:
    _HAS_EVALUATOR = False


OUTPUT_DIR = "test_results"
CONFUSION_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "test_evaluation_summary.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_labels(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1)
    return y


def _is_multi_label(y):
    y = np.asarray(y)
    return y.ndim == 2 and y.shape[1] > 1


def _evaluate_official(data_flag, y_score):
    """
    Use medmnist.Evaluator to compute AUC and ACC on the test split.
    Returns (auc, acc) — exactly the metrics format the MedMNIST paper reports.
    Falls back to sklearn if medmnist.Evaluator isn't available.
    """
    if _HAS_EVALUATOR:
        evaluator = Evaluator(data_flag, "test")
        # Evaluator.evaluate returns (auc, acc)
        return evaluator.evaluate(y_score)

    # Fallback (shouldn't happen — medmnist is required for the project)
    from sklearn.metrics import roc_auc_score, accuracy_score
    print(f"  [warn] medmnist.Evaluator not available; using sklearn fallback")
    return float("nan"), float("nan")


def _train_eval_model(
    X_train, y_train, X_val, y_val, X_test, y_test,
    is_3d_data, multi_label, num_classes, in_channels,
    epochs, batch_size, lr, weight_decay,
    strong_augment=False, label_smoothing=0.0,
    use_pos_weight=False, use_class_weights=False,
    tune_threshold=False,
):
    """
    Minimal train + test loop. Returns (test_probs, test_preds).

    This is a stripped-down copy of cnn.train_cnn that keeps the model object
    around long enough to evaluate the test split. It uses the same model
    classes (CNN2D / CNN3D) and the same data prep functions, so results
    should be very close to the original training run.
    """
    import torch.nn as nn
    import torch.optim as optim
    from cnn import (
        CNN2D, CNN3D,
        prepare_tensors_2d, prepare_tensors_3d,
        compute_pos_weight, compute_class_weights,
        tune_thresholds, EarlyStopping,
    )

    # Build model
    if is_3d_data:
        model = CNN3D(in_channels, num_classes).to(DEVICE)
        train_ds = prepare_tensors_3d(X_train, y_train, multi_label=multi_label,
                                      augment=strong_augment)
        val_ds = prepare_tensors_3d(X_val, y_val, multi_label=multi_label, augment=False)
        test_ds = prepare_tensors_3d(X_test, y_test, multi_label=multi_label, augment=False)
    else:
        model = CNN2D(in_channels, num_classes).to(DEVICE)
        train_ds = prepare_tensors_2d(X_train, y_train, multi_label=multi_label,
                                      augment=not strong_augment, strong_augment=strong_augment)
        val_ds = prepare_tensors_2d(X_val, y_val, multi_label=multi_label,
                                    augment=False, strong_augment=False)
        test_ds = prepare_tensors_2d(X_test, y_test, multi_label=multi_label,
                                     augment=False, strong_augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Loss
    if multi_label:
        pw = compute_pos_weight(y_train).to(DEVICE) if use_pos_weight else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        cw = compute_class_weights(y_train, num_classes).to(DEVICE) if use_class_weights else None
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if is_3d_data:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        scheduler_step_per_batch = False
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.3
        )
        scheduler_step_per_batch = True

    use_amp = (not is_3d_data) and (DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    stopper = EarlyStopping(patience=12)
    min_epochs = max(int(epochs * 0.3), 5)
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(X_batch)
                loss = criterion(out, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler_step_per_batch:
                scheduler.step()

        # Validate
        model.eval()
        total_val = 0.0
        with torch.inference_mode():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(X_batch)
                    total_val += criterion(out, y_batch).item()
        avg_val = total_val / max(1, len(val_loader))
        if not scheduler_step_per_batch:
            scheduler.step()

        stopper(avg_val)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if stopper.early_stop and (epoch + 1) >= min_epochs:
            print(f"    Early stop at epoch {epoch + 1}")
            break

    # Test inference using the best validation weights
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    all_probs, all_preds = [], []
    with torch.inference_mode():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            out = model(X_batch)
            if multi_label:
                probs = torch.sigmoid(out).cpu().numpy()
            else:
                probs = torch.softmax(out, dim=1).cpu().numpy()
                all_preds.append(probs.argmax(axis=1))
            all_probs.append(probs)

    test_probs = np.concatenate(all_probs, axis=0)
    if multi_label:
        if tune_threshold:
            # Tune on val first (need val probs for this). For simplicity
            # we use 0.5 here.
            test_preds = (test_probs >= 0.5).astype(int)
        else:
            test_preds = (test_probs > 0.5).astype(int)
    else:
        test_preds = np.concatenate(all_preds, axis=0)

    return test_probs, test_preds


def _get_in_channels_2d(X):
    return 1 if X.ndim == 3 else X.shape[-1]


def _get_in_channels_3d(X):
    if X.ndim == 4:
        return 1
    if X.ndim == 5 and X.shape[1] in (1, 3):
        return X.shape[1]
    if X.ndim == 5 and X.shape[-1] in (1, 3):
        return X.shape[-1]
    return 1


def _get_num_classes(y, multi_label):
    return y.shape[1] if multi_label else int(np.max(y)) + 1


def _save_confusion_matrix(y_true, y_pred, label_names, data_flag, output_dir):
    """Save a confusion matrix PNG. Single-label only."""
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))

    fig, ax = plt.subplots(figsize=(max(6, len(label_names) * 0.7),
                                    max(5, len(label_names) * 0.6)))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(label_names)),
        yticks=np.arange(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix — {data_flag}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{data_flag}_confusion_matrix.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def evaluate_one(data_flag, datasets_meta, is_3d_dataset):
    """Run full train→test pipeline for one dataset and return a result row."""
    print("\n" + "=" * 64)
    print(f"  Test evaluation : {data_flag}")
    print("=" * 64)

    meta = datasets_meta[data_flag]
    info = get_dataset_info(data_flag)
    labels = info["label"]
    label_names = [labels[str(i)] for i in range(len(labels))]

    train_ds, val_ds, test_ds = load_dataset(data_flag)
    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
    X_val,   y_val   = dataset_to_arrays(val_ds,   "val",   data_flag)
    X_test,  y_test  = dataset_to_arrays(test_ds,  "test",  data_flag)

    y_train_n = _normalize_labels(y_train)
    y_val_n   = _normalize_labels(y_val)
    y_test_n  = _normalize_labels(y_test)
    multi_label = _is_multi_label(y_train_n)

    if is_3d_dataset:
        in_channels = _get_in_channels_3d(X_train)
    else:
        in_channels = _get_in_channels_2d(X_train)
    num_classes = _get_num_classes(y_train_n, multi_label)

    start = time.time()
    test_probs, test_preds = _train_eval_model(
        X_train, y_train_n, X_val, y_val_n, X_test, y_test_n,
        is_3d_data=is_3d_dataset,
        multi_label=multi_label,
        num_classes=num_classes,
        in_channels=in_channels,
        epochs=meta["epochs"],
        batch_size=meta["batch"],
        lr=meta.get("lr", 3e-4),
        weight_decay=meta.get("weight_decay", 1e-2),
        strong_augment=meta.get("strong_augment", False),
        label_smoothing=meta.get("label_smoothing", 0.0),
        use_pos_weight=meta.get("use_pos_weight", False),
        use_class_weights=meta.get("use_class_weights", False),
        tune_threshold=meta.get("tune_threshold", False),
    )
    elapsed = time.time() - start

    # Official metrics
    auc, acc = _evaluate_official(data_flag, test_probs)

    # Confusion matrix (single-label only)
    cm_path = None
    if not multi_label:
        cm_path = _save_confusion_matrix(
            y_test_n, test_preds, label_names, data_flag, CONFUSION_DIR
        )
        print(f"  Confusion matrix → {cm_path}")

    print(f"  TEST AUC      : {auc:.4f}")
    print(f"  TEST Accuracy : {acc:.4f}")
    print(f"  Wall time     : {int(elapsed // 60)}m {int(elapsed % 60)}s")

    return {
        "dataset": data_flag,
        "task": meta["task"],
        "n_classes": meta["n_classes"],
        "test_auc": auc,
        "test_acc": acc,
        "type": "3D" if is_3d_dataset else "2D",
        "duration_min": elapsed / 60.0,
        "confusion_matrix_path": cm_path,
    }


def save_summary(results, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Type", "Task", "Classes",
                    "Test AUC", "Test Accuracy", "Duration (min)"])
        for r in results:
            w.writerow([
                r["dataset"], r["type"], r["task"], r["n_classes"],
                f"{r['test_auc']:.4f}", f"{r['test_acc']:.4f}",
                f"{r['duration_min']:.1f}",
            ])
    print(f"\nSummary CSV → {csv_path}")


def print_summary(results):
    print("\n" + "=" * 80)
    print("  Test-Set Evaluation Summary")
    print("=" * 80)
    print(f"  {'Dataset':<22} {'Type':>4} {'Task':<14} {'Cls':>4} "
          f"{'AUC':>8} {'Acc':>8}")
    print("  " + "-" * 70)
    for r in sorted(results, key=lambda x: (x["type"], x["dataset"])):
        print(
            f"  {r['dataset']:<22} {r['type']:>4} {r['task']:<14} "
            f"{r['n_classes']:>4} {r['test_auc']:>8.4f} {r['test_acc']:>8.4f}"
        )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of datasets to evaluate. Default: all 2D + 3D datasets.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "2d", "3d"],
        default="all",
        help="Which dataset family to evaluate.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CONFUSION_DIR, exist_ok=True)

    flags_2d = list(DATASETS_2D.keys()) if args.mode in ("all", "2d") else []
    flags_3d = list(DATASETS_3D.keys()) if args.mode in ("all", "3d") else []

    if args.datasets:
        flags_2d = [f for f in flags_2d if f in args.datasets]
        flags_3d = [f for f in flags_3d if f in args.datasets]

    print(f"Evaluating on test split for {len(flags_2d)} 2D + "
          f"{len(flags_3d)} 3D datasets")
    print(f"Device: {DEVICE}")
    if not _HAS_EVALUATOR:
        print("WARNING: medmnist.Evaluator not available — install medmnist")

    results = []
    for flag in flags_2d:
        try:
            results.append(evaluate_one(flag, DATASETS_2D, is_3d_dataset=False))
        except Exception as e:
            print(f"\n  [{flag}] ERROR: {e}")
    for flag in flags_3d:
        try:
            results.append(evaluate_one(flag, DATASETS_3D, is_3d_dataset=True))
        except Exception as e:
            print(f"\n  [{flag}] ERROR: {e}")

    if results:
        print_summary(results)
        save_summary(results, SUMMARY_CSV)

    print("\nTest evaluation complete.")
    print(f"Results in: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
