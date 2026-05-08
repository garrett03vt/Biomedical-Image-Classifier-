# train2d.py
# Train CNN2D models on all 12 2D MedMNIST datasets.
# Optimized for: RTX 4090 (24 GB VRAM) | Ryzen 9 5900X (12C/24T) | 128 GB RAM. change if you have different hardware (Garrette Ritz)
#
# Results are saved to models_2d/
#
# Dataset reference:
#   pathmnist      Colon Pathology        Multi-Class  (9)   89,996 / 10,004 /  7,180
#   chestmnist     Chest X-Ray            Multi-Label (14)   78,468 / 11,219 / 22,433
#   dermamnist     Dermatoscope           Multi-Class  (7)    7,007 /  1,003 /  2,005
#   octmnist       Retinal OCT            Multi-Class  (4)   97,477 / 10,832 /  1,000
#   pneumoniamnist Chest X-Ray            Binary-Class (2)    4,708 /    524 /    624
#   retinamnist    Fundus Camera          Ordinal Reg  (5)    1,080 /    120 /    400
#   breastmnist    Breast Ultrasound      Binary-Class (2)      546 /     78 /    156
#   bloodmnist     Blood Cell Microscope  Multi-Class  (8)   11,959 /  1,712 /  3,421
#   tissuemnist    Kidney Cortex Micro.   Multi-Class  (8)  165,466 / 23,640 / 47,280
#   organamnist    Abdominal CT           Multi-Class (11)   34,561 /  6,491 / 17,778
#   organcmnist    Abdominal CT           Multi-Class (11)   12,975 /  2,392 /  8,216
#   organsmnist    Abdominal CT           Multi-Class (11)   13,932 /  2,452 /  8,827

import os
import time
import warnings
import numpy as np
import joblib
import torch
import torch.backends.cudnn as cudnn

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm

from utils import load_dataset, dataset_to_arrays, get_dataset_info
from cnn import train_cnn

warnings.filterwarnings("ignore", category=FutureWarning)

# GPU optimizations
cudnn.benchmark     = True   # auto-tune kernels for fixed input sizes
cudnn.deterministic = False  # allow non-deterministic ops for max throughput
torch.set_float32_matmul_precision("high")  # TF32 on Ada Lovelace ≈ 8× faster matmuls

# Constants
MODELS_DIR = "models_2d"

# Per-dataset config.
# batch_size is tuned to the dataset size and task:
#   - Large datasets (>50k train): 512 — keeps the GPU saturated
#   - Medium datasets (5k–50k):   256
#   - Small datasets (<5k):       64  — avoid overfitting on tiny sets
#   - chestmnist uses BCELoss (multi-label), so slightly smaller batch for stability
#
# epochs:
#   - Large datasets need fewer epochs to see enough samples
#   - Small datasets benefit from more passes
#
# All flags are lowercase MedMNIST keys.
DATASETS_2D = {
    "pathmnist":      {"task": "multi-class",   "n_classes": 9,  "n_train": 89996,  "batch": 512, "epochs": 30},
    "chestmnist":     {"task": "multi-label",   "n_classes": 14, "n_train": 78468,  "batch": 256, "epochs": 30},
    "dermamnist":     {"task": "multi-class",   "n_classes": 7,  "n_train": 7007,   "batch": 256, "epochs": 50},
    "octmnist":       {"task": "multi-class",   "n_classes": 4,  "n_train": 97477,  "batch": 512, "epochs": 30},
    "pneumoniamnist": {"task": "binary-class",  "n_classes": 2,  "n_train": 4708,   "batch": 128, "epochs": 50},
    "retinamnist":    {"task": "ordinal-reg",   "n_classes": 5,  "n_train": 1080,   "batch": 64,  "epochs": 80},
    "breastmnist":    {"task": "binary-class",  "n_classes": 2,  "n_train": 546,    "batch": 32,  "epochs": 100},
    "bloodmnist":     {"task": "multi-class",   "n_classes": 8,  "n_train": 11959,  "batch": 256, "epochs": 50},
    "tissuemnist":    {"task": "multi-class",   "n_classes": 8,  "n_train": 165466, "batch": 512, "epochs": 25},
    "organamnist":    {"task": "multi-class",   "n_classes": 11, "n_train": 34561,  "batch": 256, "epochs": 40},
    "organcmnist":    {"task": "multi-class",   "n_classes": 11, "n_train": 12975,  "batch": 256, "epochs": 50},
    "organsmnist":    {"task": "multi-class",   "n_classes": 11, "n_train": 13932,  "batch": 256, "epochs": 50},
}

DEFAULT_LR = 3e-4


# helpers

def normalize_labels(y):
    """(N, 1) → (N,); multi-label (N, K>1) left untouched."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1)
    return y


def is_multi_label_target(y):
    y = np.asarray(y)
    return y.ndim == 2 and y.shape[1] > 1


def compute_auc(y_true, y_probs, multi_label):
    y_true = np.asarray(y_true)
    if multi_label:
        return roc_auc_score(y_true, y_probs, average="macro")
    y_true = normalize_labels(y_true)
    classes = np.unique(y_true)
    if len(classes) == 2:
        return roc_auc_score(y_true, y_probs[:, 1])
    return roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")


# persistence 

def _path(data_flag):
    return f"{MODELS_DIR}/{data_flag}_cnn2d_results.joblib"


def already_trained(data_flag):
    return os.path.exists(_path(data_flag))


def save_2d_results(data_flag, auc, acc, duration, labels, report_str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        {
            "dataset":      data_flag,
            "auc":          auc,
            "accuracy":     acc,
            "method":       "cnn2d",
            "duration":     duration,
            "labels":       labels,
            "class_report": report_str,
        },
        _path(data_flag),
    )


def load_2d_results(data_flag):
    return joblib.load(_path(data_flag))


# single dataset 

def train_single_2d(
    data_flag,
    epochs=None,       # None → use per-dataset default from DATASETS_2D
    batch_size=None,   # None → use per-dataset default from DATASETS_2D
    lr=DEFAULT_LR,
    force_retrain=False,
):
    """
    Full CNN2D pipeline for one 2D MedMNIST dataset.

    Batch size and epoch count are tuned per dataset (see DATASETS_2D table).
    Pass explicit values to override.

    1. Load train / val splits
    2. Normalise labels
    3. Train CNN2D  (defined in cnn.py)
    4. Compute AUC, accuracy, and per-class F1 on the val set
    5. Save result to models_2d/
    """
    if data_flag not in DATASETS_2D:
        raise ValueError(
            f"'{data_flag}' is not in the 2D dataset list. "
            f"Valid flags: {list(DATASETS_2D.keys())}"
        )

    meta       = DATASETS_2D[data_flag]
    batch_size = batch_size if batch_size is not None else meta["batch"]
    epochs     = epochs     if epochs     is not None else meta["epochs"]

    if not force_retrain and already_trained(data_flag):
        print(f"  [{data_flag}] Already trained — loading saved result...")
        r = load_2d_results(data_flag)
        print(f"  [{data_flag}] AUC: {r['auc']:.4f}  Accuracy: {r['accuracy']:.4f}")
        return r

    print("\n" + "=" * 64)
    print(f"  Training 2D dataset : {data_flag}")
    print("=" * 64)

    info        = get_dataset_info(data_flag)
    labels      = info["label"]
    label_names = [labels[str(i)] for i in range(len(labels))]

    print(f"  Task       : {meta['task']}")
    print(f"  Classes    : {meta['n_classes']}")
    print(f"  Train size : {meta['n_train']:,}")
    print(f"  Config     : epochs={epochs}  batch={batch_size}  lr={lr}")
    for lid, lname in labels.items():
        print(f"    [{lid}] {lname}")

    train_ds, val_ds, _ = load_dataset(data_flag)
    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
    X_val,   y_val   = dataset_to_arrays(val_ds,   "val",   data_flag)

    y_train     = normalize_labels(y_train)
    y_val       = normalize_labels(y_val)
    multi_label = is_multi_label_target(y_train)

    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    print(f"\n  X_train     : {X_train.shape}  y_train : {y_train.shape}")
    print(f"  X_val       : {X_val.shape}    y_val   : {y_val.shape}")
    print(f"  Multi-label : {multi_label}")
    print(f"  Device      : {device_name}\n")

    start_time = time.time()

    y_probs, y_preds = train_cnn(
        X_train, y_train,
        X_val,   y_val,
        is_3d_data=False,
        multi_label=multi_label,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    elapsed  = time.time() - start_time
    duration = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    auc = compute_auc(y_val, y_probs, multi_label)
    acc = accuracy_score(y_val, y_preds)

    # classification_report needs flat integer labels for single-label tasks
    report_y_val  = y_val  if not multi_label else y_val
    report_y_pred = y_preds
    report_str = classification_report(
        report_y_val, report_y_pred, target_names=label_names, zero_division=0
    )

    print(f"\n{'─' * 64}")
    print(f"  [{data_flag}] RESULTS")
    print(f"{'─' * 64}")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Time     : {duration}")
    print(f"\n  Classification Report (val set):\n{report_str}")

    if not multi_label:
        print("  Per-class accuracy (val set):")
        for class_id, class_name in labels.items():
            mask = y_val == int(class_id)
            if mask.sum() == 0:
                continue
            class_acc = accuracy_score(y_val[mask], y_preds[mask])
            print(f"    [{class_id}] {class_name:<26}: {class_acc:.4f}  (n={mask.sum()})")

    save_2d_results(data_flag, auc, acc, duration, labels, report_str)

    return {
        "dataset":      data_flag,
        "auc":          auc,
        "accuracy":     acc,
        "method":       "cnn2d",
        "duration":     duration,
        "labels":       labels,
        "class_report": report_str,
    }


# full pipeline

def train_all_2d(lr=DEFAULT_LR, force_retrain=False):
    """
    Train CNN2D on all 12 2D MedMNIST datasets.
    Each dataset uses its individually tuned batch size and epoch count.
    """
    print("=" * 72)
    print("  2D Biomedical Image Classification Pipeline")
    print("  RTX 4090 | CNN only | per-dataset batch & epoch tuning")
    print("=" * 72)
    print(
        f"\n  {'Dataset':<18} {'Task':<14} {'Cls':>4} {'Train':>8} "
        f"{'Batch':>6} {'Epochs':>7}  Status"
    )
    print("  " + "-" * 68)
    for flag, meta in DATASETS_2D.items():
        status = "✓ cached" if already_trained(flag) else "needs training"
        print(
            f"  {flag:<18} {meta['task']:<14} {meta['n_classes']:>4} "
            f"{meta['n_train']:>8,} {meta['batch']:>6} {meta['epochs']:>7}  {status}"
        )

    all_results = []
    total_start = time.time()

    for flag in tqdm(DATASETS_2D, desc="2D training", unit="dataset"):
        try:
            result = train_single_2d(flag, lr=lr, force_retrain=force_retrain)
            all_results.append(result)
        except Exception as e:
            print(f"\n  [{flag}] ERROR: {e}")
            all_results.append({
                "dataset": flag, "auc": None, "accuracy": None,
                "method": "cnn2d", "duration": "—", "labels": {}, "class_report": str(e),
            })

    total_elapsed  = time.time() - total_start
    total_duration = f"{int(total_elapsed // 60)}m {int(total_elapsed % 60)}s"

    print("\n" + "=" * 72)
    print("  2D Training Summary")
    print("=" * 72)
    print(f"  {'Dataset':<20} {'Task':<14} {'AUC':>8} {'Accuracy':>10} {'Time':>10}")
    print("  " + "-" * 67)
    for r in all_results:
        meta = DATASETS_2D[r["dataset"]]
        if r["auc"] is None:
            print(f"  {r['dataset']:<20} ERROR")
        else:
            print(
                f"  {r['dataset']:<20} {meta['task']:<14} "
                f"{r['auc']:>8.4f} {r['accuracy']:>10.4f} {r['duration']:>10}"
            )
    print("=" * 72)
    print(f"  Total wall-clock time : {total_duration}")
    print("=" * 72)

    return all_results


# entry point

if __name__ == "__main__":
    train_all_2d(lr=3e-4, force_retrain=False)