# train3d.py
# Train CNN3D models on all 3D MedMNIST datasets.
# Optimized for: RTX 4090 (24 GB VRAM) | Ryzen 9 5900X (12C/24T) | 128 GB RAM (Garrette R., 2024-06-15)
#
# Datasets:
#   adrenalmnist3d  : Binary-Class (2)  | Shape from Abdominal CT  | 1,188 / 98  / 298
#   fracturemnist3d : Multi-Class  (3)  | Chest CT                 | 1,027 / 103 / 240
#   nodulemnist3d   : Binary-Class (2)  | Chest CT                 | 1,158 / 165 / 310
#   organmnist3d    : Multi-Class  (11) | Abdominal CT             |   971 / 161 / 610
#   synapsemnist3d  : Binary-Class (2)  | Electron Microscope      | 1,230 / 177 / 352
#   vesselmnist3d   : Binary-Class (2)  | Shape from Brain MRA     | 1,335 / 191 / 382

import os
import time
import joblib
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from utils import load_dataset, dataset_to_arrays, get_dataset_info
from features import is_3d
from cnn import train_cnn


# GPU / cuDNN optimizations
# These settings are meant for an RTX 4090 for fixed-size 3D volumes. (CHANGE IF YOU HAVE A DIFFERENT GPU OR DATA SHAPE!)
cudnn.benchmark     = True   # auto-tune convolution kernels per input shape
cudnn.deterministic = False  # allow non-deterministic ops for max throughput
torch.set_float32_matmul_precision("high")  # TF32 on Ada Lovelace = big speedup


# Constants

MODELS_DIR = "models_3d"

DATASETS_3D = {
    "adrenalmnist3d":  {"task": "binary-class",  "n_classes": 2},
    "fracturemnist3d": {"task": "multi-class",   "n_classes": 3},
    "nodulemnist3d":   {"task": "binary-class",  "n_classes": 2},
    "organmnist3d":    {"task": "multi-class",   "n_classes": 11},
    "synapsemnist3d":  {"task": "binary-class",  "n_classes": 2},
    "vesselmnist3d":   {"task": "binary-class",  "n_classes": 2},
}

# Again, I'm using an RTX 4090. If you have less VRAM, reduce batch_size to 16 or 8. If you have more, try 64. (Garrette R., 2024-06-15)
# 24 GB VRAM easily fits batch_size=32 for 28^3 volumes.
# 50 epochs gives better convergence on these small datasets vs 30.
# num_workers=8 keeps the GPU fed without starving the OS on a 24-thread CPU.
DEFAULT_EPOCHS     = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR         = 3e-4
DEFAULT_WORKERS    = 8


# Label helpers

def normalize_labels(y):
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

# Persistence 

def _path(data_flag):
    return f"{MODELS_DIR}/{data_flag}_cnn3d_results.joblib"
 
 
def already_trained(data_flag):
    return os.path.exists(_path(data_flag))
 
 
def save_3d_results(data_flag, auc, acc, duration, labels, report_str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        {
            "dataset":      data_flag,
            "auc":          auc,
            "accuracy":     acc,
            "method":       "cnn3d",
            "duration":     duration,
            "labels":       labels,
            "class_report": report_str,
        },
        _path(data_flag),
    )
 
 
def load_3d_results(data_flag):
    return joblib.load(_path(data_flag))


# Single-dataset training 
def train_single_3d(
    data_flag,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LR,
    force_retrain=False,
):
    """
    Full CNN3D pipeline for one 3D MedMNIST dataset.
 
    1. Load train / val splits
    2. Normalise labels
    3. Train CNN3D (defined in cnn.py)
    4. Compute AUC, accuracy, and per-class F1 on the val set
    5. Save result to models_3d/
    """
    if not force_retrain and already_trained(data_flag):
        print(f"  [{data_flag}] Already trained — loading saved result...")
        r = load_3d_results(data_flag)
        print(f"  [{data_flag}] AUC: {r['auc']:.4f}  Accuracy: {r['accuracy']:.4f}")
        if "class_report" in r:
            print(r["class_report"])
        return r
 
    print("\n" + "=" * 64)
    print(f"  Training 3D dataset : {data_flag}")
    print("=" * 64)
 
    info        = get_dataset_info(data_flag)
    labels      = info["label"]
    label_names = [labels[str(i)] for i in range(len(labels))]
 
    print(f"  Task    : {info['task']}")
    print(f"  Classes : {len(labels)}")
    for lid, lname in labels.items():
        print(f"    {lid}: {lname}")
 
    train_ds, val_ds, _ = load_dataset(data_flag)
    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
    X_val,   y_val   = dataset_to_arrays(val_ds,   "val",   data_flag)
 
    if not is_3d(X_train):
        raise ValueError(
            f"{data_flag} does not look 3-D (shape {X_train.shape}). "
            "Use train2d.py for 2D datasets."
        )
 
    y_train     = normalize_labels(y_train)
    y_val       = normalize_labels(y_val)
    multi_label = is_multi_label_target(y_train)
 
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    print(f"\n  X_train     : {X_train.shape}  y_train : {y_train.shape}")
    print(f"  X_val       : {X_val.shape}    y_val   : {y_val.shape}")
    print(f"  Multi-label : {multi_label}")
    print(f"  Device      : {device_name}")
    print(f"  Config      : epochs={epochs}  batch={batch_size}  lr={lr}\n")
 
    start_time = time.time()
 
    y_probs, y_preds = train_cnn(
        X_train, y_train,
        X_val,   y_val,
        is_3d_data=True,
        multi_label=multi_label,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
 
    elapsed  = time.time() - start_time
    duration = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
 
    auc = compute_auc(y_val, y_probs, multi_label)
    acc = accuracy_score(y_val, y_preds)
 
    report_str = classification_report(
        y_val, y_preds, target_names=label_names, zero_division=0
    )
 
    print(f"\n{'─' * 64}")
    print(f"  [{data_flag}] RESULTS")
    print(f"{'─' * 64}")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Time     : {duration}")
    print(f"\n  Classification Report (val set):\n{report_str}")
 
    print("  Per-class accuracy (val set):")
    for class_id, class_name in labels.items():
        mask = y_val == int(class_id)
        if mask.sum() == 0:
            continue
        class_acc = accuracy_score(y_val[mask], y_preds[mask])
        print(f"    [{class_id}] {class_name:<24}: {class_acc:.4f}  (n={mask.sum()})")
 
    save_3d_results(data_flag, auc, acc, duration, labels, report_str)
 
    return {
        "dataset":      data_flag,
        "auc":          auc,
        "accuracy":     acc,
        "method":       "cnn3d",
        "duration":     duration,
        "labels":       labels,
        "class_report": report_str,
    }


# Full pipeline

def train_all_3d(
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LR,
    force_retrain=False,
):
    """Train CNN3D on all six 3D MedMNIST datasets and print a summary table."""
    print("=" * 64)
    print("  3D Biomedical Image Classification Pipeline")
    print(f"  RTX 4090 | batch={batch_size} | epochs={epochs} | lr={lr}")
    print("=" * 64)
    print("\nDatasets scheduled:")
    for flag, meta in DATASETS_3D.items():
        status = "✓ cached" if already_trained(flag) else "needs training"
        print(f"  {flag:<25} ({meta['task']}, {meta['n_classes']} cls)  [{status}]")
 
    all_results = []
    total_start = time.time()
 
    for flag in DATASETS_3D:
        result = train_single_3d(
            flag,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            force_retrain=force_retrain,
        )
        all_results.append(result)
 
    total_elapsed  = time.time() - total_start
    total_duration = f"{int(total_elapsed // 60)}m {int(total_elapsed % 60)}s"
 
    print("\n" + "=" * 72)
    print("  3D Training Summary")
    print("=" * 72)
    print(
        f"{'Dataset':<25} {'Task':<15} {'Cls':>4} "
        f"{'AUC':>8} {'Accuracy':>10} {'Time':>10}"
    )
    print("-" * 72)
    for r in all_results:
        meta = DATASETS_3D[r["dataset"]]
        print(
            f"{r['dataset']:<25} "
            f"{meta['task']:<15} "
            f"{meta['n_classes']:>4} "
            f"{r['auc']:>8.4f} "
            f"{r['accuracy']:>10.4f} "
            f"{r['duration']:>10}"
        )
    print("=" * 72)
    print(f"  Total wall-clock time : {total_duration}")
    print("=" * 72)
 
    return all_results


# Entry point

if __name__ == "__main__":
    train_all_3d(
        epochs=50,
        batch_size=32,   # safe on 24 GB VRAM for 28^3 volumes; try 64 if headroom allows 
        lr=3e-4,
        force_retrain=False,
    )
 