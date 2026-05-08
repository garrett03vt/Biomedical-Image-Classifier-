# train2d.py
# Train CNN2D models on all 12 2D MedMNIST datasets.
# Optimized for: RTX 4090 (24 GB VRAM) | Ryzen 9 5900X (12C/24T) | 128 GB RAM
#
# Results saved to models_2d/

import os
import sys
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

cudnn.benchmark     = True
cudnn.deterministic = False
torch.set_float32_matmul_precision("high")

MODELS_DIR = "models_2d"

# chestmnist fixes:
#   use_pos_weight=True   — capped inverse-frequency BCE weights (max 10×)
#   tune_threshold=True   — per-label F1-optimal threshold replaces fixed 0.5
#   lr=1e-4               — lower LR stabilises multi-label weighted BCE
#   epochs=80             — more passes needed for 14-label learning
#
# retinamnist fixes:
#   use_class_weights=True — inverse-frequency CE weights for 5 ordinal classes
#   strong_augment=True    — flips + rotation + colour jitter
#   label_smoothing=0.1    — prevents overconfident predictions between adjacent grades
#   epochs=200             — tiny dataset needs many passes
DATASETS_2D = {
    "pathmnist":      {"task": "multi-class",  "n_classes": 9,  "n_train": 89996,  "batch": 512, "epochs": 30},
    # chestmnist (multi-label, 14 sparse binary labels):
    # The official MedMNIST metric is mean per-label accuracy at threshold 0.5
    # (NOT subset/exact-match accuracy). On this very sparse dataset, the model
    # that wins is a well-regularised ResNet trained with plain BCE — pos_weight
    # and per-label F1 threshold tuning push the model towards more positive
    # predictions, which helps F1/recall but hurts mean accuracy because false
    # positives dominate. The official ResNet-18 baseline hits ~0.947 with
    # plain BCE + light augmentation + threshold 0.5, so we mirror that.
    "chestmnist":     {"task": "multi-label", "n_classes": 14, "n_train": 78468, "batch": 256, "epochs": 50,
                       "use_pos_weight": False, "tune_threshold": False,
                       "strong_augment": False, "lr": 1e-3, "weight_decay": 1e-4},
    "dermamnist":     {"task": "multi-class",  "n_classes": 7,  "n_train": 7007,   "batch": 256, "epochs": 30,
                       "strong_augment": True},
    "octmnist":       {"task": "multi-class",  "n_classes": 4,  "n_train": 97477,  "batch": 512, "epochs": 20},
    "pneumoniamnist": {"task": "binary-class", "n_classes": 2,  "n_train": 4708,   "batch": 128, "epochs": 50},
    "retinamnist":    {"task": "ordinal-reg",  "n_classes": 5,  "n_train": 1080,   "batch": 32,  "epochs": 300,
                       "strong_augment": True, "lr": 5e-4, "weight_decay": 1e-3},
    "breastmnist":    {"task": "binary-class", "n_classes": 2, "n_train": 546, "batch": 32, "epochs": 150, 
                        "lr": 5e-5, "weight_decay": 1e-2}, # Heavy decay for tiny dataset,
    "bloodmnist":     {"task": "multi-class",  "n_classes": 8,  "n_train": 11959,  "batch": 256, "epochs": 50},
    "tissuemnist":    {"task": "multi-class",  "n_classes": 8,  "n_train": 165466, "batch": 256, "epochs": 100,
                       "lr": 1e-3, "weight_decay": 1e-4, "strong_augment": True},
    "organamnist":    {"task": "multi-class",  "n_classes": 11, "n_train": 34561,  "batch": 256, "epochs": 40},
    "organcmnist":    {"task": "multi-class",  "n_classes": 11, "n_train": 12975,  "batch": 256, "epochs": 50},
    "organsmnist":    {"task": "multi-class",  "n_classes": 11, "n_train": 13932,  "batch": 256, "epochs": 35,
                       "strong_augment": True},
}

DEFAULT_LR = 3e-4


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


def compute_accuracy(y_true, y_preds, multi_label):
    """
    Match the official MedMNIST evaluator:
      * multi-label  -> mean per-label accuracy (averaged over the L labels)
      * single-label -> standard sample accuracy

    sklearn's accuracy_score on 2-D arrays returns subset (exact-match) accuracy,
    which is far stricter than what MedMNIST reports — for chestmnist that
    collapses scores from ~0.95 down to ~0.37. We replicate the official
    formula here so our numbers line up with the benchmark tables.
    """
    y_true  = np.asarray(y_true)
    y_preds = np.asarray(y_preds)

    if multi_label:
        # y_true and y_preds are (N, L) binary arrays
        per_label_acc = (y_preds == y_true).mean(axis=0)
        return float(per_label_acc.mean())

    return float(accuracy_score(normalize_labels(y_true), y_preds))


def _path(data_flag):
    return f"{MODELS_DIR}/{data_flag}_cnn2d_results.joblib"


def already_trained(data_flag):
    return os.path.exists(_path(data_flag))


def save_2d_results(data_flag, auc, acc, duration, labels, report_str, train_losses, val_losses):
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
            "train_losses": train_losses,
            "val_losses":   val_losses,
        },
        _path(data_flag),
    )


def load_2d_results(data_flag):
    return joblib.load(_path(data_flag))


def train_single_2d(
    data_flag,
    epochs=None,
    batch_size=None,
    lr=None,
    force_retrain=False,
):
    if data_flag not in DATASETS_2D:
        raise ValueError(f"'{data_flag}' is not in DATASETS_2D.")

    meta       = DATASETS_2D[data_flag]
    batch_size = batch_size if batch_size is not None else meta["batch"]
    epochs     = epochs     if epochs     is not None else meta["epochs"]
    lr         = lr         if lr         is not None else meta.get("lr", DEFAULT_LR)

    if not force_retrain and already_trained(data_flag):
        tqdm.write(f"  [{data_flag}] Already trained — loading saved result...")
        r = load_2d_results(data_flag)
        tqdm.write(f"  [{data_flag}] AUC: {r['auc']:.4f}  Accuracy: {r['accuracy']:.4f}")
        return r

    tqdm.write("\n" + "=" * 64)
    tqdm.write(f"  Training 2D dataset : {data_flag}")
    tqdm.write("=" * 64)

    info        = get_dataset_info(data_flag)
    labels      = info["label"]
    label_names = [labels[str(i)] for i in range(len(labels))]

    tqdm.write(f"  Task       : {meta['task']}")
    tqdm.write(f"  Classes    : {meta['n_classes']}")
    tqdm.write(f"  Train size : {meta['n_train']:,}")
    tqdm.write(f"  Config     : epochs={epochs}  batch={batch_size}  lr={lr}")
    for lid, lname in labels.items():
        tqdm.write(f"    [{lid}] {lname}")

    train_ds, val_ds, _ = load_dataset(data_flag)
    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
    X_val,   y_val   = dataset_to_arrays(val_ds,   "val",   data_flag)

    y_train     = normalize_labels(y_train)
    y_val       = normalize_labels(y_val)
    multi_label = is_multi_label_target(y_train)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    tqdm.write(f"\n  X_train : {X_train.shape}  y_train : {y_train.shape}")
    tqdm.write(f"  X_val   : {X_val.shape}    y_val   : {y_val.shape}")
    tqdm.write(f"  Device  : {device_name}\n")

    start_time = time.time()

    y_probs, y_preds, train_losses, val_losses = train_cnn(
        X_train, y_train,
        X_val,   y_val,
        is_3d_data=False,
        multi_label=multi_label,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_pos_weight=meta.get("use_pos_weight", False),
        use_class_weights=meta.get("use_class_weights", False),
        strong_augment=meta.get("strong_augment", False),
        label_smoothing=meta.get("label_smoothing", 0.0),
        tune_threshold=meta.get("tune_threshold", False),
        weight_decay=meta.get("weight_decay", 1e-2),
    )

    elapsed  = time.time() - start_time
    duration = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    auc = compute_auc(y_val, y_probs, multi_label)
    acc = compute_accuracy(y_val, y_preds, multi_label)

    report_str = classification_report(
        y_val, y_preds, target_names=label_names, zero_division=0
    )

    tqdm.write(f"\n{'─' * 64}")
    tqdm.write(f"  [{data_flag}] RESULTS")
    tqdm.write(f"{'─' * 64}")
    tqdm.write(f"  AUC      : {auc:.4f}")
    tqdm.write(f"  Accuracy : {acc:.4f}")
    tqdm.write(f"  Time     : {duration}")
    tqdm.write(f"\n  Classification Report (val set):\n{report_str}")

    if not multi_label:
        tqdm.write("  Per-class accuracy (val set):")
        for class_id, class_name in labels.items():
            mask = y_val == int(class_id)
            if mask.sum() == 0:
                continue
            class_acc = accuracy_score(y_val[mask], y_preds[mask])
            tqdm.write(f"    [{class_id}] {class_name:<26}: {class_acc:.4f}  (n={mask.sum()})")

    save_2d_results(data_flag, auc, acc, duration, labels, report_str, train_losses, val_losses)

    return {
        "dataset":      data_flag,
        "auc":          auc,
        "accuracy":     acc,
        "method":       "cnn2d",
        "duration":     duration,
        "labels":       labels,
        "class_report": report_str,
        "train_losses": train_losses,
        "val_losses":   val_losses,
    }


def train_all_2d(force_retrain=False):
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
    sys.stdout.flush()

    all_results = []
    total_start = time.time()

    for flag in tqdm(DATASETS_2D, desc="2D training", unit="dataset"):
        try:
            result = train_single_2d(flag, force_retrain=force_retrain)
            all_results.append(result)
        except Exception as e:
            tqdm.write(f"\n  [{flag}] ERROR: {e}")
            all_results.append({
                "dataset": flag, "auc": None, "accuracy": None,
                "method": "cnn2d", "duration": "—", "labels": {},
                "class_report": str(e), "train_losses": [], "val_losses": [],
            })

    total_elapsed  = time.time() - total_start
    total_duration = f"{int(total_elapsed // 60)}m {int(total_elapsed % 60)}s"

    sys.stdout.flush()
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
    sys.stdout.flush()

    return all_results


if __name__ == "__main__":
    train_all_2d(force_retrain=False)