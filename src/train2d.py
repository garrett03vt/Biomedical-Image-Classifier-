# src/train_2d.py
# Train CNN models on all 2D MedMNIST datasets.
# This script keeps the 2D pipeline separate from the 3D pipeline.

import os
import time
import joblib
import numpy as np

from medmnist import INFO
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils import load_dataset, dataset_to_arrays, get_dataset_info
from src.features import is_3d
from src.cnn import train_cnn


MODELS_DIR = "models_2d"


def normalize_labels(y):
    """
    Convert labels from shape (N, 1) to shape (N,)
    for single-label classification.
    """
    y = np.asarray(y)

    if y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1)

    return y


def is_multi_label_target(y):
    """
    Some MedMNIST datasets are multi-label, such as chestmnist.
    Multi-label classification means one image can belong to multiple labels.
    """
    y = np.asarray(y)
    return y.ndim == 2 and y.shape[1] > 1


def compute_auc(y_true, y_probs, multi_label):
    """
    Compute AUC for binary, multi-class, and multi-label classification.
    """
    y_true = np.asarray(y_true)

    if multi_label:
        return roc_auc_score(y_true, y_probs, average="macro")

    y_true = normalize_labels(y_true)
    classes = np.unique(y_true)

    if len(classes) == 2:
        return roc_auc_score(y_true, y_probs[:, 1])

    return roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")


def save_2d_results(data_flag, auc, acc, duration, labels):
    """
    Save training results for one 2D dataset.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    result = {
        "dataset": data_flag,
        "auc": auc,
        "accuracy": acc,
        "method": "cnn2d",
        "duration": duration,
        "labels": labels,
    }

    joblib.dump(result, f"{MODELS_DIR}/{data_flag}_cnn2d_results.joblib")


def already_trained(data_flag):
    """
    Check whether this 2D dataset has already been trained.
    """
    return os.path.exists(f"{MODELS_DIR}/{data_flag}_cnn2d_results.joblib")


def load_2d_results(data_flag):
    """
    Load saved result for one 2D dataset.
    """
    return joblib.load(f"{MODELS_DIR}/{data_flag}_cnn2d_results.joblib")


def get_2d_dataset_flags():
    """
    Automatically find all 2D MedMNIST datasets.
    This keeps 2D and 3D datasets separate.
    """
    flags_2d = []

    for data_flag in INFO.keys():
        train_ds, _, _ = load_dataset(data_flag)
        X_train, _ = dataset_to_arrays(train_ds, "train", data_flag)

        if not is_3d(X_train):
            flags_2d.append(data_flag)

    return flags_2d


def train_single_2d(data_flag, epochs=30, batch_size=64, lr=3e-4):
    """
    Train one 2D dataset using CNN.
    The program automatically:
    1. Loads the dataset
    2. Extracts labels
    3. Splits train/val data
    4. Trains the CNN
    5. Computes AUC and accuracy
    6. Saves results
    """
    if already_trained(data_flag):
        print(f"\n[{data_flag}] Already trained. Loading saved result...")
        result = load_2d_results(data_flag)
        print(
            f"[{data_flag}] AUC: {result['auc']:.4f}, "
            f"Accuracy: {result['accuracy']:.4f}"
        )
        return result

    print("\n" + "=" * 60)
    print(f"Training 2D dataset: {data_flag}")
    print("=" * 60)

    info = get_dataset_info(data_flag)
    labels = info["label"]

    print("\nLabels in this dataset:")
    for label_id, label_name in labels.items():
        print(f"  {label_id}: {label_name}")

    train_ds, val_ds, _ = load_dataset(data_flag)

    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
    X_val, y_val = dataset_to_arrays(val_ds, "val", data_flag)

    if is_3d(X_train):
        raise ValueError(f"{data_flag} is a 3D dataset. This script only handles 2D datasets.")

    y_train = normalize_labels(y_train)
    y_val = normalize_labels(y_val)

    multi_label = is_multi_label_target(y_train)

    start_time = time.time()

    y_probs, y_preds = train_cnn(
        X_train,
        y_train,
        X_val,
        y_val,
        is_3d_data=False,
        multi_label=multi_label,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    auc = compute_auc(y_val, y_probs, multi_label)
    acc = accuracy_score(y_val, y_preds)

    elapsed = time.time() - start_time
    duration = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    save_2d_results(data_flag, auc, acc, duration, labels)

    print(f"\n[{data_flag}] Finished.")
    print(f"AUC:      {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Time:     {duration}")

    return {
        "dataset": data_flag,
        "auc": auc,
        "accuracy": acc,
        "method": "cnn2d",
        "duration": duration,
        "labels": labels,
    }


def train_all_2d():
    """
    Automatically train all 2D MedMNIST datasets.
    """
    print("=" * 64)
    print(" 2D Biomedical Image Classification Pipeline")
    print("=" * 64)

    flags_2d = get_2d_dataset_flags()

    print("\n2D datasets found:")
    for flag in flags_2d:
        print(f"  - {flag}")

    all_results = []

    for flag in flags_2d:
        result = train_single_2d(flag)
        all_results.append(result)

    print("\n" + "=" * 64)
    print("2D Training Summary")
    print("=" * 64)
    print(f"{'Dataset':<25} {'AUC':>8} {'Accuracy':>10} {'Method':>10} {'Time':>10}")
    print("-" * 64)

    for r in all_results:
        print(
            f"{r['dataset']:<25} "
            f"{r['auc']:>8.4f} "
            f"{r['accuracy']:>10.4f} "
            f"{r['method']:>10} "
            f"{r['duration']:>10}"
        )

    print("=" * 64)

    return all_results


if __name__ == "__main__":
    train_all_2d()