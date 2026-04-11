# train.py - Main training script for the biomedical image classifier.
# This is where we load the datasets, extract features, train the logistic regression model, and evaluate it. 
# We use joblib to run the training in parallel across all datasets, which should speed things up significantly.

import warnings
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
from medmnist import INFO
from tqdm import tqdm

from utils import load_dataset, dataset_to_arrays
from features import extract_features

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MAX_TRAIN_SAMPLES = 20000

def train_single(data_flag):
    try:
        print(f"  [{data_flag}] Loading data...")
        train_ds, val_ds, test_ds = load_dataset(data_flag)
        X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
        X_val,   y_val   = dataset_to_arrays(val_ds,   "val",   data_flag)
        X_test,  y_test  = dataset_to_arrays(test_ds,  "test",  data_flag)

        # Subsample if dataset is too large
        if len(X_train) > MAX_TRAIN_SAMPLES:
            print(f"  [{data_flag}] Subsampling {len(X_train)} → {MAX_TRAIN_SAMPLES} samples...")
            idx = np.random.choice(len(X_train), MAX_TRAIN_SAMPLES, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        print(f"  [{data_flag}] Extracting features...")
        X_train_pca, X_val_pca, X_test_pca = extract_features(
            X_train, X_val, X_test, n_components=100
        )

        # Detect multi-label datasets
        multi_label = y_train.ndim == 2 and y_train.shape[1] > 1

        print(f"  [{data_flag}] Training...")
        start = time.time()
        if multi_label:
            clf = MultiOutputClassifier(
                LogisticRegression(C=1.0, max_iter=5000, solver="saga")
            )
        else:
            clf = LogisticRegression(C=1.0, max_iter=5000, solver="saga")
        clf.fit(X_train_pca, y_train)
        elapsed = time.time() - start

        # Evaluate
        if multi_label:
            y_score = np.column_stack([p[:, 1] for p in clf.predict_proba(X_val_pca)])
            auc = roc_auc_score(y_val, y_score, average="macro")
            acc = accuracy_score(y_val, clf.predict(X_val_pca))
        else:
            acc = accuracy_score(y_val, clf.predict(X_val_pca))
            n_classes = len(np.unique(y_val))
            if n_classes == 2:
                auc = roc_auc_score(y_val, clf.predict_proba(X_val_pca)[:, 1])
            else:
                auc = roc_auc_score(
                    y_val,
                    clf.predict_proba(X_val_pca),
                    multi_class="ovr",
                    average="macro"
                )

        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"  [{data_flag}] Done — AUC: {auc:.4f}, Accuracy: {acc:.4f} ({minutes}m {seconds}s)")
        return (data_flag, auc, acc, f"{minutes}m {seconds}s", None)

    except Exception as e:
        print(f"  [{data_flag}] ERROR: {e}")
        return (data_flag, None, None, None, str(e))

if __name__ == "__main__":
    all_flags = list(INFO.keys())
    total_start = time.time()

    print("=" * 58)
    print(" MedMNIST Classifier Pipeline")
    print("=" * 58)

    # Step 1
    print("\n[1/3] Preparing datasets...")
    for flag in tqdm(all_flags, desc="Checking datasets", unit="dataset"):
        load_dataset(flag)

    # Step 2
    print("\n[2/3] Training all datasets in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(train_single)(flag) for flag in all_flags
    )

    # Step 3
    total_elapsed = time.time() - total_start
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)

    print("\n[3/3] Results")
    print("=" * 58)
    print(f"{'Dataset':<25} {'AUC':>8} {'Accuracy':>10} {'Train Time':>12}")
    print("-" * 58)
    for flag, auc, acc, duration, err in sorted(results):
        if err:
            print(f"{flag:<25} ERROR: {err}")
        else:
            print(f"{flag:<25} {auc:>8.4f} {acc:>10.4f} {duration:>12}")
    print("=" * 58)
    print(f" Total time: {total_min}m {total_sec}s")
    print("=" * 58)