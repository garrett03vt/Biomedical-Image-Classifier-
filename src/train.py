# train.py - Main training script for the biomedical image classifier.
# This is where we load the datasets, extract features, train the logistic regression model, and evaluate it. 
# We use joblib to run the training in parallel across all datasets, which should speed things up significantly.
import os
import time
import warnings
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from medmnist import INFO
from tqdm import tqdm

from utils import load_dataset, dataset_to_arrays
from features import extract_features, is_3d
from cnn import train_cnn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODELS_DIR = "models"


def save_results(data_flag, auc, acc, best_method, duration):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        {"auc": auc, "acc": acc, "method": best_method, "duration": duration},
        f"{MODELS_DIR}/{data_flag}_results.joblib",
    )


def load_results(data_flag):
    r = joblib.load(f"{MODELS_DIR}/{data_flag}_results.joblib")
    return r["auc"], r["acc"], r["method"], r["duration"]


def already_trained(data_flag):
    return os.path.exists(f"{MODELS_DIR}/{data_flag}_results.joblib")


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


def probs_from_model(clf, Xva, multi_label):
    probs = clf.predict_proba(Xva)
    if multi_label:
        # MultiOutputClassifier returns a list of per-label probability arrays
        return np.column_stack([p[:, 1] for p in probs])
    return probs


def train_logistic(Xtr, y_train, Xva, multi_label, max_iter):
    print("    Training logistic regression...")

    if multi_label:
        base = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        solver="saga",
                        max_iter=max_iter,
                        tol=1e-3,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        clf = MultiOutputClassifier(base, n_jobs=-1)
    else:
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=max_iter,
                        tol=1e-3,
                        random_state=42,
                    ),
                ),
            ]
        )

    clf.fit(Xtr, y_train)
    probs = probs_from_model(clf, Xva, multi_label)
    preds = clf.predict(Xva)
    return probs, preds


def train_random_forest(Xtr, y_train, Xva, multi_label):
    print("    Training random forest...")

    base = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    if multi_label:
        clf = MultiOutputClassifier(base, n_jobs=1)
    else:
        clf = base

    clf.fit(Xtr, y_train)
    probs = probs_from_model(clf, Xva, multi_label)
    preds = clf.predict(Xva)
    return probs, preds


def train_classical_models(X_train, y_train, X_val, y_val, method, multi_label, max_iter):
    Xtr, Xva, _ = extract_features(
        X_train,
        X_val,
        X_val,
        n_components=100,
        method=method,
    )

    results = {}

    print(f"    [{method}] Logistic regression...")
    log_probs, log_preds = train_logistic(Xtr, y_train, Xva, multi_label, max_iter)
    results[f"{method}+logistic"] = (
        compute_auc(y_val, log_probs, multi_label),
        accuracy_score(y_val, log_preds),
    )

    print(f"    [{method}] Random forest...")
    rf_probs, rf_preds = train_random_forest(Xtr, y_train, Xva, multi_label)
    results[f"{method}+rf"] = (
        compute_auc(y_val, rf_probs, multi_label),
        accuracy_score(y_val, rf_preds),
    )

    return results


def train_single(data_flag):
    if already_trained(data_flag):
        print(f"  [{data_flag}] Already trained — loading saved results...")
        auc, acc, best_method, duration = load_results(data_flag)
        print(f"  [{data_flag}] AUC: {auc:.4f}, Accuracy: {acc:.4f} ({best_method})")
        return (data_flag, auc, acc, best_method, duration, None)

    try:
        print(f"\n{'='*50}")
        print(f"  [{data_flag}] Starting...")
        print(f"{'='*50}")

        print(f"  [{data_flag}] Loading data...")
        train_ds, val_ds, _ = load_dataset(data_flag)

        X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
        X_val, y_val = dataset_to_arrays(val_ds, "val", data_flag)

        y_train = normalize_labels(y_train)
        y_val = normalize_labels(y_val)

        multi_label = is_multi_label_target(y_train)
        is_3d_data = is_3d(X_train)

        max_iter = 10000 if data_flag in {
            "organamnist",
            "organcmnist",
            "organsmnist",
            "chestmnist",
        } else 3000

        start = time.time()
        candidates = []

        if is_3d_data:
            print(f"  [{data_flag}] 3D data — running CNN...")
            cnn_probs, cnn_preds = train_cnn(
                X_train,
                y_train,
                X_val,
                y_val,
                is_3d_data=True,
                multi_label=multi_label,
            )
            cnn_auc = compute_auc(y_val, cnn_probs, multi_label)
            cnn_acc = accuracy_score(y_val, cnn_preds)
            candidates.append((cnn_auc, cnn_acc, "cnn3d"))

        else:
            print(f"  [{data_flag}] Running flat features...")
            flat_results = train_classical_models(
                X_train, y_train, X_val, y_val, "flat", multi_label, max_iter
            )
            for name, (auc, acc) in flat_results.items():
                candidates.append((auc, acc, name))

            print(f"  [{data_flag}] Running HOG features...")
            hog_results = train_classical_models(
                X_train, y_train, X_val, y_val, "hog", multi_label, max_iter
            )
            for name, (auc, acc) in hog_results.items():
                candidates.append((auc, acc, name))

            print(f"    [cnn] Running CNN once...")
            cnn_probs, cnn_preds = train_cnn(
                X_train,
                y_train,
                X_val,
                y_val,
                is_3d_data=False,
                multi_label=multi_label,
            )
            cnn_auc = compute_auc(y_val, cnn_probs, multi_label)
            cnn_acc = accuracy_score(y_val, cnn_preds)
            candidates.append((cnn_auc, cnn_acc, "cnn"))

        best_auc, best_acc, best_method = max(candidates, key=lambda x: x[0])

        elapsed = time.time() - start
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        duration = f"{minutes}m {seconds}s"

        print(
            f"  [{data_flag}] Done — {best_method} won — "
            f"AUC: {best_auc:.4f}, Accuracy: {best_acc:.4f} ({duration})"
        )

        save_results(data_flag, best_auc, best_acc, best_method, duration)
        return (data_flag, best_auc, best_acc, best_method, duration, None)

    except Exception as e:
        print(f"  [{data_flag}] ERROR: {e}")
        return (data_flag, None, None, None, None, str(e))


if __name__ == "__main__":
    all_flags = list(INFO.keys())
    total_start = time.time()

    print("=" * 64)
    print(" MedMNIST Classifier Pipeline")
    print("=" * 64)

    print("\n[1/3] Preparing datasets...")
    for flag in tqdm(all_flags, desc="Checking datasets", unit="dataset"):
        load_dataset(flag)

    print("\n[2/3] Training all datasets sequentially...")
    results = []
    for flag in all_flags:
        result = train_single(flag)
        results.append(result)

    total_elapsed = time.time() - total_start
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)

    print("\n[3/3] Results")
    print("=" * 64)
    print(f"{'Dataset':<25} {'AUC':>8} {'Accuracy':>10} {'Method':>16} {'Train Time':>12}")
    print("-" * 64)
    for flag, auc, acc, method, duration, err in sorted(results):
        if err:
            print(f"{flag:<25} ERROR: {err}")
        else:
            print(f"{flag:<25} {auc:>8.4f} {acc:>10.4f} {method:>16} {duration:>12}")
    print("=" * 64)
    print(f" Total time: {total_min}m {total_sec}s")
    print("=" * 64)