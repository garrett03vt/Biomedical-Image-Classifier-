# src/noise_evaluation_2d.py
# Train a 2D CNN on clean images, then evaluate on clean and noisy validation images.

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils import load_dataset, dataset_to_arrays, get_dataset_info
from src.cnn import CNN2D, prepare_tensors_2d
from src.noise import add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise


OUTPUT_DIR = "noise_results_2d"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_input_channels_2d(X):
    if X.ndim == 3:
        return 1

    return X.shape[-1]


def get_num_classes(y, multi_label):
    y = np.asarray(y)

    if multi_label:
        return y.shape[1]

    return int(np.max(y)) + 1


def train_clean_2d_model(X_train, y_train, multi_label, epochs=5, batch_size=64, lr=3e-4):
    in_channels = get_input_channels_2d(X_train)
    num_classes = get_num_classes(y_train, multi_label)

    model = CNN2D(in_channels, num_classes).to(DEVICE)

    train_ds = prepare_tensors_2d(
        X_train,
        y_train,
        multi_label=multi_label,
        augment=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training clean 2D CNN model on: {DEVICE}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    return model


def evaluate_2d_model(model, X_eval, y_eval, multi_label, batch_size=64):
    model.eval()

    eval_ds = prepare_tensors_2d(
        X_eval,
        y_eval,
        multi_label=multi_label,
        augment=False,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    all_probs = []
    all_preds = []

    with torch.inference_mode():
        for X_batch, _ in eval_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
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

    auc = compute_auc(y_eval, y_probs, multi_label)
    acc = accuracy_score(y_eval, y_preds)

    return auc, acc


def save_noise_results(data_flag, results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, f"{data_flag}_noise_evaluation_2d.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Condition", "AUC", "Accuracy"])

        for condition, auc, acc in results:
            writer.writerow([condition, f"{auc:.4f}", f"{acc:.4f}"])

    print(f"\nSaved 2D noise evaluation CSV to: {csv_path}")


def run_noise_evaluation_2d(data_flag="pathmnist", epochs=5):
    print("=" * 64)
    print(f"2D Noise Robustness Evaluation: {data_flag}")
    print("=" * 64)

    get_dataset_info(data_flag)

    train_ds, val_ds, _ = load_dataset(data_flag)

    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
    X_val, y_val = dataset_to_arrays(val_ds, "val", data_flag)

    y_train = normalize_labels(y_train)
    y_val = normalize_labels(y_val)

    multi_label = is_multi_label_target(y_train)

    start = time.time()

    model = train_clean_2d_model(
        X_train,
        y_train,
        multi_label=multi_label,
        epochs=epochs,
    )

    results = []

    print("\nEvaluating clean validation images...")
    clean_auc, clean_acc = evaluate_2d_model(model, X_val, y_val, multi_label)
    results.append(("Clean", clean_auc, clean_acc))

    print("Evaluating Gaussian noise validation images...")
    X_gaussian = add_gaussian_noise(X_val, std=0.10)
    gaussian_auc, gaussian_acc = evaluate_2d_model(model, X_gaussian, y_val, multi_label)
    results.append(("Gaussian Noise std=0.10", gaussian_auc, gaussian_acc))

    print("Evaluating salt-and-pepper noise validation images...")
    X_sp = add_salt_pepper_noise(X_val, amount=0.05)
    sp_auc, sp_acc = evaluate_2d_model(model, X_sp, y_val, multi_label)
    results.append(("Salt-Pepper Noise amount=0.05", sp_auc, sp_acc))

    print("Evaluating speckle noise validation images...")
    X_speckle = add_speckle_noise(X_val, std=0.10)
    speckle_auc, speckle_acc = evaluate_2d_model(model, X_speckle, y_val, multi_label)
    results.append(("Speckle Noise std=0.10", speckle_auc, speckle_acc))

    elapsed = time.time() - start

    print("\n2D Noise Evaluation Results")
    print("=" * 64)
    print(f"{'Condition':<35} {'AUC':>8} {'Accuracy':>10}")
    print("-" * 64)

    for condition, auc, acc in results:
        print(f"{condition:<35} {auc:>8.4f} {acc:>10.4f}")

    print("=" * 64)
    print(f"Total time: {int(elapsed // 60)}m {int(elapsed % 60)}s")

    save_noise_results(data_flag, results)


def main():
    run_noise_evaluation_2d(data_flag="pathmnist", epochs=5)


if __name__ == "__main__":
    main()