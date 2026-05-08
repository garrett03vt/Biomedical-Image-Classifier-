# src/overfitting_analysis.py
# Plot epoch vs training loss and epoch vs validation loss.
# This imports CNN models from cnn.py instead of modifying cnn.py.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils import load_dataset, dataset_to_arrays, get_dataset_info
from src.features import is_3d
from src.cnn import CNN2D, CNN3D, prepare_tensors_2d, prepare_tensors_3d


OUTPUT_DIR = "overfitting_results"


def normalize_labels(y):
    y = np.asarray(y)

    if y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1)

    return y


def is_multi_label_target(y):
    y = np.asarray(y)
    return y.ndim == 2 and y.shape[1] > 1


def get_num_classes(y, multi_label):
    y = np.asarray(y)

    if multi_label:
        return y.shape[1]

    return int(np.max(y)) + 1


def get_input_channels_2d(X):
    if X.ndim == 3:
        return 1
    return X.shape[-1]


def get_input_channels_3d(X):
    """
    Supports:
    (N, D, H, W)
    (N, C, D, H, W)
    (N, D, H, W, C)
    """
    X = np.asarray(X)

    if X.ndim == 4:
        return 1

    if X.ndim == 5:
        if X.shape[1] in (1, 3):
            return X.shape[1]
        if X.shape[-1] in (1, 3):
            return X.shape[-1]

    raise ValueError(f"Cannot infer channels from shape {X.shape}")


def batch_accuracy(outputs, labels, multi_label):
    """
    Compute accuracy for one batch.
    For multi-label datasets, this uses label-wise accuracy.
    For single-label datasets, this uses normal sample accuracy.
    """
    if multi_label:
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()
        correct = (preds == labels.int()).sum().item()
        total = labels.numel()
    else:
        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)

    return correct, total


def evaluate_model(model, val_loader, criterion, device, multi_label):
    """
    Evaluate validation loss and validation accuracy for one epoch.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.inference_mode():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()

            correct, count = batch_accuracy(outputs, y_batch, multi_label)
            total_correct += correct
            total_count += count

    avg_loss = total_loss / max(1, len(val_loader))
    accuracy = total_correct / max(1, total_count)

    return avg_loss, accuracy


def plot_curves(history, data_flag):
    """
    Save training vs validation loss and accuracy curves.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{data_flag}: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()

    loss_path = os.path.join(OUTPUT_DIR, f"{data_flag}_loss_curve.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Training Accuracy")
    plt.plot(epochs, history["val_acc"], marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{data_flag}: Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    acc_path = os.path.join(OUTPUT_DIR, f"{data_flag}_accuracy_curve.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()

    print(f"Saved loss curve to: {loss_path}")
    print(f"Saved accuracy curve to: {acc_path}")


def run_training_curve_analysis(data_flag="pathmnist", epochs=10, batch_size=64, lr=3e-4):
    """
    Train a CNN and record training/validation loss and accuracy per epoch.
    This works for both 2D and 3D MedMNIST datasets.
    """
    print("=" * 64)
    print(f"Training Curve Analysis: {data_flag}")
    print("=" * 64)

    get_dataset_info(data_flag)

    train_ds, val_ds, _ = load_dataset(data_flag)

    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
    X_val, y_val = dataset_to_arrays(val_ds, "val", data_flag)

    y_train = normalize_labels(y_train)
    y_val = normalize_labels(y_val)

    multi_label = is_multi_label_target(y_train)
    is_3d_data = is_3d(X_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    num_classes = get_num_classes(y_train, multi_label)

    if is_3d_data:
        in_channels = get_input_channels_3d(X_train)
        model = CNN3D(in_channels, num_classes).to(device)

        train_dataset = prepare_tensors_3d(
            X_train,
            y_train,
            multi_label=multi_label,
        )

        val_dataset = prepare_tensors_3d(
            X_val,
            y_val,
            multi_label=multi_label,
        )

        batch_size = min(batch_size, 16)

    else:
        in_channels = get_input_channels_2d(X_train)
        model = CNN2D(in_channels, num_classes).to(device)

        train_dataset = prepare_tensors_2d(
            X_train,
            y_train,
            multi_label=multi_label,
            augment=True,
        )

        val_dataset = prepare_tensors_2d(
            X_val,
            y_val,
            multi_label=multi_label,
            augment=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            correct, count = batch_accuracy(outputs, y_batch, multi_label)
            total_train_correct += correct
            total_train_count += count

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_acc = total_train_correct / max(1, total_train_count)

        val_loss, val_acc = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            multi_label,
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train loss: {avg_train_loss:.4f}, "
            f"val loss: {val_loss:.4f}, "
            f"train acc: {train_acc:.4f}, "
            f"val acc: {val_acc:.4f}"
        )

    plot_curves(history, data_flag)

    print("\nTraining curve analysis complete.")
    return history


def main():
    # Quick 2D test. You can change the dataset name here.
    run_training_curve_analysis(
        data_flag="pathmnist",
        epochs=10,
        batch_size=64,
        lr=3e-4,
    )

    # Example for 3D:
    # run_training_curve_analysis(
    #     data_flag="adrenalmnist3d",
    #     epochs=10,
    #     batch_size=16,
    #     lr=3e-4,
    # )


if __name__ == "__main__":
    main()