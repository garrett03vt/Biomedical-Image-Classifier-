# src/bias_analysis_2d.py
# Analyze label distribution for 2D MedMNIST datasets.

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from medmnist import INFO

from src.utils import load_dataset, dataset_to_arrays, get_dataset_info
from src.features import is_3d


OUTPUT_DIR = "bias_results_2d"


def get_2d_dataset_flags():
    flags_2d = []

    for data_flag in INFO.keys():
        train_ds, _, _ = load_dataset(data_flag)
        X_train, _ = dataset_to_arrays(train_ds, "train", data_flag)

        if not is_3d(X_train):
            flags_2d.append(data_flag)

    return flags_2d


def count_single_label(y):
    y = np.asarray(y).reshape(-1)
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique.astype(int), counts.astype(int)))


def count_multi_label(y):
    y = np.asarray(y)
    positive_counts = y.sum(axis=0).astype(int)
    negative_counts = y.shape[0] - positive_counts
    return positive_counts, negative_counts


def save_single_label_csv(data_flag, label_counts, labels):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{data_flag}_label_distribution.csv")

    total = sum(label_counts.values())

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Label ID", "Label Name", "Count", "Percentage"])

        for label_id, count in sorted(label_counts.items()):
            label_name = labels.get(str(label_id), str(label_id))
            percentage = count / total * 100
            writer.writerow([label_id, label_name, count, f"{percentage:.2f}%"])

    print(f"Saved CSV: {csv_path}")


def save_multi_label_csv(data_flag, positive_counts, negative_counts, labels):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{data_flag}_label_distribution.csv")

    total = positive_counts[0] + negative_counts[0]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Label ID", "Label Name", "Positive Count", "Negative Count", "Positive Percentage"])

        for label_id in range(len(positive_counts)):
            label_name = labels.get(str(label_id), str(label_id))
            pos = int(positive_counts[label_id])
            neg = int(negative_counts[label_id])
            percentage = pos / total * 100
            writer.writerow([label_id, label_name, pos, neg, f"{percentage:.2f}%"])

    print(f"Saved CSV: {csv_path}")


def plot_single_label_distribution(data_flag, label_counts, labels):
    label_ids = sorted(label_counts.keys())
    label_names = [labels.get(str(i), str(i)) for i in label_ids]
    counts = [label_counts[i] for i in label_ids]

    plt.figure(figsize=(12, 6))
    plt.bar(label_names, counts)
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.title(f"{data_flag} Label Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"{data_flag}_label_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved graph: {output_path}")


def plot_multi_label_distribution(data_flag, positive_counts, labels):
    label_ids = list(range(len(positive_counts)))
    label_names = [labels.get(str(i), str(i)) for i in label_ids]

    plt.figure(figsize=(12, 6))
    plt.bar(label_names, positive_counts)
    plt.xlabel("Disease / Condition Label")
    plt.ylabel("Number of Positive Samples")
    plt.title(f"{data_flag} Positive Label Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"{data_flag}_positive_label_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved graph: {output_path}")


def analyze_one_dataset(data_flag):
    print("\n" + "=" * 60)
    print(f"Analyzing 2D dataset: {data_flag}")
    print("=" * 60)

    info = get_dataset_info(data_flag)
    labels = info["label"]

    train_ds, _, _ = load_dataset(data_flag)
    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)

    if is_3d(X_train):
        print(f"Skipping {data_flag}: this is a 3D dataset.")
        return

    y_train = np.asarray(y_train)

    if y_train.ndim == 2 and y_train.shape[1] > 1:
        positive_counts, negative_counts = count_multi_label(y_train)

        print("\nPositive label counts:")
        for label_id, count in enumerate(positive_counts):
            label_name = labels.get(str(label_id), str(label_id))
            print(f"  {label_id}: {label_name:<25} {count}")

        save_multi_label_csv(data_flag, positive_counts, negative_counts, labels)
        plot_multi_label_distribution(data_flag, positive_counts, labels)

    else:
        label_counts = count_single_label(y_train)

        print("\nClass counts:")
        for label_id, count in sorted(label_counts.items()):
            label_name = labels.get(str(label_id), str(label_id))
            print(f"  {label_id}: {label_name:<25} {count}")

        save_single_label_csv(data_flag, label_counts, labels)
        plot_single_label_distribution(data_flag, label_counts, labels)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    flags_2d = get_2d_dataset_flags()

    print("\n2D datasets found:")
    for flag in flags_2d:
        print(f"  - {flag}")

    for flag in flags_2d:
        analyze_one_dataset(flag)

    print("\n2D bias analysis complete.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()