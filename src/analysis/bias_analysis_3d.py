# src/bias_analysis_3d.py
# Analyze label distribution for 3D MedMNIST datasets.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from medmnist import INFO

from src.utils import load_dataset, dataset_to_arrays, get_dataset_info
from src.features import is_3d


OUTPUT_DIR = "bias_results_3d"


def get_3d_dataset_flags():
    flags_3d = []

    for data_flag in INFO.keys():
        train_ds, _, _ = load_dataset(data_flag)
        X_train, _ = dataset_to_arrays(train_ds, "train", data_flag)

        if is_3d(X_train):
            flags_3d.append(data_flag)

    return flags_3d


def count_single_label(y):
    y = np.asarray(y).reshape(-1)
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique.astype(int), counts.astype(int)))


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


def plot_single_label_distribution(data_flag, label_counts, labels):
    label_ids = sorted(label_counts.keys())
    label_names = [labels.get(str(i), str(i)) for i in label_ids]
    counts = [label_counts[i] for i in label_ids]

    plt.figure(figsize=(12, 6))
    plt.bar(label_names, counts)
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.title(f"{data_flag} 3D Label Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"{data_flag}_label_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved graph: {output_path}")


def analyze_one_dataset(data_flag):
    print("\n" + "=" * 60)
    print(f"Analyzing 3D dataset: {data_flag}")
    print("=" * 60)

    info = get_dataset_info(data_flag)
    labels = info["label"]

    train_ds, _, _ = load_dataset(data_flag)
    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)

    if not is_3d(X_train):
        print(f"Skipping {data_flag}: this is a 2D dataset.")
        return

    label_counts = count_single_label(y_train)

    print("\nClass counts:")
    for label_id, count in sorted(label_counts.items()):
        label_name = labels.get(str(label_id), str(label_id))
        print(f"  {label_id}: {label_name:<25} {count}")

    save_single_label_csv(data_flag, label_counts, labels)
    plot_single_label_distribution(data_flag, label_counts, labels)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    flags_3d = get_3d_dataset_flags()

    print("\n3D datasets found:")
    for flag in flags_3d:
        print(f"  - {flag}")

    for flag in flags_3d:
        analyze_one_dataset(flag)

    print("\n3D bias analysis complete.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()