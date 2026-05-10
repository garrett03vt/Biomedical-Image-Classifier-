# 2Dviewer.py
# Explore any 2D MedMNIST dataset.
# Shows a 20×20 grid of sample images for a chosen label.
# No files are saved — the grid pops up on screen.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from utils import load_dataset, dataset_to_arrays, get_dataset_info
from train2d import DATASETS_2D

GRID_N = 20   # 20×20 = up to 400 images per label


def show_label_grid(X, y, info, label_id):
    labels    = info["label"]
    lname     = labels.get(str(label_id), str(label_id))
    indices   = [i for i in range(len(y)) if int(y[i]) == label_id]

    if not indices:
        print(f"  No images found for label {label_id} ({lname}).")
        return

    total   = GRID_N * GRID_N
    n_avail = len(indices)

    # Evenly sample up to GRID_N^2 images so the grid is representative
    if n_avail > total:
        step    = n_avail / total
        indices = [indices[int(i * step)] for i in range(total)]
    # If fewer than total, use what we have (remaining cells stay blank)

    n_show = len(indices)
    print(f"  Showing {n_show} images for label {label_id}: {lname}  (total available: {n_avail})")

    # Detect if images are grayscale or RGB
    sample = X[indices[0]]
    is_rgb = sample.ndim == 3 and sample.shape[-1] == 3

    cell_px  = 64
    fig_size = GRID_N * cell_px / 80
    fig, axes = plt.subplots(GRID_N, GRID_N, figsize=(fig_size, fig_size + 0.5), dpi=80)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n_show:
            img = X[indices[i]]
            if is_rgb:
                ax.imshow(img.astype(np.uint8))
            else:
                ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=255)

    fig.suptitle(
        f"{info.get('name', '')}  |  Label {label_id}: {lname}  "
        f"({n_show}/{n_avail} shown)",
        fontsize=10,
    )
    fig.tight_layout(pad=0.1)
    plt.show()


def main():
    print("=" * 52)
    print("  MedMNIST 2D Viewer — 20×20 Label Grid")
    print("=" * 52)

    print("\nAvailable 2D datasets:")
    flags = list(DATASETS_2D.keys())
    for i, flag in enumerate(flags):
        meta = DATASETS_2D[flag]
        print(f"  [{i:>2}] {flag:<20} {meta['task']}  ({meta['n_classes']} classes)")

    choice = input("\nEnter number or dataset name: ").strip()
    if choice.isdigit() and int(choice) < len(flags):
        data_flag = flags[int(choice)]
    elif choice in DATASETS_2D:
        data_flag = choice
    else:
        print(f"  Defaulting to pathmnist.")
        data_flag = "pathmnist"

    print(f"\n  Loading {data_flag} ...")
    train_ds, _, _ = load_dataset(data_flag)
    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
    info = get_dataset_info(data_flag)

    print(f"  Loaded {len(X_train)} images  |  shape: {X_train.shape}")
    print("\n  Labels:")
    for lid, lname in info["label"].items():
        count = int((y_train == int(lid)).sum())
        print(f"    [{lid}] {lname:<30} (n={count:,})")

    while True:
        choice = input("\n  Enter label ID to view (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            break
        if not choice.isdigit():
            print("  Please enter a valid integer label ID.")
            continue
        label_id = int(choice)
        if str(label_id) not in info["label"]:
            print(f"  Label {label_id} not found in this dataset.")
            continue
        show_label_grid(X_train, y_train, info, label_id)


if __name__ == "__main__":
    main()