# utils.py - Utility functions for loading datasets and converting them to numpy arrays.
# this acts as a bridge between the medmnist datasets and the scikit-learn training code, allowing us to use the datasets without needing to rewrite them in PyTorch.

import numpy as np
import medmnist
from medmnist import INFO
from tqdm import tqdm
import matplotlib.pyplot as plt
import math


# Get datasets for a given flag (e.g., "pathmnist", "dermamnist", etc.)
def load_dataset(data_flag):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    train_ds = DataClass(split="train", download=True)
    val_ds   = DataClass(split="val",   download=True)
    test_ds  = DataClass(split="test",  download=True)

    return train_ds, val_ds, test_ds


# Turn datasets into numpy arrays. This is a bottleneck, but it allows us to use scikit-learn which is much faster for training than PyTorch for this task. We can optimize this later if needed.
def dataset_to_arrays(dataset, split_name="", flag=""):
    imgs, labels = [], []
    print(f"  [{flag}] Loading {split_name}...")
    for img, lbl in dataset:
        imgs.append(np.array(img))
        labels.append(lbl)
    X = np.array(imgs)
    y = np.array(labels)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    return X, y


# get dataset info
def get_dataset_info(data_flag):
    info = INFO[data_flag]

    print("\n=== DATASET INFO ===")
    print(f"Name: {data_flag}")
    print(f"Task: {info['task']}")
    print(f"Channels: {info['n_channels']}")
    print(f"Number of Classes: {len(info['label'])}")

    print("\nClass Labels:")
    for k, v in info['label'].items():
        print(f"  {k}: {v}")

    return info


# label distribution
def show_label_distribution(y, info):
    unique, counts = np.unique(y, return_counts=True)

    print("\n=== LABEL DISTRIBUTION ===")
    for u, c in zip(unique, counts):
        label_name = info['label'][str(u)] if isinstance(info['label'], dict) else str(u)
        print(f"{u} ({label_name}): {c}")


# visualize some samples from the dataset
def show_samples(X, y, info, num_samples=10):
    plt.figure(figsize=(12, 5))

    for i in range(num_samples):
        plt.subplot(2, num_samples // 2, i + 1)

        img = X[i]
        label = y[i]

        # Handle grayscale vs RGB
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)

        # Convert label to name
        label_name = info['label'][str(label)] if isinstance(info['label'], dict) else str(label)

        plt.title(f"{label}\n{label_name}", fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# viewer
def interactive_viewer(X, y, info):
    print("\n=== INTERACTIVE VIEWER ===")
    print("Press ENTER to go to next image, 'q' to quit.\n")

    i = 0
    while True:
        img = X[i]
        label = y[i]
        label_name = info['label'][str(label)] if isinstance(info['label'], dict) else str(label)

        plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
        plt.title(f"Index: {i} | Label: {label} ({label_name})")
        plt.axis('off')
        plt.show()

        cmd = input("Next? (ENTER/q): ")
        if cmd.lower() == 'q':
            break

        i = (i + 1) % len(X)


# filter by label
def filter_by_label(X, y, label_id):
    """
    Returns all images that match a given label.
    """
    mask = (y == label_id)
    return X[mask], y[mask]


# show all images for a label
def show_all_images_for_label(X, y, info, label_id, batch_size=64):
    """
    Shows ALL images for a specific label in paginated grids.
    """

    label_name = info['label'][str(label_id)] if isinstance(info['label'], dict) else str(label_id)

    X_label = X[y == label_id]

    print(f"\nFound {len(X_label)} images for label {label_id} ({label_name})")

    if len(X_label) == 0:
        print("No images found for this label.")
        return

    page = 0
    per_page = batch_size

    while True:
        start = page * per_page
        end = start + per_page
        batch = X_label[start:end]

        if len(batch) == 0:
            print("No more images.")
            break

        cols = int(math.sqrt(per_page))
        rows = int(math.ceil(len(batch) / cols))

        plt.figure(figsize=(12, 12))

        for i, img in enumerate(batch):
            plt.subplot(rows, cols, i + 1)

            if img.ndim == 2:
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)

            plt.axis("off")

        plt.suptitle(f"Label {label_id}: {label_name} | Page {page + 1}", fontsize=14)
        plt.tight_layout()
        plt.show()

        cmd = input("Next page? (ENTER/q): ")
        if cmd.lower() == "q":
            break

        page += 1