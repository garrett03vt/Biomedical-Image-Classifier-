# utils.py - Utility functions for loading datasets and converting them to numpy arrays.
# this acts as a bridge between the medmnist datasets and the scikit-learn training code, allowing us to use the datasets without needing to rewrite them in PyTorch.

import numpy as np
import medmnist
from medmnist import INFO
from tqdm import tqdm

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