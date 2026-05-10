# src/viewers/visualize_noise_samples_2d.py
# Visualize noise injection on one 2D MedMNIST dataset.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils import load_dataset, dataset_to_arrays
from noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
    save_2d_noise_comparison,
)


def main():
    data_flag = "pathmnist"

    print(f"Loading 2D dataset: {data_flag}")
    train_ds, _, _ = load_dataset(data_flag)

    X_train, _ = dataset_to_arrays(train_ds, "train", data_flag)

    samples = X_train[:8]

    gaussian = add_gaussian_noise(samples, std=0.10)
    salt_pepper = add_salt_pepper_noise(samples, amount=0.05)
    speckle = add_speckle_noise(samples, std=0.10)

    save_2d_noise_comparison(
        samples,
        gaussian,
        "2D Gaussian Noise Example",
        "noise_results_2d/gaussian_noise_example.png",
    )

    save_2d_noise_comparison(
        samples,
        salt_pepper,
        "2D Salt-and-Pepper Noise Example",
        "noise_results_2d/salt_pepper_noise_example.png",
    )

    save_2d_noise_comparison(
        samples,
        speckle,
        "2D Speckle Noise Example",
        "noise_results_2d/speckle_noise_example.png",
    )

    print("2D noise visualization complete.")


if __name__ == "__main__":
    main()