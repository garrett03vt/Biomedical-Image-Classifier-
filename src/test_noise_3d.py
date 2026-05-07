# src/test_noise_3d.py
# Visualize noise injection on one 3D MedMNIST dataset.

from src.utils import load_dataset, dataset_to_arrays
from src.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
    save_3d_noise_comparison,
)


def main():
    data_flag = "adrenalmnist3d"

    print(f"Loading 3D dataset: {data_flag}")
    train_ds, _, _ = load_dataset(data_flag)

    X_train, _ = dataset_to_arrays(train_ds, "train", data_flag)

    samples = X_train[:6]

    gaussian = add_gaussian_noise(samples, std=0.10)
    salt_pepper = add_salt_pepper_noise(samples, amount=0.05)
    speckle = add_speckle_noise(samples, std=0.10)

    save_3d_noise_comparison(
        samples,
        gaussian,
        "3D Gaussian Noise Example",
        "noise_results_3d/gaussian_noise_example.png",
    )

    save_3d_noise_comparison(
        samples,
        salt_pepper,
        "3D Salt-and-Pepper Noise Example",
        "noise_results_3d/salt_pepper_noise_example.png",
    )

    save_3d_noise_comparison(
        samples,
        speckle,
        "3D Speckle Noise Example",
        "noise_results_3d/speckle_noise_example.png",
    )

    print("3D noise visualization complete.")


if __name__ == "__main__":
    main()