# src/noise.py
# Shared noise functions for 2D and 3D biomedical image data.

import os
import numpy as np
import matplotlib.pyplot as plt


def normalize_image_array(X):
    """
    Convert image array to float32 in range [0, 1].
    Supports 2D images and 3D volumes.
    """
    X = np.asarray(X).astype(np.float32)

    if X.max() > 1.0:
        X = X / 255.0

    return X


def restore_image_range(X_noisy, original_X):
    """
    Restore noisy array back to original range.
    If original images were 0-255, return uint8.
    If original images were 0-1, return float32.
    """
    if np.asarray(original_X).max() > 1.0:
        return np.clip(X_noisy * 255.0, 0, 255).astype(np.uint8)

    return np.clip(X_noisy, 0, 1).astype(np.float32)


def add_gaussian_noise(X, mean=0.0, std=0.1):
    """
    Add Gaussian noise.

    std:
    0.05 = light noise
    0.10 = medium noise
    0.20 = strong noise
    """
    X_norm = normalize_image_array(X)
    noise = np.random.normal(mean, std, X_norm.shape)
    X_noisy = X_norm + noise
    X_noisy = np.clip(X_noisy, 0, 1)

    return restore_image_range(X_noisy, X)


def add_salt_pepper_noise(X, amount=0.05):
    """
    Add salt-and-pepper noise.

    amount:
    0.01 = light noise
    0.05 = medium noise
    0.10 = strong noise
    """
    X_norm = normalize_image_array(X)
    X_noisy = X_norm.copy()

    random_values = np.random.random(X_norm.shape)

    salt_mask = random_values < amount / 2
    pepper_mask = (random_values >= amount / 2) & (random_values < amount)

    X_noisy[salt_mask] = 1.0
    X_noisy[pepper_mask] = 0.0

    return restore_image_range(X_noisy, X)


def add_speckle_noise(X, std=0.1):
    """
    Add speckle noise.
    Speckle noise is multiplicative:
    image + image * random_noise
    """
    X_norm = normalize_image_array(X)
    noise = np.random.normal(0.0, std, X_norm.shape)
    X_noisy = X_norm + X_norm * noise
    X_noisy = np.clip(X_noisy, 0, 1)

    return restore_image_range(X_noisy, X)


def get_middle_slice(volume):
    """
    Return the middle slice from one 3D volume.
    Supports:
    (D, H, W)
    (D, H, W, C)
    (C, D, H, W)
    """
    volume = np.asarray(volume)

    if volume.ndim == 3:
        return volume[volume.shape[0] // 2]

    if volume.ndim == 4:
        # Channel-first: (C, D, H, W)
        if volume.shape[0] in (1, 3):
            return volume[0, volume.shape[1] // 2]

        # Channel-last: (D, H, W, C)
        if volume.shape[-1] in (1, 3):
            return volume[volume.shape[0] // 2]

    raise ValueError(f"Unsupported 3D volume shape: {volume.shape}")


def save_2d_noise_comparison(original_images, noisy_images, title, output_path, num_samples=8):
    """
    Save clean vs noisy comparison for 2D images.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_samples = min(num_samples, len(original_images))

    plt.figure(figsize=(14, 4))

    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        img = original_images[i]

        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)

        plt.axis("off")
        if i == 0:
            plt.ylabel("Clean")

        plt.subplot(2, num_samples, num_samples + i + 1)
        noisy = noisy_images[i]

        if noisy.ndim == 2:
            plt.imshow(noisy, cmap="gray")
        else:
            plt.imshow(noisy)

        plt.axis("off")
        if i == 0:
            plt.ylabel("Noisy")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved 2D noise comparison to: {output_path}")


def save_3d_noise_comparison(original_volumes, noisy_volumes, title, output_path, num_samples=6):
    """
    Save clean vs noisy comparison for 3D volumes.
    This displays the middle slice of each volume.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_samples = min(num_samples, len(original_volumes))

    plt.figure(figsize=(14, 4))

    for i in range(num_samples):
        clean_slice = get_middle_slice(original_volumes[i])
        noisy_slice = get_middle_slice(noisy_volumes[i])

        plt.subplot(2, num_samples, i + 1)

        if clean_slice.ndim == 2:
            plt.imshow(clean_slice, cmap="gray")
        else:
            plt.imshow(clean_slice)

        plt.axis("off")
        if i == 0:
            plt.ylabel("Clean")

        plt.subplot(2, num_samples, num_samples + i + 1)

        if noisy_slice.ndim == 2:
            plt.imshow(noisy_slice, cmap="gray")
        else:
            plt.imshow(noisy_slice)

        plt.axis("off")
        if i == 0:
            plt.ylabel("Noisy")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved 3D noise comparison to: {output_path}")