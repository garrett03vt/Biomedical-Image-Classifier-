# features.py - Functions for feature extraction and dimensionality reduction.
# This is where we implement the HOG feature extraction and PCA dimensionality reduction. 
# The extract_features function will be called from the training code to process the datasets before training the logistic regression model.

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage.color import rgb2gray

def extract_hog_features(img):
    """Extract HOG features from a single 2D image."""
    if img.ndim == 3 and img.shape[-1] == 3:
        img = rgb2gray(img)
    elif img.ndim == 3:
        img = img.mean(axis=-1)
    img = img.astype(np.float64) / 255.0 if img.max() > 1.0 else img.astype(np.float64)
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),   # smaller cells for 28x28 images
        cells_per_block=(2, 2),
        feature_vector=True
    )
    return features

def extract_features_2d(X):
    """Extract HOG features from a batch of 2D images."""
    return np.array([extract_hog_features(img) for img in X])

def is_3d(X):
    """
    3D volumetric data has shape (N, D, H, W) or (N, D, H, W, C).
    MedMNIST 3D datasets are (N, 28, 28, 28) — 4 dims with equal spatial dims.
    2D datasets are (N, 28, 28) or (N, 28, 28, 3).
    """
    if X.ndim == 5:
        return True
    if X.ndim == 4:
        # If last dim is 1 or 3 it's a 2D image with channels, otherwise 3D volume
        return X.shape[-1] not in (1, 3)
    return False

def extract_features_3d(X):
    """
    Extract HOG features from 3D volumes slice by slice then average.
    X shape: (N, D, H, W) or (N, D, H, W, C)
    """
    all_features = []
    for volume in X:
        slice_features = []
        for slc in volume:
            feat = extract_hog_features(slc)
            slice_features.append(feat)
        pooled = np.mean(slice_features, axis=0)
        all_features.append(pooled)
    return np.array(all_features)

def extract_features(X_train, X_val, X_test, n_components=100):
    print("    Extracting HOG features...")

    if is_3d(X_train):
        print("    Detected 3D data — processing slice by slice...")
        X_train_feat = extract_features_3d(X_train)
        X_val_feat   = extract_features_3d(X_val)
        X_test_feat  = extract_features_3d(X_test)
    else:
        X_train_feat = extract_features_2d(X_train)
        X_val_feat   = extract_features_2d(X_val)
        X_test_feat  = extract_features_2d(X_test)

    print(f"    Feature shape: {X_train_feat.shape}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_val_scaled   = scaler.transform(X_val_feat)
    X_test_scaled  = scaler.transform(X_test_feat)

    # PCA
    n_components = min(n_components, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    print(f"    Reduced to {X_train_pca.shape[1]} PCA components")

    return X_train_pca, X_val_pca, X_test_pca