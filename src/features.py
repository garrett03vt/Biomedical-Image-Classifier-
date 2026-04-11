# features.py - Functions for feature extraction and dimensionality reduction.
# This is where we implement the HOG feature extraction and PCA dimensionality reduction. 
# The extract_features function will be called from the training code to process the datasets before training the logistic regression model.
import numpy as np
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage.color import rgb2gray


def extract_hog_features(img):
    if img.ndim == 3 and img.shape[-1] == 3:
        img = rgb2gray(img)
    elif img.ndim == 3:
        img = img.mean(axis=-1)

    img = img.astype(np.float64) / 255.0 if img.max() > 1.0 else img.astype(np.float64)

    return hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        feature_vector=True
    )


def extract_features_2d(X):
    return np.array([extract_hog_features(img) for img in X])


def is_3d(X):
    if X.ndim == 5:
        return True
    if X.ndim == 4:
        return X.shape[-1] not in (1, 3)
    return False


def extract_features_3d(X):
    all_features = []
    for volume in X:
        slice_features = [extract_hog_features(slc) for slc in volume]
        pooled = np.mean(slice_features, axis=0)
        all_features.append(pooled)
    return np.array(all_features)


def flatten_pixels(X):
    return X.reshape(X.shape[0], -1).astype(np.float32) / 255.0


def run_pca(X_train_feat, X_val_feat, X_test_feat, n_components):
    n = min(n_components, X_train_feat.shape[1])
    pca = PCA(n_components=n)

    return (
        pca.fit_transform(X_train_feat),
        pca.transform(X_val_feat),
        pca.transform(X_test_feat)
    )


def extract_features(X_train, X_val, X_test, n_components=100, method="flat"):
    if is_3d(X_train):
        return run_pca(
            extract_features_3d(X_train),
            extract_features_3d(X_val),
            extract_features_3d(X_test),
            n_components
        )

    if method == "hog":
        return run_pca(
            extract_features_2d(X_train),
            extract_features_2d(X_val),
            extract_features_2d(X_test),
            n_components
        )

    return run_pca(
        flatten_pixels(X_train),
        flatten_pixels(X_val),
        flatten_pixels(X_test),
        n_components
    )