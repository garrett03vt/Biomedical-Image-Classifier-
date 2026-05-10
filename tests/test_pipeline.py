# tests/test_pipeline.py
# Lightweight unit tests for the MedMNIST CNN pipeline.
#
# Run with:  pytest tests/
#
# These tests verify:
#   1. Noise functions preserve shape and dtype
#   2. Noise functions produce different outputs from inputs
#   3. CNN2D and CNN3D forward passes work end-to-end on tiny synthetic data
#   4. Data preparation utilities handle 2D and 3D inputs correctly
#   5. is_3d() correctly distinguishes 2D from 3D data
#
# These tests use synthetic data so they don't require downloading any
# MedMNIST datasets — they run in seconds and on CPU.

import os
import sys
import numpy as np
import pytest
import torch

# Path bootstrap so tests/ can import from src/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
    normalize_image_array,
    get_middle_slice,
)
from features import is_3d, extract_features
from cnn import CNN2D, CNN3D, prepare_tensors_2d, prepare_tensors_3d


# Noise function tests

class TestNoiseFunctions:
    def test_gaussian_preserves_shape(self):
        X = np.random.randint(0, 256, (8, 28, 28, 3), dtype=np.uint8)
        noisy = add_gaussian_noise(X, std=0.1)
        assert noisy.shape == X.shape

    def test_gaussian_preserves_dtype_uint8(self):
        X = np.random.randint(0, 256, (8, 28, 28, 3), dtype=np.uint8)
        noisy = add_gaussian_noise(X, std=0.1)
        assert noisy.dtype == np.uint8

    def test_gaussian_actually_modifies_image(self):
        X = np.full((4, 28, 28, 3), 128, dtype=np.uint8)
        noisy = add_gaussian_noise(X, std=0.1)
        # std=0 would mean identical; std=0.1 should yield different output
        assert not np.array_equal(X, noisy)

    def test_gaussian_zero_std_is_identity(self):
        X = np.random.randint(0, 256, (4, 28, 28, 3), dtype=np.uint8)
        noisy = add_gaussian_noise(X, std=0.0)
        np.testing.assert_array_equal(X, noisy)

    def test_salt_pepper_preserves_shape(self):
        X = np.random.randint(0, 256, (8, 28, 28), dtype=np.uint8)
        noisy = add_salt_pepper_noise(X, amount=0.1)
        assert noisy.shape == X.shape

    def test_salt_pepper_creates_extreme_values(self):
        # With amount=0.5, we expect lots of 0s and 255s
        X = np.full((4, 28, 28), 128, dtype=np.uint8)
        noisy = add_salt_pepper_noise(X, amount=0.5)
        # Should contain some 0 (pepper) and some 255 (salt)
        assert (noisy == 0).any()
        assert (noisy == 255).any()

    def test_speckle_preserves_shape(self):
        X = np.random.randint(0, 256, (8, 28, 28, 3), dtype=np.uint8)
        noisy = add_speckle_noise(X, std=0.1)
        assert noisy.shape == X.shape

    def test_works_on_3d_volumes(self):
        # 3D volume input: (N, D, H, W)
        X = np.random.randint(0, 256, (4, 28, 28, 28), dtype=np.uint8)
        for fn in [
            lambda x: add_gaussian_noise(x, std=0.1),
            lambda x: add_salt_pepper_noise(x, amount=0.05),
            lambda x: add_speckle_noise(x, std=0.1),
        ]:
            noisy = fn(X)
            assert noisy.shape == X.shape

    def test_normalize_image_array_to_unit_range(self):
        X = np.array([[0, 128, 255]], dtype=np.uint8)
        norm = normalize_image_array(X)
        assert norm.dtype == np.float32
        assert norm.min() >= 0.0 and norm.max() <= 1.0

    def test_get_middle_slice_3d(self):
        vol = np.random.rand(28, 28, 28).astype(np.float32)
        mid = get_middle_slice(vol)
        assert mid.shape == (28, 28)


# Feature / dimensionality tests

class TestFeatures:
    def test_is_3d_detects_grayscale_3d(self):
        # (N, D, H, W) — 3D grayscale
        X = np.random.rand(4, 28, 28, 28)
        assert is_3d(X)

    def test_is_3d_rejects_2d_grayscale(self):
        # (N, H, W) — 2D grayscale
        X = np.random.rand(4, 28, 28)
        assert not is_3d(X)

    def test_is_3d_rejects_2d_rgb(self):
        # (N, H, W, 3) — 2D RGB
        X = np.random.rand(4, 28, 28, 3)
        assert not is_3d(X)

    def test_is_3d_detects_5d_volume(self):
        # (N, C, D, H, W) — explicit 3D with channels
        X = np.random.rand(4, 1, 28, 28, 28)
        assert is_3d(X)


# CNN forward pass tests

class TestCNNForwardPass:
    def test_cnn2d_grayscale(self):
        model = CNN2D(in_channels=1, num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_cnn2d_rgb(self):
        model = CNN2D(in_channels=3, num_classes=9)
        x = torch.randn(2, 3, 28, 28)
        out = model(x)
        assert out.shape == (2, 9)

    def test_cnn3d_grayscale(self):
        model = CNN3D(in_channels=1, num_classes=2)
        x = torch.randn(2, 1, 28, 28, 28)
        out = model(x)
        assert out.shape == (2, 2)

    def test_cnn3d_rgb(self):
        model = CNN3D(in_channels=3, num_classes=11)
        x = torch.randn(2, 3, 28, 28, 28)
        out = model(x)
        assert out.shape == (2, 11)

    def test_cnn2d_gradient_flows(self):
        """Gradients reach all parameters — guards against accidentally
        disconnecting part of the graph."""
        model = CNN2D(in_channels=1, num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        y = torch.randint(0, 10, (2,))
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_cnn3d_gradient_flows(self):
        model = CNN3D(in_channels=1, num_classes=2)
        x = torch.randn(2, 1, 28, 28, 28)
        y = torch.randint(0, 2, (2,))
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# Data preparation tests

class TestDataPreparation:
    def test_prepare_tensors_2d_grayscale(self):
        X = np.random.randint(0, 256, (4, 28, 28), dtype=np.uint8)
        y = np.array([0, 1, 2, 3])
        ds = prepare_tensors_2d(X, y)
        assert len(ds) == 4
        x_one, y_one = ds[0]
        # After ToTensor + Normalize, values should be roughly in [-1, 1]
        assert x_one.shape == (1, 28, 28)
        assert isinstance(y_one.item(), int)

    def test_prepare_tensors_2d_rgb(self):
        X = np.random.randint(0, 256, (4, 28, 28, 3), dtype=np.uint8)
        y = np.array([0, 1, 2, 3])
        ds = prepare_tensors_2d(X, y)
        x_one, _ = ds[0]
        assert x_one.shape == (3, 28, 28)

    def test_prepare_tensors_2d_multi_label(self):
        X = np.random.randint(0, 256, (4, 28, 28), dtype=np.uint8)
        y = np.random.randint(0, 2, (4, 14)).astype(np.float32)
        ds = prepare_tensors_2d(X, y, multi_label=True)
        _, y_one = ds[0]
        assert y_one.shape == (14,)
        assert y_one.dtype == torch.float32

    def test_prepare_tensors_3d_grayscale(self):
        # Input (N, D, H, W) → channel-first (N, 1, D, H, W)
        X = np.random.randint(0, 256, (4, 28, 28, 28), dtype=np.uint8)
        y = np.array([0, 1, 0, 1])
        ds = prepare_tensors_3d(X, y)
        assert len(ds) == 4
        x_one, y_one = ds[0]
        assert x_one.shape == (1, 28, 28, 28)


# Synthetic end-to-end smoke test

class TestSmoke:
    def test_2d_pipeline_runs(self):
        """Tiny 2D end-to-end: data → model → loss → backward → step."""
        X = np.random.randint(0, 256, (8, 28, 28), dtype=np.uint8)
        y = np.random.randint(0, 3, 8)
        ds = prepare_tensors_2d(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        model = CNN2D(in_channels=1, num_classes=3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for x_batch, y_batch in loader:
            opt.zero_grad()
            out = model(x_batch)
            loss = torch.nn.functional.cross_entropy(out, y_batch)
            loss.backward()
            opt.step()
        assert True  # If we got here, the pipeline works

    def test_3d_pipeline_runs(self):
        X = np.random.randint(0, 256, (4, 28, 28, 28), dtype=np.uint8)
        y = np.random.randint(0, 2, 4)
        ds = prepare_tensors_3d(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        model = CNN3D(in_channels=1, num_classes=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for x_batch, y_batch in loader:
            opt.zero_grad()
            out = model(x_batch)
            loss = torch.nn.functional.cross_entropy(out, y_batch)
            loss.backward()
            opt.step()
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
