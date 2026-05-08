# cnn.py - CNN model definitions and training function for 2D and 3D classification.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T


class CNN2D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNN3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def prepare_tensors_2d(X, y=None, multi_label=False, augment=False, strong_augment=False):
    X = np.asarray(X)

    if strong_augment:
        # Stronger augmentation for small / imbalanced datasets (retinamnist, fracturemnist3d)
        transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
        ])
    elif augment:
        transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
        ])
    else:
        transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
        ])

    X_list = []
    for img in X:
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        X_list.append(transform(img))

    X_tensor = torch.stack(X_list)

    if y is None:
        return X_tensor

    y = np.asarray(y)
    if not multi_label and y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)

    if multi_label:
        y_tensor = torch.from_numpy(y.astype(np.float32))
    else:
        y_tensor = torch.from_numpy(y.astype(np.int64).reshape(-1))

    return TensorDataset(X_tensor, y_tensor)


def prepare_tensors_3d(X, y=None, multi_label=False, augment=False):
    X = np.asarray(X, dtype=np.float32) / 255.0

    if X.ndim == 4:
        X = np.expand_dims(X, axis=1)
    elif X.ndim == 5 and X.shape[-1] in (1, 3):
        X = X.transpose(0, 4, 1, 2, 3)

    if augment:
        # Random horizontal flip along W axis for 3D volumes
        flip_mask = np.random.rand(X.shape[0]) > 0.5
        X[flip_mask] = X[flip_mask, :, :, :, ::-1].copy()

    X_tensor = torch.from_numpy(X)

    if y is None:
        return X_tensor

    y = np.asarray(y)
    if not multi_label and y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)

    if multi_label:
        y_tensor = torch.from_numpy(y.astype(np.float32))
    else:
        y_tensor = torch.from_numpy(y.astype(np.int64).reshape(-1))

    return TensorDataset(X_tensor, y_tensor)


def compute_pos_weight(y_train):
    """
    Compute per-label positive weights for BCEWithLogitsLoss to handle class imbalance.
    pos_weight[i] = (# negative samples) / (# positive samples) for label i.
    Used for chestmnist multi-label training.
    """
    y = np.asarray(y_train, dtype=np.float32)
    pos = y.sum(axis=0).clip(min=1)
    neg = (y.shape[0] - y.sum(axis=0)).clip(min=1)
    return torch.from_numpy(neg / pos).float()


def compute_class_weights(y_train, num_classes):
    """
    Inverse-frequency class weights for CrossEntropyLoss.
    Used for fracturemnist3d (imbalanced 3-class) and retinamnist (small ordinal dataset).
    """
    y = np.asarray(y_train).reshape(-1)
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = counts.clip(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalise so mean weight = 1
    return torch.from_numpy(weights).float()


def train_cnn(
    X_train,
    y_train,
    X_val,
    y_val,
    is_3d_data=False,
    multi_label=False,
    epochs=30,
    batch_size=64,
    lr=3e-4,
    # Special flags for problem datasets
    use_pos_weight=False,       # chestmnist: weighted BCE for label imbalance
    use_class_weights=False,    # fracturemnist3d / retinamnist: weighted CE
    strong_augment=False,       # retinamnist / fracturemnist3d: heavier augmentation
    label_smoothing=0.0,        # retinamnist: smooths ordinal targets slightly
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    CNN training on: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    y_train = np.asarray(y_train)
    y_val   = np.asarray(y_val)

    num_classes = y_train.shape[1] if multi_label else int(np.max(y_train)) + 1

    if is_3d_data:
        if X_train.ndim == 4:
            in_channels = 1
        elif X_train.ndim == 5:
            if X_train.shape[1] in (1, 3):
                in_channels = X_train.shape[1]
            elif X_train.shape[-1] in (1, 3):
                in_channels = X_train.shape[-1]
            else:
                raise ValueError(f"Cannot infer 3D channels from shape {X_train.shape}")
        else:
            raise ValueError(f"Unsupported 3D input shape: {X_train.shape}")

        model    = CNN3D(in_channels, num_classes).to(device)
        train_ds = prepare_tensors_3d(X_train, y_train, multi_label=multi_label, augment=strong_augment)
        val_ds   = prepare_tensors_3d(X_val,   y_val,   multi_label=multi_label, augment=False)
    else:
        in_channels = 1 if X_train.ndim == 3 else X_train.shape[-1]
        model    = CNN2D(in_channels, num_classes).to(device)
        train_ds = prepare_tensors_2d(X_train, y_train, multi_label=multi_label,
                                      augment=not strong_augment, strong_augment=strong_augment)
        val_ds   = prepare_tensors_2d(X_val,   y_val,   multi_label=multi_label,
                                      augment=False, strong_augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # Loss function selection
    if multi_label:
        if use_pos_weight:
            pw = compute_pos_weight(y_train).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            print(f"    Using weighted BCEWithLogitsLoss (pos_weight range: {pw.min():.2f}–{pw.max():.2f})")
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        if use_class_weights:
            cw = compute_class_weights(y_train, num_classes).to(device)
            criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
            print(f"    Using weighted CrossEntropyLoss (weights: {cw.cpu().numpy().round(3)})")
        elif label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            print(f"    Using CrossEntropyLoss with label_smoothing={label_smoothing}")
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # CosineAnnealingLR gives better final performance than StepLR for small/imbalanced sets
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out  = model(X_batch)
                loss = criterion(out, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"    Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

    model.eval()
    all_probs, all_preds = [], []

    with torch.inference_mode():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            out     = model(X_batch)

            if multi_label:
                probs = torch.sigmoid(out).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            else:
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

            all_probs.append(probs)
            all_preds.append(preds)

    y_probs = np.concatenate(all_probs, axis=0)
    y_preds = np.concatenate(all_preds, axis=0)

    return y_probs, y_preds