# cnn.py - CNN model definitions and training function for 2D and 3D classification.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T


# Standard 2D CNN
class CNN2D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

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


# Standard 3D CNN
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


# Data prep — 2D

def prepare_tensors_2d(X, y=None, multi_label=False, augment=False, strong_augment=False):
    X = np.asarray(X)

    # Note: Using 0.5 for mean/std is standard for grayscale MedMNIST images
    normalize = T.Normalize(mean=[0.5], std=[0.5])

    if strong_augment:
        transform = T.Compose([
            T.ToPILImage(),
            T.Pad(4),                     # Requested: pad by 4 pixels
            T.RandomCrop(32),             # Requested: random crop 32x32
            T.RandomHorizontalFlip(p=0.5),# Requested: 50% horizontal flip
            T.ToTensor(),
            normalize                     # Requested: Data Normalization
        ])
    elif augment:
        transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            normalize
        ])
    else:
        transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            normalize                     # Must apply to val set too!
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


# Data prep — 3D

def prepare_tensors_3d(X, y=None, multi_label=False, augment=False):
    X = np.asarray(X, dtype=np.float32) / 255.0

    if X.ndim == 4:
        X = np.expand_dims(X, axis=1)
    elif X.ndim == 5 and X.shape[-1] in (1, 3):
        X = X.transpose(0, 4, 1, 2, 3)

    if augment:
        for ax in (2, 3, 4):
            mask = np.random.rand(X.shape[0]) > 0.5
            X[mask] = np.flip(X[mask], axis=ax).copy()
        noise = np.random.normal(0, 0.02, X.shape).astype(np.float32)
        X = np.clip(X + noise, 0.0, 1.0)

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


# Loss helpers

def compute_pos_weight(y_train):
    # Cap at 10 to prevent extreme weights on very rare labels destabilising training
    y = np.asarray(y_train, dtype=np.float32)
    pos = y.sum(axis=0).clip(min=1)
    neg = (y.shape[0] - y.sum(axis=0)).clip(min=1)
    return torch.from_numpy((neg / pos).clip(max=10.0)).float()


def compute_class_weights(y_train, num_classes):
    y = np.asarray(y_train).reshape(-1)
    counts = np.bincount(y, minlength=num_classes).astype(np.float32).clip(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights).float()


# Per-label threshold tuning — chestmnist
# Sweeps thresholds 0.05–0.95 for each of the 14 labels and picks the one
# that maximises that label's F1 on the validation set.
# This replaces the fixed 0.5 threshold which collapses accuracy on rare labels.

def tune_thresholds(y_true, y_probs, n_steps=50):
    thresholds = np.linspace(0.05, 0.95, n_steps)
    n_labels   = y_probs.shape[1]
    best_t     = np.full(n_labels, 0.5)

    for j in range(n_labels):
        best_f1 = -1.0
        for t in thresholds:
            preds = (y_probs[:, j] >= t).astype(int)
            tp    = int(((preds == 1) & (y_true[:, j] == 1)).sum())
            fp    = int(((preds == 1) & (y_true[:, j] == 0)).sum())
            fn    = int(((preds == 0) & (y_true[:, j] == 1)).sum())
            denom = 2 * tp + fp + fn
            f1    = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1   = f1
                best_t[j] = t

    return best_t


# Main training function.
# Returns: y_probs, y_preds, train_losses, val_losses

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
    use_pos_weight=False,
    use_class_weights=False,
    strong_augment=False,
    label_smoothing=0.0,
    tune_threshold=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    CNN training on: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    y_train = np.asarray(y_train)
    y_val   = np.asarray(y_val)

    num_classes = y_train.shape[1] if multi_label else int(np.max(y_train)) + 1

    # Data Preparation
    if is_3d_data:
        # (Assuming prepare_tensors_3d is updated with normalization if needed)
        model    = CNN3D(in_channels=1, num_classes=num_classes).to(device)
        train_ds = prepare_tensors_3d(X_train, y_train, multi_label=multi_label, augment=strong_augment)
        val_ds   = prepare_tensors_3d(X_val,   y_val,   multi_label=multi_label, augment=False)
    else:
        in_channels = 1 if X_train.ndim == 3 else X_train.shape[-1]
        model    = CNN2D(in_channels, num_classes).to(device)
        # prepare_tensors_2d now handles the Pad/Crop/Flip/Normalize logic
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

    # Loss Function Configuration
    if multi_label:
        if use_pos_weight:
            pw = compute_pos_weight(y_train).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        if use_class_weights:
            cw = compute_class_weights(y_train, num_classes).to(device)
            criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer & Scheduler (Changes implemented here)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # One Cycle Learning Rate Policy
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs,
        pct_start=0.3 # Spends 30% of time warming up
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    train_losses = []
    val_losses   = []

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

            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Step scheduler after every BATCH
            scheduler.step()
            
            total_loss += loss.item()

        avg_train = total_loss / max(1, len(train_loader))
        train_losses.append(avg_train)

        # Validation Step
        model.eval()
        v_loss = 0.0
        with torch.inference_mode():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    out  = model(X_batch)
                    loss = criterion(out, y_batch)
                v_loss += loss.item()

        avg_val = v_loss / max(1, len(val_loader))
        val_losses.append(avg_val)

        print(f"    Epoch {epoch + 1}/{epochs}  train={avg_train:.4f}  val={avg_val:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

    # Final Inference for Evaluation
    model.eval()
    all_probs, all_preds = [], []

    with torch.inference_mode():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            out     = model(X_batch)
            if multi_label:
                probs = torch.sigmoid(out).cpu().numpy()
            else:
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                all_preds.append(preds)
            all_probs.append(probs)

    y_probs = np.concatenate(all_probs, axis=0)

    if multi_label:
        if tune_threshold:
            thresholds = tune_thresholds(y_val, y_probs)
            y_preds = (y_probs >= thresholds).astype(int)
        else:
            y_preds = (y_probs > 0.5).astype(int)
    else:
        y_preds = np.concatenate(all_preds, axis=0)

    return y_probs, y_preds, train_losses, val_losses