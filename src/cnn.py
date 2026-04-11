import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CNN2D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
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
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def prepare_tensors_2d(X, y=None, multi_label=False):
    X = np.asarray(X, dtype=np.float32) / 255.0
    if X.ndim == 3:
        X = np.expand_dims(X, axis=1)      # (N, 1, H, W)
    else:
        X = X.transpose(0, 3, 1, 2)        # (N, C, H, W)

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


def prepare_tensors_3d(X, y=None, multi_label=False):
    X = np.asarray(X, dtype=np.float32) / 255.0
    if X.ndim == 4:
        X = np.expand_dims(X, axis=1)      # (N, 1, D, H, W)
    else:
        X = X.transpose(0, 4, 1, 2, 3)     # (N, C, D, H, W)

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


def train_cnn(
    X_train,
    y_train,
    X_val,
    y_val,
    is_3d_data=False,
    multi_label=False,
    epochs=20,
    batch_size=64,
    lr=1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    CNN training on: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    num_classes = y_train.shape[1] if multi_label else int(np.max(y_train)) + 1

    if is_3d_data:
        in_channels = 1 if X_train.ndim == 4 else X_train.shape[-1]
        model = CNN3D(in_channels, num_classes).to(device)
        train_ds = prepare_tensors_3d(X_train, y_train, multi_label=multi_label)
        val_ds = prepare_tensors_3d(X_val, y_val, multi_label=multi_label)
    else:
        in_channels = 1 if X_train.ndim == 3 else X_train.shape[-1]
        model = CNN2D(in_channels, num_classes).to(device)
        train_ds = prepare_tensors_2d(X_train, y_train, multi_label=multi_label)
        val_ds = prepare_tensors_2d(X_val, y_val, multi_label=multi_label)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(X_batch)
                loss = criterion(out, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"    Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

    model.eval()
    all_probs = []
    all_preds = []

    with torch.inference_mode():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            out = model(X_batch)

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