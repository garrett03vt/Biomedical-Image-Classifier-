# cnn.py - CNN model definitions and training function for 2D and 3D biomedical image classification.
# This is where we define the CNN architectures for both 2D and 3D data, as well as the training loop. The train_cnn function will be called from the main training code to train the CNN on each dataset and evaluate its performance.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T


# For 2D data, we can use a simple CNN architecture with a few convolutional layers followed by fully connected layers. This should be sufficient for the relatively small and simple images in the medmnist datasets.
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


# For 3D data, we can use a similar architecture but with 3D convolutional layers. This will allow us to capture spatial features across the depth dimension of the volumes.
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


# The train_cnn function will handle the training loop, including data preparation, model training, and evaluation. It will return the predicted probabilities and class labels for the validation set, which can then be used to calculate AUC and accuracy.
def prepare_tensors_2d(X, y=None, multi_label=False, augment=False):
    X = np.asarray(X)

    # Define transforms
    if augment:
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

    # Apply transforms image-by-image
    X_list = []
    for img in X:
        if img.ndim == 2:  # grayscale
            img = np.expand_dims(img, axis=-1)

        img = transform(img)
        X_list.append(img)

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


# For 3D data, we need to ensure the input shape is correct for 3D convolutions. We also need to handle the labels similarly to the 2D case.
def prepare_tensors_3d(X, y=None, multi_label=False):
    X = np.asarray(X, dtype=np.float32) / 255.0
    
    if X.ndim == 4:
        X = np.expand_dims(X, axis=1)      # (N, 1, D, H, W)
    elif X.ndim == 5 and X.shape[-1] in (1, 3):
        # Only transpose if channels are explicitly at the end (N, D, H, W, C)
        X = X.transpose(0, 4, 1, 2, 3)     # (N, C, D, H, W)
    # If ndim == 5 and shape is (N, 1, 28, 28, 28), it's already in (N, C, D, H, W) format.

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


# The main training function for the CNN. This will be called from the main training code to train the CNN on each dataset and evaluate its performance. It returns the predicted probabilities and class labels for the validation set.
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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    CNN training on: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    num_classes = y_train.shape[1] if multi_label else int(np.max(y_train)) + 1

    # Infer the number of input channels for the CNN based on the shape of X_train. This is important for both 2D and 3D data, as the channel dimension can be in different positions depending on the dataset.
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

        model = CNN3D(in_channels, num_classes).to(device)
        train_ds = prepare_tensors_3d(X_train, y_train, multi_label=multi_label)
        val_ds = prepare_tensors_3d(X_val, y_val, multi_label=multi_label)
    else:
        in_channels = 1 if X_train.ndim == 3 else X_train.shape[-1]
        model = CNN2D(in_channels, num_classes).to(device)
        train_ds = prepare_tensors_2d(X_train, y_train, multi_label=multi_label, augment=True)
        val_ds = prepare_tensors_2d(X_val, y_val, multi_label=multi_label, augment=False)

    # We set num_workers=0 here to avoid issues with multiprocessing on Windows. If you are running this code on Linux or macOS, you can set num_workers to a higher value (e.g., 4) to speed up data loading.
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

    # Training loop
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

    # We use torch.inference_mode() here to disable gradient tracking and reduce memory usage during evaluation. This is important for larger models and datasets, especially when running on a GPU.
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