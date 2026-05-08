# cnn.py - CNN model definitions and training function for 2D and 3D classification.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # This is the "Skip Connection"
        return torch.relu(out)

class CNN2D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_planes = 64
        
        # Initial Layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet Layers
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return self.classifier(out)


# Standard 3D CNN
# A 3D ResNet that mirrors the 2D version above. The previous implementation
# collapsed spatial dims to 1x1x1 with AdaptiveAvgPool3d *before* the final conv,
# which threw away all spatial structure and is why several 3D datasets came
# back below 0.5 AUC. This version keeps real spatial reasoning all the way
# through and only pools to 1x1x1 right before the classifier.
#
# We use GroupNorm rather than BatchNorm3d because the 3D MedMNIST datasets
# are small (~1k volumes) and batches are correspondingly small (32). With
# BatchNorm under aggressive augmentation, the running mean/var stats tracked
# during training drift wildly, so the model behaves very differently at eval
# (which uses those stored stats) than during training (which uses per-batch
# stats). Symptom: train loss drops normally while val loss explodes into the
# dozens. GroupNorm has no running stats, so train and eval modes behave
# identically — the failure mode is eliminated by construction.

def _gn(num_channels):
    # Pick a group count up to 32 that evenly divides num_channels.
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0 and g <= num_channels:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1   = _gn(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2   = _gn(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _gn(out_channels),
            )

    def forward(self, x):
        out = torch.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class CNN3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_planes = 32

        # Stem
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1   = _gn(32)

        # Three residual stages: 32 -> 64 -> 128 channels with stride-2
        # downsampling at stages 2-3. Spatial dims go 28^3 -> 14^3 -> 7^3.
        # Width is tighter than the 2D version because 3D conv cubes the
        # parameter & FLOP cost.
        self.layer1 = self._make_layer(32,  2, stride=1)
        self.layer2 = self._make_layer(64,  2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)

        # Only collapse to a single voxel right before the classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Widened classifier with a hidden FC layer. The previous head was
        # 128 -> num_classes with Dropout(0.5), which together with strong
        # augmentation + weight_decay 1e-3 was over-regularising tiny (~1k
        # volume) datasets and preventing the model from fitting at all.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock3D(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return self.classifier(out)


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
# Returns one of:
#   * a TensorDataset (when augment=False) — fast, no random ops at __getitem__
#   * a MedMNIST3DDataset (when augment=True) — augments per sample per epoch,
#     so the model sees a different volume every time. The previous code
#     applied "augmentation" once at dataset construction, which means every
#     epoch saw the *same* augmented samples — equivalent to no augmentation
#     at all from epoch 2 onward.

class MedMNIST3DDataset(torch.utils.data.Dataset):
    """
    3D dataset with on-the-fly random flips + light intensity jitter.

    `flip_axes` controls which spatial axes can be randomly flipped. For
    most datasets all three is fine, but datasets with laterality labels
    (e.g. organmnist3d's "kidney-left" vs "kidney-right") MUST exclude
    the left-right axis or the model is being trained with wrong labels.

    Convention after the channel-first transpose in prepare_tensors_3d:
        axis 1 = depth (axial slice index, head-foot in body coords)
        axis 2 = height (typically anterior-posterior)
        axis 3 = width  (typically left-right) ← skip this for laterality

    Default: only axis 1 (depth/axial), which is anatomically safe for
    every dataset.
    """

    def __init__(self, X, y, multi_label=False, flip_axes=(1,), noise_std=0.0):
        # X: float32 in [0, 1], shape (N, C, D, H, W)
        self.X = X
        self.y = y
        self.multi_label = multi_label
        self.flip_axes = flip_axes
        self.noise_std = noise_std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        vol = self.X[idx]  # (C, D, H, W) float32

        # Random flips on the allowed axes only
        for ax in self.flip_axes:
            if np.random.rand() < 0.5:
                vol = np.flip(vol, axis=ax)

        # ascontiguousarray to drop the negative strides from np.flip
        # (PyTorch can't take negative-strided arrays).
        vol = np.ascontiguousarray(vol)

        if self.noise_std > 0:
            vol = vol + np.random.normal(0, self.noise_std, vol.shape).astype(np.float32)
            vol = np.clip(vol, 0.0, 1.0)

        x_tensor = torch.from_numpy(vol)

        if self.multi_label:
            y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        else:
            y_tensor = torch.tensor(int(self.y[idx]), dtype=torch.int64)

        return x_tensor, y_tensor


def prepare_tensors_3d(X, y=None, multi_label=False, augment=False, flip_axes=(1,), noise_std=0.0):
    X = np.asarray(X, dtype=np.float32) / 255.0

    if X.ndim == 4:
        X = np.expand_dims(X, axis=1)
    elif X.ndim == 5 and X.shape[-1] in (1, 3):
        X = X.transpose(0, 4, 1, 2, 3)

    if y is None:
        return torch.from_numpy(X)

    y = np.asarray(y)
    if not multi_label and y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)

    if augment:
        return MedMNIST3DDataset(X, y, multi_label=multi_label,
                                 flip_axes=flip_axes, noise_std=noise_std)

    # Validation / no augmentation: a plain TensorDataset is faster
    X_tensor = torch.from_numpy(X)
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
    label_smoothing=0.1,
    tune_threshold=False, 
    weight_decay=1e-2,  # High default to combat overfitting
    flip_axes=(1,),     # 3D only: which spatial axes to allow flipping on
    noise_std=0.0,      # 3D only: stddev of intensity jitter (0 disables it)
    use_amp=None,       # None = auto (off for 3D, on for 2D); bool to force
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    CNN training on: {device}")

    # Disable AMP for 3D by default. The 3D models are small (~2M params) so
    # the fp16 speedup is modest, and AMP + 3D conv + GroupNorm combinations
    # have been a source of numerical instability.
    if use_amp is None:
        use_amp = (not is_3d_data) and (device.type == "cuda")

    # DATA PREPARATION 
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    num_classes = y_train.shape[1] if multi_label else int(np.max(y_train)) + 1

    if is_3d_data:
        # Derive 3D input channels from the array shape:
        #   (N, D, H, W)        -> 1 channel
        #   (N, C, D, H, W)     -> C channels (channel-first)
        #   (N, D, H, W, C)     -> C channels (channel-last)
        if X_train.ndim == 4:
            in_channels = 1
        elif X_train.ndim == 5 and X_train.shape[1] in (1, 3):
            in_channels = X_train.shape[1]
        elif X_train.ndim == 5 and X_train.shape[-1] in (1, 3):
            in_channels = X_train.shape[-1]
        else:
            in_channels = 1
        model = CNN3D(in_channels=in_channels, num_classes=num_classes).to(device)
        train_ds = prepare_tensors_3d(X_train, y_train, multi_label=multi_label,
                                      augment=strong_augment, flip_axes=flip_axes, noise_std=noise_std)
        val_ds = prepare_tensors_3d(X_val, y_val, multi_label=multi_label, augment=False)
    else:
        in_channels = 1 if X_train.ndim == 3 else X_train.shape[-1]
        model = CNN2D(in_channels, num_classes).to(device)
        train_ds = prepare_tensors_2d(X_train, y_train, multi_label=multi_label, 
                                      augment=not strong_augment, strong_augment=strong_augment)
        val_ds = prepare_tensors_2d(X_val, y_val, multi_label=multi_label, 
                                    augment=False, strong_augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # LOSS, OPTIMIZER, & SCHEDULER
    if multi_label:
        pw = compute_pos_weight(y_train).to(device) if use_pos_weight else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        cw = compute_class_weights(y_train, num_classes).to(device) if use_class_weights else None
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # OneCycleLR works well for 2D where there are many batches per epoch and
    # the dataset is large enough to absorb the high-LR phase. For 3D the
    # datasets are ~1k volumes (~30 batches/epoch), and the high-LR ramp was
    # destabilising training — switch to plain cosine decay from `lr` down to
    # `lr/100`. CosineAnnealingLR steps once per epoch (not per batch).
    if is_3d_data:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        scheduler_step_per_batch = False
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.3
        )
        scheduler_step_per_batch = True

    # TRAINING LOOP WITH EARLY STOPPING
    # Patience widened from 5 to 12 because OneCycleLR can produce big val-loss
    # bumps during the high-LR phase that aren't really plateaus. We also
    # freeze the early-stop counter for the first 30% of epochs (the LR
    # warm-up phase) so we never abort during the ramp-up — checkpoints saved
    # during that phase are usually noisy and not representative.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    stopper = EarlyStopping(patience=12)
    min_epochs = max(int(epochs * 0.3), 5)
    best_model_state = None

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(X_batch)
                loss = criterion(out, y_batch)
            
            scaler.scale(loss).backward()
            
            # Gradient Clipping to prevent "Exploding Gradients"
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            if scheduler_step_per_batch:
                scheduler.step() # OneCycleLR: update LR every batch
            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        with torch.inference_mode():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(X_batch)
                    v_loss = criterion(out, y_batch)
                total_val_loss += v_loss.item()
        
        avg_val = total_val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        # Cosine scheduler steps once per epoch (after the val pass)
        if not scheduler_step_per_batch:
            scheduler.step()

        print(f"    Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Check Early Stopping & Save Best State
        stopper(avg_val)
        if best_model_state is None or avg_val <= stopper.best_loss:
            # We clone to CPU to save GPU memory
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if stopper.early_stop and (epoch + 1) >= min_epochs:
            print(f"    >>> Early stopping triggered at epoch {epoch+1}")
            break

    # FINAL EVAL
    # Load the best weights (not the last ones) to ensure we didn't return an overfitted model
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    all_probs = []
    all_preds = []

    with torch.inference_mode():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            if multi_label:
                probs = torch.sigmoid(out).cpu().numpy()
            else:
                probs = torch.softmax(out, dim=1).cpu().numpy()
                all_preds.append(probs.argmax(axis=1))
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