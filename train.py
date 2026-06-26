"""
train.py — Phase 2: CNN RF Signal Classifier
Project: NN_RF
Architecture: 3x Conv1D blocks → GlobalAvgPool → concat(ptm, ifv, sk) → FC → Softmax
Input: (batch, 2, 1024) spectrum + energy envelope
Scalars: PTM (peak-to-mean), IFV (instantaneous frequency variance), SK (spectral kurtosis), CNR (carrier-to-noise)
Output: 5-class softmax (fm_broadcast, ads_b, noaa_wx, noise_floor, unknown)

No frequency injection — frequency gating handled at inference time.
"""

import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from scipy.stats import kurtosis as scipy_kurtosis

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATASET_PATH    = "data/rf_dataset.h5"
CHECKPOINT_PATH = "models/rf_cnn_best.pt"
RESULTS_DIR     = "results"

BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 1e-3
VAL_SPLIT   = 0.2
SEED        = 42

CLASS_NAMES  = ["fm_broadcast", "ads_b", "noaa_wx", "noise_floor", "unknown"]
NUM_CLASSES  = 5
NUM_SCALARS  = 4   # PTM, IFV, SK, CNR to Linear(256+4, 128)

os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────
# SCALAR EXTRACTION
# ─────────────────────────────────────────
def extract_scalars(complex_iq_np: np.ndarray) -> np.ndarray:
    """
    Compute PTM, IFV, SK from a complex IQ vector.
    Returns normalized float32 array of shape (3,).

    PTM  — peak-to-mean amplitude ratio. High for ADS-B bursts, low for noise.
    IFV  — instantaneous frequency variance. High for FM/NFM, near-zero for noise.
    SK   — spectral kurtosis. High for impulsive signals, ~3 for Gaussian noise.
    """
    amp = np.abs(complex_iq_np)

    # PTM — normalize by 30 (empirical max ~25 for ADS-B)
    ptm = float(amp.max() / (amp.mean() + 1e-9)) / 30.0

    # IFV — instantaneous frequency from phase derivative
    phase    = np.unwrap(np.angle(complex_iq_np))
    inst_freq = np.diff(phase)
    ifv      = float(np.var(inst_freq))
    ifv_norm = np.clip(ifv / 10.0, 0.0, 1.0)   # FM ~2-8, noise ~0.01-0.1

    # SK — spectral kurtosis (excess kurtosis, Gaussian = 0)
    spectrum  = np.abs(np.fft.fft(complex_iq_np))
    sk        = float(scipy_kurtosis(spectrum, fisher=True))   # Fisher: Gaussian=0
    sk_norm   = np.clip((sk + 3.0) / 30.0, 0.0, 1.0)          # shift to ~0-1 range

    # CNR — peak bin vs mean; no dominant carrier in noise floor (~1.0)
    peak      = float(spectrum.max())
    cnr       = peak / (float(spectrum.mean()) + 1e-9)
    cnr_norm  = float(np.clip(cnr / 100.0, 0.0, 1.0))

    return np.array([ptm, ifv_norm, sk_norm, cnr_norm], dtype=np.float32)


# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────
class RFDataset(Dataset):
    def __init__(self, path):
        with h5py.File(path, 'r') as f:
            self.iq     = torch.tensor(f['iq_frames'][:], dtype=torch.float32)
            self.labels = torch.tensor(f['labels'][:],    dtype=torch.long)
            self.gains  = torch.tensor(f['gains'][:],     dtype=torch.float32)

        print(f"Loaded dataset: {len(self.labels)} frames")
        unique, counts = torch.unique(self.labels, return_counts=True)
        for cls, cnt in zip(unique.tolist(), counts.tolist()):
            print(f"  Class {cls} ({CLASS_NAMES[cls]}): {cnt} frames")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        iq         = self.iq[idx]                      # (2, 1024)
        complex_iq = iq[0] + 1j * iq[1]

        # Channel 0: magnitude spectrum
        spectrum = torch.log1p(torch.abs(torch.fft.fft(complex_iq)))

        # Channel 1: time domain energy envelope
        energy = torch.log1p(torch.abs(complex_iq))

        x = torch.stack([spectrum, energy], dim=0)     # (2, 1024)

        # Scalars: PTM, IFV, SK
        scalars = extract_scalars(complex_iq.numpy())
        scalars_t = torch.tensor(scalars, dtype=torch.float32)  # (3,)

        return x, scalars_t, self.labels[idx]


# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


class RFClassifier(nn.Module):
    def __init__(self, num_classes=5, num_scalars=3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(2, 64),       # (B, 2, 1024) to (B, 64, 512)
            ConvBlock(64, 128),     # to (B, 128, 256)
            ConvBlock(128, 256),    # to (B, 256, 128)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # to (B, 256, 1)
        fc_in = 256 + num_scalars                        # 256 + 3 = 259
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, scalars):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)                  # (B, 256)
        x = torch.cat([x, scalars], dim=1)        # (B, 259)
        return self.classifier(x)


# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, scalars, labels in loader:
        x, scalars, labels = x.to(device), scalars.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(x, scalars)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, scalars, labels in loader:
            x, scalars, labels = x.to(device), scalars.to(device), labels.to(device)
            out   = model(x, scalars)
            loss  = criterion(out, labels)
            total_loss += loss.item() * len(labels)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total   += len(labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────
def plot_loss_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses,   label='Val')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(True)
    ax2.plot(train_accs, label='Train')
    ax2.plot(val_accs,   label='Val')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/loss_curves.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/loss_curves.png")


def plot_confusion_matrix(all_labels, all_preds):
    cm   = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    plt.title('Confusion Matrix — Validation Set')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/confusion_matrix.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    dataset    = RFDataset(DATASET_PATH)
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"\nTrain: {train_size} | Val: {val_size}")

    model = RFClassifier(NUM_CLASSES, NUM_SCALARS).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0

    print("\n── Training ──")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc             = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, preds, lbs = val_epoch(model, val_loader, criterion)
        scheduler.step(vl_loss)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ✓ New best model saved ({best_val_acc:.4f})")

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    _, final_acc, final_preds, final_labels = val_epoch(model, val_loader, criterion)

    print("\n── Classification Report ──")
    print(classification_report(final_labels, final_preds, target_names=CLASS_NAMES))

    plot_loss_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(final_labels, final_preds)

    print("\nPhase 2 complete.")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Plots:      {RESULTS_DIR}/")


if __name__ == "__main__":
    main()