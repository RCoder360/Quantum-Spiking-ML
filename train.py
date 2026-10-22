"""
Training pipeline for the Hybrid QSNN on CIFAR-10.

Training Schedule:
    Phase 1 (epochs 1–10):  ConvSNN + ANN_bypass only (quantum bypassed)
    Phase 2 (epochs 11–30): Full system — ConvSNN + Quantum + ANN_quantum

Features:
    - Rate encoding across all RGB channels
    - Gaussian noise augmentation (5%)
    - Adam optimiser with L2 regularisation (weight decay)
    - Early stopping with patience=5 (Phase 2 only)
    - Train/val accuracy printed each epoch
    - Loss curves plotted and saved
    - Final test evaluation with classification report
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from model import QSNN
from utils import (
    GaussianNoise,
    EarlyStopping,
    rate_encode,
    plot_losses,
    plot_accuracies,
)


# ═════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════

# Paths
DATA_DIR = r"F:\Academics\Research Paper\QCNN\code\data"
SAVE_DIR = r"F:\Academics\Research Paper\QCNN\code\QSNN\results"

# Model hyper-parameters
NUM_STEPS    = 25       # SNN simulation time steps
BETA         = 0.9      # LIF neuron decay rate
NUM_CLASSES  = 10       # CIFAR-10 classes
DROPOUT      = 0.3      # ANN dropout rate

# Training hyper-parameters
BATCH_SIZE   = 128
PHASE1_EPOCHS = 10      # Train Conv+ANN only
PHASE2_EPOCHS = 20      # Train full system (Conv+Quantum+ANN)
TOTAL_EPOCHS  = PHASE1_EPOCHS + PHASE2_EPOCHS  # 30

LR_PHASE1    = 1e-3
LR_PHASE2    = 5e-4
WEIGHT_DECAY = 1e-4     # L2 regularisation

EARLY_STOP_PATIENCE = 5

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# ═════════════════════════════════════════════════════════════
# Data Loading
# ═════════════════════════════════════════════════════════════

def get_data_loaders():
    """Load CIFAR-10 and split into 80/10/10 train/val/test sets.

    Transforms:
        - ToTensor (values in [0, 1])
        - GaussianNoise (std=0.05)

    Note: We do NOT normalise with mean/std because rate encoding
    requires pixel values in [0, 1].

    Returns:
        train_loader, val_loader, test_loader
    """
    transform_train = T.Compose([
        T.ToTensor(),
        GaussianNoise(std=0.05),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
    ])

    # Combine CIFAR-10's 50,000 training images and 10,000 test images.
    full_train = ConcatDataset([
        torchvision.datasets.CIFAR10(
            root=DATA_DIR, train=True, download=True, transform=transform_train
        ),
        torchvision.datasets.CIFAR10(
            root=DATA_DIR, train=False, download=True, transform=transform_train
        ),
    ])

    # Deterministic split
    generator = torch.Generator().manual_seed(42)
    total = len(full_train)  # 60,000
    train_size = int(0.8 * total)   # 48,000
    val_size   = int(0.1 * total)   # 6,000
    test_size  = total - train_size - val_size  # 6,000

    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices   = indices[train_size:train_size + val_size]
    test_indices  = indices[train_size + val_size:]

    train_dataset = Subset(full_train, train_indices)

    # Validation and test use clean transforms (no noise)
    full_clean = ConcatDataset([
        torchvision.datasets.CIFAR10(
            root=DATA_DIR, train=True, download=False, transform=transform_test
        ),
        torchvision.datasets.CIFAR10(
            root=DATA_DIR, train=False, download=False, transform=transform_test
        ),
    ])
    val_dataset  = Subset(full_clean, val_indices)
    test_dataset = Subset(full_clean, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"Dataset splits: Train={len(train_dataset)}, "
          f"Val={len(val_dataset)}, Test={len(test_dataset)}")

    return train_loader, val_loader, test_loader


# ═════════════════════════════════════════════════════════════
# Training Loop
# ═════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device,
                    use_quantum=False, num_steps=NUM_STEPS):
    """Train the model for one epoch.

    Args:
        model: QSNN model.
        loader: Training DataLoader.
        optimizer: Optimiser instance.
        criterion: Loss function.
        device: torch device.
        use_quantum: Whether to route through the quantum layer.
        num_steps: Number of SNN time steps for rate encoding.

    Returns:
        avg_loss, accuracy (%)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device)
        targets = targets.to(device)

        # Rate-encode the images into spike trains
        spike_data = rate_encode(data, num_steps=num_steps)  # (T, B, C, H, W)
        spike_data = spike_data.to(device)

        optimizer.zero_grad()
        output = model(spike_data, use_quantum=use_quantum)  # (B, 10)

        loss = criterion(output, targets)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(targets).sum().item()
   
        # Progress indicator every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx + 1}/{len(loader)}  |  "
                  f"Loss: {loss.item():.4f}", flush=True)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device,
             use_quantum=False, num_steps=NUM_STEPS):
    """Evaluate the model on a dataset.

    Args:
        model: QSNN model.
        loader: DataLoader for evaluation.
        criterion: Loss function.
        device: torch device.
        use_quantum: Whether to route through the quantum layer.
        num_steps: Number of SNN time steps for rate encoding.

    Returns:
        avg_loss, accuracy (%), all_preds, all_targets
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)

        spike_data = rate_encode(data, num_steps=num_steps)
        spike_data = spike_data.to(device)

        output = model(spike_data, use_quantum=use_quantum)
        loss = criterion(output, targets)

        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(targets).sum().item()
        total += data.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

def main():
    # ── Device setup ──
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA not available, using CPU (training will be slow)")

    # ── Create results directory ──
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load data ──
    print("\n" + "=" * 60)
    print("Loading CIFAR-10 dataset...")
    print("=" * 60)
    train_loader, val_loader, test_loader = get_data_loaders()

    # ── Build model ──
    print("\n" + "=" * 60)
    print("Building QSNN model...")
    print("=" * 60)
    model = QSNN(
        beta=BETA,
        num_steps=NUM_STEPS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ── Loss function ──
    criterion = nn.CrossEntropyLoss()

    # ── Tracking ──
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # ══════════════════════════════════════════════════════════
    # PHASE 1: Train ConvSNN + ANN_bypass (Quantum Bypassed)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"PHASE 1: Training ConvSNN + ANN (epochs 1–{PHASE1_EPOCHS})")
    print("Quantum layer: INACTIVE (bypassed)")
    print("=" * 60)

    # Only optimise ConvSNN and bypass ANN parameters
    phase1_params = (
        list(model.conv_snn.parameters()) +
        list(model.ann_bypass.parameters())
    )
    optimizer1 = torch.optim.Adam(
        phase1_params, lr=LR_PHASE1, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer1, criterion, device,
            use_quantum=False, num_steps=NUM_STEPS
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device,
            use_quantum=False, num_steps=NUM_STEPS
        )

        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"\nEpoch [{epoch}/{TOTAL_EPOCHS}]  (Phase 1)  "
              f"Time: {elapsed:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.2f}%")
        print("-" * 60)

    # ══════════════════════════════════════════════════════════
    # PHASE 2: Train Full System (ConvSNN + Quantum + ANN)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"PHASE 2: Training full QSNN (epochs {PHASE1_EPOCHS + 1}–{TOTAL_EPOCHS})")
    print("Quantum layer: ACTIVE")
    print("=" * 60)

    # Optimise ALL parameters (conv + quantum + ann_quantum)
    optimizer2 = torch.optim.Adam(
        model.parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY
    )

    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, verbose=True)

    for epoch in range(PHASE1_EPOCHS + 1, TOTAL_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer2, criterion, device,
            use_quantum=True, num_steps=NUM_STEPS
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device,
            use_quantum=True, num_steps=NUM_STEPS
        )

        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"\nEpoch [{epoch}/{TOTAL_EPOCHS}]  (Phase 2 — Quantum)  "
              f"Time: {elapsed:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.2f}%")

        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n>> Stopped early at epoch {epoch}.")
            early_stopping.restore_best_model(model)
            break

        print("-" * 60)

    # ══════════════════════════════════════════════════════════
    # Plot Losses & Accuracies
    # ══════════════════════════════════════════════════════════
    plot_losses(
        train_losses, val_losses,
        save_path=os.path.join(SAVE_DIR, "loss_plot.png"),
        phase1_epochs=PHASE1_EPOCHS,
    )
    plot_accuracies(
        train_accs, val_accs,
        save_path=os.path.join(SAVE_DIR, "accuracy_plot.png"),
        phase1_epochs=PHASE1_EPOCHS,
    )

    # ══════════════════════════════════════════════════════════
    # FINAL EVALUATION on Test Set
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FINAL EVALUATION on Test Set")
    print("=" * 60)

    test_loss, test_acc, preds, targets = evaluate(
        model, test_loader, criterion, device,
        use_quantum=True, num_steps=NUM_STEPS
    )

    print(f"\n  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")

    # Classification report
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(classification_report(
        targets, preds, target_names=CIFAR10_CLASSES, digits=4
    ))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(targets, preds)
    print(cm)

    # ── Save model ──
    model_path = os.path.join(SAVE_DIR, "qsnn_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
    }, model_path)
    print(f"\n[✓] Model saved to: {model_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
