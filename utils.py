"""
Utility classes and functions for the QSNN project.

Includes:
    - GaussianNoise: Custom torchvision transform for adding Gaussian noise
    - EarlyStopping: Monitors validation loss and stops training on overfitting
    - plot_losses: Plots train vs validation loss curves
    - rate_encode: Wrapper around snntorch rate encoding
"""

import os
import copy
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from snntorch import spikegen


# ─────────────────────────────────────────────────────────────
# Custom Transform: Gaussian Noise
# ─────────────────────────────────────────────────────────────

class GaussianNoise:
    """Adds Gaussian noise to a tensor and clamps to [0, 1].

    Args:
        std (float): Standard deviation of the noise. Default: 0.05
    """

    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"


# ─────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """Monitors validation loss and stops training when overfitting is detected.

    Args:
        patience (int): Number of epochs to wait after last improvement.
        min_delta (float): Minimum decrease in loss to qualify as improvement.
        verbose (bool): If True, prints a message when early stopping triggers.
    """

    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} "
                      f"(best val_loss: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("  >> Early stopping triggered. Restoring best model.")

    def restore_best_model(self, model):
        """Restore the model to the state with the best validation loss."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# ─────────────────────────────────────────────────────────────
# Rate Encoding
# ─────────────────────────────────────────────────────────────

def rate_encode(data, num_steps=25):
    """Rate-encode a batch of images into spike trains.

    Each pixel value is treated as a firing probability. At each time step,
    spikes are sampled from a Bernoulli distribution parameterised by the
    pixel intensity.

    Args:
        data (Tensor): Input images of shape (B, C, H, W) with values in [0, 1].
        num_steps (int): Number of time steps for the spike train.

    Returns:
        Tensor: Spike train of shape (num_steps, B, C, H, W) with binary values.
    """
    return spikegen.rate(data, num_steps=num_steps)


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_losses(train_losses, val_losses, save_path="loss_plot.png",
                phase1_epochs=10):
    """Plot training and validation loss on the same graph.

    A vertical dashed line marks the transition from Phase 1 (Conv+ANN only)
    to Phase 2 (full system with quantum layer).

    Args:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
        save_path (str): File path to save the plot.
        phase1_epochs (int): Number of epochs in Phase 1 (for annotation).
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', markersize=4, label='Train Loss')
    plt.plot(epochs, val_losses, 'r-o', markersize=4, label='Validation Loss')

    # Mark phase transition
    if phase1_epochs < len(train_losses):
        plt.axvline(x=phase1_epochs + 0.5, color='gray', linestyle='--',
                     linewidth=1.5, label=f'Phase 1→2 (epoch {phase1_epochs})')

    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.title('Training vs Validation Loss — QSNN', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n[✓] Loss plot saved to: {os.path.abspath(save_path)}")


def plot_accuracies(train_accs, val_accs, save_path="accuracy_plot.png",
                    phase1_epochs=10):
    """Plot training and validation accuracy on the same graph.

    Args:
        train_accs (list): Training accuracy per epoch.
        val_accs (list): Validation accuracy per epoch.
        save_path (str): File path to save the plot.
        phase1_epochs (int): Number of epochs in Phase 1.
    """
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-o', markersize=4, label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-o', markersize=4, label='Validation Accuracy')

    if phase1_epochs < len(train_accs):
        plt.axvline(x=phase1_epochs + 0.5, color='gray', linestyle='--',
                     linewidth=1.5, label=f'Phase 1→2 (epoch {phase1_epochs})')

    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Accuracy (%)', fontsize=13)
    plt.title('Training vs Validation Accuracy — QSNN', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Accuracy plot saved to: {os.path.abspath(save_path)}")
