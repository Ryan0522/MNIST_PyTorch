import os
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_loss_acc(
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    val_acc: Sequence[float],
    out_dir: str,
    prefix: str,
):
    ensure_dir(out_dir)
    epochs = np.arange(1, len(train_loss) + 1)

    # Loss curves
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Loss Curves — {prefix}")
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"loss_curve_{prefix}.png"), dpi=150)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, val_acc, label="val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve — {prefix}")
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"acc_curve_{prefix}.png"), dpi=150)
    plt.close()

def plot_confusion_matrix(
    C_norm: torch.Tensor,
    out_path: str,
    title: str = "Confusion Matrix",
    annotate: bool = True,
):
    """
    C_norm: (10,10) float, row-normalized.
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure(figsize=(8, 7))
    plt.imshow(C_norm.cpu().numpy(), cmap="viridis", aspect="equal")
    plt.colorbar()
    classes = np.arange(10)
    plt.xticks(range(10), classes)
    plt.yticks(range(10), classes)
    if annotate:
        for i in range(10):
            for j in range(10):
                val = C_norm[i, j].item()
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
