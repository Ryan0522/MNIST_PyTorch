from typing import Dict, Tuple
import torch
import numpy as np
import csv
import os

@torch.no_grad()
def evaluate(model, criterion, loader, device="cpu") -> Dict[str, float]:
    """
    Eval loop returning avg loss and accuracy.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += float(loss.item())
        pred = logits.argmax(dim=1)
        total_correct += int((pred == y).sum().item())
        total_samples += int(y.size(0))

    avg_loss = running_loss / len(loader)
    acc = total_correct / max(1, total_samples)
    return {"loss": avg_loss, "acc": acc}

@torch.no_grad()
def confusion_matrix(model, loader, num_classes=10, device="cpu") -> torch.Tensor:
    """
    Integer 10x10 confusion matrix C where C[true, pred] counts samples.
    """
    C = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    model.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        for t, p in zip(y, pred):
            C[int(t), int(p)] += 1
    return C

def save_confusion_csv(C: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(C.size(0)):
            row = [int(C[i, j].item()) for j in range(C.size(1))]
            writer.writerow(row)

def normalize_confusion(C: torch.Tensor) -> torch.Tensor:
    """
    Row-normalize (per true class). NaNs converted to 0.
    """
    C = C.float()
    row_sum = C.sum(dim=1, keepdim=True).clamp_min(1.0)
    N = C / row_sum
    N = torch.nan_to_num(N)
    return N
