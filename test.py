import argparse
import os
import torch
import torch.nn as nn

from utils.data import get_dataloaders
from utils.metrics import evaluate, confusion_matrix, normalize_confusion, save_confusion_csv
from utils.plotting import plot_confusion_matrix
from models.logistic_regression import LogisticRegression
from models.mnist_mlp import MNISTClassifier

def get_model_from_key(key: str):
    key = key.lower()
    if key in {"logreg", "lr", "linear"}:
        return LogisticRegression(), "logreg"
    elif key in {"mlp", "nn"}:
        return MNISTClassifier(), "mlp"
    else:
        raise ValueError(f"Unknown model key: {key}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="mlp", choices=["mlp", "logreg"])
    p.add_argument("--ckpt", type=str, required=True, help="Path to *.pt checkpoint")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="results/test_run")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Data
    _, test_loader = get_dataloaders(data_root=args.data_root, batch_size=args.batch)

    # Model
    model, model_key = get_model_from_key(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, criterion, test_loader, device=device)
    print(f"Test  loss={metrics['loss']:.4f}  acc={metrics['acc']*100:5.2f}%")

    # Confusion matrix for verification
    C = confusion_matrix(model, test_loader, num_classes=10, device=device)
    save_confusion_csv(C, os.path.join(args.out_dir, f"confusion_matrix_{model_key}.csv"))
    Cn = normalize_confusion(C)
    plot_confusion_matrix(
        Cn,
        os.path.join(args.out_dir, f"confusion_matrix_{model_key}.png"),
        title=f"Confusion Matrix — {model_key}",
        annotate=True,
    )

if __name__ == "__main__":
    main()
