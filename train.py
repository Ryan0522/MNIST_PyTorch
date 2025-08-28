import argparse
import os
from datetime import date
import torch
import torch.nn as nn

from utils.data import get_dataloaders
from utils.metrics import evaluate, confusion_matrix, save_confusion_csv, normalize_confusion
from utils.plotting import plot_loss_acc, plot_confusion_matrix
from models.logistic_regression import LogisticRegression
from models.mnist_mlp import MNISTClassifier

def set_seed(s: int = 42):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(name: str):
    name = name.lower()
    if name in {"logreg", "lr", "linear"}:
        return LogisticRegression(), "logreg"
    elif name in {"mlp", "nn"}:
        return MNISTClassifier(), "mlp"
    else:
        raise ValueError(f"Unknown model: {name}")

def get_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr), "adam"
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9), "sgd"
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="mlp", choices=["mlp", "logreg"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    # Paths
    today = date.today().isoformat()
    results_root = args.results_dir or os.path.join("results", today)
    ckpt_dir = "checkpoints"
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data
    train_loader, test_loader = get_dataloaders(
        data_root=args.data_root, batch_size=args.batch
    )

    # Model / Opt / Loss
    model, model_key = get_model(args.model)
    optimizer, opt_key = get_optimizer(args.optimizer, model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Experiment tag for filenames
    exp = f"mnist-{model_key}-b{args.batch}-{opt_key}-lr{args.lr}-seed{args.seed}"

    best_acc = -1.0
    train_losses, val_losses, val_accs = [], [], []

    # Training
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        train_loss = running / len(train_loader)

        # Evaluation
        metrics = evaluate(model, criterion, test_loader, device=device)
        val_loss = metrics["loss"]
        val_acc = metrics["acc"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc*100:5.2f}%")

        # Save checkpoints
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "config": vars(args),
            "exp": exp,
        }
        torch.save(state, os.path.join(ckpt_dir, f"{exp}-last.pt"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(state, os.path.join(ckpt_dir, f"{exp}-best.pt"))

    # Curves
    plot_loss_acc(train_losses, val_losses, val_accs, results_root, exp)

    # Confusion matrix (counts + normalized heatmap)
    C = confusion_matrix(model, test_loader, num_classes=10, device=device)
    save_confusion_csv(C, os.path.join(results_root, f"confusion_matrix_{exp}.csv"))
    Cn = normalize_confusion(C)
    plot_confusion_matrix(
        Cn,
        os.path.join(results_root, f"confusion_matrix_{exp}.png"),
        title=f"Confusion Matrix — {exp}",
        annotate=True,
    )

    print(f"\nArtifacts saved to:\n  - {results_root}\n  - {ckpt_dir}")

if __name__ == "__main__":
    main()
