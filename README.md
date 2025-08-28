# MNIST — Logistic Regression vs MLP (PyTorch)

A compact, reproducible MNIST baseline comparing a **logistic regression** (linear) and a **2-layer MLP** (nonlinear) — all in pure PyTorch (no sklearn).

## Results (Week 1, Day 4)

- Dataset: MNIST (60k train / 10k test)
- Batch size: 64
- Optimizers: LogReg=SGD(lr=0.01), MLP=Adam(lr=1e-3)
- Epochs: 10

| Model | Val Acc (≈) | Best Epoch | Notes |
|------:|:-----------:|:----------:|------|
| Logistic Regression | ~90–91% | 10 | Linear baseline |
| MLP (784→128→10 + ReLU) | ~97–98% | 7–10 | Nonlinear, better separation |

## Project Structure

```
MNIST_PyTorch/
├─ train.py
├─ predict.py
├─ models/
│  ├─ logistic_regression.py
│  └─ mnist_mlp.py
├─ utils/
│  ├─ data.py
│  ├─ metrics.py
│  └─ plotting.py
├─ checkpoints/
├─ results/
│  └─ 2025-08-27/
├─ data/
├─ README.md
└─ requirements.txt
```

## Quickstart

```bash
# 1) create env & install
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) train (choose model: logreg | mlp)
python train.py --model logreg --epochs 10 --batch 64 --lr 0.01 --optimizer sgd
python train.py --model mlp    --epochs 10 --batch 64 --lr 0.001 --optimizer adam

# 3) find artifacts
#   checkpoints/<exp>-best.pt / -last.pt
#   results/2025-08-27/{loss_curve.png, acc_curve.png, confusion_matrix_*.png}

# 4) predict with a single image
python predict.py --model mlp --ckpt checkpoints/<exp>-best.pt --img path/to/digit.png
```

## Notes

- **Confusion Matrix** implemented without sklearn (pure PyTorch).
- Plots saved via matplotlib.
- Fixed random seed recommended for reproducibility.

## License

MIT
