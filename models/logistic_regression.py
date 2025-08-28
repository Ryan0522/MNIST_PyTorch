import torch.nn as nn
import torch

class LogisticRegression(nn.Module):
    """
    Linear baseline: 784 -> 10, no activation.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1) # [batch, 1, 28, 28] -> [batch, 784]
        return self.fc(x)
