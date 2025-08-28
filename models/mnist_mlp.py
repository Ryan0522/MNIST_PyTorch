import torch.nn as nn
import torch

class MNISTClassifier(nn.Module):
    """
    2-layer MLP: 784 -> 128 -> 10 + ReLU
    """
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1) # [batch, 1, 28, 28] -> [batch, 784]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
