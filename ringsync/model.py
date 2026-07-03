"""
Model definition for RingSync.

Deliberately small: this project's point is to prove the distributed
*training mechanics* are correct, not to chase SOTA accuracy. A small
CNN trains fast enough that a correctness comparison (baseline vs.
distributed loss curves) takes minutes, not hours, and a shallow model
still exercises real gradient tensors of varying shapes (conv kernels,
linear weights, biases), which is what actually stresses the ring
all-reduce implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """A small CNN for CIFAR-10 (3x32x32 input, 10 classes)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))      # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))      # 16x16 -> 8x8
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_model(seed: int = 42) -> SimpleCNN:
    """
    Build a model with a fixed seed.

    Fixing the seed here (not just for data shuffling) is what lets the
    baseline and distributed runs start from *identical* initial weights.
    Without this, any divergence between the two loss curves is
    ambiguous -- you wouldn't know if it's a bug in your all-reduce or
    just different random init.
    """
    torch.manual_seed(seed)
    return SimpleCNN()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = build_model()
    print(m)
    print(f"Total parameters: {count_parameters(m):,}")
    dummy = torch.randn(4, 3, 32, 32)
    out = m(dummy)
    print(f"Output shape: {out.shape}")
