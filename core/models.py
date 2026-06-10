"""CNN architectures for FL experiments. Key models:
- MicroMNIST (~4.3K params): cross-device, Fashion-MNIST, client-level DP
- SVHNCNN (~90K params): cross-silo, SVHN, record-level DP
- TinyMNIST (~9-12K params): archived, original from iteration 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class MicroMNIST(nn.Module):
    """Smallest viable CNN. ~4.3K params. Ceiling ~90-92% on MNIST.
    Architecture: conv1(1→4,3×3) → pool → conv2(4→8,3×3) → pool → fc(8×7×7→10).
    d=4266: SNR_MK=0.251 at ε=10 (σ=1.707, f=28, n_active=40).
    Purpose: attacks visible (ceiling low enough) AND DP viable (d small enough).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        pooled_size = input_size // 4
        self.fc = nn.Linear(8 * pooled_size * pooled_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MiniMNIST(nn.Module):
    """Intermediate CNN. ~6.6K params. Between Micro and Tiny.
    Architecture: conv1(1→6,3×3) → pool → conv2(6→12,3×3) → pool → fc(12×7×7→10).
    d=6610: SNR_MK=0.202 at ε=10 — right at the boundary.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        pooled_size = input_size // 4
        self.fc = nn.Linear(12 * pooled_size * pooled_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TinyMNIST(nn.Module):
    """2-conv CNN. ~9-12K params depending on input. No BatchNorm (important for DP-FL SNR).
    MNIST (1×28×28, d=9098): SNR ≈ 0.27 at σ=1.71 (ε=10, n=200, q=0.2).
    CIFAR/SVHN (3×32×32, d=11642): SNR ≈ 0.22 at σ=1.71 — still above 0.19 threshold.
    Architecture: conv1(in→8,3×3) → pool → conv2(8→16,3×3) → pool → fc(flat→10).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        pooled_size = input_size // 4  # two 2×2 pools
        self.fc = nn.Linear(16 * pooled_size * pooled_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SVHNCNN(nn.Module):
    """3-conv CNN for SVHN (3×32×32). ~90K params. No BatchNorm.
    Architecture: conv1(3→16,3×3) → pool → conv2(16→32,3×3) → pool →
                  conv3(32→64,3×3) → pool → fc1(1024→64) → fc2(64→10).
    d ≈ 89,834. For record-level DP: SNR >> 1 (sensitivity ~ R/(n·|D_i|) very small).
    For client-level DP at ε=10: SNR ≈ 0.03 — too low, use record-level instead.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 32→16
        x = self.pool(F.relu(self.conv2(x)))   # 16→8
        x = self.pool(F.relu(self.conv3(x)))   # 8→4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CIFAR10CNN(nn.Module):
    """3-conv CNN for CIFAR-10. ~290K params."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32→16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 16→8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 8→4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dropout(x)))
        return self.fc2(x)


class FashionMNISTCNN(nn.Module):
    """2-conv CNN for Fashion-MNIST. ~60K params."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 28→14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 14→7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dropout(x)))
        return self.fc2(x)


def get_model(dataset: Literal["mnist", "cifar10", "fashion_mnist", "svhn"],
              model_name: str | None = None) -> nn.Module:
    """Factory: returns model for the given dataset (or explicit model_name override).
    model_name='tiny' forces TinyMNIST with appropriate in_channels/input_size.
    """
    # Explicit model overrides
    if model_name == "svhn_cnn":
        return SVHNCNN()

    if model_name == "micro":
        if dataset in ("cifar10", "svhn"):
            return MicroMNIST(in_channels=3, input_size=32)
        elif dataset == "fashion_mnist":
            return MicroMNIST(in_channels=1, input_size=28)
        else:
            return MicroMNIST()  # MNIST: ~4.3K params

    if model_name == "mini":
        if dataset in ("cifar10", "svhn"):
            return MiniMNIST(in_channels=3, input_size=32)
        elif dataset == "fashion_mnist":
            return MiniMNIST(in_channels=1, input_size=28)
        else:
            return MiniMNIST()  # MNIST: ~6.6K params

    if model_name == "tiny":
        if dataset in ("cifar10", "svhn"):
            return TinyMNIST(in_channels=3, input_size=32)  # ~11.6K params
        elif dataset == "fashion_mnist":
            return TinyMNIST(in_channels=1, input_size=28)  # ~9K params
        else:
            return TinyMNIST()  # MNIST default

    # Default models per dataset
    if dataset == "mnist":
        return TinyMNIST()
    elif dataset == "cifar10":
        return CIFAR10CNN()
    elif dataset == "svhn":
        return SVHNCNN()
    elif dataset == "fashion_mnist":
        return FashionMNISTCNN()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_weights(model: nn.Module) -> list[torch.Tensor]:
    """Extract model weights as list of tensors (detached, on CPU)."""
    return [p.data.detach().cpu().clone() for p in model.parameters()]


def set_weights(model: nn.Module, weights: list[torch.Tensor]) -> None:
    """Set model parameters from list of tensors."""
    for p, w in zip(model.parameters(), weights):
        p.data = w.to(p.device)


def weights_to_flat(weights: list[torch.Tensor]) -> torch.Tensor:
    """Flatten list of weight tensors to a single 1D tensor."""
    return torch.cat([w.flatten() for w in weights])


def flat_to_weights(flat: torch.Tensor, reference: list[torch.Tensor]) -> list[torch.Tensor]:
    """Unflatten 1D tensor back to list of tensors matching reference shapes."""
    result = []
    offset = 0
    for ref in reference:
        numel = ref.numel()
        result.append(flat[offset : offset + numel].reshape(ref.shape).clone())
        offset += numel
    return result


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
