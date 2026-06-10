"""Evaluation metrics: MTA, ASR, TPR, FPR."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Optional


@torch.no_grad()
def compute_mta(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Main Task Accuracy on clean test set."""
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def compute_per_class_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    n_classes: int = 10,
) -> dict[int, float]:
    """Per-class accuracy on clean test set."""
    model.eval()
    correct = torch.zeros(n_classes)
    total = torch.zeros(n_classes)

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        for c in range(n_classes):
            mask = (y == c)
            correct[c] += (preds[mask] == y[mask]).sum().item()
            total[c] += mask.sum().item()

    return {c: (correct[c] / total[c]).item() if total[c] > 0 else 0.0
            for c in range(n_classes)}


@torch.no_grad()
def compute_label_flip_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    source_class: int,
    target_class: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Metrics for targeted label-flip attack (source_class -> target_class).

    Returns:
        source_acc: accuracy on samples of source_class (how well the class is preserved).
        target_confusion: fraction of source_class samples predicted as target_class
                          (analogue of ASR for label flipping).
    """
    model.eval()
    correct = 0
    flipped = 0
    total = 0
    for x, y in test_loader:
        mask = (y == source_class)
        if mask.sum() == 0:
            continue
        x_s = x[mask].to(device)
        preds = model(x_s).argmax(dim=1)
        correct += (preds == source_class).sum().item()
        flipped += (preds == target_class).sum().item()
        total += x_s.size(0)
    if total == 0:
        return 0.0, 0.0
    return correct / total, flipped / total


@torch.no_grad()
def compute_asr(
    model: nn.Module,
    test_loader: DataLoader,
    trigger_fn: Callable[[torch.Tensor], torch.Tensor],
    target_label: int,
    device: torch.device,
    source_class: Optional[int] = None,
) -> float:
    """
    Attack Success Rate: fraction of (optionally source-class) samples
    with trigger classified as target_label.

    Args:
        trigger_fn: function that applies backdoor trigger to a batch
        source_class: if not None, only test on samples from this class
    """
    model.eval()
    success = 0
    total = 0

    for x, y in test_loader:
        if source_class is not None:
            mask = (y == source_class)
            if mask.sum() == 0:
                continue
            x, y = x[mask], y[mask]

        x_triggered = trigger_fn(x).to(device)
        preds = model(x_triggered).argmax(dim=1)
        success += (preds == target_label).sum().item()
        total += x.size(0)

    return success / total if total > 0 else 0.0


def compute_tpr_fpr(
    selected_indices: list[int],
    malicious_indices: set[int],
    n_clients: int,
) -> tuple[float, float]:
    """
    Compute TPR and FPR for Multi-Krum detection of malicious clients.

    selected = indices of clients SELECTED (included in aggregation).
    rejected = all - selected.

    TPR = (rejected ∩ malicious) / |malicious|   (NaN → 0.0 when no malicious)
    FPR = (rejected ∩ honest)   / |honest|
    """
    n_malicious = len(malicious_indices)
    n_honest = n_clients - n_malicious

    all_indices = set(range(n_clients))
    honest_indices = all_indices - malicious_indices
    rejected = all_indices - set(selected_indices)

    tpr = len(rejected & malicious_indices) / n_malicious if n_malicious > 0 else 0.0
    fpr = len(rejected & honest_indices) / n_honest if n_honest > 0 else 0.0

    return tpr, fpr


class RoundMetrics:
    """Accumulates per-round metrics across seeds."""

    def __init__(self):
        self.records: list[dict] = []

    def log(self, round_idx: int, seed: int, **kwargs):
        self.records.append({"round": round_idx, "seed": seed, **kwargs})

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.records)

    def summary(self) -> dict:
        """Mean ± std across seeds for the final round."""
        import pandas as pd
        df = pd.DataFrame(self.records)
        last = df[df["round"] == df["round"].max()]
        result = {}
        for col in last.columns:
            if col in ("round", "seed"):
                continue
            result[f"{col}_mean"] = last[col].mean()
            result[f"{col}_std"] = last[col].std()
        return result
