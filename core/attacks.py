"""Poisoning attacks: LabelFlip, ALIE, IPM, Fang/Min-Max, Backdoor (Model Replacement)."""

import math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm as scipy_norm
from torch.utils.data import DataLoader, Dataset
from typing import Optional


# ---------------------------------------------------------------------------
# Label Flipping (data poisoning)
# ---------------------------------------------------------------------------

class LabelFlipDataset(Dataset):
    """Wraps a dataset, flipping source_class labels to target_class."""

    def __init__(self, dataset, source_class: int, target_class: int,
                 flip_ratio: float = 1.0, seed: int = 0):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class

        # Pre-compute flip mask deterministically (same pattern as BackdoorDataset)
        rng = np.random.default_rng(seed)
        if hasattr(dataset, "targets"):
            targets = np.array(dataset.targets)
        else:
            targets = np.array([dataset[i][1] for i in range(len(dataset))])
        is_source = targets == source_class
        random_mask = rng.random(len(targets)) < flip_ratio
        self.flip_mask = is_source & random_mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.flip_mask[idx]:
            y = self.target_class
        return x, y


def label_flip_attack(
    client_dataset,
    source_class: int = 5,
    target_class: int = 3,
    flip_ratio: float = 1.0,
    seed: int = 0,
):
    """Return a poisoned version of client_dataset."""
    return LabelFlipDataset(client_dataset, source_class, target_class, flip_ratio, seed)


class RandomLabelFlipDataset(Dataset):
    """Wraps a dataset, replacing all labels with a random class permutation.
    All n_classes labels are remapped: label c → perm[c] (bijective).
    This corrupts ~90% of labels (all classes except fixed points of perm).
    Stronger than class-specific flip; still deterministic given seed.
    """

    def __init__(self, dataset, n_classes: int = 10, seed: int = 0):
        self.dataset = dataset
        rng = np.random.default_rng(seed)
        # Generate a random permutation with no fixed points (derangement).
        # Simple approach: shuffle until no fixed point exists.
        perm = np.arange(n_classes)
        while np.any(perm == np.arange(n_classes)):
            rng.shuffle(perm)
        self.perm = perm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, int(self.perm[y])


def random_label_flip_attack(client_dataset, n_classes: int = 10, seed: int = 0):
    """Return a dataset with all labels randomly permuted (derangement)."""
    return RandomLabelFlipDataset(client_dataset, n_classes, seed)



# ---------------------------------------------------------------------------
# IPM: Inner Product Manipulation (untargeted model poisoning)
# ---------------------------------------------------------------------------

def ipm_attack(
    honest_updates: list[torch.Tensor],
    epsilon: float = 0.1,
) -> torch.Tensor:
    """
    Inner Product Manipulation attack (Xie et al., ICML 2020).

    Crafts malicious update that makes the inner product between the
    aggregate and the true mean gradient negative, causing the model
    to move away from the optimum.

    V_mal = -ε · mean(honest_updates)

    Args:
        honest_updates: all honest client updates
        epsilon: scaling factor (default 0.1; larger = more aggressive but detectable)

    Returns:
        malicious flat update tensor (same for all Byzantine clients)
    """
    mu = torch.stack(honest_updates).mean(dim=0)
    return -epsilon * mu


# ---------------------------------------------------------------------------
# Fang Attack (adaptive model poisoning targeting Krum)
# ---------------------------------------------------------------------------

def fang_attack(
    honest_updates: list[torch.Tensor],
    n_malicious: int,
    m_assumed: int | None = None,
) -> torch.Tensor:
    """
    Fang attack (Fang et al., USENIX Security 2020).

    Crafts malicious updates to pass Multi-Krum while maximizing deviation.
    Finds direction opposite to honest mean, then searches for maximum
    perturbation λ such that the malicious updates still get selected by Krum.

    Adapted from FedRoLA (Yan et al., KDD 2024) implementation.

    Args:
        honest_updates: list of honest client update tensors (1D)
        n_malicious: number of malicious clients to generate
        m_assumed: Krum's m parameter (default: n_malicious)

    Returns:
        malicious flat update tensor (same for all Byzantine clients)
    """
    from .aggregation import multi_krum

    if n_malicious <= 0:
        return torch.stack(honest_updates).mean(dim=0)

    if m_assumed is None:
        m_assumed = n_malicious

    mu = torch.stack(honest_updates).mean(dim=0)
    # Direction: negative of honest mean (push away from convergence)
    direction = -mu / (mu.norm() + 1e-8)

    # Binary search for maximum λ such that at least one malicious
    # update gets selected by Multi-Krum
    lam = 1.0
    best_update = mu.clone()
    n_honest = len(honest_updates)

    for _ in range(20):  # max iterations
        candidate = mu + lam * direction
        # Simulate: n_malicious copies of candidate + honest updates
        test_updates = [candidate] * n_malicious + honest_updates
        n_total = len(test_updates)

        result = multi_krum(test_updates, m=m_assumed)
        selected = result.selected or []

        # Check if any malicious (indices 0..n_malicious-1) got selected
        if any(idx < n_malicious for idx in selected):
            best_update = candidate
            lam *= 2.0  # try larger
        else:
            lam *= 0.5  # too aggressive, reduce

        if lam < 1e-6:
            break

    return best_update


# ---------------------------------------------------------------------------
# Min-Max Attack (adaptive model poisoning)
# ---------------------------------------------------------------------------

def minmax_attack(
    honest_updates: list[torch.Tensor],
    n_malicious: int,
) -> torch.Tensor:
    """
    Min-Max attack (Shejwalkar & Houmansadr, USENIX Security 2021).

    Crafts malicious update along the negative direction of the honest mean,
    maximizing deviation while keeping max distance to any honest update
    below the max pairwise distance among honest updates.

    Adapted from FedRoLA (Yan et al., KDD 2024) implementation.

    Args:
        honest_updates: list of honest client update tensors (1D)
        n_malicious: number of malicious clients

    Returns:
        malicious flat update tensor (same for all Byzantine clients)
    """
    if n_malicious <= 0:
        return torch.stack(honest_updates).mean(dim=0)

    stacked = torch.stack(honest_updates)  # (n_honest, d)
    mu = stacked.mean(dim=0)

    # Direction: negative of honest mean
    direction = -mu / (mu.norm() + 1e-8)

    # Compute max pairwise distance among honest updates
    n_honest = len(honest_updates)
    max_dist_honest = 0.0
    for i in range(n_honest):
        for j in range(i + 1, n_honest):
            d = (honest_updates[i] - honest_updates[j]).norm().item()
            max_dist_honest = max(max_dist_honest, d)

    # Binary search for maximum γ such that max distance to honest < max_dist_honest
    gamma = 1.0
    best_update = mu.clone()

    for _ in range(20):
        candidate = mu + gamma * direction
        max_dist_to_honest = max(
            (candidate - honest_updates[i]).norm().item()
            for i in range(n_honest)
        )

        if max_dist_to_honest <= max_dist_honest:
            best_update = candidate
            gamma *= 2.0
        else:
            gamma *= 0.5

        if gamma < 1e-6:
            break

    return best_update


# ---------------------------------------------------------------------------
# ALIE: A Little Is Enough (model poisoning)
# ---------------------------------------------------------------------------

def compute_alie_z_max(n: int, m: int) -> float:
    """
    Compute optimal z for ALIE (Baruch et al. 2019, Algorithm 1).

    z = Φ⁻¹((n - m - s) / (n - m))
    where s = ⌊n/2⌋ + 1 - m

    The z value ensures that malicious updates μ + z·σ per coordinate
    fall within the statistical range of honest updates, evading
    distance-based defenses like Krum.

    For n=20, m=4:  z ≈ 0.157
    For n=40, m=8:  z ≈ 0.237
    For n=40, m=12: z ≈ 0.466
    """
    s = n // 2 + 1 - m
    n_honest = n - m
    if n_honest <= 0 or s <= 0 or s >= n_honest:
        # Degenerate case: majority Byzantine or invalid parameters.
        # Fall back to small z that still shifts updates subtly.
        return 0.1
    p = (n_honest - s) / n_honest
    return scipy_norm.ppf(p)


def alie_attack(
    honest_updates: list[torch.Tensor],
    n: int,
    m: int,
    z: Optional[float] = None,
) -> torch.Tensor:
    """
    ALIE attack (full knowledge): compute malicious update that passes Krum.

    V_mal[j] = μ_j + z · σ_j  (per coordinate)

    Args:
        honest_updates: all honest client updates (used for μ, σ estimation)
        n: total number of clients
        m: number of malicious clients
        z: perturbation scale (default: z_max for evading Krum)

    Returns:
        malicious flat update tensor
    """
    if z is None:
        z = compute_alie_z_max(n, m)

    stacked = torch.stack(honest_updates)  # (n_honest, d)
    mu = stacked.mean(dim=0)
    sigma = stacked.std(dim=0)

    return mu + z * sigma


# ---------------------------------------------------------------------------
# Backdoor via Model Replacement (targeted model poisoning)
# ---------------------------------------------------------------------------

class BackdoorDataset(Dataset):
    """
    Inserts trigger pattern into a fraction of the dataset, changing labels to target.
    """

    def __init__(
        self,
        dataset,
        target_label: int = 0,
        poison_ratio: float = 0.5,
        trigger_size: int = 5,
        trigger_value: Optional[torch.Tensor] = None,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.trigger_size = trigger_size
        # trigger_value: (C, 1, 1) tensor — white pixel in normalized space.
        # If None, falls back to 1.0 (unnormalized, kept for backward compat).
        self.trigger_value = trigger_value

        rng = np.random.default_rng(seed)
        n = len(dataset)
        self.poisoned_mask = rng.random(n) < poison_ratio

    def _apply_trigger(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 5×5 trigger to bottom-right corner (white pixel in normalized space)."""
        x = x.clone()
        s = self.trigger_size
        # trigger_value is (C, 1, 1) → broadcasts to (C, s, s)
        x[:, -s:, -s:] = self.trigger_value if self.trigger_value is not None else 1.0
        return x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.poisoned_mask[idx]:
            x = self._apply_trigger(x)
            y = self.target_label
        return x, y


def apply_trigger(
    x: torch.Tensor,
    trigger_size: int = 5,
    trigger_value: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply backdoor trigger to a batch or single image tensor."""
    x = x.clone()
    if x.dim() == 3:  # (C, H, W)
        # trigger_value (C, 1, 1) broadcasts to (C, s, s)
        x[:, -trigger_size:, -trigger_size:] = (
            trigger_value if trigger_value is not None else 1.0
        )
    elif x.dim() == 4:  # (B, C, H, W)
        # trigger_value (C, 1, 1) → (1, C, 1, 1) broadcasts to (B, C, s, s)
        val = trigger_value.unsqueeze(0) if trigger_value is not None else 1.0
        x[:, :, -trigger_size:, -trigger_size:] = val
    return x


def backdoor_model_replacement(
    malicious_update: torch.Tensor,
    gamma: float,
    n: int,
) -> torch.Tensor:
    """
    Scale malicious update for Model Replacement attack.
    gamma ∈ {1, n/2, n} — scaling factor.
    With FedAvg, server averages all n updates → malicious client
    boosts its update by γ to replace the global model.
    """
    return gamma * malicious_update


def train_backdoor_client(
    model: nn.Module,
    clean_loader: DataLoader,
    poisoned_dataset,
    optimizer,
    criterion,
    device: torch.device,
    backdoor_epochs: int = 10,
    gamma: float = 1.0,
    n_clients: int = 20,
) -> torch.Tensor:
    """
    Train a backdoor client: extra epochs on poisoned data + model replacement scaling.
    Returns flat malicious update (delta w).
    """
    from .models import get_weights, weights_to_flat

    initial_weights = get_weights(model)
    initial_flat = weights_to_flat(initial_weights)

    # Train more epochs on poisoned data
    from torch.utils.data import DataLoader as DL
    poisoned_loader = DL(poisoned_dataset, batch_size=32, shuffle=True, num_workers=2)

    model.train()
    for _ in range(backdoor_epochs):
        for x, y in poisoned_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    trained_weights = get_weights(model)
    trained_flat = weights_to_flat(trained_weights)

    delta = trained_flat - initial_flat
    return backdoor_model_replacement(delta, gamma=gamma, n=n_clients)
