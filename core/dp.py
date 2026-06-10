"""Differential Privacy: norm clipping, Gaussian noise, RDP accounting.

Supports two DP granularity levels:
- Record-level (cross-silo): protects individual data records.
  Sensitivity = R / (n · |D_i|), where R = per-sample clip, n = clients, |D_i| = samples/client.
  Noise is tiny → large models viable.
- Client-level (cross-device): protects entire client datasets.
  Sensitivity = C / n, where C = per-client clip, n = clients in round.
  Noise is large → only small models viable.
"""

import logging
import math
import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DP Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPConfig:
    """DP parameters for a scenario."""
    level: str = "client"          # "record" or "client"
    epsilon: float = 10.0
    delta: float = 1e-5
    # Client-level clipping (applied on server to per-client updates)
    client_clip: float = 1.0       # C: per-client update L2 norm bound
    # Record-level clipping (applied on client to per-sample gradients)
    record_clip: float = 1.0       # R: per-sample gradient L2 norm bound
    # Adaptive clipping
    adaptive_clip: bool = True     # use median of norms for client_clip
    clip_max: float | None = None  # cap for adaptive clipping
    # Accounting
    sigma: float | None = None     # auto-computed if None
    sample_rate: float = 1.0       # q = clients_per_round / n_clients
    n_rounds: int = 200            # T: total rounds for accounting


# ---------------------------------------------------------------------------
# Norm clipping
# ---------------------------------------------------------------------------

def clip_update(update: torch.Tensor, S: float) -> torch.Tensor:
    """Clip a single flat update to L2 norm S."""
    norm = update.norm(p=2)
    if norm > S:
        return update * (S / norm)
    return update


def adaptive_clip(
    updates: list[torch.Tensor],
    S: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> tuple[list[torch.Tensor], float]:
    """
    Clip all updates to S. If S is None, use S = median(||Δ_k||).
    If clip_max is set, cap S at clip_max to prevent feedback loops.
    Returns (clipped_updates, S_used).
    """
    norms = torch.tensor([u.norm(p=2).item() for u in updates])
    if S is None:
        S = float(norms.median())
    if clip_max is not None:
        S = min(S, clip_max)
    clipped = [clip_update(u, S) for u in updates]
    return clipped, S


# ---------------------------------------------------------------------------
# Gaussian noise
# ---------------------------------------------------------------------------

def add_dp_noise(
    agg_update: torch.Tensor,
    sigma: float,
    sensitivity: float,
) -> torch.Tensor:
    """
    Add Gaussian noise for Central DP.
    noise ~ N(0, (σ · sensitivity)² · I)

    For client-level: sensitivity = C / f_eff
    For record-level: sensitivity = R / (n_active · |D_min|)
    """
    if sigma <= 0:
        return agg_update
    noise_std = sigma * sensitivity
    noise = torch.randn_like(agg_update) * noise_std
    return agg_update + noise


def compute_sensitivity_client(client_clip: float, f_eff: float) -> float:
    """Client-level DP sensitivity: adding/removing one client changes aggregate by at most C/f_eff."""
    return client_clip / f_eff


def compute_sensitivity_record(
    record_clip: float,
    n_active: int,
    samples_per_client: int,
) -> float:
    """Record-level DP sensitivity: adding/removing one record changes aggregate by at most R/(n·|D_i|)."""
    return record_clip / (n_active * samples_per_client)


# ---------------------------------------------------------------------------
# Legacy noise functions (backward compatibility)
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    agg_update: torch.Tensor,
    sigma: float,
    S: float,
    f: int,
) -> torch.Tensor:
    """Add Gaussian noise for Central DP after Multi-Krum averaging.
    Sensitivity = S/f. Noise ~ N(0, (σ · S/f)² · I).
    """
    return add_dp_noise(agg_update, sigma, S / f)


def add_gaussian_noise_fedavg(
    agg_update: torch.Tensor,
    sigma: float,
    S: float,
    n: int,
) -> torch.Tensor:
    """DP noise for plain FedAvg (sensitivity = S/n)."""
    return add_dp_noise(agg_update, sigma, S / n)


# ---------------------------------------------------------------------------
# RDP accounting — compute noise multiplier for target (ε, δ)
# ---------------------------------------------------------------------------

def compute_sigma_opacus(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
) -> float:
    """Compute noise multiplier via Opacus RDP accountant."""
    try:
        from opacus.accountants.utils import get_noise_multiplier
        sigma = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant="rdp",
        )
        return float(sigma)
    except ImportError:
        raise ImportError("Opacus not installed. Run: pip install opacus==1.5.2")


def compute_sigma(
    target_epsilon: float,
    target_delta: float,
    T: int,
    sample_rate: float = 1.0,
    method: str = "opacus",
) -> float:
    """
    Compute noise multiplier σ for target (ε, δ)-DP.

    Args:
        target_epsilon: privacy budget ε
        target_delta: failure probability δ
        T: number of FL rounds (composition steps)
        sample_rate: client sampling rate (q = clients_per_round / n_clients)
        method: "opacus"
    """
    if target_epsilon == float("inf") or target_epsilon > 1000:
        return 0.0

    return compute_sigma_opacus(target_epsilon, target_delta, sample_rate, T)


# ---------------------------------------------------------------------------
# Privacy accounting: compute actual ε spent so far
# ---------------------------------------------------------------------------

def compute_epsilon_spent(
    sigma: float,
    sample_rate: float,
    steps: int,
    delta: float,
    method: str = "opacus",
) -> float:
    """Compute ε spent after `steps` rounds of Gaussian mechanism."""
    if sigma == 0.0:
        return float("inf")

    try:
        from opacus.accountants import RDPAccountant
        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
        eps, _ = accountant.get_privacy_spent(delta=delta)
        return float(eps)
    except ImportError:
        pass

    return sigma  # placeholder
