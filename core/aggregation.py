"""Aggregation algorithms: FedAvg, Multi-Krum, TrimmedMean, FedRoLA-PCSI."""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple

logger = logging.getLogger(__name__)


class AggResult(NamedTuple):
    update: torch.Tensor          # aggregated flat update (1D)
    selected: list[int] | None    # indices selected by Multi-Krum (None for others)


def fedavg(updates: list[torch.Tensor]) -> AggResult:
    """Simple average of all updates."""
    stacked = torch.stack(updates)
    return AggResult(update=stacked.mean(dim=0), selected=None)


def multi_krum(
    updates: list[torch.Tensor],
    m: int,
    f: int | None = None,
) -> AggResult:
    """
    Multi-Krum aggregation.

    Args:
        updates: list of n flat update tensors (1D, same shape)
        m: assumed number of Byzantine clients (upper bound)
        f: number of updates to select (default: n - m)

    Returns:
        AggResult with averaged selected updates and their indices.
    """
    n = len(updates)
    if f is None:
        f = n - m

    assert m < n, f"m={m} must be < n={n}"
    assert f >= 1, f"f={f} must be >= 1"
    if n < 2 * m + 3:
        logger.warning(
            f"Multi-Krum: n={n} < 2m+3={2*m+3} — Byzantine resilience guarantees do not hold."
        )

    stacked = torch.stack(updates)  # (n, d)

    # Compute pairwise squared distances
    # ||u_i - u_j||² = ||u_i||² + ||u_j||² - 2 u_i·u_j
    norms_sq = (stacked ** 2).sum(dim=1)  # (n,)
    dot = stacked @ stacked.T             # (n, n)
    dist_sq = norms_sq.unsqueeze(1) + norms_sq.unsqueeze(0) - 2 * dot
    dist_sq = dist_sq.clamp(min=0)        # numerical safety

    # Krum score: for each i, sum of n-m-2 smallest distances to others
    n_neighbors = max(1, n - m - 2)
    scores = torch.zeros(n)
    for i in range(n):
        dists_i = dist_sq[i].clone()
        dists_i[i] = float("inf")         # exclude self
        sorted_dists, _ = dists_i.sort()
        scores[i] = sorted_dists[:n_neighbors].sum()

    # Select f updates with lowest scores
    _, top_indices = scores.topk(f, largest=False)
    selected = top_indices.tolist()

    agg = stacked[top_indices].mean(dim=0)
    return AggResult(update=agg, selected=selected)


def krum(updates: list[torch.Tensor], m: int) -> AggResult:
    """Standard Krum: selects f=1 update. Wrapper around multi_krum."""
    return multi_krum(updates, m=m, f=1)


def trimmed_mean(updates: list[torch.Tensor], beta: float) -> AggResult:
    """
    Coordinate-wise trimmed mean.

    Args:
        updates: list of n flat update tensors
        beta: fraction to trim from each end (e.g., 0.1 = trim 10% from each side)
    """
    n = len(updates)
    k = int(n * beta)
    assert 2 * k < n, f"Too much trimming: 2*{k} >= {n}"

    stacked = torch.stack(updates)  # (n, d)
    sorted_updates, _ = stacked.sort(dim=0)
    trimmed = sorted_updates[k : n - k]  # (n - 2k, d)
    return AggResult(update=trimmed.mean(dim=0), selected=None)


class FedRoLA:
    """FedRoLA-PCSI: layer-based robust aggregation (Yan et al., KDD 2024).

    Analyzes pairwise cosine similarity per layer to detect malicious clients.
    Uses probabilistic layer selection (layers that detected more get sampled more),
    client reputation tracking across rounds, and soft weighting (detected clients
    get reduced weight rather than being removed entirely).

    Key difference from Krum: operates on per-layer similarity, not full-vector distance.
    This catches attacks that concentrate in specific layers (e.g., last FC for backdoor).

    Args:
        layer_sizes: list of parameter counts per layer (from model.parameters())
        chi: discount factor for detected clients (default 0.1, per paper)
    """

    def __init__(
        self,
        layer_sizes: list[int],
        chi: float = 0.1,
        full_layer: bool = False,
    ):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.chi = chi
        self.full_layer = full_layer  # if True, evaluate ALL layers each round
        # Per-layer detection success/failure counts (for probabilistic selection)
        self.alpha = {i: 1 for i in range(self.num_layers)}
        self.beta = {i: 0 for i in range(self.num_layers)}
        # Per-client reputation (persistent across rounds)
        self.alpha_client: dict[int, int] = {}
        self.beta_client: dict[int, int] = {}
        self.round = 0
        # Last aggregation info (for DP noise calibration + diagnostics)
        self.last_weights: list[float] = []
        self.last_f_eff: float = 1.0
        self.last_selected_layers: list[int] = []
        self.last_n_detected: int = 0
        self.last_score_min: float = 0.0
        self.last_score_max: float = 0.0
        self.last_score_mean: float = 0.0
        self.last_disc: float = 0.0

    def _split_layers(self, flat: torch.Tensor) -> list[torch.Tensor]:
        """Split flat 1D tensor into per-layer 1D tensors."""
        layers = []
        offset = 0
        for size in self.layer_sizes:
            layers.append(flat[offset:offset + size])
            offset += size
        return layers

    def _get_disc(self) -> float:
        """Discrimination factor: sigmoid-like, increases with rounds.
        Early rounds: close to 0 (gentle). Later: close to 1 (strong penalization).
        """
        return 2.0 / (1.0 + np.exp(-self.chi * self.round)) - 1.0

    def _get_layer_probs(self) -> np.ndarray:
        """Probability distribution over layers based on past detection success."""
        total = np.array([self.alpha[i] + self.beta[i]
                          for i in range(self.num_layers)], dtype=float)
        probs = np.array([self.alpha[i] for i in range(self.num_layers)], dtype=float)
        probs = probs / total
        s = probs.sum()
        if s < 1e-8:
            return np.ones(self.num_layers) / self.num_layers
        return probs / s

    def detect(
        self,
        updates: list[torch.Tensor],
        client_ids: list[int] | None = None,
    ) -> AggResult:
        """Detect malicious clients and return soft-weighted aggregate.

        Args:
            updates: list of n flat update tensors (1D, same shape)
            client_ids: unique ids for reputation tracking (default: range(n))

        Returns:
            AggResult with weighted average.
            selected = indices of NON-detected clients (for TPR/FPR compatibility).
        """
        n = len(updates)
        if client_ids is None:
            client_ids = list(range(n))

        # Init reputation for new clients
        for uid in client_ids:
            if uid not in self.alpha_client:
                self.alpha_client[uid] = 1
                self.beta_client[uid] = 1

        # Split all updates into per-layer tensors
        all_layers = [self._split_layers(u) for u in updates]

        # Choose layers — full sweep (debug/sanity) or probabilistic (default)
        if self.full_layer:
            chosen = list(range(self.num_layers))
            n_choose = self.num_layers
        else:
            n_choose = min(3, self.num_layers)
            probs = self._get_layer_probs()
            chosen = np.random.choice(
                self.num_layers, n_choose, replace=False, p=probs
            )

        # Per-layer detection via pairwise cosine similarity (PCSI)
        vote_counts: dict[int, int] = {}
        # Diagnostic: collect mean_sims across all chosen layers for distribution stats
        all_mean_sims: list[float] = []

        for layer_idx in chosen:
            layer_vecs = [all_layers[i][layer_idx] for i in range(n)]
            stacked = torch.stack(layer_vecs)
            norms = stacked.norm(dim=1, keepdim=True).clamp(min=1e-8)
            normalized = stacked / norms
            sim_matrix = (normalized @ normalized.T).cpu().numpy()

            # Mean of top-2 pairwise similarities (excluding self)
            mean_sims = []
            for i in range(n):
                sims_i = sim_matrix[i].copy()
                sims_i[i] = -999.0
                top2 = np.sort(sims_i)[-2:]
                mean_sims.append(float(np.mean(top2)))
            all_mean_sims.extend(mean_sims)

            # Threshold search: find suspiciously similar clients
            threshold = 0.9
            bad_ids: list[int] = []
            found = False
            while threshold >= -0.05:
                flagged = [i for i, s in enumerate(mean_sims) if s >= threshold]
                if len(flagged) < n / 2:
                    bad_ids = flagged
                    found = True
                    self.alpha[layer_idx] += 1
                    break
                threshold -= 0.1

            if not found:
                self.beta[layer_idx] += 1

            for idx in bad_ids:
                vote_counts[idx] = vote_counts.get(idx, 0) + 1

        # Voting: start with unanimous, relax until we have detections
        detected_indices: list[int] = []
        vote_threshold = n_choose
        while not detected_indices and vote_threshold >= 1:
            detected_indices = [
                idx for idx, cnt in vote_counts.items()
                if cnt >= vote_threshold
            ]
            vote_threshold -= 1

        # Update client reputation
        detected_set = set(client_ids[idx] for idx in detected_indices)
        for uid in client_ids:
            if uid in detected_set:
                self.beta_client[uid] += 1
            else:
                self.alpha_client[uid] += 1

        # Soft weighting: all clients participate with reputation-based weights
        disc = self._get_disc()
        weights = []
        for i, uid in enumerate(client_ids):
            prob = self.alpha_client[uid] / (
                self.alpha_client[uid] + self.beta_client[uid]
            )
            if uid in detected_set:
                prob *= disc
            weights.append(max(prob, 1e-8))

        # Normalize weights
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        # Weighted average
        weight_tensor = torch.tensor(weights, dtype=updates[0].dtype)
        stacked_all = torch.stack(updates)
        agg = (stacked_all * weight_tensor.unsqueeze(1)).sum(dim=0)

        # Store for DP noise calibration + diagnostics
        self.last_weights = weights
        self.last_f_eff = 1.0 / max(weights)  # effective number of clients
        self.last_selected_layers = [int(x) for x in chosen]
        self.last_n_detected = len(detected_indices)
        if all_mean_sims:
            self.last_score_min = float(min(all_mean_sims))
            self.last_score_max = float(max(all_mean_sims))
            self.last_score_mean = float(sum(all_mean_sims) / len(all_mean_sims))
        self.last_disc = float(disc)

        self.round += 1

        # selected = non-detected (for TPR/FPR: rejected = all - selected)
        detected_pos_set = set(detected_indices)
        selected = [i for i in range(n) if i not in detected_pos_set]

        logger.debug(
            f"FedRoLA: round {self.round}, detected {len(detected_indices)}/{n}, "
            f"f_eff={self.last_f_eff:.1f}, disc={disc:.3f}"
        )

        return AggResult(update=agg, selected=selected if detected_indices else None)


def median(updates: list[torch.Tensor]) -> AggResult:
    """Coordinate-wise median."""
    stacked = torch.stack(updates)  # (n, d)
    med = stacked.median(dim=0).values
    return AggResult(update=med, selected=None)


def _fedrola_dispatch(updates, cfg):
    inst = cfg.get("_fedrola_inst")
    if inst is None:
        inst = FedRoLA(layer_sizes=cfg["layer_sizes"], chi=cfg.get("chi", 0.1))
        cfg["_fedrola_inst"] = inst
    return inst.detect(updates, cfg.get("client_ids"))


# Registry for easy lookup by name
AGGREGATION_REGISTRY = {
    "fedavg": lambda updates, cfg: fedavg(updates),
    "multi_krum": lambda updates, cfg: multi_krum(updates, m=cfg["m_assumed"], f=cfg.get("f")),
    "krum": lambda updates, cfg: krum(updates, m=cfg["m_assumed"]),
    "trimmed_mean": lambda updates, cfg: trimmed_mean(updates, beta=cfg.get("trim_beta", 0.1)),
    "median": lambda updates, cfg: median(updates),
    "fedrola": _fedrola_dispatch,
}


def aggregate(
    updates: list[torch.Tensor],
    method: str,
    cfg: dict,
) -> AggResult:
    """Dispatch aggregation by name."""
    if method not in AGGREGATION_REGISTRY:
        raise ValueError(f"Unknown aggregation method: {method}. "
                         f"Available: {list(AGGREGATION_REGISTRY)}")
    return AGGREGATION_REGISTRY[method](updates, cfg)
