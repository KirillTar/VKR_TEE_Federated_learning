"""
Flower FL server with Multi-Krum + Central DP aggregation strategy.
Designed to run inside SCONE/SGX enclave.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Optional

import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import get_model, get_weights, set_weights, weights_to_flat, flat_to_weights
from core.aggregation import multi_krum, fedavg, aggregate
from core.dp import adaptive_clip, add_gaussian_noise, compute_sigma

logger = logging.getLogger("fl.server")
logging.basicConfig(level=logging.INFO)


class KrumDPStrategy(fl.server.strategy.Strategy):
    """
    Custom Flower aggregation strategy implementing:
      1. Norm clipping (adaptive S = median(||Δ_k||))
      2. Multi-Krum selection
      3. Central DP noise
    """

    def __init__(
        self,
        n_clients: int,
        m_assumed: int,
        epsilon: float,
        delta: float = 1e-5,
        n_rounds: int = 200,
        aggregation: str = "multi_krum",
        sigma: Optional[float] = None,
        min_fit_clients: int = 20,
        min_evaluate_clients: int = 20,
        min_available_clients: int = 20,
    ):
        super().__init__()
        self.n_clients = n_clients
        self.m_assumed = m_assumed
        self.epsilon = epsilon
        self.delta = delta
        self.n_rounds = n_rounds
        self.aggregation = aggregation
        self.f = n_clients - m_assumed
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        # Compute or use provided sigma
        if sigma is not None:
            self.sigma = sigma
        elif epsilon < float("inf"):
            self.sigma = compute_sigma(epsilon, delta, T=n_rounds)
            logger.info(f"Computed σ={self.sigma:.4f} for ε={epsilon}, δ={delta}")
        else:
            self.sigma = 0.0
            logger.info("No DP noise (ε=∞)")

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return None  # clients initialize from scratch

    def configure_fit(self, server_round, parameters, client_manager):
        config = {"server_round": server_round}
        fit_ins = fl.common.FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(c, fit_ins) for c in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {}
        eval_ins = fl.common.EvaluateIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(c, eval_ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        # Extract weight deltas (client sends delta = trained - initial)
        ndarrays_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Convert to flat tensors
        flat_updates = [
            torch.tensor(np.concatenate([a.flatten() for a in ndarrays]), dtype=torch.float32)
            for ndarrays in ndarrays_list
        ]

        # Step 1: Norm clipping
        clipped, S = adaptive_clip(flat_updates)

        # Step 2: Aggregation
        agg_cfg = {"m_assumed": self.m_assumed, "f": self.f}
        result = aggregate(clipped, self.aggregation, agg_cfg)
        agg_update = result.update
        selected = result.selected

        # Step 3: DP noise
        if self.sigma > 0:
            agg_update = add_gaussian_noise(agg_update, self.sigma, S, f=self.f)

        # TPR/FPR logging (requires ground truth — skip in production TEE)
        metrics = {
            "round": server_round,
            "clip_S": float(S),
            "n_selected": len(selected) if selected else len(results),
        }
        logger.info(f"Round {server_round}: S={S:.4f}, selected={selected}")

        # Convert back to Parameters
        # We return the aggregated DELTA — clients must apply it to their model
        agg_np = agg_update.numpy()
        # Reconstruct list of ndarrays matching original shapes
        reference_ndarrays = ndarrays_list[0]
        shapes = [a.shape for a in reference_ndarrays]
        reconstructed = []
        offset = 0
        for shape in shapes:
            size = int(np.prod(shape))
            reconstructed.append(agg_np[offset:offset+size].reshape(shape))
            offset += size

        parameters = ndarrays_to_parameters(reconstructed)
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        # Average client evaluation losses
        total_examples = sum(r.num_examples for _, r in results)
        weighted_loss = sum(r.loss * r.num_examples for _, r in results) / total_examples
        weighted_acc = sum(
            r.metrics.get("accuracy", 0) * r.num_examples for _, r in results
        ) / total_examples
        logger.info(f"Round {server_round} eval: loss={weighted_loss:.4f}, acc={weighted_acc:.4f}")
        return weighted_loss, {"accuracy": weighted_acc}

    def evaluate(self, server_round, parameters):
        return None  # centralized evaluation not used


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=20)
    parser.add_argument("--m_assumed", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--aggregation", default="multi_krum")
    parser.add_argument("--server_address", default="0.0.0.0:8080")
    parser.add_argument("--min_fit_clients", type=int, default=None,
                        help="defaults to n_clients")
    parser.add_argument("--min_evaluate_clients", type=int, default=None,
                        help="defaults to n_clients")
    parser.add_argument("--min_available_clients", type=int, default=None,
                        help="defaults to n_clients")
    args = parser.parse_args()

    min_fit = args.min_fit_clients or args.n_clients
    min_eval = args.min_evaluate_clients or args.n_clients
    min_avail = args.min_available_clients or args.n_clients

    strategy = KrumDPStrategy(
        n_clients=args.n_clients,
        m_assumed=args.m_assumed,
        epsilon=args.epsilon,
        delta=args.delta,
        n_rounds=args.n_rounds,
        aggregation=args.aggregation,
        min_fit_clients=min_fit,
        min_evaluate_clients=min_eval,
        min_available_clients=min_avail,
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.n_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
