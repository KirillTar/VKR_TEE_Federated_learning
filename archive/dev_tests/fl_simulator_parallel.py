"""
Parallel FL simulator: trains clients in parallel using torch.multiprocessing.
Each client runs in a separate process, all sharing the same GPU.
Expected speedup: 3-5× over sequential simulator.
"""

import copy
import json
import logging
import os
from typing import Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from core.models import get_model, weights_to_flat
from core.data import load_dataset, dirichlet_split, get_test_dataloader
from core.aggregation import aggregate
from core.dp import adaptive_clip, add_gaussian_noise, compute_sigma
from core.attacks import (
    label_flip_attack, sign_flip_from_mean, alie_attack,
    BackdoorDataset, apply_trigger, backdoor_model_replacement,
)
from core.metrics import compute_mta, compute_asr, compute_tpr_fpr, RoundMetrics
from experiments.fl_simulator import FLConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker function (runs in a separate process)
# ---------------------------------------------------------------------------

# Module-level cache — lives for the lifetime of the worker process
_worker_cache: dict = {}


def _worker_init(gpu_id: int, dataset_name: str, data_dir: str):
    """Initialize CUDA and pre-load dataset once per worker process."""
    torch.cuda.set_device(gpu_id)
    train_ds, _ = load_dataset(dataset_name, data_dir)
    _worker_cache["train_ds"] = train_ds


def _train_one_client(args: tuple) -> torch.Tensor:
    """
    Worker: train one client and return flat delta (CPU tensor).
    Called via multiprocessing pool.

    args: (client_id, global_state_cpu, client_indices, dataset_name, data_dir,
           cfg_dict, attack, malicious_set, device_str)
    """
    (client_id, global_state_cpu, client_indices,
     dataset_name, data_dir, cfg_dict, attack, malicious_set, device_str) = args

    device = torch.device(device_str)
    is_malicious = client_id in malicious_set

    # Use pre-loaded dataset from worker initializer (avoids per-call reload)
    train_ds = _worker_cache.get("train_ds")
    if train_ds is None:
        train_ds, _ = load_dataset(dataset_name, data_dir)
        _worker_cache["train_ds"] = train_ds

    # Prepare (possibly poisoned) dataset
    ds = train_ds
    if is_malicious and attack == "label_flip":
        ds = label_flip_attack(ds, cfg_dict["flip_source"], cfg_dict["flip_target"])
    elif is_malicious and attack == "backdoor":
        ds = BackdoorDataset(
            ds,
            target_label=cfg_dict["backdoor_target_label"],
            poison_ratio=cfg_dict["backdoor_poison_ratio"],
            trigger_size=cfg_dict["backdoor_trigger_size"],
            seed=cfg_dict["seed"] + client_id,
        )

    subset = Subset(ds, client_indices)
    loader = DataLoader(subset, batch_size=cfg_dict["batch_size"], shuffle=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    # Build model and load global weights
    model = get_model(dataset_name).to(device)
    model.load_state_dict({k: v.to(device) for k, v in global_state_cpu.items()})
    model.train()

    n_epochs = cfg_dict["backdoor_epochs"] if (is_malicious and attack == "backdoor") else cfg_dict["local_epochs"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg_dict["lr"],
        momentum=cfg_dict["momentum"],
        weight_decay=cfg_dict["weight_decay"],
    )

    param_keys = {n for n, _ in model.named_parameters()}

    for _ in range(n_epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # Compute delta over trainable parameters only
    trained_flat = weights_to_flat([p.data.detach().cpu() for p in model.parameters()])
    init_flat = weights_to_flat([v for k, v in global_state_cpu.items() if k in param_keys])
    return trained_flat - init_flat


# ---------------------------------------------------------------------------
# Parallel FL Simulator
# ---------------------------------------------------------------------------

class FLSimulatorParallel:
    """Parallel FL simulator using torch.multiprocessing."""

    def __init__(self, cfg: FLConfig, n_workers: Optional[int] = None):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        torch.manual_seed(cfg.seed)

        # Load data
        self.train_ds, self.test_ds = load_dataset(cfg.dataset, cfg.data_dir)
        self.client_indices = dirichlet_split(self.train_ds, cfg.n_clients, cfg.alpha, cfg.seed)
        self.test_loader = get_test_dataloader(self.test_ds, num_workers=2)

        self.malicious_set = set(range(cfg.n_clients - cfg.m_real, cfg.n_clients))
        logger.info(f"Malicious clients: {sorted(self.malicious_set)}")

        self.model = get_model(cfg.dataset).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # DP
        if cfg.use_dp and cfg.sigma is None and cfg.epsilon < float("inf"):
            cfg.sigma = compute_sigma(cfg.epsilon, cfg.delta, T=cfg.n_rounds,
                                      method=cfg.dp_accounting)
            logger.info(f"σ={cfg.sigma:.4f} for ε={cfg.epsilon}")
        elif not cfg.use_dp or cfg.epsilon >= 1e9:
            cfg.sigma = 0.0

        if cfg.f is None:
            cfg.f = cfg.n_clients - cfg.m_assumed

        # Number of parallel workers: min(n_clients, available CPU cores)
        self.n_workers = n_workers or min(cfg.n_clients, os.cpu_count() or 4)
        logger.info(f"Parallel workers: {self.n_workers}")

        self._param_keys = [n for n, _ in self.model.named_parameters()]
        self._param_keys_set = set(self._param_keys)
        self._global_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        self.metrics = RoundMetrics()
        self._trigger_fn = lambda x: apply_trigger(x, cfg.backdoor_trigger_size)

    def _make_worker_args(self, client_id: int, global_state: dict) -> tuple:
        cfg = self.cfg
        cfg_dict = dict(
            batch_size=cfg.batch_size, lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay, local_epochs=cfg.local_epochs,
            backdoor_epochs=cfg.backdoor_epochs, backdoor_target_label=cfg.backdoor_target_label,
            backdoor_poison_ratio=cfg.backdoor_poison_ratio,
            backdoor_trigger_size=cfg.backdoor_trigger_size,
            flip_source=cfg.flip_source, flip_target=cfg.flip_target, seed=cfg.seed,
        )
        return (
            client_id,
            global_state,
            self.client_indices[client_id],
            cfg.dataset,
            cfg.data_dir,
            cfg_dict,
            cfg.attack,
            self.malicious_set,
            str(self.device),
        )

    def _aggregate_and_apply(self, updates: list[torch.Tensor], global_state: dict):
        cfg = self.cfg
        clipped, S = adaptive_clip(updates)
        agg_cfg = {"m_assumed": cfg.m_assumed, "f": cfg.f, "trim_beta": cfg.trim_beta}
        result = aggregate(clipped, cfg.aggregation, agg_cfg)
        agg_update, selected = result.update, result.selected

        if cfg.sigma and cfg.sigma > 0:
            f_eff = cfg.f if cfg.aggregation in ("multi_krum", "krum") else cfg.n_clients
            agg_update = add_gaussian_noise(agg_update, cfg.sigma, S, f=f_eff)

        param_keys = self._param_keys
        new_state = dict(global_state)
        offset = 0
        for key in param_keys:
            size = global_state[key].numel()
            shape = global_state[key].shape
            old = global_state[key].flatten()
            new_state[key] = (old + agg_update[offset:offset+size]).reshape(shape).clone()
            offset += size

        return new_state, selected, float(S)

    def run(self) -> RoundMetrics:
        cfg = self.cfg
        ctx = mp.get_context("spawn")
        logger.info(f"Parallel FL: rounds={cfg.n_rounds}, workers={self.n_workers}")

        global_state = self._global_state

        gpu_id = self.device.index if (self.device.type == "cuda" and self.device.index is not None) else 0
        with ctx.Pool(
            processes=self.n_workers,
            initializer=_worker_init,
            initargs=(gpu_id, cfg.dataset, cfg.data_dir),
        ) as pool:

            for t in range(cfg.n_rounds):
                # --- Build args for honest clients ---
                honest_args = [
                    self._make_worker_args(k, global_state)
                    for k in range(cfg.n_clients)
                    if k not in self.malicious_set
                ]

                # --- Train honest clients in parallel ---
                honest_deltas: list[torch.Tensor] = pool.map(_train_one_client, honest_args)
                honest_client_ids = [k for k in range(cfg.n_clients)
                                     if k not in self.malicious_set]

                all_updates: list[torch.Tensor | None] = [None] * cfg.n_clients
                for kid, delta in zip(honest_client_ids, honest_deltas):
                    all_updates[kid] = delta

                # --- Malicious clients (need honest_deltas first for ALIE) ---
                if cfg.attack in ("label_flip", "backdoor", "none"):
                    # Data poisoning: still train in parallel
                    mal_args = [
                        self._make_worker_args(k, global_state)
                        for k in self.malicious_set
                    ]
                    mal_deltas = pool.map(_train_one_client, mal_args)
                    for kid, delta in zip(sorted(self.malicious_set), mal_deltas):
                        if cfg.attack == "backdoor":
                            delta = backdoor_model_replacement(delta, cfg.backdoor_gamma, cfg.n_clients)
                        all_updates[kid] = delta

                elif cfg.attack == "sign_flip":
                    bad_delta = sign_flip_from_mean(honest_deltas)
                    for k in self.malicious_set:
                        all_updates[k] = bad_delta.clone()

                elif cfg.attack == "alie":
                    bad_delta = alie_attack(honest_deltas, n=cfg.n_clients, m=cfg.m_real)
                    for k in self.malicious_set:
                        all_updates[k] = bad_delta.clone()

                # --- Aggregate ---
                global_state, selected, S = self._aggregate_and_apply(all_updates, global_state)

                # --- Metrics ---
                if (t + 1) % cfg.log_every == 0 or t == cfg.n_rounds - 1:
                    self.model.load_state_dict(
                        {k: v.to(self.device) for k, v in global_state.items()}
                    )
                    mta = compute_mta(self.model, self.test_loader, self.device)
                    asr = 0.0
                    if cfg.attack == "backdoor":
                        asr = compute_asr(self.model, self.test_loader,
                                          self._trigger_fn, cfg.backdoor_target_label, self.device)
                    tpr, fpr = 0.0, 0.0
                    if selected is not None:
                        tpr, fpr = compute_tpr_fpr(selected, self.malicious_set, cfg.n_clients)

                    self.metrics.log(round_idx=t+1, seed=cfg.seed,
                                     mta=mta, asr=asr, tpr=tpr, fpr=fpr, clip_S=S)
                    logger.info(
                        f"Round {t+1:3d}/{cfg.n_rounds} | "
                        f"MTA={mta:.4f} | ASR={asr:.4f} | TPR={tpr:.3f} | FPR={fpr:.3f}"
                    )

        return self.metrics
