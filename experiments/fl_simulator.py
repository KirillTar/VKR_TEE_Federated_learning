"""
Single-process FL simulator.
Supports two scenarios:
- Cross-silo: N=50, record-level DP, larger models (~90K), E=1
- Cross-device: N=200, client-level DP, small models (~4K), E=5
"""

import copy
import itertools
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as TF
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from core.models import (
    get_model, get_weights, set_weights, weights_to_flat, flat_to_weights,
    count_parameters,
)
from core.data import (
    load_dataset, dirichlet_split, get_test_dataloader,
    CIFAR10_MEAN, CIFAR10_STD, SVHN_MEAN, SVHN_STD, FMNIST_MEAN, FMNIST_STD, MNIST_MEAN, MNIST_STD,
)
from core.aggregation import aggregate, FedRoLA, AggResult
from core.dp import (
    adaptive_clip, add_dp_noise, compute_sigma,
    compute_sensitivity_client, compute_sensitivity_record,
)
from core.attacks import (
    label_flip_attack, random_label_flip_attack,
    alie_attack, ipm_attack, fang_attack, minmax_attack,
    BackdoorDataset, apply_trigger, backdoor_model_replacement,
)
from core.metrics import (
    compute_mta, compute_asr, compute_tpr_fpr, compute_label_flip_metrics, RoundMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class FLConfig:
    # Data
    dataset: str = "fashion_mnist"
    data_dir: str = "./data"
    alpha: float = 0.3              # Dirichlet non-IID parameter
    seed: int = 42
    model_name: Optional[str] = None

    # FL
    n_clients: int = 200
    clients_per_round: int = 20
    n_rounds: int = 200
    local_epochs: int = 5
    batch_size: int = 32
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Aggregation
    aggregation: str = "fedavg"     # "fedavg", "multi_krum", "fedrola", "trimmed_mean"
    m_assumed: int = 6
    f: Optional[int] = None
    trim_beta: float = 0.1
    fedrola_chi: float = 0.1        # FedRoLA discount factor
    fedrola_full_layer: bool = False  # if True, evaluate ALL layers each round (sanity)

    # DP
    use_dp: bool = True
    dp_level: str = "client"        # "client" or "record"
    epsilon: float = 10.0
    delta: float = 1e-5
    sigma: Optional[float] = None
    # Client-level clipping (applied on server to per-client updates)
    client_clip: float = 1.0
    adaptive_client_clip: bool = True  # use median of norms
    clip_max: Optional[float] = None
    # Record-level clipping (applied on client to per-sample gradients via Opacus)
    record_clip: float = 1.0

    # Attacks
    attack: str = "none"            # "none", "label_flip", "random_label_flip", "alie", "ipm", "fang", "minmax", "backdoor"
    m_real: int = 0
    flip_source: int = 5
    flip_target: int = 3
    ipm_epsilon: float = 0.1        # IPM scaling factor
    backdoor_gamma: float = 20.0
    backdoor_trigger_size: int = 5
    backdoor_target_label: int = 0
    backdoor_poison_ratio: float = 0.5
    backdoor_epochs: int = 10

    # LR schedule
    lr_end: Optional[float] = None   # if set, linear decay lr → lr_end over T rounds

    # DP mode
    dp_mode: str = "central"         # "central" (server adds noise) or "local" (client adds noise)

    # Logging
    log_every: int = 10
    save_dir: str = "./results"


class FLSimulator:
    """Federated learning single-process simulator."""

    def __init__(self, cfg: FLConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        torch.manual_seed(cfg.seed)

        # Load data
        self.train_ds, self.test_ds = load_dataset(cfg.dataset, cfg.data_dir)
        self.client_indices = dirichlet_split(self.train_ds, cfg.n_clients, cfg.alpha, cfg.seed)
        self.test_loader = get_test_dataloader(self.test_ds, num_workers=0)

        # Samples per client (for record-level DP sensitivity)
        self._samples_per_client = [len(idx) for idx in self.client_indices]
        self._min_samples = min(self._samples_per_client) if self._samples_per_client else 1

        # Malicious clients: last m_real indices
        self.malicious_set = set(range(cfg.n_clients - cfg.m_real, cfg.n_clients))
        logger.info(f"Malicious: {cfg.m_real}/{cfg.n_clients}, attack={cfg.attack}")

        # Trigger value for backdoor (white pixel in normalized space)
        _ds_stats = {
            "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
            "svhn": (SVHN_MEAN, SVHN_STD),
            "fashion_mnist": (FMNIST_MEAN, FMNIST_STD),
            "mnist": (MNIST_MEAN, MNIST_STD),
        }
        _mean, _std = _ds_stats.get(cfg.dataset, (MNIST_MEAN, MNIST_STD))
        self._trigger_value = torch.tensor(
            [(1.0 - m) / s for m, s in zip(_mean, _std)]
        ).view(-1, 1, 1)

        # Global model
        self.model = get_model(cfg.dataset, cfg.model_name).to(self.device)
        d = count_parameters(self.model)
        self.criterion = nn.CrossEntropyLoss()

        # Pre-build per-client loaders
        self.client_loaders: list[DataLoader] = []
        for k in range(cfg.n_clients):
            is_malicious = k in self.malicious_set
            ds = self.train_ds

            if is_malicious and cfg.attack == "label_flip":
                ds = label_flip_attack(ds, cfg.flip_source, cfg.flip_target, seed=cfg.seed + k)
            elif is_malicious and cfg.attack == "random_label_flip":
                ds = random_label_flip_attack(ds, n_classes=10, seed=cfg.seed + k)
            elif is_malicious and cfg.attack == "backdoor":
                ds = BackdoorDataset(
                    ds, target_label=cfg.backdoor_target_label,
                    poison_ratio=cfg.backdoor_poison_ratio,
                    trigger_size=cfg.backdoor_trigger_size,
                    trigger_value=self._trigger_value,
                    seed=cfg.seed + k,
                )

            subset = Subset(ds, self.client_indices[k])
            nw = 0 if cfg.n_clients >= 50 else 2
            pm = (self.device.type == "cuda") and (nw > 0)
            loader = DataLoader(
                subset, batch_size=cfg.batch_size, shuffle=True,
                num_workers=nw, pin_memory=pm,
                persistent_workers=(nw > 0),
            )
            self.client_loaders.append(loader)

        # Local model (reused for client-level DP; record-level creates fresh each time)
        self.local_model = get_model(cfg.dataset, cfg.model_name).to(self.device)
        self.local_model.train()

        # DP sigma computation
        _sample_rate = cfg.clients_per_round / cfg.n_clients
        self._sigma_local = 0.0  # Local DP noise multiplier for Opacus

        if cfg.use_dp and cfg.sigma is None and cfg.epsilon < float("inf"):
            cfg.sigma = compute_sigma(
                target_epsilon=cfg.epsilon,
                target_delta=cfg.delta,
                T=cfg.n_rounds,
                sample_rate=_sample_rate,
            )
            logger.info(
                f"σ={cfg.sigma:.4f} for ε={cfg.epsilon}, δ={cfg.delta}, "
                f"q={_sample_rate:.3f}, dp_level={cfg.dp_level}"
            )
        elif not cfg.use_dp or cfg.epsilon >= 1e9:
            cfg.sigma = 0.0

        # Local DP: compute per-client noise multiplier
        if cfg.dp_mode == "local" and cfg.use_dp:
            import math
            expected_participations = cfg.n_rounds * _sample_rate

            if cfg.dp_level == "record":
                # DP-SGD path (Opacus): noise on per-sample gradients.
                batches_per_epoch = math.ceil(self._min_samples / cfg.batch_size)
                total_local_steps = int(expected_participations * cfg.local_epochs * batches_per_epoch)
                local_sample_rate = cfg.batch_size / self._min_samples
                self._sigma_local = compute_sigma(
                    target_epsilon=cfg.epsilon,
                    target_delta=cfg.delta,
                    T=total_local_steps,
                    sample_rate=local_sample_rate,
                )
                logger.info(
                    f"Local DP (record): σ_local={self._sigma_local:.4f} for ε={cfg.epsilon}, "
                    f"total_steps={total_local_steps}, p={local_sample_rate:.4f}"
                )
            else:
                # Client-level: each participation = one Gaussian release on Δ
                # (sensitivity = client_clip, fixed). Composition over E[participations].
                self._sigma_local = compute_sigma(
                    target_epsilon=cfg.epsilon,
                    target_delta=cfg.delta,
                    T=max(1, int(round(expected_participations))),
                    sample_rate=1.0,
                )
                logger.info(
                    f"Local DP (client): σ_local={self._sigma_local:.4f} for ε={cfg.epsilon}, "
                    f"E[participations]={expected_participations:.1f}, C={cfg.client_clip}"
                )

        # f for Multi-Krum
        if cfg.f is None:
            cfg.f = cfg.clients_per_round - cfg.m_assumed

        # FedRoLA: stateful aggregator (persists across rounds)
        self._fedrola: FedRoLA | None = None
        if cfg.aggregation == "fedrola":
            layer_sizes = [p.numel() for p in self.model.parameters()]
            self._fedrola = FedRoLA(
                layer_sizes, chi=cfg.fedrola_chi, full_layer=cfg.fedrola_full_layer,
            )
            logger.info(
                f"FedRoLA initialized: {len(layer_sizes)} layers, "
                f"chi={cfg.fedrola_chi}, full_layer={cfg.fedrola_full_layer}"
            )

        self.metrics = RoundMetrics()
        self._trigger_fn = lambda x: apply_trigger(x, cfg.backdoor_trigger_size, self._trigger_value)

        # Param keys vs buffer keys
        self._param_keys: list[str] = [n for n, _ in self.model.named_parameters()]
        self._param_keys_set: set[str] = set(self._param_keys)

        # Cache initial global state
        self._global_state: dict = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }

        logger.info(
            f"Model: {cfg.model_name or 'default'}, d={d}, "
            f"agg={cfg.aggregation}, dp_level={cfg.dp_level}"
        )

    # -----------------------------------------------------------------
    # Client training
    # -----------------------------------------------------------------

    def _train_client(
        self,
        client_id: int,
        global_state: dict,
        n_epochs: int,
        lr: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Standard client training (no per-sample clipping)."""
        self.local_model.load_state_dict(
            {k: v.to(self.device) for k, v in global_state.items()}
        )
        self.local_model.train()

        optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=lr if lr is not None else self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )

        loader = self.client_loaders[client_id]
        data_stream = itertools.chain.from_iterable(
            iter(loader) for _ in range(n_epochs)
        )
        for x, y in data_stream:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self.local_model(x), y)
            loss.backward()
            optimizer.step()

        trained_flat = weights_to_flat(
            [p.data.detach().cpu() for p in self.local_model.parameters()]
        )
        init_flat = weights_to_flat([global_state[k] for k in self._param_keys])
        param_delta = trained_flat - init_flat

        buf_state = {
            k: v.detach().cpu().clone()
            for k, v in self.local_model.state_dict().items()
            if k not in self._param_keys_set
        }

        return param_delta, buf_state

    def _train_client_dpsgd(
        self,
        client_id: int,
        global_state: dict,
        n_epochs: int,
        lr: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Train with per-sample gradient clipping (DP-SGD).
        Central DP: clipping on client, noise on server (noise_multiplier=0).
        Local DP: clipping + noise on client (noise_multiplier=σ_local).
        Creates a fresh model each call because Opacus wraps the module.
        """
        from opacus import PrivacyEngine

        # Fresh model (Opacus wraps it)
        model = get_model(self.cfg.dataset, self.cfg.model_name).to(self.device)
        model.load_state_dict(
            {k: v.to(self.device) for k, v in global_state.items()}
        )

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr if lr is not None else self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )

        # Fresh loader (Opacus replaces the sampler)
        subset = Subset(
            self.client_loaders[client_id].dataset.dataset
            if hasattr(self.client_loaders[client_id].dataset, 'dataset')
            else self.train_ds,
            self.client_indices[client_id],
        )
        loader = DataLoader(subset, batch_size=self.cfg.batch_size, shuffle=True)

        # Local DP: Opacus adds noise per-step; Central DP: no local noise
        noise_mult = self._sigma_local if self.cfg.dp_mode == "local" else 0.0

        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=noise_mult,
            max_grad_norm=self.cfg.record_clip,
        )

        model.train()
        for epoch in range(n_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = TF.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()

        # Unwrap Opacus model to get parameters
        unwrapped = model._module
        trained_flat = weights_to_flat(
            [p.data.detach().cpu() for p in unwrapped.parameters()]
        )
        init_flat = weights_to_flat([global_state[k] for k in self._param_keys])
        param_delta = trained_flat - init_flat

        buf_state = {
            k: v.detach().cpu().clone()
            for k, v in unwrapped.state_dict().items()
            if k not in self._param_keys_set
        }

        return param_delta, buf_state

    def _do_train_client(
        self, client_id: int, global_state: dict, n_epochs: int,
        lr: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Dispatch to appropriate training method based on DP level."""
        cfg = self.cfg
        if cfg.dp_level == "record" and cfg.use_dp:
            return self._train_client_dpsgd(client_id, global_state, n_epochs, lr=lr)

        delta, buf = self._train_client(client_id, global_state, n_epochs, lr=lr)

        # Client-level Local DP: clip Δ to fixed C on client, then add Gaussian.
        # Fixed C (not adaptive) is required — adaptive would depend on peers.
        if (cfg.dp_mode == "local" and cfg.use_dp
                and cfg.dp_level == "client" and self._sigma_local > 0):
            C = cfg.client_clip
            norm = delta.norm(p=2).item()
            if norm > C:
                delta = delta * (C / norm)
            delta = delta + torch.randn_like(delta) * (self._sigma_local * C)

        return delta, buf

    # -----------------------------------------------------------------
    # Aggregation + DP noise
    # -----------------------------------------------------------------

    def _aggregate_and_apply(
        self,
        updates: list[torch.Tensor],
        buf_states: list[dict],
        global_state: dict,
        client_ids: list[int] | None = None,
    ) -> tuple[dict, list[int] | None, float]:
        """Pipeline: clip → aggregate → DP noise → apply to global state."""
        cfg = self.cfg

        # 1. Norm clipping (per-client, on server)
        clip_S = cfg.client_clip
        if cfg.adaptive_client_clip:
            clipped, clip_S = adaptive_clip(updates, clip_max=cfg.clip_max)
        else:
            clipped, clip_S = adaptive_clip(updates, S=cfg.client_clip, clip_max=cfg.clip_max)

        # 2. Aggregation
        if cfg.aggregation == "fedrola":
            result = self._fedrola.detect(clipped, client_ids)
        else:
            agg_cfg = {"m_assumed": cfg.m_assumed, "f": cfg.f, "trim_beta": cfg.trim_beta}
            result = aggregate(clipped, cfg.aggregation, agg_cfg)

        agg_update = result.update
        selected = result.selected

        # 3. DP noise — Central DP only.
        # Local DP noise already added on client (record: Opacus per-step;
        # client-level: Gaussian on Δ in _do_train_client).
        if cfg.sigma and cfg.sigma > 0 and cfg.dp_mode != "local":
            if cfg.dp_level == "record":
                sensitivity = compute_sensitivity_record(
                    cfg.record_clip,
                    n_active=len(updates),
                    samples_per_client=self._min_samples,
                )
            else:
                # Client-level: sensitivity = C / f_eff
                if cfg.aggregation == "fedrola" and self._fedrola is not None:
                    f_eff = self._fedrola.last_f_eff
                elif cfg.aggregation in ("multi_krum", "krum"):
                    f_eff = cfg.f
                elif selected is not None:
                    f_eff = len(selected)
                else:
                    f_eff = len(updates)
                sensitivity = compute_sensitivity_client(clip_S, f_eff)

            agg_update = add_dp_noise(agg_update, cfg.sigma, sensitivity)

        # 4. Apply delta to parameters
        new_state = dict(global_state)
        offset = 0
        for key in self._param_keys:
            size = global_state[key].numel()
            shape = global_state[key].shape
            old = global_state[key].flatten()
            new_state[key] = (old + agg_update[offset:offset + size]).reshape(shape).clone()
            offset += size

        # 5. FedAvg BN buffers across ALL clients
        if buf_states:
            buf_keys = list(buf_states[0].keys())
            for bk in buf_keys:
                stacked = torch.stack([bs[bk].float() for bs in buf_states])
                new_state[bk] = stacked.mean(dim=0).to(global_state[bk].dtype)

        return new_state, selected, clip_S

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------

    def run(self) -> RoundMetrics:
        cfg = self.cfg
        q = cfg.clients_per_round / cfg.n_clients
        logger.info(
            f"FL start: T={cfg.n_rounds}, N={cfg.n_clients}, "
            f"active/round={cfg.clients_per_round} (q={q:.2f}), "
            f"attack={cfg.attack}, agg={cfg.aggregation}, "
            f"dp_level={cfg.dp_level}, dp_mode={cfg.dp_mode}, "
            f"ε={cfg.epsilon if cfg.use_dp else '∞'}, σ={cfg.sigma:.4f}"
            f"{f', σ_local={self._sigma_local:.4f}' if cfg.dp_mode == 'local' else ''}"
            f"{f', lr_end={cfg.lr_end}' if cfg.lr_end is not None else ''}"
        )

        global_state = self._global_state
        rng = torch.Generator()
        rng.manual_seed(cfg.seed)

        for t in range(cfg.n_rounds):
            # --- LR schedule ---
            if cfg.lr_end is not None:
                current_lr = cfg.lr + (cfg.lr_end - cfg.lr) * t / max(cfg.n_rounds - 1, 1)
            else:
                current_lr = None  # use default cfg.lr

            # --- Subsample clients ---
            perm = torch.randperm(cfg.n_clients, generator=rng).tolist()
            active_clients = sorted(perm[:cfg.clients_per_round])
            active_malicious = [k for k in active_clients if k in self.malicious_set]
            active_honest = [k for k in active_clients if k not in self.malicious_set]

            n_active = len(active_clients)
            n_honest_active = len(active_honest)
            # Positional indices: 0..n_honest-1 = honest, n_honest..n_active-1 = malicious
            active_malicious_pos = set(range(n_honest_active, n_active))

            updates: list[torch.Tensor] = []
            buf_states: list[dict | None] = []
            honest_updates: list[torch.Tensor] = []
            # Real client IDs (for FedRoLA reputation tracking)
            client_ids: list[int] = []

            # --- Honest clients ---
            for k in active_honest:
                delta, buf = self._do_train_client(k, global_state, cfg.local_epochs, lr=current_lr)
                updates.append(delta)
                buf_states.append(buf)
                honest_updates.append(delta)
                client_ids.append(k)

            # --- Malicious clients ---
            for k in active_malicious:
                attack = cfg.attack
                buf = None

                if attack in ("label_flip", "random_label_flip", "none"):
                    delta, buf = self._do_train_client(k, global_state, cfg.local_epochs, lr=current_lr)

                elif attack == "alie":
                    base_delta = alie_attack(
                        honest_updates, n=n_active, m=len(active_malicious)
                    )
                    perturb_std = 0.01 * base_delta.norm().item()
                    delta = base_delta + torch.randn_like(base_delta) * perturb_std

                elif attack == "ipm":
                    delta = ipm_attack(honest_updates, epsilon=cfg.ipm_epsilon)

                elif attack == "fang":
                    delta = fang_attack(
                        honest_updates, n_malicious=len(active_malicious),
                        m_assumed=cfg.m_assumed,
                    )

                elif attack == "minmax":
                    delta = minmax_attack(
                        honest_updates, n_malicious=len(active_malicious),
                    )

                elif attack == "backdoor":
                    delta, buf = self._do_train_client(k, global_state, cfg.backdoor_epochs, lr=current_lr)
                    delta = backdoor_model_replacement(delta, cfg.backdoor_gamma, n_active)

                else:
                    delta, buf = self._do_train_client(k, global_state, cfg.local_epochs, lr=current_lr)

                updates.append(delta)
                buf_states.append(buf)
                client_ids.append(k)

            buf_states_list = [b for b in buf_states if b is not None]

            # --- Aggregate ---
            global_state, selected, S = self._aggregate_and_apply(
                updates, buf_states_list, global_state, client_ids=client_ids,
            )

            # --- Metrics ---
            if (t + 1) % cfg.log_every == 0 or t == cfg.n_rounds - 1:
                self.model.load_state_dict(
                    {k: v.to(self.device) for k, v in global_state.items()}
                )
                mta = compute_mta(self.model, self.test_loader, self.device)
                asr = 0.0
                source_acc = 0.0
                if cfg.attack == "backdoor":
                    asr = compute_asr(
                        self.model, self.test_loader,
                        self._trigger_fn, cfg.backdoor_target_label, self.device,
                    )
                elif cfg.attack == "label_flip":
                    # For targeted label flip, ASR := P(pred=target | y=source),
                    # source_acc := P(pred=source | y=source).
                    source_acc, asr = compute_label_flip_metrics(
                        self.model, self.test_loader,
                        cfg.flip_source, cfg.flip_target, self.device,
                    )

                tpr, fpr = 0.0, 0.0
                if selected is not None:
                    tpr, fpr = compute_tpr_fpr(selected, active_malicious_pos, n_active)

                # FedRoLA diagnostic fields (0/empty when not using FedRoLA)
                frola_fields = {}
                if self._fedrola is not None:
                    frola_fields = {
                        "frola_n_detected": self._fedrola.last_n_detected,
                        "frola_f_eff": round(self._fedrola.last_f_eff, 4),
                        "frola_score_min": round(self._fedrola.last_score_min, 4),
                        "frola_score_max": round(self._fedrola.last_score_max, 4),
                        "frola_score_mean": round(self._fedrola.last_score_mean, 4),
                        "frola_disc": round(self._fedrola.last_disc, 4),
                        "frola_layers": ",".join(str(x) for x in self._fedrola.last_selected_layers),
                    }

                self.metrics.log(
                    round_idx=t + 1, seed=cfg.seed,
                    mta=mta, asr=asr, source_acc=source_acc,
                    tpr=tpr, fpr=fpr, clip_S=float(S),
                    **frola_fields,
                )
                logger.info(
                    f"Round {t+1:3d}/{cfg.n_rounds} | "
                    f"MTA={mta:.4f} | ASR={asr:.4f} | SrcAcc={source_acc:.4f} | "
                    f"TPR={tpr:.3f} | FPR={fpr:.3f} | "
                    f"mal/round={len(active_malicious)}"
                )

        return self.metrics
