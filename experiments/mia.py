#!/usr/bin/env python3
"""
Membership Inference Attack (MIA) experiment.

Trains FL models (FedAvg, no attacks) with/without DP, then evaluates
loss-based MIA (Yeom et al. 2018) to demonstrate privacy leakage.

The attack exploits the fact that models tend to have lower loss on
training samples (members) than on unseen samples (non-members).
AUC > 0.5 indicates privacy leakage; AUC ≈ 0.5 means DP is effective.

Usage:
    python experiments/mia.py --config experiments/configs/cross_device/exp1_base.yaml \
        --seeds 0 1 2 --save_dir results/mia
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.fl_simulator import FLConfig, FLSimulator
from core.models import get_model, count_parameters
from core.data import load_dataset, get_test_dataloader
from core.metrics import compute_mta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mia")


def compute_per_sample_loss_and_labels(model, loader, device):
    """Compute per-sample cross-entropy loss and true labels."""
    model.eval()
    losses = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="none")
            losses.append(loss.cpu())
            all_labels.append(y.cpu())
    return torch.cat(losses).numpy(), torch.cat(all_labels).numpy()


def compute_per_sample_confidence_and_labels(model, loader, device):
    """Compute per-sample max softmax confidence and true labels."""
    model.eval()
    confs = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = F.softmax(model(x), dim=1)
            conf = probs.gather(1, y.unsqueeze(1)).squeeze(1)
            confs.append(conf.cpu())
            all_labels.append(y.cpu())
    return torch.cat(confs).numpy(), torch.cat(all_labels).numpy()


def per_class_threshold_auc(member_scores, member_labels, nonmember_scores, nonmember_labels):
    """
    Per-class threshold MIA (stronger than global threshold).

    For each class, compute a separate optimal threshold. Then aggregate
    per-class decisions into a single AUC. This is stronger than Yeom's
    global threshold because it accounts for class-specific loss distributions.
    """
    from sklearn.metrics import roc_auc_score

    all_classes = np.unique(np.concatenate([member_labels, nonmember_labels]))
    per_sample_scores = []
    per_sample_labels = []

    for cls in all_classes:
        m_mask = member_labels == cls
        nm_mask = nonmember_labels == cls
        if m_mask.sum() == 0 or nm_mask.sum() == 0:
            continue

        # Per-class: normalize scores by class mean/std for better separation
        cls_scores = np.concatenate([member_scores[m_mask], nonmember_scores[nm_mask]])
        cls_mean = cls_scores.mean()
        cls_std = cls_scores.std() + 1e-8
        normed = (cls_scores - cls_mean) / cls_std

        cls_labels = np.concatenate([np.ones(m_mask.sum()), np.zeros(nm_mask.sum())])
        per_sample_scores.append(normed)
        per_sample_labels.append(cls_labels)

    if not per_sample_scores:
        return 0.5

    all_scores = np.concatenate(per_sample_scores)
    all_labels = np.concatenate(per_sample_labels)
    return float(roc_auc_score(all_labels, all_scores))


def mia_evaluate(model, train_ds, test_ds, device, n_member_samples=10000):
    """
    Loss-based MIA evaluation.

    Args:
        model: trained FL model
        train_ds: full training dataset (members)
        test_ds: test dataset (non-members)
        n_member_samples: subsample members to match non-member count

    Returns:
        dict with AUC (loss-based and confidence-based), and ROC curves
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    # Subsample members to roughly match test set size for balanced evaluation
    n_test = len(test_ds)
    n_use = min(n_member_samples, len(train_ds), n_test)

    rng = np.random.default_rng(42)
    member_idx = rng.choice(len(train_ds), size=n_use, replace=False).tolist()
    nonmember_idx = rng.choice(n_test, size=n_use, replace=False).tolist()

    member_loader = DataLoader(
        Subset(train_ds, member_idx), batch_size=256, shuffle=False, num_workers=2,
    )
    nonmember_loader = DataLoader(
        Subset(test_ds, nonmember_idx), batch_size=256, shuffle=False, num_workers=2,
    )

    # Loss-based MIA: members have LOWER loss
    member_loss, member_loss_labels = compute_per_sample_loss_and_labels(model, member_loader, device)
    nonmember_loss, nonmember_loss_labels = compute_per_sample_loss_and_labels(model, nonmember_loader, device)

    labels = np.concatenate([np.ones(len(member_loss)), np.zeros(len(nonmember_loss))])
    # Score = negative loss (higher = more likely member)
    scores_loss = np.concatenate([-member_loss, -nonmember_loss])
    auc_loss = roc_auc_score(labels, scores_loss)
    fpr_loss, tpr_loss, _ = roc_curve(labels, scores_loss)

    # Confidence-based MIA: members have HIGHER confidence on true class
    member_conf, member_conf_labels = compute_per_sample_confidence_and_labels(model, member_loader, device)
    nonmember_conf, nonmember_conf_labels = compute_per_sample_confidence_and_labels(model, nonmember_loader, device)
    scores_conf = np.concatenate([member_conf, nonmember_conf])
    auc_conf = roc_auc_score(labels, scores_conf)
    fpr_conf, tpr_conf, _ = roc_curve(labels, scores_conf)

    # Per-class threshold MIA (stronger attack)
    auc_loss_perclass = per_class_threshold_auc(
        -member_loss, member_loss_labels, -nonmember_loss, nonmember_loss_labels,
    )
    auc_conf_perclass = per_class_threshold_auc(
        member_conf, member_conf_labels, nonmember_conf, nonmember_conf_labels,
    )

    return {
        "auc_loss": float(auc_loss),
        "auc_conf": float(auc_conf),
        "auc_loss_perclass": float(auc_loss_perclass),
        "auc_conf_perclass": float(auc_conf_perclass),
        "member_loss_mean": float(member_loss.mean()),
        "nonmember_loss_mean": float(nonmember_loss.mean()),
        "member_conf_mean": float(member_conf.mean()),
        "nonmember_conf_mean": float(nonmember_conf.mean()),
        "roc_loss": (fpr_loss, tpr_loss),
        "roc_conf": (fpr_conf, tpr_conf),
    }


def train_centralized(dataset, model_name, data_dir, epochs, batch_size, lr,
                      momentum, weight_decay, seed):
    """Train a model centrally (no FL) on full training set. Returns (model, train_ds, test_ds, device)."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = load_dataset(dataset, data_dir)
    model = get_model(dataset, model_name).to(device)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                        pin_memory=(device.type == "cuda"))
    test_loader = get_test_dataloader(test_ds, num_workers=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        if ep % 10 == 0 or ep == epochs:
            mta = compute_mta(model, test_loader, device)
            logger.info(f"  Centralized epoch {ep}/{epochs}: MTA={mta:.4f}")

    return model, train_ds, test_ds, device


def run_mia_centralized(config_path, seeds, save_dir, epochs=200):
    """Train centralized model and evaluate MIA (upper bound for privacy leakage)."""
    import yaml

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    seed_results = []
    for seed in seeds:
        logger.info(f"=== MIA centralized, seed={seed} ===")
        model, train_ds, test_ds, device = train_centralized(
            dataset=base_cfg["dataset"],
            model_name=base_cfg.get("model_name"),
            data_dir=base_cfg.get("data_dir", "./data"),
            epochs=epochs,
            batch_size=base_cfg.get("batch_size", 32),
            lr=base_cfg.get("lr", 0.01),
            momentum=base_cfg.get("momentum", 0.9),
            weight_decay=base_cfg.get("weight_decay", 1e-4),
            seed=seed,
        )

        test_loader = get_test_dataloader(test_ds, num_workers=0)
        mta = compute_mta(model, test_loader, device)
        mia_result = mia_evaluate(model, train_ds, test_ds, device)

        result = {
            "mta": round(mta, 4),
            "auc_loss": round(mia_result["auc_loss"], 4),
            "auc_conf": round(mia_result["auc_conf"], 4),
            "auc_loss_perclass": round(mia_result["auc_loss_perclass"], 4),
            "auc_conf_perclass": round(mia_result["auc_conf_perclass"], 4),
            "member_loss_mean": round(mia_result["member_loss_mean"], 4),
            "nonmember_loss_mean": round(mia_result["nonmember_loss_mean"], 4),
            "member_conf_mean": round(mia_result["member_conf_mean"], 4),
            "nonmember_conf_mean": round(mia_result["nonmember_conf_mean"], 4),
            "seed": seed,
        }
        seed_results.append(result)

        np.savez(
            save_dir / f"mia_centralized_seed{seed}_roc.npz",
            fpr_loss=mia_result["roc_loss"][0],
            tpr_loss=mia_result["roc_loss"][1],
            fpr_conf=mia_result["roc_conf"][0],
            tpr_conf=mia_result["roc_conf"][1],
        )

        logger.info(
            f"  MTA={mta:.4f}, AUC_loss={mia_result['auc_loss']:.4f}, "
            f"AUC_conf={mia_result['auc_conf']:.4f}, "
            f"AUC_loss_pc={mia_result['auc_loss_perclass']:.4f}, "
            f"AUC_conf_pc={mia_result['auc_conf_perclass']:.4f}"
        )

    agg = {}
    for key in ["mta", "auc_loss", "auc_conf",
                 "auc_loss_perclass", "auc_conf_perclass",
                 "member_loss_mean", "nonmember_loss_mean",
                 "member_conf_mean", "nonmember_conf_mean"]:
        vals = [r[key] for r in seed_results]
        agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{key}_std"] = round(float(np.std(vals)), 4)
    agg["n_seeds"] = len(seeds)

    logger.info(f"  centralized summary: AUC_loss={agg['auc_loss_mean']:.4f}±{agg['auc_loss_std']:.4f}")
    return {"centralized": agg}


def run_mia(config_path, seeds, save_dir, epsilons=None, skip_no_dp=False):
    """Train FL models with/without DP and evaluate MIA."""
    import yaml

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    # MIA configurations: vary DP settings only
    if epsilons is None:
        epsilons = [10.0, 20.0]

    mia_configs = {}
    if not skip_no_dp:
        mia_configs["no_dp"] = {"use_dp": False, "epsilon": 999.0}
    for eps in epsilons:
        eps_tag = str(eps).replace(".", "p") if eps != int(eps) else str(int(eps))
        mia_configs[f"dp_eps{eps_tag}"] = {"use_dp": True, "epsilon": float(eps)}

    logger.info(f"MIA configs: {list(mia_configs.keys())}")

    all_results = {}

    for dp_name, dp_overrides in mia_configs.items():
        seed_results = []

        for seed in seeds:
            logger.info(f"=== MIA: {dp_name}, seed={seed} ===")

            # Build config: clean FedAvg + DP variant
            cfg_dict = {**base_cfg}
            cfg_dict.update({
                "aggregation": "fedavg",
                "attack": "none",
                "m_real": 0,
                "seed": seed,
                "name": f"mia_{dp_name}",
                "log_every": 50,
                "save_dir": str(save_dir),
            })
            cfg_dict.update(dp_overrides)

            cfg = FLConfig(**{
                k: v for k, v in cfg_dict.items()
                if k in FLConfig.__dataclass_fields__
            })
            cfg.save_dir = str(save_dir)

            # Train
            sim = FLSimulator(cfg)
            sim.run()

            # Get final model and evaluate MIA
            device = sim.device
            mta = compute_mta(sim.model, sim.test_loader, device)

            mia_result = mia_evaluate(
                sim.model, sim.train_ds, sim.test_ds, device,
            )

            result = {
                "mta": round(mta, 4),
                "auc_loss": round(mia_result["auc_loss"], 4),
                "auc_conf": round(mia_result["auc_conf"], 4),
                "auc_loss_perclass": round(mia_result["auc_loss_perclass"], 4),
                "auc_conf_perclass": round(mia_result["auc_conf_perclass"], 4),
                "member_loss_mean": round(mia_result["member_loss_mean"], 4),
                "nonmember_loss_mean": round(mia_result["nonmember_loss_mean"], 4),
                "member_conf_mean": round(mia_result["member_conf_mean"], 4),
                "nonmember_conf_mean": round(mia_result["nonmember_conf_mean"], 4),
                "seed": seed,
            }
            seed_results.append(result)

            # Save ROC curve data per seed
            np.savez(
                save_dir / f"mia_{dp_name}_seed{seed}_roc.npz",
                fpr_loss=mia_result["roc_loss"][0],
                tpr_loss=mia_result["roc_loss"][1],
                fpr_conf=mia_result["roc_conf"][0],
                tpr_conf=mia_result["roc_conf"][1],
            )

            logger.info(
                f"  MTA={mta:.4f}, AUC_loss={mia_result['auc_loss']:.4f}, "
                f"AUC_conf={mia_result['auc_conf']:.4f}, "
                f"AUC_loss_pc={mia_result['auc_loss_perclass']:.4f}, "
                f"AUC_conf_pc={mia_result['auc_conf_perclass']:.4f}"
            )

        # Aggregate across seeds
        agg = {}
        for key in ["mta", "auc_loss", "auc_conf",
                     "auc_loss_perclass", "auc_conf_perclass",
                     "member_loss_mean", "nonmember_loss_mean",
                     "member_conf_mean", "nonmember_conf_mean"]:
            vals = [r[key] for r in seed_results]
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{key}_std"] = round(float(np.std(vals)), 4)
        agg["n_seeds"] = len(seeds)
        all_results[dp_name] = agg

        logger.info(f"  {dp_name} summary: AUC_loss={agg['auc_loss_mean']:.4f}±{agg['auc_loss_std']:.4f}")

    # Save combined summary
    summary_path = save_dir / "mia_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"MIA summary saved to {summary_path}")

    # Print final table
    print("\n=== MIA Results ===")
    print(f"{'Config':<15} {'MTA':>8} {'AUC_loss':>10} {'AUC_conf':>10} {'AUC_l_pc':>10} {'AUC_c_pc':>10}")
    print("-" * 65)
    for name, r in all_results.items():
        print(
            f"{name:<15} {r['mta_mean']:>8.4f} {r['auc_loss_mean']:>10.4f} "
            f"{r['auc_conf_mean']:>10.4f} {r['auc_loss_perclass_mean']:>10.4f} "
            f"{r['auc_conf_perclass_mean']:>10.4f}"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(description="MIA experiment")
    parser.add_argument("--config", required=True, help="Base YAML config")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--save_dir", default="results/mia")
    parser.add_argument("--epsilons", nargs="+", type=float, default=None,
                        help="DP epsilon values to test (default: 10, 20)")
    parser.add_argument("--centralized", action="store_true",
                        help="Run centralized training baseline instead of FL")
    parser.add_argument("--centralized_epochs", type=int, default=200,
                        help="Number of epochs for centralized training")
    parser.add_argument("--no_dp_only", action="store_true",
                        help="Only run no-DP config (skip DP variants)")
    parser.add_argument("--skip_no_dp", action="store_true",
                        help="Skip no-DP config (only run DP variants)")
    args = parser.parse_args()

    if args.centralized:
        run_mia_centralized(args.config, args.seeds, args.save_dir, args.centralized_epochs)
    else:
        epsilons = [] if args.no_dp_only else args.epsilons
        run_mia(args.config, args.seeds, args.save_dir, epsilons,
                skip_no_dp=args.skip_no_dp)


if __name__ == "__main__":
    main()
