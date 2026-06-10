#!/usr/bin/env python3
"""
Gradient Inversion Attack (GIA) demonstration.

Implements DLG (Deep Leakage from Gradients, Zhu et al. 2019) on a simplified
scenario: single client, batch_size=1. Shows that gradients leak training data,
and DP noise prevents reconstruction.

This is a worst-case demonstration for the thesis: in real FL (N=50, batch_size=32),
GIA is much harder. TEE prevents GIA by hiding individual updates.

Usage:
    python experiments/gia.py --dataset svhn --model svhn_cnn \
        --n_images 5 --save_dir results/gia
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import get_model
from core.data import load_dataset, SVHN_MEAN, SVHN_STD, FMNIST_MEAN, FMNIST_STD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gia")

# Dataset normalization stats for denormalization
DENORM_STATS = {
    "svhn": (SVHN_MEAN, SVHN_STD),
    "fashion_mnist": (FMNIST_MEAN, FMNIST_STD),
}


def denormalize(img_tensor, dataset):
    """Convert normalized tensor back to [0, 1] range for visualization."""
    mean, std = DENORM_STATS.get(dataset, ([0.0], [1.0]))
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1)


def compute_gradient(model, x, y, device):
    """Compute gradient of loss w.r.t. model parameters for a single (x, y) pair."""
    model.zero_grad()
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.long)
    x = x.to(device)
    y = y.to(device)
    logits = model(x.unsqueeze(0))
    loss = F.cross_entropy(logits, y.unsqueeze(0))
    grads = torch.autograd.grad(loss, model.parameters())
    return [g.detach().clone() for g in grads]


def add_noise_to_gradients(grads, noise_std):
    """Add Gaussian noise to gradients (simulates DP)."""
    return [g + torch.randn_like(g) * noise_std for g in grads]


def dlg_attack(model, true_grads, img_shape, n_classes, device,
               n_steps=1000, lr=0.1):
    """
    DLG: reconstruct input from gradients.

    Optimizes dummy_x, dummy_y to match the observed gradients.
    Returns (reconstructed_image, reconstructed_label, loss_history).
    """
    dummy_x = torch.randn(1, *img_shape, device=device, requires_grad=True)
    dummy_label = torch.randn(1, n_classes, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([dummy_x, dummy_label], lr=lr)
    loss_history = []

    for step in range(n_steps):
        def closure():
            optimizer.zero_grad()
            model.zero_grad()
            logits = model(dummy_x)
            dummy_y = F.softmax(dummy_label, dim=1)
            loss = F.cross_entropy(logits, dummy_y)
            dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            grad_diff = sum(
                ((dg - tg) ** 2).sum()
                for dg, tg in zip(dummy_grads, true_grads)
            )
            grad_diff.backward()
            return grad_diff

        loss = optimizer.step(closure)
        loss_history.append(float(loss))

        if step % 200 == 0:
            logger.info(f"    DLG step {step}/{n_steps}, grad_diff={float(loss):.6f}")

    pred_label = int(dummy_label.argmax(dim=1).item())
    return dummy_x.detach().cpu().squeeze(0), pred_label, loss_history


def run_gia(dataset, model_name, n_images, save_dir, noise_levels, seed=42):
    """Run GIA experiment: reconstruct images with/without DP noise."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_ds, _ = load_dataset(dataset, "./data")
    model = get_model(dataset, model_name).to(device)
    model.eval()

    img_shape = train_ds[0][0].shape
    n_classes = 10
    logger.info(f"Dataset: {dataset}, model: {model_name}, img_shape={img_shape}")

    # Select random images
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_ds), size=n_images, replace=False)

    results = {}

    for i, idx in enumerate(indices):
        x, y = train_ds[idx]
        true_label = int(y)
        logger.info(f"Image {i+1}/{n_images}: idx={idx}, true_label={true_label}")

        # Compute true gradient (what server would see from 1 client, batch_size=1)
        true_grads = compute_gradient(model, x, y, device)

        for noise_name, noise_std in noise_levels.items():
            logger.info(f"  Noise: {noise_name} (std={noise_std})")

            if noise_std > 0:
                noisy_grads = add_noise_to_gradients(true_grads, noise_std)
            else:
                noisy_grads = true_grads

            # Run DLG reconstruction
            recon, pred_label, loss_hist = dlg_attack(
                model, noisy_grads, img_shape, n_classes, device,
                n_steps=600, lr=0.1,
            )

            # Save results
            key = f"img{i}_{noise_name}"
            results[key] = {
                "true_label": true_label,
                "pred_label": pred_label,
                "final_loss": loss_hist[-1],
                "label_correct": true_label == pred_label,
            }

            # Save images
            orig_denorm = denormalize(x, dataset)
            recon_denorm = denormalize(recon, dataset)

            np.savez(
                save_dir / f"{key}.npz",
                original=orig_denorm.numpy(),
                reconstructed=recon_denorm.numpy(),
                true_label=true_label,
                pred_label=pred_label,
                loss_history=np.array(loss_hist),
            )

    # Save summary
    import json
    summary = {
        "dataset": dataset,
        "model": model_name,
        "n_images": n_images,
        "noise_levels": {k: v for k, v in noise_levels.items()},
        "results": results,
    }

    # Aggregate stats per noise level
    for noise_name in noise_levels:
        keys = [k for k in results if k.endswith(f"_{noise_name}")]
        losses = [results[k]["final_loss"] for k in keys]
        correct = [results[k]["label_correct"] for k in keys]
        summary[f"{noise_name}_mean_loss"] = float(np.mean(losses))
        summary[f"{noise_name}_label_acc"] = float(np.mean(correct))
        logger.info(
            f"  {noise_name}: mean_grad_diff={np.mean(losses):.6f}, "
            f"label_acc={np.mean(correct):.2f}"
        )

    with open(save_dir / "gia_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {save_dir / 'gia_summary.json'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="GIA experiment (DLG)")
    parser.add_argument("--dataset", default="svhn", choices=["svhn", "fashion_mnist"])
    parser.add_argument("--model", default="svhn_cnn")
    parser.add_argument("--n_images", type=int, default=5)
    parser.add_argument("--save_dir", default="results/gia")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_std_dp", type=float, default=0.1,
                        help="Noise std simulating DP (default: 0.1)")
    args = parser.parse_args()

    # Noise levels to compare
    noise_levels = {
        "no_dp": 0.0,
        "dp_low": args.noise_std_dp,
        "dp_high": args.noise_std_dp * 5,
    }

    run_gia(args.dataset, args.model, args.n_images, args.save_dir,
            noise_levels, args.seed)


if __name__ == "__main__":
    main()
