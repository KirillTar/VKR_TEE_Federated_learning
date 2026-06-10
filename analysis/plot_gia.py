#!/usr/bin/env python3
"""Visualize GIA reconstructions: grid of (original, no_dp, dp_low, dp_high).

Usage:
    python analysis/plot_gia.py \
        --in_dir results/local_dp/gia \
        --out results/local_dp/gia/gia_grid.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def chw_to_hwc(img: np.ndarray) -> np.ndarray:
    """(C,H,W) -> (H,W,C) for matplotlib; grayscale (1,H,W) -> (H,W)."""
    if img.ndim == 3 and img.shape[0] in (1, 3):
        if img.shape[0] == 1:
            return img[0]
        return np.transpose(img, (1, 2, 0))
    return img


def load_image(path: Path, key: str) -> np.ndarray:
    return chw_to_hwc(np.load(path)[key])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_images", type=int, default=5)
    p.add_argument("--noise_levels", nargs="+",
                   default=["no_dp", "dp_low", "dp_high"])
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    cols = ["original"] + args.noise_levels
    n_rows = args.n_images

    fig, axes = plt.subplots(n_rows, len(cols),
                             figsize=(2.2 * len(cols), 2.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = {
        "original": "Оригинал",
        "no_dp": "DLG без DP",
        "dp_low": "DLG, σ=0.1",
        "dp_high": "DLG, σ=0.5",
    }

    for i in range(n_rows):
        npz_no_dp = in_dir / f"img{i}_no_dp.npz"
        data_no_dp = np.load(npz_no_dp)
        axes[i, 0].imshow(chw_to_hwc(data_no_dp["original"]))
        axes[i, 0].set_ylabel(f"true={int(data_no_dp['true_label'])}",
                              fontsize=9)

        for j, noise in enumerate(args.noise_levels, start=1):
            npz = in_dir / f"img{i}_{noise}.npz"
            d = np.load(npz)
            axes[i, j].imshow(chw_to_hwc(d["reconstructed"]))
            axes[i, j].set_title(
                f"pred={int(d['pred_label'])}" if i == 0
                else f"pred={int(d['pred_label'])}",
                fontsize=8,
            )

        for j in range(len(cols)):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    for j, c in enumerate(cols):
        axes[0, j].set_title(f"{col_titles.get(c, c)}", fontsize=10)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
