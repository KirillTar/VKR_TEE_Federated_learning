#!/usr/bin/env python3
"""SNR vs MTA scatter from existing S-curve data.

SNR formula (from CLAUDE.md): per-round SNR = n_averaged / (σ · √d)
where σ is DP noise multiplier (Opacus RDP), n_averaged is f_eff clients,
d is model dim. Threshold for learning: SNR > 0.19 (empirical).

Reads no-attack clean runs from both scenarios, computes σ via Opacus,
assembles a single figure.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.dp import compute_sigma


def load_seeds(dir_, pattern_glob):
    """Load mta_mean across seed files matching pattern. Return list of floats."""
    vals = []
    for f in sorted(Path(dir_).glob(pattern_glob)):
        with open(f) as fh:
            d = json.load(fh)
        if "mta_mean" in d and "mta_std" not in d:
            vals.append(d["mta_mean"])
        elif "mta_mean" in d:
            # seed-level summary: single mean per seed (std=0)
            vals.append(d["mta_mean"])
    return vals


def collect_epsilons(dir_, prefix, epsilons):
    """For each ε, collect seed-level MTA → mean, std."""
    points = []
    for eps in epsilons:
        eps_tag = str(eps).rstrip('0').rstrip('.') if '.' in str(eps) else str(eps)
        tried = [
            f"{prefix}{eps}_none_seed*_summary.json",
            f"{prefix}{eps_tag}_none_seed*_summary.json",
            f"{prefix}{float(eps):g}_none_seed*_summary.json",
        ]
        seen = set()
        mtas = []
        for pat in tried:
            for f in sorted(Path(dir_).glob(pat)):
                if f in seen:
                    continue
                seen.add(f)
                with open(f) as fh:
                    d = json.load(fh)
                mtas.append(d["mta_mean"])
        if not mtas:
            print(f"  no data for ε={eps} (pat={prefix}*)")
            continue
        points.append({
            "epsilon": eps,
            "mta_mean": float(np.mean(mtas)),
            "mta_std": float(np.std(mtas)) if len(mtas) > 1 else 0.0,
            "n_seeds": len(mtas),
        })
    return points


def compute_snr_client(eps, T, q, d, f_eff=28, C=1.0):
    """Client-level DP SNR. Noise per-coord std = σ·C/f_eff.
    Signal per-coord ~ C/√d (assuming unit-norm averaged Δ). SNR = f_eff/(σ·√d)."""
    sigma = compute_sigma(eps, 1e-5, T=T, sample_rate=q)
    return f_eff / (sigma * np.sqrt(d)), sigma


def compute_snr_record(eps, T, q, d, n_active=40, samples_per_client=1400, R=5.0, f_eff=28):
    """Record-level DP SNR. Sensitivity = R/(n·|D|), much smaller → higher SNR.
    Noise per-coord std = σ·R/(n·|D|). Signal per-coord ~ C/√d with C_effective≈f_eff·R/(n·|D|)·T?
    Using the CLAUDE.md formula: SNR = n_averaged / (σ·√d), but adjusted for record sensitivity:
    Actually noise std scaling differs; we use the same normalized form but with record-level σ."""
    sigma = compute_sigma(eps, 1e-5, T=T, sample_rate=q)
    # Record-level sensitivity is R/(n·|D|) instead of C/f_eff.
    # Normalize to same formula: SNR ∝ 1 / (σ_effective · √d), where
    # σ_effective = σ · sensitivity / (typical signal magnitude per coord).
    # Assume signal per-coord ~ 1/√d → SNR = n_active · samples / (σ · R · √d)
    return (n_active * samples_per_client) / (sigma * R * np.sqrt(d)), sigma


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/snr_vs_mta.png")
    args = p.parse_args()

    # CD: client-level, d=4266, T=200, q=0.2, f_eff≈28
    # CD data: cd_diag_eps{2,3,4,5,6,8} + cd_exp2_eps{1,10,15,20}
    cd_epsilons = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    print("CD (client-level, d=4266):")
    cd_points = []
    # diag covers 2-8; exp2 covers 1,10,15,20
    cd_points += collect_epsilons("results/cross_device", "cd_diag_eps", [2, 3, 4, 5, 6, 8])
    cd_points += collect_epsilons("results/cross_device", "cd_exp2_eps", [1, 10, 15, 20])

    for pt in cd_points:
        snr, sigma = compute_snr_client(pt["epsilon"], T=200, q=0.2, d=4266)
        pt["snr"] = snr
        pt["sigma"] = sigma
    cd_points.sort(key=lambda p: p["epsilon"])
    for pt in cd_points:
        print(f"  ε={pt['epsilon']:>4} σ={pt['sigma']:.3f} SNR={pt['snr']:.3f} "
              f"MTA={pt['mta_mean']:.3f}±{pt['mta_std']:.3f} n={pt['n_seeds']}")

    # CS: record-level, d=89834, T=200, q=0.8
    cs_epsilons = [0.5, 0.75, 1, 1.5, 2, 3, 4, 8]
    print("\nCS (record-level, d=89834):")
    cs_points = []
    cs_points += collect_epsilons("results/cross_silo", "cs_p3_eps", [0.5, 0.75, 1, 1.5, 3])
    cs_points += collect_epsilons("results/cross_silo", "cs_exp2_eps", [2, 4, 8])

    for pt in cs_points:
        snr, sigma = compute_snr_record(pt["epsilon"], T=200, q=0.8, d=89834)
        pt["snr"] = snr
        pt["sigma"] = sigma
    cs_points.sort(key=lambda p: p["epsilon"])
    for pt in cs_points:
        print(f"  ε={pt['epsilon']:>5} σ={pt['sigma']:.3f} SNR={pt['snr']:.3f} "
              f"MTA={pt['mta_mean']:.3f}±{pt['mta_std']:.3f} n={pt['n_seeds']}")

    fig, ax = plt.subplots(figsize=(7, 5))

    cd_snr = [p["snr"] for p in cd_points]
    cd_mta = [p["mta_mean"] for p in cd_points]
    cd_err = [p["mta_std"] for p in cd_points]
    ax.errorbar(cd_snr, cd_mta, yerr=cd_err, fmt="o-", label="CD (client-level, d=4266)",
                capsize=3, color="C0")

    cs_snr = [p["snr"] for p in cs_points]
    cs_mta = [p["mta_mean"] for p in cs_points]
    cs_err = [p["mta_std"] for p in cs_points]
    ax.errorbar(cs_snr, cs_mta, yerr=cs_err, fmt="s-", label="CS (record-level, d=89834)",
                capsize=3, color="C1")

    for p_ in cd_points:
        ax.annotate(f"ε={p_['epsilon']}", (p_["snr"], p_["mta_mean"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7, color="C0")
    for p_ in cs_points:
        ax.annotate(f"ε={p_['epsilon']}", (p_["snr"], p_["mta_mean"]),
                    xytext=(4, -10), textcoords="offset points", fontsize=7, color="C1")

    ax.axvline(0.19, color="red", linestyle="--", alpha=0.5, label="SNR=0.19 threshold")
    ax.axhline(0.1, color="gray", linestyle=":", alpha=0.5, label="chance (MTA=0.1)")

    ax.set_xscale("log")
    ax.set_xlabel("Per-round SNR (log scale)")
    ax.set_ylabel("MTA")
    ax.set_title("Тезис 6: применимость DP-FL определяется per-round SNR")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")

    # Also dump data as JSON for further analysis
    data_out = out.with_suffix(".json")
    with open(data_out, "w") as fh:
        json.dump({"cd": cd_points, "cs": cs_points}, fh, indent=2)
    print(f"Saved {data_out}")


if __name__ == "__main__":
    main()
