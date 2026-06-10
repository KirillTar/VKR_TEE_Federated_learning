#!/usr/bin/env python3
"""Build figures for section 5.6 (TEE overhead).

Reads JSONL measurements from scone/measurements/server_e2386g_long/, renders
fig_5_13_overhead.png and fig_5_14_jitter.png into docs/images/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scone" / "measurements" / "server_e2386g_long"
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.4,
    "figure.dpi": 150,
})

AGGS = ["fedavg", "multi_krum", "krum", "trimmed_mean", "median", "fedrola"]
AGG_LABELS = {
    "fedavg":       "FedAvg",
    "multi_krum":   "Multi-Krum",
    "krum":         "Krum",
    "trimmed_mean": "TrimmedMean",
    "median":       "Median",
    "fedrola":      "FedRoLA",
}


def load(name: str) -> np.ndarray:
    """Return per-round dt_ms (excluding warmup)."""
    rounds = []
    f = SRC / f"{name}.jsonl"
    for line in f.open():
        d = json.loads(line)
        if d.get("event") == "round" and not d.get("warmup"):
            rounds.append(d["dt_ms"])
    return np.array(rounds)


def stats(arr: np.ndarray) -> dict:
    return {
        "median": float(np.median(arr)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
        "max":    float(arr.max()),
        "std":    float(arr.std()),
    }


def fig_overhead():
    """Bar chart: median per aggregation, native vs HW, two subplots (CS, CD)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    x = np.arange(len(AGGS))
    width = 0.38

    for ax, sc, title in [
        (axes[0], "cs", "Межорганизационный сценарий ($d=89834$)"),
        (axes[1], "cd", "Межустройственный сценарий ($d=4266$)"),
    ]:
        nat = [stats(load(f"native_{sc}_{a}"))["median"] for a in AGGS]
        hw  = [stats(load(f"hw_{sc}_{a}"))["median"] for a in AGGS]
        b1 = ax.bar(x - width/2, nat, width, label="native (без TEE)",  color="#7fb069")
        b2 = ax.bar(x + width/2, hw,  width, label="SCONE HW (Intel SGX)", color="#c1666b")

        # Annotate HW bars with overhead ratio
        for i, (n, h) in enumerate(zip(nat, hw)):
            ax.text(i + width/2, h, f"×{h/n:.0f}", ha="center", va="bottom",
                    fontsize=8, color="#7a3033")
            ax.text(i - width/2, n, f"{n:.1f}", ha="center", va="bottom",
                    fontsize=8, color="#3a5a26")

        ax.set_xticks(x)
        ax.set_xticklabels([AGG_LABELS[a] for a in AGGS], rotation=20, ha="right")
        ax.set_ylabel("Медиана $dt$ раунда, мс")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.set_ylim(0.6, 400)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = OUT / "fig_5_13_overhead.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def fig_jitter():
    """Per-round timeline + histogram for FedRoLA and Multi-Krum under HW."""
    cases = [
        ("hw_cs_multi_krum", "Multi-Krum, межорг. (HW)"),
        ("hw_cs_fedrola",    "FedRoLA,    межорг. (HW)"),
        ("hw_cd_multi_krum", "Multi-Krum, межустр. (HW)"),
        ("hw_cd_fedrola",    "FedRoLA,    межустр. (HW)"),
    ]
    fig, axes = plt.subplots(len(cases), 2, figsize=(11, 2.4 * len(cases)),
                             gridspec_kw={"width_ratios": [3, 1.2]})

    for row, (name, title) in enumerate(cases):
        arr = load(name)
        s = stats(arr)

        # Timeline (zoomed first 1500 rounds, otherwise too dense)
        ax = axes[row, 0]
        n_show = min(1500, len(arr))
        ax.plot(arr[:n_show], lw=0.4, color="#2b6cb0", alpha=0.75)
        ax.axhline(s["median"], color="#2f855a", lw=1, ls="--",
                   label=f"медиана {s['median']:.1f}")
        ax.axhline(s["p99"],    color="#dd6b20", lw=1, ls="--",
                   label=f"p99 {s['p99']:.1f}")
        ax.axhline(s["max"],    color="#c53030", lw=1, ls=":",
                   label=f"max {s['max']:.1f}")
        ax.set_xlabel("номер раунда (первые 1500)")
        ax.set_ylabel("$dt$, мс")
        ax.set_title(f"{title} — последовательность")
        ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.grid(alpha=0.3)

        # Histogram (log y)
        ax = axes[row, 1]
        ax.hist(arr, bins=80, color="#4a5568", alpha=0.85, edgecolor="none")
        ax.axvline(s["median"], color="#2f855a", lw=1, ls="--")
        ax.axvline(s["p99"],    color="#dd6b20", lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("$dt$, мс")
        ax.set_ylabel("число раундов (log)")
        ax.set_title("распределение")
        ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = OUT / "fig_5_14_jitter.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    fig_overhead()
    fig_jitter()
