"""Visualization: convergence curves, heatmaps, trade-off plots."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


STYLE = {
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
}

ATTACK_LABELS = {
    "none": "No Attack",
    "label_flip": "Label Flip",
    "sign_flip": "Sign Flip",
    "alie": "ALIE",
    "backdoor": "Backdoor",
}

DEFENSE_LABELS = {
    "fedavg_nodp": "No Defense",
    "mkrum_nodp": "Multi-Krum",
    "fedavg_cdp4": "CDP (ε=4)",
    "mkrum_cdp4": "MKrum+CDP (ε=4)",
    "mkrum_cdp1": "MKrum+CDP (ε=1)",
}

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load_results(results_dir: str, pattern: str = "*.csv") -> pd.DataFrame:
    """Load all CSV result files from a directory into one DataFrame."""
    frames = []
    for path in Path(results_dir).glob(pattern):
        df = pd.read_csv(path)
        df["experiment"] = path.stem
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSVs found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def plot_convergence(
    df: pd.DataFrame,
    metric: str = "mta",
    group_by: str = "experiment",
    title: str = "Convergence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot mean ± std convergence curves across seeds."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, (name, group) in enumerate(df.groupby(group_by)):
            agg = group.groupby("round")[metric].agg(["mean", "std"])
            rounds = agg.index.values
            mean = agg["mean"].values
            std = agg["std"].values

            label = DEFENSE_LABELS.get(name, ATTACK_LABELS.get(name, str(name)))
            color = COLORS[i % len(COLORS)]
            ax.plot(rounds, mean, label=label, color=color)
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_xlabel("Communication Round")
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.legend(loc="lower right")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved: {save_path}")
    return fig


def plot_epsilon_tradeoff(
    df: pd.DataFrame,
    metric: str = "mta",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Privacy-accuracy trade-off: MTA (or ASR) vs ε."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        for attack, group in df.groupby("attack"):
            agg = group.groupby("epsilon")[metric].agg(["mean", "std"])
            eps = agg.index.astype(float)
            ax.errorbar(
                eps, agg["mean"], yerr=agg["std"],
                marker="o", label=ATTACK_LABELS.get(attack, attack), capsize=4,
            )

        ax.set_xlabel("Privacy Budget ε")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Privacy-Accuracy Trade-off ({metric.upper()} vs ε)")
        ax.set_xscale("log")
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_tpr_fpr_by_attack(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart: TPR and FPR per attack type."""
    with plt.rc_context(STYLE):
        last = df[df["round"] == df["round"].max()]
        attacks = last["attack"].unique()
        x = np.arange(len(attacks))
        width = 0.35

        tpr_means = [last[last["attack"] == a]["tpr"].mean() for a in attacks]
        fpr_means = [last[last["attack"] == a]["fpr"].mean() for a in attacks]
        tpr_stds  = [last[last["attack"] == a]["tpr"].std() for a in attacks]
        fpr_stds  = [last[last["attack"] == a]["fpr"].std() for a in attacks]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, tpr_means, width, yerr=tpr_stds, label="TPR", capsize=5, color="steelblue")
        ax.bar(x + width/2, fpr_means, width, yerr=fpr_stds, label="FPR", capsize=5, color="tomato")
        ax.set_xticks(x)
        ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks], rotation=15)
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("Multi-Krum Detection: TPR and FPR by Attack")
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_mta_asr_matrix(
    summary: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of MTA and ASR: defenses × attacks."""
    attacks = summary["attack"].unique()
    defenses = summary["defense"].unique()

    mta_matrix = summary.pivot(index="defense", columns="attack", values="mta_mean")
    asr_matrix = summary.pivot(index="defense", columns="attack", values="asr_mean")

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, matrix, title, fmt in [
            (axes[0], mta_matrix, "Main Task Accuracy (MTA)", ".2f"),
            (axes[1], asr_matrix, "Attack Success Rate (ASR)", ".2f"),
        ]:
            im = ax.imshow(matrix.values, cmap="RdYlGn" if "MTA" in title else "RdYlGn_r",
                           vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(matrix.columns)))
            ax.set_yticks(range(len(matrix.index)))
            ax.set_xticklabels([ATTACK_LABELS.get(c, c) for c in matrix.columns], rotation=30)
            ax.set_yticklabels([DEFENSE_LABELS.get(r, r) for r in matrix.index])
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
            for i in range(len(matrix.index)):
                for j in range(len(matrix.columns)):
                    val = matrix.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=9)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_sensitivity_heatmap(
    df: pd.DataFrame,
    x_col: str = "m_assumed",
    y_col: str = "m_real",
    value_col: str = "mta_mean",
    title: str = "Sensitivity to m",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap for m_real × m_assumed sensitivity."""
    pivot = df.pivot(index=y_col, columns=x_col, values=value_col)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("m_assumed")
        ax.set_ylabel("m_real")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return fig
