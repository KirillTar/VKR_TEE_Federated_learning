#!/usr/bin/env python3
"""Build all figures for chapter 5.

Reads summary.json files from results/, renders PNGs into docs/images/.
Each figure is one function; missing data is skipped with a warning.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.6,
    "figure.dpi": 150,
})


def load_json(path: Path):
    if not path.exists():
        print(f"  [missing] {path.relative_to(ROOT) if path.is_absolute() else path}")
        return None
    with open(path) as fh:
        return json.load(fh)


def aggregate_mta(prefix_dir: Path, prefix: str) -> tuple[float, float, int]:
    """Return mean, std, n_seeds across cd_*_seedN_summary.json files."""
    files = sorted(prefix_dir.glob(f"{prefix}_seed*_summary.json"))
    vals = []
    for f in files:
        d = load_json(f)
        if d is None:
            continue
        v = d.get("mta_mean")
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def _aggregate_field(prefix_dir: Path, prefix: str, field: str) -> tuple[float, float, int]:
    files = sorted(prefix_dir.glob(f"{prefix}_seed*_summary.json"))
    vals = []
    for f in files:
        d = load_json(f)
        if d is None:
            continue
        v = d.get(field)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


# =====================================================================
# Fig 5.1 — convergence baselines
# =====================================================================
def fig_5_1_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    panels = [
        (axes[0], RES / "cross_device", [
            ("cd_bl1_fedavg", "FedAvg"),
            ("cd_bl3_fedrola", "FedRoLA"),
            ("cd_bl2_fedavg_dp", "FedAvg + ДП ε=10"),
            ("cd_bl4_fedrola_dp", "FedRoLA + ДП ε=10"),
            ("cd_bl5_mkrum_dp", "Multi-Krum + ДП ε=10"),
        ], "Межустройственный сценарий"),
        (axes[1], RES / "cross_silo", [
            ("cs_bl1_fedavg", "FedAvg"),
            ("cs_bl3_fedrola", "FedRoLA"),
            ("cs_bl2_fedavg_dp", "FedAvg + ДП ε=4"),
            ("cs_bl4_fedrola_dp", "FedRoLA + ДП ε=4"),
            ("cs_bl5_mkrum_dp", "Multi-Krum + ДП ε=4"),
        ], "Межорганизационный сценарий"),
    ]
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#ff7f0e"]
    for ax, dir_, configs, title in panels:
        for (prefix, label), c in zip(configs, colors):
            seed_csvs = [p for p in sorted(dir_.glob(f"{prefix}_seed*.csv"))
                         if "summary" not in p.name and "_R1" not in p.name]
            if not seed_csvs:
                print(f"  [missing] {dir_.name}/{prefix}_seed*.csv")
                continue
            dfs = [pd.read_csv(p) for p in seed_csvs]
            min_rounds = min(len(df) for df in dfs)
            mat = np.array([df["mta"].values[:min_rounds] for df in dfs])
            mean = mat.mean(0)
            std = mat.std(0)
            rounds = dfs[0]["round"].values[:min_rounds]
            ax.plot(rounds, mean, label=label, color=c)
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.15, color=c)
        ax.set_xlabel("Раунд обучения t")
        ax.set_ylabel("ТОЗ")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_ylim(0, 1.0)

    fig.suptitle("Рисунок 5.1 — Кривые сходимости базовых конфигураций (без атак)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_1_convergence.png", bbox_inches="tight")
    print("  saved fig_5_1_convergence.png")
    plt.close(fig)


# =====================================================================
# Fig 5.2 / 5.3 — Attack × Defense matrices (CD, CS)
# =====================================================================
def _matrix_panel(ax, dir_: Path, prefix_pat: str, attacks: list[tuple[str, str]],
                  defenses: list[tuple[str, str]], title: str):
    """Render heatmap MTA. attacks/defenses: list of (key, label)."""
    M = np.full((len(attacks), len(defenses)), np.nan)
    for i, (atk, _) in enumerate(attacks):
        for j, (defn, _) in enumerate(defenses):
            if atk == "none":
                # baselines: cd_bl{1..5}; cs same
                bl_map_cd = {
                    "fedavg": "cd_bl1_fedavg", "fedavg_dp": "cd_bl2_fedavg_dp",
                    "fedrola": "cd_bl3_fedrola", "fedrola_dp": "cd_bl4_fedrola_dp",
                    "mkrum_dp": "cd_bl5_mkrum_dp",
                }
                bl_map_cs = {
                    "fedavg": "cs_bl1_fedavg", "fedavg_dp": "cs_bl2_fedavg_dp",
                    "fedrola": "cs_bl3_fedrola", "fedrola_dp": "cs_bl4_fedrola_dp",
                    "mkrum_dp": "cs_bl5_mkrum_dp",
                }
                bl_map = bl_map_cd if dir_.name == "cross_device" else bl_map_cs
                prefix = bl_map.get(defn)
            else:
                prefix = prefix_pat.format(attack=atk, defense=defn)
            if prefix is None:
                continue
            mean, _, n = aggregate_mta(dir_, prefix)
            if n > 0:
                M[i, j] = mean

    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(defenses)))
    ax.set_yticks(range(len(attacks)))
    ax.set_xticklabels([d[1] for d in defenses], rotation=20, ha="right", fontsize=8)
    ax.set_yticklabels([a[1] for a in attacks], fontsize=9)
    ax.set_title(title)
    for i in range(len(attacks)):
        for j in range(len(defenses)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    return im


def fig_5_2_matrix_cd():
    attacks = [
        ("none", "Без атаки"),
        ("alie", "ALIE"),
        ("ipm", "IPM"),
        ("minmax", "Min-Max"),
        ("backdoor", "Закладка"),
    ]
    defenses = [
        ("fedavg", "FedAvg"),
        ("fedavg_dp", "FedAvg+ДП"),
        ("fedrola", "FedRoLA"),
        ("fedrola_dp", "FedRoLA+ДП"),
        ("mkrum_dp", "MKrum+ДП"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = _matrix_panel(ax, RES / "cross_device", "cd_exp1_{attack}_{defense}",
                       attacks, defenses,
                       "Рисунок 5.2 — Матрица «атака × защита», ТОЗ, межустройственный сценарий")
    fig.colorbar(im, ax=ax, label="ТОЗ")
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_2_matrix_cd.png", bbox_inches="tight")
    print("  saved fig_5_2_matrix_cd.png")
    plt.close(fig)


def fig_5_3_matrix_cs():
    attacks = [
        ("none", "Без атаки"),
        ("alie", "ALIE"),
        ("ipm", "IPM"),
        ("minmax", "Min-Max"),
        ("backdoor", "Закладка"),
        ("label_flip", "Подмена 5→3"),
        ("random_label_flip", "Случайная перестановка"),
    ]
    defenses = [
        ("fedavg", "FedAvg"),
        ("fedavg_dp", "FedAvg+ДП"),
        ("fedrola", "FedRoLA"),
        ("fedrola_dp", "FedRoLA+ДП"),
        ("mkrum_dp", "MKrum+ДП"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = _matrix_panel(ax, RES / "cross_silo", "cs_exp1_{attack}_{defense}",
                       attacks, defenses,
                       "Рисунок 5.3 — Матрица «атака × защита», ТОЗ, межорганизационный сценарий")
    fig.colorbar(im, ax=ax, label="ТОЗ")
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_3_matrix_cs.png", bbox_inches="tight")
    print("  saved fig_5_3_matrix_cs.png")
    plt.close(fig)


# =====================================================================
# Fig 5.4 — Backdoor ASR vs m_real (CD)
# =====================================================================
def fig_5_4_backdoor_mreal():
    m_pcts = [(20, 10), (40, 20), (60, 30), (80, 40)]
    asrs, asr_stds = [], []
    mtas, mta_stds = [], []
    for m_real, _pct in m_pcts:
        prefix = f"cd_diag_backdoor_m{m_real}_fedrola_dp"
        a_m, a_s, _ = _aggregate_field(RES / "cross_device", prefix, "asr_mean")
        t_m, t_s, _ = _aggregate_field(RES / "cross_device", prefix, "mta_mean")
        asrs.append(a_m); asr_stds.append(a_s)
        mtas.append(t_m); mta_stds.append(t_s)

    pcts = [p for _, p in m_pcts]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(pcts, asrs, yerr=asr_stds, marker="o", capsize=4,
                label="ТЗА (атака закладкой)", color="#d62728")
    ax.errorbar(pcts, mtas, yerr=mta_stds, marker="s", capsize=4,
                label="ТОЗ", color="#1f77b4")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="порог 0,5")
    ax.set_xlabel("Доля вредоносных, %")
    ax.set_ylabel("Метрика")
    ax.set_ylim(0, 1.05)
    ax.set_title("Рисунок 5.4 — Атака со скрытой закладкой: ТЗА и ТОЗ vs доля вредоносных\n"
                 "(CD, FedRoLA + ДП ε=10, три seeds)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_4_backdoor_mreal.png", bbox_inches="tight")
    print("  saved fig_5_4_backdoor_mreal.png")
    plt.close(fig)


# =====================================================================
# Fig 5.5 — TPR vs m_real for ALIE & Backdoor (CD)
# =====================================================================
def fig_5_5_tpr_mreal():
    pcts = [10, 20, 30, 40]
    m_to_alie = {10: 20, 20: 40, 30: 60, 40: 80}  # m_real labels
    m_to_back = {10: 20, 20: 40, 30: 60, 40: 80}

    alie_tpr, alie_std = [], []
    back_tpr, back_std = [], []
    for p in pcts:
        a_m, a_s, _ = _aggregate_field(RES / "cross_device",
                                       f"cd_p3_alie_m{m_to_alie[p]}_fedrola_dp",
                                       "tpr_mean")
        # cd_p3 is ε=15 (per phase3 plan); fall back to diag if needed
        if np.isnan(a_m):
            a_m, a_s, _ = _aggregate_field(RES / "cross_device",
                                           f"cd_diag_alie_m{m_to_alie[p]}_fedrola_dp",
                                           "tpr_mean")
        alie_tpr.append(a_m); alie_std.append(a_s)

        b_m, b_s, _ = _aggregate_field(RES / "cross_device",
                                       f"cd_diag_backdoor_m{m_to_back[p]}_fedrola_dp",
                                       "tpr_mean")
        back_tpr.append(b_m); back_std.append(b_s)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(pcts, alie_tpr, yerr=alie_std, marker="o", capsize=4,
                label="ALIE", color="#1f77b4")
    ax.errorbar(pcts, back_tpr, yerr=back_std, marker="s", capsize=4,
                label="Атака со скрытой закладкой", color="#d62728")
    ax.set_xlabel("Доля вредоносных, %")
    ax.set_ylabel("TPR детектора FedRoLA")
    ax.set_ylim(0, 1.05)
    ax.set_title("Рисунок 5.5 — TPR FedRoLA vs доля вредоносных\n"
                 "(CD, FedRoLA + ДП, три seeds)")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_5_tpr_mreal.png", bbox_inches="tight")
    print("  saved fig_5_5_tpr_mreal.png")
    plt.close(fig)


# =====================================================================
# Fig 5.6 — S-curves CD/CS
# =====================================================================
def fig_5_6_scurves():
    cd_eps = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    cs_eps = [0.5, 0.75, 1, 1.5, 2, 3, 4, 8]

    def collect(dir_: Path, prefix_for_eps):
        means, stds = [], []
        for eps in (cd_eps if dir_.name == "cross_device" else cs_eps):
            tried = prefix_for_eps(eps)
            mtas = []
            for prefix in tried:
                files = sorted(dir_.glob(f"{prefix}_seed*_summary.json"))
                for f in files:
                    d = load_json(f)
                    if d is None:
                        continue
                    v = d.get("mta_mean")
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        mtas.append(v)
                if mtas:
                    break
            if mtas:
                means.append(float(np.mean(mtas)))
                stds.append(float(np.std(mtas)))
            else:
                means.append(float("nan"))
                stds.append(float("nan"))
        return means, stds

    def cd_prefix(eps):
        # cd_diag_eps{2..8}_none, cd_exp2_eps{1,10,15,20}_none
        return [f"cd_diag_eps{eps}_none", f"cd_exp2_eps{eps}_none"]

    def cs_prefix(eps):
        eps_str = str(eps)
        # cs_p3_eps{0.5,0.75,1,1.5,3}_none, cs_exp2_eps{2,4,8}_none
        return [f"cs_p3_eps{eps_str}_none", f"cs_exp2_eps{eps_str}_none",
                f"cs_p3_eps{eps_str}_none"]

    cd_mean, cd_std = collect(RES / "cross_device", cd_prefix)
    cs_mean, cs_std = collect(RES / "cross_silo", cs_prefix)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].errorbar(cd_eps, cd_mean, yerr=cd_std, marker="o", capsize=4,
                     color="#1f77b4")
    axes[0].axvline(4, color="red", linestyle="--", alpha=0.5, label="перегиб ε≈4")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Бюджет приватности ε (лог)")
    axes[0].set_ylabel("ТОЗ")
    axes[0].set_title("Межустройственный сценарий\n(ДП на уровне клиента, d=4266)")
    axes[0].legend()
    axes[0].set_ylim(0, 1.0)

    axes[1].errorbar(cs_eps, cs_mean, yerr=cs_std, marker="s", capsize=4,
                     color="#ff7f0e")
    axes[1].axvline(1, color="red", linestyle="--", alpha=0.5, label="перегиб ε≈1")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Бюджет приватности ε (лог)")
    axes[1].set_ylabel("ТОЗ")
    axes[1].set_title("Межорганизационный сценарий\n(ДП на уровне записей, d=89834)")
    axes[1].legend()
    axes[1].set_ylim(0, 1.0)

    fig.suptitle("Рисунок 5.6 — S-кривые ТОЗ от бюджета приватности (FedRoLA + ДП, без атак)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_6_scurves.png", bbox_inches="tight")
    print("  saved fig_5_6_scurves.png")
    plt.close(fig)


# =====================================================================
# Fig 5.7 — SNR vs MTA (existing PNG, copy)
# =====================================================================
def fig_5_7_snr_copy():
    src = RES / "snr_vs_mta.png"
    dst = OUT / "fig_5_7_snr_vs_mta.png"
    if src.exists():
        shutil.copy(src, dst)
        print(f"  copied {dst.name}")
    else:
        print(f"  [missing] {src}")


# =====================================================================
# Fig 5.8 — Alpha (Dirichlet) sensitivity
# =====================================================================
def fig_5_8_alpha():
    alphas = [0.1, 0.3, 1.0, 10.0]
    alpha_tags = ["0p1", "0p3", "1p0", "10p0"]

    def gather(dir_: Path, scenario_tag: str, attack: str):
        means, stds = [], []
        for tag in alpha_tags:
            prefix = f"{scenario_tag}_exp5_a{tag}_{attack}"
            m, s, _ = _aggregate_field(dir_, prefix, "mta_mean")
            means.append(m)
            stds.append(s)
        return means, stds

    cd_none_m, cd_none_s = gather(RES / "cross_device", "cd", "none")
    cd_alie_m, cd_alie_s = gather(RES / "cross_device", "cd", "alie")
    cs_none_m, cs_none_s = gather(RES / "cross_silo", "cs", "none")
    cs_alie_m, cs_alie_s = gather(RES / "cross_silo", "cs", "alie")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(alphas, cd_none_m, yerr=cd_none_s, marker="o",
                capsize=4, label="CD, без атаки", color="#1f77b4")
    ax.errorbar(alphas, cd_alie_m, yerr=cd_alie_s, marker="o",
                capsize=4, linestyle="--", label="CD, ALIE", color="#1f77b4", alpha=0.6)
    ax.errorbar(alphas, cs_none_m, yerr=cs_none_s, marker="s",
                capsize=4, label="CS, без атаки", color="#ff7f0e")
    ax.errorbar(alphas, cs_alie_m, yerr=cs_alie_s, marker="s",
                capsize=4, linestyle="--", label="CS, ALIE", color="#ff7f0e", alpha=0.6)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.6, label="базовая α=0,5")
    ax.set_xscale("log")
    ax.set_xlabel("Параметр Дирихле α (лог)")
    ax.set_ylabel("ТОЗ")
    ax.set_ylim(0, 1.0)
    ax.set_title("Рисунок 5.8 — Чувствительность ТОЗ к неоднородности данных\n"
                 "(FedRoLA + ДП, три seeds)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_8_alpha.png", bbox_inches="tight")
    print("  saved fig_5_8_alpha.png")
    plt.close(fig)


# =====================================================================
# Fig 5.9 — T=500 scatter
# =====================================================================
def fig_5_9_t500():
    configs = [
        ("ALIE + ДП ε=4", "t500_alie_fedrola_dp", "#d62728"),
        ("Min-Max + ДП ε=4", "t500_minmax_fedrola_dp", "#9467bd"),
        ("ALIE без ДП", "t500_alie_fedrola", "#1f77b4"),
        ("Min-Max без ДП", "t500_minmax_fedrola", "#2ca02c"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, prefix, color in configs:
        files = sorted((RES / "t500_probe").glob(f"{prefix}_seed*_summary.json"))
        seeds, mtas = [], []
        for f in files:
            d = load_json(f)
            if d is None:
                continue
            seed = int(f.stem.split("seed")[1].split("_")[0])
            v = d.get("mta_mean")
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            seeds.append(seed); mtas.append(v)
        if not seeds:
            continue
        ax.scatter(seeds, mtas, color=color, s=60, label=label, edgecolor="black")
        if len(mtas) > 1:
            mean = float(np.mean(mtas))
            std = float(np.std(mtas))
            ax.axhline(mean, color=color, linestyle=":", alpha=0.5)
            ax.fill_between([min(seeds) - 0.3, max(seeds) + 0.3],
                            mean - std, mean + std, color=color, alpha=0.08)
    ax.set_xlabel("Seed")
    ax.set_ylabel("ТОЗ")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(0, 1.0)
    ax.set_title("Рисунок 5.9 — Длинногоризонтная проба T=500: разброс ТОЗ по seeds\n"
                 "(межорганизационный сценарий, FedRoLA)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_9_t500.png", bbox_inches="tight")
    print("  saved fig_5_9_t500.png")
    plt.close(fig)


# =====================================================================
# Fig 5.10 — MIA AUC bar chart, range 0.43–0.52
# =====================================================================
def fig_5_10_mia():
    d = load_json(RES / "mia_cs" / "mia_summary.json")
    if d is None:
        return
    keys = [("no_dp", "FL без ДП"),
            ("dp_eps4", "FL + ДП ε=4"),
            ("dp_eps8", "FL + ДП ε=8")]
    auc_loss_m = [d[k]["auc_loss_mean"] for k, _ in keys]
    auc_loss_s = [d[k]["auc_loss_std"] for k, _ in keys]
    labels = [v for _, v in keys]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, auc_loss_m, yerr=auc_loss_s, capsize=5,
                  color=["#1f77b4", "#9467bd", "#d62728"], edgecolor="black")
    ax.axhline(0.5, color="red", linestyle="--", label="порог случайного угадывания")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC ROC (loss-based MIA)")
    ax.set_ylim(0.42, 0.52)
    ax.set_title("Рисунок 5.10 — AUC атаки определения членства\n"
                 "(межорганизационный сценарий, три seeds)")
    for b, v in zip(bars, auc_loss_m):
        ax.text(b.get_x() + b.get_width()/2, v + 0.003, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_10_mia_auc.png", bbox_inches="tight")
    print("  saved fig_5_10_mia_auc.png")
    plt.close(fig)


# =====================================================================
# Fig 5.11 — GIA grid (existing PNG, copy)
# =====================================================================
def fig_5_11_gia_copy():
    src = RES / "local_dp" / "gia" / "gia_grid.png"
    dst = OUT / "fig_5_11_gia_grid.png"
    if src.exists():
        shutil.copy(src, dst)
        print(f"  copied {dst.name}")
    else:
        print(f"  [missing] {src}")


# =====================================================================
# Fig 5.12 — Central vs Local DP bars
# =====================================================================
def fig_5_12_central_vs_local():
    cd_central, cd_central_s, _ = _aggregate_field(
        RES / "local_dp_cd", "local_dp_cd_central", "mta_mean")
    cd_local, cd_local_s, _ = _aggregate_field(
        RES / "local_dp_cd", "local_dp_cd_local_fixed", "mta_mean")
    cd_central_alie, cd_central_alie_s, _ = _aggregate_field(
        RES / "cross_device", "cd_exp1_alie_fedrola_dp", "mta_mean")
    cd_local_alie, cd_local_alie_s, _ = _aggregate_field(
        RES / "local_dp_attack", "local_dp_cd_alie", "mta_mean")
    cs_local_alie, cs_local_alie_s, _ = _aggregate_field(
        RES / "local_dp_attack", "local_dp_cs_alie", "mta_mean")
    cs_central_alie, cs_central_alie_s, _ = _aggregate_field(
        RES / "cross_silo", "cs_exp1_alie_fedrola_dp", "mta_mean")

    groups = ["CD без атак\n(ε=10)", "CD + ALIE\n(ε=10)", "CS + ALIE\n(ε=4)"]
    central = [cd_central, cd_central_alie, cs_central_alie]
    central_err = [cd_central_s, cd_central_alie_s, cs_central_alie_s]
    local = [cd_local, cd_local_alie, cs_local_alie]
    local_err = [cd_local_s, cd_local_alie_s, cs_local_alie_s]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    b1 = ax.bar(x - width/2, central, width, yerr=central_err, capsize=4,
                label="Центральная ДП", color="#2ca02c", edgecolor="black")
    b2 = ax.bar(x + width/2, local, width, yerr=local_err, capsize=4,
                label="Локальная ДП", color="#d62728", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("ТОЗ")
    ax.set_ylim(0, 1.0)
    ax.set_title("Рисунок 5.12 — Центральная и локальная дифференциальная приватность\n"
                 "(FedRoLA, фиксированный ε)")
    ax.legend(loc="upper right")
    for bars, vals in [(b1, central), (b2, local)]:
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, v + 0.02,
                        f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_12_central_vs_local.png", bbox_inches="tight")
    print("  saved fig_5_12_central_vs_local.png")
    plt.close(fig)


# =====================================================================
# Run all
# =====================================================================
def main():
    print(f"Output dir: {OUT}")
    fig_5_1_convergence()
    fig_5_2_matrix_cd()
    fig_5_3_matrix_cs()
    fig_5_4_backdoor_mreal()
    fig_5_5_tpr_mreal()
    fig_5_6_scurves()
    fig_5_7_snr_copy()
    fig_5_8_alpha()
    fig_5_9_t500()
    fig_5_10_mia()
    fig_5_11_gia_copy()
    fig_5_12_central_vs_local()
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
