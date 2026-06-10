#!/usr/bin/env python3
"""Dump all numeric values used in chapter 5 tables — for text reconciliation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def field(prefix_dir: Path, prefix: str, key: str = "mta_mean"):
    files = sorted(prefix_dir.glob(f"{prefix}_seed*_summary.json"))
    vals = [json.load(open(f)).get(key) for f in files]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def fmt(m, s, n):
    if m is None:
        return "—"
    return f"{m:.3f} ± {s:.3f} (n={n})"


# ============= Table 5.2 baselines =============
print("=== Table 5.2 — Baselines (without attacks) ===")
for d, prefix, label in [
    (RES / "cross_device", "cd_bl1_fedavg", "CD FedAvg"),
    (RES / "cross_device", "cd_bl3_fedrola", "CD FedRoLA"),
    (RES / "cross_device", "cd_bl2_fedavg_dp", "CD FedAvg+DP"),
    (RES / "cross_device", "cd_bl4_fedrola_dp", "CD FedRoLA+DP"),
    (RES / "cross_device", "cd_bl5_mkrum_dp", "CD MKrum+DP"),
    (RES / "cross_silo", "cs_bl1_fedavg", "CS FedAvg"),
    (RES / "cross_silo", "cs_bl3_fedrola", "CS FedRoLA"),
    (RES / "cross_silo", "cs_bl2_fedavg_dp", "CS FedAvg+DP"),
    (RES / "cross_silo", "cs_bl4_fedrola_dp", "CS FedRoLA+DP"),
    (RES / "cross_silo", "cs_bl5_mkrum_dp", "CS MKrum+DP"),
]:
    print(f"  {label}: MTA = {fmt(*field(d, prefix))}")

# ============= Table 5.3 — CD attack matrix =============
print("\n=== Table 5.3 — CD attack matrix (MTA, TPR) ===")
defenses_cd = ["fedavg", "fedavg_dp", "fedrola", "fedrola_dp", "mkrum_dp"]
attacks = ["alie", "ipm", "minmax", "backdoor"]
for atk in attacks:
    print(f"  -- {atk} --")
    for defn in defenses_cd:
        prefix = f"cd_exp1_{atk}_{defn}"
        mta = field(RES / "cross_device", prefix, "mta_mean")
        tpr = field(RES / "cross_device", prefix, "tpr_mean")
        asr = field(RES / "cross_device", prefix, "asr_mean")
        print(f"    {defn}: MTA={fmt(*mta)}, TPR={fmt(*tpr)}, ASR={fmt(*asr)}")

# ============= Table 5.4 — CS attack matrix =============
print("\n=== Table 5.4 — CS attack matrix ===")
for atk in ["alie", "ipm", "minmax", "backdoor", "label_flip", "random_label_flip"]:
    print(f"  -- {atk} --")
    for defn in defenses_cd:
        prefix = f"cs_exp1_{atk}_{defn}"
        mta = field(RES / "cross_silo", prefix, "mta_mean")
        tpr = field(RES / "cross_silo", prefix, "tpr_mean")
        asr = field(RES / "cross_silo", prefix, "asr_mean")
        sa = field(RES / "cross_silo", prefix, "source_acc_mean")
        print(f"    {defn}: MTA={fmt(*mta)}, TPR={fmt(*tpr)}, ASR={fmt(*asr)}", end="")
        if sa[0] is not None:
            print(f", source_acc={fmt(*sa)}")
        else:
            print()

# ============= Table 5.5 — Backdoor m_real =============
print("\n=== Table 5.5/5.8 — Backdoor m_real (CD, FedRoLA+DP) ===")
for m in [20, 40, 60, 80]:
    prefix = f"cd_diag_backdoor_m{m}_fedrola_dp"
    mta = field(RES / "cross_device", prefix, "mta_mean")
    asr = field(RES / "cross_device", prefix, "asr_mean")
    tpr = field(RES / "cross_device", prefix, "tpr_mean")
    print(f"  m={m} (={m//2}%): MTA={fmt(*mta)}, ASR={fmt(*asr)}, TPR={fmt(*tpr)}")

# ============= Table 5.7 — ALIE m_real =============
print("\n=== Table 5.7 — ALIE m_real (CD, FedRoLA+DP, ε=?) ===")
for m in [20, 40, 60, 80]:
    prefix = f"cd_p3_alie_m{m}_fedrola_dp"
    mta = field(RES / "cross_device", prefix, "mta_mean")
    tpr = field(RES / "cross_device", prefix, "tpr_mean")
    fpr = field(RES / "cross_device", prefix, "fpr_mean")
    print(f"  m={m} (={m//2}%): MTA={fmt(*mta)}, TPR={fmt(*tpr)}, FPR={fmt(*fpr)}")

# ============= Table 5.9/5.10 — S-curves =============
print("\n=== Table 5.9 — CD S-curve (none, FedRoLA+DP) ===")
for eps in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
    for prefix in [f"cd_diag_eps{eps}_none", f"cd_exp2_eps{eps}_none"]:
        mta = field(RES / "cross_device", prefix, "mta_mean")
        if mta[0] is not None:
            print(f"  ε={eps}: {fmt(*mta)} ({prefix})")
            break
    else:
        print(f"  ε={eps}: missing")

print("\n=== Table 5.10 — CS S-curve (none, FedRoLA+DP) ===")
for eps in [0.5, 0.75, 1, 1.5, 2, 3, 4, 8]:
    for prefix in [f"cs_p3_eps{eps}_none", f"cs_exp2_eps{eps}_none"]:
        mta = field(RES / "cross_silo", prefix, "mta_mean")
        if mta[0] is not None:
            print(f"  ε={eps}: {fmt(*mta)} ({prefix})")
            break
    else:
        print(f"  ε={eps}: missing")

# ============= Table 5.11 — Ablation =============
print("\n=== Table 5.11 — Ablation (CD, m_real=80) ===")
for atk in ["alie", "backdoor", "ipm"]:
    print(f"  -- {atk} --")
    for cfg, prefix_suffix in [
        ("Neither (FedAvg, no DP)", "fedavg_nodp"),
        ("DP only (FedAvg+DP)", "fedavg_dp"),
        ("FedRoLA only", "fedrola_nodp"),
        ("FedRoLA+DP", "fedrola_dp"),
    ]:
        prefix = f"cd_p3_ablation_{atk}_{prefix_suffix}"
        mta = field(RES / "cross_device", prefix, "mta_mean")
        tpr = field(RES / "cross_device", prefix, "tpr_mean")
        asr = field(RES / "cross_device", prefix, "asr_mean")
        print(f"    {cfg}: MTA={fmt(*mta)}, TPR={fmt(*tpr)}, ASR={fmt(*asr)}")

# ============= Table 5.12 — alpha sweep =============
print("\n=== Table 5.12 — Alpha sweep (FedRoLA+DP) ===")
for tag, alpha in [("0p1", 0.1), ("0p3", 0.3), ("1p0", 1.0), ("10p0", 10.0)]:
    print(f"  -- α={alpha} --")
    for sce, dir_, atk_label, atk in [
        ("CD none", RES / "cross_device", "no attack", "none"),
        ("CD ALIE", RES / "cross_device", "ALIE", "alie"),
        ("CS none", RES / "cross_silo", "no attack", "none"),
        ("CS ALIE", RES / "cross_silo", "ALIE", "alie"),
    ]:
        sce_tag = "cd" if "CD" in sce else "cs"
        prefix = f"{sce_tag}_exp5_a{tag}_{atk}"
        mta = field(dir_, prefix, "mta_mean")
        print(f"    {sce} {atk_label}: MTA={fmt(*mta)}")

# ============= Table 5.13 — T=500 probe =============
print("\n=== Table 5.13 — T=500 probe (CS, FedRoLA) ===")
for label, prefix in [
    ("ALIE+DP", "t500_alie_fedrola_dp"),
    ("MinMax+DP", "t500_minmax_fedrola_dp"),
    ("ALIE no_dp", "t500_alie_fedrola"),
    ("MinMax no_dp", "t500_minmax_fedrola"),
]:
    files = sorted((RES / "t500_probe").glob(f"{prefix}_seed*_summary.json"))
    pairs = []
    for f in files:
        seed = int(f.stem.split("seed")[1].split("_")[0])
        d = json.load(open(f))
        v = d.get("mta_mean")
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            pairs.append((seed, v))
    pairs.sort()
    seeds_str = ", ".join(f"seed{s}={v:.3f}" for s, v in pairs)
    if pairs:
        m = float(np.mean([v for _, v in pairs]))
        s = float(np.std([v for _, v in pairs])) if len(pairs) > 1 else 0.0
        print(f"  {label}: {seeds_str} | mean={m:.3f}, std={s:.3f}, n={len(pairs)}")
    else:
        print(f"  {label}: missing")

# ============= Table 5.14 — MIA =============
print("\n=== Table 5.14 — MIA (CS) ===")
mia = json.load(open(RES / "mia_cs" / "mia_summary.json"))
for k in ["no_dp", "dp_eps4", "dp_eps8"]:
    print(f"  {k}: MTA={mia[k]['mta_mean']:.4f}±{mia[k]['mta_std']:.4f}, "
          f"AUC_loss={mia[k]['auc_loss_mean']:.4f}±{mia[k]['auc_loss_std']:.4f}")

# ============= Table 5.16 — Central vs Local DP =============
print("\n=== Table 5.16 — Central vs Local DP ===")
for label, dir_, prefix in [
    ("CD Central (no attack)", RES / "local_dp_cd", "local_dp_cd_central"),
    ("CD Local fixed (no attack)", RES / "local_dp_cd", "local_dp_cd_local_fixed"),
    ("CD Central + ALIE (exp1)", RES / "cross_device", "cd_exp1_alie_fedrola_dp"),
    ("CD Local + ALIE", RES / "local_dp_attack", "local_dp_cd_alie"),
    ("CS Central + ALIE (exp1)", RES / "cross_silo", "cs_exp1_alie_fedrola_dp"),
    ("CS Local + ALIE", RES / "local_dp_attack", "local_dp_cs_alie"),
]:
    mta = field(dir_, prefix, "mta_mean")
    tpr = field(dir_, prefix, "tpr_mean")
    print(f"  {label}: MTA={fmt(*mta)}, TPR={fmt(*tpr)}")
