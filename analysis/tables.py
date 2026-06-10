"""Generate LaTeX tables from experiment results."""

import pandas as pd
import numpy as np


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
    "fedavg_cdp4": "CDP ($\\varepsilon=4$)",
    "mkrum_cdp4": "MKrum+CDP ($\\varepsilon=4$)",
    "mkrum_cdp1": "MKrum+CDP ($\\varepsilon=1$)",
}


def summarize_last_round(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    """Aggregate metrics at final round, mean ± std across seeds."""
    last = df[df["round"] == df["round"].max()]
    agg = last.groupby(groupby).agg(
        mta_mean=("mta", "mean"),
        mta_std=("mta", "std"),
        asr_mean=("asr", "mean"),
        asr_std=("asr", "std"),
        tpr_mean=("tpr", "mean"),
        tpr_std=("tpr", "std"),
        fpr_mean=("fpr", "mean"),
        fpr_std=("fpr", "std"),
    ).reset_index()
    return agg


def format_mean_std(mean: float, std: float, pct: bool = True, bold_best: bool = False) -> str:
    """Format as 'xx.x ± y.y%'."""
    scale = 100 if pct else 1
    fmt = f"{mean * scale:.1f} $\\pm$ {std * scale:.1f}"
    if bold_best:
        fmt = f"\\textbf{{{fmt}}}"
    return fmt


def make_main_table(
    summary: pd.DataFrame,
    attack_col: str = "attack",
    defense_col: str = "defense",
    caption: str = "Main Task Accuracy (MTA) and Attack Success Rate (ASR) across attacks and defenses. Mean $\\pm$ std over 3 seeds.",
    label: str = "tab:main_results",
) -> str:
    """
    Generate LaTeX table: rows = defenses, columns = attacks.
    Each cell: MTA / ASR.
    """
    attacks = summary[attack_col].unique()
    defenses = summary[defense_col].unique()

    col_spec = "l" + "c" * len(attacks)
    header_attacks = " & ".join(
        [f"\\multicolumn{{1}}{{c}}{{{ATTACK_LABELS.get(a, a)}}}" for a in attacks]
    )

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"Defense & {header_attacks} \\\\",
        "\\midrule",
    ]

    for defense in defenses:
        row_values = []
        for attack in attacks:
            mask = (summary[attack_col] == attack) & (summary[defense_col] == defense)
            row = summary[mask]
            if row.empty:
                row_values.append("—")
            else:
                mta = f"{row['mta_mean'].values[0]*100:.1f}"
                mta_s = f"{row['mta_std'].values[0]*100:.1f}"
                asr = f"{row['asr_mean'].values[0]*100:.1f}"
                asr_s = f"{row['asr_std'].values[0]*100:.1f}"
                cell = f"{mta}$\\pm${mta_s} / {asr}$\\pm${asr_s}"
                row_values.append(cell)
        defense_label = DEFENSE_LABELS.get(defense, defense)
        lines.append(f"{defense_label} & " + " & ".join(row_values) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def make_tpr_fpr_table(
    summary: pd.DataFrame,
    attack_col: str = "attack",
    caption: str = "Multi-Krum detection performance (TPR/FPR) by attack type.",
    label: str = "tab:tpr_fpr",
) -> str:
    """Generate TPR/FPR table."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Attack & TPR (\\%) & FPR (\\%) \\\\",
        "\\midrule",
    ]

    for attack in summary[attack_col].unique():
        row = summary[summary[attack_col] == attack]
        if row.empty:
            continue
        tpr = f"{row['tpr_mean'].values[0]*100:.1f} $\\pm$ {row['tpr_std'].values[0]*100:.1f}"
        fpr = f"{row['fpr_mean'].values[0]*100:.1f} $\\pm$ {row['fpr_std'].values[0]*100:.1f}"
        lines.append(f"{ATTACK_LABELS.get(attack, attack)} & {tpr} & {fpr} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def make_epsilon_table(
    summary: pd.DataFrame,
    epsilon_col: str = "epsilon",
    attack_col: str = "attack",
    caption: str = "MTA vs privacy budget $\\varepsilon$ (Multi-Krum + Central DP).",
    label: str = "tab:epsilon_sweep",
) -> str:
    """ε sweep table."""
    epsilons = sorted(summary[epsilon_col].unique())
    attacks = summary[attack_col].unique()

    col_spec = "l" + "c" * len(epsilons)
    header = " & ".join([f"$\\varepsilon={e}$" for e in epsilons])

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"Attack & {header} \\\\",
        "\\midrule",
    ]

    for attack in attacks:
        row_values = []
        for eps in epsilons:
            mask = (summary[epsilon_col] == eps) & (summary[attack_col] == attack)
            row = summary[mask]
            if row.empty:
                row_values.append("—")
            else:
                mta = f"{row['mta_mean'].values[0]*100:.1f}"
                mta_s = f"{row['mta_std'].values[0]*100:.1f}"
                row_values.append(f"{mta}$\\pm${mta_s}")
        lines.append(f"{ATTACK_LABELS.get(attack, attack)} & " + " & ".join(row_values) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def save_table(latex: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX table: {path}")
