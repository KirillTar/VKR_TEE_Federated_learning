"""
Run all Phase 0 baselines sequentially: BL-0 through BL-6.
Each baseline uses 3 seeds. Results saved to results/baselines/.

Red flags (as per CLAUDE.md):
  BL-1: MTA < 0.75 or > 0.88  → FL-loop bug
  BL-2: MTA < 0.65             → data split bug
  BL-3: MTA drop > 0.08 vs BL-2 → aggregation bug
  BL-4: MTA(ε=2) < MTA(ε=1)   → DP accounting bug
  BL-5: attacks don't degrade  → attack implementation bug
  BL-6: TPR < 1.0              → Krum bug (sign-flip is max outlier)
"""
import sys, os, json, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.fl_simulator import FLSimulator, FLConfig

os.makedirs("results/baselines", exist_ok=True)

# Logging with immediate flush
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        FlushFileHandler("results/baselines/run.log"),
    ]
)
logger = logging.getLogger(__name__)

SEEDS = [0, 1, 2]


def run(name: str, cfg_kwargs: dict, seeds=SEEDS, red_flags: dict = None):
    import pandas as pd
    records = []
    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"START: {name}  seed={seed}")
        logger.info(f"{'='*60}")
        cfg = FLConfig(seed=seed, **cfg_kwargs)
        sim = FLSimulator(cfg)
        m = sim.run()
        records.extend(m.records)

    df = pd.DataFrame(records)
    df.to_csv(f"results/baselines/{name}.csv", index=False)

    last = df[df["round"] == df["round"].max()]
    summary = {
        "mta_mean": round(float(last["mta"].mean()), 4),
        "mta_std":  round(float(last["mta"].std()),  4),
        "asr_mean": round(float(last["asr"].mean()), 4),
        "tpr_mean": round(float(last["tpr"].mean()), 4),
        "fpr_mean": round(float(last["fpr"].mean()), 4),
    }
    with open(f"results/baselines/{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"DONE: {name} → MTA={summary['mta_mean']:.4f}±{summary['mta_std']:.4f}  "
                f"ASR={summary['asr_mean']:.4f}  TPR={summary['tpr_mean']:.3f}")

    # Check red flags
    if red_flags:
        for flag, (condition, msg) in red_flags.items():
            if condition(summary):
                logger.warning(f"  ⚠ RED FLAG [{flag}]: {msg}")
            else:
                logger.info(f"  ✓ {flag}: OK")

    return summary


# Common hyperparameters (from CLAUDE.md)
COMMON = dict(
    dataset="cifar10", data_dir="./data",
    n_rounds=200, local_epochs=5, batch_size=32,
    lr=0.01, momentum=0.9, weight_decay=1e-4,
    log_every=20,
)


# BL-0: Centralized (upper bound ~86-88%)
bl0 = run("bl0_centralized", {**COMMON,
    "alpha": 10.0, "n_clients": 1, "n_rounds": 50, "local_epochs": 10,
    "aggregation": "fedavg", "m_assumed": 0,
    "use_dp": False, "epsilon": float("inf"), "attack": "none", "m_real": 0,
}, red_flags={
    "BL-0 MTA>75%": (lambda s: s["mta_mean"] < 0.75, "Centralized < 75% — model or data bug"),
})

# BL-1: FedAvg, IID (~83-86%)
bl1 = run("bl1_fedavg_iid", {**COMMON,
    "alpha": 10.0, "aggregation": "fedavg", "m_assumed": 0,
    "use_dp": False, "epsilon": float("inf"), "attack": "none", "m_real": 0,
}, red_flags={
    "BL-1 75%<MTA<88%": (lambda s: not (0.75 < s["mta_mean"] < 0.88),
                          f"FedAvg IID out of expected range [0.75, 0.88]"),
})

# BL-2: FedAvg, non-IID alpha=0.5 (~76-80%)
bl2 = run("bl2_fedavg_noniid", {**COMMON,
    "alpha": 0.5, "aggregation": "fedavg", "m_assumed": 0,
    "use_dp": False, "epsilon": float("inf"), "attack": "none", "m_real": 0,
}, red_flags={
    "BL-2 MTA>65%": (lambda s: s["mta_mean"] < 0.65, "FedAvg non-IID < 65% — split bug"),
})

# BL-3: Multi-Krum, no attacks (MTA ≈ BL-2 minus 1-3%, FPR ~ 30%)
bl3 = run("bl3_mkrum_no_attack", {**COMMON,
    "alpha": 0.5, "aggregation": "multi_krum", "m_assumed": 6, "f": 14,
    "use_dp": False, "epsilon": float("inf"), "attack": "none", "m_real": 0,
}, red_flags={
    "BL-3 Krum drop<8%": (lambda s: (bl2["mta_mean"] - s["mta_mean"]) > 0.08,
                           "Krum drops MTA > 8% — aggregation bug"),
    "BL-3 FPR~30%": (lambda s: not (0.20 < s["fpr_mean"] < 0.40),
                      "FPR not ~30% — Krum selection bug"),
})

# BL-4: FedAvg + CDP, eps=4, no attacks (~75-80%)
# Note: we run only eps=4 here; full sweep done in exp4
bl4 = run("bl4_cdp_eps4", {**COMMON,
    "alpha": 0.5, "aggregation": "fedavg", "m_assumed": 0,
    "use_dp": True, "epsilon": 4.0, "delta": 1e-5, "dp_accounting": "opacus",
    "attack": "none", "m_real": 0,
}, red_flags={
    "BL-4 MTA reasonable": (lambda s: s["mta_mean"] < 0.60,
                             "CDP eps=4 < 60% — DP noise too large or bug"),
})

# BL-5: Attacks without defense — verify attacks actually work
for attack, extra_kwargs, name_suffix, flags in [
    ("label_flip", {}, "labelflip",
     {"LF degrades": (lambda s: s["mta_mean"] > 0.76, "Label flip doesn't degrade accuracy")}),
    ("sign_flip", {}, "signflip",
     {"SF diverges": (lambda s: s["mta_mean"] > 0.50, "Sign flip: MTA > 50% — attack not working")}),
    ("alie", {}, "alie",
     {"ALIE degrades": (lambda s: s["mta_mean"] > 0.76, "ALIE doesn't degrade accuracy")}),
    ("backdoor", {"backdoor_gamma": 20.0}, "backdoor_gammaK",
     {"Backdoor ASR": (lambda s: s["asr_mean"] < 0.50, "Backdoor ASR < 50% — attack not working")}),
]:
    run(f"bl5_{name_suffix}_no_defense", {**COMMON,
        "alpha": 0.5, "aggregation": "fedavg", "m_assumed": 0,
        "use_dp": False, "epsilon": float("inf"),
        "attack": attack, "m_real": 4,
        **extra_kwargs,
    }, red_flags=flags)

# BL-6: Multi-Krum + Sign-Flip → TPR should be 100%
run("bl6_mkrum_signflip", {**COMMON,
    "alpha": 0.5, "aggregation": "multi_krum", "m_assumed": 6, "f": 14,
    "use_dp": False, "epsilon": float("inf"), "attack": "sign_flip", "m_real": 4,
}, red_flags={
    "BL-6 TPR=100%": (lambda s: s["tpr_mean"] < 0.99,
                       "Krum doesn't catch all sign-flip outliers — Krum bug"),
})

logger.info("\nALL PHASE 0 BASELINES DONE")
