"""
Main experiment runner.

Usage:
    python run_experiment.py --config configs/bl_1.yaml
    python run_experiment.py --config configs/exp1.yaml --seeds 0 1 2
    python run_experiment.py --config configs/exp1.yaml --override attack=alie epsilon=4
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.fl_simulator import FLSimulator, FLConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


def load_config(config_path: str, overrides: list[str] = None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for override in overrides:
            key, value = override.split("=", 1)
            # Try to cast to int/float/bool
            for cast in (int, float):
                try:
                    value = cast(value)
                    break
                except ValueError:
                    pass
            if value == "true":
                value = True
            elif value == "false":
                value = False
            cfg[key] = value
    return cfg


def run_seeds(cfg_dict: dict, seeds: list[int], save_dir: str) -> None:
    """Run experiment across multiple seeds and save results.

    When called with a single seed (parallel mode), writes to per-seed files
    ({name}_seed{N}.csv) to avoid race conditions when multiple processes run
    the same experiment in parallel. When called with multiple seeds (sequential
    mode), writes to the combined {name}.csv as before.
    """
    import pandas as pd

    all_records = []

    for seed in seeds:
        cfg_dict["seed"] = seed
        cfg = FLConfig(**{k: v for k, v in cfg_dict.items() if k in FLConfig.__dataclass_fields__})
        cfg.save_dir = save_dir

        logger.info(f"=== Seed {seed} ===")
        sim = FLSimulator(cfg)
        metrics = sim.run()
        all_records.extend(metrics.records)

    os.makedirs(save_dir, exist_ok=True)
    exp_name = cfg_dict.get("name", "experiment")

    # Use per-seed filename when running a single seed to avoid overwrite races
    single_seed = len(seeds) == 1
    suffix = f"_seed{seeds[0]}" if single_seed else ""

    out_path = os.path.join(save_dir, f"{exp_name}{suffix}.csv")
    df = pd.DataFrame(all_records)
    df.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")

    last = df[df["round"] == df["round"].max()]
    summary = {}
    for col in ["mta", "asr", "tpr", "fpr"]:
        if col in last.columns:
            summary[f"{col}_mean"] = round(float(last[col].mean()), 4)
            summary[f"{col}_std"] = round(float(last[col].std()), 4) if len(last) > 1 else 0.0

    logger.info("=== Summary (final round) ===")
    for k, v in summary.items():
        logger.info(f"  {k}: {v:.4f}")

    summary_path = os.path.join(save_dir, f"{exp_name}{suffix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="FL Experiment Runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values (e.g. --override attack=alie epsilon=4)")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    cfg_dict = load_config(args.config, args.override)

    save_dir = args.save_dir or cfg_dict.get("save_dir", "./results")
    run_seeds(cfg_dict, seeds=args.seeds, save_dir=save_dir)


if __name__ == "__main__":
    main()
