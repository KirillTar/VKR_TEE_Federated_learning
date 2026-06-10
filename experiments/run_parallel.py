"""
Launch multiple FL experiment processes in parallel (different seeds or configs).
Each process is fully independent — no shared memory, no CUDA context sharing.
The GPU handles multiple concurrent training jobs via hardware multi-tasking.

Usage:
    # Run BL-1 with 3 seeds in parallel:
    python run_parallel.py --config configs/bl_1_fedavg_iid.yaml --seeds 0 1 2

    # Run multiple configs in parallel (e.g. different attacks):
    python run_parallel.py --configs configs/bl_5_sign_flip.yaml configs/bl_5_alie.yaml --seeds 0 1 2

    # Run with override:
    python run_parallel.py --config configs/exp4_epsilon_sweep.yaml \
        --override_grid epsilon=1,2,4,8 attack=alie,label_flip --seeds 0 1 2
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

PYTHON = str(Path(__file__).parent.parent / ".venv" / "bin" / "python")
RUNNER = str(Path(__file__).parent / "run_experiment.py")


def launch(config: str, seed: int, overrides: list[str], save_dir: str) -> subprocess.Popen:
    # Resolve config to absolute path before changing cwd in subprocess
    config = str(Path(config).resolve())
    cmd = [PYTHON, RUNNER, "--config", config, "--seeds", str(seed), "--save_dir", save_dir]
    if overrides:
        cmd += ["--override"] + overrides
    # If 'name' is overridden, use it as the log prefix (cleaner filenames)
    log_name = Path(config).stem
    other_overrides = []
    for ov in overrides:
        k, v = ov.split("=", 1)
        if k == "name":
            log_name = v
        else:
            other_overrides.append(ov)
    if other_overrides:
        log_name += "_" + "_".join(o.replace("=", "") for o in other_overrides)
    log_name += f"_seed{seed}.log"
    log_path = Path(save_dir) / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Launching: {Path(config).name} seed={seed} overrides={overrides}")
    with open(log_path, "a") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(Path(__file__).parent.parent)
        )
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Single config file")
    parser.add_argument("--configs", nargs="+", help="Multiple config files (run each in parallel)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--override_grid", nargs="*", default=[], metavar="KEY=v1,v2,...",
                        help="Grid of overrides: runs cartesian product")
    parser.add_argument("--sweep", nargs="+", metavar="KEY=V1,KEY=V2,...",
                        help="Paired sweep: each item is comma-separated KEY=VAL pairs "
                             "for one config point, e.g. --sweep epsilon=1,name=bl_eps1 epsilon=2,name=bl_eps2")
    parser.add_argument("--save_dir", default="./results")
    parser.add_argument("--max_parallel", type=int, default=6,
                        help="Max number of simultaneous processes")
    args = parser.parse_args()

    configs = args.configs if args.configs else [args.config]

    # Build override combinations
    if args.sweep:
        # Each sweep item is a comma-separated list of KEY=VAL pairs for one config point
        override_combos = [item.split(",") for item in args.sweep]
    elif args.override_grid:
        grid_keys = []
        grid_values = []
        for item in args.override_grid:
            k, vals = item.split("=")
            grid_keys.append(k)
            grid_values.append(vals.split(","))
        override_combos = [
            [f"{k}={v}" for k, v in zip(grid_keys, combo)]
            for combo in itertools.product(*grid_values)
        ]
    else:
        override_combos = [args.override] if args.override else [[]]

    # Build all jobs
    jobs = []
    for config in configs:
        for seed in args.seeds:
            for ov in override_combos:
                jobs.append((config, seed, ov))

    print(f"Total jobs: {len(jobs)} (max {args.max_parallel} parallel)")

    # Run with limited parallelism
    running: list[subprocess.Popen] = []
    for i, (config, seed, ov) in enumerate(jobs):
        while len(running) >= args.max_parallel:
            running = [p for p in running if p.poll() is None]
            time.sleep(2)

        proc = launch(config, seed, ov, args.save_dir)
        running.append(proc)

    # Wait for all
    for p in running:
        p.wait()

    print(f"\nAll {len(jobs)} jobs done.")

    # Aggregate per-seed results into combined summaries
    _aggregate_per_seed_results(configs, override_combos, args.seeds, args.save_dir)


TAIL_ROUNDS = 5  # number of last evaluation points used for stable-phase averaging
SKIP_COLS = {"round", "seed", "clip_S"}


def _aggregate_per_seed_results(
    configs: list[str],
    override_combos: list[list[str]],
    seeds: list[int],
    save_dir: str,
) -> None:
    """Read per-seed CSV/JSON files and write combined summary for each experiment.

    Metrics are averaged over the last TAIL_ROUNDS evaluation points (stable phase),
    not just the final round. Per-round TPR/FPR are noisy because they depend on
    which clients were sampled in that specific round; the tail-mean is far more
    representative. All numeric columns from the CSV (mta, asr, tpr, fpr,
    source_acc, ...) are aggregated automatically.
    """
    import math

    import pandas as pd

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    def _pstd(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = _mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    for config in configs:
        for ov in override_combos:
            with open(config) as f:
                cfg = yaml.safe_load(f)
            exp_name = cfg.get("name", Path(config).stem)
            for item in ov:
                k, v = item.split("=", 1)
                if k == "name":
                    exp_name = v

            dfs = []
            seed_finals: dict[str, list[float]] = {}
            for seed in seeds:
                csv_path = Path(save_dir) / f"{exp_name}_seed{seed}.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                dfs.append(df)
                tail = df.tail(TAIL_ROUNDS)
                for col in df.columns:
                    if col in SKIP_COLS:
                        continue
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        continue
                    seed_finals.setdefault(col, []).append(float(tail[col].mean()))

            if not dfs:
                continue

            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_csv(Path(save_dir) / f"{exp_name}.csv", index=False)

            summary = {}
            for col, vals in seed_finals.items():
                if vals:
                    summary[f"{col}_mean"] = round(_mean(vals), 4)
                    summary[f"{col}_std"] = round(_pstd(vals), 4)
            summary["n_seeds"] = len(dfs)
            summary["tail_rounds"] = TAIL_ROUNDS

            with open(Path(save_dir) / f"{exp_name}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(f"  Aggregated {len(dfs)} seeds (tail={TAIL_ROUNDS}) → {exp_name}_summary.json")
            for k, v in summary.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
