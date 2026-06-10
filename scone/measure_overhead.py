"""
Measure per-round wall-clock of the aggregation hot path.

Same script can run:
  - native CPU (python scone/measure_overhead.py)
  - inside SCONE SIM/HW container (CMD-override on sconified image)

Output: JSON line per round + final summary (median, p95, mean) → stdout.
No real Flower / network — only the aggregation+DP path that lives inside the enclave.
"""
import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

# Allow import from project root when run from /app inside container or repo root locally
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from core.aggregation import aggregate
from core.dp import adaptive_clip, add_gaussian_noise

# compute_sigma needs opacus; fallback to pre-computed values when opacus unavailable
# (e.g. inside SCONE images that don't ship opacus). The exact sigma only affects
# noise magnitude; timing of the noise addition is identical.
def _sigma_fallback(target_epsilon, target_delta, T, sample_rate=1.0, **_):
    return 13.14 if sample_rate >= 0.5 else 1.71

try:
    from core.dp import compute_sigma as _compute_sigma_real
    _ = _compute_sigma_real(4.0, 1e-5, 200, 0.8)  # probe
    compute_sigma = _compute_sigma_real
except Exception:
    compute_sigma = _sigma_fallback


def cur_rss_mib():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def synth_round(n_clients, d, rng):
    """Generate n_clients random updates of dimension d."""
    return [
        torch.from_numpy(rng.standard_normal(d, dtype=np.float32).astype(np.float32))
        for _ in range(n_clients)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["cs", "cd"], default="cs",
                   help="cs=cross-silo SVHNCNN d=89834, cd=cross-device MicroMNIST d=4266")
    p.add_argument("--aggregation", default="multi_krum",
                   help="multi_krum | fedavg | fedrola | trimmed_mean")
    p.add_argument("--n_clients", type=int, default=40)
    p.add_argument("--m_assumed", type=int, default=12)
    p.add_argument("--epsilon", type=float, default=4.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--n_rounds", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="native", help="tag for the output line")
    args = p.parse_args()

    d = 89834 if args.scenario == "cs" else 4266
    f = args.n_clients - args.m_assumed
    sample_rate = 0.8 if args.scenario == "cs" else 0.2

    sigma = compute_sigma(
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        T=200,
        sample_rate=sample_rate,
    )

    print(json.dumps({
        "event": "config",
        "label": args.label,
        "scenario": args.scenario,
        "aggregation": args.aggregation,
        "d": d,
        "n_clients": args.n_clients,
        "m_assumed": args.m_assumed,
        "f": f,
        "sigma": sigma,
        "n_rounds": args.n_rounds,
        "warmup": args.warmup,
    }), flush=True)

    rng = np.random.default_rng(args.seed)
    agg_cfg = {"m_assumed": args.m_assumed, "f": f}

    if args.aggregation == "fedrola":
        from core.models import get_model
        if args.scenario == "cs":
            _model = get_model("svhn")
        else:
            _model = get_model("fashion_mnist", model_name="micro")
        layer_sizes = [p.numel() for p in _model.parameters()]
        assert sum(layer_sizes) == d, f"layer_sizes sum {sum(layer_sizes)} != d {d}"
        agg_cfg["layer_sizes"] = layer_sizes
        agg_cfg["chi"] = 0.1
        agg_cfg["client_ids"] = list(range(args.n_clients))

    times = []

    total_rounds = args.warmup + args.n_rounds
    for r in range(total_rounds):
        gc.collect()
        flat_updates = synth_round(args.n_clients, d, rng)

        t0 = time.perf_counter()
        clipped, S = adaptive_clip(flat_updates)
        result = aggregate(clipped, args.aggregation, agg_cfg)
        agg = result.update
        if sigma > 0:
            agg = add_gaussian_noise(agg, sigma, S, f=f)
        # force materialization
        _ = float(agg.sum().item())
        t1 = time.perf_counter()

        dt_ms = (t1 - t0) * 1000.0
        is_warmup = r < args.warmup
        if not is_warmup:
            times.append(dt_ms)

        print(json.dumps({
            "event": "round",
            "label": args.label,
            "round": r,
            "warmup": is_warmup,
            "dt_ms": round(dt_ms, 3),
            "rss_mib": round(cur_rss_mib(), 1),
            "peak_rss_mib": round(peak_rss_mib(), 1),
        }), flush=True)

    arr = np.array(times)
    print(json.dumps({
        "event": "summary",
        "label": args.label,
        "scenario": args.scenario,
        "aggregation": args.aggregation,
        "n_rounds_measured": len(times),
        "mean_ms": round(float(arr.mean()), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "std_ms": round(float(arr.std()), 3),
        "peak_rss_mib": round(peak_rss_mib(), 1),
    }), flush=True)


if __name__ == "__main__":
    main()
