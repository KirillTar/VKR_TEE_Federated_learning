"""
Speed benchmark: measure time per FL round and estimate total experiment time.
Runs BL-1 config for 5 rounds, reports timing.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from experiments.fl_simulator import FLSimulator, FLConfig

cfg = FLConfig(
    dataset="cifar10",
    data_dir="./data",
    alpha=10.0,
    seed=42,
    n_clients=20,
    n_rounds=5,
    local_epochs=5,
    batch_size=32,
    lr=0.01,
    aggregation="fedavg",
    m_assumed=0,
    use_dp=False,
    epsilon=float("inf"),
    attack="none",
    m_real=0,
    log_every=1,
    save_dir="./results/speed_test",
)

print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print("Running 5 rounds...")

t0 = time.time()
sim = FLSimulator(cfg)
t_init = time.time() - t0
print(f"Init (data download + split): {t_init:.1f}s")

t1 = time.time()
metrics = sim.run()
t_total = time.time() - t1

per_round = t_total / 5
estimated_200 = per_round * 200 / 60

print(f"\n=== Speed Results ===")
print(f"5 rounds took: {t_total:.1f}s")
print(f"Per round: {per_round:.1f}s")
print(f"Estimated 200 rounds: {estimated_200:.1f} min")
print(f"Estimated per 3-seed run: {estimated_200*3:.1f} min")
print(f"  CLAUDE.md estimate: ~15 min per run")

# Print final MTA
if metrics.records:
    last = metrics.records[-1]
    print(f"\nMTA after 5 rounds: {last['mta']:.4f} (expect low, model not converged yet)")
