"""Speed benchmark: sequential vs parallel simulator."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from experiments.fl_simulator import FLSimulator, FLConfig
from experiments.fl_simulator_parallel import FLSimulatorParallel

BASE_CFG = dict(
    dataset="cifar10", data_dir="./data", alpha=10.0, seed=42,
    n_clients=20, n_rounds=3, local_epochs=5, batch_size=32, lr=0.01,
    momentum=0.9, weight_decay=1e-4, aggregation="fedavg", m_assumed=0,
    use_dp=False, epsilon=float("inf"), attack="none", m_real=0, log_every=1,
)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CPU cores: {__import__('os').cpu_count()}")

# Sequential
cfg_seq = FLConfig(**BASE_CFG)
t0 = time.time()
FLSimulator(cfg_seq).run()
t_seq = time.time() - t0
print(f"\nSequential: {t_seq:.1f}s for 3 rounds → {t_seq/3:.1f}s/round → {t_seq/3*200/60:.1f} min/200 rounds")

# Parallel
for n_workers in [4, 8, 20]:
    cfg_par = FLConfig(**BASE_CFG)
    t0 = time.time()
    FLSimulatorParallel(cfg_par, n_workers=n_workers).run()
    t_par = time.time() - t0
    speedup = t_seq / t_par
    print(f"Parallel({n_workers} workers): {t_par:.1f}s → {t_par/3:.1f}s/round → "
          f"{t_par/3*200/60:.1f} min/200 rounds  (speedup: {speedup:.1f}×)")
