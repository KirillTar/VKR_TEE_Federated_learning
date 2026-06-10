"""
Correctness test: compare sequential vs parallel simulator.
Same seed, same config → deltas should be identical (within float32 precision).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from experiments.fl_simulator import FLSimulator, FLConfig
from experiments.fl_simulator_parallel import FLSimulatorParallel

CFG = dict(
    dataset="cifar10", data_dir="./data", alpha=0.5, seed=123,
    n_clients=20, n_rounds=1, local_epochs=5, batch_size=32, lr=0.01,
    momentum=0.9, weight_decay=1e-4, aggregation="multi_krum", m_assumed=6, f=14,
    use_dp=False, epsilon=float("inf"), attack="none", m_real=0, log_every=1,
)

print("=== Sequential (1 round) ===")
t0 = time.time()
m_seq = FLSimulator(FLConfig(**CFG)).run()
t_seq = time.time() - t0
mta_seq = m_seq.records[-1]["mta"]
print(f"  MTA={mta_seq:.5f}  time={t_seq:.1f}s")

print("\n=== Parallel (1 round, 8 workers) ===")
t0 = time.time()
m_par = FLSimulatorParallel(FLConfig(**CFG), n_workers=8).run()
t_par = time.time() - t0
mta_par = m_par.records[-1]["mta"]
print(f"  MTA={mta_par:.5f}  time={t_par:.1f}s")

print(f"\nMTA difference: {abs(mta_seq - mta_par):.6f}")
print(f"Speedup: {t_seq/t_par:.1f}×")
print(f"Extrapolated 200 rounds → par: {t_par*200/60:.1f} min")
