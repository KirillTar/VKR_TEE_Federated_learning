# Федеративное обучение с FedRoLA + Central DP + TEE

Экспериментальная реализация к ВКР. Эшелонированная защита для FL:
**FedRoLA-PCSI** (робастная агрегация) → **Central DP** (record- или client-level)
→ **TEE** (SCONE/SGX для integrity и confidentiality серверной стороны).

Два сценария:

| | Cross-device (CD) | Cross-silo (CS) |
|---|---|---|
| Датасет / модель | Fashion-MNIST / MicroMNIST (d=4266) | SVHN / SVHNCNN (d=89834) |
| N / q / E | 200 / 0.2 / 5 | 50 / 0.8 / 1 |
| DP | client-level | record-level (R=5) |
| m_real (доля вредоносных) | 80 (40%) | 20 (40%) |

## Структура репозитория

```
├── core/                       Device-agnostic код FL
│   ├── models.py               Архитектуры (MicroMNIST, SVHNCNN, …)
│   ├── data.py                 Загрузка + Dirichlet split
│   ├── aggregation.py          FedAvg, Multi-Krum, TrimmedMean, FedRoLA-PCSI
│   ├── dp.py                   Norm clipping, Gaussian noise, RDP accounting
│   ├── attacks.py              ALIE, IPM, Fang, Min-Max, Backdoor, LabelFlip
│   └── metrics.py              MTA, ASR, TPR, FPR
│
├── experiments/                GPU-симуляция
│   ├── fl_simulator.py         FL-loop (FLConfig, FLSimulator)
│   ├── run_experiment.py       Запуск по YAML + --override
│   ├── run_parallel.py         Параллельные seeds/sweeps (max_parallel=6)
│   ├── gia.py                  DLG — gradient inversion demo
│   ├── mia.py                  Membership Inference Attack (Yeom 2018)
│   ├── configs/                YAML-конфиги (cross_device/, cross_silo/)
│   └── EXPERIMENTS.md          Реестр проведённых прогонов + команды
│
├── scone/                      SCONE-деплоймент (TEE overhead)
│   ├── server.py, client.py    Flower server/client
│   ├── Dockerfile
│   └── scone_build.sh          Сконификация (SIM | HW)
│
├── analysis/                   Графики и таблицы
│   ├── plots.py                Стандартные графики
│   ├── plot_gia.py             Визуализация GIA
│   ├── snr_vs_mta.py           SNR диагностика
│   ├── tables.py               LaTeX-таблицы
│   └── notebooks/
│
├── requirements/
│   ├── gpu.txt                 RTX 5070Ti / WSL2 / CUDA 12.8 / torch nightly
│   └── scone.txt               CPU-only для SCONE-контейнера
│
└── results/                    Артефакты прогонов (в git-ignored)
```

## Установка

### GPU-стенд (ML-эксперименты)

Требуется WSL2 + CUDA 12.8 + GPU с поддержкой sm_120 (Blackwell).
Для других GPU — замените torch nightly на stable.

```bash
python3.12 -m venv .venv
.venv/bin/pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
.venv/bin/pip install -r requirements/gpu.txt
```

### SCONE-стенд (TEE overhead)

```bash
python3.11 -m venv .venv_scone
.venv_scone/bin/pip install -r requirements/scone.txt
# Либо собрать образ через scone/scone_build.sh
```

## Быстрый старт

```bash
# 1 эксперимент: FedRoLA + ALIE на CD, 3 seeds
.venv/bin/python experiments/run_experiment.py \
    --config experiments/configs/cross_device/exp1_base.yaml \
    --seeds 0 1 2 \
    --override attack=alie m_real=80 aggregation=fedrola name=quickstart

# Параллельный sweep по epsilon на CS
.venv/bin/python experiments/run_parallel.py \
    --config experiments/configs/cross_silo/exp1_base.yaml \
    --sweep "epsilon=2,name=qs_eps2" "epsilon=4,name=qs_eps4" "epsilon=8,name=qs_eps8" \
    --seeds 0 1 2 \
    --save_dir ./results/quickstart \
    --max_parallel 3
```

Выходы пишутся в `--save_dir`:
- `<name>_seed<N>.csv` — per-round MTA/TPR/FPR/…
- `<name>_seed<N>.log` — stdout/stderr симулятора
- `<name>_summary.json` — сводка (MTA/ASR/AUC на конец обучения, через нескольких seeds)

## Воспроизведение экспериментов главы 5

Полный реестр прогонов (Phase 2–6: baselines, sensitivity, privacy) с командами
для воспроизведения — `experiments/EXPERIMENTS.md`.

## Формат `--override`

- Любой ключ из `FLConfig` (`experiments/fl_simulator.py`) или YAML-базы.
- Типизация автоматическая: `true`/`false` → bool, числа → int/float.
- Синтаксис `--sweep` (run_parallel.py) — список строк вида
  `"key1=v1,key2=v2,...,name=<run_id>"`; каждая строка запускается × `--seeds`.

Ключевые override-поля:

| Ключ | Значения |
|---|---|
| `attack` | `none`, `alie`, `ipm`, `fang`, `minmax`, `backdoor`, `random_label_flip`, `label_flip` (CS only) |
| `aggregation` | `fedavg`, `multi_krum`, `trimmed_mean`, `median`, `fedrola` |
| `use_dp` / `dp_mode` / `dp_level` | `true`/`false` • `central`/`local` • `client`/`record` |
| `epsilon`, `delta`, `clip_max`, `record_clip` | DP-параметры |
| `m_real`, `m_assumed`, `f` | Количество вредоносных / предполагаемых / трим-фактор |
| `n_rounds`, `lr`, `lr_end`, `alpha` | Длина обучения, LR decay, non-IID концентрация |
| `fedrola_chi`, `fedrola_full_layer` | FedRoLA-гиперпараметры |


## Железо / окружение разработки

- RTX 5070Ti (Blackwell, sm_120) через WSL2, Python 3.12, torch nightly cu128
- Опции параллелизма: `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=2`, `max_parallel=6`
- Типичная скорость: CD FedRoLA ~5 c/раунд, CS DP ~4.5 run/ч (T=200)
