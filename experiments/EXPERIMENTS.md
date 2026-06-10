# Реестр экспериментов

Полная история прогонов итерации 2, замещающая удалённые shell-скрипты.
Каждый раздел содержит: цель, параметры, команду запуска, директорию вывода.

Общие параметры окружения для всех phase-прогонов:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PYTHON=".venv/bin/python"
RUNNER="experiments/run_parallel.py"
MAX_P=6                                    # параллельных процессов
SEEDS="0 1 2"                              # стандартный набор
CD_BASE="experiments/configs/cross_device/exp1_base.yaml"
CS_BASE="experiments/configs/cross_silo/exp1_base.yaml"
```

Запуск одиночного эксперимента — `run_experiment.py`; запуск сетки (sweep) — `run_parallel.py`.

---

## Сводная таблица фаз

| Фаза | Назначение | Runs | Время (wall) | Выход (results/) | Логи |
|---|---|---|---|---|---|
| Phase 1–2 `exp1`/`exp2` | Baselines + attack × defense matrix (CD+CS) | ~170 | несколько суток | `cross_device/cd_exp1_*`, `cross_silo/cs_exp1_*`, `*_exp2_*` | — |
| Phase 2 diagnostics | S-кривая CD, full-layer, clip_max, m_real, instrumented | 111 | ~12 ч | `cross_device/cd_diag_*`, `cross_silo/cs_diag_*` | `phase2_diagnostics.log` |
| Phase 3 sensitivity | CS S-кривая, m_real ALIE, ablation, α sweep | 117 | ~13 ч | `cross_device/cd_p3_*`, `cross_silo/cs_p3_*` | `phase3_sensitivity.log` |
| Phase 4 | m_real gap-fill + α sweep + (устаревший) MIA CD | 75 | ~10 ч | `cross_device/cd_exp3_*`, `cs_exp3_*`, `cd_exp5_*`, `cs_exp5_*` | `archive/old_logs/phase4.log` |
| Phase 5b | **CS MIA N=10** (centralized + FL no-DP + FL DP ε=4) | 8 | ~14 ч | `mia_n10/` | `phase5b_mia_n10.log` |
| Phase 5c | Local vs Central DP (CS) + GIA demo | 5 | ~6 ч | `local_dp/` | `phase5c_local_dp.log` |
| Phase 5d | T=500 + LR decay probe (CS) | 8 | ~11 ч | `t500_probe/` | `phase5d_t500_probe.log` |
| Phase 5e | T=500 extra seeds + CD Local DP + GIA rerun | 9 | ~6 ч | `t500_probe/`, `local_dp_cd/`, `local_dp/gia/` | `phase5e_t500_gia.log` |
| Phase 6 | Тезисы 2/3/5 gap-fills | 15 | ~7 ч | `local_dp_attack/`, `thesis5_neg/`, `local_dp_cd/` | `phase6_tomorrow.log` |

Не вошли в главу 5 (см. `archive/README.md`): phase 6c/6d (CD MIA N=20/50 — модель не обучилась под DP).

---

## Phase 2 diagnostics (завершено)

**Цель:** закрыть открытые вопросы Phase 2 (S-кривая CD, full-layer FedRoLA, clip_max sweep, m_real backdoor, instrumented re-runs с новыми FedRoLA-полями).

### Section A — CD ε gap-fill (42 runs)
ε ∈ {1.5, 2, 3, 4, 5, 6, 8} × {none, alie} × 3 seeds, FedRoLA+DP.

```bash
A_SWEEP=()
for EPS in 1.5 2 3 4 5 6 8; do
  for ATK in none alie; do
    MR=80; [[ "$ATK" == "none" ]] && MR=0
    A_SWEEP+=("epsilon=${EPS},attack=${ATK},m_real=${MR},aggregation=fedrola,use_dp=true,name=cd_diag_eps${EPS}_${ATK}")
  done
done
$PYTHON $RUNNER --config $CD_BASE --sweep "${A_SWEEP[@]}" --seeds 0 1 2 \
  --save_dir ./results/cross_device --max_parallel 6
```

### Section B — FedRoLA full-layer sanity (6 runs)
Backdoor, fedrola_full_layer=true, CD (no DP) + CS (DP).

```bash
$PYTHON $RUNNER --config $CD_BASE \
  --sweep "attack=backdoor,m_real=80,aggregation=fedrola,use_dp=false,fedrola_full_layer=true,name=cd_diag_backdoor_fedrola_full" \
  --seeds 0 1 2 --save_dir ./results/cross_device --max_parallel 6

$PYTHON $RUNNER --config $CS_BASE \
  --sweep "attack=backdoor,m_real=20,aggregation=fedrola,use_dp=true,fedrola_full_layer=true,name=cs_diag_backdoor_fedrola_full_dp" \
  --seeds 0 1 2 --save_dir ./results/cross_silo --max_parallel 6
```

### Section C — clip_max sweep против backdoor (24 runs)
clip_max ∈ {0.3, 0.5, 1.0, 1.5} × backdoor × FedRoLA+DP × 3 seeds × {CD, CS}.

```bash
# CD (m_real=80)
for CM in 0.3 0.5 1.0 1.5; do
  SWEEP+=("attack=backdoor,m_real=80,aggregation=fedrola,use_dp=true,clip_max=${CM},name=cd_diag_backdoor_clip${CM}_fedrola_dp")
done
$PYTHON $RUNNER --config $CD_BASE --sweep "${SWEEP[@]}" --seeds 0 1 2 \
  --save_dir ./results/cross_device --max_parallel 6
# CS (m_real=20) — аналогично, с cs_diag_backdoor_clip*
```

### Section D — m_real backdoor sweep CD (12 runs)
m_real ∈ {20, 40, 60, 80} × FedRoLA+DP × 3 seeds.

```bash
for MR in 20 40 60 80; do
  SWEEP+=("attack=backdoor,m_real=${MR},aggregation=fedrola,use_dp=true,name=cd_diag_backdoor_m${MR}_fedrola_dp")
done
$PYTHON $RUNNER --config $CD_BASE --sweep "${SWEEP[@]}" --seeds 0 1 2 \
  --save_dir ./results/cross_device --max_parallel 6
```

### Section E — Instrumented re-runs (27 runs)
9 ключевых конфигов с новым FedRoLA-логом (frola_n_detected, frola_f_eff, frola_score_*, frola_layers, frola_disc).

```bash
# CD (5 конфигов)
E_CD=(
  "attack=backdoor,m_real=80,aggregation=fedrola,use_dp=false,name=cd_diag_inst_backdoor_fedrola"
  "attack=backdoor,m_real=80,aggregation=fedrola,use_dp=true,name=cd_diag_inst_backdoor_fedrola_dp"
  "attack=alie,m_real=80,aggregation=fedrola,use_dp=false,name=cd_diag_inst_alie_fedrola"
  "epsilon=15,attack=alie,m_real=80,aggregation=fedrola,use_dp=true,name=cd_diag_inst_alie_fedrola_dp_eps15"
  "epsilon=20,attack=alie,m_real=80,aggregation=fedrola,use_dp=true,name=cd_diag_inst_alie_fedrola_dp_eps20"
)
$PYTHON $RUNNER --config $CD_BASE --sweep "${E_CD[@]}" --seeds 0 1 2 \
  --save_dir ./results/cross_device --max_parallel 6

# CS (4 конфига)
E_CS=(
  "attack=backdoor,m_real=20,aggregation=fedrola,use_dp=true,name=cs_diag_inst_backdoor_fedrola_dp"
  "epsilon=4,attack=alie,m_real=20,aggregation=fedrola,use_dp=true,name=cs_diag_inst_alie_fedrola_dp_eps4"
  "epsilon=4,attack=none,m_real=0,aggregation=fedrola,use_dp=true,name=cs_diag_inst_clean_fedrola_dp_eps4"
  "epsilon=2,attack=none,m_real=0,aggregation=fedrola,use_dp=true,name=cs_diag_inst_clean_fedrola_dp_eps2"
)
$PYTHON $RUNNER --config $CS_BASE --sweep "${E_CS[@]}" --seeds 0 1 2 \
  --save_dir ./results/cross_silo --max_parallel 6
```

---

## Phase 3 sensitivity (завершено)

**Цель:** чувствительность к non-IID, m_real, ablation защит.

### Section A2 — повтор CD m_real backdoor (12 runs)
Идентично Phase 2 Section D (сделано заново для консистентности naming).

### Section A3 — повтор instrumented re-runs (27 runs)
Идентично Phase 2 Section E.

### Section B — CS ε dense grid (30 runs)
ε ∈ {0.5, 0.75, 1, 1.5, 3} × {none, alie} × 3 seeds, FedRoLA+DP.

```bash
B_SWEEP=()
for EPS in 0.5 0.75 1 1.5 3; do
  for ATK in none alie; do
    MR=20; [[ "$ATK" == "none" ]] && MR=0
    B_SWEEP+=("epsilon=${EPS},attack=${ATK},m_real=${MR},aggregation=fedrola,use_dp=true,name=cs_p3_eps${EPS}_${ATK}")
  done
done
$PYTHON $RUNNER --config $CS_BASE --sweep "${B_SWEEP[@]}" --seeds 0 1 2 \
  --save_dir ./results/cross_silo --max_parallel 6
```

### Section D — CD m_real ALIE sweep (12 runs)
m_real ∈ {20, 40, 60, 80} × ALIE × FedRoLA+DP × 3 seeds.

```bash
for MR in 20 40 60 80; do
  D_SWEEP+=("attack=alie,m_real=${MR},aggregation=fedrola,use_dp=true,name=cd_p3_alie_m${MR}_fedrola_dp")
done
$PYTHON $RUNNER --config $CD_BASE --sweep "${D_SWEEP[@]}" --seeds 0 1 2 \
  --save_dir ./results/cross_device --max_parallel 6
```

### Section E — Ablation CD (36 runs)
4 комбинации защит × 3 атаки × 3 seeds.
Защиты: FedRoLA-only, DP-only (FedAvg+DP), FedRoLA+DP, neither (FedAvg, no DP).
Атаки: ALIE, backdoor, IPM.

```bash
E_SWEEP=()
for ATK in alie backdoor ipm; do
  E_SWEEP+=("attack=${ATK},m_real=80,aggregation=fedrola,use_dp=false,name=cd_p3_ablation_${ATK}_fedrola_nodp")
  E_SWEEP+=("attack=${ATK},m_real=80,aggregation=fedavg,use_dp=true,name=cd_p3_ablation_${ATK}_fedavg_dp")
  E_SWEEP+=("attack=${ATK},m_real=80,aggregation=fedrola,use_dp=true,name=cd_p3_ablation_${ATK}_fedrola_dp")
  E_SWEEP+=("attack=${ATK},m_real=80,aggregation=fedavg,use_dp=false,name=cd_p3_ablation_${ATK}_fedavg_nodp")
done
$PYTHON $RUNNER --config $CD_BASE --sweep "${E_SWEEP[@]}" --seeds 0 1 2 \
  --save_dir ./results/cross_device --max_parallel 6
```

---

## Phase 4 (завершено; часть данных уехала в архив)

**Цель:** добить m_real gap-fill, α (non-IID) sweep, первый MIA (CD).

### Section A — m_real ALIE gap-fill (18 runs)
CD: m_real ∈ {120, 160} (60%, 80%); CS: m_real ∈ {10, 20, 30, 40} (20-80% от N=50).

```bash
# CD
for MR in 120 160; do
  CD_A+=("attack=alie,m_real=${MR},aggregation=fedrola,use_dp=true,name=cd_exp3_alie_m${MR}_fedrola_dp")
done
$PYTHON $RUNNER --config $CD_BASE --sweep "${CD_A[@]}" --seeds 0 1 2 \
  --save_dir ./results/cross_device --max_parallel 6

# CS
for MR in 10 20 30 40; do
  CS_A+=("attack=alie,m_real=${MR},aggregation=fedrola,use_dp=true,name=cs_exp3_alie_m${MR}_fedrola_dp")
done
$PYTHON $RUNNER --config $CS_BASE --sweep "${CS_A[@]}" --seeds 0 1 2 \
  --save_dir ./results/cross_silo --max_parallel 6
```

### Section B — α Dirichlet sweep (48 runs)
α ∈ {0.1, 0.3, 1.0, 10.0} × {none, alie} × FedRoLA+DP × 3 seeds × {CD, CS}.

```bash
# CD (CS аналогично, с cs_exp5_a*, m_real=20/0)
for ALPHA in 0.1 0.3 1.0 10.0; do
  A_TAG=$(echo $ALPHA | tr '.' 'p')
  for ATK in none alie; do
    MR=80; [[ "$ATK" == "none" ]] && MR=0
    CD_B+=("alpha=${ALPHA},attack=${ATK},m_real=${MR},aggregation=fedrola,use_dp=true,name=cd_exp5_a${A_TAG}_${ATK}")
  done
done
$PYTHON $RUNNER --config $CD_BASE --sweep "${CD_B[@]}" --seeds 0 1 2 \
  --save_dir ./results/cross_device --max_parallel 6
```

### Section C — устаревший CD MIA
Замещён CS MIA (Phase 5b). Результаты перенесены в `archive/abandoned_cd_mia/mia_phase4/`.

---

## Phase 5b — CS MIA N=10 (завершено)

**Цель:** обосновать DP эмпирически (MIA AUC без DP vs с DP).

Конфиг: `experiments/configs/cross_silo/mia_n10.yaml` (N=10, q=1.0, E=10, T=100, SVHN, SVHNCNN).

### Section A — Centralized baseline (200 epochs)
```bash
$PYTHON experiments/mia.py \
  --config experiments/configs/cross_silo/mia_n10.yaml \
  --seeds 0 1 \
  --centralized --centralized_epochs 200 \
  --save_dir ./results/mia_n10/centralized
```

### Section B — FL N=10, no DP
```bash
$PYTHON experiments/mia.py \
  --config experiments/configs/cross_silo/mia_n10.yaml \
  --seeds 0 1 --no_dp_only \
  --save_dir ./results/mia_n10/fl_no_dp
```

### Section C — FL N=10, DP ε=4
```bash
$PYTHON experiments/mia.py \
  --config experiments/configs/cross_silo/mia_n10.yaml \
  --seeds 0 1 --epsilons 4 --skip_no_dp \
  --save_dir ./results/mia_n10/fl_dp_eps4
```

**Результат:** AUC≈0.49–0.51 во всех трёх конфигурациях — модель не переобучается; MIA ≈ random. Для ВКР — график 0.47–0.52 + цитаты (Naseri et al., Yeom et al., Nasr et al.).

---

## Phase 5c — Local vs Central DP + GIA (завершено)

**Цель:** эмпирика Тезиса 2 (Central DP > Local DP) на CS; визуальная демонстрация GIA.

### Section A — Central DP ε=4, FedRoLA, no attack (2 seeds)
```bash
for SEED in 0 1; do
  $PYTHON experiments/run_experiment.py --config $CS_BASE --seeds $SEED \
    --override name=local_dp_central aggregation=fedrola attack=none m_real=0 \
               use_dp=true epsilon=4.0 dp_mode=central save_dir=./results/local_dp
done
```

### Section B — Local DP ε=4 (то же, dp_mode=local) (2 seeds)
```bash
for SEED in 0 1; do
  $PYTHON experiments/run_experiment.py --config $CS_BASE --seeds $SEED \
    --override name=local_dp_local aggregation=fedrola attack=none m_real=0 \
               use_dp=true epsilon=4.0 dp_mode=local save_dir=./results/local_dp
done
```

### Section C — GIA demonstration (DLG, 5 images)
```bash
$PYTHON experiments/gia.py --dataset svhn --model svhn_cnn \
  --n_images 5 --noise_std_dp 0.1 --save_dir ./results/local_dp/gia
```

---

## Phase 5d — T=500 + LR decay probe (завершено)

**Цель:** проверить, улучшает ли длинное обучение + LR decay MTA под DP (inspired by DP-BREM T=2000).

Все конфиги: CS FedRoLA × {ALIE, MinMax} × {no_dp, ε=4} × 2 seeds.
Добавлен параметр `lr_end=0.001` (линейный decay с 0.01 за T раундов).

```bash
COMMON="aggregation=fedrola m_real=20 n_rounds=500 lr_end=0.001 save_dir=./results/t500_probe"

for SEED in 0 1; do
  # ALIE no_dp / dp
  $PYTHON experiments/run_experiment.py --config $CS_BASE --seeds $SEED --override \
    name=t500_alie_fedrola attack=alie use_dp=false $COMMON
  $PYTHON experiments/run_experiment.py --config $CS_BASE --seeds $SEED --override \
    name=t500_alie_fedrola_dp attack=alie use_dp=true epsilon=4.0 $COMMON

  # MinMax no_dp / dp
  $PYTHON experiments/run_experiment.py --config $CS_BASE --seeds $SEED --override \
    name=t500_minmax_fedrola attack=minmax use_dp=false $COMMON
  $PYTHON experiments/run_experiment.py --config $CS_BASE --seeds $SEED --override \
    name=t500_minmax_fedrola_dp attack=minmax use_dp=true epsilon=4.0 $COMMON
done
```

**Результат:** no_dp стабилен (~0.91), DP-режим даёт высокую дисперсию между seeds (0.40–0.79) — DP на длинных горизонтах неустойчив.

---

## Phase 5e — T=500 extra seeds + CD Local DP + GIA rerun (завершено)

**Цель:** добрать seeds {2, 3} к T=500 DP-конфигам; выполнить CD Local DP (Тезис 2 на CD); пересобрать GIA (был баг с y как int).

### Section A — T=500 DP extra seeds (4 runs)
```bash
COMMON="aggregation=fedrola m_real=20 n_rounds=500 lr_end=0.001 save_dir=./results/t500_probe"
for SEED in 2 3; do
  $PYTHON experiments/run_experiment.py --config $CS_BASE --seeds $SEED --override \
    name=t500_alie_fedrola_dp attack=alie use_dp=true epsilon=4.0 $COMMON
  $PYTHON experiments/run_experiment.py --config $CS_BASE --seeds $SEED --override \
    name=t500_minmax_fedrola_dp attack=minmax use_dp=true epsilon=4.0 $COMMON
done
```

### Section B — CD Local vs Central DP (4 runs)
ε=10, FedRoLA, no attack, 2 seeds × {central, local}.
```bash
for MODE in central local; do
  for SEED in 0 1; do
    $PYTHON experiments/run_experiment.py --config $CD_BASE --seeds $SEED --override \
      name=local_dp_cd_${MODE} aggregation=fedrola attack=none m_real=0 \
      use_dp=true epsilon=10.0 dp_mode=$MODE save_dir=./results/local_dp_cd
  done
done
```

### Section C — GIA rerun (bug fixed)
```bash
$PYTHON experiments/gia.py --dataset svhn --model svhn_cnn \
  --n_images 5 --noise_std_dp 0.1 --save_dir ./results/local_dp/gia
```

---

## Phase 6 — Gap-fills по тезисам (завершено)

**Цель:** закрыть пробелы в поддержке Тезисов 2, 3, 5.

### Section A — Тезис 2: Local DP + ALIE (6 runs)
CD (ε=10 client-level) и CS (ε=4 record-level) с ALIE, FedRoLA, dp_mode=local, 3 seeds.

```bash
# A1 — CD
$PYTHON $RUNNER --config $CD_BASE \
  --sweep "attack=alie,m_real=80,aggregation=fedrola,use_dp=true,epsilon=10.0,dp_mode=local,name=local_dp_cd_alie" \
  --seeds 0 1 2 --save_dir ./results/local_dp_attack --max_parallel 6

# A2 — CS
$PYTHON $RUNNER --config $CS_BASE \
  --sweep "attack=alie,m_real=20,aggregation=fedrola,use_dp=true,epsilon=4.0,dp_mode=local,name=local_dp_cs_alie" \
  --seeds 0 1 2 --save_dir ./results/local_dp_attack --max_parallel 6
```

### Section B — Тезис 5: client-level DP на d=90K (negative control, 3 runs)
Ожидание: MTA≈chance; подтверждает необходимость record-level DP.

```bash
$PYTHON $RUNNER --config $CS_BASE \
  --sweep "attack=none,m_real=0,aggregation=fedrola,use_dp=true,epsilon=4.0,dp_level=client,clip_max=2.0,name=cs_client_dp_negcontrol" \
  --seeds 0 1 2 --save_dir ./results/thesis5_neg --max_parallel 6
```

### Section D — Тезис 2 extra seeds (CD Local vs Central, 4 runs)
Добор seeds {3, 4} к phase5e Section B.

```bash
# D1 — Central
$PYTHON $RUNNER --config $CD_BASE \
  --sweep "attack=none,m_real=0,aggregation=fedrola,use_dp=true,epsilon=10.0,dp_mode=central,dp_level=client,name=local_dp_cd_central" \
  --seeds 3 4 --save_dir ./results/local_dp_cd --max_parallel 6

# D2 — Local
$PYTHON $RUNNER --config $CD_BASE \
  --sweep "attack=none,m_real=0,aggregation=fedrola,use_dp=true,epsilon=10.0,dp_mode=local,dp_level=client,name=local_dp_cd_local_fixed" \
  --seeds 3 4 --save_dir ./results/local_dp_cd --max_parallel 6
```

### Section C — Тезис 3: CD MIA (закрыт, в архив)
Cерия прогонов CD MIA (N=10 в phase6; N=20 в phase6c; N=50 в phase6d) показала, что на Fashion-MNIST + MicroMNIST DP-режим не обучается (MTA≈10–15%), AUC-сравнение некорректно. Результаты перемещены в `archive/abandoned_cd_mia/`. В главе 5 MIA представлен по CS (Phase 5b).

---

## Конфиги (experiments/configs/)

### cross_device/ (Fashion-MNIST + MicroMNIST)
| Файл | Назначение |
|---|---|
| `base.yaml` | Общая база для CD |
| `exp1_base.yaml` | База матрицы attack × defense (N=200, q=0.2, E=5, T=200) |
| `bl1_fedavg.yaml` … `bl5_mkrum_dp.yaml` | Baseline-прогоны (без атак) |

### cross_silo/ (SVHN + SVHNCNN)
| Файл | Назначение |
|---|---|
| `base.yaml` | Общая база для CS |
| `exp1_base.yaml` | База матрицы (N=50, q=0.8, E=1, T=200, record-level DP, R=5) |
| `bl1_fedavg.yaml` … `bl5_mkrum_dp.yaml` | Baseline-прогоны |
| `mia_n10.yaml` | Спецификация для Phase 5b CS MIA (N=10, q=1.0, E=10, T=100) |

---

## Воспроизведение одиночного эксперимента

```bash
# Минимальный пример: BL FedAvg CD, 3 seeds, no attack
.venv/bin/python experiments/run_experiment.py \
    --config experiments/configs/cross_device/exp1_base.yaml \
    --seeds 0 1 2 \
    --override attack=none m_real=0 aggregation=fedavg name=demo_fedavg

# Параллельный sweep: ε ∈ {4, 8, 10} с ALIE
.venv/bin/python experiments/run_parallel.py \
    --config experiments/configs/cross_silo/exp1_base.yaml \
    --sweep "epsilon=4,attack=alie,use_dp=true,name=demo_eps4_alie" \
            "epsilon=8,attack=alie,use_dp=true,name=demo_eps8_alie" \
            "epsilon=10,attack=alie,use_dp=true,name=demo_eps10_alie" \
    --seeds 0 1 2 \
    --save_dir ./results/demo \
    --max_parallel 3
```

Ключевые `--override`-параметры (полный список — `core/`, `experiments/fl_simulator.py:FLConfig`):

| Ключ | Значения | Замечание |
|---|---|---|
| `attack` | `none`, `alie`, `ipm`, `fang`, `minmax`, `backdoor`, `random_label_flip`, `label_flip` | `label_flip` — CS only |
| `aggregation` | `fedavg`, `multi_krum`, `trimmed_mean`, `median`, `fedrola` | `fedrola` — primary |
| `use_dp` | `true`/`false` | |
| `dp_mode` | `central`, `local` | `local` требует Opacus |
| `dp_level` | `client`, `record` | `record` — только CS |
| `epsilon` | float | δ=1e-5 фиксирован |
| `m_real` | int | CD: 0-200, CS: 0-50 |
| `n_rounds` | int | |
| `lr_end` | float \| None | Линейный decay от `lr` к `lr_end` |
| `alpha` | float | Dirichlet concentration для non-IID split |
| `fedrola_full_layer` | bool | Per-layer FedRoLA по всем слоям vs 3 sampled |
| `clip_max` | float | Adaptive clipping ceiling |

---

## Карта результатов (что идёт в главу 5)

| Подраздел главы 5 | Источники `results/` |
|---|---|
| 5.2 Baselines | `cross_device/cd_exp1_*_fedavg_*` + `cross_silo/cs_exp1_*_fedavg_*` (no attack) |
| 5.3 Attack × defense | `cross_device/cd_exp1_*`, `cross_silo/cs_exp1_*` (все атаки × защиты × DP) |
| 5.4 Sensitivity | CD S-кривая: `cd_diag_eps*`; CS S-кривая: `cs_p3_eps*`; m_real: `cd_p3_alie_m*`, `cd_exp3_alie_m*`, `cs_exp3_alie_m*`, `cd_diag_backdoor_m*`; α sweep: `cd_exp5_a*`, `cs_exp5_a*`; T=500: `t500_probe/`; ablation: `cd_p3_ablation_*`; SNR: `snr_vs_mta.{json,png}` |
| 5.5 Privacy | CS MIA: `mia_n10/`, `mia_cs/`; Local vs Central: `local_dp/`, `local_dp_cd/`; Тезис 2 + ALIE: `local_dp_attack/`; Тезис 5 negative control: `thesis5_neg/`; GIA: `local_dp/gia/` |
| 5.6 TEE overhead | (pending) — собрать на Azure DCsv3, писать в отдельную папку |

Детальные FL-логи по конкретному run: `results/{cross_device,cross_silo}/<name>_seed<N>.log`.
