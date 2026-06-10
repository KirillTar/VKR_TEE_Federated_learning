# Статус экспериментов ВКР

## Легенда
`✅ Done` | `🔄 Running` | `⏳ Pending` | `❌ Failed` | `📦 Archived`

---

## CIFAR-10 результаты (архив)

| ID   | Описание                        | Статус    | MTA (mean ± std)     | Заметки                                  |
|------|---------------------------------|-----------|----------------------|------------------------------------------|
| BL-0 | Централизованное обучение       | 📦 Done   | 85.88%               | Архив: results/archived_cifar10/         |
| BL-1 | FedAvg, IID (α=10)              | 📦 Done   | 86.73% ± 0.30%       | Архив                                    |
| BL-2 | FedAvg, α=0.5, без атак         | 📦 Done   | 85.58% ± 0.12%       | Архив                                    |
| BL-3 | FedAvg + Multi-Krum, без атак   | 📦 Done   | 84.06% ± 0.15%       | Архив                                    |
| BL-4 | FedAvg + Central DP             | ❌ Failed  | ~10% (случайный)     | SNR=0.004 при ε=8, d=290K. Обучение невозможно. Мотивация перехода на MNIST. |

Детали провала: `docs/cifar10_lessons_learned.md`

---

## ФАЗА 0: Baselines (MNIST, n=100, q=0.2)

| ID    | Описание                                  | Статус     | MTA                  | Заметки                                  |
|-------|-------------------------------------------|------------|----------------------|------------------------------------------|
| BL-3.5| CIFAR-10 DP failure (мотивация)           | 📦 Done    | —                    | Архивные данные, нет доп. прогонов       |
| BL-0  | Централизованное обучение (MNIST)         | ⏳ Pending | —                    | Ожидаем ~98-99%                          |
| BL-1  | FedAvg, IID (α=10), n=100, q=0.2         | ⏳ Pending | —                    | Ожидаем ~96-98%                          |
| BL-2  | FedAvg, α=0.5, без атак                  | ⏳ Pending | —                    | Ожидаем ~93-96%                          |
| BL-3  | FedAvg + Multi-Krum, без атак             | ⏳ Pending | —                    | Ожидаем BL-2 − 1-3%, FPR=30%            |
| BL-4  | FedAvg + Central DP, ε sweep             | ⏳ Pending | —                    | ε∈{1,5,10,20,∞}, sample_rate=0.2        |
| BL-5  | Атаки без защит                          | ⏳ Pending | —                    | Верификация что атаки работают           |
| BL-6  | MKrum + Sign-Flip                        | ⏳ Pending | —                    | TPR должен быть ~100%                    |

---

## ФАЗА 1: Основные эксперименты

| ID     | Описание                        | Статус     | Оценка прогонов |
|--------|---------------------------------|------------|-----------------|
| Exp-1  | Атаки × Защиты (главная таблица)| ⏳ Pending | ~60 прогонов, ~15 ч |
| Exp-3  | TPR/FPR по атакам               | ⏳ Pending | Из Exp-1 (логировать selected) |
| Exp-4  | Sweep по ε                      | ⏳ Pending | ~48 прогонов, ~12 ч |
| Exp-Sub| Sweep по sample_rate q          | ⏳ Pending | ~15 прогонов, ~4 ч |

## ФАЗА 2: Важные эксперименты

| ID     | Описание                        | Статус     | Оценка прогонов |
|--------|---------------------------------|------------|-----------------|
| Exp-2  | MKrum vs Krum vs TrimmedMean    | ⏳ Pending | ~36 прогонов, ~9 ч |
| Exp-5+6| Sensitivity m_real / m_assumed  | ⏳ Pending | ~35 прогонов, ~9 ч |

## ФАЗА 3: Non-IID анализ

| ID     | Описание                        | Статус     | Оценка прогонов |
|--------|---------------------------------|------------|-----------------|
| Exp-9  | Non-IID sweep (α sweep)         | ⏳ Pending | ~24 прогона, ~6 ч |

## ФАЗА 4: Бонусные (если время)

| ID     | Описание                        | Статус     |
|--------|---------------------------------|------------|
| Exp-7  | Central DP vs Local DP          | ⏳ Pending |
| Exp-8  | Ablation study                  | ⏳ Pending |
| Exp-10 | Fashion-MNIST                   | ⏳ Pending |
| Exp-11 | TEE overhead (Azure DCsv3)      | ⏳ Pending |

---

## Исправленные баги

| Файл              | Описание                                                                 | Когда  |
|-------------------|--------------------------------------------------------------------------|--------|
| `core/attacks.py` | BN buffers не передавались на сервер → MTA зависала на 0.42             | сессия 1 |
| `core/attacks.py` | `compute_tpr_fpr` делил на ноль при m_real=0                             | сессия 1 |
| `core/aggregation.py` | `multi_krum` assert вместо warning при n < 2m+3                      | сессия 1 |
| `experiments/fl_simulator.py` | Порядок итерации state_dict (param_keys)                   | сессия 1 |
| `core/attacks.py` | `LabelFlipDataset`: `np.random.random()` в `__getitem__` не seeded → pre-compute mask в `__init__` | сессия 2 |
| `core/attacks.py` | Триггер `1.0` ≠ белый пиксель в normalized space. Исправлен на `(1-mean)/std` per channel | сессия 2 |
| `experiments/run_parallel.py` + `run_experiment.py` | Race condition: 3 процесса перезаписывали один `_summary.json` → std=NaN | сессия 2 |
| `experiments/fl_simulator.py` | DP: `sample_rate=1/n` (трюк из сессии 2) физически неправильно для полного участия. Нужно: `sample_rate = clients_per_round / n_clients` | сессия 3 |

---

## Параметры, зафиксированные в методике (MNIST, актуальные)

- Датасет: MNIST, модель: TinyMNIST (~9K params, 2-conv, без BatchNorm)
- n=100 клиентов, clients_per_round=20 (q=0.2), E=5 лок. эпох, B=32, lr=0.01 (SGD), T=200 раундов
- Основная задача: α=0.5, m_real=20 (20%), m_assumed=6 (30% per-round)
- Multi-Krum: f = 14 (per-round)
- DP: Central, adaptive clipping S=median(‖Δ‖), RDP accounting, sample_rate=0.2
- ε sweep: {1, 5, 10, 20, ∞}, δ=1e-5
- Label Flip: random all classes; ALIE: α=0.1 для сильного сигнала
- 3 seeds, отчёт mean ± std

---

## Скорость (ориентир, нужно перемерить для MNIST n=100)

- CIFAR-10 n=20: ~23 с/раунд (1 процесс)
- MNIST n=100 (ожидаем): ~8-15 с/раунд (меньшая модель, но больше клиентов)
- Оценка: 200 раундов × 1 seed ≈ 30-50 мин → 3 seeds параллельно ≈ 40-60 мин
