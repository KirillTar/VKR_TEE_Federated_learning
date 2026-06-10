# Archive

Артефакты, не используемые в текущей итерации 2 и не идущие в главу 5 ВКР.
Сохранены для истории / возможного возврата. В GitHub-репозиторий не копировать.

## Структура

| Подпапка | Что внутри | Почему архив |
|---|---|---|
| `iteration1/configs/` | YAML конфиги итерации 1 (CIFAR-10, MNIST tiny, старые baselines) | Итерация 1 завершена; её модели/датасеты заменены на Fashion-MNIST + SVHN |
| `iteration1/results/` | Результаты итерации 1 (333 runs, Multi-Krum + FLAME) | Итерация 1 завершена; см. CLAUDE.md «Эксперименты итерации 1» |
| `abandoned_cd_mia/` | `results/mia_cd`, `mia_cd_n10`, `mia_cd_n20`, `mia_cd_n50` | CD MIA на MicroMNIST не даёт контраста: во всех конфигурациях AUC≈0.5; DP-режим не обучается. Вывод: опираемся на CS MIA |
| `old_logs/` | Старые логи из корня `results/` (overnight, phase4, phase5_mia, finish_phase2, и т.п.) | Логи шелл-скриптов итерации 1 или замещённых скриптов; подробные per-run логи лежат в `results/{cross_device,cross_silo}/*.log` |
| `old_configs/` | `mia_cd_n20.yaml`, `mia_cd_n50.yaml` из cross_device | Конфигурации заброшенного CD MIA направления |
| `dev_tests/` | `correctness_test.py`, `speed_test.py`, `speed_test_parallel.py`, `fl_simulator_parallel.py` | Ad-hoc проверки скорости и корректности; не часть пайплайна |
| `smoke/` | `results/smoke_local_dp/` | Smoke-test для local DP, заменён полноценными фазовыми прогонами |

## Иерархия

```
archive/
├── README.md                       (этот файл)
├── iteration1/
│   ├── configs/                    (было: experiments/configs/archived_iteration1/)
│   └── results/                    (было: results/archived_iteration1/)
├── abandoned_cd_mia/
│   ├── mia_cd/
│   ├── mia_cd_n10/
│   ├── mia_cd_n20/
│   └── mia_cd_n50/
├── old_logs/                       (15 *.log из корня results/)
├── old_configs/                    (2 yaml из cross_device)
├── dev_tests/                      (4 py-файла)
└── smoke/                          (smoke_local_dp/)
```

## Итерация 1 — краткое резюме

333 runs на Fashion-MNIST + MicroMNIST с Multi-Krum и FLAME.
Ключевые выводы (подробно в `CLAUDE.md`):

- ALIE полностью обходит Multi-Krum (TPR=0%, FPR=55%)
- FLAME + DP даёт каскадный провал в non-IID
- Multi-Krum + DP ε=10: overhead 13% (MTA 87.3% → 74.0%)
- Backdoor ASR ≥ 82% при всех конфигурациях Multi-Krum+DP

Эти результаты мотивировали переход в итерации 2:
FedRoLA-PCSI (вместо Multi-Krum) и record-level DP (для CS).

## Заброшенное CD MIA направление (2026-04-17 … 2026-04-19)

Гипотеза: на малом числе клиентов (N=10/20/50) с q=1.0 и E=10 FL-модель
начнёт переобучаться → MIA AUC вырастет → DP будет виден контраст.

Факт: σ от RDP-аккаунтанта (σ≈5.3 при ε=10) доминирует над сигналом на
d=4266 (MicroMNIST). DP-модель не обучается (MTA≈10–15%), сравнение
AUC некорректно. Решение (2026-04-19): опираться на CS MIA, где
record-level DP позволяет обучить модель; CD MIA исключён из главы 5.
