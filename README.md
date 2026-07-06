# World Model + VLM Scorer для MiniGrid

Компактное демо, объединяющее **world model в стиле Dreamer/PlaNet (RSSM)** с
**VLM-based goal scorer (CLIP)** для управления агентом в MiniGrid через
**MPC-планирование по imagined rollouts**.

В режиме с VLM агент **не использует reward задачи** во время планирования: он
воображает будущие кадры с помощью world model и спрашивает у предобученного CLIP
«насколько это близко к цели?» — оценивая именно *imagined future frames* (а не
только текущее наблюдение), как того требует задание.

## Pipeline

```
obs ─► CNN encoder ─► RSSM state ─┬─ sample N action sequences (random shooting / CEM)
                                  ├─ imagine H steps in latent space (только prior)
                                  ├─ decode imagined future frames
                                  ├─ score каждого кадра: WM reward head и/или CLIP(goal)
                                  └─ выбрать лучшую последовательность ► сделать первый action ► повторить
```

- **Environment:** `MiniGrid-Empty-6x6-v0`, полностью наблюдаемая RGB, приведена к 128×128.
- **World model:** RSSM (детерминированное состояние GRU + гауссовский stochastic latent),
  CNN encoder/decoder, reward head. Обучается по Dreamer ELBO
  (reconstruction + reward + KL с free-nats).
- **VLM scorer:** open_clip `ViT-B-32`. Цель задаётся текстом. Score — это контрастивная
  CLIP-similarity, применённая к imagined future frames.
- **Planner:** random shooting MPC (CEM опционально через `--cem`).
- **Baselines:** `random`, `wm` (планирование по WM-predicted reward, без VLM),
  `wm_vlm` (планирование по VLM-score).

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
# 1. сбор данных + обучение world model (кэширует dataset + checkpoint)
python -m src.train_wm --steps 4000

# 2. количественное сравнение трёх режимов
python -m src.evaluate --episodes 10 --seeds 0 1 2

# 3. визуализации (behavior GIFs + imagined-rollout filmstrip с VLM-скорами)
python -m src.visualize --seed 0
```

Результаты пишутся в `outputs/` (`checkpoints/`, `results/`, `gifs/`).

## Результаты

MiniGrid-Empty-6x6, success = агент дошёл до зелёной цели в пределах лимита шагов.
`wm` и `wm_vlm` используют непересекающиеся objective (reward-only vs VLM-only), чтобы
изолировать вклад каждого сигнала.

| Режим | Success rate | Mean return | Mean steps | N |
|------|:-----------:|:-----------:|:---------:|:--:|
| Random | 0.37 | 0.176 | 54.1 | 30 |
| **WM planning (без VLM)** | **0.93** | **0.422** | **40.6** | 30 |
| WM planning + VLM (VLM-only) | 0.13 | 0.084 | 59.0 | 15 |

Планирование в world model решает задачу; планирование по одному лишь CLIP-score даёт
результат *хуже случайного* — goal-detection AUC на декодированных кадрах равен 0.30
(< 0.5, анти-коррелирован). Полный анализ — в `report/`, визуализации — в `outputs/gifs/`.

## Ключевой вывод про VLM (см. отчёт)

CLIP не умеет разрешать пространственное отношение «агент **на** цели» на абстрактных
тайлах MiniGrid. Надёжный признак — то, что **зелёная клетка цели закрывается агентом**
при достижении, поэтому цель формулируется как *«зелёного больше нет»*. Это даёт
идеально разделяющий, но **терминальный** детектор цели (без approach gradient), и —
что важнее — этот высокочастотный признак **выживает только при высоком разрешении**:
на декодированных 128px кадрах он теряется (AUC 0.30). Это главный trade-off, разобранный
в отчёте.

## Структура репозитория

```
src/
  config.py        # все гиперпараметры (дефолты под ноутбук)
  env.py           # обёртка MiniGrid RGB
  data.py          # сбор rollouts (random + scripted BFS) + батчинг
  models/
    networks.py    # CNN encoder / decoder / reward head
    rssm.py        # recurrent state-space model
    world_model.py # ELBO loss + imagination
  vlm_scorer.py    # CLIP goal scorer
  planner.py       # random shooting + CEM
  agent.py         # closed-loop MPC agent
  evaluate.py      # количественное сравнение трёх режимов
  visualize.py     # GIFs + imagined-rollout filmstrip
report/            # отчёт (report.md, report.tex — Overleaf/pdfLaTeX)
outputs/           # results/ (таблицы) и gifs/ (визуализации)
```
