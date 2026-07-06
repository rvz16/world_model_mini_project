# World Model + VLM Scorer for MiniGrid

A compact demo that combines a **Dreamer/PlaNet-style world model (RSSM)** with a
**VLM-based goal scorer (CLIP)** to control an agent in MiniGrid via **MPC planning
over imagined rollouts**.

The agent never uses the task reward at plan time in the VLM mode: it imagines
future frames with the world model and asks a pretrained CLIP "how close is this
to the goal?" — scoring the *imagined future frames* (not just the current
observation), exactly as required.

## Pipeline

```
obs ─► CNN encoder ─► RSSM state ─┬─ sample N action sequences (random shooting / CEM)
                                  ├─ imagine H steps in latent space (prior only)
                                  ├─ decode imagined future frames
                                  ├─ score each frame: WM reward head and/or CLIP(goal)
                                  └─ pick best sequence ► execute first action ► repeat
```

- **Environment:** `MiniGrid-Empty-6x6-v0`, fully-observable RGB, resized to 64×64.
- **World model:** RSSM (GRU deterministic state + Gaussian stochastic latent),
  CNN encoder/decoder, reward head. Trained with the Dreamer ELBO
  (reconstruction + reward + KL with free-nats).
- **VLM scorer:** open_clip `ViT-B-32`. Goal given as text. Score is a contrastive
  CLIP similarity applied to imagined future frames.
- **Planner:** random shooting MPC (CEM optional via `--cem`).
- **Baselines:** `random`, `wm` (plan on WM-predicted reward, no VLM),
  `wm_vlm` (plan on VLM score).

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# 1. collect data + train the world model (caches dataset + checkpoint)
python -m src.train_wm --steps 6000

# 2. quantitative comparison of the 3 modes
python -m src.evaluate --episodes 20 --seeds 0 1 2

# 3. visualizations (behavior GIFs + imagined-rollout filmstrip with VLM scores)
python -m src.visualize --seed 0
```

Outputs land in `outputs/` (`checkpoints/`, `results/`, `gifs/`).

## Results

MiniGrid-Empty-6x6, success = reaching the green goal within the step cap.
`wm` and `wm_vlm` use disjoint objectives (reward-only vs VLM-only) to isolate each signal.

| Mode | Success rate | Mean return | Mean steps | N |
|------|:-----------:|:-----------:|:---------:|:--:|
| Random | 0.37 | 0.176 | 54.1 | 30 |
| **WM planning (no VLM)** | **0.93** | **0.422** | **40.6** | 30 |
| WM planning + VLM (VLM-only) | 0.13 | 0.084 | 59.0 | 15 |

World-model planning solves the task; planning on the CLIP score alone does *worse
than random* — the decoded-frame goal-detection AUC is 0.30 (< 0.5, anti-correlated).
See `report/` for the full analysis and `outputs/gifs/` for visualizations.

## Key finding on the VLM (see report)

CLIP cannot resolve the spatial relation "agent **on** the goal" on abstract
MiniGrid tiles. The reliable cue is that the **green goal tile gets occluded by
the agent** once reached, so the goal is framed as *"green is gone"*. This gives
a perfectly-separating but **terminal** goal detector (no approach gradient) —
the central trade-off discussed in the report.

## Repo layout

```
src/
  config.py        # all hyperparameters (laptop-friendly defaults)
  env.py           # MiniGrid RGB wrapper
  data.py          # rollout collection (random + scripted BFS) + batching
  models/
    networks.py    # CNN encoder / decoder / reward head
    rssm.py        # recurrent state-space model
    world_model.py # ELBO loss + imagination
  vlm_scorer.py    # CLIP goal scorer
  planner.py       # random shooting + CEM
  agent.py         # closed-loop MPC agent
  evaluate.py      # 3-mode quantitative comparison
  visualize.py     # GIFs + imagined-rollout filmstrip
```
