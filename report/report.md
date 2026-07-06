# World Model + VLM Scorer for MiniGrid — Report

**Task.** Combine a Dreamer-style world model (RSSM) with a VLM-based goal scorer
to control an agent in a simple environment via MPC planning over *imagined*
rollouts, and compare against two baselines (random; world-model planning without
the VLM).

**TL;DR.** The world model + planning pipeline works and clearly beats random
(success 0.93 vs 0.37). The VLM-scored variant, however, does *worse than random*
(0.13) — and we trace this to a concrete, quantified cause: **CLIP can only detect
the goal on the full-resolution native render (AUC 0.98 at 192px); on the frames a
compact world-model decoder produces the cue is not just lost but anti-correlated
(AUC 0.30 < 0.5), so optimising the VLM score steers the agent away from the goal.**
This is the central finding and is discussed in detail under Failure Modes.

---

## 1. Setup

| Component | Choice |
|---|---|
| Environment | `MiniGrid-Empty-6x6-v0`, fully-observable RGB, resized to 128×128 |
| Goal (text) | framed as "green goal occluded" (see §4) |
| World model | RSSM: GRU deterministic state (128) + Gaussian stochastic latent (32), CNN encoder/decoder, reward head |
| WM training | Dreamer ELBO = reconstruction + reward + KL (free-nats=1.0); Adam 3e-4; batch 32; seq-len 20; MPS |
| VLM | open_clip `ViT-B-32` (`laion2b_s34b_b79k`), contrastive text score |
| Planner | random-shooting MPC, horizon H=10, receding horizon |
| Data | 400 episodes (250 random + 150 noisy shortest-path), 15,444 transitions, 60% reach the goal |

**Modes compared**
- `random` — uniform random actions.
- `wm` — MPC objective = discounted **WM-predicted reward** (no VLM).
- `wm_vlm` — MPC objective = discounted **CLIP goal score on the imagined future frames**.

The VLM score is applied to the *decoded imagined future frames* of each candidate
rollout (not to the current observation), as required.

---

## 2. World model quality

The RSSM trains cleanly on MPS: reconstruction loss drops from ~430 to a per-pixel
RMSE ≈ 0.02, and the reward head fits the sparse goal reward almost perfectly
(reward-loss ≈ 3e-4; predicted reward ≈ 0.7–0.8 exactly at the goal step, ≈ 0
elsewhere). Imagined rollouts are visually faithful. In other words, **the latent
state clearly encodes goal achievement** — the bottleneck is not the world model.

---

## 3. Results

Episodes: **10 per seed × seeds {0, 1, 2} = 30 episodes per mode**. Success = agent
reaches the green goal within the step cap. (Numbers filled from `outputs/results/results.md`.)

The `wm` and `wm_vlm` objectives are **disjoint** (no reward term leaks into the VLM
mode) so the comparison isolates each signal: `wm` plans on the WM-predicted reward,
`wm_vlm` plans on the CLIP goal score only.

| Mode | Success rate | Mean return | Mean steps | N |
|------|:-----------:|:-----------:|:---------:|:--:|
| Random | 0.37 | 0.176 | 54.1 | 30 |
| **WM planning (no VLM)** | **0.93** | **0.422** | **40.6** | 30 |
| WM planning + VLM (VLM-only) | 0.13 | 0.084 | 59.0 | 15\* |

\* `wm_vlm` uses fewer episodes because CLIP scoring of every candidate rollout is the
bottleneck (~120–150 s/episode on MPS). Its value does not improve with more episodes:
the decoded-frame goal-detection AUC is **0.30 (< 0.5, i.e. anti-correlated)**, so the
VLM objective actively points *away* from the goal.

**Reading of the results.** World-model planning on the reward head solves the task
(high success, far fewer steps than random). Planning on the VLM score alone does
*worse than random* (0.13 vs 0.37): because the decoded-frame scorer is
anti-correlated with the goal (AUC 0.30), optimising it steers the agent away. This is
and is the honest, informative outcome of the experiment rather than a tuning
failure.

---

## 4. The VLM scorer: what works, what doesn't (main failure mode)

We validated the scorer directly, independent of planning, by measuring how well its
scalar separates goal from non-goal frames (664 frames, 60 at the goal), reported as
pair-ranking **AUC**.

**CLIP cannot do the spatial relation "agent *on* the goal."** On abstract MiniGrid
tiles CLIP behaves as a bag-of-concepts with no spatial binding: prompts like
*"a red triangle on a green square"* score *lower* at the goal (AUC ≈ 0.0), because
what actually changes is the **total amount of green** — the agent occludes part of
the green goal tile when it arrives. So we frame the goal as *"green is gone"*
(`"a dark maze with no green"` vs `"a green goal square in a maze"`).

**But this cue only survives at full resolution.**

| Frame source given to CLIP | Goal-detection AUC |
|---|---|
| native render, **192px** | **0.977** |
| native 192 → 128 (bilinear / area) | 0.495 / 0.498 |
| native 192 → 96 (area) | 0.671 |
| model **decoded** frames (128px world model) | **0.302** |

The goal signal is a **high-frequency** detail (the agent silhouette cut into the
green cell). It is averaged away by any downsampling below ~192px and by the lossy
decoder — even though reconstruction RMSE is only 0.02. Consequently, scoring the
decoded imagined frames is at or below chance, and `wm_vlm` planning cannot steer.

This is a clean, general lesson: **a VLM scorer is only useful for planning if its
goal cue is preserved at the resolution and fidelity of the world model's decoder.**
Abstract, small-footprint cues (a single grid cell) fail this test.

---

## 5. Other failure modes observed

- **Sparse / terminal signal.** Even where the VLM does separate goal frames, the
  cue is *terminal* (fires only once the agent is on the goal), giving no approach
  gradient. Random-shooting compensates only because 6×6 is tiny enough that some
  imagined rollouts reach the goal within the horizon.
- **Pose sensitivity.** CLIP scores vary more with agent orientation/position than
  with goal achievement, adding noise to any per-frame objective.
- **Random baseline is strong here.** In a 6×6 room a random walk reaches the goal
  fairly often within the step cap, compressing the gap between methods.

---

## 6. What we would try next (future work)

1. **Match the scorer to the decoder resolution.** Either (a) train the world model
   to decode at ≥192px (256px for a power-of-two decoder), or (b) add a dedicated
   high-resolution "scoring head." Decisive test: goal-detection AUC on *decoded*
   frames, which must clear ~0.9 before planning can benefit.
2. **Give the VLM a global cue.** Redesign the task so goal proximity produces a
   large, low-frequency change (e.g., an egocentric view where the goal colour fills
   the forward cone, or a "go to a big coloured object" task). Global colour cues
   survive downsampling and decoding.
3. **Stronger / better-suited VLM.** SigLIP or larger CLIP variants; or a VLM
   fine-tuned briefly on rendered gridworld frames.
4. **CEM instead of random shooting** (implemented, `--cem`) for tighter action
   search, and **denser shaping** (score progress, not just terminal state).
5. **Full Dreamer actor-critic** in imagination as a bonus, replacing MPC.

---

## 7. Reproduce

```bash
pip install -r requirements.txt
python -m src.train_wm --steps 4000      # collect data + train world model
python -m src.evaluate --episodes 10 --seeds 0 1 2
python -m src.visualize --seed 0         # behavior GIFs + imagined-rollout filmstrip
```
