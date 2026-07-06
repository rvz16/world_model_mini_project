"""Visualizations: behavior GIFs per mode, and an imagined-rollout filmstrip
showing the world model's decoded future frames with their VLM scores.
"""
import argparse
import os
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw

from .config import Config
from .models import WorldModel
from .vlm_scorer import VLMScorer
from .agent import Agent
from .evaluate import load_world_model


def _hstack_png(frames, out_path, k=6):
    """Save a horizontal montage of k evenly-spaced frames as a PNG (for LaTeX)."""
    idx = np.linspace(0, len(frames) - 1, min(k, len(frames))).astype(int)
    strip = np.concatenate([np.asarray(frames[i], dtype=np.uint8) for i in idx], axis=1)
    imageio.imwrite(out_path, strip)
    print(f"[viz] montage -> {out_path}")


def save_behavior_gif(cfg: Config, wm, scorer, mode: str, seed: int, path: str):
    agent = Agent(cfg, wm, scorer)
    out = agent.run_episode(mode, seed, collect_frames=True)
    frames = [np.asarray(f, dtype=np.uint8) for f in out["frames"]]
    imageio.mimsave(path, frames, duration=0.2, loop=0)
    _hstack_png(frames, path.replace(".gif", ".png"))   # PNG still for the report
    print(f"[viz] {mode} seed={seed} success={out['success']} -> {path}")
    return out


def _annotate(frame_chw: np.ndarray, text: str, size: int = 128) -> np.ndarray:
    img = (np.clip(frame_chw.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img).resize((size, size), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, size - 14, size, size], fill=(0, 0, 0))
    draw.text((2, size - 13), text, fill=(255, 255, 0))
    return np.asarray(img)


def save_imagined_filmstrip(cfg: Config, wm: WorldModel, scorer: VLMScorer,
                            seed: int, path: str):
    """Plan the best action sequence from the start state, then decode and score
    the imagined future frames (the VLM operates on these, per the spec)."""
    from .env import MiniGridImage
    from .planner import Planner

    env = MiniGridImage(cfg.env)
    frame, _ = env.reset(seed=seed)
    device = cfg.device
    rng = np.random.default_rng(seed)

    obs_t = torch.from_numpy(frame).unsqueeze(0).to(device)
    init_state = wm.init_state_from_obs(obs_t)

    planner = Planner(cfg.plan, env.num_actions, device)
    N, H = cfg.plan.num_candidates, cfg.plan.horizon
    idx = torch.from_numpy(rng.integers(0, env.num_actions, size=(N, H))).long().to(device)
    rollout = wm.imagine_rollouts(init_state, planner._onehot(idx))
    vlm = scorer.score_rollout(rollout["frames"])            # (N, H)
    best = (vlm * planner.discount).sum(1).argmax().item()

    best_frames = rollout["frames"][best].cpu().numpy()      # (H, C, H, W)
    best_scores = vlm[best].numpy()
    strip = [_annotate(frame, "start (real)")]
    strip += [_annotate(best_frames[h], f"t+{h+1} vlm={best_scores[h]:+.3f}")
              for h in range(H)]
    imageio.mimsave(path, strip, duration=0.4, loop=0)
    # PNG filmstrip (all frames side-by-side) for the report
    png = np.concatenate(strip, axis=1)
    imageio.imwrite(path.replace(".gif", ".png"), png)
    env.close()
    print(f"[viz] imagined rollout -> {path} (+ .png)")


def main(cfg: Config, seed: int):
    os.makedirs("outputs/gifs", exist_ok=True)
    wm = load_world_model(cfg)
    scorer = VLMScorer(cfg.vlm, device=cfg.device)
    for mode in ["random", "wm", "wm_vlm"]:
        save_behavior_gif(cfg, wm, scorer, mode, seed,
                          f"outputs/gifs/behavior_{mode}.gif")
    save_imagined_filmstrip(cfg, wm, scorer, seed,
                            "outputs/gifs/imagined_rollout.gif")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(Config(), args.seed)
