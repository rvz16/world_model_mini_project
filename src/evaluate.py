"""Quantitative comparison of the three modes over multiple episodes/seeds.

Writes a results table (CSV + markdown) under outputs/results/.
"""
import argparse
import os
import csv
import numpy as np
import torch

from .config import Config
from .models import WorldModel
from .vlm_scorer import VLMScorer
from .agent import Agent

MODES = ["random", "wm", "wm_vlm"]
MODE_LABELS = {
    "random": "Random",
    "wm": "WM planning (no VLM)",
    "wm_vlm": "WM planning + VLM",
}


def load_world_model(cfg: Config) -> WorldModel:
    ckpt = torch.load(cfg.train.ckpt_path, map_location=cfg.device, weights_only=False)
    wm = WorldModel(len(cfg.env.action_set), cfg.rssm, image_size=cfg.env.image_size).to(cfg.device)
    wm.load_state_dict(ckpt["model"])
    wm.eval()
    return wm


def evaluate(cfg: Config, n_episodes: int, seeds, modes=MODES):
    wm = load_world_model(cfg)
    scorer = VLMScorer(cfg.vlm, device=cfg.device)
    agent = Agent(cfg, wm, scorer)

    rows = []
    per_mode = {}
    for mode in modes:
        succ, rets, steps = [], [], []
        for seed in seeds:
            for ep in range(n_episodes):
                ep_seed = seed * 100000 + ep
                out = agent.run_episode(mode, ep_seed)
                succ.append(out["success"])
                rets.append(out["return"])
                steps.append(out["steps"])
        per_mode[mode] = {
            "success_rate": float(np.mean(succ)),
            "success_std": float(np.std(succ)),
            "return": float(np.mean(rets)),
            "steps": float(np.mean(steps)),
            "n": len(succ),
        }
        m = per_mode[mode]
        print(f"[eval] {MODE_LABELS[mode]:24s} "
              f"success={m['success_rate']:.2f} return={m['return']:.3f} "
              f"steps={m['steps']:.1f} (n={m['n']})")
        rows.append([MODE_LABELS[mode], m["success_rate"], m["return"],
                     m["steps"], m["n"]])
    write_results(rows, n_episodes, seeds)
    return per_mode


def write_results(rows, n_episodes, seeds):
    os.makedirs("outputs/results", exist_ok=True)
    csv_path = "outputs/results/results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "success_rate", "mean_return", "mean_steps", "n_eps"])
        w.writerows(rows)
    md_path = "outputs/results/results.md"
    with open(md_path, "w") as f:
        f.write(f"# Results ({n_episodes} episodes x seeds {list(seeds)})\n\n")
        f.write("| Mode | Success rate | Mean return | Mean steps | N |\n")
        f.write("|------|-------------|-------------|-----------|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]:.2f} | {r[2]:.3f} | {r[3]:.1f} | {r[4]} |\n")
    print(f"[eval] wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--modes", type=str, nargs="+", default=MODES)
    ap.add_argument("--cem", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    if args.cem:
        cfg.plan.cem = True
    evaluate(cfg, args.episodes, args.seeds, args.modes)
