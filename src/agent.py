"""Closed-loop agent that runs MPC planning in the world model.

Re-encodes the current observation into an RSSM posterior state every step
(receding-horizon MPC), then plans. Works for all three modes.
"""
from typing import Optional
import numpy as np
import torch

from .config import Config
from .env import MiniGridImage
from .models import WorldModel
from .planner import Planner
from .vlm_scorer import VLMScorer


class Agent:
    def __init__(self, cfg: Config, wm: Optional[WorldModel],
                 scorer: Optional[VLMScorer]):
        self.cfg = cfg
        self.wm = wm
        self.scorer = scorer
        self.planner = Planner(cfg.plan, len(cfg.env.action_set), cfg.device)

    def run_episode(self, mode: str, seed: int, collect_frames: bool = False):
        env = MiniGridImage(self.cfg.env)
        rng = np.random.default_rng(seed)
        frame, _ = env.reset(seed=seed)
        device = self.cfg.device

        total_reward, steps, success = 0.0, 0, False
        frames_rgb = [env.render_uint8()] if collect_frames else None

        done = False
        while not done and steps < self.cfg.env.max_steps:
            if mode == "random":
                action = int(rng.integers(0, env.num_actions))
            else:
                obs_t = torch.from_numpy(frame).unsqueeze(0).to(device)
                init_state = self.wm.init_state_from_obs(obs_t)
                action, _ = self.planner.plan(self.wm, self.scorer, init_state, mode, rng)

            frame, r, term, trunc, _ = env.step(action)
            total_reward += r
            steps += 1
            if collect_frames:
                frames_rgb.append(env.render_uint8())
            if env.at_goal() or r > 0:
                success = True
            done = term or trunc

        env.close()
        return {
            "success": float(success),
            "return": total_reward,
            "steps": steps,
            "frames": frames_rgb,
        }
