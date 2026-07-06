"""MiniGrid environment wrappers.

We expose a fully-observable RGB view, resized to a fixed square resolution and
returned as a float CHW tensor in [0, 1]. We also keep the raw uint8 frame for
visualization and for feeding ground-truth frames to the VLM (sanity checks).
"""
from typing import Optional
import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid envs)
from minigrid.wrappers import RGBImgObsWrapper
from PIL import Image

from .config import EnvConfig


def _resize(frame: np.ndarray, size: int) -> np.ndarray:
    """uint8 HWC -> uint8 HWC resized to (size, size)."""
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    img = Image.fromarray(frame).resize((size, size), Image.NEAREST)
    return np.asarray(img)


def to_chw_float(frame_uint8: np.ndarray) -> np.ndarray:
    """uint8 HWC [0,255] -> float32 CHW [0,1]."""
    return frame_uint8.astype(np.float32).transpose(2, 0, 1) / 255.0


class MiniGridImage:
    """Thin wrapper returning fixed-size RGB frames and exposing a small action set.

    Not a gym.Env subclass on purpose: planning code wants explicit, simple
    control over reset/step and access to both float and uint8 frames.
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        env = gym.make(cfg.env_id, max_steps=cfg.max_steps)
        self.env = RGBImgObsWrapper(env)
        self.action_set = list(cfg.action_set)
        self.num_actions = len(self.action_set)
        self.size = cfg.image_size
        self._last_uint8: Optional[np.ndarray] = None

    # -- observation helpers -------------------------------------------------
    def _obs_from_dict(self, obs_dict) -> np.ndarray:
        frame = _resize(obs_dict["image"], self.size)
        self._last_uint8 = frame
        return to_chw_float(frame)

    @property
    def last_frame_uint8(self) -> np.ndarray:
        return self._last_uint8

    # -- gym-like API --------------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        obs_dict, info = self.env.reset(seed=seed)
        return self._obs_from_dict(obs_dict), info

    def step(self, action_idx: int):
        """action_idx indexes into action_set (not the raw MiniGrid action)."""
        raw_action = self.action_set[action_idx]
        obs_dict, reward, terminated, truncated, info = self.env.step(raw_action)
        return self._obs_from_dict(obs_dict), float(reward), terminated, truncated, info

    def render_uint8(self) -> np.ndarray:
        """Full-resolution RGB frame for GIFs."""
        return self.unwrapped.get_frame(highlight=False)

    # -- oracle for success + scripted data ----------------------------------
    @property
    def unwrapped(self):
        return self.env.unwrapped

    def at_goal(self) -> bool:
        """True if the agent currently stands on the goal cell."""
        u = self.unwrapped
        cell = u.grid.get(*u.agent_pos)
        return cell is not None and cell.type == "goal"

    def close(self):
        self.env.close()
