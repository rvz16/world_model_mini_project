"""Data collection and replay storage for world-model training.

We gather two kinds of episodes:
  * random-policy rollouts (broad state coverage), and
  * noisy shortest-path rollouts (so the dataset actually contains goal-reaching
    transitions and the reward head has positive signal to learn from).

Each episode stores the float CHW frames, the action indices, rewards and
done flags. Training samples fixed-length windows across episodes.
"""
from collections import deque
from typing import List
import numpy as np

from .config import EnvConfig, DataConfig
from .env import MiniGridImage


# MiniGrid direction vectors: 0=right, 1=down, 2=left, 3=up
DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


class Episode:
    def __init__(self):
        self.frames: List[np.ndarray] = []   # T+1 float CHW
        self.actions: List[int] = []         # T action indices
        self.rewards: List[float] = []       # T
        self.dones: List[float] = []         # T

    @staticmethod
    def _to_uint8(frame):
        # env gives float CHW [0,1]; store as uint8 to shrink the dataset ~4x
        return (np.clip(frame, 0, 1) * 255).astype(np.uint8)

    def add_first(self, frame):
        self.frames.append(self._to_uint8(frame))

    def add(self, action, frame, reward, done):
        self.actions.append(action)
        self.frames.append(self._to_uint8(frame))
        self.rewards.append(reward)
        self.dones.append(float(done))

    def __len__(self):
        return len(self.actions)

    def as_arrays(self):
        # frames stored as uint8 (CHW, 0-255) to keep the dataset small; callers
        # convert to float [0,1].
        return (
            np.asarray(self.frames, dtype=np.uint8),
            np.asarray(self.actions, dtype=np.int64),
            np.asarray(self.rewards, dtype=np.float32),
            np.asarray(self.dones, dtype=np.float32),
        )


def _bfs_first_step(env: MiniGridImage):
    """Return the next grid cell toward the goal via BFS, or None."""
    u = env.unwrapped
    start = tuple(u.agent_pos)
    goal = None
    for j in range(u.grid.height):
        for i in range(u.grid.width):
            c = u.grid.get(i, j)
            if c is not None and c.type == "goal":
                goal = (i, j)
    if goal is None:
        return None

    def passable(pos):
        c = u.grid.get(*pos)
        return c is None or c.can_overlap()

    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for dx, dy in DIR_VEC.values():
            nxt = (cur[0] + dx, cur[1] + dy)
            if 0 <= nxt[0] < u.grid.width and 0 <= nxt[1] < u.grid.height:
                if nxt not in prev and passable(nxt):
                    prev[nxt] = cur
                    q.append(nxt)
    if goal not in prev:
        return None
    # walk back to the cell right after start
    node = goal
    while prev[node] != start:
        node = prev[node]
        if node is None:
            return None
    return node


def scripted_action_idx(env: MiniGridImage) -> int:
    """Greedy turn-toward-then-forward controller indexed into action_set."""
    u = env.unwrapped
    nxt = _bfs_first_step(env)
    aset = env.action_set
    fwd = aset.index(2)
    left = aset.index(0)
    right = aset.index(1)
    if nxt is None:
        return fwd
    pos = tuple(u.agent_pos)
    desired = (nxt[0] - pos[0], nxt[1] - pos[1])
    desired_dir = next((d for d, v in DIR_VEC.items() if v == desired), u.agent_dir)
    cur_dir = u.agent_dir
    if cur_dir == desired_dir:
        return fwd
    # choose the cheaper turn
    return right if (desired_dir - cur_dir) % 4 == 1 else left


def collect_dataset(env_cfg: EnvConfig, data_cfg: DataConfig) -> List[Episode]:
    env = MiniGridImage(env_cfg)
    rng = np.random.default_rng(data_cfg.seed)
    episodes: List[Episode] = []

    def run_episode(seed, scripted):
        frame, _ = env.reset(seed=seed)
        ep = Episode()
        ep.add_first(frame)
        done = False
        while not done and len(ep) < env_cfg.max_steps:
            if scripted and rng.random() > data_cfg.scripted_noise:
                a = scripted_action_idx(env)
            else:
                a = int(rng.integers(env.num_actions))
            frame, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep.add(a, frame, r, term)  # store terminal (goal) as done, not truncation
        episodes.append(ep)

    base = 10_000 + data_cfg.seed * 1000
    for k in range(data_cfg.n_random_episodes):
        run_episode(base + k, scripted=False)
    for k in range(data_cfg.n_scripted_episodes):
        run_episode(base + 500_000 + k, scripted=True)

    env.close()
    return episodes


def sample_batch(episodes: List[Episode], batch_size: int, seq_len: int,
                 num_actions: int, rng: np.random.Generator) -> dict:
    """Sample fixed-length windows with right-padding + masks.

    Returns numpy arrays:
      obs:    (B, L+1, C, H, W) float
      action: (B, L) int, action_onehot: (B, L, A)
      reward: (B, L) float
      mask:   (B, L) 1 for valid transitions (loss masking)
    Padding repeats the last frame and uses mask=0 so short scripted episodes
    (which carry the goal reward) can still be used.
    """
    L = seq_len
    C, H, W = episodes[0].frames[0].shape
    obs = np.zeros((batch_size, L + 1, C, H, W), dtype=np.float32)
    action = np.zeros((batch_size, L), dtype=np.int64)
    reward = np.zeros((batch_size, L), dtype=np.float32)
    mask = np.zeros((batch_size, L), dtype=np.float32)

    lengths = np.array([len(e) for e in episodes], dtype=np.float64)
    probs = lengths / lengths.sum()
    for b in range(batch_size):
        ep = episodes[rng.choice(len(episodes), p=probs)]
        T = len(ep)
        start = int(rng.integers(0, max(1, T)))  # transition start
        k = min(L, T - start)
        f, a, r, _ = ep.as_arrays()
        f = f.astype(np.float32) / 255.0     # uint8 -> float [0,1]
        obs[b, : k + 1] = f[start : start + k + 1]
        if k + 1 <= L:
            obs[b, k + 1 :] = f[start + k]  # pad frames with last valid
        action[b, :k] = a[start : start + k]
        reward[b, :k] = r[start : start + k]
        mask[b, :k] = 1.0

    action_onehot = np.eye(num_actions, dtype=np.float32)[action]
    return {
        "obs": obs,
        "action": action,
        "action_onehot": action_onehot,
        "reward": reward,
        "mask": mask,
    }


def episode_stats(episodes: List[Episode]) -> dict:
    lengths = [len(e) for e in episodes]
    successes = [1.0 if (len(e) and e.rewards[-1] > 0) else 0.0 for e in episodes]
    total_transitions = int(sum(lengths))
    return {
        "n_episodes": len(episodes),
        "transitions": total_transitions,
        "mean_len": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
    }
