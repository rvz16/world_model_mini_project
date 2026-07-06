"""Train the RSSM world model on collected MiniGrid rollouts."""
import argparse
import os
import pickle
import time
import numpy as np
import torch

from .config import Config
from .data import collect_dataset, sample_batch, episode_stats
from .models import WorldModel


def to_tensors(batch, device):
    return {
        "obs": torch.from_numpy(batch["obs"]).to(device),
        "action_onehot": torch.from_numpy(batch["action_onehot"]).to(device),
        "reward": torch.from_numpy(batch["reward"]).to(device),
        "mask": torch.from_numpy(batch["mask"]).to(device),
    }


def get_or_build_dataset(cfg: Config, cache="outputs/dataset.pkl"):
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            episodes = pickle.load(f)
        print(f"[data] loaded cached dataset: {cache}")
    else:
        print("[data] collecting rollouts ...")
        episodes = collect_dataset(cfg.env, cfg.data)
        with open(cache, "wb") as f:
            pickle.dump(episodes, f)
    print("[data]", episode_stats(episodes))
    return episodes


def train(cfg: Config):
    device = cfg.device
    print(f"[train] device = {device}")
    torch.manual_seed(cfg.data.seed)
    np.random.seed(cfg.data.seed)
    rng = np.random.default_rng(cfg.data.seed)

    episodes = get_or_build_dataset(cfg)
    num_actions = len(cfg.env.action_set)

    wm = WorldModel(num_actions, cfg.rssm, image_size=cfg.env.image_size).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=cfg.train.lr)

    t0 = time.time()
    for step in range(1, cfg.train.steps + 1):
        batch = sample_batch(episodes, cfg.train.batch_size, cfg.train.seq_len,
                             num_actions, rng)
        batch = to_tensors(batch, device)
        loss, metrics = wm.loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg.train.grad_clip)
        opt.step()

        if step % cfg.train.log_every == 0 or step == 1:
            dt = time.time() - t0
            print(f"[{step:5d}/{cfg.train.steps}] "
                  f"loss={metrics['loss']:.3f} recon={metrics['recon']:.3f} "
                  f"reward={metrics['reward']:.4f} kl={metrics['kl']:.3f} "
                  f"({dt:.0f}s)")

        # periodic checkpoint so an interrupted run is still usable / resumable
        if step % cfg.train.ckpt_every == 0:
            save_ckpt(wm, cfg, step)

    save_ckpt(wm, cfg, cfg.train.steps)
    return wm


def save_ckpt(wm, cfg, step):
    os.makedirs(os.path.dirname(cfg.train.ckpt_path), exist_ok=True)
    torch.save({"model": wm.state_dict(), "cfg": cfg.to_dict(), "step": step},
               cfg.train.ckpt_path)
    print(f"[train] saved checkpoint @ step {step} -> {cfg.train.ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    cfg = Config()
    if args.steps is not None:
        cfg.train.steps = args.steps
    train(cfg)
