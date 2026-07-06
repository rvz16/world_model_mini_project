"""MPC planning over imagined rollouts: random shooting (+ optional CEM).

At each environment step we:
  1. encode the current observation into an RSSM state,
  2. sample N candidate action sequences of length H,
  3. imagine them forward in latent space and decode the future frames,
  4. score each rollout (WM reward and/or VLM goal score on the *future* frames),
  5. take the first action of the best sequence (receding horizon).

Three objective modes:
  * "random" : ignore the model, act uniformly at random (baseline)
  * "wm"     : objective = discounted WM-predicted reward only (baseline)
  * "wm_vlm" : objective = discounted WM reward + VLM goal score on imagined frames
"""
import numpy as np
import torch

from .config import PlanConfig


class Planner:
    def __init__(self, cfg: PlanConfig, num_actions: int, device: str):
        self.cfg = cfg
        self.num_actions = num_actions
        self.device = device
        h = cfg.horizon
        self.discount = (cfg.gamma ** torch.arange(h, dtype=torch.float32))  # (H,)

    def _onehot(self, idx):
        return torch.eye(self.num_actions, device=self.device)[idx]

    def _objective(self, rollout, scorer, use_vlm):
        # Clean, non-overlapping objectives so the comparison isolates each signal:
        #   wm     mode: objective = WM-predicted reward only
        #   wm_vlm mode: objective = VLM goal score only (no reward) -- tests whether
        #               the VLM alone can drive planning, which is the point of the task.
        per_step_vlm = None
        if use_vlm and scorer is not None:
            per_step_vlm = scorer.score_rollout(rollout["frames"])  # (N, H)
            obj = self.cfg.vlm_weight * per_step_vlm
        else:
            obj = self.cfg.reward_weight * rollout["rewards"].cpu()  # (N, H)
        total = (obj * self.discount).sum(dim=1)           # (N,)
        return total, per_step_vlm

    def _sample_uniform(self, n, rng):
        idx = torch.from_numpy(
            rng.integers(0, self.num_actions, size=(n, self.cfg.horizon))).long()
        return idx.to(self.device)

    def plan(self, wm, scorer, init_state, mode, rng):
        """Return (best_first_action_idx:int, info:dict)."""
        if mode == "random":
            return int(rng.integers(0, self.num_actions)), {}

        use_vlm = (mode == "wm_vlm")
        # CLIP scoring dominates cost in VLM mode, so use fewer candidates there
        N = self.cfg.vlm_num_candidates if use_vlm else self.cfg.num_candidates
        H = self.cfg.horizon

        if not self.cfg.cem:
            idx = self._sample_uniform(N, rng)
            rollout = wm.imagine_rollouts(init_state, self._onehot(idx))
            total, _ = self._objective(rollout, scorer, use_vlm)
            best = int(idx[total.argmax(), 0].item())
            return best, {"best_value": float(total.max())}

        # --- CEM: per-timestep categorical, refit on elites -----------------
        logits = torch.zeros(H, self.num_actions, device=self.device)
        n_elite = max(1, int(self.cfg.cem_elite_frac * N))
        best_overall = None
        for _ in range(self.cfg.cem_iters):
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            samples = np.stack(
                [rng.choice(self.num_actions, size=N, p=probs[t]) for t in range(H)],
                axis=1)  # (N, H)
            idx = torch.from_numpy(samples).long().to(self.device)
            rollout = wm.imagine_rollouts(init_state, self._onehot(idx))
            total, _ = self._objective(rollout, scorer, use_vlm)
            elite = idx[total.topk(n_elite).indices]       # (n_elite, H)
            counts = torch.zeros(H, self.num_actions, device=self.device)
            for t in range(H):
                counts[t] = torch.bincount(elite[:, t], minlength=self.num_actions).float()
            logits = torch.log(counts + 1.0)               # Laplace-smoothed refit
            best_overall = int(elite[0, 0].item())
        return best_overall, {}
