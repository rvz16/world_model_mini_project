"""Recurrent State-Space Model (PlaNet / Dreamer style), Gaussian stochastic.

State = (deterministic h from a GRU, stochastic z ~ Normal). The model supports:
  * obs_step : prior(h,z,a) then posterior(h, obs-embed)  -- used with observations
  * img_step : prior only                                 -- used for imagination
The concatenation [h, z] is the feature consumed by decoder / reward head /
the VLM-scoring decode.
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.distributions as D


class RSSM(nn.Module):
    def __init__(self, action_dim: int, deter_dim=128, stoch_dim=32,
                 hidden_dim=128, embed_dim=256, min_std=0.1):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.min_std = min_std

        self.fc_input = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.fc_prior = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim))
        self.fc_post = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim))

    @property
    def feat_dim(self):
        return self.deter_dim + self.stoch_dim

    def initial(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        return {
            "deter": torch.zeros(batch_size, self.deter_dim, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    def feat(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    def _dist(self, params) -> D.Normal:
        mean, std = torch.chunk(params, 2, dim=-1)
        std = torch.nn.functional.softplus(std) + self.min_std
        return D.Normal(mean, std)

    def _prior(self, prev_state, prev_action) -> Tuple[Dict, D.Normal]:
        x = torch.cat([prev_state["stoch"], prev_action], dim=-1)
        x = self.fc_input(x)
        deter = self.gru(x, prev_state["deter"])
        dist = self._dist(self.fc_prior(deter))
        stoch = dist.rsample()
        return {"deter": deter, "stoch": stoch}, dist

    def img_step(self, prev_state, prev_action):
        state, _ = self._prior(prev_state, prev_action)
        return state

    def obs_step(self, prev_state, prev_action, embed):
        prior_state, prior_dist = self._prior(prev_state, prev_action)
        x = torch.cat([prior_state["deter"], embed], dim=-1)
        post_dist = self._dist(self.fc_post(x))
        stoch = post_dist.rsample()
        post_state = {"deter": prior_state["deter"], "stoch": stoch}
        return post_state, prior_dist, post_dist

    def observe(self, embed_seq, action_seq, mask_first=None):
        """Roll the posterior over a sequence.

        embed_seq:  (B, T, embed)   observation embeddings o_0..o_{T-1}
        action_seq: (B, T, A)       action taken AT each step (a_t at o_t);
                                    a_{t-1} is the previous action used at step t.
        Returns stacked posterior feats and lists of prior/post dists (t>=1).
        """
        B, T, _ = embed_seq.shape
        device = embed_seq.device
        state = self.initial(B, device)
        zero_action = torch.zeros_like(action_seq[:, 0])
        feats, prior_dists, post_dists = [], [], []
        prev_action = zero_action
        for t in range(T):
            state, prior_dist, post_dist = self.obs_step(state, prev_action, embed_seq[:, t])
            feats.append(self.feat(state))
            if t >= 1:
                prior_dists.append(prior_dist)
                post_dists.append(post_dist)
            prev_action = action_seq[:, t]
            last_state = state
        feats = torch.stack(feats, dim=1)  # (B,T,feat)
        return feats, prior_dists, post_dists, last_state

    def imagine(self, init_state, action_seq):
        """Roll the prior forward under a candidate action sequence.

        init_state: dict of (B, ...) state to start from.
        action_seq: (B, H, A) actions a_0..a_{H-1}.
        Returns feats (B, H, feat) for the H imagined next-states.
        """
        H = action_seq.shape[1]
        state = init_state
        feats = []
        for t in range(H):
            state = self.img_step(state, action_seq[:, t])
            feats.append(self.feat(state))
        return torch.stack(feats, dim=1)
