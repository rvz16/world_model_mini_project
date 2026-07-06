"""World model: encoder + RSSM + decoder + reward head, with loss + imagination.

Training objective (Dreamer/PlaNet ELBO):
    L = recon_nll(frames) + reward_scale * reward_nll + kl_scale * KL(post||prior)
KL uses free-nats. Everything is masked so right-padded short episodes contribute
only their valid steps.
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.distributions as D

from .networks import ConvEncoder, ConvDecoder, RewardHead
from .rssm import RSSM


class WorldModel(nn.Module):
    def __init__(self, action_dim: int, rssm_cfg, image_size: int = 64):
        super().__init__()
        self.encoder = ConvEncoder(image_size=image_size, embed_dim=rssm_cfg.embed_dim)
        self.rssm = RSSM(
            action_dim=action_dim,
            deter_dim=rssm_cfg.deter_dim,
            stoch_dim=rssm_cfg.stoch_dim,
            hidden_dim=rssm_cfg.hidden_dim,
            embed_dim=rssm_cfg.embed_dim,
        )
        self.decoder = ConvDecoder(self.rssm.feat_dim, image_size=image_size)
        self.reward_head = RewardHead(self.rssm.feat_dim, rssm_cfg.hidden_dim)
        self.cfg = rssm_cfg
        self.action_dim = action_dim

    # -- encoding helpers ----------------------------------------------------
    def encode_frames(self, frames):
        """(B, T, C, H, W) -> (B, T, embed)."""
        B, T = frames.shape[:2]
        flat = frames.reshape(B * T, *frames.shape[2:])
        emb = self.encoder(flat)
        return emb.reshape(B, T, -1)

    def decode_feats(self, feats):
        """(B, T, feat) -> (B, T, C, H, W)."""
        B, T = feats.shape[:2]
        out = self.decoder(feats.reshape(B * T, -1))
        return out.reshape(B, T, *out.shape[1:])

    # -- training ------------------------------------------------------------
    def loss(self, batch: Dict[str, torch.Tensor]):
        obs = batch["obs"]            # (B, L+1, C,H,W)
        action = batch["action_onehot"]  # (B, L, A)
        reward = batch["reward"]      # (B, L)
        mask = batch["mask"]          # (B, L)  valid transitions
        B, Lp1 = obs.shape[:2]
        L = Lp1 - 1

        # frame validity mask: frame 0 always valid; frame t valid iff transition t-1 valid
        obs_mask = torch.cat([torch.ones(B, 1, device=obs.device), mask], dim=1)  # (B, L+1)

        embed = self.encode_frames(obs)                       # (B, L+1, embed)
        action_full = torch.cat(
            [action, torch.zeros(B, 1, self.action_dim, device=obs.device)], dim=1)
        feats, prior_dists, post_dists, _ = self.rssm.observe(embed, action_full)

        # reconstruction NLL (unit-variance Gaussian == 0.5 * SE), per valid frame
        recon = self.decode_feats(feats)                      # (B, L+1, C,H,W)
        se = ((recon - obs) ** 2).flatten(2).sum(-1)          # (B, L+1)
        recon_nll = (0.5 * se * obs_mask).sum() / obs_mask.sum().clamp(min=1)

        # reward NLL: predict reward[t] (transition t) from state at t+1
        reward_pred = self.reward_head(feats[:, 1:])          # (B, L)
        rew_se = (reward_pred - reward) ** 2
        reward_nll = (0.5 * rew_se * mask).sum() / mask.sum().clamp(min=1)

        # KL(post || prior) with free nats, steps t=1..L
        kl_per_step = []
        for post, prior in zip(post_dists, prior_dists):
            kl = D.kl_divergence(post, prior).sum(-1)         # (B,)
            kl_per_step.append(kl)
        kl = torch.stack(kl_per_step, dim=1)                  # (B, L)
        kl = torch.clamp(kl, min=self.cfg.free_nats)
        kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1)

        total = recon_nll + self.cfg.kl_scale * kl_loss + reward_nll
        metrics = {
            "loss": total.item(),
            "recon": recon_nll.item(),
            "reward": reward_nll.item(),
            "kl": kl_loss.item(),
        }
        return total, metrics

    # -- inference for planning ---------------------------------------------
    @torch.no_grad()
    def init_state_from_obs(self, frame):
        """Single frame (1, C, H, W) -> posterior state dict (batch 1)."""
        emb = self.encoder(frame)                             # (1, embed)
        state = self.rssm.initial(1, frame.device)
        zero_a = torch.zeros(1, self.action_dim, device=frame.device)
        state, _, _ = self.rssm.obs_step(state, zero_a, emb)
        return state

    @torch.no_grad()
    def imagine_rollouts(self, init_state, action_seqs):
        """init_state: dict batch 1; action_seqs: (N, H, A).

        Returns dict with feats (N,H,feat), frames (N,H,C,H,W), rewards (N,H).
        """
        N, H, _ = action_seqs.shape
        state = {k: v.expand(N, -1).contiguous() for k, v in init_state.items()}
        feats = self.rssm.imagine(state, action_seqs)         # (N, H, feat)
        frames = self.decode_feats(feats)
        rewards = self.reward_head(feats)
        return {"feats": feats, "frames": frames, "rewards": rewards}
