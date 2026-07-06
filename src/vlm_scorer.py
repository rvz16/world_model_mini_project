"""VLM-based goal scorer using a pretrained CLIP (open_clip).

Turns an image (or a batch of imagined future frames) into a single scalar that
estimates how well it matches a natural-language goal. By default we use a
*contrastive* score: sim(goal) - sim(negative), which sharpens the signal on the
abstract MiniGrid renders.

The scorer operates on float CHW frames in [0,1] (the same format the world-model
decoder produces), resizing + normalizing to CLIP's expected input internally.
"""
from typing import Optional
import torch
import torch.nn.functional as F
import open_clip
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from .config import VLMConfig


class VLMScorer:
    def __init__(self, cfg: VLMConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            cfg.model_name, pretrained=cfg.pretrained)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(cfg.model_name)

        mean = getattr(self.model.visual, "image_mean", None) or OPENAI_DATASET_MEAN
        std = getattr(self.model.visual, "image_std", None) or OPENAI_DATASET_STD
        self.register_norm(mean, std)
        self.input_res = self.model.visual.image_size
        if isinstance(self.input_res, (tuple, list)):
            self.input_res = self.input_res[0]

        with torch.no_grad():
            self._goal_feat = self._encode_text(cfg.goal_prompt)
            self._neg_feat = self._encode_text(cfg.neg_prompt) if cfg.use_contrast else None

    def register_norm(self, mean, std):
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

    def _encode_text(self, prompt: str) -> torch.Tensor:
        tok = self.tokenizer([prompt]).to(self.device)
        feat = self.model.encode_text(tok)
        return F.normalize(feat, dim=-1)

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (N, 3, H, W) float, possibly outside [0,1] (decoder output)
        frames = frames.clamp(0, 1).to(self.device)
        frames = F.interpolate(frames, size=(self.input_res, self.input_res),
                               mode="bilinear", align_corners=False)
        return (frames - self.mean) / self.std

    @torch.no_grad()
    def score(self, frames: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """frames: (N, 3, H, W) in [0,1] -> (N,) goal-achievement scores (CPU).

        Processed in chunks so large candidate batches (planning) don't OOM.
        """
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        outs = []
        for i in range(0, frames.shape[0], chunk):
            x = self._preprocess(frames[i:i + chunk])
            img = F.normalize(self.model.encode_image(x), dim=-1)
            s = img @ self._goal_feat.T  # (n,1)
            if self._neg_feat is not None:
                s = s - img @ self._neg_feat.T
            outs.append(s.squeeze(-1).float().cpu())
        return torch.cat(outs)

    @torch.no_grad()
    def score_rollout(self, frames_nh: torch.Tensor) -> torch.Tensor:
        """frames_nh: (N, H, 3, Hgt, Wd) -> (N, H) per-frame scores (CPU)."""
        N, H = frames_nh.shape[:2]
        flat = frames_nh.reshape(N * H, *frames_nh.shape[2:])
        return self.score(flat).reshape(N, H)
