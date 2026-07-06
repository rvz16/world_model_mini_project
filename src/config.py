"""Central configuration for the world-model + VLM-scorer demo.

All knobs live here so the project can be scaled between a laptop (MPS/CPU)
and a GPU box without touching the code. Defaults are tuned for a Mac/MPS.
"""
from dataclasses import dataclass, field, asdict
import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class EnvConfig:
    env_id: str = "MiniGrid-Empty-6x6-v0"
    image_size: int = 128          # square CHW float[0,1]; must be power of two.
    # 128px is needed so the VLM can read the goal off decoded frames: at 64px the
    # green-occlusion goal cue is too small for CLIP (goal-detection AUC ~0.36),
    # while at higher resolution it is reliable.
    max_steps: int = 64            # episode cap used for data + eval
    # MiniGrid action subset that actually matters for Empty: 0=left,1=right,2=forward
    action_set: tuple = (0, 1, 2)


@dataclass
class DataConfig:
    n_random_episodes: int = 250   # random-policy rollouts
    n_scripted_episodes: int = 150 # noisy shortest-path rollouts (cover goal states)
    scripted_noise: float = 0.25   # prob of a random action in scripted rollouts
    seed: int = 0


@dataclass
class RSSMConfig:
    deter_dim: int = 128           # GRU hidden (deterministic state h)
    stoch_dim: int = 32            # stochastic latent z (Gaussian)
    hidden_dim: int = 128          # MLP width inside RSSM
    embed_dim: int = 256           # CNN encoder output
    free_nats: float = 1.0         # KL free bits
    kl_scale: float = 1.0


@dataclass
class TrainConfig:
    batch_size: int = 32
    seq_len: int = 20              # training sequence length (BPTT window)
    steps: int = 6000              # gradient steps
    lr: float = 3e-4
    grad_clip: float = 100.0
    recon_scale: float = 1.0
    reward_scale: float = 1.0
    log_every: int = 100
    ckpt_every: int = 500          # periodic checkpoint (interrupted runs stay usable)
    ckpt_path: str = "outputs/checkpoints/wm.pt"


@dataclass
class PlanConfig:
    horizon: int = 10              # imagined rollout length H
    num_candidates: int = 100      # random-shooting samples per step
    vlm_num_candidates: int = 24   # fewer candidates in VLM mode (CLIP is the bottleneck)
    cem: bool = False              # use CEM refinement instead of plain shooting
    cem_iters: int = 3
    cem_elite_frac: float = 0.1
    reward_weight: float = 1.0     # weight on WM-predicted reward in objective
    vlm_weight: float = 10.0       # weight on VLM score in objective
    gamma: float = 0.95            # discount over the horizon


@dataclass
class VLMConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    # NOTE on prompt design: CLIP cannot do the spatial relation "agent ON the
    # goal" on abstract MiniGrid renders (bag-of-concepts, no spatial binding).
    # The reliable visual cue is that the green goal tile becomes occluded by the
    # agent once reached. So we frame the goal as "green is gone". Validated to
    # give pair_acc=1.00 separating goal vs non-goal frames at 64px and 192px.
    goal_prompt: str = "a dark maze with no green"
    neg_prompt: str = "a green goal square in a maze"
    use_contrast: bool = True      # score = sim(goal) - sim(neg)


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    plan: PlanConfig = field(default_factory=PlanConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    device: str = field(default_factory=get_device)

    def to_dict(self):
        return asdict(self)


DEFAULT = Config()
