"""Shared configuration for the CerviSpan model and training pipeline."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    # Explicit in manuscript
    spacing_mm: float = 1.5
    hu_min: float = -1000.0
    hu_max: float = 1000.0
    global_size: int = 160
    global_patch: int = 16
    local_size: int = 48
    local_patch: int = 8
    blocks: int = 16
    embed_dim: int = 864
    prototypes: int = 8192
    dino_weight: float = 1.0
    ibot_weight: float = 1.0
    koleo_weight: float = 0.1
    correspondence_weight: float = 0.3
    pretrain_lr: float = 1e-3
    layer_decay: float = 0.9
    head_lr_multiplier: float = 0.2
    patch_lr_multiplier: float = 0.2
    # Explicit choices for missing implementation detail; not author values
    attention_heads: int = 12
    mlp_ratio: float = 4.0
    n_global: int = 2
    n_local: int = 4
    mask_ratio: float = 0.3
    teacher_momentum: float = 0.996
    teacher_temperature: float = 0.04
    student_temperature: float = 0.1
    weight_decay: float = 0.04
    seed: int = 42

    def dictionary(self) -> dict:
        return asdict(self)


CFG = Config()