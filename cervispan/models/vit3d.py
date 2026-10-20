"""Flexible-patch 3D Vision Transformer encoder used by CerviSpan."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from ..configs.config import CFG


def position_3d(grid: tuple[int, int, int], width: int, device, dtype):
    # Additive sinusoidal positional encoding for the three spatial axes.
    if width % 6:
        raise ValueError("embedding width must be divisible by six")
    axes = torch.meshgrid(*[torch.arange(n, device=device, dtype=torch.float32) for n in grid], indexing="ij")
    channels = width // 6
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(channels, device=device) / max(channels - 1, 1))
    pieces = []
    for coordinate in axes:
        phase = coordinate.reshape(-1, 1) * frequencies
        pieces.extend((phase.sin(), phase.cos()))
    return torch.cat(pieces, dim=-1)[None].to(dtype)


class FlexiblePatchEmbed(nn.Module):
    def __init__(self, width: int, base_patch: int = 8):
        super().__init__()
        self.base_patch = base_patch
        self.weight = nn.Parameter(torch.empty(width, 1, base_patch, base_patch, base_patch))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, image: torch.Tensor, patch: int) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if any(n % patch for n in image.shape[-3:]):
            raise ValueError("All image dimensions must be divisible by patch size")
        weight = self.weight
        if patch != self.base_patch:
            weight = F.interpolate(weight, size=(patch,) * 3, mode="trilinear", align_corners=False)
            weight = weight * (self.base_patch / patch) ** 1.5
        feature = F.conv3d(image, weight, self.bias, stride=patch)
        return feature.flatten(2).transpose(1, 2), tuple(feature.shape[-3:])


class Block(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))
        self.heads = heads

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        q, k, v = self.qkv(self.norm1(tokens)).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        tokens = tokens + self.proj(attended)
        return tokens + self.mlp(self.norm2(tokens))


class CerviSpanEncoder(nn.Module):
    def __init__(self, width: int = CFG.embed_dim, depth: int = CFG.blocks, heads: int = CFG.attention_heads,
                 mlp_ratio: float = CFG.mlp_ratio):
        super().__init__()
        self.patch_embed = FlexiblePatchEmbed(width)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList([Block(width, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward_features(self, image: torch.Tensor, patch: int = 16, mask: torch.Tensor | None = None,
                         stages: bool = False) -> dict:
        tokens, grid = self.patch_embed(image, patch)
        if mask is not None:
            if mask.shape != tokens.shape[:2]:
                raise ValueError("mask shape must equal [batch, patches]")
            tokens = torch.where(mask[..., None], self.mask_token.to(tokens.dtype), tokens)
        tokens = tokens + position_3d(grid, tokens.shape[-1], tokens.device, tokens.dtype)
        tokens = torch.cat([self.cls_token.expand(len(tokens), -1, -1), tokens], dim=1)
        intermediate = []
        for index, block in enumerate(self.blocks, 1):
            tokens = block(tokens)
            if stages and index in (4, 8, 12, 16):
                intermediate.append(self.norm(tokens[:, 1:]))
        tokens = self.norm(tokens)
        return {"cls": tokens[:, 0], "patches": tokens[:, 1:], "grid": grid, "stages": intermediate}

    def forward(self, image: torch.Tensor, patch: int = 16) -> torch.Tensor:
        return self.forward_features(image, patch)["cls"]


def load_encoder(path, device="cpu") -> CerviSpanEncoder:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["architecture"]
    model = CerviSpanEncoder(cfg["width"], cfg["depth"], cfg["heads"], cfg["mlp_ratio"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def save_encoder(model: CerviSpanEncoder, path, extra: dict | None = None):
    torch.save({"architecture": {"width": model.cls_token.shape[-1], "depth": len(model.blocks),
                                 "heads": model.blocks[0].heads,
                                 "mlp_ratio": model.blocks[0].mlp[0].out_features / model.cls_token.shape[-1]},
                "state_dict": model.state_dict(), "extra": extra or {}}, path)