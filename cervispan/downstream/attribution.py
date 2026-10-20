"""Patch-level Grad-CAM for a trained binary linear probe."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..configs.config import CFG
from ..data.dataset import centered
from ..models.vit3d import load_encoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(a.checkpoint, device).requires_grad_(False)
    payload = torch.load(a.head, map_location="cpu", weights_only=False)
    width = encoder.cls_token.shape[-1]
    if payload["feature_dim"] != 2 * width:
        raise ValueError("Linear head and encoder dimensions differ")
    head = nn.Linear(2 * width, 1).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval().requires_grad_(False)
    captured = {}
    def save_activation(_module, _input, output):
        captured["tokens"] = output
    handle = encoder.blocks[-2].register_forward_hook(save_activation)
    try:
        with np.load(a.image) as file:
            image = centered(file["ct"].astype(np.float32), CFG.global_size)
        tensor = torch.from_numpy(image.copy())[None, None].to(device).requires_grad_(True)
        features = encoder.forward_features(tensor, CFG.global_patch)
        logit = head(torch.cat((features["cls"], features["patches"].mean(1)), dim=1)).squeeze()
        activation = captured["tokens"][:, 1:]
        gradient = torch.autograd.grad(logit, captured["tokens"])[0][:, 1:]
        weights = gradient.mean(dim=1, keepdim=True)
        importance = F.relu((activation * weights).sum(dim=-1))
        grid = features["grid"]
        importance = importance.reshape(1, 1, *grid)
        importance = F.interpolate(importance, size=image.shape, mode="trilinear", align_corners=False)[0, 0]
        heatmap = importance.detach().float().cpu().numpy()
        heatmap /= max(float(heatmap.max()), 1e-8)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.output, ct=image, heatmap=heatmap, probability=float(torch.sigmoid(logit)))
    finally:
        handle.remove()


if __name__ == "__main__":
    main()