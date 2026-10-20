"""Frozen encoder extraction: layer-normalized CLS concatenated with mean patches."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..configs.config import CFG
from ..data.dataset import centered, rows_from_manifest
from ..models.vit3d import load_encoder


@torch.inference_mode()
def encode(model, volume: np.ndarray, device: torch.device):
    image = torch.from_numpy(centered(volume, CFG.global_size).copy())[None, None].to(device)
    features = model.forward_features(image, CFG.global_patch)
    return torch.cat((features["cls"], features["patches"].mean(1)), dim=1)[0].float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    rows = [r for r in rows_from_manifest(a.manifest) if r["split"] != "pretrain"]
    if not rows:
        raise ValueError("No downstream patients")
    device = torch.device(a.device)
    model = load_encoder(a.checkpoint, device).requires_grad_(False)
    vectors = []
    for row in rows:
        with np.load(row["npz_path"]) as file:
            vectors.append(encode(model, file["ct"].astype(np.float32), device))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, X=np.stack(vectors), case_id=np.array([r["case_id"] for r in rows]),
                        split=np.array([r["split"] for r in rows]))


if __name__ == "__main__":
    main()