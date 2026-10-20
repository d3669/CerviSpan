"""Frozen-encoder four-stage CTV decoder, five-fold selection, Gaussian window inference."""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

from ..configs.config import CFG
from ..data.dataset import crop, rows_from_manifest
from ..models.vit3d import load_encoder


def conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding=1), nn.InstanceNorm3d(out_channels), nn.ReLU(inplace=True))


class CTVModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        width = encoder.cls_token.shape[-1]
        self.projections = nn.ModuleList([nn.Conv3d(width, channels, 1) for channels in (32, 64, 128, 256)])
        self.up3, self.up2, self.up1 = (nn.ConvTranspose3d(a, b, 2, stride=2) for a, b in ((256, 128), (128, 64), (64, 32)))
        self.fuse3, self.fuse2, self.fuse1 = conv(256, 128), conv(128, 64), conv(64, 32)
        self.shallow = conv(1, 32)
        self.head = nn.Sequential(conv(64, 32), nn.Conv3d(32, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, image):
        if any(size % CFG.global_patch for size in image.shape[-3:]):
            raise ValueError("Input dimensions must be multiples of 16")
        with torch.no_grad():
            source = self.encoder.forward_features(image, CFG.global_patch, stages=True)
        if len(source["stages"]) != 4:
            raise RuntimeError("Expected four intermediate encoder stages")
        grid = source["grid"]
        features = []
        for index, (tokens, layer) in enumerate(zip(source["stages"], self.projections)):
            volume = tokens.transpose(1, 2).reshape(len(image), tokens.shape[-1], *grid)
            target_size = tuple(size // (2 ** (index + 1)) for size in image.shape[-3:])
            features.append(F.interpolate(layer(volume), size=target_size, mode="trilinear", align_corners=False))
        a, b, c, d = features
        x = self.fuse3(torch.cat((self.up3(d), c), dim=1))
        x = self.fuse2(torch.cat((self.up2(x), b), dim=1))
        x = self.fuse1(torch.cat((self.up1(x), a), dim=1))
        x = F.interpolate(x, size=image.shape[-3:], mode="trilinear", align_corners=False)
        return self.head(torch.cat((x, self.shallow(image)), dim=1))


def dice_ce(logits, truth):
    probability = torch.sigmoid(logits)
    axes = tuple(range(2, truth.ndim))
    dice = (2 * (probability * truth).sum(axes) + 1) / (probability.sum(axes) + truth.sum(axes) + 1)
    return 1 - dice.mean() + F.binary_cross_entropy_with_logits(logits, truth)


def dice_score(probability, truth):
    predicted = probability >= 0.5
    truth = truth > 0
    return float((2 * (predicted & truth).sum() + 1) / (predicted.sum() + truth.sum() + 1))


def patch_from_case(path, rng, positive=True):
    with np.load(path) as file:
        ct = file["ct"].astype(np.float32)
        if "mask" not in file:
            raise ValueError(f"No CTV mask in {path}")
        mask = file["mask"].astype(np.float32)
    foreground = np.argwhere(mask > 0)
    if positive and len(foreground):
        center = foreground[int(rng.integers(len(foreground)))]
    else:
        center = np.array([rng.integers(n) for n in ct.shape])
    start = tuple(int(c - CFG.global_size // 2) for c in center)
    return (torch.from_numpy(crop(ct, start, CFG.global_size).copy())[None, None],
            torch.from_numpy(crop(mask, start, CFG.global_size).copy())[None, None])


def fit(model, rows, epochs, seed, device):
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        for index in rng.permutation(len(rows)):
            image, truth = patch_from_case(rows[index]["npz_path"], rng)
            optimizer.zero_grad(set_to_none=True)
            loss = dice_ce(model(image.to(device)), truth.to(device))
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch + 1} training_loss={float(loss):.5f}", flush=True)
    return model.eval()


@torch.inference_mode()
def sliding_predict(model, volume, device, size=160):
    original_shape = volume.shape
    volume = np.pad(volume, [(0, max(0, size - n)) for n in original_shape])
    positions = []
    for dimension in volume.shape:
        values = list(range(0, max(dimension - size + 1, 1), size // 2))
        if values[-1] != dimension - size:
            values.append(dimension - size)
        positions.append(values)
    axes = np.meshgrid(*[np.linspace(-1, 1, size)] * 3, indexing="ij")
    weights = np.exp(-sum(a * a for a in axes) / 0.5).astype(np.float32)
    numerator = np.zeros(volume.shape, dtype=np.float32)
    denominator = np.zeros(volume.shape, dtype=np.float32)
    for start in product(*positions):
        selection = tuple(slice(i, i + size) for i in start)
        tensor = torch.from_numpy(volume[selection].copy())[None, None].to(device)
        probability = torch.sigmoid(model(tensor))[0, 0].float().cpu().numpy()
        numerator[selection] += probability * weights
        denominator[selection] += weights
    full = numerator / np.maximum(denominator, 1e-8)
    return full[tuple(slice(0, n) for n in original_shape)]


def train_main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("160³ CTV training requires a CUDA GPU")
    rows = [r for r in rows_from_manifest(args.manifest) if r["split"] == "train"]
    if len(rows) < 10:
        raise ValueError("At least ten training patients required for five folds")
    device = torch.device("cuda")
    encoder = load_encoder(args.checkpoint, device)
    folds = list(KFold(5, shuffle=True, random_state=args.seed).split(rows))
    selection = {}
    for epochs in args.epochs:
        values = []
        for fold, (train_index, validation_index) in enumerate(folds):
            model = fit(CTVModel(encoder), [rows[i] for i in train_index], epochs, args.seed + fold, device)
            fold_dice = []
            for index in validation_index:
                with np.load(rows[index]["npz_path"]) as file:
                    ct, truth = file["ct"], file["mask"]
                probability = sliding_predict(model, ct.astype(np.float32), device)
                fold_dice.append(dice_score(probability, truth))
            values.append(float(np.mean(fold_dice)))
            del model
            torch.cuda.empty_cache()
        selection[epochs] = values
    chosen = max(selection, key=lambda value: np.mean(selection[value]))
    final = fit(CTVModel(encoder), rows, chosen, args.seed, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.cpu() for key, value in final.state_dict().items() if not key.startswith("encoder.")}
    torch.save({"decoder": state, "selected_epochs": chosen, "cv_dice": selection}, args.output)


def predict_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(args.checkpoint, device)
    model = CTVModel(encoder).to(device)
    payload = torch.load(args.decoder, map_location="cpu", weights_only=False)
    mismatch = model.load_state_dict(payload["decoder"], strict=False)
    if mismatch.unexpected_keys or any(not key.startswith("encoder.") for key in mismatch.missing_keys):
        raise RuntimeError(f"Incompatible decoder state: {mismatch}")
    model.eval()
    with np.load(args.image) as file:
        image = file["ct"].astype(np.float32)
        truth = file["mask"] if "mask" in file else None
    probability = sliding_predict(model, image, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, probability=probability, mask=(probability >= 0.5).astype(np.uint8))
    if truth is not None:
        print(f"Dice={dice_score(probability, truth):.4f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", nargs="+", type=int, default=[10, 20, 40], help="Candidate training epochs for cross-validation")
    train.add_argument("--seed", type=int, default=42)
    predict = sub.add_parser("predict")
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--decoder", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    (train_main if args.command == "train" else predict_main)(args)


if __name__ == "__main__":
    main()