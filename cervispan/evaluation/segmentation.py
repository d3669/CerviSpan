"""Evaluate CTV Dice per patient and cohort with patient-level bootstrap CI."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import rows_from_manifest
from ..models.vit3d import load_encoder
from ..downstream.segmentation import CTVModel, dice_score, sliding_predict


def percentile_interval(values, replicates=20000, seed=42):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for index in range(replicates):
        means[index] = values[rng.integers(0, len(values), len(values))].mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=20000)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CTVModel(load_encoder(a.checkpoint, device)).to(device)
    payload = torch.load(a.decoder, map_location="cpu", weights_only=False)
    mismatch = model.load_state_dict(payload["decoder"], strict=False)
    if mismatch.unexpected_keys or any(not key.startswith("encoder.") for key in mismatch.missing_keys):
        raise RuntimeError(f"Incompatible decoder: {mismatch}")
    model.eval()
    scores = []
    for row in rows_from_manifest(a.manifest):
        if row["split"] in ("train", "pretrain"):
            continue
        with np.load(row["npz_path"]) as file:
            if "mask" not in file:
                continue
            image, truth = file["ct"].astype(np.float32), file["mask"]
        probability = sliding_predict(model, image, device)
        scores.append({"case_id": row["case_id"], "split": row["split"], "dice": dice_score(probability, truth)})
    if not scores:
        raise ValueError("No labelled test cases")
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / "per_case.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "split", "dice"])
        writer.writeheader()
        writer.writerows(scores)
    summary = {}
    for split in sorted({row["split"] for row in scores}):
        values = [row["dice"] for row in scores if row["split"] == split]
        summary[split] = {"n": len(values), "mean_dice": float(np.mean(values)),
                          "ci_95": percentile_interval(values, a.bootstrap) if a.bootstrap else None}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()