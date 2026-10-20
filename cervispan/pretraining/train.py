"""Standalone DINO/iBOT/KoLeo/spatial-correspondence training from the paper."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..configs.config import CFG
from ..data.dataset import PretrainDataset, rows_from_manifest
from ..models.vit3d import CerviSpanEncoder, save_encoder


class ProjectionHead(nn.Module):
    def __init__(self, width: int, prototypes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 2048), nn.GELU(), nn.Linear(2048, 256), nn.GELU(), nn.Linear(256, prototypes))

    def forward(self, tokens):
        return self.net(tokens)


@torch.no_grad()
def sinkhorn(logits: torch.Tensor, temperature: float = 0.04, iterations: int = 3) -> torch.Tensor:
    # Single-process Sinkhorn-Knopp. Distributed all-reduce is an unreported implementation choice.
    q = torch.exp((logits.float() - logits.float().max()) / temperature).T
    q = q / q.sum().clamp_min(1e-12)
    k, batch = q.shape
    for _ in range(iterations):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12) / k
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12) / batch
    return (q * batch).T


def dino_ce(student_logits, teacher_target, student_temperature=0.1):
    return -(teacher_target * F.log_softmax(student_logits / student_temperature, dim=-1)).sum(dim=-1).mean()


def koleo_loss(features: torch.Tensor) -> torch.Tensor:
    if len(features) < 2:
        return features.sum() * 0
    features = F.normalize(features.float(), dim=-1)
    distance = torch.cdist(features, features)
    distance = distance.masked_fill(torch.eye(len(features), device=features.device, dtype=torch.bool), float("inf"))
    return -distance.min(dim=1).values.clamp_min(1e-6).log().mean()


def correspondence_loss(local: torch.Tensor, global_tokens: torch.Tensor,
                        local_box: torch.Tensor, global_box: torch.Tensor,
                        local_grid: tuple[int, int, int], global_grid: tuple[int, int, int]):
    batch, _, channels = local.shape
    local_feature = local.transpose(1, 2).reshape(batch, channels, *local_grid)
    global_feature = global_tokens.transpose(1, 2).reshape(batch, channels, *global_grid)
    axes = [torch.arange(n, device=local.device, dtype=torch.float32) for n in local_grid]
    cells = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    local_start, local_end = local_box[:, 0], local_box[:, 1]
    global_start, global_end = global_box[:, 0], global_box[:, 1]
    position = local_start[:, None, None, None] + (cells[None] + 0.5) * (
        (local_end - local_start)[:, None, None, None] / torch.tensor(local_grid, device=local.device))
    normalized = 2 * (position - global_start[:, None, None, None]) / (
        global_end - global_start)[:, None, None, None] - 1
    inside = ((normalized >= -1) & (normalized <= 1)).all(-1)
    sampled = F.grid_sample(global_feature, normalized[..., [2, 1, 0]], mode="bilinear", align_corners=False)
    cosine = F.cosine_similarity(local_feature, sampled, dim=1)
    return (1 - cosine)[inside].mean() if inside.any() else local.sum() * 0


class SelfSupervised(nn.Module):
    def __init__(self, encoder: CerviSpanEncoder, prototypes: int = CFG.prototypes,
                 local_patch: int = CFG.local_patch, correspondence_weight: float = CFG.correspondence_weight):
        super().__init__()
        self.local_patch = local_patch
        self.correspondence_weight = correspondence_weight
        self.student = encoder
        self.teacher = copy.deepcopy(encoder).eval().requires_grad_(False)
        width = encoder.cls_token.shape[-1]
        self.dino_student = ProjectionHead(width, prototypes)
        self.dino_teacher = copy.deepcopy(self.dino_student).eval().requires_grad_(False)
        self.ibot_student = ProjectionHead(width, prototypes)
        self.ibot_teacher = copy.deepcopy(self.ibot_student).eval().requires_grad_(False)

    @torch.no_grad()
    def ema(self, momentum: float):
        for source, target in ((self.student, self.teacher), (self.dino_student, self.dino_teacher),
                               (self.ibot_student, self.ibot_teacher)):
            for p, q in zip(source.parameters(), target.parameters()):
                q.lerp_(p, 1 - momentum)

    def forward(self, global_images, local_images, global_boxes, local_boxes, mask_ratio=CFG.mask_ratio):
        batch, n_global = global_images.shape[:2]
        n_local = local_images.shape[1]
        teacher_global = []
        student_global = []
        student_local = []
        masks = []
        for view in range(n_global):
            image = global_images[:, view]
            with torch.no_grad():
                teacher_global.append(self.teacher.forward_features(image, CFG.global_patch))
            patches = teacher_global[-1]["patches"].shape[1]
            mask = torch.rand(batch, patches, device=image.device) < mask_ratio
            masks.append(mask)
            student_global.append(self.student.forward_features(image, CFG.global_patch, mask=mask))
        for view in range(n_local):
            student_local.append(self.student.forward_features(local_images[:, view], self.local_patch))
        teacher_target = [sinkhorn(self.dino_teacher(value["cls"].detach()), CFG.teacher_temperature)
                          for value in teacher_global]
        cls_logits = [self.dino_student(value["cls"]) for value in student_global + student_local]
        dino_terms = [dino_ce(cls_logits[index], teacher_target[teacher_index], CFG.student_temperature)
                      for teacher_index in range(n_global) for index in range(n_global + n_local)
                      if index != teacher_index]
        dino = torch.stack(dino_terms).mean()
        ibot_terms = []
        for teacher_value, student_value, mask in zip(teacher_global, student_global, masks):
            teacher_logits = self.ibot_teacher(teacher_value["patches"].detach())
            target = sinkhorn(teacher_logits.flatten(0, 1), CFG.teacher_temperature).reshape_as(teacher_logits)
            predicted = self.ibot_student(student_value["patches"])
            if mask.any():
                ibot_terms.append(dino_ce(predicted[mask], target[mask], CFG.student_temperature))
        ibot = torch.stack(ibot_terms).mean() if ibot_terms else dino * 0
        koleo = koleo_loss(torch.cat([v["cls"] for v in student_global], dim=0))
        correspondence = []
        for local_index, local_value in enumerate(student_local):
            for global_index, global_value in enumerate(student_global):
                correspondence.append(correspondence_loss(
                    local_value["patches"], global_value["patches"], local_boxes[:, local_index],
                    global_boxes[:, global_index], local_value["grid"], global_value["grid"]))
        corr = torch.stack(correspondence).mean()
        loss = dino + ibot + CFG.koleo_weight * koleo + self.correspondence_weight * corr
        return loss, {"dino": float(dino.detach()), "ibot": float(ibot.detach()),
                      "koleo": float(koleo.detach()), "correspondence": float(corr.detach())}


def parameter_groups(model: SelfSupervised, lr: float):
    groups = []
    depth = len(model.student.blocks)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        multiplier = 1.0
        if name.startswith("student.patch_embed"):
            multiplier = CFG.patch_lr_multiplier * CFG.layer_decay ** depth
        elif name.startswith("student.blocks."):
            block = int(name.split(".")[2])
            multiplier = CFG.layer_decay ** (depth - 1 - block)
        elif name.startswith(("dino_student", "ibot_student")):
            multiplier = CFG.head_lr_multiplier
        groups.append({"params": [parameter], "lr": lr * multiplier})
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100, help="Number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of volumes per pretraining batch")
    parser.add_argument("--lr", type=float, default=CFG.pretrain_lr)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--ablation", choices=["baseline", "multiscale", "full"], default="full")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Full-scale 3D pretraining requires a CUDA GPU")
    torch.manual_seed(CFG.seed)
    rows = [r for r in rows_from_manifest(args.manifest) if r["split"] == "pretrain"]
    if not rows:
        raise ValueError("No pretrain patients")
    loader = DataLoader(PretrainDataset(rows, CFG.global_size, CFG.local_size, CFG.n_local),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    local_patch = CFG.global_patch if args.ablation == "baseline" else CFG.local_patch
    correlation_weight = CFG.correspondence_weight if args.ablation == "full" else 0.0
    model = SelfSupervised(CerviSpanEncoder(), local_patch=local_patch,
                           correspondence_weight=correlation_weight).cuda()
    optimizer = torch.optim.AdamW(parameter_groups(model, args.lr), weight_decay=CFG.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "configuration.json").write_text(json.dumps({**CFG.dictionary(), "ablation": args.ablation,
        "effective_local_patch": local_patch, "effective_correspondence_weight": correlation_weight}, indent=2), encoding="utf-8")
    for epoch in range(args.epochs):
        model.train()
        model.teacher.eval()
        totals = []
        for global_views, local_views, global_boxes, local_boxes in loader:
            tensors = [x.cuda(non_blocking=True) for x in (global_views, local_views, global_boxes, local_boxes)]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model(*tensors)
            loss.backward()
            optimizer.step()
            model.ema(CFG.teacher_momentum)
            totals.append(float(loss.detach()))
        print(f"epoch={epoch+1} loss={np.mean(totals):.5f} components={parts}", flush=True)
        save_encoder(model.teacher, args.output / "teacher.pt", {"epoch": epoch + 1, "paper_only": True})


if __name__ == "__main__":
    main()