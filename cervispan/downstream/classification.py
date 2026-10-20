"""Frozen-feature binary linear probing with train-only five-fold selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from ..evaluation.metrics import bootstrap_ci


def asymmetric_focal(logits, labels, gamma_negative=4.0, gamma_positive=1.0):
    # Focal-loss weighting for negative and positive samples.
    positive = torch.sigmoid(logits)
    negative = 1 - positive
    return -(labels * negative.pow(gamma_positive) * positive.clamp_min(1e-8).log()
             + (1 - labels) * positive.pow(gamma_negative) * negative.clamp_min(1e-8).log()).mean()


def train_head(x, y, epochs, seed):
    torch.manual_seed(seed)
    head = nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    inputs = torch.as_tensor(x, dtype=torch.float32)
    targets = torch.as_tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = asymmetric_focal(head(inputs).squeeze(-1), targets)
        loss.backward()
        optimizer.step()
    return head.eval()


@torch.inference_mode()
def probabilities(head, x):
    return torch.sigmoid(head(torch.as_tensor(x, dtype=torch.float32)).squeeze(-1)).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, nargs="+", default=[25, 50, 100, 200], help="Candidate training epochs for cross-validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=20000)
    a = p.parse_args()
    with np.load(a.features) as file:
        x, ids, split = file["X"], file["case_id"].astype(str), file["split"].astype(str)
    with a.labels.open(newline="", encoding="utf-8-sig") as stream:
        labels = {r["case_id"]: int(r["label"]) for r in csv.DictReader(stream)}
    train = np.array([i for i, (case_id, part) in enumerate(zip(ids, split)) if part == "train" and case_id in labels])
    y = np.array([labels[ids[i]] for i in train], dtype=np.int64)
    if len(train) < 10 or len(np.unique(y)) != 2 or np.bincount(y).min() < 5:
        raise ValueError("Five folds require at least five patients from each class in train")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(train, y))
    scores = {}
    for epochs in a.epochs:
        fold_auc = []
        for fold, (tr, val) in enumerate(folds):
            head = train_head(x[train[tr]], y[tr], epochs, a.seed + fold)
            fold_auc.append(float(roc_auc_score(y[val], probabilities(head, x[train[val]]))))
        scores[epochs] = fold_auc
    selected = max(scores, key=lambda epoch: np.mean(scores[epoch]))
    # Threshold from out-of-fold training predictions, never from held-out cohorts.
    out_of_fold = np.zeros(len(train))
    for fold, (tr, val) in enumerate(folds):
        head = train_head(x[train[tr]], y[tr], selected, a.seed + fold)
        out_of_fold[val] = probabilities(head, x[train[val]])
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold = float(max(thresholds, key=lambda value: balanced_accuracy_score(y, out_of_fold >= value)))
    final = train_head(x[train], y, selected, a.seed)
    a.output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final.state_dict(), "feature_dim": x.shape[1], "threshold": threshold}, a.output / "head.pt")
    (a.output / "selection.json").write_text(json.dumps({"cv_auc": scores, "epochs": selected, "threshold": threshold}, indent=2), encoding="utf-8")
    with (a.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "split", "label", "probability", "prediction"])
        predicted = probabilities(final, x)
        for case_id, part, score in zip(ids, split, predicted):
            if part != "train":
                writer.writerow([case_id, part, labels.get(case_id, ""), float(score), int(score >= threshold)])
    metrics = {}
    for part in sorted(set(split) - {"train"}):
        index = np.array([i for i, s in enumerate(split) if s == part and ids[i] in labels])
        if len(index) and len({labels[ids[i]] for i in index}) == 2:
            target = np.array([labels[ids[i]] for i in index])
            probability = predicted[index]
            binary = probability >= threshold
            tn, fp, fn, tp = confusion_matrix(target, binary, labels=[0, 1]).ravel()
            metrics[part] = {"n": len(index), "auc": roc_auc_score(target, probability),
                             "auc_ci_95": bootstrap_ci(len(index), lambda take: roc_auc_score(target[take], probability[take]), a.bootstrap, a.seed),
                             "accuracy": accuracy_score(target, binary),
                             "balanced_accuracy": balanced_accuracy_score(target, binary),
                             "f1": f1_score(target, binary),
                             "precision": precision_score(target, binary, zero_division=0),
                             "recall": recall_score(target, binary, zero_division=0),
                             "specificity": float(tn / max(tn + fp, 1))}
    (a.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()