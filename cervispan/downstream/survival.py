"""Paper-specified Cox MLP, pairwise ranking, Breslow DFS and test metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

from ..evaluation.metrics import bootstrap_ci


class CoxHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def cox_breslow_loss(risk, time, event):
    values = []
    for instant in torch.unique(time[event.bool()]):
        deaths = (time == instant) & event.bool()
        risk_set = time >= instant
        values.append(risk[deaths].sum() - deaths.sum() * torch.logsumexp(risk[risk_set], 0))
    if not values:
        raise ValueError("Training set contains no DFS events")
    return -torch.stack(values).sum() / event.sum()


def pairwise_ranking_loss(risk, time, event):
    pairs = (time[:, None] < time[None, :]) & event.bool()[:, None]
    return F.softplus(-(risk[:, None] - risk[None, :])[pairs]).mean() if pairs.any() else risk.sum() * 0


def fit(x, time, event, epochs, seed):
    torch.manual_seed(seed)
    head = CoxHead(x.shape[1])
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    x, time, event = (torch.as_tensor(array, dtype=torch.float32) for array in (x, time, event))
    head.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        risk = head(x)
        loss = cox_breslow_loss(risk, time, event) + 0.3 * pairwise_ranking_loss(risk, time, event)
        loss.backward()
        optimizer.step()
    return head.eval()


@torch.inference_mode()
def predict(head, x):
    return head(torch.as_tensor(x, dtype=torch.float32)).numpy()


def breslow_hazard(time, event, risk):
    instants = np.unique(time[event == 1])
    increments = [np.sum((time == t) & (event == 1)) / np.exp(risk[time >= t]).sum() for t in instants]
    return instants, np.cumsum(increments)


def score(time, event, risk):
    pairs = (time[:, None] < time[None, :]) & (event[:, None] == 1)
    if not pairs.any():
        return float("nan")
    difference = risk[:, None] - risk[None, :]
    return float(((difference > 0) * pairs + 0.5 * (difference == 0) * pairs).sum() / pairs.sum())


def censor_survival(train_time, train_event):
    times = np.unique(train_time)
    survival = []
    current = 1.0
    for instant in times:
        at_risk = np.sum(train_time >= instant)
        censored = np.sum((train_time == instant) & (train_event == 0))
        current *= 1 - censored / at_risk
        survival.append(current)
    return times, np.asarray(survival)


def dynamic_auc(train_time, train_event, test_time, test_event, risk, horizon):
    """Cumulative/dynamic AUC with training-cohort censoring Kaplan-Meier IPCW."""
    case = (test_time <= horizon) & (test_event == 1)
    control = test_time > horizon
    if not case.any() or not control.any():
        return float("nan")
    censor_time, censor_probability = censor_survival(train_time, train_event)
    # Left limit of G at each event time.
    position = np.searchsorted(censor_time, test_time[case], side="left") - 1
    weights = np.where(position >= 0, censor_probability[np.maximum(position, 0)], 1.0)
    if np.any(weights <= 0):
        return float("nan")
    weights = 1 / weights
    comparison = risk[case, None] - risk[None, control]
    correct = (comparison > 0) + 0.5 * (comparison == 0)
    return float((correct * weights[:, None]).sum() / (weights.sum() * control.sum()))


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
        labels = {row["case_id"]: (float(row["time_months"]), int(row["event"])) for row in csv.DictReader(stream)}
    train = np.array([i for i, (case_id, part) in enumerate(zip(ids, split)) if part == "train" and case_id in labels])
    if len(train) < 10:
        raise ValueError("At least 10 training patients required")
    time = np.array([labels[ids[i]][0] for i in train])
    event = np.array([labels[ids[i]][1] for i in train])
    folds = list(KFold(5, shuffle=True, random_state=a.seed).split(train))
    scores = {}
    for epochs in a.epochs:
        values = []
        for fold, (tr, val) in enumerate(folds):
            if not event[tr].any() or not event[val].any():
                raise ValueError("Each training and validation fold must contain DFS events")
            head = fit(x[train[tr]], time[tr], event[tr], epochs, a.seed + fold)
            values.append(score(time[val], event[val], predict(head, x[train[val]])))
        scores[epochs] = values
    selected = max(scores, key=lambda epoch: np.mean(scores[epoch]))
    head = fit(x[train], time, event, selected, a.seed)
    risk = predict(head, x)
    cutoff = float(np.median(risk[train]))
    baseline_times, cumulative_hazard = breslow_hazard(time, event, risk[train])
    a.output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "feature_dim": x.shape[1]}, a.output / "cox_head.pt")
    np.savez_compressed(a.output / "breslow.npz", time=baseline_times, hazard=cumulative_hazard)
    (a.output / "selection.json").write_text(json.dumps({"cv_c_index": scores, "epochs": selected, "training_median": cutoff}, indent=2), encoding="utf-8")
    with (a.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "split", "time_months", "event", "log_risk", "risk_group", "dfs_12", "dfs_36", "dfs_60"])
        for i, (case_id, part) in enumerate(zip(ids, split)):
            if part == "train":
                continue
            prediction = []
            for month in (12, 36, 60):
                position = np.searchsorted(baseline_times, month, side="right") - 1
                hazard = cumulative_hazard[position] if position >= 0 else 0.0
                prediction.append(float(np.exp(-hazard * np.exp(risk[i]))))
            writer.writerow([case_id, part, *labels.get(case_id, ("", "")), float(risk[i]),
                             "high" if risk[i] >= cutoff else "low", *prediction])
    metrics = {}
    for part in sorted(set(split) - {"train"}):
        index = np.array([i for i, s in enumerate(split) if s == part and ids[i] in labels])
        if len(index) < 2:
            continue
        observed_time = np.array([labels[ids[i]][0] for i in index])
        observed_event = np.array([labels[ids[i]][1] for i in index])
        if not observed_event.any():
            continue
        entry = {"n": len(index), "c_index": score(observed_time, observed_event, risk[index]),
                 "c_index_ci_95": bootstrap_ci(len(index), lambda take: score(observed_time[take], observed_event[take], risk[index][take]), a.bootstrap, a.seed)}
        for month in (12, 36, 60):
            # IPCW requires the horizon inside train and test follow-up support.
            if month >= min(time.max(), observed_time.max()):
                continue
            auc = dynamic_auc(time, event, observed_time, observed_event, risk[index], float(month))
            if np.isfinite(auc):
                entry[f"auc_{month}"] = auc
                entry[f"auc_{month}_ci_95"] = bootstrap_ci(
                    len(index),
                    lambda take: dynamic_auc(time, event, observed_time[take], observed_event[take], risk[index][take], float(month)),
                    a.bootstrap, a.seed + month,
                )
        metrics[part] = entry
    (a.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()