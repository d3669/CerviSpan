"""Phenotype concordance of image-only nearest neighbours."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..downstream.retrieval import nearest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--attributes", type=Path, required=True, help="CSV: case_id,lnm,stage,tumor_size_mm")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--size-tolerance-mm", type=float, default=10.0,
                   help="Absolute tumour-size difference for a match, in mm")
    a = p.parse_args()
    with np.load(a.features) as file:
        ids = file["case_id"].astype(str).tolist()
        x = file["X"]
    with a.attributes.open(newline="", encoding="utf-8-sig") as stream:
        attributes = {row["case_id"]: row for row in csv.DictReader(stream)}
    result = {}
    for k in a.k:
        pairs = nearest(x, ids, k)
        values = {"lnm": [], "stage": [], "tumor_size": [], "joint": []}
        for item in pairs:
            query, neighbour = attributes.get(item["query_id"]), attributes.get(item["neighbour_id"])
            if query is None or neighbour is None:
                continue
            matches = {
                "lnm": query["lnm"] == neighbour["lnm"],
                "stage": query["stage"] == neighbour["stage"],
                "tumor_size": abs(float(query["tumor_size_mm"]) - float(neighbour["tumor_size_mm"])) <= a.size_tolerance_mm,
            }
            for key, value in matches.items():
                values[key].append(value)
            values["joint"].append(all(matches.values()))
        result[f"k_{k}"] = {key: (float(np.mean(value)) if value else None) for key, value in values.items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()