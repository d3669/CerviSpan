"""Paper preprocessing from DICOM/NIfTI with patient-level manifest validation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ..configs.config import CFG


def load_image(path: Path) -> sitk.Image:
    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        identifiers = reader.GetGDCMSeriesIDs(str(path))
        if len(identifiers) != 1:
            raise ValueError(f"Expected exactly one DICOM series in {path}, found {len(identifiers)}")
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(path), identifiers[0]))
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(path))
    return sitk.DICOMOrient(image, "LPS")


def preprocess(ct_path: Path, mask_path: Path | None = None) -> dict[str, np.ndarray]:
    image = load_image(ct_path)
    spacing = [CFG.spacing_mm] * 3
    size = [max(1, round(a * b / CFG.spacing_mm)) for a, b in zip(image.GetSize(), image.GetSpacing())]
    ct = sitk.Resample(image, size, sitk.Transform(), sitk.sitkLinear,
                      image.GetOrigin(), spacing, image.GetDirection(),
                      CFG.hu_min, sitk.sitkFloat32)
    hu = np.clip(sitk.GetArrayFromImage(ct).astype(np.float32), CFG.hu_min, CFG.hu_max)
    normalized = (hu - hu.mean()) / max(float(hu.std()), 1e-6)
    result = {"ct": normalized, "spacing_zyx": np.array(spacing[::-1], dtype=np.float32)}
    if mask_path:
        mask = sitk.Resample(load_image(mask_path), ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        result["mask"] = (sitk.GetArrayFromImage(mask) > 0).astype(np.uint8)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--out-manifest", type=Path, required=True)
    a = p.parse_args()
    with a.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"case_id", "split", "ct_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest requires {sorted(required)}")
    identifiers = [r["case_id"] for r in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Patient IDs must be unique; aggregate scans for a patient before splitting")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for r in rows:
        ct = Path(r["ct_path"])
        if not ct.is_absolute():
            ct = a.manifest.parent / ct
        mask = Path(r["mask_path"]) if r.get("mask_path") else None
        if mask is not None and not mask.is_absolute():
            mask = a.manifest.parent / mask
        target = a.output_dir / f"{r['case_id']}.npz"
        np.savez_compressed(target, **preprocess(ct, mask))
        result.append({"case_id": r["case_id"], "split": r["split"], "npz_path": str(target.resolve())})
    a.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with a.out_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "split", "npz_path"])
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    main()