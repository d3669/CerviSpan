# CerviSpan

**A CT foundation model spanning the cervical cancer care pathway**

CerviSpan is a disease-centred, self-supervised 3D CT representation model designed to support multiple tasks across the cervical cancer care pathway.

## Method overview

The encoder is pretrained on unlabelled CT volumes using a DINOv2-style teacher–student framework, with an iBOT masked-token objective and KoLeo regularization. CerviSpan incorporates two components tailored to volumetric anatomy:

1. **Asymmetric multi-scale views:** Global crops of 160³ voxels use 16³-voxel patches, while local crops of 48³ voxels use 8³-voxel patches.
2. **Anatomical correspondence learning:** Features at spatially corresponding positions in local and global views are aligned using a cosine correspondence loss.

For downstream tasks, the encoder remains frozen while task-specific prediction heads or a CTV decoder are trained. Zero-shot retrieval uses image embeddings without a retrieval-specific prediction head.

## Installation

Python 3.10 or later is required. Self-supervised pretraining and CTV decoder training require a CUDA-capable GPU. Install a PyTorch build compatible with your CUDA driver, then install the project dependencies:

```bash
git clone https://github.com/d3669/CerviSpan.git
cd CerviSpan
python -m pip install -r requirements.txt
```

### Preprocessing manifest

[The preprocessing module](https://github.com/d3669/CerviSpan/blob/main/cervispan/data/preprocessing.py) expects a CSV file with the following columns:

| Column | Required | Description |
|---|---|---|
| `case_id` | Yes | Pseudonymous patient identifier, unique within the manifest |
| `split` | Yes | Cohort label, such as `pretrain`, `train`, `validation`, or `test` |
| `ct_path` | Yes | Path to a CT volume file or a directory containing exactly one DICOM series |
| `mask_path` | No | Path to an optional rasterized mask image for segmentation |

Run preprocessing:

```bash
python -m cervispan.data.preprocessing \
  --manifest /path/to/ct_manifest.csv \
  --output-dir /path/to/processed_npz \
  --out-manifest /path/to/processed_manifest.csv
```

Each output NPZ file contains a normalized `ct` volume and, when provided, a `mask`. The generated manifest includes `case_id`, `split`, and `npz_path`. The code checks that patient IDs are unique within each manifest. Users are responsible for preventing patient overlap between independently prepared manifests.

### Task labels

- Classification CSV: `case_id,label` (prepare one binary endpoint at a time).
- Survival CSV: `case_id,time_months,event`, where `event=1` indicates a DFS event and `event=0` indicates censoring.
- Retrieval attributes CSV: `case_id,lnm,stage,tumor_size_mm`.
- Segmentation masks: stored in the processed NPZ file under `mask`.

Use pseudonymous identifiers. Do not commit patient-level labels, identifying file paths, DICOM headers, images, masks, or clinical records.

## Example workflow

The commands below illustrate the project workflow. Provide your own manifests, labels, and output directories.

```bash
# 1. Self-supervised pretraining using rows marked split=pretrain
python -m cervispan.pretraining.train \
  --manifest /path/to/processed_manifest.csv \
  --output /path/to/pretrain_run \
  --epochs 100 --batch-size 2

# 2. Extract features for rows outside the pretraining split
python -m cervispan.downstream.features \
  --manifest /path/to/processed_manifest.csv \
  --checkpoint /path/to/pretrain_run/teacher.pt \
  --output /path/to/features.npz

# 3. Binary linear probing
python -m cervispan.downstream.classification \
  --features /path/to/features.npz \
  --labels /path/to/labels.csv \
  --output /path/to/classification_results

# 4. Disease-free survival analysis
python -m cervispan.downstream.survival \
  --features /path/to/features.npz \
  --labels /path/to/dfs_labels.csv \
  --output /path/to/survival_results

# 5. CTV decoder training
python -m cervispan.downstream.segmentation train \
  --manifest /path/to/processed_manifest.csv \
  --checkpoint /path/to/pretrain_run/teacher.pt \
  --output /path/to/ctv_decoder.pt

# 6. Zero-shot nearest-neighbour retrieval and phenotype evaluation
python -m cervispan.downstream.retrieval \
  --features /path/to/features.npz \
  --output /path/to/neighbours.csv --k 5
python -m cervispan.evaluation.retrieval \
  --features /path/to/features.npz \
  --attributes /path/to/retrieval_attributes.csv \
  --output /path/to/retrieval_metrics.json
```

Use the `--help` option on each module to see its full list of arguments. Select hyperparameters and thresholds using training or validation data only, and reserve external test cohorts for final evaluation.

## Data access and privacy

Raw CT images are not publicly available. As stated in the manuscript, supporting data may be requested from the corresponding authors, subject to a signed data-access agreement. This repository does not include patient data, annotations, pretrained checkpoints, or clinical labels. Do not upload these materials here.
