# nuScenes VLM Phase 1 — Baseline Inference & Evaluation Pipeline

> **Phase 1 only.**  No fine-tuning, quantisation, causal analysis, or edge deployment.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Analysis](#2-dataset-analysis)
3. [Phase-1 Subset Strategy](#3-phase-1-subset-strategy)
4. [VLM Selection](#4-vlm-selection)
5. [Project Structure](#5-project-structure)
6. [Environment Setup](#6-environment-setup)
7. [Configuration](#7-configuration)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Running the Tests](#9-running-the-tests)
10. [Output Files](#10-output-files)
11. [Evaluation Assumptions & Limitations](#11-evaluation-assumptions--limitations)
12. [Scope Restrictions (Phase 1)](#12-scope-restrictions-phase-1)

---

## 1. Project Overview

This repository implements a reproducible baseline pipeline for running
pretrained Vision-Language Model (VLM) inference on the
[nuScenes](https://www.nuscenes.org/) autonomous-driving dataset.

**Goals**

- Understand the nuScenes dataset structure in sufficient depth to select a
  scientifically justified Phase-1 subset.
- Run zero-shot VLM inference on front-camera images.
- Evaluate scene-description quality against ground-truth 3-D annotations.
- Produce structured outputs (manifest CSV, JSONL predictions, JSON metrics,
  text report) for downstream research.

---

## 2. Dataset Analysis

### 2.1 nuScenes Data Model

```
scene (∼ 20 s clip)
  └── sample (annotated keyframe, ∼ 2 Hz)
        ├── sample_data (per-sensor raw recording at each timestamp)
        │     CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
        │     CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT,
        │     LIDAR_TOP, RADAR_FRONT, …
        └── sample_annotation (3-D bounding box per object)
              ├── category (vehicle.car, human.pedestrian.adult, …)
              ├── attribute (e.g. vehicle.moving)
              ├── visibility_token (1 = 0–40 % visible … 4 = 80–100 %)
              ├── num_lidar_pts / num_radar_pts
              └── translation, rotation, size (in global frame)
```

Additional metadata tables:
- **calibrated_sensor** — extrinsics + intrinsic matrix per sensor instance.
- **ego_pose** — global pose of the ego vehicle at each sensor timestamp.
- **map** — 2-D semantic map for the recording location.
- **log** — recording date, vehicle, location.

### 2.2 Dataset Versions

| Version | Scenes | Samples | Purpose |
|---|---|---|---|
| `v1.0-mini` | 10 | ∼ 404 | Rapid pipeline validation |
| `v1.0-trainval` | 850 | ∼ 34 149 | Full research experiments |

### 2.3 Phase-1 Modality Decisions

| Modality | Phase-1 Decision | Justification |
|---|---|---|
| `CAM_FRONT` images | ✅ **Used** | Primary VLM input; single forward-facing view |
| Other 5 cameras | ⏳ Deferred | Increases coverage but not needed for baseline |
| LiDAR (`LIDAR_TOP`) | ❌ Excluded | Not consumed by image-only VLMs; large file size |
| Radar | ❌ Excluded | As above; radar data not directly interpretable by VLMs |
| 3-D annotations | ✅ **Used (eval only)** | Ground truth for recall and hallucination evaluation |
| Calibration matrices | ✅ **Used (eval only)** | Required for 3-D → 2-D projection in evaluation |
| Ego poses | ✅ **Used (eval only)** | Required for global → camera coordinate transform |
| Map / log | ❌ Excluded | Not needed for image-level scene description |

---

## 3. Phase-1 Subset Strategy

### 3.1 Pipeline Validation (mini)

- Dataset: `v1.0-mini`
- Samples: up to **10** (configurable via `selection.mini.max_samples`)
- Split: mini (uses both `mini_train` and `mini_val` names from the devkit)
- Purpose: verify image loading, annotation retrieval, VLM inference end-to-end

### 3.2 Research Pilot (trainval)

- Dataset: `v1.0-trainval`
- Scenes: up to **50 train + 10 val** (configurable)
- Sample selection: all annotated keyframes per selected scene, or one midpoint
  sample per scene (toggle `one_sample_per_scene`)
- Camera: `CAM_FRONT` keyframe image per sample
- Ground truth: all `sample_annotation` records for each sample

**Split integrity** is enforced: the official `nuscenes.utils.splits.{train,val}`
scene-name lists are used directly.  A scene is never split across train and val.

### 3.3 Diversity Assessment

Run `src/dataset_audit.py` to inspect:
- Object-category distribution across selected scenes
- Visibility-level distribution (levels 1–4)
- Scene descriptions
- Image availability statistics

---

## 4. VLM Selection

| Key | Model | Parameters | Use |
|---|---|---|---|
| `qwen2_5_vl_3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | 3 B | **Primary baseline** |
| `smolvlm2_2b` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | 2.2 B | Optional alternative |

Both models use the HuggingFace `transformers` API.
Qwen2.5-VL uses `Qwen2_5_VLForConditionalGeneration` + `qwen-vl-utils`.
SmolVLM2 uses `AutoModelForVision2Seq`.

Only **one model is loaded at a time**.

---

## 5. Project Structure

```
nuscenes_vlm_phase1/
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
│
├── configs/
│   ├── config.yaml              ← Master configuration (paths, model, counts)
│   └── prompt.txt               ← Fixed VLM inference prompt
│
├── src/
│   ├── utils.py                 ← Logging, config/JSONL I/O, timing, seeds
│   ├── dataset_loader.py        ← NuScenesLoader: devkit wrapper
│   ├── scene_selector.py        ← Reproducible scene/sample selection
│   ├── model_loader.py          ← VLM lifecycle: load, generate, unload
│   ├── infer.py                 ← Main inference pipeline (CLI entry)
│   ├── dataset_audit.py         ← Manifest generation + audit report (CLI)
│   └── evaluation.py            ← Evaluation metrics computation (CLI)
│
├── tests/
│   ├── conftest.py              ← pytest sys.path setup
│   ├── test_dataset.py          ← Unit tests: loader, selector, evaluation
│   └── test_output_schema.py    ← Schema/contract tests: JSONL, CSV, metrics
│
├── outputs/
│   ├── manifests/               ← dataset_manifest_<version>.csv
│   ├── predictions/             ← predictions_<version>_<model>.jsonl
│   ├── metrics/                 ← metrics_<predictions_stem>.json
│   └── reports/                 ← Baseline text reports
│
└── notebooks/
    └── baseline_analysis.ipynb  ← Exploratory analysis (optional)
```

---

## 6. Environment Setup

### 6.1 Prerequisites

- Python ≥ 3.10 (3.11 or 3.12 recommended)
- CUDA 11.8+ (optional but strongly recommended for GPU inference)
- nuScenes dataset files downloaded from [nuscenes.org](https://www.nuscenes.org/download)

### 6.2 Create a conda environment

```bash
conda create -n nuscenes_vlm python=3.11 -y
conda activate nuscenes_vlm
```

### 6.3 Install PyTorch

```bash
# CUDA 12.1 (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU-only fallback
pip install torch torchvision
```

### 6.4 Install project dependencies

```bash
cd /csehome/m25csa031/nuscenes_vlm_phase1
pip install -r requirements.txt
```

### 6.5 (Optional) Flash Attention for faster GPU inference

Only supported on Ampere+ GPUs (A100, RTX 3090+):

```bash
pip install flash-attn --no-build-isolation
```

Then set `attn_implementation: flash_attention_2` in the model config
(automatic if the package is importable).

### 6.6 Set the dataset path

Edit `configs/config.yaml`:

```yaml
dataset:
  dataroot: "/path/to/your/nuscenes"   # ← change this
  version: "v1.0-mini"
```

Expected directory layout:
```
/path/to/your/nuscenes/
  v1.0-mini/          ← JSON metadata files
  v1.0-trainval/      ← JSON metadata files
  samples/            ← image files
    CAM_FRONT/
    CAM_FRONT_LEFT/
    …
```

---

## 7. Configuration

All runtime settings are in [`configs/config.yaml`](configs/config.yaml).
Key sections:

| Section | Key fields |
|---|---|
| `dataset` | `dataroot`, `version`, `camera_channel` |
| `selection.mini` | `max_samples`, `seed` |
| `selection.trainval` | `max_train_scenes`, `max_val_scenes`, `seed`, `one_sample_per_scene` |
| `model` | `name`, `dtype`, `max_new_tokens`, `do_sample`, `device`, `cache_dir` |
| `inference` | `prompt_file`, `batch_size`, `checkpoint_interval` |
| `evaluation` | `focus_categories`, `min_visibility_level`, `hallucination_review_n` |
| `outputs` | `root`, `manifests`, `predictions`, `metrics`, `reports` |

The prompt is stored separately in [`configs/prompt.txt`](configs/prompt.txt)
and instructs the model to describe road layout, road users, signs,
infrastructure, hazards, and uncertainty.

---

## 8. Running the Pipeline

All commands assume you are in the project root and the conda environment is active.

### Step 1 — Dataset Audit

```bash
# Mini split
python src/dataset_audit.py --config configs/config.yaml --version v1.0-mini

# Trainval split
python src/dataset_audit.py --config configs/config.yaml --version v1.0-trainval
```

Produces:
- `outputs/manifests/manifest_v1.0-mini.csv`
- Logged audit report (scene counts, category distribution, missing images)

---

### Step 2 — Inference

#### Mini experiment (pipeline validation, ∼ 10 samples)

```bash
python src/infer.py \
    --config configs/config.yaml \
    --version v1.0-mini \
    --model qwen2_5_vl_3b
```

#### Research pilot — training split

```bash
python src/infer.py \
    --config configs/config.yaml \
    --version v1.0-trainval \
    --model qwen2_5_vl_3b \
    --split train
```

#### Research pilot — validation split

```bash
python src/infer.py \
    --config configs/config.yaml \
    --version v1.0-trainval \
    --model qwen2_5_vl_3b \
    --split val
```

#### Both splits in one run

```bash
python src/infer.py \
    --config configs/config.yaml \
    --version v1.0-trainval \
    --split both
```

#### Use the alternative model

```bash
python src/infer.py \
    --config configs/config.yaml \
    --version v1.0-mini \
    --model smolvlm2_2b
```

**Resume support**: re-running any command skips already-completed sample tokens.

Produces:
- `outputs/predictions/predictions_v1.0-mini_qwen2_5_vl_3b.jsonl`
- `outputs/predictions/predictions_v1.0-trainval_train_qwen2_5_vl_3b.jsonl`
- `outputs/predictions/predictions_v1.0-trainval_val_qwen2_5_vl_3b.jsonl`

---

### Step 3 — Evaluation

```bash
# Mini
python src/evaluation.py \
    --config configs/config.yaml \
    --predictions outputs/predictions/predictions_v1.0-mini_qwen2_5_vl_3b.jsonl \
    --version v1.0-mini

# Trainval val split
python src/evaluation.py \
    --config configs/config.yaml \
    --predictions outputs/predictions/predictions_v1.0-trainval_val_qwen2_5_vl_3b.jsonl \
    --version v1.0-trainval
```

Produces:
- `outputs/metrics/metrics_predictions_v1.0-mini_qwen2_5_vl_3b.json`

---

## 9. Running the Tests

Tests do **not** require dataset downloads or model weights.

```bash
# From project root
cd /csehome/m25csa031/nuscenes_vlm_phase1

# All tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Individual test files
python -m pytest tests/test_dataset.py -v
python -m pytest tests/test_output_schema.py -v
```

### What the tests cover

| Test module | Coverage |
|---|---|
| `test_dataset.py` | `utils.py` helpers, `NuScenesLoader` (mocked), `scene_selector` split integrity, `evaluation.py` metric functions |
| `test_output_schema.py` | JSONL prediction schema, CSV manifest schema, resume mechanism, metrics JSON schema |

### What is NOT tested (requires real data / GPU)

- Loading the actual nuScenes dataset files
- Downloading and running VLM weights
- Full end-to-end inference on real images
- GPU memory measurement

---

## 10. Output Files

### `outputs/manifests/manifest_<version>.csv`

One row per selected sample.  Columns: `dataset_version`, `split`,
`scene_token`, `scene_name`, `scene_description`, `sample_token`,
`timestamp_us`, `image_path`, `image_available`, `image_width`,
`image_height`, `n_annotations`, `vis_1`–`vis_4`, `categories_json`.

### `outputs/predictions/<name>.jsonl`

One JSON record per processed sample:

```json
{
  "dataset_version": "v1.0-mini",
  "split": "mini",
  "scene_token": "...",
  "sample_token": "...",
  "timestamp_us": 1531883530449377,
  "image_path": "/data/nuscenes/samples/CAM_FRONT/xxx.jpg",
  "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
  "model_key": "qwen2_5_vl_3b",
  "prompt": "...",
  "generated_description": "1. ROAD LAYOUT\nThe road is ...",
  "latency_sec": 2.34,
  "n_tokens_generated": 298,
  "tokens_per_sec": 127.4,
  "device": "cuda:0",
  "dtype": "bfloat16",
  "error": null,
  "recorded_at": "2024-11-01T12:34:56Z"
}
```

### `outputs/metrics/<name>.json`

Aggregated evaluation metrics including recall, visibility-stratified recall,
hallucination rate, latency statistics, and caveats.

---

## 11. Evaluation Assumptions & Limitations

1. **3-D annotations ≠ image-visible objects.**  A 3-D bounding box may be
   behind the vehicle, heavily occluded, or outside the camera FOV.
   `visibility_token` (1–4) is used as a proxy but does not guarantee
   image visibility.

2. **Keyword-based recall.**  Category recall is estimated using a hand-crafted
   keyword list.  A model might describe a "sedan" which maps to `vehicle.car`,
   but miss a "van" that also maps to the same category.  The estimate is a
   lower bound.

3. **Hallucination detection is heuristic.**  An "unsupported mention" is any
   category keyword in the VLM output for which no annotation exists in the
   sample.  False positives occur when the model correctly perceives an object
   that falls outside nuScenes' annotation protocol (e.g. road surface, sky).
   Manual review of flagged samples is required.

4. **Fluent output ≠ factual correctness.**  We explicitly do not treat
   grammatically correct or well-structured descriptions as evidence of
   accuracy.

5. **No semantic similarity.**  ROUGE or embedding-based similarity against
   the nuScenes `scene.description` field is *not* implemented in Phase 1.
   `scene.description` contains only brief human-curated annotations, not
   full scene descriptions, making them unreliable evaluation targets.

---

## 12. Scope Restrictions (Phase 1)

The following are **explicitly out of scope** for this phase:

- Causal graph extraction
- Causal token importance scoring
- Token pruning or early-exit
- Knowledge distillation
- Quantisation-aware training
- Edge deployment or ONNX export
- Continual adaptation or online learning
- Model fine-tuning of any kind
- Multi-camera fusion (beyond CAM_FRONT)
- LiDAR or radar processing

---

*Generated by Antigravity (Phase 1 pipeline) — October 2026.*
