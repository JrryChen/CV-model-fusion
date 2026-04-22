# Multi-Model Fusion for Robust 2D Upper-Limb Pose Estimation

Master's Degree Project -- University of Pittsburgh, 2026

## Overview

This project implements a learned fusion framework that combines predictions
from multiple complementary 2D pose estimators to produce robust upper-limb
keypoint estimates for exercise and rehabilitation video. The MLP Attention
Fusion model achieves **100% accuracy** (at threshold tau=0.05) on the test
set, eliminating all catastrophic outlier predictions that affect individual
models.

## Key Results

| Fusion Model | Accuracy | Mean L2 (px) |
|---|---|---|
| **MLP Fusion** | **100.00%** | 3.215 |
| Confidence-Weighted | 99.92% | 2.074 |
| Transformer Fusion | 99.69% | 4.492 |
| Internal Transformer | 99.58% | 3.441 |
| Uniform (1/3) | 98.39% | 3.052 |

Evaluated on 16,038 samples (5,346 frames x 3 joints). All results above are
from the 3-model fusion variants (HRNet/DEKR + OpenPose + MediaPipe). The
4-model fusion variants (which add ViTPose) have not been evaluated yet.

### Baseline Models

| Model | Accuracy | Mean L2 (px) | Mean L2 (norm) | Samples |
|---|---|---|---|---|
| HRNet/DEKR | 99.66% | 1.795 | 0.0050 | 16,038 |
| MediaPipe | 98.74% | 2.532 | 0.0070 | 15,972 |
| OpenPose | 98.18% | 2.612 | 0.0072 | 14,916 |
| DEKR (Trained) | 99.95% | 6.273 | 0.0173 | 16,038 |
| OpenPose (Frozen BN) | 91.71% | 9.778 | 0.0270 | 16,038 |
| OpenPose (Trained) | 88.50% | 11.378 | 0.0314 | 16,038 |
| DEKR (Frozen Backbone) | 70.72% | 29.733 | 0.0821 | 16,038 |

### MLP Fusion Improvement vs Baselines

| Baseline | Accuracy Δ | Per-Frame Error Δ |
|---|---|---|
| vs HRNet/DEKR | +0.34 pp | -79.1% (better) |
| vs OpenPose | +1.82 pp | -23.2% (better) |
| vs MediaPipe | +1.26 pp | -27.0% (better) |
| vs DEKR (Trained) | +89.38 pp | +95.3% (worse) |
| vs DEKR (Frozen) | +100.00 pp | +95.7% (worse) |
| vs OpenPose (Trained) | +11.50 pp | +71.7% (worse) |

## Repository Structure

```
pose-fusion/
├── models/
│   ├── mlp_fusion.py                  # MLP Attention Fusion (3-model)
│   ├── mlp_fusion_4.py                # MLP Attention Fusion (4-model, +ViTPose)
│   ├── transformer_fusion.py          # Transformer Encoder-Decoder Fusion
│   ├── transformer_fusion_4.py        # Transformer Fusion (4-model)
│   ├── transformer_internal_fusion.py # Internal-Attention Transformer Fusion
│   └── transformer_internal_fusion_4.py
├── data/
│   └── pytorch_dataset.py             # Dataset class for video keypoint data
├── training/
│   ├── train_mlp.py                   # MLP fusion training + shared data pipeline
│   ├── train_mlp_improved.py          # Improved MLP training with better metrics
│   ├── train_mlp_4.py                 # 4-model MLP fusion training
│   ├── train_transformer.py           # Transformer fusion training
│   ├── train_transformer_4.py         # 4-model transformer training
│   ├── train_internal.py              # Internal transformer fusion training
│   └── train_internal_4.py            # 4-model internal transformer training
├── evaluate_fusion.py                 # Unified evaluation pipeline
├── run_inference.py                   # Run inference and export predictions
├── plot_importance_comparison.py      # Importance weight comparison plots
├── requirements.txt
└── LICENSE
```

## Backbone Models

The fusion models combine predictions from these pose estimators:

| Model | Source | Role |
|---|---|---|
| **HRNet/DEKR** | [HRNet/DEKR](https://github.com/HRNet/DEKR) | Bottom-up keypoint regression (COCO-pretrained) |
| **OpenPose** | [CMU OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) | Part Affinity Fields (BODY_25) |
| **MediaPipe** | [Google MediaPipe](https://github.com/google/mediapipe) | Lightweight mobile pose estimation |
| **ViTPose** *(optional 4th)* | [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) via [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) | Vision Transformer pose estimation |

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Dataset Source

The video data was gathered from the
[UCO Physical Rehabilitation](https://github.com/AVAuco/ucophyrehab) dataset,
a multi-camera exercise/rehabilitation dataset recorded at the University of
Cordoba. It contains 2,160 video sequences of 27 subjects performing 16
upper- and lower-body rehabilitation exercises captured from 5 RGB camera
viewpoints, with 3D ground-truth keypoints provided by an OptiTrack motion
capture system.

### Data Format

The training pipeline expects a dataset organized as follows:

```
dataset_root/
├── clips_mp4/
│   ├── <patient_id>/
│   │   ├── <exercise_id>/
│   │   │   ├── cam0.mp4              # Video file
│   │   │   ├── cam0_p2d.txt          # Ground truth annotations
│   │   │   └── ...
│   │   └── ...
│   └── p2d_cache/                     # Precomputed model predictions
│       ├── <patient_id>/
│       │   └── <exercise_id>/
│       │       ├── cam0_dekr_p2d.txt
│       │       ├── cam0_openpose_p2d.txt
│       │       ├── cam0_mediapipe_p2d.txt
│       │       └── cam0_vitpose_p2d.txt  # Optional (4-model only)
│       └── ...
├── train.txt                          # List of training videos
├── val.txt                            # List of validation videos
└── test.txt                           # List of test videos
```

Each `*_p2d.txt` file contains one line per annotated frame with space-separated
values: `x1 y1 conf1 x2 y2 conf2 x3 y3 conf3` for each joint (left shoulder,
left elbow, left wrist). List files contain relative paths like
`clips_mp4/1/09/cam0.mp4`, one per line.

### Training (MLP Fusion)

```bash
python -m training.train_mlp_improved \
    --list path/to/train.txt \
    --root path/to/dataset_root \
    --val-list path/to/val.txt \
    --save-dir checkpoints/mlp_fusion \
    --device cuda \
    --epochs 20 \
    --batch-size 8
```

### Training (Transformer Fusion)

```bash
python -m training.train_transformer \
    --list path/to/train.txt \
    --root path/to/dataset_root \
    --val-list path/to/val.txt \
    --save-dir checkpoints/transformer_fusion \
    --device cuda
```

### Evaluation (all models)

```bash
python evaluate_fusion.py \
    --eval-all \
    --list path/to/test.txt \
    --root path/to/dataset_root \
    --output-dir eval_results \
    --device cuda \
    --extract-importance \
    --visualize-importance
```

### Inference

```bash
python run_inference.py \
    --checkpoints checkpoints/mlp_fusion/checkpoint_best.pt \
    --list path/to/test.txt \
    --root path/to/dataset_root \
    --out-dir predictions
```

## Optional: Trained Backbone Evaluation

The evaluation script can optionally load trained DEKR and OpenPose models
for comparison. Set these environment variables to point to their
repositories:

```bash
export DEKR_ROOT=/path/to/DEKR
export OPENPOSE_ROOT=/path/to/pytorch-openpose
```

## Author

Jerry Chen (jzc23@pitt.edu)
