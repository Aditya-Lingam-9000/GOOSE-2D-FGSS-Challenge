# GOOSE-M2F

> **GOOSE-M2F** is a task-specific adaptation of [Mask2Former](https://github.com/facebookresearch/Mask2Former) for the [GOOSE 2D Semantic Segmentation Challenge](https://goose-dataset.de/) — 64-class unstructured outdoor terrain parsing.

**Official Composite mIoU: 70.08%** (Fine mIoU + Coarse mIoU / 2)

---

## What is GOOSE-M2F?

The GOOSE dataset presents one of the most challenging real-world segmentation benchmarks: 64 fine-grained classes across diverse unstructured outdoor environments including forests, gravel paths, construction zones, and agricultural terrain — with a severely long-tailed class distribution.

GOOSE-M2F extends the baseline Mask2Former (Swin-Large backbone) with three key modifications engineered specifically for this challenge:

| Modification | Problem Solved | Impact |
|---|---|---|
| **200 Object Queries** (vs 100) | Query saturation in 64-class scenes | +2-3% composite mIoU |
| **Feature Refinement Module (FRM)** — ASPP-lite + CBAM | Over-segmentation of amorphous terrain classes | +3-4% on Vegetation/Terrain |
| **Auxiliary Supervision Head** at H/4 resolution | Vanishing gradients for tiny/thin classes | +5-8% on rare classes |

---

## Architecture

```
Input Image [B, 3, H, W]
      │
      ▼
Swin-Large Backbone (Hierarchical, 4 stages)
  Stage 1-4: channels {192, 384, 768, 1536}, resolutions {H/4 → H/32}
      │
      ▼
MSDeformAttn Pixel Decoder (6-layer FPN)
  Output: mask_features [B, 256, H/4, W/4]
      │
      ├──────────────────────────────────────┐
      ▼                                      ▼
[NEW] Feature Refinement Module        [NEW] Auxiliary Head
  ASPP-lite: dilations {1, 3, 6, 12}     Conv(256→256→64)
  + Global Average Pooling               DB-weighted CE loss
  + CBAM Dual-Attention (Ch + Sp)        Supervised at H/4
      │
      ▼
Transformer Decoder (9 layers)
  [MOD] 200 Object Queries (was 100)
  Masked Cross-Attention
      │
      ▼
Class Head [B, 200, 65] × Mask Head [B, 200, H/4, W/4]
      │
      ▼
Hungarian Matching → Semantic Prediction
```

---

## Training Strategy

| Technique | Description |
|---|---|
| **Distribution-Balanced (DB) Loss** | `w_c = (1-β)/(1-β^n_c)`, β=0.9999. Amplifies gradients for rare classes. |
| **Rare-Class Copy-Paste (RCCP)** | Pre-extracted rare-class cutouts pasted onto training images at 85% probability. |
| **Dynamic IoU-Aware Weights** | Per-class loss weights updated every epoch from validation IoU (0%→4x, 80%+→1x). |
| **10x LR Jump (V4)** | Backbone 1e-5, Decoder 5e-5 — broke the model out of a local minimum at ~55%. |
| **EMA (decay=0.9995)** | Shadow weights consistently +1.0–1.5% over raw model on validation. |
| **Class-Aware Repeat Sampling** | Oversamples images containing rare classes proportional to their rarity. |
| **Polynomial LR Decay** | Gradual decay after warmup, with annealing in final sessions. |

### Training Progression (V1 → V8)

| Session | Base LR | Backbone LR | Official Score |
|---------|---------|-------------|----------------|
| V1 (S3) | 5e-6 | 1e-6 | 50.68% |
| V2 (S4) | 5e-6 | 1e-6 | 54.62% |
| V3 (S5) | 5e-6 | 1e-6 | 55.64% |
| V4 (S6) | **5e-5** | **1e-5** | 56.38% ← **10x LR Jump** |
| V5 (S7) | 5e-5 | 1e-5 | 57.59% |
| V6 (S8) | 5e-5 | 1e-5 | 58.58% |
| V7 (S9) | 5e-5 | 1e-5 | 59.23% |
| V8 (S10) | **2.5e-5** | **5e-6** | 59.51% ← Annealing |
| **Inference** | — | — | **70.08%** ← +10.57% from TTA |

---

## Inference Engine

The final performance leap from 59.51% (training) to 70.08% (submission) came entirely from the inference pipeline:

| Technique | Gain | Description |
|---|---|---|
| **Dense Sliding Window** | +4-5% | 896×896 crops, stride=384px (57% overlap) |
| **2D Gaussian Kernel Blending** | Eliminates artifacts | Center pixels weighted higher, edges down-weighted |
| **4-Scale TTA** | +3-4% | Scales: 0.5×, 0.75×, 1.0×, 1.5× |
| **H-Flip TTA** | +1-2% | 8 total views per image (4 scales × 2 flips) |
| **EMA Weights** | +1-1.5% | Shadow weights used instead of raw training weights |
| **AuxHead Stripping** | VRAM savings | Removed before inference — not needed for prediction |

---

## Project Structure

```
goose-m2f/
├── src/
│   ├── model.py          ← GOOSEMask2Former (FRM + AuxHead + 200 queries)
│   ├── features.py       ← Dataset, augmentations, EMA, metrics
│   ├── train.py          ← Training engine (Trainer class)
│   └── inference.py      ← Dense Gaussian patch-blending inference
├── configs/
│   ├── train_config.yaml ← All training hyperparameters
│   └── infer_config.yaml ← TTA and inference settings
├── data/raw/             ← Dataset (symlink or copy)
├── models/               ← Manually placed checkpoints
├── outputs/
│   ├── checkpoints/      ← best_model.pth, latest.pth, charts
│   └── predictions/      ← Output PNG predictions
├── tests/
│   └── test_model.py     ← pytest unit tests
├── instructions/
│   └── instructions.md   ← Full setup + usage guide
└── requirements.txt
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/your-username/goose-m2f.git
cd goose-m2f

conda create -n goose python=3.11 -y && conda activate goose
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
accelerate config   # Configure for your GPU setup
```

### 2. Configure Paths

Edit `configs/train_config.yaml`:
```yaml
data_dir: "/path/to/goose_dataset"
csv_path: "/path/to/goose_label_mapping.csv"
output_dir: "outputs/checkpoints/session_01"
```

### 3. Train

```bash
# Single GPU
python -m src.train --config configs/train_config.yaml

# Multi-GPU
accelerate launch --num_processes 2 -m src.train --config configs/train_config.yaml
```

### 4. Inference

Edit `configs/infer_config.yaml` with the checkpoint path and image directory, then:
```bash
python -m src.inference --config configs/infer_config.yaml
```

### 5. Tests

```bash
pytest tests/ -v
```

---

## Results

### Official Leaderboard Performance (Final Submission)

| Metric | Score |
|---|---|
| Fine mIoU | ~68.5% |
| Coarse mIoU | ~71.6% |
| **Official Composite** | **70.08%** |

### Coarse Category Breakdown

| Category | mIoU |
|---|---|
| Sky | 94.6% |
| Road | 91.0% |
| Vehicle | 89.8% |
| Vegetation | 89.8% |
| Construction | 75.5% |
| Terrain | 78.9% |
| Human | 62.8% |
| Sign | 62.4% |
| Water | 33.9% |
| Object | 51.3% |
| Animal | 0.0% |

---

## Requirements

| Package | Version |
|---|---|
| torch | ≥ 2.1.0 |
| transformers | ≥ 4.38.0 |
| accelerate | ≥ 0.27.0 |
| albumentations | ≥ 1.3.1 |
| opencv-python | ≥ 4.9.0 |
| numpy | ≥ 1.24.0 |

See `requirements.txt` for the complete list.

---

## Citation

If you use this work, please cite:

```bibtex
@techreport{goosem2f2025,
  title     = {GOOSE-M2F: Adapting Mask2Former for High-Fidelity,
               Long-Tailed Semantic Segmentation},
  author    = {Team GOOSE},
  year      = {2025},
  institution = {ICRA 2025 GOOSE 2D Segmentation Challenge}
}
```

---

## References

- **Mask2Former**: Cheng et al., *Masked-Attention Mask Transformer for Universal Image Segmentation*, CVPR 2022
- **Swin Transformer**: Liu et al., ICCV 2021
- **CBAM**: Woo et al., *Convolutional Block Attention Module*, ECCV 2018
- **DeepLab**: Chen et al., *Rethinking Atrous Convolution*, TPAMI 2017

---

> For detailed setup instructions, training tips, and troubleshooting, see [instructions/instructions.md](instructions/instructions.md).
