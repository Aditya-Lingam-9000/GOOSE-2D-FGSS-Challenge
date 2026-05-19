# GOOSE-M2F: Setup, Training & Inference Guide

A complete step-by-step guide for setting up the environment, preparing data, training the model, and running inference on any machine — from scratch to a 70%+ mIoU submission.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Project Structure Walkthrough](#3-project-structure-walkthrough)
4. [Configuration Guide](#4-configuration-guide)
5. [Training: Step-by-Step](#5-training-step-by-step)
6. [Resuming Training](#6-resuming-training)
7. [Inference: Step-by-Step](#7-inference-step-by-step)
8. [Running Tests](#8-running-tests)
9. [Common Errors & Fixes](#9-common-errors--fixes)
10. [Architecture Notes](#10-architecture-notes)

---

## 1. Environment Setup

### 1.1 Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|----------------|-------|
| Python      | 3.10           | 3.11+ recommended |
| CUDA        | 11.8           | 12.x preferred |
| VRAM (GPU)  | 24 GB          | 40 GB for batch_size=2 |
| RAM         | 32 GB          | 64 GB preferred |
| Disk Space  | ~50 GB         | Dataset + model weights |

### 1.2 Clone the Repository

```bash
git clone https://github.com/your-username/goose-m2f.git
cd goose-m2f
```

### 1.3 Create a Virtual Environment

```bash
# Using conda (recommended)
conda create -n goose python=3.11 -y
conda activate goose

# OR using venv
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### 1.4 Install PyTorch (CUDA)

Always install PyTorch first, separately, to ensure correct CUDA compatibility:

```bash
# For CUDA 12.1 (most common on modern GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Expected: True  12.1
```

### 1.5 Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 1.6 Configure Accelerate (Multi-GPU)

HuggingFace Accelerate handles distributed training. Run the configuration wizard:

```bash
accelerate config
```

Answer the prompts:
- **Machine type**: `This machine`
- **Multi-GPU?**: `yes` (if you have multiple GPUs)
- **Number of GPUs**: `2` (or 1)
- **Mixed precision**: `fp16`
- **Distributed type**: `multi-GPU`

This saves a config at `~/.cache/huggingface/accelerate/default_config.yaml`.

### 1.7 Download the Pretrained Weights (One-Time)

The model downloads automatically on first use. To pre-cache it manually:

```bash
python -c "
from transformers import Mask2FormerForUniversalSegmentation
Mask2FormerForUniversalSegmentation.from_pretrained(
    'facebook/mask2former-swin-large-cityscapes-semantic',
    use_safetensors=True
)
print('Downloaded successfully.')
"
```

The weights are cached at `~/.cache/huggingface/hub/` (~866 MB).

---

## 2. Dataset Preparation

### 2.1 Download the GOOSE Dataset

Download the GOOSE 2D Semantic Segmentation dataset from the official challenge page:
- https://goose-dataset.de/

The dataset should have the following structure:
```
goose_dataset/
├── goose_2d_train/
│   ├── images/train/               ← Training images (.png)
│   └── labels/train/               ← Label ID maps (*_labelids.png)
├── goose_2d_val/
│   ├── images/val/
│   └── labels/val/
├── gooseEx_2d_train/               ← Extended dataset (color labels)
│   ├── images/
│   └── labels/
└── goose_label_mapping.csv         ← Class names, IDs, and hex colors
```

Place the dataset directory anywhere on disk. You will set the path in `configs/train_config.yaml`.

### 2.2 Place Dataset Files

Update the paths in `configs/train_config.yaml`:
```yaml
data_dir:  "/absolute/path/to/goose_dataset"
csv_path:  "/absolute/path/to/goose_dataset/goose_label_mapping.csv"
```

### 2.3 (Optional) Extract Rare-Class Cutouts for Copy-Paste

For the Rare-Class Copy-Paste (RCCP) augmentation to work, you need pre-extracted cutouts
of rare semantic classes (e.g., traffic_cone, wire, animal).

The cutout directory should contain pairs of files:
```
goose_rare_cutouts/
├── cutout_1_img.png       ← Image crop of class 1 (traffic_cone)
├── cutout_1_mask.png      ← Binary mask for the crop above
├── cutout_7_img.png       ← Image crop of class 7 (bikeway)
├── cutout_7_mask.png
└── ...
```

The filename format is: `<unique_id>_<class_id>_img.png` and `<unique_id>_<class_id>_mask.png`.

Set the path in the config:
```yaml
cutout_dir:      "/absolute/path/to/goose_rare_cutouts"
copy_paste_prob: 0.85
```

If you don't have cutouts, set `copy_paste_prob: 0.0` to disable.

### 2.4 Verify Dataset Indexing

Run a quick sanity check to ensure the dataset is discoverable:

```bash
python - <<'EOF'
from src.features import GooseDataset, build_augmentations
ds = GooseDataset(
    base_path="/your/dataset/path",
    csv_path="/your/dataset/path/goose_label_mapping.csv",
    splits="train",
    transform=build_augmentations(576, 1152, "val"),
)
print(f"Found {len(ds)} training samples.")
img = ds[0]
print(f"Sample keys: {list(img.keys())}")
EOF
```

Expected output:
```
[GooseDataset] Found 18xxx image-label pairs for split=train
Found 18xxx training samples.
Sample keys: ['pixel_values', 'pixel_mask', 'mask_labels', 'class_labels', 'original_mask']
```

---

## 3. Project Structure Walkthrough

```
goose-m2f/
│
├── src/                        ← Core Python package
│   ├── __init__.py             ← Package exports
│   ├── model.py                ← GOOSEMask2Former architecture (FRM + AuxHead)
│   ├── features.py             ← Dataset, augmentations, EMA, metrics
│   ├── train.py                ← Training engine (Trainer class)
│   └── inference.py            ← Dense patch-blending inference engine
│
├── configs/
│   ├── train_config.yaml       ← All training hyperparameters
│   └── infer_config.yaml       ← Inference hyperparameters + TTA settings
│
├── data/
│   ├── raw/                    ← Symlink or copy your dataset here
│   └── processed/              ← Pre-extracted cutouts, processed files
│
├── models/                     ← Manually downloaded/placed checkpoint files
│
├── notebooks/                  ← Exploratory notebooks (EDA, visualization)
│
├── outputs/
│   ├── checkpoints/            ← best_model.pth, checkpoint_latest.pth
│   └── predictions/            ← Inference output PNG files
│
├── tests/
│   ├── __init__.py
│   └── test_model.py           ← Unit tests (pytest)
│
├── instructions/
│   └── instructions.md         ← This file
│
├── requirements.txt
└── README.md
```

### Key Design Decisions

| File | Role |
|------|------|
| `src/model.py` | **Only place** the GOOSE-M2F architecture is defined. Single source of truth. |
| `src/features.py` | Everything data-related: dataset, augmentations, EMA, metrics. |
| `src/train.py` | Training logic completely separated from model/data. |
| `src/inference.py` | Inference pipeline completely separated — AuxHead stripped here. |
| `configs/*.yaml` | **All** hyperparameters live here. No magic numbers in code. |

---

## 4. Configuration Guide

### train_config.yaml — Key Settings Explained

```yaml
# ─── Most commonly changed settings ───────────────────────────

# How many epochs to train in this session
num_epochs: 3

# The model base — do not change unless you know what you are doing
model_name: "facebook/mask2former-swin-large-cityscapes-semantic"

# ─── Learning Rate Strategy ────────────────────────────────────

# The Swin-Large backbone should always have a LOWER LR than the decoder.
# In early sessions: backbone_lr=5e-6, decoder_lr=5e-6
# In mid sessions (V4+): backbone_lr=1e-5, decoder_lr=5e-5  ← 10x LR jump
# In late sessions (V8): backbone_lr=5e-6, decoder_lr=2.5e-5 ← annealing
backbone_lr: 1.0e-5
decoder_lr:  5.0e-5

# ─── Auxiliary Loss ────────────────────────────────────────────

# Weight of the Auxiliary Head CE loss relative to M2F loss.
# Range: 0.3-0.5. Higher = stronger per-pixel supervision for rare classes.
aux_weight: 0.4

# ─── Copy-Paste Augmentation ───────────────────────────────────

# Probability of pasting rare-class cutouts onto each image.
# 0.0 = disabled (for final fine-tuning sessions to avoid distribution shift)
# 0.85 = recommended for early/mid training sessions
copy_paste_prob: 0.85

# ─── EMA Decay ─────────────────────────────────────────────────

# Higher decay = slower, more stable averaging.
# 0.999 = faster absorption (better for high-LR phases)
# 0.9995 = slower, more stable (better for annealing phases)
ema_decay: 0.9995
```

---

## 5. Training: Step-by-Step

### Step 1: Edit the Config

Open `configs/train_config.yaml` and set:
- `data_dir` → your dataset root
- `csv_path` → path to `goose_label_mapping.csv`
- `output_dir` → where you want checkpoints saved
- `mode: "train"` for a fresh run

### Step 2: Run Training

**Single GPU:**
```bash
python -m src.train --config configs/train_config.yaml
```

**Multi-GPU (using Accelerate):**
```bash
accelerate launch --num_processes 2 -m src.train --config configs/train_config.yaml
```

**On Kaggle / Colab (adapt paths in config, then):**
```bash
# Kaggle (2x T4 or 2x P100)
accelerate launch --num_processes 2 -m src.train --config configs/train_config.yaml
```

### Step 3: Monitor Progress

After every epoch you will see:
```
================================================================
  Epoch 1 | Train=52.3% | Val=48.7% | Official=49.1%
  Best: 49.1%
================================================================
```

Three charts are saved to `output_dir`:
- `chart1_iou_ep1.png` — per-class IoU vs previous epoch
- `chart2_weights_ep1.png` — dynamic loss weight distribution
- `chart3_progress_ep1.png` — composite score and loss curves

### Step 4: Locate Checkpoints

After each epoch:
```
outputs/checkpoints/session_01/
├── checkpoint_latest.pth     ← Overwritten every epoch
├── best_model.pth            ← Saved only when official composite improves
├── training_history.json     ← Full epoch-by-epoch metric log
├── chart1_iou_ep1.png
├── chart2_weights_ep1.png
└── chart3_progress_ep1.png
```

---

## 6. Resuming Training

To continue from a saved checkpoint (e.g., after session 1, starting session 2):

1. Edit `configs/train_config.yaml`:
```yaml
mode:              "resume"
resume_checkpoint: "outputs/checkpoints/session_01/best_model.pth"
output_dir:        "outputs/checkpoints/session_02"

# Optionally adjust LR for the new session:
backbone_lr: 5.0e-6   # Anneal for fine-tuning
decoder_lr:  2.5e-5
```

2. Run as normal:
```bash
accelerate launch --num_processes 2 -m src.train --config configs/train_config.yaml
```

**What gets restored from checkpoint:**
- Model weights (with automatic query-count truncation if `num_queries` differs)
- EMA shadow weights
- Optimizer state
- Epoch number and training history
- Best composite score

**What gets overridden (intentionally):**
- Learning rates (always taken from config, not checkpoint)
- Warmup (disabled when `mode=resume` to prevent LR drop-to-zero)

---

## 7. Inference: Step-by-Step

### Step 1: Edit the Inference Config

Open `configs/infer_config.yaml` and set:
```yaml
image_dir:       "/path/to/test/images"
csv_path:        "/path/to/goose_label_mapping.csv"
checkpoint_path: "outputs/checkpoints/session_02/best_model.pth"
output_dir:      "outputs/predictions"
gpu_id:          0
```

### Step 2: Run Inference

```bash
python -m src.inference --config configs/infer_config.yaml
```

### Step 3: TTA Settings

The default TTA uses 4 scales × 2 flips = **8 augmented views per image**:
```yaml
tta_scales: [0.5, 0.75, 1.0, 1.5]   # Each scale is also H-flipped
stride:     384                        # Dense stride (57% overlap for 896x896 crops)
```

To speed up inference (at the cost of accuracy), reduce TTA:
```yaml
tta_scales: [0.75, 1.0]   # 4 views — faster
stride:     512            # Less dense — much faster
```

### Step 4: Output Format

For each input image `<stem>.png`, two files are saved:
- `<stem>_pred.png` — Colorized RGB prediction (GOOSE color palette)
- `<stem>_ids.png` — Raw grayscale label-ID map (uint8, 0-63)

---

## 8. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only architecture tests
pytest tests/test_model.py::TestFeatureRefinementModule -v

# Run only metric tests
pytest tests/test_model.py::TestOfficialCompositeMetric -v
```

Expected output:
```
tests/test_model.py::TestFeatureRefinementModule::test_output_shape_unchanged PASSED
tests/test_model.py::TestFeatureRefinementModule::test_residual_non_trivial    PASSED
tests/test_model.py::TestAuxiliaryHead::test_output_channels                   PASSED
...
```

---

## 9. Common Errors & Fixes

### `CUDA out of memory`

**Cause**: Batch size or crop size too large for your GPU.
**Fix**:
```yaml
# In train_config.yaml:
crop_h: 512       # Reduce from 576
crop_w: 1024      # Reduce from 1152
batch_size: 1     # Keep at 1
grad_accum_steps: 32   # Increase to compensate effective batch
```

Also ensure gradient checkpointing is enabled (it is by default in `Trainer.run()`).

### `RuntimeError: loaded state dict contains a key that does not exist`

**Cause**: Checkpoint was saved with a different model config (e.g., different `num_queries`).
**Fix**: This is handled automatically — mismatched query weights are truncated. If the error persists, add `strict=False` to `load_state_dict`.

### `KeyError: 'loss'` during training

**Cause**: `mask_labels` or `class_labels` is empty (no valid instances in crop).
**Fix**: The collate function handles empty batches, but if your dataset has very sparse labels, increase crop size so at least one labeled pixel is always present.

### `FileNotFoundError` for label files

**Cause**: Dataset directory structure does not match what the dataset indexer expects.
**Fix**: Check that the path pattern matches:
```
<data_dir>/goose_2d_train/images/train/*.png  ← images
<data_dir>/goose_2d_train/labels/train/*_labelids.png  ← labels
```

### Accelerate multi-GPU hangs indefinitely

**Cause**: Model download lock contention between processes.
**Fix**: Pre-download the model to a local directory before launching:
```bash
python -c "from transformers import Mask2FormerForUniversalSegmentation; \
           Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-large-cityscapes-semantic')"
```
Then set `model_name` to the local cache path (find it at `~/.cache/huggingface/hub/`).

---

## 10. Architecture Notes

### Why 200 Queries?

Mask2Former uses bipartite matching between N object queries and ground-truth segments. With 64 classes and complex GOOSE scenes, 100 queries create "representational saturation" — multiple semantically distinct regions share one query, leading to segmentation collisions. 200 queries provide sufficient headroom for even the most cluttered scenes.

### Why ASPP-lite in the FRM?

Amorphous classes (forest, bush, moss) have no clear boundary — they "leak" into neighboring terrain without sufficient receptive field context. ASPP's multi-rate dilated convolutions (d=1,3,6,12) explicitly capture context at four spatial scales simultaneously, allowing the model to integrate wide-area evidence before the transformer decoder.

### Why the Auxiliary Head?

Ultra-thin classes (wire, pole, traffic_sign) often occupy fewer than 50 pixels in a 576×1152 crop. No object query will "win" the Hungarian matching assignment for them, leaving them with zero gradient signal. The Auxiliary Head provides direct pixel-level CE supervision at H/4 resolution, ensuring these classes receive gradients regardless of whether any query claimed them.

### Why Gaussian Patch Blending?

Standard sliding-window inference with max-pooling creates visible grid artifacts at crop boundaries because the model is most confident at crop centers. By weighting predictions with a 2D Gaussian kernel (higher weight at center, lower at edges) and mean-pooling overlapping crops, the boundary artifacts are mathematically eliminated.

### The 10x LR Jump (V4)

Training sessions V1–V3 used conservative LRs (backbone=1e-6, decoder=5e-6). The model was stuck in a local minimum, especially for Object and Water categories. In V4, we executed a 10x LR jump (backbone=1e-5, decoder=5e-5) which forced rapid adaptation, pulling the official composite from ~55% to ~56.38% in a single session and unlocking further gains in subsequent sessions.
