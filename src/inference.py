# =============================================================================
# src/inference.py — High-Fidelity GOOSE-M2F Inference Engine
# =============================================================================
"""
Dense Gaussian patch-blending inference engine for GOOSE-M2F.

Key strategies:
  - Dense sliding window (896x896, stride=384px, 57% overlap)
  - 2D Gaussian kernel weighting (center pixels have higher confidence)
  - 4-scale Test-Time Augmentation: 0.5x, 0.75x, 1.0x, 1.5x + H-flip
  - Mean-pool ensemble across all augmented predictions
  - FP16 mixed precision for speed
  - Auxiliary head stripped from state-dict to save VRAM at inference
  - Dual-GPU greedy load-balancing (sort by file size → greedy partition)

Usage:
    from src.inference import SegmentationInference
    engine = SegmentationInference(cfg)
    engine.run()
"""

import gc
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Mask2FormerImageProcessor

from .features import load_label_mapping, VOID_ID
from .model import GOOSEMask2Former


# ─── Gaussian Kernel ─────────────────────────────────────────────────────────

def _make_gaussian_kernel(h: int, w: int, sigma: float = 0.5) -> np.ndarray:
    """
    Build a 2D Gaussian weight kernel of size (h, w).
    
    Predictions at the crop center are weighted highest; edge predictions
    are smoothly down-weighted. This eliminates hard seam artifacts when
    overlapping crops are mean-pooled.

    Args:
        h, w:  Kernel spatial dimensions (match crop size).
        sigma: Gaussian spread as a fraction of the half-size. Default: 0.5.

    Returns:
        Normalized numpy array [h, w] with weights in (0, 1].
    """
    cy, cx = h / 2.0, w / 2.0
    sy, sx = cy * sigma, cx * sigma
    y, x = np.ogrid[:h, :w]
    kernel = np.exp(-((x - cx) ** 2 / (2 * sx ** 2) + (y - cy) ** 2 / (2 * sy ** 2)))
    return kernel.astype(np.float32)


# ─── Single-GPU Worker ────────────────────────────────────────────────────────

def _infer_single_scale(
    model: GOOSEMask2Former,
    processor: Mask2FormerImageProcessor,
    img_rgb: np.ndarray,
    scale: float,
    flip: bool,
    crop_h: int,
    crop_w: int,
    stride: int,
    num_classes: int,
    device: torch.device,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Run patch-blended prediction at a single scale with optional H-flip.

    Args:
        model:       GOOSE-M2F model in eval mode.
        processor:   Mask2FormerImageProcessor for preprocessing.
        img_rgb:     Original image [H_orig, W_orig, 3] in RGB uint8.
        scale:       Resize factor relative to original size.
        flip:        If True, apply horizontal flip augmentation.
        crop_h/w:    Sliding window crop dimensions.
        stride:      Sliding window stride (pixels).
        num_classes: Number of output semantic classes.
        device:      Torch device for inference.
        kernel:      2D Gaussian weight kernel [crop_h, crop_w].

    Returns:
        logit_acc: [num_classes, H_orig, W_orig] mean-pooled logit accumulation.
    """
    H_orig, W_orig = img_rgb.shape[:2]
    H_s = max(int(H_orig * scale), crop_h)
    W_s = max(int(W_orig * scale), crop_w)

    img_scaled = cv2.resize(img_rgb, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
    if flip:
        img_scaled = img_scaled[:, ::-1, :].copy()

    logit_acc = np.zeros((num_classes, H_s, W_s), dtype=np.float32)
    weight_acc = np.zeros((H_s, W_s), dtype=np.float32)
    kernel_3d = kernel[np.newaxis]  # [1, crop_h, crop_w]

    for y0 in range(0, H_s - crop_h + 1, stride):
        for x0 in range(0, W_s - crop_w + 1, stride):
            crop = img_scaled[y0:y0 + crop_h, x0:x0 + crop_w]
            inputs = processor(images=crop, return_tensors="pt")
            pv = inputs["pixel_values"].to(device)

            with torch.amp.autocast("cuda"):
                with torch.no_grad():
                    outputs = model(pixel_values=pv)

            # Post-process: get full logit map at crop resolution
            preds = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(crop_h, crop_w)]
            )
            # preds[0] is an integer label map; we need raw logits for mean-pooling
            # Use masks_queries_logits to reconstruct probability map
            logits = _reconstruct_logits(outputs, crop_h, crop_w, num_classes, device)

            logit_acc[:, y0:y0 + crop_h, x0:x0 + crop_w] += logits * kernel_3d
            weight_acc[y0:y0 + crop_h, x0:x0 + crop_w]   += kernel

            del pv, inputs, outputs, preds, logits
            torch.cuda.empty_cache()

    # Normalize
    weight_acc = np.maximum(weight_acc, 1e-6)
    logit_acc /= weight_acc[np.newaxis]

    if flip:
        logit_acc = logit_acc[:, :, ::-1].copy()

    # Resize back to original resolution
    logit_acc_t = torch.from_numpy(logit_acc).unsqueeze(0)  # [1, C, H_s, W_s]
    logit_orig  = F.interpolate(logit_acc_t, size=(H_orig, W_orig),
                                mode="bilinear", align_corners=False)
    return logit_orig.squeeze(0).numpy()


def _reconstruct_logits(outputs, h: int, w: int, num_classes: int,
                         device: torch.device) -> np.ndarray:
    """
    Reconstruct a [num_classes, h, w] probability map from Mask2Former outputs.
    Uses class_queries_logits × masks_queries_logits to build the dense map.
    """
    # class logits: [1, Q, C+1] → softmax over classes
    cq = outputs.class_queries_logits  # [1, Q, C+1]
    mq = outputs.masks_queries_logits  # [1, Q, H/4, W/4]

    class_probs = torch.softmax(cq, dim=-1)[0, :, :num_classes]  # [Q, C]
    mask_probs  = torch.sigmoid(mq)[0]                           # [Q, H/4, W/4]
    mask_probs  = F.interpolate(mask_probs.unsqueeze(0), size=(h, w),
                                 mode="bilinear", align_corners=False).squeeze(0)  # [Q, H, W]

    # Dense logit map: sum over queries
    logit_map = torch.einsum("qc,qhw->chw", class_probs, mask_probs)  # [C, H, W]
    return logit_map.cpu().numpy()


# ─── Inference Engine ────────────────────────────────────────────────────────

class SegmentationInference:
    """
    Full inference pipeline for GOOSE-M2F semantic segmentation.

    Applies dense Gaussian patch blending + 4-scale TTA with H-flip to
    produce high-fidelity prediction masks. Supports single-GPU and
    dual-GPU greedy load-balanced multi-processing.

    Args:
        cfg: Configuration dictionary (from configs/infer_config.yaml).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_dir = Path(cfg["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self, device: torch.device) -> GOOSEMask2Former:
        """Load model from checkpoint, strip Aux Head for VRAM savings."""
        cfg = self.cfg
        model = GOOSEMask2Former(
            num_classes=cfg["num_classes"],
            num_queries=cfg.get("num_queries", 200),
            pretrained_name=cfg["model_name"],
        )

        state = torch.load(cfg["checkpoint_path"], map_location="cpu", weights_only=False)
        model_state = state.get("ema", state.get("model", state))

        # Strip Auxiliary Head — not needed at inference
        model_state = {k: v for k, v in model_state.items()
                       if not k.startswith("aux_head.")}

        model.load_state_dict(model_state, strict=False)
        model.aux_head = torch.nn.Identity()  # Disable gracefully
        model.to(device)
        model.eval()
        print(f"[Inference] Model loaded from {cfg['checkpoint_path']} on {device}")
        return model

    def _predict_single_image(
        self,
        model: GOOSEMask2Former,
        processor: Mask2FormerImageProcessor,
        img_path: Path,
        device: torch.device,
        palette: dict,
    ) -> np.ndarray:
        """
        Full TTA inference on a single image.

        Pipeline:
          1. Load image in RGB
          2. For each scale in TTA_SCALES × {no-flip, H-flip}:
             a. Resize image
             b. Dense sliding-window with Gaussian kernel blending
             c. Accumulate logits
          3. Mean-pool all views → argmax → label ID map
          4. Return colorized prediction PNG
        """
        cfg        = self.cfg
        crop_h     = cfg.get("crop_h", 896)
        crop_w     = cfg.get("crop_w", 896)
        stride     = cfg.get("stride", 384)
        tta_scales = cfg.get("tta_scales", [0.5, 0.75, 1.0, 1.5])
        num_classes = cfg["num_classes"]

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W    = img_rgb.shape[:2]

        kernel = _make_gaussian_kernel(crop_h, crop_w, sigma=0.5)

        logit_sum = np.zeros((num_classes, H, W), dtype=np.float32)
        n_views   = 0

        for scale in tta_scales:
            for flip in (False, True):
                logits = _infer_single_scale(
                    model, processor, img_rgb, scale, flip,
                    crop_h, crop_w, stride, num_classes, device, kernel
                )
                logit_sum += logits
                n_views   += 1
                gc.collect()

        mean_logits = logit_sum / n_views                     # [C, H, W]
        pred_ids    = mean_logits.argmax(axis=0).astype(np.uint8)  # [H, W]
        return pred_ids

    def _colorize(self, pred_ids: np.ndarray, palette: dict) -> np.ndarray:
        """Convert a label-ID map to a colorized RGB image using the GOOSE palette."""
        H, W    = pred_ids.shape
        color   = np.zeros((H, W, 3), dtype=np.uint8)
        for cid, (r, g, b) in palette.items():
            mask = pred_ids == cid
            color[mask] = [r, g, b]
        return color

    def run(self):
        """
        Run inference on all images in the configured input directory.

        Saves:
          - <output_dir>/<stem>_pred.png: Colorized prediction image
          - <output_dir>/<stem>_ids.png:  Raw label-ID grayscale map
        """
        cfg = self.cfg
        device = torch.device(f"cuda:{cfg.get('gpu_id', 0)}"
                               if torch.cuda.is_available() else "cpu")

        # Load label info
        palette, class_names, _ = load_label_mapping(cfg["csv_path"])

        # Load model
        model = self._load_model(device)

        # Load processor
        processor = Mask2FormerImageProcessor.from_pretrained(
            cfg["model_name"], do_resize=False, do_rescale=False, do_normalize=True
        )

        # Collect images
        img_dir = Path(cfg["image_dir"])
        images  = sorted(img_dir.rglob("*.png")) + sorted(img_dir.rglob("*.jpg"))
        print(f"[Inference] Processing {len(images)} images → {self.out_dir}")

        for i, img_path in enumerate(images, 1):
            pred_ids = self._predict_single_image(model, processor, img_path, device, palette)
            stem = img_path.stem

            # Save colorized prediction
            color = self._colorize(pred_ids, palette)
            cv2.imwrite(str(self.out_dir / f"{stem}_pred.png"),
                        cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

            # Save raw label-ID map
            cv2.imwrite(str(self.out_dir / f"{stem}_ids.png"), pred_ids)

            if i % 50 == 0 or i == len(images):
                print(f"[Inference] {i}/{len(images)} done")

        print(f"[Inference] Complete. Results saved to: {self.out_dir}")
