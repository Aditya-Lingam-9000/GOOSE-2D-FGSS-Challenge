# =============================================================================
# src/train.py — GOOSE-M2F Training Engine
# =============================================================================
"""
Full training loop for GOOSE-M2F with:
  - Multi-GPU distributed training via HuggingFace Accelerate
  - Dynamic IoU-Aware loss weighting (updated every epoch)
  - Auxiliary head CE loss (DB-weighted) at H/4 resolution
  - EMA weight tracking with automatic best-model checkpointing
  - Polynomial LR schedule with configurable warmup
  - Per-epoch matplotlib charts and JSON history logging
  - Overfitting diagnostics (train-val gap tracking)

Usage (single GPU):
    python -m src.train --config configs/train_config.yaml

Usage (multi-GPU with Accelerate):
    accelerate launch --num_processes 2 -m src.train --config configs/train_config.yaml
"""

import gc
import json
import os
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta
from torch.utils.data import DataLoader
from transformers import Mask2FormerImageProcessor

from .features import (
    GooseDataset,
    OfficialCompositeMetric,
    EMAModel,
    build_augmentations,
    compute_db_weights,
    build_cas_sampler,
    m2f_collate_fn,
    load_label_mapping,
    VOID_ID,
)
from .model import GOOSEMask2Former


# ─── LR Schedule ─────────────────────────────────────────────────────────────

def get_poly_lr_lambda(warmup_iters: int, total_iters: int,
                       power: float = 1.0, min_lr_ratio: float = 0.01):
    """
    Polynomial learning rate decay with linear warmup.

    Args:
        warmup_iters: Number of warmup steps (LR ramps from 0 to base LR).
        total_iters:  Total number of training iterations.
        power:        Polynomial decay power. 1.0 = linear decay.
        min_lr_ratio: Minimum LR as a fraction of base LR (avoids zero LR).

    Returns:
        Callable lr_lambda for torch.optim.lr_scheduler.LambdaLR.
    """
    def lr_lambda(current_iter: int) -> float:
        if current_iter < warmup_iters:
            return max(current_iter / max(warmup_iters, 1), 0.01)
        progress = (current_iter - warmup_iters) / max(total_iters - warmup_iters, 1)
        return max((1.0 - progress) ** power, min_lr_ratio)
    return lr_lambda


# ─── Dynamic IoU-Aware Weights ────────────────────────────────────────────────

def iou_to_tier(iou_pct: float):
    """
    Map per-class IoU% to a loss weight multiplier and tier label.
    
    Classes with 0% IoU get the maximum gradient pressure (4x).
    Classes at ≥80% are protected from over-weighting (1x).
    """
    if iou_pct == 0.0:      return 4.0, "CRITICAL"
    elif iou_pct < 15.0:    return 5.0, "VERY HIGH"
    elif iou_pct < 30.0:    return 3.0, "HIGH"
    elif iou_pct < 45.0:    return 2.0, "MED HIGH"
    elif iou_pct < 60.0:    return 1.5, "MEDIUM"
    elif iou_pct < 80.0:    return 1.2, "LOW"
    else:                    return 1.0, "NORMAL"


def compute_dynamic_iou_weights(
    per_class_iou: dict,
    class_names: dict,
    num_classes: int = 64,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build dynamic per-class loss weights from the latest validation IoU results.
    Called after every epoch to re-prioritize struggling classes.

    Args:
        per_class_iou: {"class_name": iou_pct} from OfficialCompositeMetric.compute().
        class_names:   {class_id: "class_name"} mapping.
        num_classes:   Total number of classes.
        device:        Target device for the returned tensor.

    Returns:
        torch.Tensor [num_classes] of loss weights.
    """
    weights = np.ones(num_classes, dtype=np.float32)
    weights[VOID_ID] = 0.0
    for cid in range(1, num_classes):
        name = class_names.get(cid, f"class_{cid}")
        iou  = per_class_iou.get(name, 0.0)
        weights[cid], _ = iou_to_tier(iou)
    return torch.from_numpy(weights).to(device)


# ─── Charting ────────────────────────────────────────────────────────────────

def save_epoch_charts(epoch_num: int, history: list, val_ema_results: dict,
                      prev_per_class: dict, dynamic_weights_np: np.ndarray,
                      class_names: dict, num_classes: int, out_dir: Path):
    """Save three diagnostic matplotlib charts to disk after every epoch."""
    import warnings
    warnings.filterwarnings("ignore")

    cids  = list(range(1, num_classes))
    names = [class_names.get(c, f"cls_{c}") for c in cids]
    curr_ious = [val_ema_results["per_class"].get(n, 0.0) for n in names]
    prev_ious = [prev_per_class.get(n, 0.0)               for n in names]
    weights   = [float(dynamic_weights_np[c])              for c in cids]
    x = np.arange(len(cids))

    # Chart 1: Per-class IoU comparison
    fig, ax = plt.subplots(figsize=(24, 8))
    ax.bar(x - 0.2, prev_ious, 0.38, label="Prev Epoch", color="#4a90d9", alpha=0.70)
    ax.bar(x + 0.2, curr_ious, 0.38, label="Curr Epoch", color="#e74c3c", alpha=0.85)
    ax.axhline(70, color="#27ae60", linestyle="--", lw=1.5, label="Target 70%")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_title(f"Epoch {epoch_num} — Per-Class IoU", fontsize=14, fontweight="bold")
    ax.set_ylabel("IoU %"); ax.set_ylim(0, 105); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_dir / f"chart1_iou_ep{epoch_num}.png"), dpi=120)
    plt.close()

    # Chart 2: Dynamic weight distribution
    colors = ["#922b21" if w >= 4 else "#e74c3c" if w >= 3 else
              "#e67e22" if w >= 2 else "#f39c12" if w >= 1.5 else
              "#27ae60" if w >= 1.2 else "#3498db" for w in weights]
    fig, ax = plt.subplots(figsize=(24, 6))
    ax.bar(x, weights, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_title(f"Epoch {epoch_num} — Dynamic Loss Weights", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss Weight"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_dir / f"chart2_weights_ep{epoch_num}.png"), dpi=120)
    plt.close()

    # Chart 3: Training progress curves
    if len(history) >= 2:
        eps = [h["epoch"] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        ax1.plot(eps, [h.get("official_composite", 0) for h in history],
                 "o-", color="#27ae60", lw=2.5, label="Official Composite")
        ax1.plot(eps, [h.get("val_ema_composite",  0) for h in history],
                 "s--", color="#3498db", lw=1.5, label="Val EMA Composite")
        ax1.plot(eps, [h.get("train_composite",    0) for h in history],
                 "^:", color="#e74c3c", lw=1.5, label="Train Composite")
        ax1.axhline(70, color="green", linestyle="--", lw=1.5)
        ax1.set_title("Composite Score History", fontweight="bold")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("%"); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(eps, [h.get("train_loss", 0) for h in history], "o-",
                 color="#e74c3c", lw=2, label="Train Loss")
        ax2.plot(eps, [h.get("val_loss",   0) for h in history], "s--",
                 color="#3498db", lw=2, label="Val Loss")
        ax2.set_title("Loss History", fontweight="bold")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(out_dir / f"chart3_progress_ep{epoch_num}.png"), dpi=120)
        plt.close()


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training engine for GOOSE-M2F.

    Handles dataset setup, model initialization, distributed training,
    evaluation, checkpointing, and metric logging.

    Args:
        cfg: Configuration dictionary (from configs/train_config.yaml).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_dir = Path(cfg["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Entry point: builds everything and runs the full training loop."""
        cfg = self.cfg

        # ── Accelerate ──
        timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        accelerator = Accelerator(
            mixed_precision=cfg.get("mixed_precision", "fp16"),
            gradient_accumulation_steps=cfg.get("grad_accum_steps", 16),
            kwargs_handlers=[timeout],
        )
        device    = accelerator.device
        is_main   = accelerator.is_main_process

        # ── Processor ──
        processor = Mask2FormerImageProcessor.from_pretrained(
            cfg["model_name"], do_resize=False, do_rescale=False, do_normalize=True
        )

        # ── Label Mapping ──
        _, class_names, fine_to_coarse = load_label_mapping(cfg["csv_path"])

        # ── Datasets ──
        train_tf = build_augmentations(cfg["crop_h"], cfg["crop_w"], mode="train")
        val_tf   = build_augmentations(cfg["crop_h"], cfg["crop_w"], mode="val")

        train_ds = GooseDataset(
            base_path=cfg["data_dir"], csv_path=cfg["csv_path"],
            splits="train", transform=train_tf, is_train=True,
            cutout_dir=cfg.get("cutout_dir"), copy_paste_prob=cfg.get("copy_paste_prob", 0.0),
            processor=processor,
        )
        val_ds = GooseDataset(
            base_path=cfg["data_dir"], csv_path=cfg["csv_path"],
            splits="val", transform=val_tf, is_train=False,
            processor=processor,
        )

        # ── Samplers & Loaders ──
        _, pixel_counts = compute_db_weights(train_ds, num_classes=cfg["num_classes"],
                                             beta=cfg.get("db_beta", 0.9999))
        sampler = build_cas_sampler(train_ds, pixel_counts,
                                    num_classes=cfg["num_classes"],
                                    threshold=cfg.get("cas_thresh", 0.06))

        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], sampler=sampler,
            collate_fn=m2f_collate_fn, num_workers=cfg.get("num_workers", 4),
            drop_last=True, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, collate_fn=m2f_collate_fn,
            num_workers=2, pin_memory=True,
        )

        # ── Model ──
        model = GOOSEMask2Former(
            num_classes=cfg["num_classes"],
            num_queries=cfg.get("num_queries", 200),
            pretrained_name=cfg["model_name"],
        )
        model.enable_gradient_checkpointing()

        # ── Optimizer ──
        param_groups = model.get_param_groups(cfg["backbone_lr"], cfg["decoder_lr"])
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.get("weight_decay", 0.03))

        # ── Resume ──
        start_epoch, best_composite, history = 0, 0.0, []
        mode = cfg.get("mode", "train")
        resume_ckpt = cfg.get("resume_checkpoint")

        if mode == "resume" and resume_ckpt:
            if is_main:
                print(f"[Trainer] Resuming from: {resume_ckpt}")
            ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt)
            # Truncate queries if checkpoint has different count
            for key in ["base_model.model.transformer_module.queries_embedder.weight",
                        "base_model.model.transformer_module.queries_features.weight"]:
                if key in state and state[key].shape[0] != cfg.get("num_queries", 200):
                    state[key] = state[key][:cfg.get("num_queries", 200)]
            model.load_state_dict(state, strict=False)
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                except Exception:
                    pass
            # Override LRs from config
            for i, pg in enumerate(optimizer.param_groups):
                pg["lr"] = cfg["backbone_lr"] if i == 0 else cfg["decoder_lr"]
                pg["initial_lr"] = pg["lr"]
            start_epoch = ckpt.get("epoch", 0)
            history     = ckpt.get("history", [])
            best_composite = ckpt.get("best_composite", 0.0)
            if is_main:
                print(f"[Trainer] Resumed epoch {start_epoch} | best={best_composite:.2f}%")

        # ── LR Schedule ──
        steps_per_epoch = len(train_loader)
        total_iters     = steps_per_epoch * cfg["num_epochs"]
        eff_warmup      = 0 if mode == "resume" else cfg.get("warmup_iters", 1000)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=get_poly_lr_lambda(eff_warmup, total_iters)
        )

        # ── Prepare Distributed ──
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

        # ── EMA ──
        unwrapped = accelerator.unwrap_model(model)
        ema = EMAModel(unwrapped, decay=cfg.get("ema_decay", 0.9995))
        if mode == "resume" and resume_ckpt:
            ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
            if "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])

        # ── Dynamic Weights Initialization ──
        init_iou = (history[-1]["per_class"] if history and "per_class" in history[-1]
                    else {class_names.get(k, f"class_{k}"): 0.0 for k in range(cfg["num_classes"])})
        dynamic_weights = compute_dynamic_iou_weights(
            init_iou, class_names, cfg["num_classes"], device
        )
        prev_per_class = init_iou.copy()
        aux_weight     = cfg.get("aux_weight", 0.4)
        label_smoothing = cfg.get("label_smoothing", 0.10)

        # ─────────────────────── TRAINING LOOP ──────────────────────────────
        for epoch in range(cfg["num_epochs"]):
            actual_epoch = start_epoch + epoch
            model.train()
            metric_train = OfficialCompositeMetric(cfg["num_classes"], fine_to_coarse, class_names)
            epoch_loss, epoch_start = 0.0, time.time()

            for step, (pv, pm, ml, cl, om) in enumerate(train_loader):
                with accelerator.accumulate(model):
                    outputs = model(pixel_values=pv, pixel_mask=pm,
                                    mask_labels=ml, class_labels=cl)
                    total_loss = outputs.loss

                    if outputs.aux_logits is not None:
                        aux_up = F.interpolate(outputs.aux_logits, size=om.shape[-2:],
                                               mode="bilinear", align_corners=False)
                        aux_ce = F.cross_entropy(
                            aux_up, om,
                            weight=dynamic_weights, ignore_index=VOID_ID,
                            label_smoothing=label_smoothing,
                        )
                        total_loss = total_loss + aux_weight * aux_ce

                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 5.0))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    ema.update(unwrapped)

                epoch_loss += total_loss.item()

                if step % 10 == 0:
                    with torch.no_grad():
                        preds = processor.post_process_semantic_segmentation(
                            outputs, target_sizes=[(cfg["crop_h"], cfg["crop_w"])] * pv.shape[0]
                        )
                        metric_train.update(preds[0], om[0])

                if step % 50 == 0 and is_main:
                    mem = torch.cuda.memory_allocated(device) / 1e9
                    eta = ((time.time() - epoch_start) / max(step, 1)) * (len(train_loader) - step)
                    print(f"  E{actual_epoch+1} [{step}/{len(train_loader)}] "
                          f"Loss={total_loss.item():.4f} | VRAM={mem:.1f}GB | ETA={eta/60:.1f}m")

            avg_train_loss = epoch_loss / len(train_loader)
            train_results  = metric_train.compute()
            gc.collect(); torch.cuda.empty_cache()

            # ── Validation ──
            def _validate(use_ema: bool) -> tuple:
                if use_ema:
                    ema.apply_shadow(unwrapped)
                model.eval()
                m = OfficialCompositeMetric(cfg["num_classes"], fine_to_coarse, class_names)
                loss_sum = 0.0
                with torch.no_grad():
                    for pv, pm, ml, cl, om in val_loader:
                        out = model(pixel_values=pv, pixel_mask=pm,
                                    mask_labels=ml, class_labels=cl)
                        loss_sum += out.loss.item()
                        preds = processor.post_process_semantic_segmentation(
                            out, target_sizes=[(cfg["crop_h"], cfg["crop_w"])] * pv.shape[0]
                        )
                        m.update(preds[0], om[0])
                        gc.collect()
                if use_ema:
                    ema.restore(unwrapped)
                return m.compute(), loss_sum / len(val_loader)

            val_raw_results, avg_val_loss = _validate(use_ema=False)
            val_ema_results, _            = _validate(use_ema=True)

            # ── Update dynamic weights from EMA val IoU ──
            new_weights = compute_dynamic_iou_weights(
                val_ema_results["per_class"], class_names, cfg["num_classes"], device
            )
            dynamic_weights.copy_(new_weights)
            if accelerator.num_processes > 1:
                import torch.distributed as dist
                dist.broadcast(dynamic_weights, src=0)

            # ── Best check & save ──
            is_best = val_ema_results["official_composite"] > best_composite
            if is_best:
                best_composite = val_ema_results["official_composite"]

            if is_main:
                h = {
                    "epoch":            actual_epoch + 1,
                    "train_loss":       avg_train_loss,
                    "val_loss":         avg_val_loss,
                    "train_composite":  train_results["all_composite"],
                    "val_ema_composite":val_ema_results["all_composite"],
                    "official_composite":val_ema_results["official_composite"],
                    "per_class":        val_ema_results["per_class"],
                }
                history.append(h)

                # Console summary
                print(f"\n{'='*65}")
                print(f"  Epoch {actual_epoch+1} | "
                      f"Train={train_results['all_composite']:.2f}% | "
                      f"Val={val_ema_results['all_composite']:.2f}% | "
                      f"Official={val_ema_results['official_composite']:.2f}%")
                print(f"  Best: {best_composite:.2f}%")
                print(f"{'='*65}\n")

                # Charts
                try:
                    save_epoch_charts(
                        actual_epoch + 1, history, val_ema_results,
                        prev_per_class, dynamic_weights.cpu().numpy(),
                        class_names, cfg["num_classes"], self.out_dir,
                    )
                except Exception as e:
                    print(f"[Trainer] Chart generation failed (non-critical): {e}")

                prev_per_class = val_ema_results["per_class"].copy()

                # Checkpoint
                state = {
                    "epoch":           actual_epoch + 1,
                    "model":           unwrapped.state_dict(),
                    "ema":             ema.state_dict(),
                    "optimizer":       optimizer.state_dict(),
                    "scheduler":       scheduler.state_dict(),
                    "best_composite":  best_composite,
                    "history":         history,
                    "config":          cfg,
                }
                torch.save(state, self.out_dir / "checkpoint_latest.pth")
                if is_best:
                    torch.save(state, self.out_dir / "best_model.pth")
                    print(f"[Trainer] NEW BEST saved: {best_composite:.2f}%")
                with open(self.out_dir / "training_history.json", "w") as f:
                    json.dump(history, f, indent=2)

            gc.collect(); torch.cuda.empty_cache()

        if is_main:
            print(f"\n[Trainer] Training complete. Best: {best_composite:.2f}%")
            print(f"[Trainer] Outputs saved to: {self.out_dir}")
