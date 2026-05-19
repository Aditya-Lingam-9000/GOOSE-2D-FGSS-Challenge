# =============================================================================
# src/model.py — GOOSE-M2F: Custom Mask2Former Architecture
# =============================================================================
"""
GOOSE Mask2Former: A task-specific adaptation of Mask2Former for the
GOOSE 2D Semantic Segmentation Challenge (64 classes, unstructured terrain).

Key Modifications over baseline Mask2Former:
  1. 200 Object Queries  — better coverage for 64 classes in complex scenes
  2. Feature Refinement Module (FRM) — ASPP-lite + CBAM dual-attention
  3. Auxiliary Supervision Head — direct per-pixel CE, bypasses query bottleneck
  4. Smart Query Initialization — pretrained weights + small Gaussian perturbations
  5. Gradient Checkpointing — VRAM-efficient training on large images

Weight initialization:
  facebook/mask2former-swin-large-cityscapes-semantic
    → Load into base_model (Swin-Large backbone + full M2F decoder)
    → Expand queries: 100 → 200 (copy + Gaussian perturbation)
    → New modules (FRM, aux_head) initialized with Kaiming/Xavier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from transformers.modeling_outputs import ModelOutput


# ─── Output Dataclass ────────────────────────────────────────────────────────

@dataclass
class GOOSEModelOutput(ModelOutput):
    """Structured output container for a GOOSEMask2Former forward pass."""
    loss: Optional[torch.Tensor] = None
    aux_logits: Optional[torch.Tensor] = None
    class_queries_logits: Optional[torch.Tensor] = None
    masks_queries_logits: Optional[torch.Tensor] = None


# ─── Attention Building Blocks ────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style Channel Attention (CBAM component).
    
    Learns WHAT feature channels are important by aggregating both
    average-pooled and max-pooled channel statistics through a shared MLP.
    
    Args:
        channels: Number of input/output feature channels.
        reduction: Bottleneck reduction ratio for the MLP. Default: 16.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (CBAM component).
    
    Learns WHERE to focus in the feature map by computing a spatial
    attention map from channel-pooled descriptors.
    
    Args:
        kernel_size: Conv kernel for the spatial gate. Default: 7.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_map = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_map


# ─── Feature Refinement Module ────────────────────────────────────────────────

class FeatureRefinementModule(nn.Module):
    """
    ASPP-lite + CBAM Feature Refinement Module.
    
    Positioned after the pixel decoder output to add multi-scale context
    (via dilated convolutions) and spatial/channel attention (via CBAM).
    
    Design rationale for GOOSE:
      - Small dilation (d=1, d=3): tiny objects — traffic_cone, wire, pole
      - Medium dilation (d=6): mid-scale — person, bicycle, motorcycle
      - Large dilation (d=12): large amorphous — building, forest, asphalt
      - Global pooling: scene-level — sky, snow

    Args:
        channels: Input/output channels (256 in standard Mask2Former).
        mid_channels: Bottleneck channels for each ASPP branch. Default: 64.
    """
    def __init__(self, channels: int = 256, mid_channels: int = 64):
        super().__init__()

        def _branch(dilation):
            return nn.Sequential(
                nn.Conv2d(channels, mid_channels, 3 if dilation > 1 else 1,
                          padding=dilation if dilation > 1 else 0,
                          dilation=dilation, bias=False),
                nn.GroupNorm(16, mid_channels),
                nn.ReLU(inplace=True),
            )

        self.branch_d1  = _branch(1)
        self.branch_d3  = _branch(3)
        self.branch_d6  = _branch(6)
        self.branch_d12 = _branch(12)

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.GroupNorm(16, mid_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 5, channels, 1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        # CBAM: sequential channel then spatial attention
        self.channel_attn = ChannelAttention(channels, reduction=16)
        self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        d1   = self.branch_d1(x)
        d3   = self.branch_d3(x)
        d6   = self.branch_d6(x)
        d12  = self.branch_d12(x)
        glob = F.interpolate(self.global_branch(x), size=(h, w),
                             mode='bilinear', align_corners=False)

        fused = self.fuse(torch.cat([d1, d3, d6, d12, glob], dim=1))

        # Residual connection + CBAM
        out = x + fused
        out = self.channel_attn(out)
        out = self.spatial_attn(out)
        return out


# ─── Auxiliary Head ──────────────────────────────────────────────────────────

class AuxiliaryHead(nn.Module):
    """
    Auxiliary per-pixel classification head.
    
    Provides DIRECT pixel-level supervision at H/4 resolution, preventing
    rare-class gradient starvation that occurs when no object query "claims"
    a tiny or rare semantic class (e.g., traffic_cone, wire, pipe).
    
    Removed at inference time to save VRAM (see inference.py).
    
    Args:
        in_channels: Input channels from the pixel decoder (default: 256).
        num_classes: Number of semantic classes (default: 64).
        mid_channels: Hidden dimension inside the head (default: 256).
    """
    def __init__(self, in_channels: int = 256, num_classes: int = 64,
                 mid_channels: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ─── Main Model ──────────────────────────────────────────────────────────────

class GOOSEMask2Former(nn.Module):
    """
    GOOSE-Optimized Mask2Former for 64-class Semantic Segmentation.
    
    Wraps a pretrained Mask2Former (Swin-Large backbone) and adds:
      - 200 object queries (expanded from 100 baseline)
      - Feature Refinement Module (FRM) with ASPP-lite + CBAM
      - Auxiliary Supervision Head at H/4 resolution
    
    Architecture flow:
        Input [B, 3, H, W]
          → Swin-Large Backbone (hierarchical, 4 stages)
          → MSDeformAttn Pixel Decoder (6-layer FPN)
          → [NEW] Feature Refinement Module (ASPP-lite + CBAM)
              → [NEW] Auxiliary Head (CE loss at H/4)
          → Transformer Decoder (9 layers, 200 queries)
          → Class Head [B, 200, 65] + Mask Head [B, 200, H/4, W/4]
    
    Args:
        num_classes: Number of semantic classes. Default: 64 (GOOSE dataset).
        num_queries: Object queries for the transformer. Default: 200.
        pretrained_name: HuggingFace model ID or local path.
    """

    def __init__(
        self,
        num_classes: int = 64,
        num_queries: int = 200,
        pretrained_name: str = "facebook/mask2former-swin-large-cityscapes-semantic",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        # 1. Load pretrained Mask2Former base
        from transformers import Mask2FormerForUniversalSegmentation
        print(f"[GOOSE-M2F] Loading base model: {pretrained_name}")
        self.base_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )

        # 2. Expand object queries: 100 → num_queries
        self._upgrade_queries(num_queries)

        # 3. Feature Refinement Module (after pixel decoder)
        hidden_dim = self.base_model.config.hidden_dim  # 256
        self.feature_refinement = FeatureRefinementModule(hidden_dim, mid_channels=64)

        # 4. Auxiliary per-pixel head
        self.aux_head = AuxiliaryHead(hidden_dim, num_classes, mid_channels=256)

        # 5. Hook to capture mask_features from the pixel level module
        self._captured_mask_features: Optional[torch.Tensor] = None
        self._hook_handle = self.base_model.model.pixel_level_module.register_forward_hook(
            self._capture_pixel_features_hook
        )

        # 6. Initialize new modules
        self._init_new_modules()

        # Summary
        total     = sum(p.numel() for p in self.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        print(f"[GOOSE-M2F] Total params: {total:.1f}M | Trainable: {trainable:.1f}M | Queries: {num_queries}")

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def _capture_pixel_features_hook(self, module, input, output):
        """Captures mask_features [B, 256, H/4, W/4] from the pixel level module."""
        if hasattr(output, "decoder_last_hidden_state"):
            self._captured_mask_features = output.decoder_last_hidden_state
        elif hasattr(output, "to_tuple"):
            self._captured_mask_features = output.to_tuple()[0]
        elif isinstance(output, dict):
            self._captured_mask_features = list(output.values())[0]
        elif isinstance(output, (tuple, list)):
            self._captured_mask_features = output[0]
        else:
            self._captured_mask_features = output

    # ── Query Upgrade ─────────────────────────────────────────────────────────

    def _upgrade_queries(self, new_num: int):
        """
        Expands all query embedding layers from 100 to new_num.
        New queries are initialized as small Gaussian perturbations of existing
        pretrained queries, preserving the learned query space.
        """
        transformer = self.base_model.model.transformer_module
        old_num = transformer.queries_features.num_embeddings
        if new_num <= old_num:
            print(f"[GOOSE-M2F] Queries: {old_num} (no expansion needed)")
            return

        print(f"[GOOSE-M2F] Expanding queries: {old_num} → {new_num}")

        def _expand(old_embed: nn.Embedding) -> nn.Embedding:
            dim = old_embed.embedding_dim
            new_embed = nn.Embedding(new_num, dim)
            with torch.no_grad():
                new_embed.weight[:old_num] = old_embed.weight.clone()
                for i in range(old_num, new_num):
                    src = i % old_num
                    new_embed.weight[i] = old_embed.weight[src] + 0.02 * torch.randn(dim)
            return new_embed

        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Embedding) and module.num_embeddings == old_num:
                parts = name.split('.')
                parent = self.base_model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], _expand(module))
                print(f"   Upgraded: {name}")

    # ── Weight Initialization ─────────────────────────────────────────────────

    def _init_new_modules(self):
        """Kaiming / Xavier initialization for newly added FRM and AuxHead."""
        for module_group in [self.feature_refinement, self.aux_head]:
            for m in module_group.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    # ── Forward Pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
        mask_labels=None,
        class_labels=None,
    ) -> GOOSEModelOutput:
        """
        Forward pass.

        Args:
            pixel_values:  [B, 3, H, W] — normalized input images.
            pixel_mask:    [B, H, W] — valid pixel mask (1=valid, 0=padding).
            mask_labels:   list of [N_i, H, W] per-image instance masks (train only).
            class_labels:  list of [N_i] per-image class IDs (train only).

        Returns:
            GOOSEModelOutput with loss, aux_logits, class_queries_logits,
            and masks_queries_logits.
        """
        self._captured_mask_features = None

        base_outputs = self.base_model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        aux_logits = None
        if self._captured_mask_features is not None:
            mask_features = self._captured_mask_features
            from torch.utils.checkpoint import checkpoint as grad_checkpoint
            if self.training and mask_features.requires_grad:
                refined = grad_checkpoint(self.feature_refinement, mask_features, use_reentrant=False)
                aux_logits = grad_checkpoint(self.aux_head, refined, use_reentrant=False)
            else:
                refined = self.feature_refinement(mask_features)
                aux_logits = self.aux_head(refined)

        return GOOSEModelOutput(
            loss=base_outputs.loss,
            aux_logits=aux_logits,
            class_queries_logits=base_outputs.class_queries_logits,
            masks_queries_logits=base_outputs.masks_queries_logits,
        )

    # ── Utility Methods ───────────────────────────────────────────────────────

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing across the full backbone to save VRAM."""
        try:
            if hasattr(self.base_model, "gradient_checkpointing_enable"):
                self.base_model.gradient_checkpointing_enable()
            else:
                self.base_model.model.pixel_level_module.encoder.gradient_checkpointing_enable()
            print("[GOOSE-M2F] Gradient checkpointing enabled.")
            return True
        except Exception as e:
            print(f"[GOOSE-M2F] Gradient checkpointing unavailable: {e}")
            return False

    def get_param_groups(self, backbone_lr: float, decoder_lr: float,
                         new_module_lr: Optional[float] = None):
        """
        Build differential learning-rate parameter groups.

        Args:
            backbone_lr: LR for Swin-Large backbone (smallest, preserves pretraining).
            decoder_lr: LR for pixel decoder + transformer decoder.
            new_module_lr: LR for FRM + AuxHead (defaults to decoder_lr).

        Returns:
            List of dicts compatible with torch.optim.
        """
        if new_module_lr is None:
            new_module_lr = decoder_lr

        backbone_params, decoder_params, new_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "base_model.model.pixel_level_module.encoder" in name:
                backbone_params.append(param)
            elif "feature_refinement" in name or "aux_head" in name:
                new_params.append(param)
            else:
                decoder_params.append(param)

        print(f"[GOOSE-M2F] Param groups — "
              f"Backbone: {sum(p.numel() for p in backbone_params)/1e6:.1f}M (lr={backbone_lr:.1e}) | "
              f"Decoder: {sum(p.numel() for p in decoder_params)/1e6:.1f}M (lr={decoder_lr:.1e}) | "
              f"New: {sum(p.numel() for p in new_params)/1e6:.1f}M (lr={new_module_lr:.1e})")

        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": decoder_params,  "lr": decoder_lr},
            {"params": new_params,       "lr": new_module_lr},
        ]

    def freeze_backbone(self):
        """Freeze the Swin-Large backbone for fine-tuning the decoder only."""
        n = sum(1 for name, p in self.named_parameters()
                if "pixel_level_module.encoder" in name and not p.requires_grad == False)
        for name, p in self.named_parameters():
            if "base_model.model.pixel_level_module.encoder" in name:
                p.requires_grad = False
        print(f"[GOOSE-M2F] Backbone frozen ({n} parameter tensors).")

    def unfreeze_all(self):
        """Unfreeze all parameters for end-to-end training."""
        for p in self.parameters():
            p.requires_grad = True
        print("[GOOSE-M2F] All parameters unfrozen.")
