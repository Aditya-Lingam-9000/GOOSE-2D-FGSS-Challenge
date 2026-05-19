# src/__init__.py
# GOOSE-M2F: Custom Mask2Former for Semantic Segmentation
from .model import GOOSEMask2Former, GOOSEModelOutput
from .features import GooseDataset, build_augmentations, m2f_collate_fn
from .train import Trainer
from .inference import SegmentationInference

__all__ = [
    "GOOSEMask2Former",
    "GOOSEModelOutput",
    "GooseDataset",
    "build_augmentations",
    "m2f_collate_fn",
    "Trainer",
    "SegmentationInference",
]
