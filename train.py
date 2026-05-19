#!/usr/bin/env python3
# =============================================================================
# train.py — Top-level training entry point
# =============================================================================
"""
Entry point for training GOOSE-M2F.

Usage:
    # Single GPU
    python train.py --config configs/train_config.yaml

    # Multi-GPU via Accelerate
    accelerate launch --num_processes 2 train.py --config configs/train_config.yaml
"""

import argparse
import yaml
from src.train import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train GOOSE-M2F Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to training config YAML file.")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
