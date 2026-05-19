#!/usr/bin/env python3
# =============================================================================
# infer.py — Top-level inference entry point
# =============================================================================
"""
Entry point for running inference with GOOSE-M2F.

Usage:
    python infer.py --config configs/infer_config.yaml
"""

import argparse
import yaml
from src.inference import SegmentationInference


def parse_args():
    parser = argparse.ArgumentParser(description="Run GOOSE-M2F Inference")
    parser.add_argument("--config", type=str, default="configs/infer_config.yaml",
                        help="Path to inference config YAML file.")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    engine = SegmentationInference(cfg)
    engine.run()


if __name__ == "__main__":
    main()
