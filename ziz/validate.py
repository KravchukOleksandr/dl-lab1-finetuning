#!/usr/bin/env python3
"""Validate a HelmetMicroNeXt checkpoint on one factory validation split."""

from __future__ import annotations

import argparse
from pathlib import Path

from train import (
    PROJECT_ROOT,
    _load_checkpoint,
    _project_path,
    _read_yaml_config,
    _resolve_device,
    _set_seed,
    _write_json,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "validate_config.yaml"


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    known_args, _ = config_parser.parse_known_args()
    config_path = _project_path(known_args.config).resolve()
    yaml_defaults = _read_yaml_config(config_path)

    parser = argparse.ArgumentParser(
        description="Validate a checkpoint on val/helmet and val/no_helmet."
    )
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument(
        "--output-json", type=Path, default=Path("models/validation_metrics.json")
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=False
    )

    valid_keys = {
        action.dest for action in parser._actions if action.dest not in {"help", "config"}
    }
    unknown_keys = sorted(set(yaml_defaults) - valid_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown parameters in {config_path}: {', '.join(unknown_keys)}"
        )
    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args()
    args.config = _project_path(args.config).resolve()
    for name in ("data_root", "checkpoint", "output_json"):
        setattr(args, name, _project_path(Path(getattr(args, name))).resolve())
    return args


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    try:
        import cv2  # noqa: F401
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise SystemExit(
            "Validation requires torch, numpy, PyYAML and opencv-python."
        ) from error

    from model import HelmetMicroNeXt, count_trainable_parameters
    from training.data import (
        HelmetCropDataset,
        cell_counts,
        discover_single_dataset_samples,
        seed_worker,
    )
    from training.evaluation import evaluate_model, load_model_weights

    device = _resolve_device(torch, args.device)
    amp_enabled = device.type == "cuda" and args.amp
    _set_seed(torch, np, args.seed, args.deterministic)

    samples = discover_single_dataset_samples(args.data_root, "val")
    dataset = HelmetCropDataset(samples, train=False)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    checkpoint = _load_checkpoint(torch, args.checkpoint)
    model = HelmetMicroNeXt().to(device)
    load_model_weights(model, checkpoint)
    metrics = evaluate_model(model, loader, device, amp_enabled, ["factory"])
    overall = metrics["overall"]
    counts = {
        class_name: count
        for (_dataset, class_name), count in cell_counts(samples).items()
    }
    payload = {
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "class_mapping": {"helmet": 0, "no_helmet": 1},
        "counts": counts,
        "model_parameters": count_trainable_parameters(model),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_json, payload)

    print(f"Checkpoint: {args.checkpoint}")
    print(
        f"Validation: {len(samples):,} files "
        f"(helmet {counts['helmet']:,}, no_helmet {counts['no_helmet']:,})"
    )
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"ROC-AUC: {overall['roc_auc']:.6f}")
    print(f"PR-AUC:  {overall['pr_auc']:.6f}")
    print(
        f"FPR@R95: {100.0 * overall['fpr_at_recall_95']:.3f}% "
        f"at threshold {overall['threshold_at_recall_95']:.6f}; "
        f"recall {100.0 * overall['achieved_recall_at_95']:.3f}%"
    )
    print(f"Metrics saved to: {args.output_json}")


if __name__ == "__main__":
    main()
