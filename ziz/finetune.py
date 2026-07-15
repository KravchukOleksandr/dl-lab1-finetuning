#!/usr/bin/env python3
"""Fine-tune HelmetMicroNeXt on one target-domain dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
from typing import Any

from train import (
    PROJECT_ROOT,
    _append_history,
    _learning_rate,
    _load_checkpoint,
    _make_grad_scaler,
    _project_path,
    _read_yaml_config,
    _resolve_device,
    _save_checkpoint,
    _set_seed,
    _write_json,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "finetune_config.yaml"


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    known_args, _ = config_parser.parse_known_args()
    config_path = _project_path(known_args.config).resolve()
    yaml_defaults = _read_yaml_config(config_path)

    parser = argparse.ArgumentParser(
        description="Fine-tune HelmetMicroNeXt on train/ and validate on val/."
    )
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models/finetune"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--batches-per-epoch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--selection-min-delta", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=5)
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
    for name in ("data_root", "initial_checkpoint", "output_dir", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, _project_path(Path(value)).resolve())
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0 or args.batch_size % 2 != 0:
        raise ValueError("--batch-size must be positive and divisible by 2")
    if args.val_batch_size <= 0:
        raise ValueError("--val-batch-size must be positive")
    if args.batches_per_epoch is not None and args.batches_per_epoch <= 0:
        raise ValueError("--batches-per-epoch must be positive or null")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if not 0.0 <= args.warmup_epochs < args.epochs:
        raise ValueError("--warmup-epochs must be >= 0 and < --epochs")
    if not 0.0 <= args.min_lr <= args.lr:
        raise ValueError("Expected 0 <= --min-lr <= --lr")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative")
    if not 1 <= args.min_epochs <= args.epochs:
        raise ValueError("--min-epochs must be between 1 and --epochs")
    if args.selection_min_delta < 0:
        raise ValueError("--selection-min-delta cannot be negative")
    if args.resume is None and args.initial_checkpoint is None:
        raise ValueError("Set --initial-checkpoint or --resume")


def _history_row(
    epoch: int,
    train_metrics: dict[str, float],
    validation: dict[str, Any],
) -> dict[str, Any]:
    overall = validation["overall"]
    cells = validation["cells"]
    return {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "lr": train_metrics["lr"],
        "val_loss": validation["overall_loss"],
        "val_macro_class_loss": validation["macro_cell_loss"],
        "val_accuracy": overall["accuracy"],
        "val_roc_auc": overall["roc_auc"],
        "val_pr_auc": overall["pr_auc"],
        "val_fpr_at_recall_95": overall["fpr_at_recall_95"],
        "val_threshold_at_recall_95": overall["threshold_at_recall_95"],
        "val_achieved_recall_at_95": overall["achieved_recall_at_95"],
        "val_helmet_loss": cells["factory/helmet"]["loss"],
        "val_helmet_accuracy": cells["factory/helmet"]["accuracy"],
        "val_no_helmet_loss": cells["factory/no_helmet"]["loss"],
        "val_no_helmet_accuracy": cells["factory/no_helmet"]["accuracy"],
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)

    try:
        import cv2  # noqa: F401
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise SystemExit(
            "Fine-tuning requires torch, numpy, PyYAML and opencv-python."
        ) from error

    from model import HelmetMicroNeXt, count_trainable_parameters
    from training.data import (
        DatasetClassBalancedBatchSampler,
        HelmetCropDataset,
        cell_counts,
        discover_single_dataset_samples,
        seed_worker,
    )
    from training.evaluation import evaluate_model, load_model_weights

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    if args.resume is None and history_path.exists():
        raise FileExistsError(
            f"A previous run already exists in {output_dir}. Use --resume or "
            "choose another --output-dir."
        )

    device = _resolve_device(torch, args.device)
    amp_enabled = device.type == "cuda" and args.amp
    _set_seed(torch, np, args.seed, args.deterministic)

    train_samples = discover_single_dataset_samples(args.data_root, "train")
    val_samples = discover_single_dataset_samples(args.data_root, "val")
    train_dataset = HelmetCropDataset(train_samples, train=True)
    val_dataset = HelmetCropDataset(val_samples, train=False)
    train_sampler = DatasetClassBalancedBatchSampler(
        train_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        batches_per_epoch=args.batches_per_epoch,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_common = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **loader_common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        **loader_common,
    )

    model = HelmetMicroNeXt().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = _make_grad_scaler(torch, amp_enabled)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = round(args.warmup_epochs * len(train_loader))
    start_epoch = 1
    global_step = 0
    best_fpr_at_recall_95 = math.inf
    epochs_without_improvement = 0

    if args.resume is not None:
        checkpoint = _load_checkpoint(torch, args.resume)
        load_model_weights(model, checkpoint)
        if "optimizer" not in checkpoint:
            raise ValueError("Resume checkpoint does not contain optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(
            checkpoint.get("global_step", (start_epoch - 1) * len(train_loader))
        )
        best_fpr_at_recall_95 = float(
            checkpoint.get("best_fpr_at_recall_95", math.inf)
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
    else:
        initial = _load_checkpoint(torch, args.initial_checkpoint)
        load_model_weights(model, initial)

    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint completed epoch {start_epoch - 1}, but epochs={args.epochs}"
        )

    config = vars(args).copy()
    config.update(
        {
            "resolved_device": str(device),
            "amp_enabled": amp_enabled,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "batches_per_epoch_resolved": len(train_loader),
            "samples_per_class_per_batch": train_sampler.per_cell,
            "train_class_counts": {
                class_name: count
                for (_dataset, class_name), count in cell_counts(train_samples).items()
            },
            "val_class_counts": {
                class_name: count
                for (_dataset, class_name), count in cell_counts(val_samples).items()
            },
            "class_mapping": {"helmet": 0, "no_helmet": 1},
            "best_checkpoint_metric": "val_fpr_at_recall_95",
        }
    )
    _write_json(output_dir / "config.json", config)

    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"Initial checkpoint: {args.initial_checkpoint if args.resume is None else args.resume}")
    print(f"Model parameters: {count_trainable_parameters(model):,}")
    print(
        f"Train: {len(train_samples):,} files, {len(train_loader):,} batches/epoch; "
        f"validation: {len(val_samples):,} files"
    )
    print(
        f"Each train batch: helmet {train_sampler.per_cell}, "
        f"no_helmet {train_sampler.per_cell}"
    )

    def train_one_epoch(epoch: int) -> dict[str, float]:
        nonlocal global_step
        model.train()
        train_sampler.set_epoch(epoch - 1)
        loss_sum = 0.0
        correct = 0
        sample_count = 0
        last_lr = args.lr

        for images, targets, _dataset_ids in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            last_lr = _learning_rate(
                global_step,
                total_steps,
                warmup_steps,
                args.lr,
                args.min_lr,
            )
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = functional.binary_cross_entropy_with_logits(logits, targets)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_count = int(targets.numel())
            loss_sum += float(loss.detach()) * batch_count
            correct += int(((logits.detach() >= 0) == (targets >= 0.5)).sum())
            sample_count += batch_count
            global_step += 1
        return {
            "loss": loss_sum / sample_count,
            "accuracy": correct / sample_count,
            "lr": last_lr,
        }

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.monotonic()
        train_metrics = train_one_epoch(epoch)
        validation = evaluate_model(
            model, val_loader, device, amp_enabled, ["factory"]
        )
        overall = validation["overall"]
        selection_value = overall["fpr_at_recall_95"]
        improved = (
            selection_value
            < best_fpr_at_recall_95 - args.selection_min_delta
        )
        if improved:
            best_fpr_at_recall_95 = selection_value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        _append_history(history_path, _history_row(epoch, train_metrics, validation))
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_fpr_at_recall_95": best_fpr_at_recall_95,
            "epochs_without_improvement": epochs_without_improvement,
            "validation": validation,
            "operating_threshold": overall["threshold_at_recall_95"],
            "config": config,
        }
        _save_checkpoint(torch, output_dir / "last.pt", payload)
        if improved:
            _save_checkpoint(torch, output_dir / "best.pt", payload)
            _write_json(
                output_dir / "best_metrics.json",
                {
                    "epoch": epoch,
                    "fpr_at_recall_95": selection_value,
                    "operating_threshold": overall["threshold_at_recall_95"],
                    "validation": validation,
                },
            )
        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_checkpoint(torch, output_dir / f"epoch_{epoch:03d}.pt", payload)

        marker = "  BEST" if improved else ""
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train loss {train_metrics['loss']:.4f}, "
            f"val loss {validation['overall_loss']:.4f}, "
            f"AUC {overall['roc_auc']:.4f}, "
            f"FPR@R95 {100.0 * selection_value:.2f}% "
            f"(thr {overall['threshold_at_recall_95']:.4f}, "
            f"R {100.0 * overall['achieved_recall_at_95']:.2f}%), "
            f"{time.monotonic() - started:.1f}s{marker}"
        )

        if (
            args.early_stopping_patience > 0
            and epoch >= args.min_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping: FPR@R95 did not improve by at least "
                f"{args.selection_min_delta:g} for "
                f"{epochs_without_improvement} epochs."
            )
            break

    print(f"Done. Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
