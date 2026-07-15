#!/usr/bin/env python3
"""Pretrain HelmetMicroNeXt on the prepared public crop datasets.

Every training batch is balanced across all dataset x class combinations.
Validation always uses every crop exactly once and remains unbalanced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "train_config.yaml"
SELECTION_METRICS = (
    "macro_cell_loss",
    "macro_dataset_fpr_at_recall_95",
    "worst_dataset_fpr_at_recall_95",
    "overall_fpr_at_recall_95",
    "dataset_fpr_at_recall_95",
)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise SystemExit(
            "PyYAML is required to read the training config: pip install pyyaml"
        ) from error

    if not path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"The YAML root must be a mapping: {path}")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError("All YAML parameter names must be strings")
    return payload


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    known_args, _ = config_parser.parse_known_args()
    config_path = _project_path(known_args.config).resolve()
    yaml_defaults = _read_yaml_config(config_path)

    parser = argparse.ArgumentParser(
        description=(
            "Train HelmetMicroNeXt with exact dataset x class balancing "
            "inside every training batch."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="YAML config. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data_prep",
        help="Directory containing <dataset>/{train,val}/{helmet,no_helmet}.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["shel5k", "sfchd", "shwd"],
        help="Prepared dataset directory names.",
    )
    parser.add_argument(
        "--dataset-weights",
        nargs="*",
        default=None,
        help="Dataset weights as name=value; YAML may use a mapping.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="Must be divisible by 2 * number of datasets (96 = 16 per cell).",
    )
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=None,
        help="Balanced train batches per epoch; by default uses ceil(N / batch_size).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-start-factor", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument(
        "--ema", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0 or mps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "internet_pretrain",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint to resume from (normally last.pt).",
    )
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default="macro_cell_loss",
        help="Validation metric minimized when selecting best.pt.",
    )
    parser.add_argument(
        "--selection-dataset",
        default=None,
        help="Target dataset when selection_metric=dataset_fpr_at_recall_95.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--selection-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer reproducibility over CUDA speed.",
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
    args.data_root = Path(args.data_root)
    args.output_dir = Path(args.output_dir)
    args.resume = Path(args.resume) if args.resume is not None else None
    if args.dataset_weights is None:
        args.dataset_weights = {}
    elif isinstance(args.dataset_weights, dict):
        args.dataset_weights = {
            str(name): float(weight)
            for name, weight in args.dataset_weights.items()
        }
    else:
        parsed_weights: dict[str, float] = {}
        for item in args.dataset_weights:
            if "=" not in item:
                raise ValueError(
                    f"Invalid dataset weight {item!r}; expected name=value"
                )
            name, raw_weight = item.split("=", maxsplit=1)
            parsed_weights[name] = float(raw_weight)
        args.dataset_weights = parsed_weights
    return args


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(torch: Any, np: Any, seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _make_grad_scaler(torch: Any, enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _load_checkpoint(torch: Any, path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _learning_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
    warmup_start_factor: float = 0.0,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        if warmup_steps == 1:
            return max_lr
        progress = step / (warmup_steps - 1)
        factor = warmup_start_factor + (1.0 - warmup_start_factor) * progress
        return max_lr * factor
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def _save_checkpoint(torch: Any, path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _append_history(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _flatten_metrics(epoch: int, train: dict[str, float], val: dict[str, Any]) -> dict[str, Any]:
    overall = val["overall"]
    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train["loss"],
        "train_accuracy": train["accuracy"],
        "lr": train["lr"],
        "val_macro_cell_loss": val["macro_cell_loss"],
        "val_macro_cell_accuracy": val["macro_cell_accuracy"],
        "val_macro_dataset_fpr_at_recall_95": val[
            "macro_dataset_fpr_at_recall_95"
        ],
        "val_worst_dataset_fpr_at_recall_95": val[
            "worst_dataset_fpr_at_recall_95"
        ],
        "val_overall_loss": val["overall_loss"],
        "val_accuracy": overall["accuracy"],
        "val_roc_auc": overall["roc_auc"],
        "val_pr_auc": overall["pr_auc"],
        "val_fpr_at_recall_95": overall["fpr_at_recall_95"],
        "val_threshold_at_recall_95": overall["threshold_at_recall_95"],
        "val_achieved_recall_at_95": overall["achieved_recall_at_95"],
    }
    for dataset_name, metrics in val["datasets"].items():
        prefix = f"val_{dataset_name}"
        row[f"{prefix}_loss"] = metrics["loss"]
        row[f"{prefix}_accuracy"] = metrics["accuracy"]
        row[f"{prefix}_roc_auc"] = metrics["roc_auc"]
        row[f"{prefix}_fpr_at_recall_95"] = metrics["fpr_at_recall_95"]
        row[f"{prefix}_threshold_at_recall_95"] = metrics[
            "threshold_at_recall_95"
        ]
        row[f"{prefix}_achieved_recall_at_95"] = metrics[
            "achieved_recall_at_95"
        ]
    for cell_name, metrics in val["cells"].items():
        dataset_name, class_name = cell_name.split("/", maxsplit=1)
        prefix = f"val_{dataset_name}_{class_name}"
        row[f"{prefix}_loss"] = metrics["loss"]
        row[f"{prefix}_accuracy"] = metrics["accuracy"]
    return row


def _checkpoint_selection(
    validation: dict[str, Any],
    selection_metric: str,
    selection_dataset: str | None,
) -> tuple[float, float | None]:
    """Return the minimized value and its operating threshold, if defined."""
    if selection_metric == "overall_fpr_at_recall_95":
        overall = validation["overall"]
        return (
            float(overall["fpr_at_recall_95"]),
            float(overall["threshold_at_recall_95"]),
        )
    if selection_metric == "dataset_fpr_at_recall_95":
        if selection_dataset is None:
            raise ValueError(
                "selection_dataset is required for dataset_fpr_at_recall_95"
            )
        dataset_metrics = validation["datasets"][selection_dataset]
        return (
            float(dataset_metrics["fpr_at_recall_95"]),
            float(dataset_metrics["threshold_at_recall_95"]),
        )
    return float(validation[selection_metric]), None


def _checkpoint_selection_rank(
    validation: dict[str, Any],
    selection_metric: str,
    selection_dataset: str | None,
) -> tuple[float, ...]:
    """Rank FPR checkpoints like the previous stable training pipeline."""
    value, threshold = _checkpoint_selection(
        validation, selection_metric, selection_dataset
    )
    if selection_metric == "overall_fpr_at_recall_95":
        metrics = validation["overall"]
    elif selection_metric == "dataset_fpr_at_recall_95":
        if selection_dataset is None:
            raise ValueError("selection_dataset is required")
        metrics = validation["datasets"][selection_dataset]
    else:
        return (value,)
    return (
        value,
        -float(metrics["achieved_recall_at_95"]),
        -float(metrics["roc_auc"]),
        -float(threshold),
    )


def _selection_rank_is_better(
    current: tuple[float, ...],
    best: tuple[float, ...] | None,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    epsilon = 1e-12
    if current[0] < best[0] - min_delta:
        return True
    if abs(current[0] - best[0]) > epsilon:
        return False
    for current_item, best_item in zip(current[1:], best[1:]):
        if current_item < best_item - epsilon:
            return True
        if current_item > best_item + epsilon:
            return False
    return False


def main() -> None:
    args = parse_args()

    # Heavy dependencies are intentionally imported after argparse, so
    # `python train.py --help` also works in lightweight environments.
    try:
        import cv2  # noqa: F401 - verifies the data pipeline dependency
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise SystemExit(
            "Training dependencies are missing. Install torch, numpy and "
            "opencv-python in the active environment."
        ) from error

    from model import HelmetMicroNeXt, count_trainable_parameters
    from training.data import (
        DatasetClassBalancedBatchSampler,
        HelmetCropDataset,
        PRIVATE_RGB_MEAN,
        PRIVATE_RGB_STD,
        cell_counts,
        discover_samples,
        seed_worker,
    )
    from training.metrics import validation_metrics
    from training.ema import ModelEMA

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0 or args.val_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if not 0.0 <= args.warmup_epochs < args.epochs:
        raise ValueError("--warmup-epochs must be >= 0 and < --epochs")
    if not 0.0 < args.warmup_start_factor <= 1.0:
        raise ValueError("--warmup-start-factor must be in (0, 1]")
    if not 0.0 <= args.min_lr <= args.lr:
        raise ValueError("Expected 0 <= --min-lr <= --lr")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.batches_per_epoch is not None and args.batches_per_epoch <= 0:
        raise ValueError("--batches-per-epoch must be positive or null")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative")
    if not 1 <= args.min_epochs <= args.epochs:
        raise ValueError("--min-epochs must be between 1 and --epochs")
    if args.selection_min_delta < 0:
        raise ValueError("--selection-min-delta cannot be negative")
    if args.ema and not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in (0, 1)")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("--datasets must not contain duplicate names")
    if args.selection_metric == "dataset_fpr_at_recall_95":
        if args.selection_dataset not in args.datasets:
            raise ValueError(
                "--selection-dataset must name one of --datasets when "
                "selection_metric=dataset_fpr_at_recall_95"
            )
    unknown_weight_names = sorted(set(args.dataset_weights) - set(args.datasets))
    if unknown_weight_names:
        raise ValueError(
            "Dataset weights contain names absent from --datasets: "
            + ", ".join(unknown_weight_names)
        )
    args.dataset_weights = {
        dataset_name: float(args.dataset_weights.get(dataset_name, 1.0))
        for dataset_name in args.datasets
    }

    data_root = _project_path(args.data_root).resolve()
    output_dir = _project_path(args.output_dir).resolve()
    args.data_root = data_root
    args.output_dir = output_dir
    if args.resume is not None:
        args.resume = _project_path(args.resume).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    if args.resume is None and history_path.exists():
        raise FileExistsError(
            f"A previous run already exists in {output_dir}. Use --resume "
            "or choose another --output-dir."
        )
    device = _resolve_device(torch, args.device)
    amp_enabled = device.type == "cuda" and args.amp
    _set_seed(torch, np, args.seed, args.deterministic)

    train_samples = discover_samples(data_root, "train", args.datasets)
    val_samples = discover_samples(data_root, "val", args.datasets)
    train_dataset = HelmetCropDataset(train_samples, train=True)
    val_dataset = HelmetCropDataset(val_samples, train=False)
    train_sampler = DatasetClassBalancedBatchSampler(
        train_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        batches_per_epoch=args.batches_per_epoch,
        dataset_weights=args.dataset_weights,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)
    loader_common = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        generator=train_generator,
        **loader_common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        generator=val_generator,
        **loader_common,
    )

    model = HelmetMicroNeXt().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = _make_grad_scaler(torch, amp_enabled)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = round(args.warmup_epochs * len(train_loader))
    global_step = 0
    start_epoch = 1
    best_selection_value = math.inf
    best_selection_rank: tuple[float, ...] | None = None
    epochs_without_improvement = 0
    checkpoint: dict[str, Any] | None = None

    if args.resume is not None:
        checkpoint = _load_checkpoint(torch, args.resume)
        if list(checkpoint.get("dataset_names", [])) != list(args.datasets):
            raise ValueError(
                "Checkpoint dataset order differs from --datasets; refusing to resume"
            )
        model.load_state_dict(checkpoint.get("raw_model", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", (start_epoch - 1) * len(train_loader)))
        checkpoint_selection_metric = checkpoint.get(
            "selection_metric", "macro_cell_loss"
        )
        if checkpoint_selection_metric != args.selection_metric:
            raise ValueError(
                "Checkpoint selection metric differs from --selection-metric; "
                "start a new output directory instead of resuming"
            )
        checkpoint_selection_dataset = checkpoint.get("selection_dataset")
        if checkpoint_selection_dataset != args.selection_dataset:
            raise ValueError(
                "Checkpoint selection dataset differs from "
                "--selection-dataset; start a new output directory"
            )
        best_selection_value = float(
            checkpoint.get(
                "best_selection_value",
                checkpoint.get("best_macro_cell_loss", math.inf),
            )
        )
        saved_rank = checkpoint.get("best_selection_rank")
        best_selection_rank = (
            tuple(float(item) for item in saved_rank)
            if saved_rank is not None
            else (best_selection_value,)
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint already completed epoch {start_epoch - 1}, "
                f"but --epochs={args.epochs}"
            )

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None
    if ema is not None and checkpoint is not None:
        ema_state = checkpoint.get("ema_model", checkpoint.get("model"))
        if ema_state is not None:
            ema.module.load_state_dict(ema_state)

    config = vars(args).copy()
    config.update(
        {
            "resolved_device": str(device),
            "amp_enabled": amp_enabled,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "batches_per_epoch": len(train_loader),
            "cells_per_batch": len(train_sampler.groups),
            "train_batch_cell_counts": {
                f"{train_sampler.dataset_names[dataset_id]}/"
                f"{'helmet' if label == 0 else 'no_helmet'}": count
                for (dataset_id, label), count in train_sampler.cell_batch_counts.items()
            },
            "train_cell_counts": {
                f"{dataset}/{class_name}": count
                for (dataset, class_name), count in sorted(cell_counts(train_samples).items())
            },
            "val_cell_counts": {
                f"{dataset}/{class_name}": count
                for (dataset, class_name), count in sorted(cell_counts(val_samples).items())
            },
            "normalization_rgb_mean": PRIVATE_RGB_MEAN.reshape(-1).tolist(),
            "normalization_rgb_std": PRIVATE_RGB_STD.reshape(-1).tolist(),
            "class_mapping": {"helmet": 0, "no_helmet": 1},
            "best_checkpoint_metric": args.selection_metric,
        }
    )
    _write_json(output_dir / "config.json", config)

    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"EMA: {args.ema} (decay {args.ema_decay:g})")
    print(f"Model parameters: {count_trainable_parameters(model):,}")
    print(
        f"Train: {len(train_samples):,} files, {len(train_loader):,} batches/epoch; "
        f"validation: {len(val_samples):,} files"
    )
    print(f"Each train batch: {args.batch_size} samples")
    for dataset_id, dataset_name in train_sampler.dataset_names.items():
        helmet_count = train_sampler.cell_batch_counts[(dataset_id, 0)]
        no_helmet_count = train_sampler.cell_batch_counts[(dataset_id, 1)]
        print(
            f"  batch {dataset_name:7s}: weight "
            f"{train_sampler.dataset_weights[dataset_name]:g}, "
            f"helmet {helmet_count}, no_helmet {no_helmet_count}"
        )
    for (dataset_name, class_name), count in sorted(cell_counts(train_samples).items()):
        print(f"  train {dataset_name:7s}/{class_name:9s}: {count:6,d}")

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
                args.warmup_start_factor,
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
            if ema is not None:
                ema.update(model)

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

    @torch.inference_mode()
    def validate() -> dict[str, Any]:
        evaluation_model = ema.module if ema is not None else model
        evaluation_model.eval()
        all_logits: list[Any] = []
        all_targets: list[Any] = []
        all_losses: list[Any] = []
        all_dataset_ids: list[Any] = []
        for images, targets, dataset_ids in val_loader:
            images = images.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = evaluation_model(images)
                losses = functional.binary_cross_entropy_with_logits(
                    logits,
                    targets_device,
                    reduction="none",
                )
            all_logits.append(logits.float().cpu().numpy())
            all_targets.append(targets.numpy())
            all_losses.append(losses.float().cpu().numpy())
            all_dataset_ids.append(dataset_ids.numpy())
        return validation_metrics(
            logits=np.concatenate(all_logits),
            targets=np.concatenate(all_targets).astype(np.int64),
            losses=np.concatenate(all_losses),
            dataset_ids=np.concatenate(all_dataset_ids),
            dataset_names=args.datasets,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.monotonic()
        train_metrics = train_one_epoch(epoch)
        val_metrics = validate()
        selection_value, selection_threshold = _checkpoint_selection(
            val_metrics,
            args.selection_metric,
            args.selection_dataset,
        )
        selection_rank = _checkpoint_selection_rank(
            val_metrics,
            args.selection_metric,
            args.selection_dataset,
        )
        improved = _selection_rank_is_better(
            selection_rank,
            best_selection_rank,
            args.selection_min_delta,
        )
        if improved:
            best_selection_value = selection_value
            best_selection_rank = selection_rank
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = _flatten_metrics(epoch, train_metrics, val_metrics)
        _append_history(history_path, row)
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            # `model` is always inference-ready (EMA when enabled).
            "model": (ema.module if ema is not None else model).state_dict(),
            "raw_model": model.state_dict(),
            "ema_model": ema.module.state_dict() if ema is not None else None,
            "ema_enabled": ema is not None,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "selection_metric": args.selection_metric,
            "selection_dataset": args.selection_dataset,
            "selection_value": selection_value,
            "operating_threshold": selection_threshold,
            "best_selection_value": best_selection_value,
            "best_selection_rank": list(best_selection_rank),
            "epochs_without_improvement": epochs_without_improvement,
            "validation": val_metrics,
            "dataset_names": list(args.datasets),
            "config": config,
        }
        _save_checkpoint(torch, output_dir / "last.pt", payload)
        if improved:
            _save_checkpoint(torch, output_dir / "best.pt", payload)
            _write_json(
                output_dir / "best_metrics.json",
                {
                    "epoch": epoch,
                    "selection_metric": args.selection_metric,
                    "selection_dataset": args.selection_dataset,
                    "selection_value": selection_value,
                    "operating_threshold": selection_threshold,
                    "validation": val_metrics,
                },
            )
        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_checkpoint(torch, output_dir / f"epoch_{epoch:03d}.pt", payload)

        elapsed = time.monotonic() - epoch_started
        overall = val_metrics["overall"]
        marker = "  BEST" if improved else ""
        selection_name = args.selection_metric
        if args.selection_dataset is not None:
            selection_name += f"[{args.selection_dataset}]"
        if "fpr_at_recall_95" in args.selection_metric:
            selection_display = f"{100.0 * selection_value:.2f}%"
        else:
            selection_display = f"{selection_value:.4f}"
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train loss {train_metrics['loss']:.4f}, "
            f"val macro-cell loss {val_metrics['macro_cell_loss']:.4f}, "
            f"worst-dataset FPR@R95 "
            f"{100.0 * val_metrics['worst_dataset_fpr_at_recall_95']:.2f}%, "
            f"AUC {overall['roc_auc']:.4f}, "
            f"FPR@R95 {100.0 * overall['fpr_at_recall_95']:.2f}% "
            f"(thr {overall['threshold_at_recall_95']:.4f}, "
            f"R {100.0 * overall['achieved_recall_at_95']:.2f}%), "
            f"selection {selection_name}={selection_display}, "
            f"{elapsed:.1f}s{marker}"
        )
        for dataset_name, metrics in val_metrics["datasets"].items():
            print(
                f"  {dataset_name:7s}: loss {metrics['loss']:.4f}, "
                f"AUC {metrics['roc_auc']:.4f}, "
                f"FPR@R95 {100.0 * metrics['fpr_at_recall_95']:.2f}% "
                f"(thr {metrics['threshold_at_recall_95']:.4f}, "
                f"R {100.0 * metrics['achieved_recall_at_95']:.2f}%)"
            )

        if (
            args.early_stopping_patience > 0
            and epoch >= args.min_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping: {args.selection_metric} did not improve by "
                f"at least {args.selection_min_delta:g} for "
                f"{epochs_without_improvement} epochs."
            )
            break

    print(f"Done. Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
