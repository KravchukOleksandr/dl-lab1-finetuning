"""
Training utilities for HelmetMicroNeXt.

Class mapping:
    0 = helmet     (каска есть)
    1 = no_helmet  (каски нет; positive class for precision/recall/F1/AUC)
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self, prefix: str = "") -> dict[str, float | int]:
        return {
            f"{prefix}loss": self.loss,
            f"{prefix}accuracy": self.accuracy,
            f"{prefix}precision": self.precision,
            f"{prefix}recall": self.recall,
            f"{prefix}f1": self.f1,
            f"{prefix}auc": self.auc,
            f"{prefix}tn": self.tn,
            f"{prefix}fp": self.fp,
            f"{prefix}fn": self.fn,
            f"{prefix}tp": self.tp,
        }


def _calculate_metrics(
    targets: list[int],
    probabilities: list[float],
    average_loss: float,
    threshold: float = 0.5,
) -> EpochMetrics:
    y_true = np.asarray(targets, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return EpochMetrics(
        loss=float(average_loss),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    max_grad_norm: float | None,
    threshold: float,
    epoch: int,
    total_epochs: int,
) -> EpochMetrics:
    model.train()

    running_loss = 0.0
    sample_count = 0
    targets: list[int] = []
    probabilities: list[float] = []

    progress = tqdm(
        loader,
        desc=f"Train {epoch:03d}/{total_epochs:03d}",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels, _paths in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        if max_grad_norm is not None and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.shape[0]
        running_loss += float(loss.detach().item()) * batch_size
        sample_count += batch_size

        batch_probabilities = torch.sigmoid(logits.detach()).cpu().numpy()
        probabilities.extend(batch_probabilities.tolist())
        targets.extend(labels.detach().cpu().to(torch.int64).numpy().tolist())

        progress.set_postfix(loss=f"{running_loss / max(sample_count, 1):.4f}")

    return _calculate_metrics(
        targets=targets,
        probabilities=probabilities,
        average_loss=running_loss / max(sample_count, 1),
        threshold=threshold,
    )


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    threshold: float,
    epoch: int,
    total_epochs: int,
) -> EpochMetrics:
    model.eval()

    running_loss = 0.0
    sample_count = 0
    targets: list[int] = []
    probabilities: list[float] = []

    progress = tqdm(
        loader,
        desc=f"Val   {epoch:03d}/{total_epochs:03d}",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels, _paths in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size

        batch_probabilities = torch.sigmoid(logits).cpu().numpy()
        probabilities.extend(batch_probabilities.tolist())
        targets.extend(labels.cpu().to(torch.int64).numpy().tolist())

        progress.set_postfix(loss=f"{running_loss / max(sample_count, 1):.4f}")

    return _calculate_metrics(
        targets=targets,
        probabilities=probabilities,
        average_loss=running_loss / max(sample_count, 1),
        threshold=threshold,
    )


def _metric_value(metrics: EpochMetrics, metric_name: str) -> float:
    if not hasattr(metrics, metric_name):
        raise ValueError(
            f"Unknown best metric '{metric_name}'. "
            "Choose one of: loss, accuracy, precision, recall, f1, auc"
        )
    value = float(getattr(metrics, metric_name))
    if math.isnan(value):
        return -math.inf
    return -value if metric_name == "loss" else value


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_score: float,
    train_metrics: EpochMetrics,
    val_metrics: EpochMetrics,
    extra_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score": best_score,
            "train_metrics": train_metrics.as_dict(),
            "val_metrics": val_metrics.as_dict(),
            "config": extra_config,
            "class_mapping": {0: "helmet", 1: "no_helmet"},
        },
        path,
    )


def _append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _format_metrics(metrics: EpochMetrics) -> str:
    auc_text = "nan" if math.isnan(metrics.auc) else f"{metrics.auc:.4f}"
    return (
        f"loss={metrics.loss:.4f} | acc={metrics.accuracy:.4f} | "
        f"precision={metrics.precision:.4f} | recall={metrics.recall:.4f} | "
        f"f1={metrics.f1:.4f} | auc={auc_text}"
    )


def fit(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epochs: int,
    output_directory: str | Path,
    use_amp: bool = True,
    max_grad_norm: float | None = 5.0,
    threshold: float = 0.5,
    best_metric: str = "auc",
    early_stopping_patience: int | None = 20,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    history_path = output_directory / "history.csv"
    best_checkpoint_path = output_directory / "best.pt"
    last_checkpoint_path = output_directory / "last.pt"

    # Start a new history file for a new run.
    if history_path.exists():
        history_path.unlink()

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    run_start = time.perf_counter()
    config = dict(extra_config or {})

    print("\nPositive class for metrics: 1 = no_helmet (каски нет)")
    print(f"Checkpoint selection metric: val_{best_metric}")
    print(f"Automatic mixed precision: {'enabled' if amp_enabled else 'disabled'}\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=amp_enabled,
            max_grad_norm=max_grad_norm,
            threshold=threshold,
            epoch=epoch,
            total_epochs=epochs,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=amp_enabled,
            threshold=threshold,
            epoch=epoch,
            total_epochs=epochs,
        )

        scheduler.step()

        score = _metric_value(val_metrics, best_metric)
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_seconds = time.perf_counter() - epoch_start

        history_row: dict[str, Any] = {
            "epoch": epoch,
            "lr": current_lr,
            "seconds": epoch_seconds,
            **train_metrics.as_dict("train_"),
            **val_metrics.as_dict("val_"),
        }
        _append_history(history_path, history_row)

        checkpoint_config = {
            **config,
            "threshold": threshold,
            "best_metric": best_metric,
        }

        _save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_score,
            train_metrics,
            val_metrics,
            checkpoint_config,
        )

        if improved:
            _save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                train_metrics,
                val_metrics,
                checkpoint_config,
            )

        best_marker = "  <-- BEST" if improved else ""
        print(f"Epoch {epoch:03d}/{epochs:03d} | lr={current_lr:.3e} | {epoch_seconds:.1f}s{best_marker}")
        print(f"  train: {_format_metrics(train_metrics)}")
        print(f"  val:   {_format_metrics(val_metrics)}")
        print(
            "  val confusion matrix (positive=1 no_helmet): "
            f"TN={val_metrics.tn} FP={val_metrics.fp} "
            f"FN={val_metrics.fn} TP={val_metrics.tp}"
        )
        print(
            f"  best epoch={best_epoch:03d} | "
            f"best val_{best_metric}={best_score:.4f}\n"
        )

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping: no val_{best_metric} improvement for "
                f"{early_stopping_patience} epochs."
            )
            break

    total_seconds = time.perf_counter() - run_start
    print(f"Training finished in {total_seconds / 60.0:.1f} minutes")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    print(f"History CSV: {history_path}")

    return {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "history": str(history_path),
    }
