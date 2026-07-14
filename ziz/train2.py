"""
Training utilities for HelmetMicroNeXt.

CLASS MAPPING:
    0 = helmet     = каска есть
    1 = no_helmet  = каски нет

Positive class for precision / recall / F1 / ROC-AUC:
    1 = no_helmet
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


# Эта строка должна появиться в начале лога через main.py.
TRAIN_MODULE_VERSION = "fp32-no-amp-v3"


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


def calculate_metrics(
    targets: list[int],
    probabilities: list[float],
    average_loss: float,
    threshold: float = 0.5,
) -> EpochMetrics:
    """
    Класс 1 = no_helmet считается положительным.
    """
    y_true = np.asarray(targets, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    accuracy = float(
        accuracy_score(y_true, y_pred)
    )

    precision = float(
        precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )
    )

    try:
        auc = float(
            roc_auc_score(y_true, y_prob)
        )
    except ValueError:
        # Может произойти, если в выборке присутствует только один класс.
        auc = float("nan")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

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
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).float()

        optimizer.zero_grad(set_to_none=True)

        # Обычный FP32 forward.
        # Здесь нет torch.amp, GradScaler и autocast.
        logits = model(images)

        # Защита на случай выхода [B, 1].
        logits = logits.reshape(-1)
        labels = labels.reshape(-1)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        if (
            max_grad_norm is not None
            and max_grad_norm > 0
        ):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )

        optimizer.step()

        batch_size = labels.shape[0]

        running_loss += (
            float(loss.detach().item())
            * batch_size
        )

        sample_count += batch_size

        batch_probabilities = (
            torch.sigmoid(logits.detach())
            .cpu()
            .numpy()
        )

        probabilities.extend(
            batch_probabilities.tolist()
        )

        targets.extend(
            labels.detach()
            .cpu()
            .to(torch.int64)
            .numpy()
            .tolist()
        )

        current_loss = (
            running_loss
            / max(sample_count, 1)
        )

        progress.set_postfix(
            loss=f"{current_loss:.4f}"
        )

    average_loss = (
        running_loss
        / max(sample_count, 1)
    )

    return calculate_metrics(
        targets=targets,
        probabilities=probabilities,
        average_loss=average_loss,
        threshold=threshold,
    )


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
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
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).float()

        logits = model(images)

        logits = logits.reshape(-1)
        labels = labels.reshape(-1)

        loss = criterion(
            logits,
            labels,
        )

        batch_size = labels.shape[0]

        running_loss += (
            float(loss.item())
            * batch_size
        )

        sample_count += batch_size

        batch_probabilities = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

        probabilities.extend(
            batch_probabilities.tolist()
        )

        targets.extend(
            labels.cpu()
            .to(torch.int64)
            .numpy()
            .tolist()
        )

        current_loss = (
            running_loss
            / max(sample_count, 1)
        )

        progress.set_postfix(
            loss=f"{current_loss:.4f}"
        )

    average_loss = (
        running_loss
        / max(sample_count, 1)
    )

    return calculate_metrics(
        targets=targets,
        probabilities=probabilities,
        average_loss=average_loss,
        threshold=threshold,
    )


def get_metric_value(
    metrics: EpochMetrics,
    metric_name: str,
) -> float:
    allowed_metrics = {
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
    }

    if metric_name not in allowed_metrics:
        raise ValueError(
            f"Unknown metric: {metric_name}. "
            f"Allowed metrics: {sorted(allowed_metrics)}"
        )

    return float(
        getattr(metrics, metric_name)
    )


def metric_improved(
    current_value: float,
    best_value: float,
    metric_name: str,
) -> bool:
    if math.isnan(current_value):
        return False

    if metric_name == "loss":
        return current_value < best_value

    return current_value > best_value


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_epoch: int,
    best_metric_name: str,
    best_metric_value: float,
    train_metrics: EpochMetrics,
    val_metrics: EpochMetrics,
    extra_config: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics.as_dict(),
        "val_metrics": val_metrics.as_dict(),
        "config": extra_config,
        "class_mapping": {
            0: "helmet",
            1: "no_helmet",
        },
        "train_module_version": TRAIN_MODULE_VERSION,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = (
            scheduler.state_dict()
        )

    torch.save(
        checkpoint,
        path,
    )


def append_history(
    path: Path,
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def format_metrics(
    metrics: EpochMetrics,
) -> str:
    if math.isnan(metrics.auc):
        auc_text = "nan"
    else:
        auc_text = f"{metrics.auc:.4f}"

    return (
        f"loss={metrics.loss:.4f} | "
        f"accuracy={metrics.accuracy:.4f} | "
        f"precision={metrics.precision:.4f} | "
        f"recall={metrics.recall:.4f} | "
        f"f1={metrics.f1:.4f} | "
        f"auc={auc_text}"
    )


def fit(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epochs: int,
    output_directory: str | Path,
    use_amp: bool = False,
    max_grad_norm: float | None = 5.0,
    threshold: float = 0.5,
    best_metric: str = "auc",
    early_stopping_patience: int | None = 20,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Основной цикл обучения.

    Параметр use_amp оставлен для совместимости с main.py,
    но AMP в этой версии намеренно не используется.
    """

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        output_directory
        / "history.csv"
    )

    best_checkpoint_path = (
        output_directory
        / "best.pt"
    )

    last_checkpoint_path = (
        output_directory
        / "last.pt"
    )

    # Новый запуск — новая история.
    if history_path.exists():
        history_path.unlink()

    if best_metric == "loss":
        best_metric_value = math.inf
    else:
        best_metric_value = -math.inf

    best_epoch = 0
    epochs_without_improvement = 0

    run_start = time.perf_counter()

    config = dict(
        extra_config or {}
    )

    config["use_amp_requested"] = bool(use_amp)
    config["use_amp_effective"] = False
    config["train_module_version"] = (
        TRAIN_MODULE_VERSION
    )

    print()
    print("=" * 78)
    print(
        f"train.py version: {TRAIN_MODULE_VERSION}"
    )
    print("CLASS 0 = helmet     = каска есть")
    print("CLASS 1 = no_helmet  = каски нет")
    print(
        "Positive class for precision/recall/F1/AUC: "
        "1 = no_helmet"
    )
    print(
        f"Checkpoint metric: val_{best_metric}"
    )
    print(
        "AMP: disabled intentionally for maximum "
        "PyTorch compatibility"
    )
    print("=" * 78)
    print()

    for epoch in range(
        1,
        epochs + 1,
    ):
        epoch_start = time.perf_counter()

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
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
            threshold=threshold,
            epoch=epoch,
            total_epochs=epochs,
        )

        current_metric_value = get_metric_value(
            val_metrics,
            best_metric,
        )

        improved = metric_improved(
            current_value=current_metric_value,
            best_value=best_metric_value,
            metric_name=best_metric,
        )

        if improved:
            best_metric_value = (
                current_metric_value
            )

            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Scheduler обновляем после завершённой эпохи.
        if scheduler is not None:
            scheduler.step()

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        history_row: dict[str, Any] = {
            "epoch": epoch,
            "lr": current_lr,
            "seconds": epoch_seconds,
            **train_metrics.as_dict("train_"),
            **val_metrics.as_dict("val_"),
        }

        append_history(
            history_path,
            history_row,
        )

        checkpoint_config = {
            **config,
            "threshold": threshold,
            "best_metric": best_metric,
        }

        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_metric_name=best_metric,
            best_metric_value=best_metric_value,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            extra_config=checkpoint_config,
        )

        if improved:
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metric_name=best_metric,
                best_metric_value=best_metric_value,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                extra_config=checkpoint_config,
            )

        marker = (
            "  <-- BEST"
            if improved
            else ""
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"lr={current_lr:.3e} | "
            f"time={epoch_seconds:.1f}s"
            f"{marker}"
        )

        print(
            f"  train: {format_metrics(train_metrics)}"
        )

        print(
            f"  val:   {format_metrics(val_metrics)}"
        )

        print(
            "  val confusion matrix "
            "(positive class = 1 no_helmet): "
            f"TN={val_metrics.tn} | "
            f"FP={val_metrics.fp} | "
            f"FN={val_metrics.fn} | "
            f"TP={val_metrics.tp}"
        )

        print(
            f"  best epoch={best_epoch:03d} | "
            f"best val_{best_metric}="
            f"{best_metric_value:.4f}"
        )

        print()

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and epochs_without_improvement
            >= early_stopping_patience
        ):
            print(
                "Early stopping: "
                f"val_{best_metric} did not improve "
                f"for {early_stopping_patience} epochs."
            )
            break

    total_seconds = (
        time.perf_counter()
        - run_start
    )

    print()
    print("=" * 78)
    print(
        f"Training finished in "
        f"{total_seconds / 60.0:.1f} minutes"
    )
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Best val_{best_metric}: "
        f"{best_metric_value:.4f}"
    )
    print(
        f"Best checkpoint: "
        f"{best_checkpoint_path}"
    )
    print(
        f"Last checkpoint: "
        f"{last_checkpoint_path}"
    )
    print(
        f"History CSV: "
        f"{history_path}"
    )
    print("=" * 78)

    return {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_score": best_metric_value,
        "best_checkpoint": str(
            best_checkpoint_path
        ),
        "last_checkpoint": str(
            last_checkpoint_path
        ),
        "history": str(
            history_path
        ),
    }