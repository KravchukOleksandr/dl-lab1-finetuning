"""
Классы:
    0 = helmet     — каска есть
    1 = no_helmet  — каски нет, положительный класс

Precision, recall, F1 и PR-AUC относятся к классу 1 = no_helmet.

Specificity и FPR характеризуют ошибки на классе 0 = helmet.

Validation не балансируется:
каждый validation-пример проходит ровно один раз.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


TRAIN_MODULE_VERSION = "fp32-epoch-log-v4"


def calculate_metrics(
    targets: list[int],
    probabilities: list[float],
    loss: float,
    threshold: float,
) -> dict[str, float | int]:
    """
    Положительный класс:
        1 = no_helmet
    """

    y_true = np.asarray(
        targets,
        dtype=np.int64,
    )

    y_prob = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    y_pred = (
        y_prob >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    helmet_count = tn + fp

    if helmet_count > 0:
        specificity = tn / helmet_count
        fpr = fp / helmet_count
    else:
        specificity = 0.0
        fpr = 0.0

    try:
        auc = float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        )
    except ValueError:
        auc = float("nan")

    try:
        # Average Precision — стандартная оценка PR-кривой.
        pr_auc = float(
            average_precision_score(
                y_true,
                y_prob,
            )
        )
    except ValueError:
        pr_auc = float("nan")

    return {
        "loss": float(loss),

        # Общая метрика по двум классам.
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        # Метрики положительного класса 1 = no_helmet.
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),

        # Метрики ранжирования, не зависящие от threshold.
        "auc": auc,
        "pr_auc": pr_auc,

        # Поведение на классе 0 = helmet.
        "specificity": float(specificity),
        "fpr": float(fpr),

        # Полная confusion matrix.
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float | int]:
    """
    Если optimizer передан — train.
    Если optimizer=None — validation.

    Прогресс по batch не печатается.
    """

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    targets: list[int] = []
    probabilities: list[float] = []

    for images, labels, _paths in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).float().reshape(-1)

        if training:
            optimizer.zero_grad(
                set_to_none=True,
            )

        # AMP и autocast намеренно не используются.
        with torch.set_grad_enabled(training):
            logits = model(images).reshape(-1)

            loss = criterion(
                logits,
                labels,
            )

            if training:
                loss.backward()

                if (
                    max_grad_norm is not None
                    and max_grad_norm > 0
                ):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_grad_norm,
                    )

                optimizer.step()

        batch_size = labels.numel()

        total_loss += (
            float(loss.detach().item())
            * batch_size
        )

        total_samples += batch_size

        batch_probabilities = torch.sigmoid(
            logits.detach()
        )

        probabilities.extend(
            batch_probabilities
            .cpu()
            .tolist()
        )

        targets.extend(
            labels.detach()
            .cpu()
            .to(torch.int64)
            .tolist()
        )

    average_loss = (
        total_loss
        / max(total_samples, 1)
    )

    return calculate_metrics(
        targets=targets,
        probabilities=probabilities,
        loss=average_loss,
        threshold=threshold,
    )


def metric_is_better(
    current: float,
    best: float,
    metric_name: str,
) -> bool:
    if math.isnan(current):
        return False

    # Для loss и FPR меньше — лучше.
    if metric_name in {
        "loss",
        "fpr",
    }:
        return current < best

    return current > best


def format_metrics(
    metrics: dict[str, float | int],
) -> str:
    return (
        f"loss={metrics['loss']:.4f} | "
        f"acc={metrics['accuracy']:.4f} | "
        f"precision={metrics['precision']:.4f} | "
        f"recall={metrics['recall']:.4f} | "
        f"f1={metrics['f1']:.4f} | "
        f"auc={metrics['auc']:.4f} | "
        f"pr_auc={metrics['pr_auc']:.4f} | "
        f"specificity={metrics['specificity']:.4f} | "
        f"fpr={metrics['fpr']:.4f}"
    )


def append_history(
    path: Path,
    row: dict[str, Any],
) -> None:
    new_file = not path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if new_file:
            writer.writeheader()

        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_epoch: int,
    best_metric: str,
    best_value: float,
    train_metrics: dict[str, float | int],
    val_metrics: dict[str, float | int],
    config: dict[str, Any],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_value": best_value,

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "config": config,

        "class_mapping": {
            0: "helmet",
            1: "no_helmet",
        },

        "train_module_version": (
            TRAIN_MODULE_VERSION
        ),
    }

    torch.save(
        checkpoint,
        path,
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
    max_grad_norm: float | None = 5.0,
    threshold: float = 0.5,
    best_metric: str = "auc",
    early_stopping_patience: int | None = 20,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allowed_metrics = {
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "pr_auc",
        "specificity",
        "fpr",
    }

    if best_metric not in allowed_metrics:
        raise ValueError(
            f"Unknown BEST_METRIC={best_metric!r}. "
            f"Allowed: {sorted(allowed_metrics)}"
        )

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

    # Новый запуск создаёт новую историю.
    if history_path.exists():
        history_path.unlink()

    if best_metric in {
        "loss",
        "fpr",
    }:
        best_value = math.inf
    else:
        best_value = -math.inf

    best_epoch = 0
    epochs_without_improvement = 0
    training_started = time.perf_counter()

    config = dict(
        extra_config or {}
    )

    print(
        f"train.py version: "
        f"{TRAIN_MODULE_VERSION}"
    )

    print(
        "CLASS 0 = helmet | "
        "CLASS 1 = no_helmet "
        "(positive class)"
    )

    print(
        f"Best checkpoint metric: "
        f"val_{best_metric}"
    )

    print(
        "AMP: disabled | "
        "batch progress: disabled"
    )

    print()

    for epoch in range(
        1,
        epochs + 1,
    ):
        epoch_started = (
            time.perf_counter()
        )

        learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            optimizer=None,
            max_grad_norm=None,
        )

        current_value = float(
            val_metrics[best_metric]
        )

        improved = metric_is_better(
            current=current_value,
            best=best_value,
            metric_name=best_metric,
        )

        if improved:
            best_value = current_value
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()

        epoch_seconds = (
            time.perf_counter()
            - epoch_started
        )

        history_row: dict[str, Any] = {
            "epoch": epoch,
            "lr": learning_rate,
            "seconds": epoch_seconds,
        }

        history_row.update({
            f"train_{key}": value
            for key, value
            in train_metrics.items()
        })

        history_row.update({
            f"val_{key}": value
            for key, value
            in val_metrics.items()
        })

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
            best_metric=best_metric,
            best_value=best_value,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            config=checkpoint_config,
        )

        if improved:
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metric=best_metric,
                best_value=best_value,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                config=checkpoint_config,
            )

        best_marker = (
            " <-- BEST"
            if improved
            else ""
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"lr={learning_rate:.3e} | "
            f"time={epoch_seconds:.1f}s"
            f"{best_marker}"
        )

        print(
            f"  train: "
            f"{format_metrics(train_metrics)}"
        )

        print(
            f"  val:   "
            f"{format_metrics(val_metrics)}"
        )

        print(
            "  val confusion: "
            f"TN={val_metrics['tn']} "
            f"FP={val_metrics['fp']} "
            f"FN={val_metrics['fn']} "
            f"TP={val_metrics['tp']} | "
            f"best epoch={best_epoch:03d}, "
            f"val_{best_metric}="
            f"{best_value:.4f}"
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
                f"val_{best_metric} "
                "did not improve for "
                f"{early_stopping_patience} "
                "epochs."
            )
            break

    total_minutes = (
        time.perf_counter()
        - training_started
    ) / 60.0

    print(
        f"Training finished in "
        f"{total_minutes:.1f} min"
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
        f"History: "
        f"{history_path}"
    )

    return {
        "best_epoch": best_epoch,
        "best_value": best_value,
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