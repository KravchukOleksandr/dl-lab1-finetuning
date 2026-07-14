"""
CLASS 0 = helmet     (каска есть)
CLASS 1 = no_helmet  (каски нет, positive class)

Train balancing is configured in main.py.
Validation is never resampled.

AMP and batch-level logging are intentionally disabled.
"""

from __future__ import annotations

import copy
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
    roc_curve,
)
from torch.utils.data import DataLoader


TRAIN_MODULE_VERSION = "ema-r95-fp32-v5"


class ModelEMA:
    """
    Exponential Moving Average модели.

    EMA применяется:
        - после каждого optimizer.step();
        - для validation;
        - для выбора лучшей эпохи;
        - для model_state_dict в checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.997,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(
                "EMA_DECAY must be in (0, 1)"
            )

        self.decay = float(decay)

        self.module = copy.deepcopy(
            model
        ).eval()

        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
    ) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()

        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()

            if torch.is_floating_point(
                ema_value
            ):
                ema_value.mul_(
                    self.decay
                ).add_(
                    model_value,
                    alpha=1.0 - self.decay,
                )
            else:
                # Например BatchNorm.num_batches_tracked.
                ema_value.copy_(model_value)


def operating_point(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float,
) -> dict[str, float | int]:
    """
    Находит минимальный FPR среди всех порогов,
    обеспечивающих recall >= target_recall.

    При равном FPR:
        1. выбирается больший recall;
        2. затем более высокий threshold.
    """

    fpr_values, recall_values, thresholds = (
        roc_curve(
            y_true,
            y_prob,
            pos_label=1,
            drop_intermediate=False,
        )
    )

    valid_indices = np.flatnonzero(
        recall_values >= target_recall
    )

    if valid_indices.size == 0:
        raise RuntimeError(
            f"Target recall {target_recall:.4f} "
            "is unreachable"
        )

    minimum_fpr = np.min(
        fpr_values[valid_indices]
    )

    candidates = valid_indices[
        np.isclose(
            fpr_values[valid_indices],
            minimum_fpr,
            atol=1e-12,
            rtol=0.0,
        )
    ]

    maximum_recall = np.max(
        recall_values[candidates]
    )

    candidates = candidates[
        np.isclose(
            recall_values[candidates],
            maximum_recall,
            atol=1e-12,
            rtol=0.0,
        )
    ]

    best_index = int(
        candidates[
            np.argmax(
                thresholds[candidates]
            )
        ]
    )

    threshold = float(
        thresholds[best_index]
    )

    y_pred = (
        y_prob >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)

    return {
        "fpr": float(fpr),
        "recall": float(recall),
        "precision": float(precision),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def calculate_metrics(
    targets: list[int],
    probabilities: list[float],
    loss: float,
    threshold: float,
    target_recall: float,
) -> dict[str, float | int]:
    """
    Обычные метрики считаются при фиксированном threshold.

    Дополнительно рассчитывается рабочая точка:
        минимальный FPR при recall >= target_recall.
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

    try:
        auc = float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        )

        pr_auc = float(
            average_precision_score(
                y_true,
                y_prob,
            )
        )
    except ValueError:
        auc = float("nan")
        pr_auc = float("nan")

    fpr = fp / max(fp + tn, 1)

    target_tag = int(
        round(target_recall * 100)
    )

    point = operating_point(
        y_true=y_true,
        y_prob=y_prob,
        target_recall=target_recall,
    )

    metrics: dict[str, float | int] = {
        "loss": float(loss),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        # Для класса 1 = no_helmet.
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

        "auc": auc,
        "pr_auc": pr_auc,

        # Для класса 0 = helmet.
        "specificity": float(
            1.0 - fpr
        ),

        "fpr": float(fpr),

        # Confusion matrix при фиксированном threshold.
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    for key, value in point.items():
        metrics[
            f"{key}_at_recall_{target_tag}"
        ] = value

    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    target_recall: float,
    optimizer: torch.optim.Optimizer | None = None,
    ema: ModelEMA | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float | int]:
    """
    optimizer передан:
        train

    optimizer=None:
        validation

    Никакого вывода внутри batch-loop нет.
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
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):
            logits = model(
                images
            ).reshape(-1)

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

                if ema is not None:
                    ema.update(model)

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
        target_recall=target_recall,
    )


def is_better(
    current: dict[str, float | int],
    best: dict[str, float | int] | None,
    metric: str,
    target_recall: float,
) -> bool:
    """
    Основной порядок сравнения для FPR@R95:

        1. ниже FPR;
        2. выше фактический recall;
        3. выше ROC-AUC;
        4. выше threshold.
    """

    if best is None:
        return True

    current_value = float(
        current[metric]
    )

    best_value = float(
        best[metric]
    )

    if math.isnan(current_value):
        return False

    target_tag = int(
        round(target_recall * 100)
    )

    target_metric = (
        f"fpr_at_recall_{target_tag}"
    )

    if metric == target_metric:
        epsilon = 1e-12

        current_rank = (
            current_value,

            -float(
                current[
                    f"recall_at_recall_{target_tag}"
                ]
            ),

            -float(current["auc"]),

            -float(
                current[
                    f"threshold_at_recall_{target_tag}"
                ]
            ),
        )

        best_rank = (
            best_value,

            -float(
                best[
                    f"recall_at_recall_{target_tag}"
                ]
            ),

            -float(best["auc"]),

            -float(
                best[
                    f"threshold_at_recall_{target_tag}"
                ]
            ),
        )

        for current_item, best_item in zip(
            current_rank,
            best_rank,
        ):
            if (
                current_item
                < best_item - epsilon
            ):
                return True

            if (
                current_item
                > best_item + epsilon
            ):
                return False

        return False

    if metric in {
        "loss",
        "fpr",
    }:
        return current_value < best_value

    return current_value > best_value


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


def format_target_metrics(
    metrics: dict[str, float | int],
    target_recall: float,
) -> str:
    tag = int(
        round(target_recall * 100)
    )

    return (
        f"FPR@R{tag}="
        f"{metrics[f'fpr_at_recall_{tag}']:.4f} | "

        f"recall="
        f"{metrics[f'recall_at_recall_{tag}']:.4f} | "

        f"precision="
        f"{metrics[f'precision_at_recall_{tag}']:.4f} | "

        f"threshold="
        f"{metrics[f'threshold_at_recall_{tag}']:.6f} | "

        f"TN={metrics[f'tn_at_recall_{tag}']} "
        f"FP={metrics[f'fp_at_recall_{tag}']} "
        f"FN={metrics[f'fn_at_recall_{tag}']} "
        f"TP={metrics[f'tp_at_recall_{tag}']}"
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
            fieldnames=list(
                row.keys()
            ),
        )

        if new_file:
            writer.writeheader()

        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_epoch: int,
    best_metric: str,
    best_metrics: dict[str, float | int],
    train_metrics: dict[str, float | int],
    val_metrics: dict[str, float | int],
    config: dict[str, Any],
) -> None:
    """
    model_state_dict содержит EMA-веса,
    предназначенные для inference.

    raw_model_state_dict содержит обычные веса,
    обновляемые optimizer.
    """

    checkpoint = {
        "epoch": epoch,
        "best_epoch": best_epoch,

        "best_metric": best_metric,
        "best_value": float(
            best_metrics[best_metric]
        ),

        # Использовать для inference.
        "model_state_dict": (
            ema.module.state_dict()
        ),

        # Сырые веса optimizer-модели.
        "raw_model_state_dict": (
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
        "best_val_metrics": best_metrics,

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
    ema_decay: float,
    target_recall: float,
    max_grad_norm: float | None = 5.0,
    threshold: float = 0.5,
    best_metric: str = "fpr_at_recall_95",
    early_stopping_patience: int | None = None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_tag = int(
        round(target_recall * 100)
    )

    expected_metric = (
        f"fpr_at_recall_{target_tag}"
    )

    if best_metric != expected_metric:
        raise ValueError(
            f"For target_recall={target_recall}, "
            f"use best_metric={expected_metric!r}"
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

    best_path = (
        output_directory
        / "best.pt"
    )

    last_path = (
        output_directory
        / "last.pt"
    )

    if history_path.exists():
        history_path.unlink()

    ema = ModelEMA(
        model,
        decay=ema_decay,
    )

    best_metrics: (
        dict[str, float | int] | None
    ) = None

    best_epoch = 0
    stale_epochs = 0

    started = time.perf_counter()

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
        "Selection: minimum val FPR "
        f"with recall >= {target_recall:.2f}"
    )

    print(
        f"EMA={ema_decay:.4f} | "
        "AMP disabled | "
        "batch logging disabled"
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
            target_recall=target_recall,
            optimizer=optimizer,
            ema=ema,
            max_grad_norm=max_grad_norm,
        )

        # Validation выполняется на EMA-модели.
        val_metrics = run_epoch(
            model=ema.module,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            target_recall=target_recall,
            optimizer=None,
            ema=None,
            max_grad_norm=None,
        )

        improved = is_better(
            current=val_metrics,
            best=best_metrics,
            metric=best_metric,
            target_recall=target_recall,
        )

        if improved:
            best_metrics = dict(
                val_metrics
            )

            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        seconds = (
            time.perf_counter()
            - epoch_started
        )

        history_row: dict[str, Any] = {
            "epoch": epoch,
            "lr": learning_rate,
            "seconds": seconds,
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

        assert best_metrics is not None

        checkpoint_config = {
            **config,
            "target_recall": target_recall,
            "threshold": threshold,
            "best_metric": best_metric,
            "ema_decay": ema_decay,
        }

        save_checkpoint(
            path=last_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_metric=best_metric,
            best_metrics=best_metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            config=checkpoint_config,
        )

        if improved:
            save_checkpoint(
                path=best_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metric=best_metric,
                best_metrics=best_metrics,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                config=checkpoint_config,
            )

        marker = (
            " <-- BEST"
            if improved
            else ""
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"lr={learning_rate:.3e} | "
            f"time={seconds:.1f}s"
            f"{marker}"
        )

        print(
            f"  train:   "
            f"{format_metrics(train_metrics)}"
        )

        print(
            f"  val EMA: "
            f"{format_metrics(val_metrics)}"
        )

        print(
            "  val @ fixed threshold: "
            f"TN={val_metrics['tn']} "
            f"FP={val_metrics['fp']} "
            f"FN={val_metrics['fn']} "
            f"TP={val_metrics['tp']}"
        )

        print(
            f"  val EMA: "
            f"{format_target_metrics(val_metrics, target_recall)}"
        )

        print(
            f"  best: epoch={best_epoch:03d} | "
            f"{best_metric}="
            f"{float(best_metrics[best_metric]):.4f}"
        )

        print()

        if scheduler is not None:
            scheduler.step()

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and stale_epochs
            >= early_stopping_patience
        ):
            print(
                "Early stopping after "
                f"{stale_epochs} epochs "
                "without improvement"
            )
            break

    total_minutes = (
        time.perf_counter()
        - started
    ) / 60.0

    assert best_metrics is not None

    best_threshold = float(
        best_metrics[
            f"threshold_at_recall_{target_tag}"
        ]
    )

    print(
        f"Training finished in "
        f"{total_minutes:.1f} min"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best {best_metric}: "
        f"{float(best_metrics[best_metric]):.4f}"
    )

    print(
        f"Inference threshold: "
        f"{best_threshold:.6f}"
    )

    print(
        f"Best checkpoint: "
        f"{best_path}"
    )

    print(
        f"Last checkpoint: "
        f"{last_path}"
    )

    print(
        f"History: "
        f"{history_path}"
    )

    return {
        "best_epoch": best_epoch,
        "best_value": float(
            best_metrics[best_metric]
        ),
        "best_threshold": best_threshold,
        "best_checkpoint": str(
            best_path
        ),
        "last_checkpoint": str(
            last_path
        ),
        "history": str(
            history_path
        ),
    }