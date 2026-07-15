"""Dependency-light binary metrics for validation and checkpoint selection."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def roc_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    positive = targets == 1
    negative = targets == 0
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    if positive_count == 0 or negative_count == 0:
        return math.nan

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[positive].sum())
    return (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    positive_count = int((targets == 1).sum())
    if positive_count == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    ordered_targets = targets[order]
    true_positives = np.cumsum(ordered_targets == 1)
    positions = np.arange(1, len(targets) + 1)
    precisions = true_positives / positions
    return float(precisions[ordered_targets == 1].sum() / positive_count)


def fpr_at_recall(
    scores: np.ndarray, targets: np.ndarray, target_recall: float
) -> tuple[float, float, float]:
    positive_count = int((targets == 1).sum())
    negative_count = int((targets == 0).sum())
    if positive_count == 0 or negative_count == 0:
        return math.nan, math.nan, math.nan

    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_targets = targets[order]
    cumulative_tp = np.cumsum(ordered_targets == 1)
    cumulative_fp = np.cumsum(ordered_targets == 0)
    group_ends = np.flatnonzero(
        np.r_[ordered_scores[:-1] != ordered_scores[1:], True]
    )
    recalls = cumulative_tp[group_ends] / positive_count
    false_positive_rates = cumulative_fp[group_ends] / negative_count
    eligible = np.flatnonzero(recalls >= target_recall)
    if len(eligible) == 0:
        return math.nan, math.nan, math.nan
    minimum_fpr = float(false_positive_rates[eligible].min())
    candidates = eligible[
        np.isclose(
            false_positive_rates[eligible],
            minimum_fpr,
            atol=1e-12,
            rtol=0.0,
        )
    ]
    maximum_recall = float(recalls[candidates].max())
    candidates = candidates[
        np.isclose(
            recalls[candidates],
            maximum_recall,
            atol=1e-12,
            rtol=0.0,
        )
    ]
    # If several candidates remain, prefer the highest threshold.
    candidate_end_indices = group_ends[candidates]
    end_index = int(
        candidate_end_indices[
            np.argmax(ordered_scores[candidate_end_indices])
        ]
    )
    threshold = float(ordered_scores[end_index])
    recall = float(cumulative_tp[end_index] / positive_count)
    fpr = float(cumulative_fp[end_index] / negative_count)
    return fpr, threshold, recall


def binary_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = scores >= threshold
    positive = targets == 1
    negative = targets == 0
    tp = int(np.logical_and(predictions, positive).sum())
    fn = int(np.logical_and(~predictions, positive).sum())
    fp = int(np.logical_and(predictions, negative).sum())
    tn = int(np.logical_and(~predictions, negative).sum())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    fpr95, threshold95, achieved_recall = fpr_at_recall(scores, targets, 0.95)
    return {
        "accuracy": _safe_divide(tp + tn, tp + tn + fp + fn),
        "precision_no_helmet": precision,
        "recall_no_helmet": recall,
        "f1_no_helmet": f1,
        "fpr_helmet": _safe_divide(fp, fp + tn),
        "roc_auc": roc_auc(scores, targets),
        "pr_auc": average_precision(scores, targets),
        "fpr_at_recall_95": fpr95,
        "threshold_at_recall_95": threshold95,
        "achieved_recall_at_95": achieved_recall,
    }


def validation_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    losses: np.ndarray,
    dataset_ids: np.ndarray,
    dataset_names: Sequence[str],
) -> dict[str, Any]:
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    result: dict[str, Any] = {
        "overall_loss": float(losses.mean()),
        "overall": binary_metrics(scores, targets),
        "datasets": {},
        "cells": {},
    }

    cell_losses: list[float] = []
    cell_accuracies: list[float] = []
    dataset_fpr_at_recall_95: list[float] = []
    for dataset_id, dataset_name in enumerate(dataset_names):
        dataset_mask = dataset_ids == dataset_id
        dataset_result = binary_metrics(scores[dataset_mask], targets[dataset_mask])
        dataset_result["loss"] = float(losses[dataset_mask].mean())
        dataset_result["count"] = int(dataset_mask.sum())
        result["datasets"][dataset_name] = dataset_result
        dataset_fpr_at_recall_95.append(dataset_result["fpr_at_recall_95"])

        for label, class_name in ((0, "helmet"), (1, "no_helmet")):
            cell_mask = np.logical_and(dataset_mask, targets == label)
            if not cell_mask.any():
                raise ValueError(f"Empty validation cell: {dataset_name}/{class_name}")
            cell_loss = float(losses[cell_mask].mean())
            cell_accuracy = float(((scores[cell_mask] >= 0.5) == label).mean())
            result["cells"][f"{dataset_name}/{class_name}"] = {
                "count": int(cell_mask.sum()),
                "loss": cell_loss,
                "accuracy": cell_accuracy,
            }
            cell_losses.append(cell_loss)
            cell_accuracies.append(cell_accuracy)

    result["macro_cell_loss"] = float(np.mean(cell_losses))
    result["macro_cell_accuracy"] = float(np.mean(cell_accuracies))
    result["macro_dataset_fpr_at_recall_95"] = float(
        np.mean(dataset_fpr_at_recall_95)
    )
    result["worst_dataset_fpr_at_recall_95"] = float(
        np.max(dataset_fpr_at_recall_95)
    )
    return result
