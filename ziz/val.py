"""
Validation HelmetMicroNeXt.

Использование:
    python3 val.py helmet_finetuned.pt

или:
    python3 val.py ./models/helmet_finetuned.pt

На вход требуется только файл весов.

Классы:
    0 = helmet
    1 = no_helmet, positive class
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
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

from model import HelmetMicroNeXt
from preprocessing import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    HelmetFolderDataset,
    HelmetPreprocessor,
)


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

DATASET_ROOT = Path("./ziz-crops-202607-2")

MODELS_DIRECTORY = Path("./models")

IMAGE_SIZE = 64
BATCH_SIZE = 512
NUM_WORKERS = 4

TARGET_RECALL = 0.95
FIXED_THRESHOLD = 0.50

PIN_MEMORY = True


# =============================================================================
# ЗАГРУЗКА ВЕСОВ
# =============================================================================

def resolve_weights_path(argument: str) -> Path:
    direct_path = Path(argument)

    if direct_path.exists():
        return direct_path

    models_path = MODELS_DIRECTORY / argument

    if models_path.exists():
        return models_path

    raise FileNotFoundError(
        "Файл весов не найден.\n"
        f"Проверен путь: {direct_path.resolve()}\n"
        f"Проверен путь: {models_path.resolve()}"
    )


def load_torch_file(
    path: Path,
    map_location: str | torch.device,
) -> Any:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def looks_like_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False

    if not value:
        return False

    return all(
        isinstance(key, str)
        for key in value
    ) and any(
        torch.is_tensor(item)
        for item in value.values()
    )


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if looks_like_state_dict(checkpoint):
        return dict(checkpoint)

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Unsupported weights object: "
            f"{type(checkpoint)}"
        )

    for key in (
        "model_state_dict",
        "state_dict",
        "ema_state_dict",
        "weights",
        "model",
    ):
        value = checkpoint.get(key)

        if looks_like_state_dict(value):
            print(
                f"State dict source: "
                f"checkpoint[{key!r}]"
            )
            return dict(value)

    raise KeyError(
        "State dict not found. "
        f"Checkpoint keys: {list(checkpoint.keys())}"
    )


def strip_common_prefixes(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "network.",
        "net.",
        "ema.",
        "ema_model.",
    )

    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
                break

    return key


def load_model_weights(
    model: torch.nn.Module,
    weights_path: Path,
) -> None:
    checkpoint = load_torch_file(
        weights_path,
        map_location="cpu",
    )

    source_state = extract_state_dict(
        checkpoint
    )

    target_state = model.state_dict()

    normalized_source = {
        strip_common_prefixes(key): value
        for key, value in source_state.items()
    }

    mapped_state: dict[str, torch.Tensor] = {}

    for target_key, target_value in target_state.items():
        source_value = normalized_source.get(
            target_key
        )

        if source_value is None:
            continue

        if tuple(source_value.shape) != tuple(target_value.shape):
            raise RuntimeError(
                f"Shape mismatch for {target_key}: "
                f"weights={tuple(source_value.shape)}, "
                f"model={tuple(target_value.shape)}"
            )

        mapped_state[target_key] = source_value

    missing_keys = [
        key
        for key in target_state
        if key not in mapped_state
        and not key.endswith(
            "num_batches_tracked"
        )
    ]

    if missing_keys:
        raise RuntimeError(
            "Weights are not compatible with model.py.\n"
            "Missing keys:\n"
            + "\n".join(
                f"  {key}"
                for key in missing_keys[:30]
            )
        )

    result = model.load_state_dict(
        mapped_state,
        strict=False,
    )

    unexpected_missing = [
        key
        for key in result.missing_keys
        if not key.endswith(
            "num_batches_tracked"
        )
    ]

    if unexpected_missing:
        raise RuntimeError(
            f"Missing model keys: "
            f"{unexpected_missing}"
        )

    print(
        f"Loaded tensors: "
        f"{len(mapped_state)}/{len(target_state)}"
    )


# =============================================================================
# МЕТРИКИ
# =============================================================================

def calculate_fixed_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (
        y_prob >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    fpr = fp / max(fp + tn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
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
        "specificity": float(specificity),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def calculate_operating_point(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float,
) -> dict[str, float | int]:
    """
    Выбирает точку с минимальным FPR при recall >= target_recall.

    При одинаковом FPR:
        1. выбирается больший recall;
        2. затем более высокий threshold.
    """

    fpr_values, recall_values, thresholds = roc_curve(
        y_true,
        y_prob,
        pos_label=1,
        drop_intermediate=False,
    )

    valid_indices = np.flatnonzero(
        recall_values >= target_recall
    )

    if valid_indices.size == 0:
        raise RuntimeError(
            f"Recall {target_recall:.4f} unreachable"
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

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    if precision + recall > 0:
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "specificity": float(1.0 - fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =============================================================================
# VALIDATION
# =============================================================================

@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    targets: list[int] = []
    probabilities: list[float] = []

    for images, labels, _paths in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images
        ).reshape(-1)

        batch_probabilities = torch.sigmoid(
            logits
        )

        probabilities.extend(
            batch_probabilities
            .cpu()
            .tolist()
        )

        targets.extend(
            labels
            .to(torch.int64)
            .reshape(-1)
            .tolist()
        )

    return (
        np.asarray(targets, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )


# =============================================================================
# MAIN
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate HelmetMicroNeXt weights"
        )
    )

    parser.add_argument(
        "weights",
        type=str,
        help=(
            "Weights path or filename inside ./models"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    weights_path = resolve_weights_path(
        args.weights
    )

    validation_directory = (
        DATASET_ROOT
        / "val"
    )

    if not validation_directory.exists():
        raise FileNotFoundError(
            f"Validation directory not found: "
            f"{validation_directory.resolve()}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = HelmetMicroNeXt()

    load_model_weights(
        model=model,
        weights_path=weights_path,
    )

    model = model.to(device)
    model.eval()

    preprocessor = HelmetPreprocessor(
        training=False,
        output_size=IMAGE_SIZE,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
    )

    dataset = HelmetFolderDataset(
        split_directory=validation_directory,
        preprocessor=preprocessor,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=None,
        num_workers=NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and device.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        drop_last=False,
    )

    y_true, y_prob = collect_predictions(
        model=model,
        loader=loader,
        device=device,
    )

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

    fixed = calculate_fixed_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=FIXED_THRESHOLD,
    )

    operating = calculate_operating_point(
        y_true=y_true,
        y_prob=y_prob,
        target_recall=TARGET_RECALL,
    )

    class_counts = dataset.class_counts()

    print()
    print("=" * 78)
    print("CLASS 0 = helmet")
    print("CLASS 1 = no_helmet, positive class")
    print(f"Weights: {weights_path.resolve()}")
    print(f"Device: {device}")
    print(
        f"Validation images: {len(dataset)} "
        f"(helmet={class_counts[0]}, "
        f"no_helmet={class_counts[1]})"
    )
    print("-" * 78)

    print(f"ROC-AUC: {auc:.6f}")
    print(f"PR-AUC:  {pr_auc:.6f}")

    print("-" * 78)
    print(
        f"Fixed threshold = "
        f"{FIXED_THRESHOLD:.6f}"
    )

    print(
        f"accuracy={fixed['accuracy']:.6f} | "
        f"precision={fixed['precision']:.6f} | "
        f"recall={fixed['recall']:.6f} | "
        f"f1={fixed['f1']:.6f}"
    )

    print(
        f"specificity={fixed['specificity']:.6f} | "
        f"fpr={fixed['fpr']:.6f}"
    )

    print(
        f"TN={fixed['tn']} "
        f"FP={fixed['fp']} "
        f"FN={fixed['fn']} "
        f"TP={fixed['tp']}"
    )

    print("-" * 78)
    print(
        "Operating point: minimum FPR "
        f"at recall >= {TARGET_RECALL:.2f}"
    )

    print(
        f"threshold={operating['threshold']:.9f}"
    )

    print(
        f"recall={operating['recall']:.6f} | "
        f"fpr={operating['fpr']:.6f} | "
        f"precision={operating['precision']:.6f} | "
        f"f1={operating['f1']:.6f}"
    )

    print(
        f"specificity="
        f"{operating['specificity']:.6f}"
    )

    print(
        f"TN={operating['tn']} "
        f"FP={operating['fp']} "
        f"FN={operating['fn']} "
        f"TP={operating['tp']}"
    )

    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
