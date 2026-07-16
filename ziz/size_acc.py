from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from model import HelmetMicroNeXt
from preprocessing import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    HelmetFolderDataset,
    HelmetPreprocessor,
)


# =============================================================================
# CONFIG
# =============================================================================

DATASET_ROOT = Path("./ziz-crops-202607-2")

WEIGHTS_PATH = Path(
    "./models/helmet_finetuned.pt"
)

DECISION_THRESHOLD = 0.358260

IMAGE_SIZE = 64
BATCH_SIZE = 512
NUM_WORKERS = 4

CROP_EXPANSION_FACTOR = 1.25

REQUESTED_BIN_COUNT = 6

MIN_TOTAL_PER_BIN = 150
MIN_HELMET_PER_BIN = 80
MIN_NO_HELMET_PER_BIN = 25

PIN_MEMORY = True

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

OUTPUT_IMAGE_PATH = (
    SCRIPT_DIRECTORY
    / f"{WEIGHTS_PATH.stem}_size_error_distribution.png"
)

OUTPUT_CSV_PATH = (
    SCRIPT_DIRECTORY
    / f"{WEIGHTS_PATH.stem}_size_error_distribution.csv"
)


# =============================================================================
# CHECKPOINT
# =============================================================================

def load_torch_file(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Weights not found: {path.resolve()}"
        )

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


def looks_like_state_dict(
    value: Any,
) -> bool:
    if not isinstance(value, Mapping):
        return False

    if not value:
        return False

    return all(
        isinstance(key, str)
        and torch.is_tensor(tensor)
        for key, tensor in value.items()
    )


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    if looks_like_state_dict(checkpoint):
        return dict(checkpoint)

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}"
        )

    candidate_keys = (
        "model_state_dict",
        "ema_state_dict",
        "state_dict",
        "weights",
        "model",
        "raw_model_state_dict",
    )

    for key in candidate_keys:
        value = checkpoint.get(key)

        if looks_like_state_dict(value):
            return dict(value)

    raise KeyError(
        "State dict not found. "
        f"Checkpoint keys: {list(checkpoint.keys())}"
    )


def normalize_state_key(
    key: str,
) -> str:
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
    checkpoint: Any,
) -> None:
    source_state = extract_state_dict(checkpoint)
    target_state = model.state_dict()

    normalized_source = {
        normalize_state_key(key): tensor
        for key, tensor in source_state.items()
    }

    mapped_state: dict[str, torch.Tensor] = {}

    for target_key, target_tensor in target_state.items():
        source_tensor = normalized_source.get(target_key)

        if source_tensor is None:
            continue

        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise RuntimeError(
                f"Shape mismatch for {target_key}: "
                f"checkpoint={tuple(source_tensor.shape)}, "
                f"model={tuple(target_tensor.shape)}"
            )

        mapped_state[target_key] = source_tensor

    missing_keys = [
        key
        for key in target_state
        if key not in mapped_state
        and not key.endswith("num_batches_tracked")
    ]

    if missing_keys:
        raise RuntimeError(
            "Weights are incompatible with model.py.\n"
            + "\n".join(
                f"Missing: {key}"
                for key in missing_keys[:30]
            )
        )

    model.load_state_dict(
        mapped_state,
        strict=False,
    )


# =============================================================================
# SIZE
# =============================================================================

def estimate_original_min_side(
    image_path: str | Path,
) -> float:
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: {image_path}"
        )

    height, width = image.shape[:2]

    return float(
        min(width, height)
        / CROP_EXPANSION_FACTOR
    )


# =============================================================================
# PREDICTIONS
# =============================================================================

@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model.eval()

    labels: list[int] = []
    probabilities: list[float] = []
    sizes: list[float] = []

    for batch in loader:
        images = batch[0]
        batch_labels = batch[1]
        paths = batch[2]

        images = images.to(
            device,
            non_blocking=True,
        )

        logits = model(images).reshape(-1)

        batch_probabilities = torch.sigmoid(
            logits
        )

        labels.extend(
            batch_labels
            .reshape(-1)
            .to(torch.int64)
            .cpu()
            .tolist()
        )

        probabilities.extend(
            batch_probabilities
            .cpu()
            .tolist()
        )

        sizes.extend(
            estimate_original_min_side(path)
            for path in paths
        )

    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(sizes, dtype=np.float64),
    )


# =============================================================================
# CONFIDENCE INTERVAL
# =============================================================================

def wilson_interval(
    error_count: int,
    sample_count: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if sample_count <= 0:
        return float("nan"), float("nan")

    probability = error_count / sample_count
    z_squared = z * z

    denominator = (
        1.0
        + z_squared / sample_count
    )

    center = (
        probability
        + z_squared / (2.0 * sample_count)
    ) / denominator

    half_width = (
        z
        * np.sqrt(
            probability
            * (1.0 - probability)
            / sample_count
            + z_squared
            / (4.0 * sample_count**2)
        )
        / denominator
    )

    return (
        max(0.0, float(center - half_width)),
        min(1.0, float(center + half_width)),
    )


# =============================================================================
# ADAPTIVE BINS
# =============================================================================

def create_adaptive_bins(
    sizes: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    maximum_bin_count = min(
        REQUESTED_BIN_COUNT,
        len(np.unique(sizes)),
    )

    for requested_count in range(
        maximum_bin_count,
        1,
        -1,
    ):
        edges = np.quantile(
            sizes,
            np.linspace(
                0.0,
                1.0,
                requested_count + 1,
            ),
        )

        edges = np.unique(edges)

        bin_count = len(edges) - 1

        if bin_count < 2:
            continue

        bin_indices = np.searchsorted(
            edges[1:-1],
            sizes,
            side="right",
        )

        valid = True

        for bin_index in range(bin_count):
            mask = bin_indices == bin_index

            total_count = int(
                np.sum(mask)
            )

            helmet_count = int(
                np.sum(
                    mask
                    & (labels == 0)
                )
            )

            no_helmet_count = int(
                np.sum(
                    mask
                    & (labels == 1)
                )
            )

            if total_count < MIN_TOTAL_PER_BIN:
                valid = False
                break

            if helmet_count < MIN_HELMET_PER_BIN:
                valid = False
                break

            if no_helmet_count < MIN_NO_HELMET_PER_BIN:
                valid = False
                break

        if valid:
            return edges, bin_indices

    raise RuntimeError(
        "Cannot create stable bins. "
        "Reduce MIN_TOTAL_PER_BIN, "
        "MIN_HELMET_PER_BIN or "
        "MIN_NO_HELMET_PER_BIN."
    )


# =============================================================================
# STATISTICS
# =============================================================================

def calculate_bin_statistics(
    sizes: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    edges: np.ndarray,
    bin_indices: np.ndarray,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []

    for bin_index in range(len(edges) - 1):
        mask = bin_indices == bin_index

        helmet_mask = (
            mask
            & (labels == 0)
        )

        no_helmet_mask = (
            mask
            & (labels == 1)
        )

        bin_sizes = sizes[mask]

        total_count = int(
            np.sum(mask)
        )

        helmet_count = int(
            np.sum(helmet_mask)
        )

        no_helmet_count = int(
            np.sum(no_helmet_mask)
        )

        overall_error_count = int(
            np.sum(
                predictions[mask]
                != labels[mask]
            )
        )

        false_positive_count = int(
            np.sum(
                predictions[helmet_mask] == 1
            )
        )

        false_negative_count = int(
            np.sum(
                predictions[no_helmet_mask] == 0
            )
        )

        overall_error_rate = (
            overall_error_count
            / total_count
        )

        false_positive_rate = (
            false_positive_count
            / helmet_count
        )

        false_negative_rate = (
            false_negative_count
            / no_helmet_count
        )

        overall_lower, overall_upper = wilson_interval(
            overall_error_count,
            total_count,
        )

        fpr_lower, fpr_upper = wilson_interval(
            false_positive_count,
            helmet_count,
        )

        fnr_lower, fnr_upper = wilson_interval(
            false_negative_count,
            no_helmet_count,
        )

        rows.append(
            {
                "bin": bin_index + 1,
                "size_min": float(
                    np.min(bin_sizes)
                ),
                "size_max": float(
                    np.max(bin_sizes)
                ),
                "size_median": float(
                    np.median(bin_sizes)
                ),
                "n_total": total_count,
                "n_helmet": helmet_count,
                "n_no_helmet": no_helmet_count,
                "overall_errors": overall_error_count,
                "overall_error_rate": overall_error_rate,
                "overall_ci_lower": overall_lower,
                "overall_ci_upper": overall_upper,
                "false_positives": false_positive_count,
                "fpr": false_positive_rate,
                "fpr_ci_lower": fpr_lower,
                "fpr_ci_upper": fpr_upper,
                "false_negatives": false_negative_count,
                "fnr": false_negative_rate,
                "fnr_ci_lower": fnr_lower,
                "fnr_ci_upper": fnr_upper,
            }
        )

    return rows


# =============================================================================
# OUTPUT
# =============================================================================

def save_statistics_csv(
    rows: list[dict[str, float | int]],
) -> None:
    with OUTPUT_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def plot_error_distribution(
    rows: list[dict[str, float | int]],
) -> None:
    medians = np.asarray(
        [
            row["size_median"]
            for row in rows
        ],
        dtype=np.float64,
    )

    minimum_sizes = np.asarray(
        [
            row["size_min"]
            for row in rows
        ],
        dtype=np.float64,
    )

    maximum_sizes = np.asarray(
        [
            row["size_max"]
            for row in rows
        ],
        dtype=np.float64,
    )

    x_errors = np.vstack(
        [
            medians - minimum_sizes,
            maximum_sizes - medians,
        ]
    )

    series = (
        (
            "Общая ошибка",
            "overall_error_rate",
            "overall_ci_lower",
            "overall_ci_upper",
            "n_total",
        ),
        (
            "FPR: helmet → no_helmet",
            "fpr",
            "fpr_ci_lower",
            "fpr_ci_upper",
            "n_helmet",
        ),
        (
            "FNR: no_helmet → helmet",
            "fnr",
            "fnr_ci_lower",
            "fnr_ci_upper",
            "n_no_helmet",
        ),
    )

    figure, axis = plt.subplots(
        figsize=(13, 8),
    )

    maximum_y = 0.0

    for (
        title,
        rate_key,
        lower_key,
        upper_key,
        count_key,
    ) in series:
        rates = np.asarray(
            [
                row[rate_key]
                for row in rows
            ],
            dtype=np.float64,
        )

        lower_bounds = np.asarray(
            [
                row[lower_key]
                for row in rows
            ],
            dtype=np.float64,
        )

        upper_bounds = np.asarray(
            [
                row[upper_key]
                for row in rows
            ],
            dtype=np.float64,
        )

        counts = np.asarray(
            [
                row[count_key]
                for row in rows
            ],
            dtype=np.int64,
        )

        y_errors = np.vstack(
            [
                rates - lower_bounds,
                upper_bounds - rates,
            ]
        )

        axis.errorbar(
            medians,
            rates,
            xerr=x_errors,
            yerr=y_errors,
            marker="o",
            linestyle="-",
            capsize=4,
            label=title,
        )

        maximum_y = max(
            maximum_y,
            float(np.max(upper_bounds)),
        )

        for x, y, count in zip(
            medians,
            rates,
            counts,
            strict=True,
        ):
            axis.annotate(
                f"n={count}",
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    axis.set_xscale(
        "log",
        base=2,
    )

    axis.set_ylim(
        0.0,
        min(
            1.0,
            max(
                0.05,
                maximum_y * 1.25 + 0.01,
            ),
        ),
    )

    axis.set_xlabel(
        "Оценка min(width, height) исходного bbox, пиксели"
    )

    axis.set_ylabel(
        "Вероятность ошибки"
    )

    axis.set_title(
        "Зависимость ошибки от размера bbox\n"
        f"threshold = {DECISION_THRESHOLD:.6f}"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend(
        loc="best",
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_IMAGE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not 0.0 <= DECISION_THRESHOLD <= 1.0:
        raise ValueError(
            "DECISION_THRESHOLD must be in [0, 1]"
        )

    validation_directory = (
        DATASET_ROOT
        / "val"
    ).resolve()

    if not validation_directory.exists():
        raise FileNotFoundError(
            f"Validation directory not found: "
            f"{validation_directory}"
        )

    checkpoint = load_torch_file(
        WEIGHTS_PATH.resolve(),
        map_location="cpu",
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = HelmetMicroNeXt()

    load_model_weights(
        model=model,
        checkpoint=checkpoint,
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

    labels, probabilities, sizes = collect_predictions(
        model=model,
        loader=loader,
        device=device,
    )

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    recall = (
        tp
        / max(tp + fn, 1)
    )

    false_positive_rate = (
        fp
        / max(fp + tn, 1)
    )

    edges, bin_indices = create_adaptive_bins(
        sizes=sizes,
        labels=labels,
    )

    rows = calculate_bin_statistics(
        sizes=sizes,
        labels=labels,
        predictions=predictions,
        edges=edges,
        bin_indices=bin_indices,
    )

    save_statistics_csv(rows)
    plot_error_distribution(rows)

    print()
    print(f"Weights: {WEIGHTS_PATH.resolve()}")
    print(f"Threshold: {DECISION_THRESHOLD:.9f}")
    print(f"Validation samples: {len(labels)}")
    print(f"Recall(no_helmet): {recall:.6f}")
    print(f"FPR(helmet): {false_positive_rate:.6f}")
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Adaptive bins: {len(rows)}")
    print()

    for row in rows:
        print(
            f"{row['size_min']:.1f}-"
            f"{row['size_max']:.1f}px | "
            f"median={row['size_median']:.1f} | "
            f"n={row['n_total']} | "
            f"error={row['overall_error_rate']:.4f} | "
            f"FPR={row['fpr']:.4f} | "
            f"FNR={row['fnr']:.4f}"
        )

    print()
    print(f"Image: {OUTPUT_IMAGE_PATH}")
    print(f"CSV: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
