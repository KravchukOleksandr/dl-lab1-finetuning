from __future__ import annotations

import csv
import shutil
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
# CONFIG
# =============================================================================

DATASET_ROOT = Path("./ziz-crops-202607-2")

WEIGHTS_PATH = Path(
    "./models/helmet_finetuned.pt"
)

DECISION_THRESHOLD = 0.358260

ERRORS_DIRECTORY = Path("./val_errors")

SAVE_ERRORS = True

# True:
#     перед каждой валидацией удалять старые FP/FN.
#
# False:
#     оставлять старые файлы и дописывать новые.
CLEAR_ERROR_DIRECTORIES = True

IMAGE_SIZE = 64
BATCH_SIZE = 512
NUM_WORKERS = 4
PIN_MEMORY = True


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
            "Unsupported checkpoint type: "
            f"{type(checkpoint)}"
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
            print(
                f"State dict source: "
                f"checkpoint[{key!r}]"
            )

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
    source_state = extract_state_dict(
        checkpoint
    )

    target_state = model.state_dict()

    normalized_source = {
        normalize_state_key(key): tensor
        for key, tensor in source_state.items()
    }

    mapped_state: dict[str, torch.Tensor] = {}

    for target_key, target_tensor in target_state.items():
        source_tensor = normalized_source.get(
            target_key
        )

        if source_tensor is None:
            continue

        if tuple(source_tensor.shape) != tuple(
            target_tensor.shape
        ):
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
        and not key.endswith(
            "num_batches_tracked"
        )
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

    print(
        f"Loaded tensors: "
        f"{len(mapped_state)}/{len(target_state)}"
    )


# =============================================================================
# ERROR DIRECTORIES
# =============================================================================

def prepare_error_directories() -> tuple[Path, Path]:
    fp_directory = (
        ERRORS_DIRECTORY
        / "FP"
    )

    fn_directory = (
        ERRORS_DIRECTORY
        / "FN"
    )

    if (
        CLEAR_ERROR_DIRECTORIES
        and ERRORS_DIRECTORY.exists()
    ):
        shutil.rmtree(
            ERRORS_DIRECTORY
        )

    fp_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fn_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        fp_directory,
        fn_directory,
    )


def create_unique_destination(
    directory: Path,
    source_path: Path,
) -> Path:
    destination = (
        directory
        / source_path.name
    )

    if not destination.exists():
        return destination

    counter = 2

    while True:
        candidate = (
            directory
            / (
                f"{source_path.stem}"
                f"__{counter}"
                f"{source_path.suffix}"
            )
        )

        if not candidate.exists():
            return candidate

        counter += 1


def copy_error_image(
    source_path: str | Path,
    destination_directory: Path,
) -> Path:
    source = Path(source_path)

    if not source.exists():
        raise FileNotFoundError(
            f"Validation image not found: {source}"
        )

    destination = create_unique_destination(
        destination_directory,
        source,
    )

    shutil.copy2(
        source,
        destination,
    )

    return destination


# =============================================================================
# VALIDATION
# =============================================================================

@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
]:
    model.eval()

    labels: list[int] = []
    probabilities: list[float] = []
    paths: list[str] = []

    for batch in loader:
        if len(batch) < 3:
            raise RuntimeError(
                "Dataset must return: "
                "(image, label, path)"
            )

        images = batch[0]
        batch_labels = batch[1]
        batch_paths = batch[2]

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

        paths.extend(
            str(path)
            for path in batch_paths
        )

    return (
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
        paths,
    )


# =============================================================================
# SAVE FP/FN
# =============================================================================

def save_validation_errors(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    paths: list[str],
) -> tuple[int, int, Path]:
    fp_directory, fn_directory = (
        prepare_error_directories()
    )

    report_path = (
        ERRORS_DIRECTORY
        / "errors.csv"
    )

    false_positive_count = 0
    false_negative_count = 0

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = (
            "error_type",
            "source_path",
            "saved_path",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "probability_no_helmet",
            "threshold",
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for (
            true_label,
            probability,
            prediction,
            source_path,
        ) in zip(
            labels,
            probabilities,
            predictions,
            paths,
            strict=True,
        ):
            error_type: str | None = None
            destination_directory: Path | None = None

            if (
                true_label == 0
                and prediction == 1
            ):
                error_type = "FP"
                destination_directory = (
                    fp_directory
                )

                false_positive_count += 1

            elif (
                true_label == 1
                and prediction == 0
            ):
                error_type = "FN"
                destination_directory = (
                    fn_directory
                )

                false_negative_count += 1

            if error_type is None:
                continue

            saved_path = copy_error_image(
                source_path=source_path,
                destination_directory=(
                    destination_directory
                ),
            )

            writer.writerow(
                {
                    "error_type": error_type,
                    "source_path": source_path,
                    "saved_path": str(
                        saved_path.resolve()
                    ),
                    "true_label": int(
                        true_label
                    ),
                    "true_class": (
                        "helmet"
                        if true_label == 0
                        else "no_helmet"
                    ),
                    "predicted_label": int(
                        prediction
                    ),
                    "predicted_class": (
                        "helmet"
                        if prediction == 0
                        else "no_helmet"
                    ),
                    "probability_no_helmet": (
                        f"{probability:.9f}"
                    ),
                    "threshold": (
                        f"{DECISION_THRESHOLD:.9f}"
                    ),
                }
            )

    return (
        false_positive_count,
        false_negative_count,
        report_path,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not 0.0 <= DECISION_THRESHOLD <= 1.0:
        raise ValueError(
            "DECISION_THRESHOLD must be "
            "inside [0, 1]"
        )

    weights_path = (
        WEIGHTS_PATH
        .expanduser()
        .resolve()
    )

    validation_directory = (
        DATASET_ROOT
        / "val"
    ).expanduser().resolve()

    if not validation_directory.exists():
        raise FileNotFoundError(
            "Validation directory not found: "
            f"{validation_directory}"
        )

    checkpoint = load_torch_file(
        weights_path,
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

    (
        labels,
        probabilities,
        paths,
    ) = collect_predictions(
        model=model,
        loader=loader,
        device=device,
    )

    predictions = (
        probabilities
        >= DECISION_THRESHOLD
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    specificity = (
        tn
        / max(tn + fp, 1)
    )

    false_positive_rate = (
        fp
        / max(fp + tn, 1)
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities,
    )

    pr_auc = average_precision_score(
        labels,
        probabilities,
    )

    if SAVE_ERRORS:
        (
            saved_fp_count,
            saved_fn_count,
            report_path,
        ) = save_validation_errors(
            labels=labels,
            probabilities=probabilities,
            predictions=predictions,
            paths=paths,
        )

        if saved_fp_count != fp:
            raise RuntimeError(
                "Saved FP count does not match "
                f"confusion matrix: "
                f"{saved_fp_count} != {fp}"
            )

        if saved_fn_count != fn:
            raise RuntimeError(
                "Saved FN count does not match "
                f"confusion matrix: "
                f"{saved_fn_count} != {fn}"
            )

    print()
    print("=" * 78)
    print("CLASS 0 = helmet")
    print("CLASS 1 = no_helmet")
    print(f"Weights: {weights_path}")
    print(f"Device: {device}")
    print(
        f"Validation images: {len(dataset)}"
    )
    print(
        f"Threshold: "
        f"{DECISION_THRESHOLD:.9f}"
    )
    print("-" * 78)
    print(
        f"Accuracy:    {accuracy:.6f}"
    )
    print(
        f"Precision:   {precision:.6f}"
    )
    print(
        f"Recall:      {recall:.6f}"
    )
    print(
        f"F1:          {f1:.6f}"
    )
    print(
        f"Specificity: {specificity:.6f}"
    )
    print(
        f"FPR:         "
        f"{false_positive_rate:.6f}"
    )
    print(
        f"ROC-AUC:     {roc_auc:.6f}"
    )
    print(
        f"PR-AUC:      {pr_auc:.6f}"
    )
    print("-" * 78)
    print(
        f"TN={tn} FP={fp} FN={fn} TP={tp}"
    )

    if SAVE_ERRORS:
        print("-" * 78)
        print(
            f"FP directory: "
            f"{(ERRORS_DIRECTORY / 'FP').resolve()}"
        )
        print(
            f"FN directory: "
            f"{(ERRORS_DIRECTORY / 'FN').resolve()}"
        )
        print(
            f"Error report: "
            f"{report_path.resolve()}"
        )

    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
