"""
Классы:
    0 = helmet     — каска есть
    1 = no_helmet  — каски нет

FP:
    настоящий класс helmet,
    модель предсказала no_helmet.

FN:
    настоящий класс no_helmet,
    модель предсказала helmet.
"""

from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader

from model import HelmetMicroNeXt
from preprocessing import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    HelmetFolderDataset,
    HelmetPreprocessor,
)


# =============================================================================
# НАСТРОЙКИ
# =============================================================================

DATASET_ROOT = Path(
    "./ziz-crops-202607-2"
)

CHECKPOINT_PATH = Path(
    "./runs/helmet_micronext_v2_ema_r95/best.pt"
)

# Папка создаётся рядом с DATASET_ROOT:
#
# parent/
# ├── ziz-crops-202607-2/
# └── val_errors/
ERRORS_ROOT = (
    DATASET_ROOT.parent
    / "val_errors"
)

IMAGE_SIZE = 64

BATCH_SIZE = 512
NUM_WORKERS = 4

# Удалить результаты предыдущего запуска.
CLEAR_OLD_ERRORS = True


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{path.resolve()}"
        )

    # Совместимость со старыми версиями PyTorch.
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def get_threshold(
    checkpoint: dict,
) -> float:
    best_metrics = checkpoint.get(
        "best_val_metrics"
    )

    if not isinstance(
        best_metrics,
        dict,
    ):
        raise KeyError(
            "Checkpoint does not contain "
            "'best_val_metrics'"
        )

    key = "threshold_at_recall_95"

    if key not in best_metrics:
        raise KeyError(
            f"Checkpoint does not contain "
            f"best_val_metrics[{key!r}]"
        )

    return float(
        best_metrics[key]
    )


def copy_with_original_name(
    source: Path,
    destination_directory: Path,
) -> None:
    destination = (
        destination_directory
        / source.name
    )

    # Не допускаем тихого перезаписывания,
    # если в разных подпапках встретятся одинаковые имена.
    if destination.exists():
        raise RuntimeError(
            "Duplicate filename detected:\n"
            f"source:      {source}\n"
            f"destination: {destination}\n"
            "Files were not overwritten."
        )

    shutil.copy2(
        source,
        destination,
    )


# =============================================================================
# MAIN
# =============================================================================

@torch.no_grad()
def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        device,
    )

    threshold = get_threshold(
        checkpoint
    )

    model = HelmetMicroNeXt().to(
        device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    val_preprocessor = HelmetPreprocessor(
        training=False,
        output_size=IMAGE_SIZE,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
    )

    val_dataset = HelmetFolderDataset(
        split_directory=(
            DATASET_ROOT
            / "val"
        ),
        preprocessor=val_preprocessor,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=None,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        drop_last=False,
    )

    if (
        CLEAR_OLD_ERRORS
        and ERRORS_ROOT.exists()
    ):
        shutil.rmtree(
            ERRORS_ROOT
        )

    fp_directory = (
        ERRORS_ROOT
        / "FP"
    )

    fn_directory = (
        ERRORS_ROOT
        / "FN"
    )

    fp_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fn_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    for images, labels, paths in val_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images
        ).reshape(-1)

        probabilities = torch.sigmoid(
            logits
        ).cpu()

        labels = labels.to(
            torch.int64
        ).reshape(-1)

        for label, probability, path_string in zip(
            labels.tolist(),
            probabilities.tolist(),
            paths,
        ):
            prediction = int(
                probability >= threshold
            )

            source_path = Path(
                path_string
            )

            if label == 0 and prediction == 0:
                tn += 1

            elif label == 0 and prediction == 1:
                fp += 1

                copy_with_original_name(
                    source=source_path,
                    destination_directory=(
                        fp_directory
                    ),
                )

            elif label == 1 and prediction == 0:
                fn += 1

                copy_with_original_name(
                    source=source_path,
                    destination_directory=(
                        fn_directory
                    ),
                )

            else:
                tp += 1

    recall = (
        tp / max(tp + fn, 1)
    )

    fpr = (
        fp / max(fp + tn, 1)
    )

    print("=" * 70)
    print(f"Device: {device}")
    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH.resolve()}"
    )
    print(
        f"Threshold: {threshold:.6f}"
    )
    print()
    print(
        f"TN={tn}  FP={fp}  "
        f"FN={fn}  TP={tp}"
    )
    print(
        f"Recall no_helmet: "
        f"{recall:.4f}"
    )
    print(
        f"FPR helmet: "
        f"{fpr:.4f}"
    )
    print()
    print(
        f"FP copied to: "
        f"{fp_directory.resolve()}"
    )
    print(
        f"FN copied to: "
        f"{fn_directory.resolve()}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()