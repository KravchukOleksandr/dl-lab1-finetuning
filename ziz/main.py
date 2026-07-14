"""
CLASS 0 = helmet     (каска есть)
CLASS 1 = no_helmet  (каски нет, positive class)

Train:
    WeightedRandomSampler, примерно 50/50.
    BCEWithLogitsLoss без pos_weight.

Validation:
    sampler отсутствует;
    каждый пример используется ровно один раз.

Выбор модели:
    минимальный FPR при recall(no_helmet) >= 0.95.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
)

import train as train_module
from model import (
    HelmetMicroNeXt,
    count_trainable_parameters,
)
from preprocessing import (
    CLASS_NAMES,
    DEFAULT_MEAN,
    DEFAULT_STD,
    HelmetFolderDataset,
    HelmetPreprocessor,
)
from train import fit


# =============================================================================
# CONFIG
# =============================================================================

DATASET_ROOT = Path(
    r"./ziz-crops-202607-2"
)

OUTPUT_DIRECTORY = Path(
    r"./runs/helmet_micronext_v2_ema_r95"
)


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------

IMAGE_SIZE = 64

TRAIN_MEAN = DEFAULT_MEAN
TRAIN_STD = DEFAULT_STD


# -----------------------------------------------------------------------------
# Geometry augmentation
# -----------------------------------------------------------------------------

SQUARE_SCALE = (
    0.90,
    1.00,
)

CENTER_JITTER = 0.03

HORIZONTAL_FLIP_PROBABILITY = 0.50


# -----------------------------------------------------------------------------
# Photometric augmentation
#
# С вероятностью 0.5 применяется ровно одна операция:
# brightness, contrast или gamma.
# -----------------------------------------------------------------------------

PHOTOMETRIC_PROBABILITY = 0.50

BRIGHTNESS_LIMIT = 0.10
CONTRAST_LIMIT = 0.10

GAMMA_RANGE = (
    0.90,
    1.10,
)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

SEED = 42

EPOCHS = 40
BATCH_SIZE = 128
NUM_WORKERS = 4

BASE_LEARNING_RATE = 1.0e-3
MIN_LEARNING_RATE = 1.0e-5

WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.20

WEIGHT_DECAY = 1.0e-4

MAX_GRAD_NORM = 5.0

EMA_DECAY = 0.997


# -----------------------------------------------------------------------------
# Metrics and model selection
# -----------------------------------------------------------------------------

# Только для диагностических метрик.
# Лучший рабочий threshold будет вычисляться автоматически.
DECISION_THRESHOLD = 0.50

TARGET_RECALL = 0.95

BEST_METRIC = "fpr_at_recall_95"

# Проходим все 40 эпох.
EARLY_STOPPING_PATIENCE = None


# -----------------------------------------------------------------------------
# Sampler and DataLoader
# -----------------------------------------------------------------------------

SAMPLER_REPLACEMENT = True

# None означает len(train_dataset).
TRAIN_SAMPLES_PER_EPOCH = None

PIN_MEMORY = True
PERSISTENT_WORKERS = True


# =============================================================================
# HELPERS
# =============================================================================

def set_global_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(
    worker_id: int,
) -> None:
    del worker_id

    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_balanced_sampler(
    dataset: HelmetFolderDataset,
) -> WeightedRandomSampler:
    """
    Каждое изображение получает вес:

        1 / количество изображений его класса

    Поэтому суммарный вес каждого класса равен примерно 1,
    и sampler выбирает классы примерно 50/50.
    """

    counts = dataset.class_counts()

    if (
        counts[0] <= 0
        or counts[1] <= 0
    ):
        raise RuntimeError(
            "Both train classes are required. "
            f"Counts: {counts}"
        )

    class_weights = {
        0: 1.0 / counts[0],
        1: 1.0 / counts[1],
    }

    sample_weights = torch.tensor(
        [
            class_weights[target]
            for target in dataset.targets
        ],
        dtype=torch.double,
    )

    if TRAIN_SAMPLES_PER_EPOCH is None:
        number_of_samples = len(dataset)
    else:
        number_of_samples = int(
            TRAIN_SAMPLES_PER_EPOCH
        )

    if number_of_samples <= 0:
        raise ValueError(
            "TRAIN_SAMPLES_PER_EPOCH must "
            "be positive or None"
        )

    sampler_generator = (
        torch.Generator()
    )

    sampler_generator.manual_seed(
        SEED
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=number_of_samples,
        replacement=SAMPLER_REPLACEMENT,
        generator=sampler_generator,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
) -> LambdaLR:
    """
    Epoch 1:
        LR = 0.2 * BASE_LR

    Epoch 2:
        LR = 0.6 * BASE_LR

    Epoch 3:
        LR = BASE_LR

    Затем cosine decay до MIN_LR к эпохе 40.

    Используется LambdaLR, поэтому предупреждения SequentialLR
    из предыдущего запуска больше не будет.
    """

    minimum_factor = (
        MIN_LEARNING_RATE
        / BASE_LEARNING_RATE
    )

    cosine_epochs = (
        EPOCHS
        - WARMUP_EPOCHS
    )

    def learning_rate_factor(
        epoch_index: int,
    ) -> float:
        # LambdaLR использует zero-based index.
        # epoch_index=0 соответствует первой эпохе.

        if (
            WARMUP_EPOCHS > 0
            and epoch_index < WARMUP_EPOCHS
        ):
            if WARMUP_EPOCHS == 1:
                return 1.0

            progress = (
                epoch_index
                / (WARMUP_EPOCHS - 1)
            )

            return (
                WARMUP_START_FACTOR
                + (
                    1.0
                    - WARMUP_START_FACTOR
                )
                * progress
            )

        cosine_step = (
            epoch_index
            - WARMUP_EPOCHS
            + 1
        )

        progress = (
            cosine_step
            / max(cosine_epochs, 1)
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return (
            minimum_factor
            + 0.5
            * (
                1.0
                - minimum_factor
            )
            * (
                1.0
                + math.cos(
                    math.pi * progress
                )
            )
        )

    return LambdaLR(
        optimizer,
        lr_lambda=learning_rate_factor,
    )


def print_dataset(
    name: str,
    dataset: HelmetFolderDataset,
) -> dict[int, int]:
    counts = dataset.class_counts()

    print(
        f"{name}: "
        f"{len(dataset)} images"
    )

    print(
        f"  class 0 "
        f"({CLASS_NAMES[0]}): "
        f"{counts[0]}"
    )

    print(
        f"  class 1 "
        f"({CLASS_NAMES[1]}): "
        f"{counts[1]}"
    )

    return counts


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    expected_metric = (
        "fpr_at_recall_"
        f"{int(round(TARGET_RECALL * 100))}"
    )

    if BEST_METRIC != expected_metric:
        raise ValueError(
            f"Set BEST_METRIC="
            f"{expected_metric!r}"
        )

    if not (
        0 <= WARMUP_EPOCHS < EPOCHS
    ):
        raise ValueError(
            "WARMUP_EPOCHS must be "
            "in [0, EPOCHS)"
        )

    if not (
        0.0 < TARGET_RECALL <= 1.0
    ):
        raise ValueError(
            "TARGET_RECALL must be in (0, 1]"
        )

    set_global_seed(SEED)

    print("=" * 78)

    print(
        "CLASS 0 = helmet     "
        "= каска есть"
    )

    print(
        "CLASS 1 = no_helmet  "
        "= каски нет "
        "(positive class)"
    )

    print("=" * 78)

    print(
        f"main.py:  "
        f"{Path(__file__).resolve()}"
    )

    print(
        f"train.py: "
        f"{Path(train_module.__file__).resolve()}"
    )

    print(
        f"train.py version: "
        f"{train_module.TRAIN_MODULE_VERSION}"
    )

    print()

    train_directory = (
        DATASET_ROOT
        / "train"
    )

    val_directory = (
        DATASET_ROOT
        / "val"
    )

    if not train_directory.exists():
        raise FileNotFoundError(
            f"Train directory not found: "
            f"{train_directory.resolve()}"
        )

    if not val_directory.exists():
        raise FileNotFoundError(
            f"Validation directory not found: "
            f"{val_directory.resolve()}"
        )

    train_preprocessor = HelmetPreprocessor(
        training=True,
        output_size=IMAGE_SIZE,
        mean=TRAIN_MEAN,
        std=TRAIN_STD,
        square_scale=SQUARE_SCALE,
        center_jitter=CENTER_JITTER,
        horizontal_flip_probability=(
            HORIZONTAL_FLIP_PROBABILITY
        ),
        photometric_probability=(
            PHOTOMETRIC_PROBABILITY
        ),
        brightness_limit=(
            BRIGHTNESS_LIMIT
        ),
        contrast_limit=(
            CONTRAST_LIMIT
        ),
        gamma_range=GAMMA_RANGE,
    )

    val_preprocessor = HelmetPreprocessor(
        training=False,
        output_size=IMAGE_SIZE,
        mean=TRAIN_MEAN,
        std=TRAIN_STD,
    )

    train_dataset = HelmetFolderDataset(
        train_directory,
        train_preprocessor,
    )

    val_dataset = HelmetFolderDataset(
        val_directory,
        val_preprocessor,
    )

    train_counts = print_dataset(
        "Train",
        train_dataset,
    )

    val_counts = print_dataset(
        "Validation",
        val_dataset,
    )

    if (
        val_counts[0] <= 0
        or val_counts[1] <= 0
    ):
        raise RuntimeError(
            "Validation must contain "
            "both classes"
        )

    sampler = build_balanced_sampler(
        train_dataset
    )

    effective_pin_memory = (
        PIN_MEMORY
        and torch.cuda.is_available()
    )

    effective_persistent_workers = (
        PERSISTENT_WORKERS
        and NUM_WORKERS > 0
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        SEED
    )

    val_generator = (
        torch.Generator()
    )

    val_generator.manual_seed(
        SEED + 1
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,

        sampler=sampler,
        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            effective_pin_memory
        ),

        persistent_workers=(
            effective_persistent_workers
        ),

        worker_init_fn=seed_worker,
        generator=train_generator,

        drop_last=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,

        sampler=None,
        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            effective_pin_memory
        ),

        persistent_workers=(
            effective_persistent_workers
        ),

        worker_init_fn=seed_worker,
        generator=val_generator,

        drop_last=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = HelmetMicroNeXt().to(
        device
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=BASE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_scheduler(
        optimizer
    )

    print(
        f"Device: "
        f"{device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Trainable parameters: "
        f"{count_trainable_parameters(model):,}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs: "
        f"{EPOCHS}"
    )

    print(
        f"Samples per train epoch: "
        f"{len(sampler)}"
    )

    print(
        "Train sampler: "
        "WeightedRandomSampler (~50/50)"
    )

    print(
        "Validation sampler: disabled"
    )

    print(
        "Loss: BCEWithLogitsLoss "
        "without pos_weight"
    )

    print(
        f"LR: "
        f"{BASE_LEARNING_RATE:.3e} "
        f"-> {MIN_LEARNING_RATE:.3e}"
    )

    print(
        f"Warmup: "
        f"{WARMUP_EPOCHS} epochs"
    )

    print(
        f"EMA decay: "
        f"{EMA_DECAY}"
    )

    print(
        "Selection: minimum FPR "
        f"with recall >= {TARGET_RECALL:.2f}"
    )

    print(
        f"Output: "
        f"{OUTPUT_DIRECTORY.resolve()}"
    )

    print()

    config = {
        "dataset_root": str(
            DATASET_ROOT.resolve()
        ),

        "class_0": "helmet",
        "class_1": "no_helmet",
        "positive_class": 1,

        "train_class_counts": (
            train_counts
        ),

        "val_class_counts": (
            val_counts
        ),

        "image_size": IMAGE_SIZE,

        "train_mean": list(
            TRAIN_MEAN
        ),

        "train_std": list(
            TRAIN_STD
        ),

        "square_scale": list(
            SQUARE_SCALE
        ),

        "center_jitter": (
            CENTER_JITTER
        ),

        "horizontal_flip_probability": (
            HORIZONTAL_FLIP_PROBABILITY
        ),

        "photometric_probability": (
            PHOTOMETRIC_PROBABILITY
        ),

        "brightness_limit": (
            BRIGHTNESS_LIMIT
        ),

        "contrast_limit": (
            CONTRAST_LIMIT
        ),

        "gamma_range": list(
            GAMMA_RANGE
        ),

        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,

        "base_learning_rate": (
            BASE_LEARNING_RATE
        ),

        "minimum_learning_rate": (
            MIN_LEARNING_RATE
        ),

        "warmup_epochs": (
            WARMUP_EPOCHS
        ),

        "warmup_start_factor": (
            WARMUP_START_FACTOR
        ),

        "weight_decay": (
            WEIGHT_DECAY
        ),

        "max_grad_norm": (
            MAX_GRAD_NORM
        ),

        "ema_decay": (
            EMA_DECAY
        ),

        "decision_threshold": (
            DECISION_THRESHOLD
        ),

        "target_recall": (
            TARGET_RECALL
        ),

        "best_metric": (
            BEST_METRIC
        ),

        "sampler": (
            "WeightedRandomSampler"
        ),

        "sampler_replacement": (
            SAMPLER_REPLACEMENT
        ),

        "train_samples_per_epoch": (
            len(sampler)
        ),

        "amp": False,
    }

    fit(
        model=model,

        train_loader=train_loader,
        val_loader=val_loader,

        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,

        device=device,

        epochs=EPOCHS,

        output_directory=(
            OUTPUT_DIRECTORY
        ),

        ema_decay=EMA_DECAY,

        target_recall=(
            TARGET_RECALL
        ),

        max_grad_norm=(
            MAX_GRAD_NORM
        ),

        threshold=(
            DECISION_THRESHOLD
        ),

        best_metric=(
            BEST_METRIC
        ),

        early_stopping_patience=(
            EARLY_STOPPING_PATIENCE
        ),

        extra_config=config,
    )


if __name__ == "__main__":
    main()