"""
Запуск обучения HelmetMicroNeXt.

КЛАССЫ:
    0 = helmet     — каска есть
    1 = no_helmet  — каски нет

Положительный класс для precision, recall, F1, ROC-AUC и PR-AUC:
    1 = no_helmet

Балансировка:
    WeightedRandomSampler применяется только к train.
    Validation проходит без sampler: каждый пример ровно один раз.

AMP не используется.
Batch-прогресс в лог не выводится.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
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
# КОНФИГУРАЦИЯ
# =============================================================================

# -----------------------------------------------------------------------------
# Пути
# -----------------------------------------------------------------------------

DATASET_ROOT = Path(
    r"./ziz-crops-202607-2"
)

OUTPUT_DIRECTORY = Path(
    r"./runs/helmet_micronext_v1"
)


# -----------------------------------------------------------------------------
# Вход модели и нормализация
# -----------------------------------------------------------------------------

IMAGE_SIZE = 64

TRAIN_MEAN = DEFAULT_MEAN
TRAIN_STD = DEFAULT_STD


# -----------------------------------------------------------------------------
# Геометрические аугментации
#
# Исходные файлы уже являются crop, расширенными примерно в 1.25 раза.
# -----------------------------------------------------------------------------

# Размер случайного квадрата относительно максимального вписанного квадрата.
SQUARE_SCALE = (
    0.90,
    1.00,
)

# Максимальный сдвиг центра относительно стороны максимального квадрата.
CENTER_JITTER = 0.03

HORIZONTAL_FLIP_PROBABILITY = 0.50


# -----------------------------------------------------------------------------
# Фотометрические аугментации
#
# С вероятностью PHOTOMETRIC_PROBABILITY применяется ровно одна операция:
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
# Обучение
# -----------------------------------------------------------------------------

SEED = 42

EPOCHS = 80
BATCH_SIZE = 64
NUM_WORKERS = 4

BASE_LEARNING_RATE = 2.0e-3
MIN_LEARNING_RATE = 1.0e-5

WARMUP_EPOCHS = 5
WARMUP_START_FACTOR = 0.20

WEIGHT_DECAY = 1.0e-4
MAX_GRAD_NORM = 5.0


# -----------------------------------------------------------------------------
# Метрики и checkpoints
#
# Метрики precision, recall и F1 относятся к классу 1 = no_helmet.
# Specificity и FPR характеризуют класс 0 = helmet.
# -----------------------------------------------------------------------------

DECISION_THRESHOLD = 0.50

# Доступные варианты:
#     loss
#     accuracy
#     precision
#     recall
#     f1
#     auc
#     pr_auc
#     specificity
#     fpr
BEST_METRIC = "auc"

# None отключает early stopping.
EARLY_STOPPING_PATIENCE = 20


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------

PIN_MEMORY = True
PERSISTENT_WORKERS = True

# WeightedRandomSampler делает примерно равный вклад двух классов.
# replacement=True означает, что изображения редкого класса могут
# повторяться внутри одной эпохи.
SAMPLER_REPLACEMENT = True

# Количество выборок sampler за эпоху.
# None означает len(train_dataset).
TRAIN_SAMPLES_PER_EPOCH = None


# -----------------------------------------------------------------------------
# Воспроизводимость
# -----------------------------------------------------------------------------

CUDNN_DETERMINISTIC = True
CUDNN_BENCHMARK = False


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def validate_configuration() -> None:
    if EPOCHS <= 0:
        raise ValueError(
            "EPOCHS must be positive"
        )

    if BATCH_SIZE <= 0:
        raise ValueError(
            "BATCH_SIZE must be positive"
        )

    if NUM_WORKERS < 0:
        raise ValueError(
            "NUM_WORKERS cannot be negative"
        )

    if BASE_LEARNING_RATE <= 0:
        raise ValueError(
            "BASE_LEARNING_RATE must be positive"
        )

    if MIN_LEARNING_RATE < 0:
        raise ValueError(
            "MIN_LEARNING_RATE cannot be negative"
        )

    if MIN_LEARNING_RATE > BASE_LEARNING_RATE:
        raise ValueError(
            "MIN_LEARNING_RATE cannot exceed "
            "BASE_LEARNING_RATE"
        )

    if WARMUP_EPOCHS < 0:
        raise ValueError(
            "WARMUP_EPOCHS cannot be negative"
        )

    if WARMUP_EPOCHS >= EPOCHS:
        raise ValueError(
            "WARMUP_EPOCHS must be smaller than EPOCHS"
        )

    if not (
        0.0
        < WARMUP_START_FACTOR
        <= 1.0
    ):
        raise ValueError(
            "WARMUP_START_FACTOR must be in (0, 1]"
        )

    if not (
        0.0
        <= DECISION_THRESHOLD
        <= 1.0
    ):
        raise ValueError(
            "DECISION_THRESHOLD must be in [0, 1]"
        )


def set_global_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = (
        CUDNN_DETERMINISTIC
    )

    torch.backends.cudnn.benchmark = (
        CUDNN_BENCHMARK
    )


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
    Создаёт sampler с примерно равной суммарной вероятностью классов.

    Для каждого изображения:
        weight = 1 / количество изображений его класса

    Поэтому суммарный вес класса 0:
        N0 * (1 / N0) = 1

    И суммарный вес класса 1:
        N1 * (1 / N1) = 1
    """

    counts = dataset.class_counts()

    helmet_count = counts[0]
    no_helmet_count = counts[1]

    if (
        helmet_count <= 0
        or no_helmet_count <= 0
    ):
        raise RuntimeError(
            "Both train classes are required. "
            f"Counts: {counts}"
        )

    class_weights = {
        0: 1.0 / helmet_count,
        1: 1.0 / no_helmet_count,
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
            "TRAIN_SAMPLES_PER_EPOCH "
            "must be positive or None"
        )

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(SEED)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=number_of_samples,
        replacement=SAMPLER_REPLACEMENT,
        generator=sampler_generator,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
):
    """
    Планировщик:

        warmup:
            BASE_LR * WARMUP_START_FACTOR
            -> BASE_LR

        затем:
            cosine decay
            -> MIN_LEARNING_RATE

    scheduler.step() вызывается один раз после каждой эпохи в train.py.
    """

    cosine_epochs = max(
        1,
        EPOCHS - WARMUP_EPOCHS,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cosine_epochs,
        eta_min=MIN_LEARNING_RATE,
    )

    if WARMUP_EPOCHS == 0:
        return cosine_scheduler

    warmup_scheduler = LinearLR(
        optimizer=optimizer,
        start_factor=WARMUP_START_FACTOR,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )

    return SequentialLR(
        optimizer=optimizer,
        schedulers=[
            warmup_scheduler,
            cosine_scheduler,
        ],
        milestones=[
            WARMUP_EPOCHS,
        ],
    )


def print_dataset_summary(
    name: str,
    dataset: HelmetFolderDataset,
) -> dict[int, int]:
    counts = dataset.class_counts()

    print(
        f"{name}: {len(dataset)} images"
    )

    print(
        f"  class 0 ({CLASS_NAMES[0]}): "
        f"{counts[0]}"
    )

    print(
        f"  class 1 ({CLASS_NAMES[1]}): "
        f"{counts[1]}"
    )

    if len(dataset) > 0:
        helmet_fraction = (
            counts[0] / len(dataset)
        )

        no_helmet_fraction = (
            counts[1] / len(dataset)
        )

        print(
            f"  class fractions: "
            f"helmet={helmet_fraction:.4f}, "
            f"no_helmet={no_helmet_fraction:.4f}"
        )

    return counts


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    validate_configuration()
    set_global_seed(SEED)

    print("=" * 78)
    print("HelmetMicroNeXt binary classification")
    print("CLASS 0 = helmet     = каска есть")
    print(
        "CLASS 1 = no_helmet  = каски нет "
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
        "train.py version: "
        f"{getattr(train_module, 'TRAIN_MODULE_VERSION', 'unknown')}"
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
        brightness_limit=BRIGHTNESS_LIMIT,
        contrast_limit=CONTRAST_LIMIT,
        gamma_range=GAMMA_RANGE,
    )

    val_preprocessor = HelmetPreprocessor(
        training=False,
        output_size=IMAGE_SIZE,
        mean=TRAIN_MEAN,
        std=TRAIN_STD,
    )

    train_dataset = HelmetFolderDataset(
        split_directory=train_directory,
        preprocessor=train_preprocessor,
    )

    val_dataset = HelmetFolderDataset(
        split_directory=val_directory,
        preprocessor=val_preprocessor,
    )

    train_counts = print_dataset_summary(
        "Train",
        train_dataset,
    )

    val_counts = print_dataset_summary(
        "Validation",
        val_dataset,
    )

    if (
        val_counts[0] == 0
        or val_counts[1] == 0
    ):
        raise RuntimeError(
            "Validation must contain both classes "
            "for ROC-AUC and PR-AUC calculation."
        )

    print(
        f"Input mean RGB: "
        f"{list(TRAIN_MEAN)}"
    )

    print(
        f"Input std  RGB: "
        f"{list(TRAIN_STD)}"
    )

    print()

    train_sampler = build_balanced_sampler(
        train_dataset
    )

    train_loader_generator = (
        torch.Generator()
    )

    train_loader_generator.manual_seed(
        SEED
    )

    val_loader_generator = (
        torch.Generator()
    )

    val_loader_generator.manual_seed(
        SEED + 1
    )

    effective_pin_memory = (
        PIN_MEMORY
        and torch.cuda.is_available()
    )

    effective_persistent_workers = (
        PERSISTENT_WORKERS
        and NUM_WORKERS > 0
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,

        # Балансировка применяется только здесь.
        sampler=train_sampler,
        shuffle=False,

        num_workers=NUM_WORKERS,
        pin_memory=effective_pin_memory,
        persistent_workers=(
            effective_persistent_workers
        ),
        worker_init_fn=seed_worker,
        generator=train_loader_generator,
        drop_last=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,

        # Validation не балансируется.
        sampler=None,
        shuffle=False,

        num_workers=NUM_WORKERS,
        pin_memory=effective_pin_memory,
        persistent_workers=(
            effective_persistent_workers
        ),
        worker_init_fn=seed_worker,
        generator=val_loader_generator,
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
        params=model.parameters(),
        lr=BASE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_scheduler(
        optimizer
    )

    train_samples_per_epoch = len(
        train_sampler
    )

    print(f"Device: {device}")

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
        f"Train samples per epoch: "
        f"{train_samples_per_epoch}"
    )

    print(
        "Train sampler: "
        "WeightedRandomSampler, "
        f"replacement={SAMPLER_REPLACEMENT}, "
        "expected class contribution ≈ 50/50"
    )

    print(
        "Validation sampler: disabled, "
        "each validation image is used once"
    )

    print(
        "Loss: BCEWithLogitsLoss, "
        "without pos_weight"
    )

    print(
        f"Base learning rate: "
        f"{BASE_LEARNING_RATE:.3e}"
    )

    print(
        f"Warmup epochs: "
        f"{WARMUP_EPOCHS}"
    )

    print(
        f"Warmup start factor: "
        f"{WARMUP_START_FACTOR:.2f}"
    )

    print(
        f"Minimum learning rate: "
        f"{MIN_LEARNING_RATE:.3e}"
    )

    print(
        f"Weight decay: "
        f"{WEIGHT_DECAY:.3e}"
    )

    print(
        f"Decision threshold: "
        f"{DECISION_THRESHOLD:.3f}"
    )

    print(
        f"Best checkpoint metric: "
        f"val_{BEST_METRIC}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIRECTORY.resolve()}"
    )

    print()

    config_for_checkpoint = {
        "dataset_root": str(
            DATASET_ROOT.resolve()
        ),
        "output_directory": str(
            OUTPUT_DIRECTORY.resolve()
        ),

        "class_0": "helmet",
        "class_1": "no_helmet",
        "positive_class": 1,

        "train_class_counts": {
            0: train_counts[0],
            1: train_counts[1],
        },

        "val_class_counts": {
            0: val_counts[0],
            1: val_counts[1],
        },

        "image_size": IMAGE_SIZE,
        "train_mean": list(TRAIN_MEAN),
        "train_std": list(TRAIN_STD),

        "square_scale": list(
            SQUARE_SCALE
        ),
        "center_jitter": CENTER_JITTER,
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
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,

        "decision_threshold": (
            DECISION_THRESHOLD
        ),
        "best_metric": BEST_METRIC,
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),

        "sampler": (
            "WeightedRandomSampler"
        ),
        "sampler_replacement": (
            SAMPLER_REPLACEMENT
        ),
        "train_samples_per_epoch": (
            train_samples_per_epoch
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
        output_directory=OUTPUT_DIRECTORY,
        max_grad_norm=MAX_GRAD_NORM,
        threshold=DECISION_THRESHOLD,
        best_metric=BEST_METRIC,
        early_stopping_patience=(
            EARLY_STOPPING_PATIENCE
        ),
        extra_config=config_for_checkpoint,
    )


if __name__ == "__main__":
    main()
