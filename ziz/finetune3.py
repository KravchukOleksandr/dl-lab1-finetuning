"""
Fine-tuning HelmetMicroNeXt.

Классы:
    0 = helmet     — каска есть
    1 = no_helmet  — каски нет, положительный класс

Train:
    WeightedRandomSampler, примерно 50/50.
    BCEWithLogitsLoss без pos_weight.

Validation:
    каждый пример используется ровно один раз.

Критерий выбора:
    минимальный FPR при recall(no_helmet) >= 0.95.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

import train as train_module
from model import HelmetMicroNeXt, count_trainable_parameters
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

DATASET_ROOT = Path("./ziz-crops-202607-2")

# Предобученная модель.
PRETRAINED_WEIGHTS = Path("./models/pretrained.pt")

# Полные checkpoints обучения.
OUTPUT_DIRECTORY = Path("./runs/helmet_finetune")

# Чистый state_dict лучшей EMA-модели.
EXPORTED_WEIGHTS = Path("./models/helmet_finetuned.pt")


# -----------------------------------------------------------------------------
# Загрузка pretrain
# -----------------------------------------------------------------------------

# False рекомендуется для multi-task pretrain:
# старая классификационная голова не загружается.
#
# True можно поставить, если pretrain имел ровно ту же бинарную задачу
# и точно такую же голову.
LOAD_CLASSIFIER_HEAD = False

# Скрипт остановится, если удалось загрузить слишком малую часть backbone.
MIN_BACKBONE_LOAD_FRACTION = 0.70


# -----------------------------------------------------------------------------
# Вход и preprocessing
# -----------------------------------------------------------------------------

IMAGE_SIZE = 64

TRAIN_MEAN = DEFAULT_MEAN
TRAIN_STD = DEFAULT_STD

SQUARE_SCALE = (0.90, 1.00)
CENTER_JITTER = 0.03
HORIZONTAL_FLIP_PROBABILITY = 0.50

PHOTOMETRIC_PROBABILITY = 0.50
BRIGHTNESS_LIMIT = 0.10
CONTRAST_LIMIT = 0.10
GAMMA_RANGE = (0.90, 1.10)


# -----------------------------------------------------------------------------
# Fine-tuning
# -----------------------------------------------------------------------------

SEED = 42

EPOCHS = 25
BATCH_SIZE = 128
NUM_WORKERS = 4

# Backbone обучается осторожнее новой головы.
BACKBONE_LEARNING_RATE = 1.0e-4
HEAD_LEARNING_RATE = 5.0e-4

MIN_LR_FACTOR = 0.05

WARMUP_EPOCHS = 2
WARMUP_START_FACTOR = 0.20

WEIGHT_DECAY = 5.0e-4
MAX_GRAD_NORM = 5.0

EMA_DECAY = 0.997

TARGET_RECALL = 0.95
DECISION_THRESHOLD = 0.50
BEST_METRIC = "fpr_at_recall_95"

EARLY_STOPPING_PATIENCE = None


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------

PIN_MEMORY = True
PERSISTENT_WORKERS = True

SAMPLER_REPLACEMENT = True
TRAIN_SAMPLES_PER_EPOCH: int | None = None


# =============================================================================
# ВОСПРОИЗВОДИМОСТЬ
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# ЗАГРУЗКА PRETRAINED WEIGHTS
# =============================================================================

def load_torch_file(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Файл весов не найден: {path.resolve()}"
        )

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        # Совместимость со старыми версиями PyTorch.
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
        for key in value.keys()
    ) and any(
        torch.is_tensor(item)
        for item in value.values()
    )


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """
    Поддерживает:
        torch.save(model.state_dict(), path)

    и checkpoints с ключами:
        model_state_dict
        state_dict
        ema_state_dict
        weights
        model
    """

    if looks_like_state_dict(checkpoint):
        return dict(checkpoint)

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Не удалось извлечь state_dict: "
            f"получен объект типа {type(checkpoint)}"
        )

    preferred_keys = (
        "model_state_dict",
        "state_dict",
        "ema_state_dict",
        "weights",
        "model",
    )

    for key in preferred_keys:
        value = checkpoint.get(key)

        if looks_like_state_dict(value):
            print(f"State dict найден в checkpoint[{key!r}]")
            return dict(value)

    raise KeyError(
        "В checkpoint не найден state_dict. "
        f"Доступные ключи: {list(checkpoint.keys())}"
    )


def strip_common_prefixes(key: str) -> str:
    """
    Удаляет стандартные обёртки.

    Например:
        module.backbone.stem.0.weight
        -> stem.0.weight
    """

    prefixes = (
        "module.",
        "model.",
        "network.",
        "net.",
        "ema.",
        "ema_model.",
        "shared_backbone.",
        "backbone.",
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


def load_pretrained_weights(
    model: nn.Module,
    weights_path: Path,
    load_classifier_head: bool,
) -> None:
    checkpoint = load_torch_file(
        weights_path,
        map_location="cpu",
    )

    source_state = extract_state_dict(
        checkpoint
    )

    target_state = model.state_dict()

    mapped_state: dict[str, torch.Tensor] = {}
    used_source_keys: set[str] = set()

    normalized_source = {
        source_key: strip_common_prefixes(source_key)
        for source_key in source_state
    }

    for target_key, target_value in target_state.items():
        if (
            not load_classifier_head
            and target_key.startswith("head.")
        ):
            continue

        candidates: list[str] = []

        # Сначала точное совпадение после удаления префиксов.
        for source_key, normalized_key in normalized_source.items():
            if normalized_key == target_key:
                candidates.append(source_key)

        # Затем совпадение по полному суффиксу.
        if not candidates:
            suffix = "." + target_key

            for source_key, normalized_key in normalized_source.items():
                if normalized_key.endswith(suffix):
                    candidates.append(source_key)

        # Используем только однозначное совпадение правильной формы.
        shape_candidates = [
            source_key
            for source_key in candidates
            if tuple(source_state[source_key].shape)
            == tuple(target_value.shape)
        ]

        if len(shape_candidates) == 1:
            source_key = shape_candidates[0]

            mapped_state[target_key] = source_state[source_key]
            used_source_keys.add(source_key)

    result = model.load_state_dict(
        mapped_state,
        strict=False,
    )

    backbone_target_keys = [
        key
        for key in target_state
        if not key.startswith("head.")
    ]

    loaded_backbone_keys = [
        key
        for key in mapped_state
        if not key.startswith("head.")
    ]

    loaded_fraction = (
        len(loaded_backbone_keys)
        / max(len(backbone_target_keys), 1)
    )

    print("=" * 78)
    print(f"Pretrained weights: {weights_path.resolve()}")
    print(f"Source tensors: {len(source_state)}")
    print(f"Loaded tensors: {len(mapped_state)}")
    print(
        "Loaded backbone fraction: "
        f"{loaded_fraction:.1%}"
    )
    print(
        "Classifier head loaded: "
        f"{load_classifier_head}"
    )

    if result.missing_keys:
        print(
            f"Not loaded target tensors: "
            f"{len(result.missing_keys)}"
        )

        for key in result.missing_keys[:20]:
            print(f"  missing: {key}")

        if len(result.missing_keys) > 20:
            print("  ...")

    unused_source_count = (
        len(source_state)
        - len(used_source_keys)
    )

    print(
        f"Unused source tensors: "
        f"{unused_source_count}"
    )
    print("=" * 78)

    if loaded_fraction < MIN_BACKBONE_LOAD_FRACTION:
        raise RuntimeError(
            "Загружена слишком малая часть backbone: "
            f"{loaded_fraction:.1%}. "
            "Вероятно, архитектура или имена слоёв отличаются."
        )


# =============================================================================
# SAMPLER
# =============================================================================

def build_balanced_sampler(
    dataset: HelmetFolderDataset,
) -> WeightedRandomSampler:
    counts = dataset.class_counts()

    helmet_count = counts[0]
    no_helmet_count = counts[1]

    if helmet_count <= 0 or no_helmet_count <= 0:
        raise RuntimeError(
            f"В train нужны оба класса. Counts: {counts}"
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
        num_samples = len(dataset)
    else:
        num_samples = int(TRAIN_SAMPLES_PER_EPOCH)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=SAMPLER_REPLACEMENT,
        generator=generator,
    )


# =============================================================================
# OPTIMIZER И SCHEDULER
# =============================================================================

def split_parameter_groups(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    backbone_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("head."):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    if not backbone_parameters:
        raise RuntimeError("Backbone parameters not found")

    if not head_parameters:
        raise RuntimeError("Classifier head parameters not found")

    return backbone_parameters, head_parameters


def build_scheduler(
    optimizer: torch.optim.Optimizer,
) -> LambdaLR:
    """
    Warmup, затем cosine decay.

    Один и тот же коэффициент применяется к backbone LR и head LR,
    поэтому их относительное соотношение сохраняется.
    """

    cosine_epochs = max(
        EPOCHS - WARMUP_EPOCHS,
        1,
    )

    def lr_factor(epoch_index: int) -> float:
        if WARMUP_EPOCHS > 0 and epoch_index < WARMUP_EPOCHS:
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
            / cosine_epochs
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return (
            MIN_LR_FACTOR
            + 0.5
            * (1.0 - MIN_LR_FACTOR)
            * (
                1.0
                + math.cos(math.pi * progress)
            )
        )

    return LambdaLR(
        optimizer,
        lr_lambda=[
            lr_factor
            for _ in optimizer.param_groups
        ],
    )


# =============================================================================
# EXPORT
# =============================================================================

def export_best_weights(
    best_checkpoint_path: str | Path,
    destination: Path,
) -> None:
    checkpoint = load_torch_file(
        Path(best_checkpoint_path),
        map_location="cpu",
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    portable_state = {
        key: value.detach().cpu()
        for key, value in state_dict.items()
    }

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        portable_state,
        destination,
    )

    print(
        "Чистые EMA-веса экспортированы: "
        f"{destination.resolve()}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    set_global_seed(SEED)

    expected_metric = (
        f"fpr_at_recall_"
        f"{int(round(TARGET_RECALL * 100))}"
    )

    if BEST_METRIC != expected_metric:
        raise ValueError(
            f"BEST_METRIC должен быть {expected_metric!r}"
        )

    train_directory = DATASET_ROOT / "train"
    val_directory = DATASET_ROOT / "val"

    if not train_directory.exists():
        raise FileNotFoundError(
            f"Train directory not found: "
            f"{train_directory.resolve()}"
        )

    if not val_directory.exists():
        raise FileNotFoundError(
            f"Val directory not found: "
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

    train_counts = train_dataset.class_counts()
    val_counts = val_dataset.class_counts()

    sampler = build_balanced_sampler(
        train_dataset
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    pin_memory = (
        PIN_MEMORY
        and device.type == "cuda"
    )

    persistent_workers = (
        PERSISTENT_WORKERS
        and NUM_WORKERS > 0
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    val_generator = torch.Generator()
    val_generator.manual_seed(SEED + 1)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
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
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        generator=val_generator,
        drop_last=False,
    )

    # Сначала создаём свежую модель.
    model = HelmetMicroNeXt()

    # Загружаем backbone до переноса на GPU.
    load_pretrained_weights(
        model=model,
        weights_path=PRETRAINED_WEIGHTS,
        load_classifier_head=LOAD_CLASSIFIER_HEAD,
    )

    model = model.to(device)

    backbone_parameters, head_parameters = (
        split_parameter_groups(model)
    )

    optimizer = AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LEARNING_RATE,
                "name": "backbone",
            },
            {
                "params": head_parameters,
                "lr": HEAD_LEARNING_RATE,
                "name": "head",
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_scheduler(
        optimizer
    )

    criterion = nn.BCEWithLogitsLoss()

    print()
    print("=" * 78)
    print("CLASS 0 = helmet")
    print("CLASS 1 = no_helmet, positive class")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"Train: helmet={train_counts[0]}, "
        f"no_helmet={train_counts[1]}"
    )

    print(
        f"Val: helmet={val_counts[0]}, "
        f"no_helmet={val_counts[1]}"
    )

    print(
        f"Trainable parameters: "
        f"{count_trainable_parameters(model):,}"
    )

    print(
        f"Backbone LR: "
        f"{BACKBONE_LEARNING_RATE:.3e}"
    )

    print(
        f"Head LR: "
        f"{HEAD_LEARNING_RATE:.3e}"
    )

    print(
        "Balanced sampler: enabled, "
        "approximately 50/50"
    )

    print(
        "Selection: minimum FPR "
        f"at recall >= {TARGET_RECALL:.2f}"
    )

    print(
        f"train.py: "
        f"{Path(train_module.__file__).resolve()}"
    )

    print(
        f"train.py version: "
        f"{train_module.TRAIN_MODULE_VERSION}"
    )

    print("=" * 78)
    print()

    config = {
        "mode": "fine_tuning",
        "pretrained_weights": str(
            PRETRAINED_WEIGHTS.resolve()
        ),
        "load_classifier_head": (
            LOAD_CLASSIFIER_HEAD
        ),
        "dataset_root": str(
            DATASET_ROOT.resolve()
        ),
        "class_0": "helmet",
        "class_1": "no_helmet",
        "positive_class": 1,
        "train_class_counts": train_counts,
        "val_class_counts": val_counts,
        "image_size": IMAGE_SIZE,
        "train_mean": list(TRAIN_MEAN),
        "train_std": list(TRAIN_STD),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "backbone_learning_rate": (
            BACKBONE_LEARNING_RATE
        ),
        "head_learning_rate": (
            HEAD_LEARNING_RATE
        ),
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "ema_decay": EMA_DECAY,
        "target_recall": TARGET_RECALL,
        "best_metric": BEST_METRIC,
        "balanced_sampler": True,
        "amp": False,
    }

    result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=EPOCHS,
        output_directory=OUTPUT_DIRECTORY,
        ema_decay=EMA_DECAY,
        target_recall=TARGET_RECALL,
        max_grad_norm=MAX_GRAD_NORM,
        threshold=DECISION_THRESHOLD,
        best_metric=BEST_METRIC,
        early_stopping_patience=(
            EARLY_STOPPING_PATIENCE
        ),
        extra_config=config,
    )

    export_best_weights(
        best_checkpoint_path=result[
            "best_checkpoint"
        ],
        destination=EXPORTED_WEIGHTS,
    )


if __name__ == "__main__":
    main()
