"""
Fine-tuning предобученной HelmetMicroNeXt.

Классы:
    0 = helmet     — каска есть
    1 = no_helmet  — каски нет, положительный класс

Train:
    WeightedRandomSampler, примерно 50/50.
    BCEWithLogitsLoss без pos_weight.

Validation:
    каждый пример используется ровно один раз.

Основная метрика:
    минимальный FPR при recall(no_helmet) >= 0.95.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
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
from model import (
    HelmetMicroNeXt,
    count_trainable_parameters,
)
from preprocessing import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    HelmetFolderDataset,
    HelmetPreprocessor,
)
from train import fit


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

DATASET_ROOT = Path(
    "./ziz-crops-202607-2"
)

# Предобученная модель или полный checkpoint.
PRETRAINED_WEIGHTS = Path(
    "./models/pretrained.pt"
)

OUTPUT_DIRECTORY = Path(
    "./runs/helmet_finetune_v2"
)

# Сюда после обучения будут экспортированы только EMA-веса.
EXPORTED_WEIGHTS = Path(
    "./models/helmet_finetuned_v2.pt"
)


# -----------------------------------------------------------------------------
# Режим загрузки pretrain
# -----------------------------------------------------------------------------

# Варианты:
#
# "full_backbone":
#     загружаются stem, stage1, down, stage2;
#     proj и head остаются новыми.
#
# "early_backbone":
#     загружаются только stem и stage1;
#     down, stage2, proj и head остаются новыми.
#
# "all_compatible":
#     загружаются все совпадающие слои, включая proj и head.
#
PRETRAIN_LOAD_MODE = "full_backbone"

MIN_LOADED_PARAMETER_FRACTION = 0.70


# -----------------------------------------------------------------------------
# Какие части модели считаются классификационной головой
# -----------------------------------------------------------------------------

HEAD_PREFIXES = (
    "proj.",
    "head.",
)

EARLY_BACKBONE_PREFIXES = (
    "stem.",
    "stage1.",
)


# -----------------------------------------------------------------------------
# Вход и preprocessing
# -----------------------------------------------------------------------------

IMAGE_SIZE = 64

TRAIN_MEAN = DEFAULT_MEAN
TRAIN_STD = DEFAULT_STD

SQUARE_SCALE = (
    0.90,
    1.00,
)

CENTER_JITTER = 0.03
HORIZONTAL_FLIP_PROBABILITY = 0.50

PHOTOMETRIC_PROBABILITY = 0.50

BRIGHTNESS_LIMIT = 0.10
CONTRAST_LIMIT = 0.10

GAMMA_RANGE = (
    0.90,
    1.10,
)


# -----------------------------------------------------------------------------
# Fine-tuning
# -----------------------------------------------------------------------------

SEED = 42

EPOCHS = 40
BATCH_SIZE = 128
NUM_WORKERS = 4

# Backbone должен заметно перестроиться под ваши камеры.
BACKBONE_LEARNING_RATE = 5.0e-4

# Новая классификационная голова обучается быстрее.
HEAD_LEARNING_RATE = 1.0e-3

# Конечный LR:
#
# backbone: 5e-4 * 0.02 = 1e-5
# head:     1e-3 * 0.02 = 2e-5
MIN_LR_FACTOR = 0.02

WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.20

WEIGHT_DECAY = 5.0e-4
MAX_GRAD_NORM = 5.0

EMA_DECAY = 0.997

TARGET_RECALL = 0.95
DECISION_THRESHOLD = 0.50

BEST_METRIC = "fpr_at_recall_95"

# Проходим все 40 эпох.
EARLY_STOPPING_PATIENCE = None


# -----------------------------------------------------------------------------
# Balanced sampler
# -----------------------------------------------------------------------------

SAMPLER_REPLACEMENT = True

# None = len(train_dataset).
TRAIN_SAMPLES_PER_EPOCH: int | None = None


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------

PIN_MEMORY = True
PERSISTENT_WORKERS = True


# =============================================================================
# ВОСПРОИЗВОДИМОСТЬ
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


# =============================================================================
# ЗАГРУЗКА CHECKPOINT
# =============================================================================

def load_torch_file(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Файл весов не найден: "
            f"{path.resolve()}"
        )

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        # Старые версии PyTorch.
        return torch.load(
            path,
            map_location=map_location,
        )


def looks_like_state_dict(
    value: Any,
) -> bool:
    if not isinstance(
        value,
        Mapping,
    ):
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
    """
    Поддерживает:

        torch.save(model.state_dict(), path)

    и checkpoints с ключами:

        model_state_dict
        raw_model_state_dict
        state_dict
        ema_state_dict
        weights
        model
    """

    if looks_like_state_dict(
        checkpoint
    ):
        print(
            "State dict source: root object"
        )

        return dict(checkpoint)

    if not isinstance(
        checkpoint,
        Mapping,
    ):
        raise TypeError(
            "Неподдерживаемый формат checkpoint: "
            f"{type(checkpoint)}"
        )

    preferred_keys = (
        "model_state_dict",
        "raw_model_state_dict",
        "state_dict",
        "ema_state_dict",
        "weights",
        "model",
    )

    for key in preferred_keys:
        value = checkpoint.get(key)

        if looks_like_state_dict(value):
            print(
                f"State dict source: "
                f"checkpoint[{key!r}]"
            )

            return dict(value)

    raise KeyError(
        "State dict не найден.\n"
        f"Ключи checkpoint: "
        f"{list(checkpoint.keys())}"
    )


def normalize_source_key(
    key: str,
) -> str:
    """
    Удаляет стандартные префиксы.

    Например:

        module.model.backbone.stem.0.weight

    превращается в:

        stem.0.weight
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
                key = key[
                    len(prefix):
                ]

                changed = True
                break

    return key


def target_key_is_allowed(
    target_key: str,
) -> bool:
    if PRETRAIN_LOAD_MODE == "full_backbone":
        return not target_key.startswith(
            HEAD_PREFIXES
        )

    if PRETRAIN_LOAD_MODE == "early_backbone":
        return target_key.startswith(
            EARLY_BACKBONE_PREFIXES
        )

    if PRETRAIN_LOAD_MODE == "all_compatible":
        return True

    raise ValueError(
        "Неизвестный PRETRAIN_LOAD_MODE: "
        f"{PRETRAIN_LOAD_MODE!r}"
    )


def load_pretrained_weights(
    model: nn.Module,
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

    source_by_normalized_key: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for source_key in source_state:
        normalized_key = (
            normalize_source_key(
                source_key
            )
        )

        source_by_normalized_key[
            normalized_key
        ].append(source_key)

    mapped_state: dict[
        str,
        torch.Tensor,
    ] = {}

    used_source_keys: set[str] = set()

    allowed_target_numel = 0
    loaded_target_numel = 0

    for target_key, target_tensor in target_state.items():
        if not target_key_is_allowed(
            target_key
        ):
            continue

        allowed_target_numel += (
            target_tensor.numel()
        )

        candidates = list(
            source_by_normalized_key.get(
                target_key,
                [],
            )
        )

        # Дополнительный поиск по суффиксу.
        if not candidates:
            target_suffix = (
                "." + target_key
            )

            for (
                normalized_key,
                source_keys,
            ) in source_by_normalized_key.items():
                if normalized_key.endswith(
                    target_suffix
                ):
                    candidates.extend(
                        source_keys
                    )

        shape_candidates = [
            source_key
            for source_key in candidates
            if tuple(
                source_state[source_key].shape
            )
            == tuple(target_tensor.shape)
        ]

        if len(shape_candidates) != 1:
            continue

        source_key = shape_candidates[0]

        mapped_state[target_key] = (
            source_state[source_key]
        )

        used_source_keys.add(
            source_key
        )

        loaded_target_numel += (
            target_tensor.numel()
        )

    load_result = model.load_state_dict(
        mapped_state,
        strict=False,
    )

    loaded_fraction = (
        loaded_target_numel
        / max(allowed_target_numel, 1)
    )

    print("=" * 78)

    print(
        f"Pretrained weights: "
        f"{weights_path.resolve()}"
    )

    print(
        f"Load mode: "
        f"{PRETRAIN_LOAD_MODE}"
    )

    print(
        f"Source tensors: "
        f"{len(source_state)}"
    )

    print(
        f"Loaded tensors: "
        f"{len(mapped_state)}"
    )

    print(
        "Loaded compatible parameter fraction: "
        f"{loaded_fraction:.2%}"
    )

    print(
        f"Unused source tensors: "
        f"{len(source_state) - len(used_source_keys)}"
    )

    missing_relevant_keys = [
        key
        for key in load_result.missing_keys
        if target_key_is_allowed(key)
        and not key.endswith(
            "num_batches_tracked"
        )
    ]

    if missing_relevant_keys:
        print(
            "Не загруженные совместимые слои:"
        )

        for key in missing_relevant_keys[:30]:
            print(
                f"  {key}"
            )

        if len(
            missing_relevant_keys
        ) > 30:
            print("  ...")

    print("=" * 78)

    if (
        loaded_fraction
        < MIN_LOADED_PARAMETER_FRACTION
    ):
        raise RuntimeError(
            "Загружена слишком малая часть "
            "выбранного backbone: "
            f"{loaded_fraction:.2%}.\n"
            "Проверь архитектуру и названия слоёв."
        )


# =============================================================================
# BATCHNORM
# =============================================================================

def reset_batchnorm_running_stats(
    model: nn.Module,
) -> int:
    """
    Сбрасывает только running statistics:

        running_mean
        running_var
        num_batches_tracked

    Обучаемые параметры BatchNorm:
        weight
        bias

    сохраняются из pretrain.
    """

    reset_count = 0

    for module in model.modules():
        if isinstance(
            module,
            nn.BatchNorm2d,
        ):
            module.reset_running_stats()
            reset_count += 1

    print(
        f"BatchNorm running statistics reset: "
        f"{reset_count} layers"
    )

    return reset_count


# =============================================================================
# BALANCED SAMPLER
# =============================================================================

def build_balanced_sampler(
    dataset: HelmetFolderDataset,
) -> WeightedRandomSampler:
    counts = dataset.class_counts()

    helmet_count = counts[0]
    no_helmet_count = counts[1]

    if (
        helmet_count <= 0
        or no_helmet_count <= 0
    ):
        raise RuntimeError(
            "В train должны присутствовать "
            f"оба класса. Counts: {counts}"
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
        num_samples = int(
            TRAIN_SAMPLES_PER_EPOCH
        )

    if num_samples <= 0:
        raise ValueError(
            "TRAIN_SAMPLES_PER_EPOCH должен "
            "быть положительным или None"
        )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=SAMPLER_REPLACEMENT,
        generator=generator,
    )


# =============================================================================
# OPTIMIZER
# =============================================================================

def is_head_parameter(
    parameter_name: str,
) -> bool:
    return parameter_name.startswith(
        HEAD_PREFIXES
    )


def split_parameter_groups(
    model: nn.Module,
) -> tuple[
    list[nn.Parameter],
    list[nn.Parameter],
]:
    backbone_parameters: list[
        nn.Parameter
    ] = []

    head_parameters: list[
        nn.Parameter
    ] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if is_head_parameter(name):
            head_parameters.append(
                parameter
            )
        else:
            backbone_parameters.append(
                parameter
            )

    if not backbone_parameters:
        raise RuntimeError(
            "Backbone parameters not found"
        )

    if not head_parameters:
        raise RuntimeError(
            "Head parameters not found. "
            f"Expected prefixes: {HEAD_PREFIXES}"
        )

    return (
        backbone_parameters,
        head_parameters,
    )


# =============================================================================
# SCHEDULER
# =============================================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
) -> LambdaLR:
    """
    Три эпохи warmup, затем cosine decay.

    Один коэффициент применяется ко всем parameter groups,
    поэтому отношение backbone LR к head LR сохраняется.
    """

    cosine_epochs = max(
        EPOCHS - WARMUP_EPOCHS,
        1,
    )

    def learning_rate_factor(
        epoch_index: int,
    ) -> float:
        if (
            WARMUP_EPOCHS > 0
            and epoch_index < WARMUP_EPOCHS
        ):
            if WARMUP_EPOCHS == 1:
                return 1.0

            warmup_progress = (
                epoch_index
                / (WARMUP_EPOCHS - 1)
            )

            return (
                WARMUP_START_FACTOR
                + (
                    1.0
                    - WARMUP_START_FACTOR
                )
                * warmup_progress
            )

        cosine_step = (
            epoch_index
            - WARMUP_EPOCHS
            + 1
        )

        cosine_progress = (
            cosine_step
            / cosine_epochs
        )

        cosine_progress = min(
            max(cosine_progress, 0.0),
            1.0,
        )

        cosine_value = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * cosine_progress
                )
            )
        )

        return (
            MIN_LR_FACTOR
            + (
                1.0
                - MIN_LR_FACTOR
            )
            * cosine_value
        )

    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=[
            learning_rate_factor
            for _ in optimizer.param_groups
        ],
    )


# =============================================================================
# EXPORT
# =============================================================================

def export_best_ema_weights(
    checkpoint_path: str | Path,
    output_path: Path,
) -> None:
    checkpoint = load_torch_file(
        Path(checkpoint_path),
        map_location="cpu",
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    cpu_state_dict = {
        key: tensor.detach().cpu()
        for key, tensor in state_dict.items()
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        cpu_state_dict,
        output_path,
    )

    print(
        "EMA weights exported to: "
        f"{output_path.resolve()}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    set_global_seed(SEED)

    expected_best_metric = (
        "fpr_at_recall_"
        f"{int(round(TARGET_RECALL * 100))}"
    )

    if BEST_METRIC != expected_best_metric:
        raise ValueError(
            f"BEST_METRIC должен быть "
            f"{expected_best_metric!r}"
        )

    if not (
        0 <= WARMUP_EPOCHS < EPOCHS
    ):
        raise ValueError(
            "WARMUP_EPOCHS должен быть "
            "меньше EPOCHS"
        )

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
        split_directory=train_directory,
        preprocessor=train_preprocessor,
    )

    val_dataset = HelmetFolderDataset(
        split_directory=val_directory,
        preprocessor=val_preprocessor,
    )

    train_counts = (
        train_dataset.class_counts()
    )

    val_counts = (
        val_dataset.class_counts()
    )

    train_sampler = (
        build_balanced_sampler(
            train_dataset
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    effective_pin_memory = (
        PIN_MEMORY
        and device.type == "cuda"
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

        sampler=train_sampler,
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

    # Свежая модель: незагруженные слои автоматически
    # остаются со случайной инициализацией.
    model = HelmetMicroNeXt()

    load_pretrained_weights(
        model=model,
        weights_path=PRETRAINED_WEIGHTS,
    )

    # Не переносим running statistics внешних датасетов.
    reset_batchnorm_running_stats(
        model
    )

    model = model.to(device)

    (
        backbone_parameters,
        head_parameters,
    ) = split_parameter_groups(
        model
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

    print(
        "CLASS 0 = helmet"
    )

    print(
        "CLASS 1 = no_helmet "
        "(positive class)"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Train: helmet={train_counts[0]}, "
        f"no_helmet={train_counts[1]}"
    )

    print(
        f"Validation: helmet={val_counts[0]}, "
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
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        "Balanced sampler: enabled, "
        "approximately 50/50"
    )

    print(
        "Loss: BCEWithLogitsLoss "
        "without pos_weight"
    )

    print(
        "Selection: minimum validation FPR "
        f"at recall >= {TARGET_RECALL:.2f}"
    )

    print(
        f"EMA decay: {EMA_DECAY}"
    )

    print(
        f"train.py: "
        f"{Path(train_module.__file__).resolve()}"
    )

    print(
        f"train.py version: "
        f"{train_module.TRAIN_MODULE_VERSION}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIRECTORY.resolve()}"
    )

    print("=" * 78)
    print()

    config = {
        "mode": "fine_tuning",

        "pretrained_weights": str(
            PRETRAINED_WEIGHTS.resolve()
        ),

        "pretrain_load_mode": (
            PRETRAIN_LOAD_MODE
        ),

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

        "backbone_learning_rate": (
            BACKBONE_LEARNING_RATE
        ),

        "head_learning_rate": (
            HEAD_LEARNING_RATE
        ),

        "minimum_lr_factor": (
            MIN_LR_FACTOR
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

        "target_recall": (
            TARGET_RECALL
        ),

        "best_metric": (
            BEST_METRIC
        ),

        "balanced_sampler": True,
        "sampler_replacement": (
            SAMPLER_REPLACEMENT
        ),

        "batchnorm_running_stats_reset": (
            True
        ),

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

    export_best_ema_weights(
        checkpoint_path=result[
            "best_checkpoint"
        ],
        output_path=EXPORTED_WEIGHTS,
    )

    print(
        f"Best validation "
        f"{BEST_METRIC}: "
        f"{result['best_value']:.6f}"
    )

    print(
        f"Best validation threshold: "
        f"{result['best_threshold']:.9f}"
    )


if __name__ == "__main__":
    main()
