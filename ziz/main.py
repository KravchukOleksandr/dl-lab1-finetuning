"""
Train HelmetMicroNeXt.

IMPORTANT CLASS MAPPING:
    0 = helmet     = каска есть
    1 = no_helmet  = каски нет (positive class for precision/recall/F1/AUC)

Edit the constants in the CONFIG section below, then run:
    python main.py
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler

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
# CONFIG: edit these values for your machine and experiments
# =============================================================================
DATASET_ROOT = Path(r"./ziz-crops-202607-2")
OUTPUT_DIRECTORY = Path(r"./runs/helmet_micronext_v1")

# Data
IMAGE_SIZE = 64
TRAIN_MEAN = DEFAULT_MEAN
TRAIN_STD = DEFAULT_STD

# Mild train geometry. Input files are already expanded x1.25.
SQUARE_SCALE = (0.90, 1.00)
CENTER_JITTER = 0.03
HORIZONTAL_FLIP_PROBABILITY = 0.50

# Mild photometric augmentation: with probability 0.50 apply exactly one.
PHOTOMETRIC_PROBABILITY = 0.50
BRIGHTNESS_LIMIT = 0.10
CONTRAST_LIMIT = 0.10
GAMMA_RANGE = (0.90, 1.10)

# Training
SEED = 42
EPOCHS = 80
BATCH_SIZE = 64
NUM_WORKERS = 4
BASE_LEARNING_RATE = 2.0e-3
MIN_LEARNING_RATE = 1.0e-5
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1.0e-4
MAX_GRAD_NORM = 5.0
USE_AMP = True

# Metrics/checkpointing. Threshold 0.5 is only the initial operating point.
DECISION_THRESHOLD = 0.50
BEST_METRIC = "auc"  # one of: loss, accuracy, precision, recall, f1, auc
EARLY_STOPPING_PATIENCE = 20  # set None to disable

# System
PIN_MEMORY = True
PERSISTENT_WORKERS = True


# =============================================================================
# Helpers
# =============================================================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is preferred for the baseline. It may slightly reduce speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_balanced_sampler(dataset: HelmetFolderDataset) -> WeightedRandomSampler:
    counts = dataset.class_counts()
    if counts[0] == 0 or counts[1] == 0:
        raise RuntimeError(f"Both classes are required in train. Counts: {counts}")

    class_weights = {
        0: 1.0 / counts[0],
        1: 1.0 / counts[1],
    }
    sample_weights = torch.tensor(
        [class_weights[target] for target in dataset.targets],
        dtype=torch.double,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    cosine_epochs = max(1, EPOCHS - WARMUP_EPOCHS)

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=MIN_LEARNING_RATE,
    )

    if WARMUP_EPOCHS <= 0:
        return cosine

    warmup = LinearLR(
        optimizer,
        start_factor=0.20,
        end_factor=1.00,
        total_iters=WARMUP_EPOCHS,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[WARMUP_EPOCHS],
    )


def print_dataset_summary(name: str, dataset: HelmetFolderDataset) -> None:
    counts = dataset.class_counts()
    print(f"{name}: {len(dataset)} images")
    print(f"  class 0 ({CLASS_NAMES[0]}): {counts[0]}")
    print(f"  class 1 ({CLASS_NAMES[1]}): {counts[1]}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("=" * 78)
    print("HelmetMicroNeXt binary classification")
    print("CLASS 0 = helmet     = каска есть")
    print("CLASS 1 = no_helmet  = каски нет (positive class for metrics)")
    print("=" * 78)

    set_global_seed(SEED)

    train_directory = DATASET_ROOT / "train"
    val_directory = DATASET_ROOT / "val"

    train_preprocessor = HelmetPreprocessor(
        training=True,
        output_size=IMAGE_SIZE,
        mean=TRAIN_MEAN,
        std=TRAIN_STD,
        square_scale=SQUARE_SCALE,
        center_jitter=CENTER_JITTER,
        horizontal_flip_probability=HORIZONTAL_FLIP_PROBABILITY,
        photometric_probability=PHOTOMETRIC_PROBABILITY,
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

    train_dataset = HelmetFolderDataset(train_directory, train_preprocessor)
    val_dataset = HelmetFolderDataset(val_directory, val_preprocessor)

    print_dataset_summary("Train", train_dataset)
    print_dataset_summary("Validation", val_dataset)
    print(f"Input mean RGB: {list(TRAIN_MEAN)}")
    print(f"Input std  RGB: {list(TRAIN_STD)}")

    sampler = build_balanced_sampler(train_dataset)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)

    effective_persistent_workers = PERSISTENT_WORKERS and NUM_WORKERS > 0
    effective_pin_memory = PIN_MEMORY and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_persistent_workers,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_persistent_workers,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HelmetMicroNeXt().to(device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Base LR: {BASE_LEARNING_RATE:.3e}")
    print(f"Warmup epochs: {WARMUP_EPOCHS}")
    print(f"Minimum LR: {MIN_LEARNING_RATE:.3e}")
    print(f"Output directory: {OUTPUT_DIRECTORY.resolve()}\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=BASE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = build_scheduler(optimizer)

    config_for_checkpoint = {
        "dataset_root": str(DATASET_ROOT),
        "image_size": IMAGE_SIZE,
        "class_0": "helmet",
        "class_1": "no_helmet",
        "train_mean": list(TRAIN_MEAN),
        "train_std": list(TRAIN_STD),
        "square_scale": list(SQUARE_SCALE),
        "center_jitter": CENTER_JITTER,
        "horizontal_flip_probability": HORIZONTAL_FLIP_PROBABILITY,
        "photometric_probability": PHOTOMETRIC_PROBABILITY,
        "brightness_limit": BRIGHTNESS_LIMIT,
        "contrast_limit": CONTRAST_LIMIT,
        "gamma_range": list(GAMMA_RANGE),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "base_learning_rate": BASE_LEARNING_RATE,
        "minimum_learning_rate": MIN_LEARNING_RATE,
        "warmup_epochs": WARMUP_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
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
        use_amp=USE_AMP,
        max_grad_norm=MAX_GRAD_NORM,
        threshold=DECISION_THRESHOLD,
        best_metric=BEST_METRIC,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        extra_config=config_for_checkpoint,
    )


if __name__ == "__main__":
    # Needed for Windows DataLoader workers.
    main()
