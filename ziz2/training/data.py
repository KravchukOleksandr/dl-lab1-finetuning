"""Datasets, preprocessing and dataset×class balanced batch sampling."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
from typing import Iterator, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, get_worker_info


CLASS_TO_LABEL = {"helmet": 0, "no_helmet": 1}
PRIVATE_RGB_MEAN = np.asarray(
    [0.5226514935493469, 0.4618634879589081, 0.4670388400554657],
    dtype=np.float32,
).reshape(3, 1, 1)
PRIVATE_RGB_STD = np.asarray(
    [0.2495836615562439, 0.2355685979127884, 0.23572568595409393],
    dtype=np.float32,
).reshape(3, 1, 1)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@contextmanager
def _silenced_native_stderr() -> Iterator[None]:
    """Temporarily silence C-library warnings when no workers are used."""
    saved_stderr = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(devnull_fd)
        os.close(saved_stderr)


@dataclass(frozen=True)
class CropSample:
    path: Path
    dataset_name: str
    dataset_id: int
    class_name: str
    label: int

    @property
    def cell(self) -> tuple[int, int]:
        return self.dataset_id, self.label


def discover_samples(
    data_root: Path,
    split: str,
    dataset_names: Sequence[str],
) -> list[CropSample]:
    samples: list[CropSample] = []
    for dataset_id, dataset_name in enumerate(dataset_names):
        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = data_root / dataset_name / split / class_name
            if not class_dir.is_dir():
                raise ValueError(f"Expected dataset directory does not exist: {class_dir}")
            paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not paths:
                raise ValueError(f"No images found in {class_dir}")
            samples.extend(
                CropSample(
                    path=path,
                    dataset_name=dataset_name,
                    dataset_id=dataset_id,
                    class_name=class_name,
                    label=label,
                )
                for path in paths
            )
    return samples


def discover_single_dataset_samples(
    data_root: Path,
    split: str,
    dataset_name: str = "factory",
) -> list[CropSample]:
    """Discover <data_root>/<split>/<class> without a dataset subdirectory."""
    samples: list[CropSample] = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = data_root / split / class_name
        if not class_dir.is_dir():
            raise ValueError(f"Expected dataset directory does not exist: {class_dir}")
        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise ValueError(f"No images found in {class_dir}")
        samples.extend(
            CropSample(
                path=path,
                dataset_name=dataset_name,
                dataset_id=0,
                class_name=class_name,
                label=label,
            )
            for path in paths
        )
    return samples


def _square_crop(image: np.ndarray, train: bool) -> np.ndarray:
    height, width = image.shape[:2]
    max_side = min(height, width)
    if train:
        square_side = int(round(random.uniform(0.90, 1.00) * max_side))
        square_side = max(1, min(square_side, max_side))
        shift_limit = 0.03 * max_side
        center_x = width / 2.0 + random.uniform(-shift_limit, shift_limit)
        center_y = height / 2.0 + random.uniform(-shift_limit, shift_limit)
        left = int(round(center_x - square_side / 2.0))
        top = int(round(center_y - square_side / 2.0))
        left = min(max(left, 0), width - square_side)
        top = min(max(top, 0), height - square_side)
    else:
        square_side = max_side
        left = (width - square_side) // 2
        top = (height - square_side) // 2
    return image[top : top + square_side, left : left + square_side]


def _photometric_augmentation(image: np.ndarray) -> np.ndarray:
    if random.random() >= 0.50:
        return image

    operation = random.randrange(3)
    factor = random.uniform(0.90, 1.10)
    float_image = image.astype(np.float32)
    if operation == 0:  # brightness
        float_image *= factor
    elif operation == 1:  # contrast
        channel_mean = float_image.mean(axis=(0, 1), keepdims=True)
        float_image = (float_image - channel_mean) * factor + channel_mean
    else:  # gamma
        normalized = np.clip(float_image / 255.0, 0.0, 1.0)
        float_image = np.power(normalized, factor) * 255.0
    return np.clip(float_image, 0.0, 255.0).astype(np.uint8)


def preprocess_crop(image: np.ndarray, train: bool) -> torch.Tensor:
    image = _square_crop(image, train=train)
    if train and random.random() < 0.50:
        image = image[:, ::-1]

    source_side = image.shape[0]
    interpolation = cv2.INTER_AREA if source_side > 64 else cv2.INTER_LINEAR
    image = cv2.resize(image, (64, 64), interpolation=interpolation)
    if train:
        image = _photometric_augmentation(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    chw = (chw - PRIVATE_RGB_MEAN) / PRIVATE_RGB_STD
    return torch.from_numpy(np.ascontiguousarray(chw, dtype=np.float32))


class HelmetCropDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: Sequence[CropSample], train: bool) -> None:
        self.samples = list(samples)
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        if get_worker_info() is None:
            # With workers, stderr is silenced once in seed_worker instead.
            with _silenced_native_stderr():
                image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read image: {sample.path}")
        tensor = preprocess_crop(image, train=self.train)
        return (
            tensor,
            torch.tensor(sample.label, dtype=torch.float32),
            torch.tensor(sample.dataset_id, dtype=torch.int64),
        )


class DatasetClassBalancedBatchSampler(Sampler[list[int]]):
    """Yield dataset-weighted batches, balanced 50/50 by class per dataset.

    Each cell is shuffled without replacement. When a smaller cell is
    exhausted it is reshuffled and cycled, so every batch remains exactly
    balanced by class while avoiding immediate independent replacement
    sampling. Dataset weights control how many class pairs each dataset gets.
    """

    def __init__(
        self,
        samples: Sequence[CropSample],
        batch_size: int,
        seed: int,
        batches_per_epoch: int | None = None,
        dataset_weights: Mapping[str, float] | None = None,
    ) -> None:
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        dataset_names: dict[int, str] = {}
        for index, sample in enumerate(samples):
            groups[sample.cell].append(index)
            previous_name = dataset_names.setdefault(
                sample.dataset_id, sample.dataset_name
            )
            if previous_name != sample.dataset_name:
                raise ValueError(
                    f"dataset_id={sample.dataset_id} has multiple names"
                )
        self.groups = {cell: indices for cell, indices in sorted(groups.items())}
        if not self.groups:
            raise ValueError("Cannot balance an empty dataset")
        self.dataset_names = dict(sorted(dataset_names.items()))
        dataset_ids = list(self.dataset_names)
        for dataset_id in dataset_ids:
            for label in (0, 1):
                if (dataset_id, label) not in self.groups:
                    raise ValueError(
                        f"Missing class {label} for dataset "
                        f"{self.dataset_names[dataset_id]}"
                    )

        if batch_size <= 0 or batch_size % 2 != 0:
            raise ValueError(
                f"batch_size={batch_size} must be positive and divisible by 2"
            )
        pairs_per_batch = batch_size // 2
        if pairs_per_batch < len(dataset_ids):
            raise ValueError(
                "batch_size is too small: every dataset needs at least one "
                "helmet/no_helmet pair"
            )

        provided_weights = dict(dataset_weights or {})
        unknown_names = sorted(set(provided_weights) - set(self.dataset_names.values()))
        if unknown_names:
            raise ValueError(
                f"Weights provided for unknown datasets: {', '.join(unknown_names)}"
            )
        weights: dict[int, float] = {}
        for dataset_id, dataset_name in self.dataset_names.items():
            weight = float(provided_weights.get(dataset_name, 1.0))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"Dataset weight must be finite and positive: "
                    f"{dataset_name}={weight}"
                )
            weights[dataset_id] = weight

        weight_sum = sum(weights.values())
        raw_pairs = {
            dataset_id: pairs_per_batch * weight / weight_sum
            for dataset_id, weight in weights.items()
        }
        dataset_pairs = {
            dataset_id: math.floor(value)
            for dataset_id, value in raw_pairs.items()
        }
        pairs_left = pairs_per_batch - sum(dataset_pairs.values())
        remainder_order = sorted(
            dataset_ids,
            key=lambda dataset_id: (
                -(raw_pairs[dataset_id] - dataset_pairs[dataset_id]),
                dataset_id,
            ),
        )
        for dataset_id in remainder_order[:pairs_left]:
            dataset_pairs[dataset_id] += 1
        zero_pair_datasets = [
            self.dataset_names[dataset_id]
            for dataset_id, count in dataset_pairs.items()
            if count == 0
        ]
        if zero_pair_datasets:
            raise ValueError(
                "These dataset weights are too small for the selected batch size: "
                f"{', '.join(zero_pair_datasets)}. Increase batch_size or weight."
            )

        self.batch_size = batch_size
        self.dataset_weights = {
            self.dataset_names[dataset_id]: weights[dataset_id]
            for dataset_id in dataset_ids
        }
        self.cell_batch_counts = {
            (dataset_id, label): dataset_pairs[dataset_id]
            for dataset_id in dataset_ids
            for label in (0, 1)
        }
        unique_cell_counts = set(self.cell_batch_counts.values())
        self.per_cell = (
            next(iter(unique_cell_counts)) if len(unique_cell_counts) == 1 else None
        )
        self.seed = seed
        self.epoch = 0
        self.batches_per_epoch = (
            batches_per_epoch
            if batches_per_epoch is not None
            else math.ceil(len(samples) / batch_size)
        )
        if self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        orders: dict[tuple[int, int], list[int]] = {}
        positions: dict[tuple[int, int], int] = {}

        def reshuffle(cell: tuple[int, int]) -> None:
            order = list(self.groups[cell])
            rng.shuffle(order)
            orders[cell] = order
            positions[cell] = 0

        for cell in self.groups:
            reshuffle(cell)

        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for cell in self.groups:
                needed = self.cell_batch_counts[cell]
                while needed > 0:
                    position = positions[cell]
                    order = orders[cell]
                    available = len(order) - position
                    take = min(needed, available)
                    batch.extend(order[position : position + take])
                    positions[cell] += take
                    needed -= take
                    if positions[cell] == len(order):
                        reshuffle(cell)
            rng.shuffle(batch)
            yield batch


def seed_worker(worker_id: int) -> None:
    del worker_id
    # libpng writes malformed ICC-profile warnings directly to the native
    # stderr file descriptor, bypassing Python warnings and OpenCV logging.
    # DataLoader exceptions still reach the parent process through its result
    # queue, so real image-reading failures remain visible with a traceback.
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
    finally:
        os.close(devnull_fd)

    try:
        cv2.setLogLevel(0)
    except AttributeError:
        pass

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def cell_counts(samples: Sequence[CropSample]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for sample in samples:
        counts[(sample.dataset_name, sample.class_name)] += 1
    return dict(counts)
