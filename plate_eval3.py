from __future__ import annotations

from collections import Counter
from pathlib import Path

from fast_alpr.default_ocr import DefaultOCR

from plate_resolver import PlateResolver
from ua import UA_PLATE_FORMAT


# =========================
# Configuration
# =========================

DATASET_DIRS = [
    Path("data/ukr_plates/batch_001"),
    Path("data/ukr_plates/batch_002"),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OCR_MODEL_PATH = Path("models/cct-s-v2-global-model/cct_s_v2_global.onnx")
OCR_CONFIG_PATH = Path("models/cct-s-v2-global-model/cct_s_v2_global_plate_config.yaml")

OCR_PROVIDERS = ["CPUExecutionProvider"]
OCR_DEVICE = "auto"


def iter_images(directory: Path) -> list[Path]:
    """
    Collect image files from a directory recursively.

    Args:
        directory: Root directory with cropped plate images.

    Returns:
        Sorted list of image paths.
    """
    images: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def extract_ground_truth(image_path: Path) -> str:
    """
    Extract the ground-truth plate string from the filename.

    This evaluator assumes that the filename stem is the plate itself:
        AC0001MM.png -> AC0001MM

    Args:
        image_path: Path to a cropped plate image.

    Returns:
        Ground-truth plate string.
    """
    stem = image_path.stem.upper()
    return "".join(ch for ch in stem if ch.isalnum())


def compare_prediction(gt: str, pred: str) -> tuple[bool, bool, list[tuple[str, str]]]:
    """
    Compare prediction with ground truth.

    Args:
        gt: Ground-truth plate.
        pred: Predicted plate.

    Returns:
        A tuple:
            - exact_match
            - wrong_length
            - substitutions as (expected_char, predicted_char)
    """
    if gt == pred:
        return True, False, []

    if len(gt) != len(pred):
        return False, True, []

    substitutions: list[tuple[str, str]] = []
    for gt_ch, pred_ch in zip(gt, pred):
        if gt_ch != pred_ch:
            substitutions.append((gt_ch, pred_ch))

    return False, False, substitutions


def init_stats() -> dict:
    """
    Create an empty statistics container.

    Returns:
        Stats dictionary.
    """
    return {
        "total": 0,
        "exact": 0,
        "no_prediction": 0,
        "wrong_length": 0,
        "substitution_samples": 0,
        "substitution_counter": Counter(),
        "error_type_counter": Counter(),
        "failed_samples": [],
    }


def update_stats(stats: dict, image_path: Path, gt: str, pred: str | None) -> None:
    """
    Update evaluation statistics for one sample.

    Args:
        stats: Statistics container.
        image_path: Current image path.
        gt: Ground-truth plate.
        pred: Predicted plate or None.
    """
    stats["total"] += 1

    if pred is None:
        stats["no_prediction"] += 1
        stats["error_type_counter"]["no_prediction"] += 1
        stats["failed_samples"].append((image_path, gt, None, "no_prediction"))
        return

    exact_match, wrong_length, substitutions = compare_prediction(gt, pred)

    if exact_match:
        stats["exact"] += 1
        return

    if wrong_length:
        stats["wrong_length"] += 1
        stats["error_type_counter"]["wrong_length"] += 1
        stats["failed_samples"].append((image_path, gt, pred, "wrong_length"))
        return

    stats["substitution_samples"] += 1
    stats["error_type_counter"]["substitution"] += 1
    stats["failed_samples"].append((image_path, gt, pred, "substitution"))

    for sub in substitutions:
        stats["substitution_counter"][sub] += 1


def merge_stats(dst: dict, src: dict) -> None:
    """
    Merge one stats dictionary into another.

    Args:
        dst: Destination stats.
        src: Source stats.
    """
    dst["total"] += src["total"]
    dst["exact"] += src["exact"]
    dst["no_prediction"] += src["no_prediction"]
    dst["wrong_length"] += src["wrong_length"]
    dst["substitution_samples"] += src["substitution_samples"]
    dst["substitution_counter"].update(src["substitution_counter"])
    dst["error_type_counter"].update(src["error_type_counter"])
    dst["failed_samples"].extend(src["failed_samples"])


def print_stats(title: str, stats: dict, max_failed_samples: int = 30) -> None:
    """
    Print evaluation statistics.

    Args:
        title: Section title.
        stats: Statistics dictionary.
        max_failed_samples: Maximum number of failed samples to print.
    """
    total = stats["total"]
    exact = stats["exact"]
    accuracy = exact / total if total else 0.0

    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total images:                {total}")
    print(f"Exact matches:              {exact}")
    print(f"Accuracy:                   {accuracy:.4%}")
    print(f"No prediction:              {stats['no_prediction']}")
    print(f"Wrong length:               {stats['wrong_length']}")
    print(f"Samples with substitutions: {stats['substitution_samples']}")
    print(f"Total substitutions:        {sum(stats['substitution_counter'].values())}")
    print()

    print("Error summary:")
    if not stats["error_type_counter"]:
        print("  No errors")
    else:
        for err_type, count in stats["error_type_counter"].most_common():
            print(f"  {err_type:15s} {count}")
    print()

    print("Character substitution statistics:")
    if not stats["substitution_counter"]:
        print("  No substitution errors")
    else:
        for (expected, predicted), count in stats["substitution_counter"].most_common():
            print(f"  {expected} -> {predicted}: {count}")
    print()

    print("Failed samples:")
    if not stats["failed_samples"]:
        print("  No failed samples")
    else:
        for image_path, gt, pred, err_type in stats["failed_samples"][:max_failed_samples]:
            print(
                f"  {err_type:15s} | "
                f"gt={gt:10s} | "
                f"pred={str(pred):10s} | "
                f"path={image_path}"
            )
    print()


def main() -> None:
    """
    Run OCR-only evaluation on cropped Ukrainian plate images.
    """
    for path in (OCR_MODEL_PATH, OCR_CONFIG_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    ocr = DefaultOCR(
        hub_ocr_model=None,
        device=OCR_DEVICE,
        providers=OCR_PROVIDERS,
        model_path=str(OCR_MODEL_PATH),
        config_path=str(OCR_CONFIG_PATH),
        force_download=False,
    )
    resolver = PlateResolver(UA_PLATE_FORMAT)

    overall_stats = init_stats()
    per_dataset_stats: dict[Path, dict] = {}

    for dataset_dir in DATASET_DIRS:
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        dataset_stats = init_stats()
        images = iter_images(dataset_dir)

        for image_path in images:
            gt = extract_ground_truth(image_path)

            ocr_result = ocr.predict(str(image_path))
            raw_text = None if ocr_result is None else ocr_result.text

            pred = None
            if raw_text:
                normalized = resolver.normalize(raw_text)
                if normalized is not None:
                    pred = normalized[0]

            update_stats(dataset_stats, image_path, gt, pred)

        per_dataset_stats[dataset_dir] = dataset_stats
        merge_stats(overall_stats, dataset_stats)

    for dataset_dir, stats in per_dataset_stats.items():
        print_stats(f"Dataset: {dataset_dir}", stats)

    print_stats("Overall", overall_stats, max_failed_samples=50)


if __name__ == "__main__":
    main()
