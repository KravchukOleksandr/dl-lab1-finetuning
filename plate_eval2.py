from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path


from plate_recognizer import PlateRecognizer
from ua import UA_PLATE_FORMAT


# =========================
# Configuration
# =========================

DATASET_DIRS = [
    Path("data/ukr_plates/batch_001"),
    Path("data/ukr_plates/batch_002"),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DETECTOR_MODEL_PATH = Path("models/yolo-v9-s-608-license-plates-end2end.onnx")
OCR_MODEL_PATH = Path("models/cct-s-v2-global-model/cct_s_v2_global.onnx")
OCR_CONFIG_PATH = Path("models/cct-s-v2-global-model/cct_s_v2_global_plate_config.yaml")

DETECTOR_CONF_THRESH = 0.0001
DETECTOR_PROVIDERS = ["CPUExecutionProvider"]
OCR_PROVIDERS = ["CPUExecutionProvider"]
OCR_DEVICE = "auto"


def iter_images(directory: Path) -> list[Path]:
    """
    Collect image files from a dataset directory recursively.

    Args:
        directory:
            Root directory containing cropped plate images.

    Returns:
        A sorted list of image paths.
    """
    images: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def extract_ground_truth(image_path: Path) -> str:
    """
    Extract the ground-truth plate string from the file name.

    This evaluator assumes that the file stem is the plate itself, for example:
        AC0001MM.png -> AC0001MM

    Non-alphanumeric characters are removed and the result is uppercased.

    Args:
        image_path:
            Path to a cropped plate image.

    Returns:
        Ground-truth plate string.
    """
    stem = image_path.stem.upper()
    return "".join(ch for ch in stem if ch.isalnum())


def compare_prediction(gt: str, pred: str) -> tuple[bool, bool, list[tuple[str, str]]]:
    """
    Compare a prediction with the ground truth.

    Args:
        gt:
            Ground-truth plate string.
        pred:
            Predicted plate string.

    Returns:
        A tuple:
            - exact_match:
                True if prediction exactly matches ground truth.
            - wrong_length:
                True if prediction length differs from ground truth length.
            - substitutions:
                A list of per-position character substitutions in the form
                (expected_char, predicted_char). This list is only filled when
                lengths are equal and the strings differ.
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


def make_recognizer() -> PlateRecognizer:
    """
    Build the recognizer used for evaluation.

    Returns:
        A configured PlateRecognizer instance.
    """
    return PlateRecognizer(
        plate_format=UA_PLATE_FORMAT,
        detector_model_path=DETECTOR_MODEL_PATH,
        ocr_model_path=OCR_MODEL_PATH,
        ocr_config_path=OCR_CONFIG_PATH,
        detector_conf_thresh=DETECTOR_CONF_THRESH,
        detector_providers=DETECTOR_PROVIDERS,
        ocr_providers=OCR_PROVIDERS,
        ocr_device=OCR_DEVICE,
    )


def init_stats() -> dict:
    """
    Create an empty statistics container.

    Returns:
        A dict with counters and error collections.
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
        stats:
            Statistics dict created by `init_stats()`.
        image_path:
            Path to the processed image.
        gt:
            Ground-truth plate string.
        pred:
            Predicted plate string, or None if the recognizer returned no result.
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
    Merge one statistics dict into another.

    Args:
        dst:
            Destination statistics dict.
        src:
            Source statistics dict.
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
    Pretty-print evaluation statistics.

    Args:
        title:
            Section title.
        stats:
            Statistics dict.
        max_failed_samples:
            Maximum number of failed samples to print.
    """
    total = stats["total"]
    exact = stats["exact"]
    accuracy = exact / total if total else 0.0

    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total images:              {total}")
    print(f"Exact matches:            {exact}")
    print(f"Accuracy:                 {accuracy:.4%}")
    print(f"No prediction:            {stats['no_prediction']}")
    print(f"Wrong length:             {stats['wrong_length']}")
    print(f"Samples with substitutions: {stats['substitution_samples']}")
    print(f"Total substitutions:      {sum(stats['substitution_counter'].values())}")
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
    Run evaluation on all configured dataset directories.
    """
    for path in (DETECTOR_MODEL_PATH, OCR_MODEL_PATH, OCR_CONFIG_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    recognizer = make_recognizer()

    overall_stats = init_stats()
    per_dataset_stats: dict[Path, dict] = {}

    for dataset_dir in DATASET_DIRS:
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        dataset_stats = init_stats()
        images = iter_images(dataset_dir)

        for image_path in images:
            gt = extract_ground_truth(image_path)
            result = recognizer.recognize(image_path)
            pred = None if result is None else result[0]
            update_stats(dataset_stats, image_path, gt, pred)

        per_dataset_stats[dataset_dir] = dataset_stats
        merge_stats(overall_stats, dataset_stats)

    for dataset_dir, stats in per_dataset_stats.items():
        print_stats(f"Dataset: {dataset_dir}", stats)

    print_stats("Overall", overall_stats, max_failed_samples=50)


if __name__ == "__main__":
    main()
