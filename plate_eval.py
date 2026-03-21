from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from plate_recognizer import PlateRecognizer
from ua import UA_PLATE_FORMAT


# =========================
# Configuration
# =========================

DATASET_DIR = Path(r"/path/to/your/plate_crops")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DETECTOR_MODEL_PATH = Path(r"/path/to/models/yolo-v9-s-608-license-plates-end2end.onnx")
OCR_MODEL_PATH = Path(r"/path/to/models/cct-s-v2-global-model/cct_s_v2_global.onnx")
OCR_CONFIG_PATH = Path(r"/path/to/models/cct-s-v2-global-model/cct_s_v2_global_plate_config.yaml")

DETECTOR_CONF_THRESH = 0.0001
DETECTOR_PROVIDERS = ["CPUExecutionProvider"]
OCR_PROVIDERS = ["CPUExecutionProvider"]
OCR_DEVICE = "auto"

# If True, only keep alphanumeric characters from the filename when extracting GT.
# Example:
#   "AA1234BC.jpg"        -> "AA1234BC"
#   "AA1234BC_001.jpg"    -> "AA1234BC001"  (bad for such naming)
#
# If your filenames contain extra suffixes/prefixes, customize extract_ground_truth().
ONLY_ALNUM_FROM_FILENAME = True


def iter_images(directory: Path, extensions: set[str]) -> Iterable[Path]:
    """
    Yield image files from a directory recursively.

    Args:
        directory: Root directory with evaluation images.
        extensions: Allowed lowercase file extensions.

    Yields:
        Paths to image files.
    """
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def extract_ground_truth(image_path: Path) -> str:
    """
    Extract the target plate string from the image filename.

    By default, this function uses the file stem and optionally keeps only
    alphanumeric characters.

    Examples:
        AA1234BC.jpg   -> AA1234BC
        aa1234bc.png   -> AA1234BC

    Adjust this function if your dataset naming scheme is different.

    Args:
        image_path: Path to an image.

    Returns:
        Ground-truth plate string.
    """
    stem = image_path.stem.upper()
    if ONLY_ALNUM_FROM_FILENAME:
        stem = "".join(ch for ch in stem if ch.isalnum())
    return stem


def compare_strings(gt: str, pred: str) -> tuple[bool, bool, list[tuple[str, str]]]:
    """
    Compare ground truth and prediction.

    Args:
        gt: Ground-truth plate.
        pred: Predicted plate.

    Returns:
        A tuple:
            - exact_match: True if strings are exactly equal
            - length_mismatch: True if lengths differ
            - substitutions: list of (expected_char, predicted_char) pairs for
              differing positions when lengths are equal
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


def main() -> None:
    """
    Run evaluation for the Ukrainian plate recognizer on a folder of plate crops.
    """
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    recognizer = PlateRecognizer(
        plate_format=UA_PLATE_FORMAT,
        detector_model_path=DETECTOR_MODEL_PATH,
        ocr_model_path=OCR_MODEL_PATH,
        ocr_config_path=OCR_CONFIG_PATH,
        detector_conf_thresh=DETECTOR_CONF_THRESH,
        detector_providers=DETECTOR_PROVIDERS,
        ocr_providers=OCR_PROVIDERS,
        ocr_device=OCR_DEVICE,
    )

    total_images = 0
    exact_matches = 0
    no_prediction = 0
    wrong_length = 0

    substitution_counter: Counter[tuple[str, str]] = Counter()
    per_image_errors: list[tuple[Path, str, str | None, str]] = []

    for image_path in iter_images(DATASET_DIR, IMAGE_EXTENSIONS):
        total_images += 1

        gt = extract_ground_truth(image_path)
        result = recognizer.recognize(image_path)

        if result is None:
            no_prediction += 1
            per_image_errors.append((image_path, gt, None, "no_prediction"))
            continue

        pred, _bbox = result

        exact_match, length_mismatch, substitutions = compare_strings(gt, pred)

        if exact_match:
            exact_matches += 1
            continue

        if length_mismatch:
            wrong_length += 1
            per_image_errors.append((image_path, gt, pred, "wrong_length"))
            continue

        for sub in substitutions:
            substitution_counter[sub] += 1

        per_image_errors.append((image_path, gt, pred, "substitution"))

    accuracy = exact_matches / total_images if total_images else 0.0

    print("=" * 80)
    print("Ukrainian Plate Recognizer Evaluation")
    print("=" * 80)
    print(f"Dataset dir:          {DATASET_DIR}")
    print(f"Total images:         {total_images}")
    print(f"Exact matches:        {exact_matches}")
    print(f"Accuracy:             {accuracy:.4%}")
    print(f"No prediction:        {no_prediction}")
    print(f"Wrong length:         {wrong_length}")
    print(f"Total substitutions:  {sum(substitution_counter.values())}")
    print()

    print("=" * 80)
    print("Error summary")
    print("=" * 80)
    error_type_counter = Counter(err_type for _, _, _, err_type in per_image_errors)
    if not error_type_counter:
        print("No errors.")
    else:
        for err_type, count in error_type_counter.most_common():
            print(f"{err_type:20s} {count}")

    print()
    print("=" * 80)
    print("Character substitution statistics")
    print("=" * 80)
    if not substitution_counter:
        print("No substitution errors.")
    else:
        for (expected, predicted), count in substitution_counter.most_common():
            print(f"{expected} -> {predicted}: {count}")

    print()
    print("=" * 80)
    print("Examples of failed samples")
    print("=" * 80)
    if not per_image_errors:
        print("No failed samples.")
    else:
        for image_path, gt, pred, err_type in per_image_errors[:100]:
            print(
                f"{err_type:15s} | "
                f"gt={gt:12s} | "
                f"pred={str(pred):12s} | "
                f"path={image_path.name}"
            )


if __name__ == "__main__":
    main()
