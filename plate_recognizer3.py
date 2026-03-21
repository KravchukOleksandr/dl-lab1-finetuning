from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fast_alpr import ALPR
from fast_alpr.base import BaseDetector, DetectionResult
from open_image_models.detection.core.yolo_v9.inference import YoloV9ObjectDetector

from plate_resolver import PlateFormat, PlateResolver


class LocalLicensePlateDetector(YoloV9ObjectDetector):
    """
    License plate detector backed by a local YOLOv9 ONNX file.

    This class mirrors the behavior of `open_image_models.LicensePlateDetector`,
    but it does not use the model registry or model download flow. Instead, it
    loads an already available local ONNX file.

    It is intended for cases where the ONNX weights are identical to one of the
    built-in open-image-models license plate detectors, but the file is stored
    locally.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_thresh: float = 0.25,
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions | None = None,
    ) -> None:
        """
        Initialize the local detector.

        Args:
            model_path:
                Local path to the detector ONNX file.
            conf_thresh:
                Confidence threshold used by the detector.
            providers:
                ONNX Runtime execution providers.
            sess_options:
                Optional ONNX Runtime session options.

        Raises:
            FileNotFoundError:
                If the ONNX file does not exist.
        """
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"File not found: {model_path}")

        super().__init__(
            model_path=str(model_path),
            conf_thresh=conf_thresh,
            class_labels=["License Plate"],
            providers=providers,
            sess_options=sess_options,
        )


class LocalDetector(BaseDetector):
    """
    fast-alpr detector wrapper for a local license plate detector.

    This adapter makes a locally loaded `LocalLicensePlateDetector` compatible
    with the detector interface expected by `fast_alpr.ALPR`.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_thresh: float = 0.25,
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions | None = None,
    ) -> None:
        """
        Initialize the detector wrapper.

        Args:
            model_path:
                Local path to the detector ONNX file.
            conf_thresh:
                Confidence threshold used by the detector.
            providers:
                ONNX Runtime execution providers.
            sess_options:
                Optional ONNX Runtime session options.
        """
        self.detector = LocalLicensePlateDetector(
            model_path=model_path,
            conf_thresh=conf_thresh,
            providers=providers,
            sess_options=sess_options,
        )

    def predict(self, frame: np.ndarray) -> list[DetectionResult]:
        """
        Detect license plates in a single image.

        Args:
            frame:
                Input image in BGR format.

        Returns:
            A list of detection results.
        """
        return self.detector.predict(frame)


class PlateRecognizer:
    """
    Recognize a plate from an image using local detector and OCR files.

    This class intentionally supports only local model files:
    - detector ONNX
    - OCR ONNX
    - OCR YAML config

    The processing flow is:

    1. Run plate detection and OCR through fast-alpr.
    2. Normalize OCR candidates using the provided `PlateFormat`.
    3. Pick the unique candidate with the smallest number of transformations.
    4. Return the normalized plate together with its bounding box.

    If no valid unique result exists, the method returns None.
    """

    def __init__(
        self,
        plate_format: PlateFormat,
        detector_model_path: str | Path,
        ocr_model_path: str | Path,
        ocr_config_path: str | Path,
        detector_conf_thresh: float = 0.02,
        detector_providers: Sequence[str | tuple[str, dict]] | None = None,
        detector_sess_options: ort.SessionOptions | None = None,
        ocr_device: str = "auto",
        ocr_providers: Sequence[str | tuple[str, dict]] | None = None,
        ocr_sess_options: ort.SessionOptions | None = None,
    ) -> None:
        """
        Initialize the recognizer from local files only.

        Args:
            plate_format:
                Plate format definition used for normalization and selection.
            detector_model_path:
                Local path to the detector ONNX file.
            ocr_model_path:
                Local path to the OCR ONNX file.
            ocr_config_path:
                Local path to the OCR YAML config file.
            detector_conf_thresh:
                Detector confidence threshold. Lower values may improve recall.
            detector_providers:
                ONNX Runtime execution providers for the detector.
            detector_sess_options:
                Optional ONNX Runtime session options for the detector.
            ocr_device:
                OCR execution device: "auto", "cpu", or "cuda".
            ocr_providers:
                ONNX Runtime execution providers for the OCR model.
            ocr_sess_options:
                Optional ONNX Runtime session options for the OCR model.

        Raises:
            FileNotFoundError:
                If any required local file does not exist.
        """
        detector_model_path = Path(detector_model_path)
        ocr_model_path = Path(ocr_model_path)
        ocr_config_path = Path(ocr_config_path)

        for path in (detector_model_path, ocr_model_path, ocr_config_path):
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")

        detector = LocalDetector(
            model_path=detector_model_path,
            conf_thresh=detector_conf_thresh,
            providers=detector_providers,
            sess_options=detector_sess_options,
        )

        self.alpr = ALPR(
            detector=detector,
            ocr_model=None,
            ocr_device=ocr_device,
            ocr_providers=ocr_providers,
            ocr_sess_options=ocr_sess_options,
            ocr_model_path=str(ocr_model_path),
            ocr_config_path=str(ocr_config_path),
            ocr_force_download=False,
        )
        self.resolver = PlateResolver(plate_format)

    def recognize(self, image: np.ndarray | str) -> tuple[str, np.ndarray] | None:
        """
        Recognize a plate and return its bounding box.

        Args:
            image:
                Either a BGR numpy array or a path to an image file.

        Returns:
            A tuple:
                (plate_string, bbox_array)

            where `bbox_array` is:
                np.array([x1, y1, x2, y2], dtype=int)

            Returns None if:
            - no OCR candidate can be normalized
            - several different best candidates tie
        """
        results = self.alpr.predict(image)

        best_plate: str | None = None
        best_bbox: np.ndarray | None = None
        best_changes: int | None = None
        ambiguous = False

        for result in results:
            if result.ocr is None or not result.ocr.text:
                continue

            normalized = self.resolver.normalize(result.ocr.text)
            if normalized is None:
                continue

            plate, changes = normalized
            bbox = result.detection.bounding_box
            bbox_array = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=int)

            if best_changes is None or changes < best_changes:
                best_plate = plate
                best_bbox = bbox_array
                best_changes = changes
                ambiguous = False
                continue

            if changes == best_changes and plate != best_plate:
                ambiguous = True

        if best_plate is None or best_bbox is None or ambiguous:
            return None

        return best_plate, best_bbox
