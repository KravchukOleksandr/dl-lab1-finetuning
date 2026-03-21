from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import overload

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from fast_alpr import ALPR
from fast_alpr.base import BaseDetector, DetectionResult
from open_image_models.detection.core.yolo_v9.inference import YoloV9ObjectDetector

from plate_resolver import PlateFormat, PlateResolver


class LocalLicensePlateDetector(YoloV9ObjectDetector):
    """
    License plate detector backed by a local YOLOv9 ONNX file.

    This class mirrors the behavior of the built-in open-image-models license
    plate detector, but loads weights from a local ONNX file instead of using
    the model registry.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
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
    Adapter that makes a local detector compatible with fast-alpr.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
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
        Detect license plates in a BGR numpy image.

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

    Supported input types:
    - image path as str
    - image path as os.PathLike
    - numpy.ndarray
    - PIL.Image.Image

    Internally, every input is converted to a BGR numpy array before being
    passed to fast-alpr.
    """

    def __init__(
        self,
        plate_format: PlateFormat,
        detector_model_path: str | os.PathLike[str],
        ocr_model_path: str | os.PathLike[str],
        ocr_config_path: str | os.PathLike[str],
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

    @overload
    def recognize(self, image: str) -> tuple[str, np.ndarray] | None: ...
    @overload
    def recognize(self, image: os.PathLike[str]) -> tuple[str, np.ndarray] | None: ...
    @overload
    def recognize(self, image: np.ndarray) -> tuple[str, np.ndarray] | None: ...
    @overload
    def recognize(self, image: Image.Image) -> tuple[str, np.ndarray] | None: ...

    def recognize(
        self,
        image: str | os.PathLike[str] | np.ndarray | Image.Image,
    ) -> tuple[str, np.ndarray] | None:
        """
        Recognize a plate and return its bounding box.

        Args:
            image:
                Input image as:
                - file path
                - numpy array
                - PIL image

        Returns:
            A tuple:
                (plate_string, bbox_array)

            where `bbox_array` is:
                np.array([x1, y1, x2, y2], dtype=int)

            Returns None if:
            - no OCR candidate can be normalized
            - several different best candidates tie
        """
        image_bgr = self._load_image(image)
        results = self.alpr.predict(image_bgr)

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

    @overload
    def _load_image(self, image: str) -> np.ndarray: ...
    @overload
    def _load_image(self, image: os.PathLike[str]) -> np.ndarray: ...
    @overload
    def _load_image(self, image: np.ndarray) -> np.ndarray: ...
    @overload
    def _load_image(self, image: Image.Image) -> np.ndarray: ...

    def _load_image(
        self,
        image: str | os.PathLike[str] | np.ndarray | Image.Image,
    ) -> np.ndarray:
        """
        Convert supported input types into a BGR numpy image.

        Args:
            image:
                Input image as path, numpy array, or PIL image.

        Returns:
            Image as a BGR numpy array.

        Raises:
            FileNotFoundError:
                If an image path does not exist.
            ValueError:
                If the path cannot be loaded or the input array has an invalid shape.
            TypeError:
                If the input type is unsupported.
        """
        if isinstance(image, (str, os.PathLike)):
            image_path = Path(image)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image from path: {image_path}")

            return img

        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            rgb = np.array(image)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if image.ndim != 3:
                raise ValueError("numpy image must have shape (H, W) or (H, W, C)")

            if image.shape[2] == 3:
                return image

            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            raise ValueError("numpy image must have 1, 3, or 4 channels")

        raise TypeError(
            "Unsupported image type. Expected str, os.PathLike, numpy.ndarray, or PIL.Image.Image"
        )
