from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import onnxruntime as ort
from fast_alpr import ALPR
from fast_plate_ocr.inference.hub import OcrModel
from open_image_models.detection.core.hub import PlateDetectorModel

from plate_resolver import PlateFormat, PlateResolver


class PlateRecognizer:
    """
    Recognize a plate from an image using fast-alpr plus deterministic
    post-processing based on a configurable plate format.

    The recognizer:
    - runs detector + OCR through fast-alpr
    - normalizes OCR outputs according to the provided PlateFormat
    - returns the final plate together with the corresponding bounding box

    The returned bounding box belongs to the candidate that produced the final
    selected plate with the smallest number of transformations.

    If no unique best result exists, the method returns None.
    """

    def __init__(
        self,
        plate_format: PlateFormat,
        detector_model: PlateDetectorModel = "yolo-v9-t-384-license-plate-end2end",
        detector_conf_thresh: float = 0.02,
        detector_providers: Sequence[str | tuple[str, dict]] | None = None,
        detector_sess_options: ort.SessionOptions | None = None,
        ocr_model: OcrModel | None = None,
        ocr_device: str = "auto",
        ocr_providers: Sequence[str | tuple[str, dict]] | None = None,
        ocr_sess_options: ort.SessionOptions | None = None,
        ocr_model_path: str | os.PathLike | None = None,
        ocr_config_path: str | os.PathLike | None = None,
        ocr_force_download: bool = False,
    ) -> None:
        """
        Initialize the recognizer.

        Args:
            plate_format:
                Plate format definition used for normalization and selection.
            detector_model:
                Name of a detector model supported by fast-alpr.
            detector_conf_thresh:
                Detector confidence threshold. Lower values may improve recall
                when deterministic post-processing is used afterward.
            detector_providers:
                ONNX Runtime providers for the detector session.
            detector_sess_options:
                Optional ONNX Runtime session options for the detector.
            ocr_model:
                OCR model name from the fast-plate-ocr hub. Set to None when
                using local OCR files.
            ocr_device:
                OCR execution device: "auto", "cpu", or "cuda".
            ocr_providers:
                ONNX Runtime providers for the OCR session.
            ocr_sess_options:
                Optional ONNX Runtime session options for the OCR model.
            ocr_model_path:
                Local path to the OCR ONNX model.
            ocr_config_path:
                Local path to the OCR plate config file.
            ocr_force_download:
                Whether to force re-download when using a hub OCR model.
        """
        self.alpr = ALPR(
            detector_model=detector_model,
            detector_conf_thresh=detector_conf_thresh,
            detector_providers=detector_providers,
            detector_sess_options=detector_sess_options,
            ocr_model=ocr_model,
            ocr_device=ocr_device,
            ocr_providers=ocr_providers,
            ocr_sess_options=ocr_sess_options,
            ocr_model_path=ocr_model_path,
            ocr_config_path=ocr_config_path,
            ocr_force_download=ocr_force_download,
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

            where bbox_array is:
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
