def recognize(
    self,
    image: str | os.PathLike[str] | np.ndarray | Image.Image,
    _allow_fallback: bool = True,
) -> tuple[str, np.ndarray] | None:
    """
    Recognize a plate and return its bounding box.

    The method runs the standard recognition pipeline on the original image.
    As a temporary workaround for models that may miss close-range plates,
    if recognition fails, it retries once on a downscaled version of the image.

    IMPORTANT:
    This fallback is model-specific and should be considered a temporary fix.
    It can be disabled via `ENABLE_CLOSE_RANGE_FALLBACK` or removed entirely
    once the detector issue is resolved.

    Args:
        image:
            Input image as:
            - file path
            - numpy array
            - PIL image
        _allow_fallback:
            Internal flag to prevent infinite recursion.

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

    if best_plate is not None and best_bbox is not None and not ambiguous:
        return best_plate, best_bbox

    # ===== fallback =====
    if not _allow_fallback or not self.ENABLE_CLOSE_RANGE_FALLBACK:
        return None

    scale = self.CLOSE_RANGE_FALLBACK_SCALE
    if not (0.0 < scale < 1.0):
        return None

    h, w = image_bgr.shape[:2]
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))

    small_image = cv2.resize(
        image_bgr,
        (small_w, small_h),
        interpolation=cv2.INTER_AREA,
    )

    # 🔥 ключевая строка — рекурсивный вызов
    return self.recognize(
        small_image,
        _allow_fallback=False,
    )
