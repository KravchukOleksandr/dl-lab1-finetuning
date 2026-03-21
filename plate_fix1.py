def predict(self, frame: np.ndarray) -> list[DetectionResult]:
    detections = self.detector.predict(frame)
    filtered: list[DetectionResult] = []

    for det in detections:
        bbox = det.bounding_box
        width = bbox.x2 - bbox.x1
        height = bbox.y2 - bbox.y1

        if width < self.min_width_px or height < self.min_height_px:
            continue

        filtered.append(det)

    return filtered
