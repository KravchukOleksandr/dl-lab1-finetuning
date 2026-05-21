from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]  # x, y, w, h


@dataclass
class KLTTrackResult:
    bbox: BBox
    state: str
    confidence: float
    need_redetect: bool

    alive_points: int
    core_points: int
    core_ratio: float

    median_flow: Tuple[float, float]
    residual_threshold: float

    points: np.ndarray          # shape: (N, 1, 2)
    core_points_xy: np.ndarray  # shape: (M, 1, 2)


class CenterGridKLTTracker:
    """
    Минимальный KLT-трекер.

    Логика:
        1. YOLO даёт bbox.
        2. Внутри bbox берём ROI:
              x: 10%..90%
              y: 50%..85%
        3. Делим ROI на сетку 6x5.
        4. Точки = центры ячеек.
        5. Трекаем точки через KLT.
        6. Плохие точки убираем через forward-backward check.
        7. Считаем median flow.
        8. Считаем residual каждой точки относительно median flow.
        9. Через IQR находим согласованные точки.
       10. Если согласованных точек достаточно — двигаем bbox.
       11. Если нет — bad_frames += 1.
       12. Если bad_frames >= max_bad_frames — need_redetect=True.
       13. Потерянные точки заново ставим в центры своих ячеек.
    """

    def __init__(
        self,
        grid_w: int = 6,
        grid_h: int = 5,

        roi_x1: float = 0.10,
        roi_x2: float = 0.90,
        roi_y1: float = 0.50,
        roi_y2: float = 0.85,

        min_core_points: int = 8,
        min_core_ratio: float = 0.60,

        fb_error_thr: float = 2.0,

        residual_thr_min: float = 1.5,
        residual_thr_max: float = 6.0,

        max_bad_frames: int = 3,
    ):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.num_cells = grid_w * grid_h

        self.roi_x1 = roi_x1
        self.roi_x2 = roi_x2
        self.roi_y1 = roi_y1
        self.roi_y2 = roi_y2

        self.min_core_points = min_core_points
        self.min_core_ratio = min_core_ratio

        self.fb_error_thr = fb_error_thr

        self.residual_thr_min = residual_thr_min
        self.residual_thr_max = residual_thr_max

        self.max_bad_frames = max_bad_frames

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

        self.initialized = False
        self.prev_gray: Optional[np.ndarray] = None
        self.bbox: Optional[BBox] = None

        self.points = np.full((self.num_cells, 2), np.nan, dtype=np.float32)
        self.alive = np.zeros((self.num_cells,), dtype=bool)

        self.bad_frames = 0

    def init(self, frame: np.ndarray, bbox: BBox):
        gray = self._to_gray(frame)

        self.bbox = tuple(map(float, bbox))
        self.prev_gray = gray.copy()

        self.points[:] = np.nan
        self.alive[:] = False

        self.bad_frames = 0
        self.initialized = True

        self._refill_dead_cells(gray)

    def update(self, frame: np.ndarray) -> KLTTrackResult:
        if not self.initialized:
            raise RuntimeError("Tracker is not initialized. Call init(frame, bbox) first.")

        gray = self._to_gray(frame)

        live_ids = np.where(self.alive)[0]

        if len(live_ids) < self.min_core_points:
            self.bad_frames += 1
            self._refill_dead_cells(gray)
            self.prev_gray = gray.copy()

            return self._make_result(
                state="UNCERTAIN",
                confidence=0.0,
                need_redetect=self.bad_frames >= self.max_bad_frames,
                core_ids=np.array([], dtype=np.int32),
                median_flow=(0.0, 0.0),
                residual_threshold=0.0,
            )

        p0 = self.points[live_ids].reshape(-1, 1, 2).astype(np.float32)

        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            p0,
            None,
            **self.lk_params,
        )

        if p1 is None or st1 is None:
            self._kill_cells(live_ids)
            self.bad_frames += 1
            self._refill_dead_cells(gray)
            self.prev_gray = gray.copy()

            return self._make_result(
                state="BAD",
                confidence=0.0,
                need_redetect=True,
                core_ids=np.array([], dtype=np.int32),
                median_flow=(0.0, 0.0),
                residual_threshold=0.0,
            )

        p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.prev_gray,
            p1,
            None,
            **self.lk_params,
        )

        if p0_back is None or st2 is None:
            self._kill_cells(live_ids)
            self.bad_frames += 1
            self._refill_dead_cells(gray)
            self.prev_gray = gray.copy()

            return self._make_result(
                state="BAD",
                confidence=0.0,
                need_redetect=True,
                core_ids=np.array([], dtype=np.int32),
                median_flow=(0.0, 0.0),
                residual_threshold=0.0,
            )

        p0_flat = p0.reshape(-1, 2)
        p1_flat = p1.reshape(-1, 2)
        p0_back_flat = p0_back.reshape(-1, 2)

        st1 = st1.reshape(-1).astype(bool)
        st2 = st2.reshape(-1).astype(bool)

        fb_error = np.linalg.norm(p0_flat - p0_back_flat, axis=1)

        good = (
            st1
            & st2
            & (fb_error <= self.fb_error_thr)
            & self._points_inside_image(p1_flat, gray.shape[1], gray.shape[0])
        )

        good_live_ids = live_ids[good]
        bad_live_ids = live_ids[~good]

        self._kill_cells(bad_live_ids)

        if len(good_live_ids) > 0:
            self.points[good_live_ids] = p1_flat[good]

        if len(good_live_ids) < self.min_core_points:
            self.bad_frames += 1
            self._refill_dead_cells(gray)
            self.prev_gray = gray.copy()

            return self._make_result(
                state="UNCERTAIN",
                confidence=0.0,
                need_redetect=self.bad_frames >= self.max_bad_frames,
                core_ids=np.array([], dtype=np.int32),
                median_flow=(0.0, 0.0),
                residual_threshold=0.0,
            )

        old_good = p0_flat[good]
        new_good = p1_flat[good]

        flows = new_good - old_good

        median_flow_vec = np.median(flows, axis=0)
        residuals = np.linalg.norm(flows - median_flow_vec, axis=1)

        inliers, residual_threshold = self._motion_inliers_by_iqr(residuals)

        core_count = int(np.sum(inliers))
        core_ratio = core_count / len(good_live_ids)

        if core_count < self.min_core_points or core_ratio < self.min_core_ratio:
            self.bad_frames += 1
            self._refill_dead_cells(gray)
            self.prev_gray = gray.copy()

            return self._make_result(
                state="UNCERTAIN",
                confidence=0.2 * core_ratio,
                need_redetect=self.bad_frames >= self.max_bad_frames,
                core_ids=np.array([], dtype=np.int32),
                median_flow=(float(median_flow_vec[0]), float(median_flow_vec[1])),
                residual_threshold=residual_threshold,
            )

        core_live_ids = good_live_ids[inliers]
        core_flows = flows[inliers]

        dx = float(np.median(core_flows[:, 0]))
        dy = float(np.median(core_flows[:, 1]))

        x, y, w, h = self.bbox
        self.bbox = (x + dx, y + dy, w, h)

        self.bad_frames = 0

        self._refill_dead_cells(gray)
        self.prev_gray = gray.copy()

        confidence = min(1.0, 0.4 + 0.6 * core_ratio)

        return self._make_result(
            state="TRACKING",
            confidence=confidence,
            need_redetect=False,
            core_ids=core_live_ids,
            median_flow=(dx, dy),
            residual_threshold=residual_threshold,
        )

    def _motion_inliers_by_iqr(self, residuals: np.ndarray):
        q1 = float(np.percentile(residuals, 25))
        q3 = float(np.percentile(residuals, 75))
        iqr = q3 - q1

        threshold = q3 + 1.5 * iqr
        threshold = float(
            np.clip(
                threshold,
                self.residual_thr_min,
                self.residual_thr_max,
            )
        )

        inliers = residuals <= threshold

        return inliers, threshold

    def _refill_dead_cells(self, gray: np.ndarray):
        if self.bbox is None:
            return

        img_h, img_w = gray.shape[:2]

        for cell_id in range(self.num_cells):
            if self.alive[cell_id]:
                continue

            point = self._cell_center(cell_id, self.bbox)

            if point is None:
                continue

            px, py = point

            if 0 <= px < img_w and 0 <= py < img_h:
                self.points[cell_id] = [px, py]
                self.alive[cell_id] = True

    def _cell_center(self, cell_id: int, bbox: BBox):
        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            return None

        gx = cell_id % self.grid_w
        gy = cell_id // self.grid_w

        rx1 = x + self.roi_x1 * w
        rx2 = x + self.roi_x2 * w
        ry1 = y + self.roi_y1 * h
        ry2 = y + self.roi_y2 * h

        roi_w = rx2 - rx1
        roi_h = ry2 - ry1

        if roi_w <= 0 or roi_h <= 0:
            return None

        cell_w = roi_w / self.grid_w
        cell_h = roi_h / self.grid_h

        px = rx1 + (gx + 0.5) * cell_w
        py = ry1 + (gy + 0.5) * cell_h

        return float(px), float(py)

    def _kill_cells(self, cell_ids: np.ndarray):
        if len(cell_ids) == 0:
            return

        self.alive[cell_ids] = False
        self.points[cell_ids] = np.nan

    @staticmethod
    def _points_inside_image(points: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        x = points[:, 0]
        y = points[:, 1]

        return (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= 0)
            & (y >= 0)
            & (x < img_w)
            & (y < img_h)
        )

    def _make_result(
        self,
        state: str,
        confidence: float,
        need_redetect: bool,
        core_ids: np.ndarray,
        median_flow: Tuple[float, float],
        residual_threshold: float,
    ) -> KLTTrackResult:
        alive_ids = np.where(self.alive)[0]

        points = self.points[alive_ids].reshape(-1, 1, 2).astype(np.float32)

        if len(core_ids) > 0:
            core_points_xy = self.points[core_ids].reshape(-1, 1, 2).astype(np.float32)
        else:
            core_points_xy = np.empty((0, 1, 2), dtype=np.float32)

        alive_points = len(alive_ids)
        core_points = len(core_ids)
        core_ratio = core_points / max(1, alive_points)

        return KLTTrackResult(
            bbox=self.bbox,
            state=state,
            confidence=float(confidence),
            need_redetect=bool(need_redetect),

            alive_points=alive_points,
            core_points=core_points,
            core_ratio=float(core_ratio),

            median_flow=median_flow,
            residual_threshold=float(residual_threshold),

            points=points,
            core_points_xy=core_points_xy,
        )

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        return np.ascontiguousarray(gray)