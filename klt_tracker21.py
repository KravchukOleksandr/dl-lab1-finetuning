import cv2
import numpy as np


class BootstrapKLTTracker:
    def __init__(
        self,
        grid_w=6,
        grid_h=5,
        roi=(0.10, 0.90, 0.50, 0.85),
        min_core_points=8,
        min_start_motion=3.0,
        flow_match_thr=3.0,
        required_matches=2,
        fb_error_thr=2.0,
        scale_limit=0.30,
        max_bootstrap_frames=8,
        max_klt_frames=60,
    ):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.n = grid_w * grid_h

        self.roi_x1, self.roi_x2, self.roi_y1, self.roi_y2 = roi

        self.min_core_points = min_core_points
        self.min_start_motion = min_start_motion
        self.flow_match_thr = flow_match_thr
        self.required_matches = required_matches
        self.fb_error_thr = fb_error_thr

        self.scale_min = 1.0 - scale_limit
        self.scale_max = 1.0 + scale_limit

        self.max_bootstrap_frames = max_bootstrap_frames
        self.max_klt_frames = max_klt_frames

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.state = "EMPTY"
        self.prev_gray = None
        self.bbox = None

        self.points = np.full((self.n, 2), np.nan, np.float32)
        self.anchor_points = np.full((self.n, 2), np.nan, np.float32)
        self.alive = np.zeros(self.n, bool)
        self.score = np.zeros(self.n, np.int32)

        self.yolo_anchor_bbox = None
        self.core_ids = np.array([], dtype=np.int32)

        self.anchor_bbox = None
        self.anchor_center = None
        self.left_offset = 0.0
        self.right_offset = 0.0
        self.top_offset = 0.0
        self.bottom_offset = 0.0

        self.bootstrap_frames = 0
        self.klt_frames = 0

    def init(self, frame, bbox):
        gray = self._gray(frame)

        self.state = "WAIT"
        self.prev_gray = gray.copy()
        self.bbox = tuple(map(float, bbox))

        self.points[:] = np.nan
        self.anchor_points[:] = np.nan
        self.alive[:] = False
        self.score[:] = 0

        self._seed_points(self.bbox)

        self.anchor_points[:] = self.points
        self.yolo_anchor_bbox = self.bbox

        self.core_ids = np.array([], dtype=np.int32)

        self.bootstrap_frames = 0
        self.klt_frames = 0

    def update(self, frame):
        gray = self._gray(frame)

        if self.state == "LOCKED":
            result = self._update_locked(gray)
        else:
            result = self._update_wait_or_bootstrap(gray)

        self.prev_gray = gray.copy()
        return result

    def feed_yolo(self, yolo_bbox):
        self.bbox = tuple(map(float, yolo_bbox))

        if self.state not in ("WAIT", "BOOTSTRAP"):
            return self._result("LOST", True)

        self.state = "BOOTSTRAP"
        self.bootstrap_frames += 1

        ids = np.where(self.alive)[0]

        if len(ids) < self.min_core_points:
            return self._result("LOST", True)

        box_flow = self._center(self.bbox) - self._center(self.yolo_anchor_bbox)

        if np.linalg.norm(box_flow) < self.min_start_motion:
            self.state = "WAIT"
            return self._result("WAIT", False)

        point_flow = self.points[ids] - self.anchor_points[ids]
        err = np.linalg.norm(point_flow - box_flow, axis=1)

        matched = err <= self.flow_match_thr

        self.score[ids[matched]] += 1
        self.score[ids[~matched]] = 0

        core_ids = ids[self.score[ids] >= self.required_matches]
        self.core_ids = core_ids

        if len(core_ids) >= self.min_core_points:
            self._lock_core(self.bbox, core_ids)
            return self._result("LOCKED", False)

        if self.bootstrap_frames >= self.max_bootstrap_frames:
            return self._result("LOST", True)

        return self._result("BOOTSTRAP", True)

    def _update_wait_or_bootstrap(self, gray):
        ids = np.where(self.alive)[0]

        if len(ids) < self.min_core_points:
            return self._result("LOST", True)

        good_ids, old, new = self._track(ids, gray)

        if len(good_ids) < self.min_core_points:
            return self._result("LOST", True)

        if self.state == "WAIT":
            displacement = self.points[good_ids] - self.anchor_points[good_ids]
            motion = np.median(np.linalg.norm(displacement, axis=1))

            if motion < self.min_start_motion:
                return self._result("WAIT", False)

            self.state = "BOOTSTRAP"
            return self._result("BOOTSTRAP", True)

        return self._result("BOOTSTRAP", True)

    def _update_locked(self, gray):
        ids = self.core_ids[self.alive[self.core_ids]]

        if len(ids) < self.min_core_points:
            return self._result("LOST", True)

        good_ids, old, new = self._track(ids, gray)
        self.core_ids = good_ids

        if len(good_ids) < self.min_core_points:
            return self._result("LOST", True)

        self.klt_frames += 1

        if self.klt_frames > self.max_klt_frames:
            return self._result("LOST", True)

        bbox = self._bbox_from_core(good_ids)

        if bbox is None:
            return self._result("LOST", True)

        self.bbox = bbox
        return self._result("LOCKED", False)

    def _track(self, ids, gray):
        p0 = self.points[ids].reshape(-1, 1, 2).astype(np.float32)

        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            p0,
            None,
            **self.lk_params,
        )

        if p1 is None:
            self._kill(ids)
            return np.array([], dtype=np.int32), None, None

        p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.prev_gray,
            p1,
            None,
            **self.lk_params,
        )

        if p0_back is None:
            self._kill(ids)
            return np.array([], dtype=np.int32), None, None

        p0 = p0.reshape(-1, 2)
        p1 = p1.reshape(-1, 2)
        p0_back = p0_back.reshape(-1, 2)

        fb_error = np.linalg.norm(p0 - p0_back, axis=1)

        good = (
            st1.reshape(-1).astype(bool)
            & st2.reshape(-1).astype(bool)
            & (fb_error <= self.fb_error_thr)
        )

        good_ids = ids[good]
        bad_ids = ids[~good]

        self._kill(bad_ids)
        self.points[good_ids] = p1[good]

        return good_ids, p0[good], p1[good]

    def _lock_core(self, bbox, core_ids):
        self.state = "LOCKED"

        self.anchor_bbox = bbox
        self.core_ids = core_ids.copy()

        core_points = self.points[core_ids]
        self.anchor_center = np.median(core_points, axis=0)

        x, y, w, h = bbox
        cx, cy = self.anchor_center

        self.left_offset = cx - x
        self.right_offset = x + w - cx
        self.top_offset = cy - y
        self.bottom_offset = y + h - cy

        self.klt_frames = 0

    def _bbox_from_core(self, ids):
        anchor = self.anchor_points[ids]
        current = self.points[ids]

        displacement = current - anchor
        center_now = self.anchor_center + np.median(displacement, axis=0)

        spread_anchor = np.median(
            np.linalg.norm(anchor - self.anchor_center, axis=1)
        )

        if spread_anchor < 1e-6:
            return None

        spread_now = np.median(
            np.linalg.norm(current - center_now, axis=1)
        )

        scale = spread_now / spread_anchor
        scale = float(np.clip(scale, self.scale_min, self.scale_max))

        cx, cy = center_now

        x = cx - self.left_offset * scale
        y = cy - self.top_offset * scale
        w = (self.left_offset + self.right_offset) * scale
        h = (self.top_offset + self.bottom_offset) * scale

        return float(x), float(y), float(w), float(h)

    def _seed_points(self, bbox):
        x, y, w, h = bbox

        rx1 = x + self.roi_x1 * w
        rx2 = x + self.roi_x2 * w
        ry1 = y + self.roi_y1 * h
        ry2 = y + self.roi_y2 * h

        cell_w = (rx2 - rx1) / self.grid_w
        cell_h = (ry2 - ry1) / self.grid_h

        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                i = gy * self.grid_w + gx

                px = rx1 + (gx + 0.5) * cell_w
                py = ry1 + (gy + 0.5) * cell_h

                self.points[i] = [px, py]
                self.alive[i] = True

    def _kill(self, ids):
        self.alive[ids] = False
        self.points[ids] = np.nan
        self.score[ids] = 0

    def _result(self, state, need_yolo):
        alive_ids = np.where(self.alive)[0]

        if self.state == "LOCKED":
            core_ids = self.core_ids
        else:
            core_ids = np.where(self.alive & (self.score > 0))[0]

        return {
            "state": state,
            "need_yolo": need_yolo,
            "bbox": self.bbox,
            "points": self.points[alive_ids].reshape(-1, 1, 2).astype(np.float32),
            "core_points": self.points[core_ids].reshape(-1, 1, 2).astype(np.float32),
            "alive": len(alive_ids),
            "core": len(core_ids),
        }

    @staticmethod
    def _center(bbox):
        x, y, w, h = bbox
        return np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float32)

    @staticmethod
    def _gray(frame):
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame