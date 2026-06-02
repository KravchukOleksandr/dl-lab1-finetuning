from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
import torch.nn.functional as F

from .point_klt_config import PointKLTConfig
from .student_mask import MicroNeXtMaskNet, SimplePointSelectorConfig, select_points_simple


@dataclass
class PointTrack:
    track_id: int
    bbox: np.ndarray
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    age: int = 0
    missed_yolo: int = 0
    confidence: float = 1.0
    needs_yolo: bool = False
    needs_refill: bool = False
    status: str = "init"


@dataclass
class PointKLTStats:
    frames_received: int = 0
    yolo_runs: int = 0
    student_runs: int = 0
    forced_yolo_runs: int = 0
    scheduled_yolo_runs: int = 0
    active_tracks: int = 0
    last_yolo_reason: str = "none"

    @property
    def yolo_saved_frames(self) -> int:
        return max(0, self.frames_received - self.yolo_runs)

    @property
    def yolo_saved_pct(self) -> float:
        if self.frames_received <= 0:
            return 0.0
        return 100.0 * self.yolo_saved_frames / self.frames_received


def clip_box(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    return np.array([
        np.clip(x1, 0, w - 1),
        np.clip(y1, 0, h - 1),
        np.clip(x2, 0, w - 1),
        np.clip(y2, 0, h - 1),
    ], dtype=np.float32)


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return 0.0 if union <= 1e-9 else float(inter / union)


def box_center(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def box_diag(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return float(np.hypot(x2 - x1, y2 - y1))


def expand_box(box: np.ndarray, img_w: int, img_h: int, scale: float) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    out = np.array([
        cx - bw * scale * 0.5,
        cy - bh * scale * 0.5,
        cx + bw * scale * 0.5,
        cy + bh * scale * 0.5,
    ], dtype=np.float32)
    return clip_box(out, img_w, img_h)


def int_crop_box(box: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    box = clip_box(box, img_w, img_h)
    x1, y1, x2, y2 = box
    x1 = max(0, min(img_w - 1, int(np.floor(x1))))
    y1 = max(0, min(img_h - 1, int(np.floor(y1))))
    x2 = max(x1 + 1, min(img_w, int(np.ceil(x2))))
    y2 = max(y1 + 1, min(img_h, int(np.ceil(y2))))
    return x1, y1, x2, y2


def shift_box(box: np.ndarray, dx: float, dy: float, img_w: int, img_h: int) -> np.ndarray:
    out = box.copy().astype(np.float32)
    out[[0, 2]] += dx
    out[[1, 3]] += dy
    return clip_box(out, img_w, img_h)


def letterbox_image(image_bgr: np.ndarray, out_w: int, out_h: int, pad_value: int = 114):
    h, w = image_bgr.shape[:2]
    scale = min(out_w / w, out_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((out_h, out_w, 3), pad_value, dtype=np.uint8)
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "new_w": new_w, "new_h": new_h}


def letterbox_point_to_crop(point_xy: Tuple[float, float], transform) -> Optional[Tuple[float, float]]:
    x, y = point_xy
    if x < transform["pad_x"] or y < transform["pad_y"]:
        return None
    if x >= transform["pad_x"] + transform["new_w"] or y >= transform["pad_y"] + transform["new_h"]:
        return None
    return (float((x - transform["pad_x"]) / transform["scale"]),
            float((y - transform["pad_y"]) / transform["scale"]))


def frame_point_to_letterbox(point_xy: Tuple[float, float], crop_origin: Tuple[int, int], transform, cfg: PointKLTConfig):
    fx, fy = point_xy
    ox, oy = crop_origin
    lx = (fx - ox) * transform["scale"] + transform["pad_x"]
    ly = (fy - oy) * transform["scale"] + transform["pad_y"]
    lx, ly = int(round(lx)), int(round(ly))
    if lx < 0 or ly < 0 or lx >= cfg.input_w or ly >= cfg.input_h:
        return None
    return lx, ly


class PointKLTPersonTracker:
    """
    Returns sv.Detections with stable tracker_id.
    YOLO is used only on schedule or when KLT confidence drops.
    """
    def __init__(self, yolo_model, cfg: PointKLTConfig, frame_hw: Tuple[int, int], logger=None):
        self.yolo_model = yolo_model
        self.cfg = cfg
        self.logger = logger
        self.frame_h, self.frame_w = frame_hw
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.student_model = self._load_student_model(cfg.student_model_path)
        self.selector_cfg = SimplePointSelectorConfig(
            mask_threshold=cfg.mask_threshold,
            torso_y_min=cfg.torso_y_min,
            torso_y_max=cfg.torso_y_max,
            max_points=cfg.target_points,
        )

        self.tracks: List[PointTrack] = []
        self.next_track_id = 1
        self.prev_gray: Optional[np.ndarray] = None
        self.stats = PointKLTStats()
        self.yolo_interval_frames = 1

    def set_fps(self, fps: float) -> None:
        fps = fps if fps and fps > 1e-6 else 25.0
        self.yolo_interval_frames = max(1, int(round(fps * self.cfg.yolo_interval_seconds)))

    def _load_student_model(self, path: str) -> MicroNeXtMaskNet:
        model = MicroNeXtMaskNet().to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
        model.eval()
        return model

    def update(self, frame_bgr: np.ndarray, allow_yolo: bool = True) -> Tuple[sv.Detections, PointKLTStats]:
        self.stats.frames_received += 1
        frame_idx = self.stats.frames_received - 1
        h, w = frame_bgr.shape[:2]
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        scheduled_yolo = frame_idx == 0 or len(self.tracks) == 0 or frame_idx % self.yolo_interval_frames == 0

        if self.prev_gray is not None and self.tracks:
            for tr in self.tracks:
                self._klt_update_track(tr, self.prev_gray, curr_gray, w, h)

        need_yolo = any(tr.needs_yolo for tr in self.tracks)
        run_yolo = allow_yolo and (scheduled_yolo or need_yolo)

        if run_yolo:
            reason = "first_frame" if frame_idx == 0 else ("scheduled" if scheduled_yolo else "low_confidence")
            self._refresh_with_yolo(frame_bgr, frame_idx)
            self.stats.yolo_runs += 1
            self.stats.last_yolo_reason = reason
            if scheduled_yolo:
                self.stats.scheduled_yolo_runs += 1
            else:
                self.stats.forced_yolo_runs += 1
        else:
            for tr in self.tracks:
                if tr.needs_refill and not tr.needs_yolo:
                    self._safe_refill(tr, frame_bgr)

            if allow_yolo and any(tr.needs_yolo for tr in self.tracks):
                self._refresh_with_yolo(frame_bgr, frame_idx)
                self.stats.yolo_runs += 1
                self.stats.forced_yolo_runs += 1
                self.stats.last_yolo_reason = "refill_rejected"

        for tr in self.tracks:
            tr.age += 1

        self.tracks = [tr for tr in self.tracks if tr.missed_yolo <= self.cfg.max_yolo_misses]
        self.stats.active_tracks = len(self.tracks)
        self.prev_gray = curr_gray

        if self.logger and self.cfg.log_every_n_frames > 0 and self.stats.frames_received % self.cfg.log_every_n_frames == 0:
            self.logger.info(
                f"PointKLT frames={self.stats.frames_received} yolo_runs={self.stats.yolo_runs} "
                f"student_runs={self.stats.student_runs} saved={self.stats.yolo_saved_frames} "
                f"saved_pct={self.stats.yolo_saved_pct:.2f}% tracks={len(self.tracks)}"
            )

        return self._to_detections(), self.stats


    def draw_debug(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Draw internal PointKLT tracks directly from PointTrack state.
        This is intentionally independent from Tracklet.draw(), because Tracklet may hide
        static/unconfirmed/lost objects while we still need to debug tracker state.
        """
        if not (self.cfg.draw_debug_points or self.cfg.draw_debug_status):
            return frame_bgr

        out = frame_bgr.copy()
        h, w = out.shape[:2]

        for tr in self.tracks:
            box = clip_box(tr.bbox, w, h)
            x1, y1, x2, y2 = map(int, box)

            if tr.needs_yolo:
                color = (0, 0, 255)
            elif tr.needs_refill:
                color = (0, 165, 255)
            else:
                rng = np.random.default_rng(tr.track_id * 12345)
                c = rng.integers(80, 255, size=3)
                color = (int(c[0]), int(c[1]), int(c[2]))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            if self.cfg.draw_debug_status:
                label = f"PKLT id {tr.track_id} pts {len(tr.points)} {tr.status}"
                cv2.putText(
                    out,
                    label,
                    (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            if self.cfg.draw_debug_points and tr.points is not None:
                for p in tr.points:
                    px, py = int(round(float(p[0]))), int(round(float(p[1])))
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(out, (px, py), 3, (0, 255, 0), -1)

        return out

    def _to_detections(self) -> sv.Detections:
        tracks = [tr for tr in self.tracks if len(tr.points) >= self.cfg.min_points_for_klt]
        if not tracks:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.array([], dtype=np.float32),
                class_id=np.array([], dtype=np.int32),
                tracker_id=np.array([], dtype=np.int32),
            )

        return sv.Detections(
            xyxy=np.array([tr.bbox for tr in tracks], dtype=np.float32),
            confidence=np.array([tr.confidence for tr in tracks], dtype=np.float32),
            class_id=np.zeros(len(tracks), dtype=np.int32),
            tracker_id=np.array([tr.track_id for tr in tracks], dtype=np.int32),
        )

    def _run_yolo(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        kwargs = dict(
            source=frame_bgr,
            imgsz=self.cfg.yolo_imgsz,
            conf=self.cfg.yolo_conf,
            classes=[self.cfg.yolo_person_class_id],
            verbose=False,
        )
        if self.cfg.yolo_device is not None:
            kwargs["device"] = self.cfg.yolo_device
        if self.cfg.yolo_half and torch.cuda.is_available():
            kwargs["half"] = True

        result = self.yolo_model.predict(**kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        return [b.astype(np.float32) for b in result.boxes.xyxy.detach().cpu().numpy()]

    def _refresh_with_yolo(self, frame_bgr: np.ndarray, frame_idx: int) -> None:
        h, w = frame_bgr.shape[:2]
        detections = [clip_box(det, w, h) for det in self._run_yolo(frame_bgr)]
        matches, unmatched_tracks, unmatched_dets = self._match_detections(detections)

        alive: List[PointTrack] = []

        for ti, di, _score in matches:
            tr = self.tracks[ti]
            tr.bbox = detections[di]
            tr.missed_yolo = 0
            tr.needs_yolo = False
            tr.needs_refill = False
            tr.status = "yolo_match"
            tr.points = np.array(self._predict_points_for_bbox(frame_bgr, tr.bbox), dtype=np.float32)
            tr.confidence = min(1.0, len(tr.points) / self.cfg.target_points)
            if len(tr.points) < self.cfg.min_points_for_klt:
                tr.needs_yolo = True
                tr.status = "yolo_match_but_few_points"
            alive.append(tr)

        for ti in unmatched_tracks:
            tr = self.tracks[ti]
            tr.missed_yolo += 1
            if tr.missed_yolo <= self.cfg.max_yolo_misses:
                tr.needs_yolo = True
                tr.needs_refill = False
                tr.status = "yolo_missed_keep"
                alive.append(tr)

        for di in unmatched_dets:
            points = self._predict_points_for_bbox(frame_bgr, detections[di])
            tr = PointTrack(
                track_id=self.next_track_id,
                bbox=detections[di],
                points=np.array(points, dtype=np.float32),
                confidence=min(1.0, len(points) / self.cfg.target_points),
                status="new_yolo",
            )
            if len(tr.points) < self.cfg.min_points_for_klt:
                tr.needs_yolo = True
                tr.status = "new_yolo_but_few_points"
            self.next_track_id += 1
            alive.append(tr)

        self.tracks = alive

    def _match_detections(self, detections: List[np.ndarray]):
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                ok, score = self._association_score(tr.bbox, det)
                if ok:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        matched_tracks, matched_dets, matches = set(), set(), []
        for score, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            matches.append((ti, di, score))

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _association_score(self, track_box: np.ndarray, det_box: np.ndarray):
        iou = box_iou(track_box, det_box)
        center_norm = float(np.linalg.norm(box_center(track_box) - box_center(det_box)) / max(1.0, box_diag(track_box)))
        ok = iou >= self.cfg.match_iou_min or center_norm <= self.cfg.match_center_norm_max
        score = iou + max(0.0, 1.0 - center_norm) * 0.25
        return ok, float(score)

    @torch.no_grad()
    def _predict_points_and_mask_for_bbox(self, frame_bgr: np.ndarray, bbox: np.ndarray):
        self.stats.student_runs += 1
        h, w = frame_bgr.shape[:2]
        work_box = expand_box(bbox, w, h, self.cfg.box_scale)
        x1, y1, x2, y2 = int_crop_box(work_box, w, h)
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            return [], None, None, None

        letter_bgr, transform = letterbox_image(crop_bgr, self.cfg.input_w, self.cfg.input_h, pad_value=114)
        letter_rgb = cv2.cvtColor(letter_bgr, cv2.COLOR_BGR2RGB)
        inp = letter_rgb.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None]
        tensor = torch.from_numpy(inp).float().to(self.device)

        logits = self.student_model(tensor)
        pred = torch.sigmoid(logits)
        pred_up = F.interpolate(pred, size=(self.cfg.input_h, self.cfg.input_w), mode="bilinear", align_corners=False)
        pred_mask = pred_up[0, 0].detach().cpu().numpy().astype(np.float32)

        points_letter = select_points_simple(letter_rgb, pred_mask, self.selector_cfg)
        frame_points = []
        for p in points_letter:
            crop_point = letterbox_point_to_crop(p, transform)
            if crop_point is None:
                continue
            px, py = crop_point
            fx, fy = x1 + px, y1 + py
            if 0 <= fx < w and 0 <= fy < h:
                frame_points.append((float(fx), float(fy)))

        return frame_points, pred_mask, transform, (x1, y1)

    def _predict_points_for_bbox(self, frame_bgr: np.ndarray, bbox: np.ndarray):
        points, _, _, _ = self._predict_points_and_mask_for_bbox(frame_bgr, bbox)
        return points

    def _count_points_inside_pred_mask(self, points: np.ndarray, pred_mask: np.ndarray, transform, crop_origin) -> int:
        if pred_mask is None or transform is None or crop_origin is None:
            return 0
        count = 0
        for p in points:
            lp = frame_point_to_letterbox((float(p[0]), float(p[1])), crop_origin, transform, self.cfg)
            if lp is None:
                continue
            lx, ly = lp
            if pred_mask[ly, lx] >= self.cfg.mask_threshold:
                count += 1
        return count

    def _klt_update_track(self, tr: PointTrack, prev_gray: np.ndarray, curr_gray: np.ndarray, img_w: int, img_h: int) -> None:
        tr.needs_refill = False
        if tr.points is None or len(tr.points) < self.cfg.min_points_for_klt:
            tr.needs_yolo = True
            tr.status = "too_few_points_before_klt"
            tr.confidence = 0.0
            return

        old_pts = tr.points.astype(np.float32).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        new_pts, st1, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, old_pts, None,
            winSize=self.cfg.klt_win_size, maxLevel=self.cfg.klt_max_level, criteria=criteria,
        )
        if new_pts is None or st1 is None:
            tr.needs_yolo = True
            tr.status = "klt_failed"
            tr.confidence = 0.0
            return

        back_pts, st2, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, new_pts, None,
            winSize=self.cfg.klt_win_size, maxLevel=self.cfg.klt_max_level, criteria=criteria,
        )
        if back_pts is None or st2 is None:
            tr.needs_yolo = True
            tr.status = "klt_back_failed"
            tr.confidence = 0.0
            return

        old = old_pts.reshape(-1, 2)
        new = new_pts.reshape(-1, 2)
        back = back_pts.reshape(-1, 2)
        st1 = st1.reshape(-1).astype(bool)
        st2 = st2.reshape(-1).astype(bool)
        fb_err = np.linalg.norm(old - back, axis=1)
        fb_good = st1 & st2 & (fb_err <= self.cfg.fb_max_error)
        fb_good_count = int(fb_good.sum())

        if fb_good_count < self.cfg.min_points_for_klt:
            tr.points = new[fb_good].astype(np.float32)
            tr.needs_yolo = True
            tr.status = f"too_few_fb_good_{fb_good_count}"
            tr.confidence = fb_good_count / self.cfg.target_points
            return

        old_good = old[fb_good]
        new_good = new[fb_good]
        flow = new_good - old_good
        med = np.median(flow, axis=0)
        residual = np.linalg.norm(flow - med[None], axis=1)

        thr = self.cfg.motion_residual_rel * box_diag(tr.bbox)
        thr = max(self.cfg.motion_residual_min, thr)
        thr = min(self.cfg.motion_residual_max, thr)

        inliers = residual <= thr
        inlier_count = int(inliers.sum())

        if fb_good_count >= 5:
            required_inliers = 4
        elif fb_good_count == 4:
            required_inliers = 4
        elif fb_good_count == 3:
            required_inliers = 3
        else:
            required_inliers = self.cfg.min_points_for_klt

        if inlier_count < required_inliers:
            tr.points = new_good[inliers].astype(np.float32)
            tr.needs_yolo = True
            tr.status = f"motion_not_consistent_{inlier_count}_of_{fb_good_count}"
            tr.confidence = inlier_count / self.cfg.target_points
            return

        flow_in = flow[inliers]
        new_in = new_good[inliers]
        dx, dy = np.median(flow_in, axis=0)
        tr.bbox = shift_box(tr.bbox, float(dx), float(dy), img_w, img_h)
        tr.points = new_in.astype(np.float32)
        tr.needs_yolo = False
        tr.confidence = min(1.0, len(tr.points) / self.cfg.target_points)

        if len(tr.points) == self.cfg.refill_only_when_points_equal:
            tr.needs_refill = True
            tr.status = "klt_ok_3pts_needs_refill"
        else:
            tr.status = f"klt_ok_{len(tr.points)}pts"

    def _safe_refill(self, tr: PointTrack, frame_bgr: np.ndarray) -> None:
        if not self.cfg.enable_safe_refill or not tr.needs_refill:
            return
        if tr.points is None or len(tr.points) != self.cfg.refill_only_when_points_equal:
            return

        candidates, pred_mask, transform, crop_origin = self._predict_points_and_mask_for_bbox(frame_bgr, tr.bbox)
        inside_old = self._count_points_inside_pred_mask(tr.points, pred_mask, transform, crop_origin)
        if inside_old < self.cfg.min_points_inside_mask_for_refill:
            tr.needs_yolo = True
            tr.needs_refill = False
            tr.status = f"refill_rejected_mask_{inside_old}_of_{len(tr.points)}"
            tr.confidence = inside_old / self.cfg.target_points
            return

        merged = self._merge_existing_and_new_points(tr.points, candidates)
        old_count = len(tr.points)
        tr.points = merged
        tr.confidence = min(1.0, len(tr.points) / self.cfg.target_points)

        if len(tr.points) < self.cfg.min_points_after_refill:
            tr.needs_yolo = True
            tr.needs_refill = False
            tr.status = f"refill_failed_{old_count}_to_{len(tr.points)}"
            return

        tr.needs_yolo = False
        tr.needs_refill = False
        tr.status = f"safe_refilled_{old_count}_to_{len(tr.points)}"

    def _merge_existing_and_new_points(self, existing_points: np.ndarray, candidate_points) -> np.ndarray:
        merged = []
        if existing_points is not None:
            for p in existing_points:
                merged.append((float(p[0]), float(p[1])))

        for p in candidate_points:
            if len(merged) >= self.cfg.target_points:
                break
            px, py = float(p[0]), float(p[1])
            too_close = any(np.hypot(px - qx, py - qy) < self.cfg.min_distance_new_point for qx, qy in merged)
            if not too_close:
                merged.append((px, py))

        return np.array(merged[:self.cfg.target_points], dtype=np.float32)
