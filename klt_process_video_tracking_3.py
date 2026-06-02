from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from model_and_selector import (
    MicroNeXtMaskNet,
    SimplePointSelectorConfig,
    select_points_simple,
)


# =========================
# PATHS
# =========================

VIDEO_PATH = "input.mp4"
OUTPUT_VIDEO_PATH = "output_tracked.mp4"

YOLO_MODEL_PATH = "yolov8n.pt"
STUDENT_CHECKPOINT_PATH = "runs/student_mask/exp_simple_points_v1/best.pt"


# =========================
# MAIN SETTINGS
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_DEVICE = 0 if DEVICE == "cuda" else "cpu"
YOLO_CONF = 0.25
YOLO_IMGSZ = 640
PERSON_CLASS_ID = 0

YOLO_INTERVAL_SECONDS = 3.0

BOX_SCALE = 1.25

INPUT_W = 64
INPUT_H = 128

TARGET_POINTS = 5

POINT_SELECTOR_CFG = SimplePointSelectorConfig(
    mask_threshold=0.80,
    torso_y_min=0.03,
    torso_y_max=0.60,
    max_points=5,
)

# KLT
KLT_WIN_SIZE = (21, 21)
KLT_MAX_LEVEL = 3
KLT_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    30,
    0.01,
)

FB_MAX_ERROR = 1.5

# Point logic
MIN_POINTS_FOR_KLT = 3

ENABLE_SAFE_REFILL = True
REFILL_ONLY_WHEN_POINTS_EQUAL = 3
MIN_POINTS_INSIDE_MASK_FOR_REFILL = 3
MIN_POINTS_AFTER_REFILL = 4
MIN_DISTANCE_NEW_POINT = 5.0

# Motion consistency
MOTION_RESIDUAL_MIN = 2.0
MOTION_RESIDUAL_REL = 0.04
MOTION_RESIDUAL_MAX = 8.0

# YOLO association
MATCH_IOU_MIN = 0.25
MATCH_CENTER_NORM_MAX = 0.55

MAX_YOLO_MISSES = 2

DRAW_POINTS = True
DRAW_WORK_BOX = False

student_runs = 0


# =========================
# DATA STRUCTURES
# =========================

@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))

    age: int = 0
    missed_yolo: int = 0
    last_yolo_frame: int = 0
    last_update_frame: int = 0

    confidence: float = 1.0
    needs_yolo: bool = False
    needs_refill: bool = False
    status: str = "init"


# =========================
# BOX UTILS
# =========================

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

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    union = box_area(a) + box_area(b) - inter

    if union <= 1e-9:
        return 0.0

    return float(inter / union)


def box_center(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def box_diag(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return float(np.hypot(x2 - x1, y2 - y1))


def expand_box(box: np.ndarray, img_w: int, img_h: int, scale: float = BOX_SCALE) -> np.ndarray:
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    new_w = bw * scale
    new_h = bh * scale

    expanded = np.array([
        cx - new_w * 0.5,
        cy - new_h * 0.5,
        cx + new_w * 0.5,
        cy + new_h * 0.5,
    ], dtype=np.float32)

    return clip_box(expanded, img_w, img_h)


def int_crop_box(box: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    box = clip_box(box, img_w, img_h)

    x1, y1, x2, y2 = box

    x1 = int(np.floor(x1))
    y1 = int(np.floor(y1))
    x2 = int(np.ceil(x2))
    y2 = int(np.ceil(y2))

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(x1 + 1, min(img_w, x2))
    y2 = max(y1 + 1, min(img_h, y2))

    return x1, y1, x2, y2


def shift_box(box: np.ndarray, dx: float, dy: float, img_w: int, img_h: int) -> np.ndarray:
    shifted = box.copy().astype(np.float32)
    shifted[[0, 2]] += dx
    shifted[[1, 3]] += dy
    return clip_box(shifted, img_w, img_h)


# =========================
# LETTERBOX UTILS
# =========================

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

    transform = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "new_w": new_w,
        "new_h": new_h,
    }

    return canvas, transform


def letterbox_point_to_crop(point_xy: Tuple[float, float], transform) -> Optional[Tuple[float, float]]:
    x, y = point_xy

    scale = transform["scale"]
    pad_x = transform["pad_x"]
    pad_y = transform["pad_y"]
    new_w = transform["new_w"]
    new_h = transform["new_h"]

    if x < pad_x or y < pad_y or x >= pad_x + new_w or y >= pad_y + new_h:
        return None

    cx = (x - pad_x) / scale
    cy = (y - pad_y) / scale

    return float(cx), float(cy)


def frame_point_to_letterbox(
    point_xy: Tuple[float, float],
    crop_origin: Tuple[int, int],
    transform,
) -> Optional[Tuple[int, int]]:
    fx, fy = point_xy
    ox, oy = crop_origin

    crop_x = fx - ox
    crop_y = fy - oy

    lx = crop_x * transform["scale"] + transform["pad_x"]
    ly = crop_y * transform["scale"] + transform["pad_y"]

    lx = int(round(lx))
    ly = int(round(ly))

    if lx < 0 or ly < 0 or lx >= INPUT_W or ly >= INPUT_H:
        return None

    return lx, ly


# =========================
# MODEL INFERENCE
# =========================

def load_student_model(path: str) -> MicroNeXtMaskNet:
    model = MicroNeXtMaskNet().to(DEVICE)

    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model


@torch.no_grad()
def predict_points_and_mask_for_bbox(
    frame_bgr: np.ndarray,
    bbox: np.ndarray,
    student_model: MicroNeXtMaskNet,
):
    global student_runs
    student_runs += 1

    img_h, img_w = frame_bgr.shape[:2]

    work_box = expand_box(bbox, img_w, img_h, BOX_SCALE)
    x1, y1, x2, y2 = int_crop_box(work_box, img_w, img_h)

    crop_bgr = frame_bgr[y1:y2, x1:x2]

    if crop_bgr.size == 0:
        return [], None, None, None

    letter_bgr, transform = letterbox_image(crop_bgr, INPUT_W, INPUT_H, pad_value=114)
    letter_rgb = cv2.cvtColor(letter_bgr, cv2.COLOR_BGR2RGB)

    inp = letter_rgb.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None]

    tensor = torch.from_numpy(inp).float().to(DEVICE)

    logits = student_model(tensor)

    pred = torch.sigmoid(logits)
    pred_up = F.interpolate(
        pred,
        size=(INPUT_H, INPUT_W),
        mode="bilinear",
        align_corners=False,
    )

    pred_mask = pred_up[0, 0].detach().cpu().numpy().astype(np.float32)

    points_letter = select_points_simple(
        image_rgb=letter_rgb,
        pred_mask=pred_mask,
        cfg=POINT_SELECTOR_CFG,
    )

    frame_points = []

    for p in points_letter:
        crop_point = letterbox_point_to_crop(p, transform)

        if crop_point is None:
            continue

        px, py = crop_point
        fx = x1 + px
        fy = y1 + py

        if 0 <= fx < img_w and 0 <= fy < img_h:
            frame_points.append((float(fx), float(fy)))

    return frame_points, pred_mask, transform, (x1, y1)


def predict_points_for_bbox(
    frame_bgr: np.ndarray,
    bbox: np.ndarray,
    student_model: MicroNeXtMaskNet,
):
    points, _, _, _ = predict_points_and_mask_for_bbox(
        frame_bgr=frame_bgr,
        bbox=bbox,
        student_model=student_model,
    )
    return points


def count_points_inside_pred_mask(
    points_frame: np.ndarray,
    pred_mask: np.ndarray,
    crop_origin: Tuple[int, int],
    transform,
) -> int:
    if pred_mask is None or transform is None or crop_origin is None:
        return 0

    count = 0

    for p in points_frame:
        lp = frame_point_to_letterbox(
            point_xy=(float(p[0]), float(p[1])),
            crop_origin=crop_origin,
            transform=transform,
        )

        if lp is None:
            continue

        lx, ly = lp

        if pred_mask[ly, lx] >= POINT_SELECTOR_CFG.mask_threshold:
            count += 1

    return count


# =========================
# YOLO + ASSOCIATION
# =========================

def run_yolo_person_detector(yolo_model: YOLO, frame_bgr: np.ndarray) -> List[np.ndarray]:
    results = yolo_model.predict(
        source=frame_bgr,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        classes=[PERSON_CLASS_ID],
        device=YOLO_DEVICE,
        verbose=False,
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confs = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

    detections = []

    for box, conf in zip(boxes, confs):
        if conf < YOLO_CONF:
            continue
        detections.append(box.astype(np.float32))

    return detections


def association_score(track_box: np.ndarray, det_box: np.ndarray) -> Tuple[bool, float]:
    iou = box_iou(track_box, det_box)

    c1 = box_center(track_box)
    c2 = box_center(det_box)

    dist = float(np.linalg.norm(c1 - c2))
    norm = max(1.0, box_diag(track_box))

    center_norm = dist / norm

    ok = (iou >= MATCH_IOU_MIN) or (center_norm <= MATCH_CENTER_NORM_MAX)

    score = iou + max(0.0, 1.0 - center_norm) * 0.25

    return ok, float(score)


def match_detections_to_tracks(tracks: List[Track], detections: List[np.ndarray]):
    pairs = []

    for ti, tr in enumerate(tracks):
        for di, det in enumerate(detections):
            ok, score = association_score(tr.bbox, det)

            if ok:
                pairs.append((score, ti, di))

    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_tracks = set()
    matched_dets = set()
    matches = []

    for score, ti, di in pairs:
        if ti in matched_tracks or di in matched_dets:
            continue

        matched_tracks.add(ti)
        matched_dets.add(di)
        matches.append((ti, di, score))

    unmatched_tracks = [
        i for i in range(len(tracks))
        if i not in matched_tracks
    ]

    unmatched_dets = [
        i for i in range(len(detections))
        if i not in matched_dets
    ]

    return matches, unmatched_tracks, unmatched_dets


def refresh_tracks_with_yolo(
    frame_bgr: np.ndarray,
    frame_idx: int,
    tracks: List[Track],
    detections: List[np.ndarray],
    student_model: MicroNeXtMaskNet,
    next_track_id: int,
):
    img_h, img_w = frame_bgr.shape[:2]

    matches, unmatched_tracks, unmatched_dets = match_detections_to_tracks(tracks, detections)

    alive_tracks: List[Track] = []

    for ti, di, score in matches:
        tr = tracks[ti]
        det = clip_box(detections[di], img_w, img_h)

        tr.bbox = det
        tr.missed_yolo = 0
        tr.last_yolo_frame = frame_idx
        tr.last_update_frame = frame_idx
        tr.needs_yolo = False
        tr.needs_refill = False
        tr.status = "yolo_match"

        points = predict_points_for_bbox(frame_bgr, tr.bbox, student_model)

        tr.points = np.array(points, dtype=np.float32)
        tr.confidence = min(1.0, len(points) / TARGET_POINTS)

        if len(tr.points) < MIN_POINTS_FOR_KLT:
            tr.needs_yolo = True
            tr.status = "yolo_match_but_few_points"

        alive_tracks.append(tr)

    for ti in unmatched_tracks:
        tr = tracks[ti]
        tr.missed_yolo += 1

        if tr.missed_yolo <= MAX_YOLO_MISSES:
            tr.status = "yolo_missed_keep"
            tr.needs_yolo = True
            tr.needs_refill = False
            alive_tracks.append(tr)

    for di in unmatched_dets:
        det = clip_box(detections[di], img_w, img_h)

        points = predict_points_for_bbox(frame_bgr, det, student_model)

        tr = Track(
            track_id=next_track_id,
            bbox=det,
            points=np.array(points, dtype=np.float32),
            age=0,
            missed_yolo=0,
            last_yolo_frame=frame_idx,
            last_update_frame=frame_idx,
            confidence=min(1.0, len(points) / TARGET_POINTS),
            needs_yolo=False,
            needs_refill=False,
            status="new_yolo",
        )

        if len(tr.points) < MIN_POINTS_FOR_KLT:
            tr.needs_yolo = True
            tr.status = "new_yolo_but_few_points"

        next_track_id += 1
        alive_tracks.append(tr)

    return alive_tracks, next_track_id


# =========================
# POINT REFILL
# =========================

def merge_existing_and_new_points(
    existing_points: np.ndarray,
    candidate_points,
    max_points: int = TARGET_POINTS,
    min_dist: float = MIN_DISTANCE_NEW_POINT,
):
    merged = []

    if existing_points is not None:
        for p in existing_points:
            merged.append((float(p[0]), float(p[1])))

    for p in candidate_points:
        if len(merged) >= max_points:
            break

        px, py = float(p[0]), float(p[1])

        too_close = False

        for qx, qy in merged:
            if np.hypot(px - qx, py - qy) < min_dist:
                too_close = True
                break

        if not too_close:
            merged.append((px, py))

    return np.array(merged[:max_points], dtype=np.float32)


def safe_refill_points_if_needed(
    frame_bgr: np.ndarray,
    track: Track,
    student_model: MicroNeXtMaskNet,
):
    if not ENABLE_SAFE_REFILL:
        return

    if not track.needs_refill:
        return

    if track.points is None or len(track.points) != REFILL_ONLY_WHEN_POINTS_EQUAL:
        return

    candidates, pred_mask, transform, crop_origin = predict_points_and_mask_for_bbox(
        frame_bgr=frame_bgr,
        bbox=track.bbox,
        student_model=student_model,
    )

    inside_old = count_points_inside_pred_mask(
        points_frame=track.points,
        pred_mask=pred_mask,
        crop_origin=crop_origin,
        transform=transform,
    )

    if inside_old < MIN_POINTS_INSIDE_MASK_FOR_REFILL:
        track.needs_yolo = True
        track.needs_refill = False
        track.status = f"refill_rejected_mask_{inside_old}_of_{len(track.points)}"
        track.confidence = inside_old / TARGET_POINTS
        return

    merged = merge_existing_and_new_points(
        existing_points=track.points,
        candidate_points=candidates,
        max_points=TARGET_POINTS,
        min_dist=MIN_DISTANCE_NEW_POINT,
    )

    old_count = len(track.points)
    track.points = merged
    track.confidence = min(1.0, len(track.points) / TARGET_POINTS)

    if len(track.points) < MIN_POINTS_AFTER_REFILL:
        track.needs_yolo = True
        track.needs_refill = False
        track.status = f"refill_failed_{old_count}_to_{len(track.points)}"
        return

    track.needs_yolo = False
    track.needs_refill = False
    track.status = f"safe_refilled_{old_count}_to_{len(track.points)}"


# =========================
# KLT TRACKING
# =========================

def klt_update_track(
    track: Track,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    img_w: int,
    img_h: int,
):
    track.needs_refill = False

    if track.points is None or len(track.points) < MIN_POINTS_FOR_KLT:
        track.needs_yolo = True
        track.status = "too_few_points_before_klt"
        track.confidence = 0.0
        return

    old_pts = track.points.astype(np.float32).reshape(-1, 1, 2)

    new_pts, st1, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        old_pts,
        None,
        winSize=KLT_WIN_SIZE,
        maxLevel=KLT_MAX_LEVEL,
        criteria=KLT_CRITERIA,
    )

    if new_pts is None or st1 is None:
        track.needs_yolo = True
        track.status = "klt_failed"
        track.confidence = 0.0
        return

    back_pts, st2, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray,
        prev_gray,
        new_pts,
        None,
        winSize=KLT_WIN_SIZE,
        maxLevel=KLT_MAX_LEVEL,
        criteria=KLT_CRITERIA,
    )

    if back_pts is None or st2 is None:
        track.needs_yolo = True
        track.status = "klt_back_failed"
        track.confidence = 0.0
        return

    old = old_pts.reshape(-1, 2)
    new = new_pts.reshape(-1, 2)
    back = back_pts.reshape(-1, 2)

    st1 = st1.reshape(-1).astype(bool)
    st2 = st2.reshape(-1).astype(bool)

    fb_err = np.linalg.norm(old - back, axis=1)
    fb_good = st1 & st2 & (fb_err <= FB_MAX_ERROR)

    fb_good_count = int(fb_good.sum())

    if fb_good_count < MIN_POINTS_FOR_KLT:
        track.points = new[fb_good].astype(np.float32)
        track.needs_yolo = True
        track.status = f"too_few_fb_good_{fb_good_count}"
        track.confidence = fb_good_count / TARGET_POINTS
        return

    old_good = old[fb_good]
    new_good = new[fb_good]

    flow = new_good - old_good

    med = np.median(flow, axis=0)
    residual = np.linalg.norm(flow - med[None], axis=1)

    adaptive_thr = MOTION_RESIDUAL_REL * box_diag(track.bbox)
    adaptive_thr = max(MOTION_RESIDUAL_MIN, adaptive_thr)
    adaptive_thr = min(MOTION_RESIDUAL_MAX, adaptive_thr)

    inliers = residual <= adaptive_thr
    inlier_count = int(inliers.sum())

    if fb_good_count >= 5:
        required_inliers = 4
    elif fb_good_count == 4:
        required_inliers = 4
    elif fb_good_count == 3:
        required_inliers = 3
    else:
        required_inliers = MIN_POINTS_FOR_KLT

    if inlier_count < required_inliers:
        track.points = new_good[inliers].astype(np.float32)
        track.needs_yolo = True
        track.status = f"motion_not_consistent_{inlier_count}_of_{fb_good_count}"
        track.confidence = inlier_count / TARGET_POINTS
        return

    flow_in = flow[inliers]
    new_in = new_good[inliers]

    dx, dy = np.median(flow_in, axis=0)

    track.bbox = shift_box(track.bbox, float(dx), float(dy), img_w, img_h)
    track.points = new_in.astype(np.float32)

    track.needs_yolo = False
    track.confidence = min(1.0, len(track.points) / TARGET_POINTS)

    if len(track.points) == 3:
        track.needs_refill = True
        track.status = "klt_ok_3pts_needs_refill"
    else:
        track.needs_refill = False
        track.status = f"klt_ok_{len(track.points)}pts"


# =========================
# DRAWING
# =========================

def color_for_id(track_id: int):
    rng = np.random.default_rng(track_id * 12345)
    color = rng.integers(60, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_tracks(frame_bgr: np.ndarray, tracks: List[Track]):
    img_h, img_w = frame_bgr.shape[:2]

    for tr in tracks:
        color = color_for_id(tr.track_id)

        x1, y1, x2, y2 = int_crop_box(tr.bbox, img_w, img_h)

        if tr.needs_yolo:
            box_color = (0, 0, 255)
        elif tr.needs_refill:
            box_color = (0, 165, 255)
        else:
            box_color = color

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_color, 2)

        label = f"ID {tr.track_id} | pts {len(tr.points)} | {tr.status}"

        cv2.putText(
            frame_bgr,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )

        if DRAW_WORK_BOX:
            wb = expand_box(tr.bbox, img_w, img_h, BOX_SCALE)
            wx1, wy1, wx2, wy2 = int_crop_box(wb, img_w, img_h)
            cv2.rectangle(frame_bgr, (wx1, wy1), (wx2, wy2), (160, 160, 160), 1)

        if DRAW_POINTS and tr.points is not None:
            for p in tr.points:
                px, py = int(round(p[0])), int(round(p[1]))
                if 0 <= px < img_w and 0 <= py < img_h:
                    cv2.circle(frame_bgr, (px, py), 3, (0, 255, 0), -1)


def draw_stats(
    frame_bgr: np.ndarray,
    frame_idx: int,
    yolo_frames: int,
    student_runs_value: int,
    last_yolo_reason: str,
    tracks_count: int,
):
    processed = frame_idx + 1
    saved = processed - yolo_frames
    saved_pct = 100.0 * saved / max(1, processed)

    lines = [
        f"frames: {processed}",
        f"YOLO runs: {yolo_frames}",
        f"YOLO saved: {saved} ({saved_pct:.1f}%)",
        f"student runs: {student_runs_value}",
        f"tracks: {tracks_count}",
        f"last YOLO: {last_yolo_reason}",
    ]

    x = 12
    y = 24

    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24


# =========================
# MAIN LOOP
# =========================

def main():
    global student_runs
    student_runs = 0

    video_path = Path(VIDEO_PATH)
    output_path = Path(OUTPUT_VIDEO_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    student_model = load_student_model(STUDENT_CHECKPOINT_PATH)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    yolo_interval_frames = max(1, int(round(fps * YOLO_INTERVAL_SECONDS)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {OUTPUT_VIDEO_PATH}")

    print(f"Video: {VIDEO_PATH}")
    print(f"Output: {OUTPUT_VIDEO_PATH}")
    print(f"FPS: {fps:.2f}")
    print(f"Size: {width}x{height}")
    print(f"Total frames metadata: {total_frames}")
    print(f"YOLO interval frames: {yolo_interval_frames}")

    tracks: List[Track] = []
    next_track_id = 1

    prev_gray = None

    yolo_frames = 0
    last_yolo_reason = "none"

    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()

        if not ok:
            break

        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        scheduled_yolo = (
            frame_idx == 0
            or frame_idx % yolo_interval_frames == 0
            or len(tracks) == 0
        )

        if prev_gray is not None and len(tracks) > 0:
            for tr in tracks:
                klt_update_track(tr, prev_gray, curr_gray, width, height)

        need_yolo = any(tr.needs_yolo for tr in tracks)
        run_yolo_now = scheduled_yolo or need_yolo

        if run_yolo_now:
            if frame_idx == 0:
                last_yolo_reason = "first_frame"
            elif scheduled_yolo:
                last_yolo_reason = "scheduled"
            else:
                last_yolo_reason = "low_confidence"

            detections = run_yolo_person_detector(yolo_model, frame_bgr)

            tracks, next_track_id = refresh_tracks_with_yolo(
                frame_bgr=frame_bgr,
                frame_idx=frame_idx,
                tracks=tracks,
                detections=detections,
                student_model=student_model,
                next_track_id=next_track_id,
            )

            yolo_frames += 1

        else:
            for tr in tracks:
                if tr.needs_refill and not tr.needs_yolo:
                    safe_refill_points_if_needed(
                        frame_bgr=frame_bgr,
                        track=tr,
                        student_model=student_model,
                    )

            if any(tr.needs_yolo for tr in tracks):
                last_yolo_reason = "refill_rejected"

                detections = run_yolo_person_detector(yolo_model, frame_bgr)

                tracks, next_track_id = refresh_tracks_with_yolo(
                    frame_bgr=frame_bgr,
                    frame_idx=frame_idx,
                    tracks=tracks,
                    detections=detections,
                    student_model=student_model,
                    next_track_id=next_track_id,
                )

                yolo_frames += 1

        for tr in tracks:
            tr.age += 1
            tr.last_update_frame = frame_idx

        tracks = [
            tr for tr in tracks
            if tr.missed_yolo <= MAX_YOLO_MISSES
        ]

        draw_tracks(frame_bgr, tracks)
        draw_stats(
            frame_bgr=frame_bgr,
            frame_idx=frame_idx,
            yolo_frames=yolo_frames,
            student_runs_value=student_runs,
            last_yolo_reason=last_yolo_reason,
            tracks_count=len(tracks),
        )

        writer.write(frame_bgr)

        prev_gray = curr_gray
        frame_idx += 1

        if frame_idx % 100 == 0:
            processed = frame_idx
            saved = processed - yolo_frames
            saved_pct = 100.0 * saved / max(1, processed)

            print(
                f"frame={processed} "
                f"yolo_runs={yolo_frames} "
                f"student_runs={student_runs} "
                f"saved={saved} "
                f"saved_pct={saved_pct:.1f}% "
                f"tracks={len(tracks)}"
            )

    cap.release()
    writer.release()

    processed = frame_idx
    saved = processed - yolo_frames
    saved_pct = 100.0 * saved / max(1, processed)

    print()
    print("Done.")
    print(f"Processed frames: {processed}")
    print(f"Total frames metadata: {total_frames}")
    print(f"YOLO runs:        {yolo_frames}")
    print(f"Student runs:     {student_runs}")
    print(f"YOLO saved:       {saved}")
    print(f"YOLO saved pct:   {saved_pct:.2f}%")
    print(f"Output video:     {OUTPUT_VIDEO_PATH}")

    if total_frames > 0 and processed < total_frames:
        print()
        print("WARNING:")
        print(f"OpenCV read only {processed} frames, metadata says {total_frames}.")
        print("Possible reason: inaccurate MKV metadata or decoder stopped early.")


if __name__ == "__main__":
    main()