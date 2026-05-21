import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# PATHS
# =========================

VIDEO_IN_PATH = "input.mp4"
VIDEO_OUT_PATH = "output_klt_graph.mp4"
YOLO_MODEL_PATH = "yolov8n.pt"


# =========================
# YOLO SETTINGS
# =========================

PERSON_CLASS_ID = 0
YOLO_CONF = 0.35
YOLO_IMGSZ = 640
YOLO_DEVICE = None              # None, "cpu", "cuda:0"
YOLO_PERIOD_FRAMES = 10
IOU_MATCH_THRESHOLD = 0.30
TRACK_DELETE_MISSES = 5


# =========================
# POINT GRID SETTINGS
# =========================

GRID_COLS = 5
GRID_ROWS = 6
GRID_MARGIN_X = 0.08
GRID_MARGIN_Y = 0.06

MAX_BAD_COUNT = 3
MIN_ACTIVE_POINTS = 12
MAX_DEAD_FRACTION = 0.55

CONF_GOOD_GAIN = 1.0
CONF_BAD_PENALTY = 1.0


# =========================
# KLT SETTINGS
# =========================

KLT_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FB_MAX_ERROR = 2.5


# =========================
# GRAPH CONSISTENCY SETTINGS
# =========================

R_MOTION_PX = 4.0
GRAPH_SPATIAL_FRAC = 0.35
R_LOCAL_SCALE_ERR = 0.20

MIN_GROUP_POINTS = 4
TARGET_GROUP_POINTS = 8

SIGMA_MOTION = 2.5
SIGMA_SHAPE = 0.08

MIN_EXTENT_X = 0.12
MIN_EXTENT_Y = 0.16

BG_MOTION_THRESHOLD = 0.8
STATIC_COMPONENT_FACTOR = 0.65

EDGE_REGION_FRAC = 0.15
PART_EXTENT_X = 0.10
PART_EXTENT_Y = 0.12
PART_SCORE_FACTOR = 0.65

ACCEPT_SCORE = 0.35
CANDIDATE_SCORE = 0.18
CANDIDATE_MOTION_DAMP = 0.5


# =========================
# BBOX UPDATE SETTINGS
# =========================

ENABLE_SCALE_UPDATE = True
MIN_SCALE_POINTS = 5
SCALE_MIN_PER_FRAME = 0.98
SCALE_MAX_PER_FRAME = 1.02
SCALE_TOTAL_MIN = 0.85
SCALE_TOTAL_MAX = 1.15


# =========================
# POINT RESTORE SETTINGS
# =========================

RESTORE_WHEN_ACTIVE_BELOW = GRID_COLS * GRID_ROWS
MAX_RESTORE_PER_FRAME = 6

RESTORE_RADIUS = 12
RESTORE_MAX_CORNERS = 5
GFTT_QUALITY = 0.01
GFTT_MIN_DISTANCE = 3
REPLACEMENT_PROBATION_FRAMES = 2


# =========================
# DRAW SETTINGS
# =========================

DRAW_POINTS = True
DRAW_TRACK_INFO = True


@dataclass
class PointSlot:
    u: float
    v: float
    pt: np.ndarray
    active: bool = True
    bad_count: int = 0
    confidence: float = 0.0
    probation: int = 0


@dataclass
class Observation:
    slot_idx: int
    old: np.ndarray
    new: np.ndarray
    motion: np.ndarray
    fb_error: float


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    yolo_ref_bbox: np.ndarray
    slots: list
    color: tuple
    frames_since_yolo: int = 0
    missed_yolo: int = 0
    needs_yolo: bool = False
    last_score: float = 0.0
    last_group_size: int = 0


def clip_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox.astype(float)
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    x2 = np.clip(x2, 0, w - 1)
    y2 = np.clip(y2, 0, h - 1)

    if x2 <= x1 + 2:
        x2 = min(w - 1, x1 + 2)
    if y2 <= y1 + 2:
        y2 = min(h - 1, y1 + 2)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def bbox_wh(bbox):
    return max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])


def bbox_center(bbox):
    return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter + 1e-6
    return inter / union


def scale_bbox_about_center(bbox, scale):
    c = bbox_center(bbox)
    w, h = bbox_wh(bbox)
    nw = w * scale
    nh = h * scale

    return np.array([
        c[0] - nw * 0.5,
        c[1] - nh * 0.5,
        c[0] + nw * 0.5,
        c[1] + nh * 0.5,
    ], dtype=np.float32)


def create_grid_slots(bbox):
    x1, y1, x2, y2 = bbox
    w, h = bbox_wh(bbox)

    slots = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            u = GRID_MARGIN_X + (col + 0.5) * (1.0 - 2.0 * GRID_MARGIN_X) / GRID_COLS
            v = GRID_MARGIN_Y + (row + 0.5) * (1.0 - 2.0 * GRID_MARGIN_Y) / GRID_ROWS

            x = x1 + u * w
            y = y1 + v * h

            slots.append(PointSlot(
                u=u,
                v=v,
                pt=np.array([x, y], dtype=np.float32),
            ))

    return slots


def make_color(track_id):
    rng = np.random.default_rng(track_id * 12345)
    return tuple(int(x) for x in rng.integers(60, 255, size=3))


def create_track(track_id, bbox):
    return Track(
        track_id=track_id,
        bbox=bbox.copy(),
        yolo_ref_bbox=bbox.copy(),
        slots=create_grid_slots(bbox),
        color=make_color(track_id),
    )


def detect_persons(model, frame):
    results = model.predict(
        frame,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        device=YOLO_DEVICE,
        verbose=False,
    )

    detections = []
    r = results[0]

    if r.boxes is None:
        return detections

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()

    for box, class_id, score in zip(xyxy, cls, conf):
        if class_id != PERSON_CLASS_ID:
            continue

        detections.append((box.astype(np.float32), float(score)))

    detections.sort(key=lambda x: x[1], reverse=True)
    return detections


def klt_observations(track, prev_gray, curr_gray):
    active_indices = []
    p0 = []

    for idx, slot in enumerate(track.slots):
        if slot.active:
            active_indices.append(idx)
            p0.append(slot.pt)

    if not p0:
        return []

    p0 = np.asarray(p0, dtype=np.float32).reshape(-1, 1, 2)

    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **KLT_PARAMS)
    p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1, None, **KLT_PARAMS)

    observations = []

    for local_idx, slot_idx in enumerate(active_indices):
        slot = track.slots[slot_idx]

        ok = bool(st1[local_idx][0]) and bool(st2[local_idx][0])
        old = p0[local_idx, 0]
        new = p1[local_idx, 0]
        back = p0_back[local_idx, 0]

        fb_error = float(np.linalg.norm(back - old))

        if not ok or fb_error > FB_MAX_ERROR:
            slot.bad_count += 1
            slot.confidence -= CONF_BAD_PENALTY
            continue

        motion = new - old

        observations.append(Observation(
            slot_idx=slot_idx,
            old=old.copy(),
            new=new.copy(),
            motion=motion.copy(),
            fb_error=fb_error,
        ))

    return observations


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.p[rb] = ra


def graph_components(observations, bbox):
    n = len(observations)
    if n < MIN_GROUP_POINTS:
        return []

    dsu = DSU(n)

    w, h = bbox_wh(bbox)
    bbox_diag = math.sqrt(w * w + h * h)
    max_spatial = GRAPH_SPATIAL_FRAC * bbox_diag

    for i in range(n):
        oi = observations[i]

        for j in range(i + 1, n):
            oj = observations[j]

            d_motion = np.linalg.norm(oi.motion - oj.motion)
            if d_motion > R_MOTION_PX:
                continue

            d_old = np.linalg.norm(oi.old - oj.old)
            if d_old < 2.0 or d_old > max_spatial:
                continue

            d_new = np.linalg.norm(oi.new - oj.new)
            local_scale_err = abs(d_new / d_old - 1.0)

            if local_scale_err > R_LOCAL_SCALE_ERR:
                continue

            dsu.union(i, j)

    groups = {}
    for i in range(n):
        root = dsu.find(i)
        groups.setdefault(root, []).append(i)

    return [g for g in groups.values() if len(g) >= MIN_GROUP_POINTS]


def median_motion(observations, group):
    motions = np.array([observations[i].motion for i in group], dtype=np.float32)
    return np.median(motions, axis=0)


def estimate_group_scale(observations, group):
    ratios = []

    for a in range(len(group)):
        for b in range(a + 1, len(group)):
            oi = observations[group[a]]
            oj = observations[group[b]]

            d_old = np.linalg.norm(oi.old - oj.old)
            if d_old < 2.0:
                continue

            d_new = np.linalg.norm(oi.new - oj.new)
            ratios.append(d_new / d_old)

    if not ratios:
        return 1.0, 0.0

    ratios = np.asarray(ratios, dtype=np.float32)
    scale = float(np.median(ratios))
    shape_err = float(np.median(np.abs(ratios - scale)))

    return scale, shape_err


def group_score(observations, group, bbox):
    x1, y1, x2, y2 = bbox
    w, h = bbox_wh(bbox)

    v_g = median_motion(observations, group)

    residuals = []
    xy_norm = []

    for idx in group:
        o = observations[idx]
        residuals.append(np.linalg.norm(o.motion - v_g))
        xy_norm.append([
            (o.old[0] - x1) / w,
            (o.old[1] - y1) / h,
        ])

    residuals = np.asarray(residuals, dtype=np.float32)
    xy_norm = np.asarray(xy_norm, dtype=np.float32)

    motion_spread = float(np.median(residuals))
    motion_score = math.exp(-((motion_spread / SIGMA_MOTION) ** 2))

    scale, shape_err = estimate_group_scale(observations, group)
    shape_score = math.exp(-((shape_err / SIGMA_SHAPE) ** 2))

    size_score = min(1.0, len(group) / TARGET_GROUP_POINTS)

    extent_x = float(np.max(xy_norm[:, 0]) - np.min(xy_norm[:, 0]))
    extent_y = float(np.max(xy_norm[:, 1]) - np.min(xy_norm[:, 1]))

    spatial_score = min(1.0, 0.5 * (extent_x / MIN_EXTENT_X + extent_y / MIN_EXTENT_Y))

    motion_mag = float(np.linalg.norm(v_g))
    static_factor = STATIC_COMPONENT_FACTOR if motion_mag < BG_MOTION_THRESHOLD else 1.0

    centroid_x = float(np.mean(xy_norm[:, 0]))
    near_side = centroid_x < EDGE_REGION_FRAC or centroid_x > 1.0 - EDGE_REGION_FRAC
    too_local = extent_x < PART_EXTENT_X and extent_y < PART_EXTENT_Y

    part_factor = PART_SCORE_FACTOR if near_side and too_local else 1.0

    scale_factor = 1.0
    if scale < 0.94 or scale > 1.06:
        scale_factor = 0.5

    score = (
        motion_score
        * shape_score
        * size_score
        * spatial_score
        * static_factor
        * part_factor
        * scale_factor
    )

    return score, v_g, scale


def find_best_body_motion_group(observations, bbox):
    components = graph_components(observations, bbox)

    best = None
    best_score = -1.0
    best_motion = np.array([0.0, 0.0], dtype=np.float32)
    best_scale = 1.0

    for group in components:
        score, motion, scale = group_score(observations, group, bbox)

        if score > best_score:
            best = group
            best_score = score
            best_motion = motion
            best_scale = scale

    return best, best_score, best_motion, best_scale


def apply_bbox_motion(track, motion, scale, frame_w, frame_h, allow_scale):
    track.bbox[0] += motion[0]
    track.bbox[2] += motion[0]
    track.bbox[1] += motion[1]
    track.bbox[3] += motion[1]

    if ENABLE_SCALE_UPDATE and allow_scale:
        scale = float(np.clip(scale, SCALE_MIN_PER_FRAME, SCALE_MAX_PER_FRAME))

        ref_w, ref_h = bbox_wh(track.yolo_ref_bbox)
        cur_w, cur_h = bbox_wh(track.bbox)

        total_scale_w = (cur_w * scale) / ref_w
        total_scale_h = (cur_h * scale) / ref_h

        if SCALE_TOTAL_MIN <= total_scale_w <= SCALE_TOTAL_MAX and SCALE_TOTAL_MIN <= total_scale_h <= SCALE_TOTAL_MAX:
            track.bbox = scale_bbox_about_center(track.bbox, scale)

    track.bbox = clip_bbox(track.bbox, frame_w, frame_h)


def restore_lost_slots(track, gray):
    active_count = sum(1 for s in track.slots if s.active)
    if active_count >= RESTORE_WHEN_ACTIVE_BELOW:
        return

    h, w = gray.shape[:2]
    restored = 0

    x1, y1, x2, y2 = track.bbox
    bw, bh = bbox_wh(track.bbox)

    for slot in track.slots:
        if restored >= MAX_RESTORE_PER_FRAME:
            break

        if slot.active:
            continue

        x_pred = x1 + slot.u * bw
        y_pred = y1 + slot.v * bh

        rx1 = int(max(0, x_pred - RESTORE_RADIUS))
        ry1 = int(max(0, y_pred - RESTORE_RADIUS))
        rx2 = int(min(w - 1, x_pred + RESTORE_RADIUS))
        ry2 = int(min(h - 1, y_pred + RESTORE_RADIUS))

        if rx2 <= rx1 + 2 or ry2 <= ry1 + 2:
            continue

        roi = gray[ry1:ry2, rx1:rx2]

        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=RESTORE_MAX_CORNERS,
            qualityLevel=GFTT_QUALITY,
            minDistance=GFTT_MIN_DISTANCE,
        )

        if corners is None:
            continue

        corners = corners.reshape(-1, 2)
        corners[:, 0] += rx1
        corners[:, 1] += ry1

        distances = np.linalg.norm(corners - np.array([x_pred, y_pred], dtype=np.float32), axis=1)
        best = corners[int(np.argmin(distances))]

        slot.pt = best.astype(np.float32)
        slot.active = True
        slot.bad_count = 0
        slot.confidence = 0.0
        slot.probation = REPLACEMENT_PROBATION_FRAMES

        restored += 1


def update_track_with_klt(track, prev_gray, curr_gray, frame_w, frame_h):
    observations = klt_observations(track, prev_gray, curr_gray)

    if len(observations) < MIN_GROUP_POINTS:
        track.needs_yolo = True
        track.last_score = 0.0
        track.last_group_size = 0
        return

    best_group, score, motion, scale = find_best_body_motion_group(observations, track.bbox)

    track.last_score = float(score)
    track.last_group_size = 0 if best_group is None else len(best_group)

    if best_group is None or score < CANDIDATE_SCORE:
        for obs in observations:
            slot = track.slots[obs.slot_idx]
            slot.pt = obs.new
            slot.bad_count += 1
            slot.confidence -= CONF_BAD_PENALTY

        track.needs_yolo = True

    else:
        if score >= ACCEPT_SCORE:
            used_motion = motion
            allow_scale = len(best_group) >= MIN_SCALE_POINTS
            track.needs_yolo = False
        else:
            used_motion = motion * CANDIDATE_MOTION_DAMP
            allow_scale = False
            track.needs_yolo = True

        apply_bbox_motion(track, used_motion, scale, frame_w, frame_h, allow_scale)

        group_set = set(best_group)

        for local_idx, obs in enumerate(observations):
            slot = track.slots[obs.slot_idx]
            slot.pt = obs.new

            if local_idx in group_set:
                slot.bad_count = 0
                slot.confidence += CONF_GOOD_GAIN
                if slot.probation > 0:
                    slot.probation -= 1
            else:
                slot.bad_count += 1
                slot.confidence -= CONF_BAD_PENALTY

    for slot in track.slots:
        if slot.active and slot.bad_count >= MAX_BAD_COUNT:
            slot.active = False

    restore_lost_slots(track, curr_gray)

    active_count = sum(1 for s in track.slots if s.active)
    dead_fraction = 1.0 - active_count / max(1, len(track.slots))

    if active_count < MIN_ACTIVE_POINTS:
        track.needs_yolo = True

    if dead_fraction > MAX_DEAD_FRACTION:
        track.needs_yolo = True

    track.frames_since_yolo += 1

    if track.frames_since_yolo >= YOLO_PERIOD_FRAMES:
        track.needs_yolo = True


def associate_yolo_detections(tracks, detections, next_track_id, frame_shape):
    h, w = frame_shape[:2]
    used_dets = set()

    for track in tracks:
        best_iou = 0.0
        best_det_idx = -1

        for det_idx, (det_bbox, _) in enumerate(detections):
            if det_idx in used_dets:
                continue

            iou = bbox_iou(track.bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_det_idx = det_idx

        if best_det_idx >= 0 and best_iou >= IOU_MATCH_THRESHOLD:
            det_bbox = detections[best_det_idx][0]
            det_bbox = clip_bbox(det_bbox, w, h)

            track.bbox = det_bbox.copy()
            track.yolo_ref_bbox = det_bbox.copy()
            track.slots = create_grid_slots(det_bbox)
            track.frames_since_yolo = 0
            track.missed_yolo = 0
            track.needs_yolo = False
            track.last_score = 1.0
            track.last_group_size = len(track.slots)

            used_dets.add(best_det_idx)

        else:
            if track.needs_yolo:
                track.missed_yolo += 1

    alive_tracks = [
        t for t in tracks
        if t.missed_yolo <= TRACK_DELETE_MISSES
    ]

    for det_idx, (det_bbox, _) in enumerate(detections):
        if det_idx in used_dets:
            continue

        det_bbox = clip_bbox(det_bbox, w, h)
        alive_tracks.append(create_track(next_track_id, det_bbox))
        next_track_id += 1

    return alive_tracks, next_track_id


def draw_tracks(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2 = track.bbox.astype(int)
        color = track.color

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if DRAW_POINTS:
            for slot in track.slots:
                if not slot.active:
                    continue

                px, py = slot.pt.astype(int)
                point_color = color if slot.probation == 0 else (0, 255, 255)
                cv2.circle(frame, (px, py), 2, point_color, -1)

        if DRAW_TRACK_INFO:
            active_count = sum(1 for s in track.slots if s.active)
            label = (
                f"ID {track.track_id} "
                f"pts {active_count}/{len(track.slots)} "
                f"g {track.last_group_size} "
                f"s {track.last_score:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    return frame


def main():
    model = YOLO(YOLO_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_IN_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_IN_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        VIDEO_OUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Empty video")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = detect_persons(model, frame)

    tracks = []
    next_track_id = 1

    for det_bbox, _ in detections:
        det_bbox = clip_bbox(det_bbox, frame_w, frame_h)
        tracks.append(create_track(next_track_id, det_bbox))
        next_track_id += 1

    draw_tracks(frame, tracks)
    writer.write(frame)

    frame_idx = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for track in tracks:
            update_track_with_klt(track, prev_gray, curr_gray, frame_w, frame_h)

        should_run_yolo = (
            frame_idx % YOLO_PERIOD_FRAMES == 0
            or any(t.needs_yolo for t in tracks)
        )

        if should_run_yolo:
            detections = detect_persons(model, frame)
            tracks, next_track_id = associate_yolo_detections(
                tracks,
                detections,
                next_track_id,
                frame.shape,
            )

        draw_tracks(frame, tracks)
        writer.write(frame)

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Saved: {VIDEO_OUT_PATH}")


if __name__ == "__main__":
    main()