# klt_yolo_pose_two_points.py

import time
import cv2
import numpy as np
from ultralytics import YOLO


# Paths
VIDEO_PATH = "input.mp4"
MODEL_PATH = "yolov8n-pose.pt"
OUTPUT_PATH = "output_pose_klt.mp4"

# YOLO Pose settings
YOLO_CONF = 0.35
KP_CONF = 0.35

# KLT settings
MAX_FB_ERROR = 3.0
MIN_POINT_DIST = 5.0

# Static point check
STATIC_MOVE_PX = 0.8
MOVING_MOVE_PX = 4.0

# Box settings
MIN_SCALE = 0.70
MAX_SCALE = 1.40
BOX_PADDING = 20

# Visualization
SHOW_WINDOW = False
DRAW_POINTS = True
DRAW_BOXES = True

# Crosshair size
CROSS_RADIUS = 12
CROSS_HALF_LEN = 18
CROSS_THICKNESS = 3


# COCO keypoint indexes
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

HEAD_IDS = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
TORSO_IDS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]


# Lucas-Kanade params
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def clamp_bbox(bbox, frame_w, frame_h):
    # Keep bbox inside image
    x1, y1, x2, y2 = bbox

    x1 = int(max(0, min(frame_w - 1, x1)))
    y1 = int(max(0, min(frame_h - 1, y1)))
    x2 = int(max(0, min(frame_w - 1, x2)))
    y2 = int(max(0, min(frame_h - 1, y2)))

    if x2 <= x1:
        x2 = min(frame_w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_h - 1, y1 + 1)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def get_group_center(kpts_xy, kpts_conf, ids, min_count):
    # Compute center from visible keypoints
    pts = []

    for idx in ids:
        if idx >= len(kpts_xy):
            continue

        if kpts_conf[idx] >= KP_CONF:
            pts.append(kpts_xy[idx])

    if len(pts) < min_count:
        return None

    return np.median(np.array(pts, dtype=np.float32), axis=0)


def run_pose(model, frame, frame_w, frame_h):
    # Run YOLO Pose and build tracks
    result = model.predict(frame, conf=YOLO_CONF, verbose=False)[0]
    tracks = []

    if result.boxes is None or result.keypoints is None:
        return tracks

    boxes = result.boxes.xyxy.cpu().numpy()
    keypoints_xy = result.keypoints.xy.cpu().numpy()
    keypoints_conf = result.keypoints.conf.cpu().numpy()

    for i in range(len(boxes)):
        bbox = clamp_bbox(boxes[i], frame_w, frame_h)

        head = get_group_center(keypoints_xy[i], keypoints_conf[i], HEAD_IDS, min_count=1)
        torso = get_group_center(keypoints_xy[i], keypoints_conf[i], TORSO_IDS, min_count=2)

        if head is None or torso is None:
            continue

        dist = float(np.linalg.norm(head - torso))
        if dist < MIN_POINT_DIST:
            continue

        track = PoseKLTTrack(
            track_id=i + 1,
            bbox=bbox,
            head=head,
            torso=torso,
        )

        tracks.append(track)

    return tracks


def klt_track_two_points(prev_gray, gray, old_points):
    # Track two points with forward-backward check
    old_points = old_points.reshape(-1, 1, 2).astype(np.float32)

    new_points, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        old_points,
        None,
        **LK_PARAMS,
    )

    if new_points is None or st_fwd is None:
        return None

    back_points, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        prev_gray,
        new_points,
        None,
        **LK_PARAMS,
    )

    if back_points is None or st_bwd is None:
        return None

    old_xy = old_points.reshape(-1, 2)
    new_xy = new_points.reshape(-1, 2)
    back_xy = back_points.reshape(-1, 2)

    fb_error = np.linalg.norm(old_xy - back_xy, axis=1)

    valid = (
        (st_fwd.reshape(-1) == 1) &
        (st_bwd.reshape(-1) == 1) &
        (fb_error < MAX_FB_ERROR)
    )

    if np.sum(valid) != 2:
        return None

    return new_xy.astype(np.float32)


def is_static_mismatch(old_points, new_points):
    # Detect case when one point moves and the other stays static
    flows = np.linalg.norm(new_points - old_points, axis=1)

    head_move = flows[0]
    torso_move = flows[1]

    head_static = head_move < STATIC_MOVE_PX
    torso_static = torso_move < STATIC_MOVE_PX

    head_moving = head_move > MOVING_MOVE_PX
    torso_moving = torso_move > MOVING_MOVE_PX

    if head_static and torso_moving:
        return True

    if torso_static and head_moving:
        return True

    return False


def draw_crosshair(img, point, color=(0, 0, 255)):
    # Draw red circle and cross
    x, y = int(point[0]), int(point[1])

    cv2.circle(img, (x, y), CROSS_RADIUS, color, CROSS_THICKNESS)
    cv2.line(img, (x - CROSS_HALF_LEN, y), (x + CROSS_HALF_LEN, y), color, CROSS_THICKNESS)
    cv2.line(img, (x, y - CROSS_HALF_LEN), (x, y + CROSS_HALF_LEN), color, CROSS_THICKNESS)


def draw_top_right_stats(img, saved_percent, avg_klt_ms):
    # Draw red stats at top right
    h, w = img.shape[:2]

    lines = [
        f"YOLO saved: {saved_percent:.1f}%",
        f"Avg KLT: {avg_klt_ms:.2f} ms",
    ]

    x = w - 310
    y = 35

    for line in lines:
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        y += 32


class PoseKLTTrack:
    def __init__(self, track_id, bbox, head, torso):
        # Store YOLO box and two anchors
        self.id = track_id

        self.ref_bbox = bbox.astype(np.float32)
        self.bbox = bbox.astype(np.float32)

        self.head = head.astype(np.float32)
        self.torso = torso.astype(np.float32)

        self.ref_head = self.head.copy()
        self.ref_torso = self.torso.copy()

        self.ref_len = float(np.linalg.norm(self.ref_head - self.ref_torso))
        self.ref_w = float(self.ref_bbox[2] - self.ref_bbox[0])
        self.ref_h = float(self.ref_bbox[3] - self.ref_bbox[1])

        ref_anchor = (self.ref_head + self.ref_torso) * 0.5
        ref_center = np.array(
            [
                (self.ref_bbox[0] + self.ref_bbox[2]) * 0.5,
                (self.ref_bbox[1] + self.ref_bbox[3]) * 0.5,
            ],
            dtype=np.float32,
        )

        self.center_offset_norm = (ref_center - ref_anchor) / max(self.ref_len, 1.0)
        self.alive = True

    def points(self):
        # Return current two points
        return np.array([self.head, self.torso], dtype=np.float32)

    def update_bbox_from_points(self, frame_w, frame_h):
        # Scale bbox using head-torso distance
        curr_len = float(np.linalg.norm(self.head - self.torso))
        if curr_len < MIN_POINT_DIST:
            self.alive = False
            return

        scale = curr_len / max(self.ref_len, 1.0)
        scale = float(np.clip(scale, MIN_SCALE, MAX_SCALE))

        new_w = self.ref_w * scale
        new_h = self.ref_h * scale

        anchor = (self.head + self.torso) * 0.5
        center = anchor + self.center_offset_norm * curr_len

        x1 = center[0] - new_w * 0.5
        y1 = center[1] - new_h * 0.5
        x2 = center[0] + new_w * 0.5
        y2 = center[1] + new_h * 0.5

        # Ensure both points are inside the box
        px1 = min(self.head[0], self.torso[0]) - BOX_PADDING
        py1 = min(self.head[1], self.torso[1]) - BOX_PADDING
        px2 = max(self.head[0], self.torso[0]) + BOX_PADDING
        py2 = max(self.head[1], self.torso[1]) + BOX_PADDING

        x1 = min(x1, px1)
        y1 = min(y1, py1)
        x2 = max(x2, px2)
        y2 = max(y2, py2)

        self.bbox = clamp_bbox([x1, y1, x2, y2], frame_w, frame_h)

    def update_klt(self, prev_gray, gray, frame_w, frame_h):
        # Update two anchors with KLT
        old_points = self.points()
        new_points = klt_track_two_points(prev_gray, gray, old_points)

        if new_points is None:
            self.alive = False
            return False

        if is_static_mismatch(old_points, new_points):
            self.alive = False
            return False

        self.head = new_points[0]
        self.torso = new_points[1]

        self.update_bbox_from_points(frame_w, frame_h)

        return self.alive


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    tracks = []
    prev_gray = None

    total_frames = 0
    yolo_frames = 0
    klt_frames = 0
    klt_times = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        need_yolo = False

        if prev_gray is None or len(tracks) == 0:
            need_yolo = True
        else:
            # KLT timing only
            t0 = time.perf_counter()

            all_ok = True
            for tr in tracks:
                if not tr.update_klt(prev_gray, gray, frame_w, frame_h):
                    all_ok = False

            t1 = time.perf_counter()

            if all_ok:
                klt_frames += 1
                klt_times.append(t1 - t0)
            else:
                need_yolo = True

        if need_yolo:
            # Re-run YOLO Pose if a point is lost or static mismatch appears
            yolo_frames += 1
            tracks = run_pose(model, frame, frame_w, frame_h)

        vis = frame.copy()

        saved_percent = 100.0 * klt_frames / max(total_frames, 1)
        avg_klt_ms = 1000.0 * sum(klt_times) / len(klt_times) if klt_times else 0.0

        for tr in tracks:
            if not tr.alive:
                continue

            x1, y1, x2, y2 = tr.bbox.astype(int)

            if DRAW_BOXES:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    vis,
                    f"ID {tr.id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if DRAW_POINTS:
                draw_crosshair(vis, tr.head)
                draw_crosshair(vis, tr.torso)

                cv2.line(
                    vis,
                    tuple(tr.head.astype(int)),
                    tuple(tr.torso.astype(int)),
                    (0, 0, 255),
                    2,
                )

        draw_top_right_stats(vis, saved_percent, avg_klt_ms)

        mode = "YOLO Pose" if need_yolo else "KLT"
        cv2.putText(
            vis,
            f"Frame {total_frames} | {mode}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(vis)

        if SHOW_WINDOW:
            cv2.imshow("YOLO Pose + KLT two-point tracking", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        prev_gray = gray

    cap.release()
    writer.release()

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    saved_percent = 100.0 * klt_frames / max(total_frames, 1)
    avg_klt_ms = 1000.0 * sum(klt_times) / len(klt_times) if klt_times else 0.0

    print(f"Total frames: {total_frames}")
    print(f"YOLO Pose frames: {yolo_frames}")
    print(f"KLT frames: {klt_frames}")
    print(f"YOLO saved: {saved_percent:.2f}%")
    print(f"Average KLT time per frame: {avg_klt_ms:.2f} ms")
    print(f"Done. Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()