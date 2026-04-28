# klt_yolo_people_track.py

import time
import cv2
import numpy as np
from ultralytics import YOLO


# Paths and main settings
VIDEO_PATH = "input.mp4"
MODEL_PATH = "yolov8n.pt"
OUTPUT_PATH = "output_klt.mp4"

# Tracking settings
NUM_POINTS = 80
YOLO_WARMUP_FRAMES = 3

# YOLO settings
PERSON_CLASS_ID = 0
YOLO_CONF = 0.35

# KLT quality thresholds
MIN_VALID_POINTS = 10
REINIT_POINTS_RATIO = 0.45
MAX_FB_ERROR = 2.5

# Visualization
DRAW_POINTS = True
DRAW_BBOXES = True
SHOW_WINDOW = True


# Lucas-Kanade optical flow params
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Feature detector params
FEATURE_PARAMS = dict(
    maxCorners=NUM_POINTS,
    qualityLevel=0.01,
    minDistance=5,
    blockSize=7,
)


def clamp_bbox(bbox, frame_w, frame_h):
    # Keep bbox inside frame
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


def bbox_iou(a, b):
    # Compute IoU between two boxes
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    return inter / (area_a + area_b - inter + 1e-6)


def detect_people_yolo(model, frame):
    # Run YOLO and keep only person boxes
    result = model.predict(frame, conf=YOLO_CONF, verbose=False)[0]
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id != PERSON_CLASS_ID:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].item())

        detections.append({
            "bbox": np.array([x1, y1, x2, y2], dtype=np.float32),
            "conf": conf,
        })

    return detections


def create_bbox_mask(gray, bbox):
    # Create mask only inside bbox
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h).astype(int)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    return mask


def init_points(gray, bbox, num_points):
    # Find good feature points inside bbox
    mask = create_bbox_mask(gray, bbox)

    pts = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        maxCorners=num_points,
        qualityLevel=FEATURE_PARAMS["qualityLevel"],
        minDistance=FEATURE_PARAMS["minDistance"],
        blockSize=FEATURE_PARAMS["blockSize"],
    )

    if pts is None:
        return np.empty((0, 1, 2), dtype=np.float32)

    return pts.astype(np.float32)


def points_inside_bbox(points, bbox):
    # Keep only points inside bbox
    if points is None or len(points) == 0:
        return np.empty((0, 1, 2), dtype=np.float32)

    x1, y1, x2, y2 = bbox
    pts = points.reshape(-1, 2)

    keep = (
        (pts[:, 0] >= x1) &
        (pts[:, 0] <= x2) &
        (pts[:, 1] >= y1) &
        (pts[:, 1] <= y2)
    )

    return pts[keep].reshape(-1, 1, 2).astype(np.float32)


def klt_track_points(prev_gray, gray, old_points):
    # Track points with LK optical flow
    if old_points is None or len(old_points) == 0:
        return None, None

    new_points, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        old_points,
        None,
        **LK_PARAMS,
    )

    if new_points is None or st_fwd is None:
        return None, None

    # Track back to filter bad points
    back_points, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        prev_gray,
        new_points,
        None,
        **LK_PARAMS,
    )

    if back_points is None or st_bwd is None:
        return None, None

    old_xy = old_points.reshape(-1, 2)
    new_xy = new_points.reshape(-1, 2)
    back_xy = back_points.reshape(-1, 2)

    fb_error = np.linalg.norm(old_xy - back_xy, axis=1)

    valid = (
        (st_fwd.reshape(-1) == 1) &
        (st_bwd.reshape(-1) == 1) &
        (fb_error < MAX_FB_ERROR)
    )

    good_old = old_xy[valid].reshape(-1, 1, 2).astype(np.float32)
    good_new = new_xy[valid].reshape(-1, 1, 2).astype(np.float32)

    return good_old, good_new


def estimate_bbox(old_bbox, old_pts, new_pts, frame_w, frame_h):
    # Move bbox using affine transform from tracked points
    old_xy = old_pts.reshape(-1, 2)
    new_xy = new_pts.reshape(-1, 2)

    if len(old_xy) >= 4:
        matrix, _ = cv2.estimateAffinePartial2D(
            old_xy,
            new_xy,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=200,
            confidence=0.99,
        )

        if matrix is not None:
            x1, y1, x2, y2 = old_bbox

            corners = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)

            moved = cv2.transform(corners, matrix).reshape(-1, 2)

            nx1 = np.min(moved[:, 0])
            ny1 = np.min(moved[:, 1])
            nx2 = np.max(moved[:, 0])
            ny2 = np.max(moved[:, 1])

            return clamp_bbox([nx1, ny1, nx2, ny2], frame_w, frame_h)

    # Fallback: use median shift
    delta = new_xy - old_xy
    dx = np.median(delta[:, 0])
    dy = np.median(delta[:, 1])

    x1, y1, x2, y2 = old_bbox
    return clamp_bbox([x1 + dx, y1 + dy, x2 + dx, y2 + dy], frame_w, frame_h)


class PersonTrack:
    def __init__(self, track_id, bbox, gray):
        # Init one tracked person
        self.id = track_id
        self.bbox = bbox.astype(np.float32)
        self.points = init_points(gray, bbox, NUM_POINTS)
        self.alive = True

    def update_with_detection(self, bbox, gray):
        # Reset track from YOLO box
        self.bbox = bbox.astype(np.float32)
        self.points = init_points(gray, self.bbox, NUM_POINTS)
        self.alive = True

    def add_points_if_needed(self, gray):
        # Refill points inside current bbox
        if len(self.points) >= int(NUM_POINTS * REINIT_POINTS_RATIO):
            return

        needed = NUM_POINTS - len(self.points)
        if needed <= 0:
            return

        new_pts = init_points(gray, self.bbox, needed)
        if len(new_pts) == 0:
            return

        if len(self.points) == 0:
            self.points = new_pts
        else:
            self.points = np.concatenate([self.points, new_pts], axis=0)

    def update_with_klt(self, prev_gray, gray, frame_w, frame_h):
        # Update bbox from KLT points
        if not self.alive:
            return

        if len(self.points) < MIN_VALID_POINTS:
            self.add_points_if_needed(gray)

            if len(self.points) < MIN_VALID_POINTS:
                self.alive = False
                return

        good_old, good_new = klt_track_points(prev_gray, gray, self.points)

        if good_old is None or good_new is None or len(good_new) < MIN_VALID_POINTS:
            self.points = np.empty((0, 1, 2), dtype=np.float32)
            self.alive = False
            return

        self.bbox = estimate_bbox(
            self.bbox,
            good_old,
            good_new,
            frame_w,
            frame_h,
        )

        self.points = points_inside_bbox(good_new, self.bbox)
        self.add_points_if_needed(gray)


def main():
    # Load YOLO
    model = YOLO(MODEL_PATH)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    # Read video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    tracks = []
    next_track_id = 1
    prev_gray = None
    frame_idx = 0

    processing_times = []

    while True:
        # Read frame outside timing
        ok, frame = cap.read()
        if not ok:
            break

        # Start processing timer
        t0 = time.perf_counter()

        # Convert to grayscale for KLT
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_idx < YOLO_WARMUP_FRAMES:
            # Run YOLO only at the beginning
            detections = detect_people_yolo(model, frame)

            for det in detections:
                det_bbox = clamp_bbox(det["bbox"], frame_w, frame_h)

                matched = False
                for tr in tracks:
                    # Avoid duplicate tracks during warmup
                    if bbox_iou(tr.bbox, det_bbox) > 0.5:
                        tr.update_with_detection(det_bbox, gray)
                        matched = True
                        break

                if not matched:
                    tracks.append(PersonTrack(next_track_id, det_bbox, gray))
                    next_track_id += 1

        elif prev_gray is not None:
            # Track all persons with KLT
            for tr in tracks:
                tr.update_with_klt(prev_gray, gray, frame_w, frame_h)

        # Draw current frame
        vis = frame.copy()

        for tr in tracks:
            if not tr.alive:
                continue

            x1, y1, x2, y2 = tr.bbox.astype(int)

            if DRAW_BBOXES:
                # Draw bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"ID {tr.id} | pts {len(tr.points)}"
                cv2.putText(
                    vis,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if DRAW_POINTS:
                # Draw KLT points
                for p in tr.points.reshape(-1, 2):
                    px, py = int(p[0]), int(p[1])
                    cv2.circle(vis, (px, py), 2, (0, 0, 255), -1)

        # Draw mode info
        mode = "YOLO init" if frame_idx < YOLO_WARMUP_FRAMES else "KLT only"
        cv2.putText(
            vis,
            f"Frame {frame_idx} | {mode}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Stop processing timer before video writing
        t1 = time.perf_counter()
        processing_times.append(t1 - t0)

        # Save frame to output video
        writer.write(vis)

        # Optional preview
        if SHOW_WINDOW:
            cv2.imshow("YOLO init + KLT tracking", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        # Save state for next frame
        prev_gray = gray
        frame_idx += 1

    # Release resources
    cap.release()
    writer.release()

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    # Print speed stats
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0

        print(f"Average processing time per frame: {avg_time * 1000:.2f} ms")
        print(f"Average algorithm FPS: {avg_fps:.2f}")

    print(f"Done. Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()