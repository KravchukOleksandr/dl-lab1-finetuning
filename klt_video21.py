import time

import cv2
import numpy as np
from ultralytics import YOLO

from klt_tracker import BootstrapKLTTracker


# =========================
# PATHS
# =========================

VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "output_tracked.mp4"
MODEL_PATH = "yolov8n.pt"


# =========================
# YOLO
# =========================

YOLO_CONF = 0.35
YOLO_IMGSZ = 640


# =========================
# TRACKER
# =========================

TRACKER = BootstrapKLTTracker(
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
)


# =========================
# DRAW
# =========================

GREEN = (0, 255, 0)
PURPLE = (255, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

SHOW_WINDOW = False


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def bbox_center(b):
    x, y, w, h = b
    return np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float32)


def bbox_area(b):
    return max(0.0, b[2]) * max(0.0, b[3])


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)

    inter = iw * ih
    union = aw * ah + bw * bh - inter

    if union <= 0:
        return 0.0

    return inter / union


def detect_persons(model, frame):
    result = model.predict(
        frame,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        classes=[0],
        verbose=False,
    )[0]

    detections = []

    if result.boxes is None:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for box, conf in zip(boxes, confs):
        detections.append((xyxy_to_xywh(box), float(conf)))

    return detections


def choose_detection(detections, previous_bbox):
    if not detections:
        return None

    if previous_bbox is None:
        return max(detections, key=lambda d: bbox_area(d[0]) * d[1])[0]

    best_bbox = None
    best_score = -1.0

    prev_center = bbox_center(previous_bbox)
    max_dist = max(80.0, 0.75 * max(previous_bbox[2], previous_bbox[3]))

    for bbox, conf in detections:
        iou = bbox_iou(previous_bbox, bbox)
        dist = np.linalg.norm(bbox_center(bbox) - prev_center)
        dist_score = max(0.0, 1.0 - dist / max_dist)

        score = 2.0 * iou + 0.7 * dist_score + 0.2 * conf

        if score > best_score:
            best_score = score
            best_bbox = bbox

    return best_bbox


def draw_bbox(frame, bbox, color, label):
    x, y, w, h = bbox

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        frame,
        label,
        (x1, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_points(frame, points, color, r):
    if points is None or len(points) == 0:
        return

    for p in points.reshape(-1, 2):
        cv2.circle(
            frame,
            (int(round(p[0])), int(round(p[1]))),
            r,
            color,
            -1,
        )


def text(frame, line, x, y):
    scale = 0.65
    thickness = 2

    size, baseline = cv2.getTextSize(
        line,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )

    w, h = size

    cv2.rectangle(
        frame,
        (x - 4, y - h - 6),
        (x + w + 4, y + baseline + 4),
        BLACK,
        -1,
    )

    cv2.putText(
        frame,
        line,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        WHITE,
        thickness,
        cv2.LINE_AA,
    )


def draw_metrics(frame, total, klt_frames, yolo_frames, fps, avg_ms, mode):
    saved = 100.0 * klt_frames / max(1, total)

    lines = [
        f"Saved by tracker: {saved:.1f}% ({klt_frames}/{total})",
        f"YOLO frames: {yolo_frames}",
        f"Processing FPS: {fps:.1f}",
        f"Tracker avg: {avg_ms:.3f} ms",
        f"Mode: {mode}",
    ]

    y = 28

    for line in lines:
        text(frame, line, 12, y)
        y += 28


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps_in <= 0:
        fps_in = 30.0

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (width, height),
    )

    total_frames = 0
    yolo_frames = 0
    tracker_frames = 0

    tracker_time = 0.0
    tracker_calls = 0

    tracker_ready = False
    last_bbox = None

    t0_all = time.perf_counter()

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        total_frames += 1

        used_yolo = False
        mode = "NONE"
        result = None

        if not tracker_ready:
            detections = detect_persons(model, frame)
            yolo_frames += 1
            used_yolo = True

            bbox = choose_detection(detections, last_bbox)

            if bbox is not None:
                TRACKER.init(frame, bbox)
                tracker_ready = True
                last_bbox = bbox
                result = TRACKER._result("YOLO_INIT", False)
                mode = "YOLO_INIT"
            else:
                mode = "YOLO_MISS"

        else:
            t0 = time.perf_counter()
            result = TRACKER.update(frame)
            tracker_time += time.perf_counter() - t0
            tracker_calls += 1

            if result["need_yolo"]:
                detections = detect_persons(model, frame)
                yolo_frames += 1
                used_yolo = True

                bbox = choose_detection(detections, result["bbox"])

                if bbox is not None:
                    if TRACKER.state == "BOOTSTRAP":
                        t0 = time.perf_counter()
                        result = TRACKER.feed_yolo(bbox)
                        tracker_time += time.perf_counter() - t0
                        tracker_calls += 1

                        if result["state"] == "LOST":
                            TRACKER.init(frame, bbox)
                            result = TRACKER._result("YOLO_REINIT", False)
                    else:
                        TRACKER.init(frame, bbox)
                        result = TRACKER._result("YOLO_REINIT", False)

                    last_bbox = bbox
                    mode = result["state"]
                else:
                    tracker_ready = False
                    mode = "LOST"

            else:
                tracker_frames += 1
                last_bbox = result["bbox"]
                mode = result["state"]

        if result is not None and result["bbox"] is not None:
            if used_yolo:
                draw_bbox(frame, result["bbox"], GREEN, "YOLO")
            else:
                draw_bbox(frame, result["bbox"], PURPLE, "KLT")

            draw_points(frame, result["points"], BLUE, 2)
            draw_points(frame, result["core_points"], YELLOW, 3)

        if mode == "LOST":
            text(frame, "LOST", 12, height - 20)

        elapsed = time.perf_counter() - t0_all
        processing_fps = total_frames / max(1e-9, elapsed)
        avg_tracker_ms = 1000.0 * tracker_time / max(1, tracker_calls)

        draw_metrics(
            frame,
            total_frames,
            tracker_frames,
            yolo_frames,
            processing_fps,
            avg_tracker_ms,
            mode,
        )

        writer.write(frame)

        if SHOW_WINDOW:
            cv2.imshow("YOLO + KLT", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    writer.release()

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0_all

    print("Done")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames: {total_frames}")
    print(f"YOLO frames: {yolo_frames}")
    print(f"Tracker frames: {tracker_frames}")
    print(f"Saved by tracker: {100.0 * tracker_frames / max(1, total_frames):.2f}%")
    print(f"Processing FPS: {total_frames / max(1e-9, elapsed):.2f}")
    print(f"Tracker avg: {1000.0 * tracker_time / max(1, tracker_calls):.4f} ms")


if __name__ == "__main__":
    main()