import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from klt_tracker import CenterGridKLTTracker, BBox


# =========================
# НАСТРОЙКИ
# =========================

VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "output_tracked.mp4"
MODEL_PATH = "yolov8n.pt"

YOLO_CONF = 0.35
YOLO_IMGSZ = 640

SHOW_WINDOW = False

# KLT tracker settings
GRID_W = 6
GRID_H = 5

ROI_X1 = 0.10
ROI_X2 = 0.90
ROI_Y1 = 0.50
ROI_Y2 = 0.85

MIN_CORE_POINTS = 8
MIN_CORE_RATIO = 0.60

FB_ERROR_THR = 2.0

RESIDUAL_THR_MIN = 1.5
RESIDUAL_THR_MAX = 6.0

MAX_BAD_FRAMES = 3

# Перехват старого трека после YOLO redetect
MIN_REID_IOU = 0.05


# =========================
# ЦВЕТА
# =========================

GREEN = (0, 255, 0)       # YOLO bbox
PURPLE = (255, 0, 255)    # KLT bbox
BLUE = (255, 0, 0)        # alive points
YELLOW = (0, 255, 255)    # core points
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)


Detection = Tuple[BBox, float]


def xyxy_to_xywh(box) -> BBox:
    x1, y1, x2, y2 = box
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def bbox_area(bbox: BBox) -> float:
    _, _, w, h = bbox
    return max(0.0, w) * max(0.0, h)


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w * 0.5, y + h * 0.5


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(np.hypot(ax - bx, ay - by))


def bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    union = aw * ah + bw * bh - inter

    if union <= 0:
        return 0.0

    return float(inter / union)


def run_yolo_persons(
    model: YOLO,
    frame: np.ndarray,
) -> List[Detection]:
    """
    YOLO ищет только людей.
    Для COCO class_id=0 — person.
    """

    results = model.predict(
        frame,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        classes=[0],
        verbose=False,
    )

    detections: List[Detection] = []

    if not results:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for box, conf in zip(boxes_xyxy, confs):
        bbox = xyxy_to_xywh(box)
        detections.append((bbox, float(conf)))

    return detections


def choose_detection_for_track(
    detections: List[Detection],
    previous_bbox: Optional[BBox],
) -> Optional[BBox]:
    """
    Если previous_bbox нет — выбираем самого крупного человека.

    Если previous_bbox есть — пытаемся перехватить старый трек:
        - сначала по IoU,
        - потом по близости центра.
    """

    if not detections:
        return None

    if previous_bbox is None:
        best_bbox, _ = max(
            detections,
            key=lambda det: bbox_area(det[0]) * max(det[1], 0.01),
        )
        return best_bbox

    _, _, prev_w, prev_h = previous_bbox
    max_reasonable_dist = max(80.0, 0.75 * max(prev_w, prev_h))

    best_bbox = None
    best_score = -1.0
    best_iou = 0.0
    best_dist = float("inf")

    for bbox, conf in detections:
        iou = bbox_iou(previous_bbox, bbox)
        dist = center_distance(previous_bbox, bbox)

        dist_score = max(0.0, 1.0 - dist / max_reasonable_dist)

        score = 2.0 * iou + 0.7 * dist_score + 0.2 * conf

        if score > best_score:
            best_score = score
            best_bbox = bbox
            best_iou = iou
            best_dist = dist

    if best_bbox is None:
        return None

    if best_iou >= MIN_REID_IOU:
        return best_bbox

    if best_dist <= max_reasonable_dist:
        return best_bbox

    return None


def draw_bbox(
    frame: np.ndarray,
    bbox: BBox,
    color,
    label: str,
    thickness: int = 2,
):
    x, y, w, h = bbox

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_points(
    frame: np.ndarray,
    points: np.ndarray,
    color,
    radius: int,
):
    if points is None or len(points) == 0:
        return

    pts = points.reshape(-1, 2)

    for p in pts:
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(frame, (x, y), radius, color, -1)


def draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color=WHITE,
    bg=BLACK,
    scale: float = 0.65,
    thickness: int = 2,
):
    x, y = org

    (tw, th), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )

    cv2.rectangle(
        frame,
        (x - 4, y - th - 6),
        (x + tw + 4, y + baseline + 4),
        bg,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_metrics(
    frame: np.ndarray,
    total_frames: int,
    klt_saved_frames: int,
    yolo_frames: int,
    processing_fps: float,
    avg_klt_ms: float,
    mode: str,
):
    saved_pct = 100.0 * klt_saved_frames / max(1, total_frames)

    lines = [
        f"Saved by KLT: {saved_pct:.1f}% ({klt_saved_frames}/{total_frames})",
        f"YOLO frames: {yolo_frames}",
        f"Processing FPS: {processing_fps:.1f}",
        f"KLT avg update: {avg_klt_ms:.3f} ms",
        f"Mode: {mode}",
    ]

    y = 28

    for line in lines:
        draw_text_with_bg(frame, line, (12, y))
        y += 28


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if src_fps <= 1e-6:
        src_fps = 30.0

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps,
        (src_w, src_h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {OUTPUT_PATH}")

    tracker = CenterGridKLTTracker(
        grid_w=GRID_W,
        grid_h=GRID_H,

        roi_x1=ROI_X1,
        roi_x2=ROI_X2,
        roi_y1=ROI_Y1,
        roi_y2=ROI_Y2,

        min_core_points=MIN_CORE_POINTS,
        min_core_ratio=MIN_CORE_RATIO,

        fb_error_thr=FB_ERROR_THR,

        residual_thr_min=RESIDUAL_THR_MIN,
        residual_thr_max=RESIDUAL_THR_MAX,

        max_bad_frames=MAX_BAD_FRAMES,
    )

    total_frames = 0
    yolo_frames = 0
    klt_saved_frames = 0

    klt_time_sum = 0.0
    klt_time_count = 0

    need_yolo = True
    last_bbox: Optional[BBox] = None

    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        total_frames += 1

        mode = "NONE"
        bbox_to_draw: Optional[BBox] = None

        points_to_draw = np.empty((0, 1, 2), dtype=np.float32)
        core_points_to_draw = np.empty((0, 1, 2), dtype=np.float32)

        # =========================
        # YOLO MODE
        # =========================

        if need_yolo:
            yolo_frames += 1

            detections = run_yolo_persons(model, frame)

            detected_bbox = choose_detection_for_track(
                detections=detections,
                previous_bbox=last_bbox,
            )

            if detected_bbox is not None:
                tracker.init(frame, detected_bbox)

                last_bbox = detected_bbox
                bbox_to_draw = detected_bbox

                mode = "YOLO"
                need_yolo = False

                alive_ids = np.where(tracker.alive)[0]
                points_to_draw = (
                    tracker.points[alive_ids]
                    .reshape(-1, 1, 2)
                    .astype(np.float32)
                )
            else:
                mode = "YOLO_MISS"
                need_yolo = True

        # =========================
        # KLT MODE
        # =========================

        else:
            t0 = time.perf_counter()
            result = tracker.update(frame)
            t1 = time.perf_counter()

            klt_time_sum += t1 - t0
            klt_time_count += 1

            if result.need_redetect:
                # KLT не уверен — пробуем YOLO на этом же кадре.
                yolo_frames += 1

                detections = run_yolo_persons(model, frame)

                detected_bbox = choose_detection_for_track(
                    detections=detections,
                    previous_bbox=result.bbox if result.bbox is not None else last_bbox,
                )

                if detected_bbox is not None:
                    tracker.init(frame, detected_bbox)

                    last_bbox = detected_bbox
                    bbox_to_draw = detected_bbox

                    mode = "YOLO"
                    need_yolo = False

                    alive_ids = np.where(tracker.alive)[0]
                    points_to_draw = (
                        tracker.points[alive_ids]
                        .reshape(-1, 1, 2)
                        .astype(np.float32)
                    )
                else:
                    last_bbox = result.bbox
                    bbox_to_draw = result.bbox

                    points_to_draw = result.points
                    core_points_to_draw = result.core_points_xy

                    mode = "LOST"
                    need_yolo = True

            else:
                klt_saved_frames += 1

                last_bbox = result.bbox
                bbox_to_draw = result.bbox

                points_to_draw = result.points
                core_points_to_draw = result.core_points_xy

                mode = f"KLT/{result.state}"

        # =========================
        # DRAW
        # =========================

        if bbox_to_draw is not None:
            if mode == "YOLO":
                draw_bbox(frame, bbox_to_draw, GREEN, "YOLO")
            elif mode == "LOST":
                draw_bbox(frame, bbox_to_draw, RED, "LOST")
            elif mode.startswith("KLT"):
                draw_bbox(frame, bbox_to_draw, PURPLE, "KLT")

        draw_points(frame, points_to_draw, BLUE, radius=2)
        draw_points(frame, core_points_to_draw, YELLOW, radius=3)

        elapsed = time.perf_counter() - t_start
        processing_fps = total_frames / max(elapsed, 1e-9)

        if klt_time_count > 0:
            avg_klt_ms = 1000.0 * klt_time_sum / klt_time_count
        else:
            avg_klt_ms = 0.0

        draw_metrics(
            frame=frame,
            total_frames=total_frames,
            klt_saved_frames=klt_saved_frames,
            yolo_frames=yolo_frames,
            processing_fps=processing_fps,
            avg_klt_ms=avg_klt_ms,
            mode=mode,
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

    elapsed = time.perf_counter() - t_start

    saved_pct = 100.0 * klt_saved_frames / max(1, total_frames)

    if klt_time_count > 0:
        avg_klt_ms = 1000.0 * klt_time_sum / klt_time_count
    else:
        avg_klt_ms = 0.0

    print("Done")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames: {total_frames}")
    print(f"YOLO frames: {yolo_frames}")
    print(f"KLT frames: {klt_saved_frames}")
    print(f"Saved by KLT: {saved_pct:.2f}%")
    print(f"Processing FPS: {total_frames / max(elapsed, 1e-9):.2f}")
    print(f"Average KLT update time: {avg_klt_ms:.4f} ms")


if __name__ == "__main__":
    main()