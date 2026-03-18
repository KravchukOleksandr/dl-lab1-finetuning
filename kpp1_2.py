# ============================================================
# KPP TRUCK FRAME EXTRACTOR WITH TRACK IDS
# ------------------------------------------------------------
# Что делает:
# 1) Берёт видео
# 2) Раз в N секунд запускает YOLO11 в режиме track()
# 3) Берёт только truck внутри ROI
# 4) Ведёт события по track_id
# 5) Для каждого track/event выбирает 1 лучший кадр
# 6) Сохраняет кадр + debug + CSV
#
# Логика для ОДНОГО КПП, где фура стоит в фиксированной зоне.
# ============================================================

from pathlib import Path
import csv
import math

import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# КОНСТАНТЫ
# =========================

VIDEO_PATH = r"kp1_big_video.mp4"
YOLO_MODEL_PATH = "yolo11m.pt"

OUTPUT_DIR = Path("kp1_frames_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Семплирование: раз в сколько секунд обрабатывать кадр
SAMPLE_EVERY_SEC = 3.0

# Детекция
YOLO_CONF = 0.25

# ROI-полигон: [x, y]
ROI_POLYGON = np.array([
    [430, 120],
    [910, 120],
    [980, 780],
    [360, 780],
], dtype=np.int32)

# Если None, берётся центроид ROI
PREFERRED_POINT = None

# Стандартные классы YOLO COCO
TARGET_VEHICLE_CLASS_NAMES = {"truck"}
BLOCKER_CLASS_NAMES = {"person"}

# Минимальная площадь bbox фуры в пикселях
MIN_TRUCK_AREA_PX = 50000

# Доля bbox, которая должна лежать в ROI
MIN_ROI_OVERLAP_RATIO = 0.35

# Через сколько sampled-шагов без появления track закрываем событие
MAX_MISSES_PER_TRACK = 2

# Минимальные требования к событию
MIN_EVENT_DURATION_SEC = 45.0
MIN_EVENT_SAMPLES = 6

# Передняя зона фуры внутри bbox
FRONT_X1_REL = 0.20
FRONT_X2_REL = 0.80
FRONT_Y1_REL = 0.45
FRONT_Y2_REL = 0.95

# Штраф за человека перед фурой
MAX_PERSON_FRONT_IOU_FOR_CLEAN = 0.02

# Резкость
MIN_SHARPNESS = 35.0

# Если хочешь top-1 кадр сохранять
SAVE_DEBUG_VIS = True

# tracking config
# Обычно ultralytics подхватывает bytetrack.yaml из коробки
TRACKER_CONFIG = "bytetrack.yaml"


# =========================
# УТИЛИТЫ
# =========================

def bbox_area_xyxy(box):
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = bbox_area_xyxy(box_a)
    area_b = bbox_area_xyxy(box_b)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def bbox_center(box):
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def clip_box(box, w, h):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def crop_box(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def point_in_polygon(point, polygon):
    x, y = float(point[0]), float(point[1])
    return cv2.pointPolygonTest(polygon.astype(np.float32), (x, y), False) >= 0


def polygon_centroid(poly):
    m = cv2.moments(poly.astype(np.float32))
    if abs(m["m00"]) < 1e-6:
        return poly.mean(axis=0)
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return np.array([cx, cy], dtype=np.float32)


def build_roi_mask(frame_shape, polygon):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def bbox_roi_overlap_ratio(box, roi_mask):
    h, w = roi_mask.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    box_mask = np.zeros_like(roi_mask)
    cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, thickness=-1)

    inter = np.logical_and(box_mask > 0, roi_mask > 0).sum()
    area = max(1, (x2 - x1) * (y2 - y1))
    return float(inter) / float(area)


def front_region_from_truck_box(box):
    x1, y1, x2, y2 = map(float, box)
    w = x2 - x1
    h = y2 - y1

    fx1 = x1 + FRONT_X1_REL * w
    fx2 = x1 + FRONT_X2_REL * w
    fy1 = y1 + FRONT_Y1_REL * h
    fy2 = y1 + FRONT_Y2_REL * h

    return np.array([fx1, fy1, fx2, fy2], dtype=np.float32)


def laplacian_sharpness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def distance_score(center, preferred_point, frame_diag):
    d = np.linalg.norm(center - preferred_point)
    return 1.0 - min(1.0, d / max(frame_diag, 1.0))


def draw_debug(frame_bgr, roi_poly, truck_box=None, front_box=None, persons=None, text=None):
    vis = frame_bgr.copy()

    cv2.polylines(vis, [roi_poly.astype(np.int32)], True, (0, 255, 255), 2)

    if truck_box is not None:
        x1, y1, x2, y2 = map(int, truck_box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if front_box is not None:
        x1, y1, x2, y2 = map(int, front_box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 128, 255), 2)

    if persons:
        for pb in persons:
            x1, y1, x2, y2 = map(int, pb)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if text:
        cv2.putText(
            vis,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def person_front_occlusion(truck_box, person_boxes):
    front_box = front_region_from_truck_box(truck_box)
    max_iou = 0.0
    for pb in person_boxes:
        iou = bbox_iou(front_box, pb)
        if iou > max_iou:
            max_iou = iou
    return max_iou, front_box


def extract_detections_with_ids(pred, model_names):
    detections = []

    if pred.boxes is None or len(pred.boxes) == 0:
        return detections

    boxes = pred.boxes.xyxy.cpu().numpy()
    confs = pred.boxes.conf.cpu().numpy()
    clss = pred.boxes.cls.cpu().numpy().astype(int)

    ids = None
    if hasattr(pred.boxes, "id") and pred.boxes.id is not None:
        ids = pred.boxes.id.cpu().numpy().astype(int)

    for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, clss)):
        track_id = int(ids[i]) if ids is not None else None
        detections.append({
            "box": box.astype(np.float32),
            "conf": float(conf),
            "cls_name": model_names[int(cls_id)],
            "track_id": track_id,
        })

    return detections


def choose_truck_candidates(detections, roi_mask, preferred_point, frame_shape):
    h, w = frame_shape[:2]
    frame_diag = math.hypot(w, h)

    candidates = []
    for det in detections:
        if det["cls_name"] not in TARGET_VEHICLE_CLASS_NAMES:
            continue
        if det["track_id"] is None:
            continue

        box = det["box"]
        area = bbox_area_xyxy(box)
        if area < MIN_TRUCK_AREA_PX:
            continue

        center = bbox_center(box)
        if not point_in_polygon(center, ROI_POLYGON):
            continue

        roi_overlap = bbox_roi_overlap_ratio(box, roi_mask)
        if roi_overlap < MIN_ROI_OVERLAP_RATIO:
            continue

        score = (
            2.5 * roi_overlap +
            1.2 * distance_score(center, preferred_point, frame_diag) +
            0.4 * det["conf"] +
            0.6 * min(1.0, area / 250000.0)
        )

        candidates.append({
            "track_id": det["track_id"],
            "box": box,
            "conf": det["conf"],
            "cls_name": det["cls_name"],
            "roi_overlap": roi_overlap,
            "raw_score": score,
        })

    return candidates


def compute_motion_stats(samples):
    if len(samples) < 2:
        return {
            "mean_center_shift": 0.0,
            "mean_area_change_ratio": 0.0,
        }

    shifts = []
    area_changes = []

    prev_center = bbox_center(samples[0]["truck_box"])
    prev_area = bbox_area_xyxy(samples[0]["truck_box"])

    for s in samples[1:]:
        c = bbox_center(s["truck_box"])
        a = bbox_area_xyxy(s["truck_box"])

        shifts.append(float(np.linalg.norm(c - prev_center)))
        area_changes.append(abs(a - prev_area) / max(prev_area, 1.0))

        prev_center = c
        prev_area = a

    return {
        "mean_center_shift": float(np.mean(shifts)) if shifts else 0.0,
        "mean_area_change_ratio": float(np.mean(area_changes)) if area_changes else 0.0,
    }


def finalize_event(event, cap, out_dir, roi_poly, save_debug=True):
    samples = event["samples"]
    if len(samples) < MIN_EVENT_SAMPLES:
        return None

    duration = samples[-1]["time_sec"] - samples[0]["time_sec"]
    if duration < MIN_EVENT_DURATION_SEC:
        return None

    center_t = 0.5 * (samples[0]["time_sec"] + samples[-1]["time_sec"])

    clean_samples = [
        s for s in samples
        if s["person_front_iou"] <= MAX_PERSON_FRONT_IOU_FOR_CLEAN and s["sharpness"] >= MIN_SHARPNESS
    ]

    # Важно: если "чистых" кадров нет, событие НЕ отбрасываем
    # Просто берём лучший доступный кадр
    candidate_pool = clean_samples if clean_samples else samples

    def rank_fn(s):
        time_penalty = abs(s["time_sec"] - center_t)
        return (time_penalty, -s["quality_score"])

    best = sorted(candidate_pool, key=rank_fn)[0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, best["frame_idx"])
    ok, frame_bgr = cap.read()
    if not ok:
        return None

    event_id = event["event_id"]
    track_id = event["track_id"]
    stem = f"event_{event_id:04d}_track_{track_id}"

    img_path = out_dir / f"{stem}.jpg"
    cv2.imwrite(str(img_path), frame_bgr)

    if save_debug:
        persons = [p["box"] for p in best["persons"]]
        debug = draw_debug(
            frame_bgr,
            roi_poly,
            truck_box=best["truck_box"],
            front_box=best["front_box"],
            persons=persons,
            text=(
                f"event={event_id} track={track_id} "
                f"t={best['time_sec']:.1f}s "
                f"occ={best['person_front_iou']:.3f} "
                f"sharp={best['sharpness']:.1f}"
            )
        )
        dbg_path = out_dir / f"{stem}_debug.jpg"
        cv2.imwrite(str(dbg_path), debug)

    motion = compute_motion_stats(samples)

    return {
        "event_id": event_id,
        "track_id": track_id,
        "image_path": str(img_path),
        "start_sec": samples[0]["time_sec"],
        "end_sec": samples[-1]["time_sec"],
        "duration_sec": duration,
        "picked_time_sec": best["time_sec"],
        "picked_frame_idx": best["frame_idx"],
        "sharpness": best["sharpness"],
        "person_front_iou": best["person_front_iou"],
        "roi_overlap": best["roi_overlap"],
        "truck_conf": best["truck_conf"],
        "sample_count": len(samples),
        "clean_frame_count": len(clean_samples),
        "mean_center_shift": motion["mean_center_shift"],
        "mean_area_change_ratio": motion["mean_area_change_ratio"],
    }


# =========================
# ОСНОВНОЙ КОД
# =========================

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0.0

    print(f"Video: {VIDEO_PATH}")
    print(f"FPS: {fps:.3f}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration_sec/60:.2f} min")

    sample_step_frames = max(1, int(round(SAMPLE_EVERY_SEC * fps)))
    print(f"Sampling every {SAMPLE_EVERY_SEC:.1f} sec => every {sample_step_frames} frames")

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    roi_mask = build_roi_mask((frame_h, frame_w, 3), ROI_POLYGON)

    preferred_point = polygon_centroid(ROI_POLYGON) if PREFERRED_POINT is None else np.array(PREFERRED_POINT, dtype=np.float32)
    print(f"Preferred point: {preferred_point.tolist()}")

    model = YOLO(YOLO_MODEL_PATH)
    model_names = model.names
    wanted_names = TARGET_VEHICLE_CLASS_NAMES | BLOCKER_CLASS_NAMES
    class_ids = [i for i, name in model_names.items() if name in wanted_names]
    print("Using class ids:", {i: model_names[i] for i in class_ids})

    active_events = {}   # track_id -> event
    results_meta = []
    next_event_id = 1

    frame_idx = 0
    processed = 0

    while frame_idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break

        time_sec = frame_idx / fps
        processed += 1

        pred = model.track(
            source=frame_bgr,
            conf=YOLO_CONF,
            classes=class_ids,
            verbose=False,
            persist=True,
            tracker=TRACKER_CONFIG
        )[0]

        detections = extract_detections_with_ids(pred, model_names)

        person_dets = [d for d in detections if d["cls_name"] in BLOCKER_CLASS_NAMES]
        person_boxes = [d["box"] for d in person_dets]

        truck_candidates = choose_truck_candidates(
            detections=detections,
            roi_mask=roi_mask,
            preferred_point=preferred_point,
            frame_shape=frame_bgr.shape
        )

        current_track_ids = set()

        # Обновляем/создаём события по track_id
        for truck in truck_candidates:
            track_id = truck["track_id"]
            current_track_ids.add(track_id)

            if track_id not in active_events:
                active_events[track_id] = {
                    "event_id": next_event_id,
                    "track_id": track_id,
                    "samples": [],
                    "misses": 0,
                }
                next_event_id += 1

            person_front_iou, front_box = person_front_occlusion(truck["box"], person_boxes)
            truck_crop = crop_box(frame_bgr, truck["box"])
            sharpness = laplacian_sharpness(truck_crop) if truck_crop is not None else 0.0

            quality_score = (
                2.0 * truck["roi_overlap"] +
                1.0 * truck["conf"] +
                0.8 * min(1.0, sharpness / 120.0) -
                3.0 * person_front_iou
            )

            active_events[track_id]["samples"].append({
                "frame_idx": frame_idx,
                "time_sec": time_sec,
                "truck_box": truck["box"].copy(),
                "truck_conf": truck["conf"],
                "roi_overlap": truck["roi_overlap"],
                "person_front_iou": person_front_iou,
                "front_box": front_box.copy(),
                "persons": person_dets,
                "sharpness": sharpness,
                "quality_score": quality_score,
            })
            active_events[track_id]["misses"] = 0

        # Для активных, которых в этом sampled-кадре не было — увеличиваем misses
        to_finalize = []
        for track_id, event in active_events.items():
            if track_id not in current_track_ids:
                event["misses"] += 1
                if event["misses"] > MAX_MISSES_PER_TRACK:
                    to_finalize.append(track_id)

        # Закрываем завершённые события
        for track_id in to_finalize:
            event = active_events.pop(track_id)
            meta = finalize_event(
                event=event,
                cap=cap,
                out_dir=OUTPUT_DIR,
                roi_poly=ROI_POLYGON,
                save_debug=SAVE_DEBUG_VIS
            )
            if meta is not None:
                results_meta.append(meta)

        if processed % 50 == 0:
            print(
                f"Processed sampled frames: {processed}, "
                f"video time: {time_sec/60:.1f} min, "
                f"active tracks: {len(active_events)}"
            )

        frame_idx += sample_step_frames

    # Закрываем хвост
    for track_id, event in list(active_events.items()):
        meta = finalize_event(
            event=event,
            cap=cap,
            out_dir=OUTPUT_DIR,
            roi_poly=ROI_POLYGON,
            save_debug=SAVE_DEBUG_VIS
        )
        if meta is not None:
            results_meta.append(meta)

    cap.release()

    # CSV
    csv_path = OUTPUT_DIR / "events.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "track_id",
                "image_path",
                "start_sec",
                "end_sec",
                "duration_sec",
                "picked_time_sec",
                "picked_frame_idx",
                "sharpness",
                "person_front_iou",
                "roi_overlap",
                "truck_conf",
                "sample_count",
                "clean_frame_count",
                "mean_center_shift",
                "mean_area_change_ratio",
            ]
        )
        writer.writeheader()
        for row in results_meta:
            writer.writerow(row)

    print("\nDone.")
    print(f"Saved events: {len(results_meta)}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()