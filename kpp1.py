# ============================================================
# KPP FRAME SAMPLER FOR TRUCKS
# ------------------------------------------------------------
# Что делает:
# 1) Берёт видео
# 2) Раз в N секунд запускает YOLO11
# 3) Ищет truck внутри заданного ROI-полигона
# 4) Склеивает последовательные детекции в одно "событие стоянки"
# 5) Для каждого события выбирает 1 лучший кадр ближе к центру стоянки
# 6) Сохраняет кадр + CSV с метаданными
#
# Логика заточена под ОДИН КПП и ОДНУ фиксированную зону остановки.
# ============================================================

from pathlib import Path
import math
import csv

import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# КОНСТАНТЫ
# =========================

VIDEO_PATH = r"kp1_big_video.mp4"
YOLO_MODEL_PATH = "yolo11m.pt"   # стандартная модель Ultralytics

OUTPUT_DIR = Path("kp1_frames_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Раз в сколько секунд анализировать видео
SAMPLE_EVERY_SEC = 3.0

# YOLO confidence
YOLO_CONF = 0.25

# ROI-полигон зоны КПП (пример; ЗАМЕНИ под свою камеру)
# Формат: [(x1, y1), (x2, y2), ...]
ROI_POLYGON = np.array([
    [430, 120],
    [910, 120],
    [980, 780],
    [360, 780],
], dtype=np.int32)

# Если хочешь вручную задать "идеальную точку" центра кадра для фуры
# Если None -> берётся центроид ROI
PREFERRED_POINT = None

# Какие классы считаем целевыми
# У стандартной YOLO COCO обычно есть "truck" и "person"
TARGET_VEHICLE_CLASS_NAMES = {"truck"}
BLOCKER_CLASS_NAMES = {"person"}

# Минимальная площадь bbox фуры в пикселях, чтобы отсечь дальний фон
MIN_TRUCK_AREA_PX = 50000

# Минимальная доля bbox фуры, лежащая в ROI
MIN_ROI_OVERLAP_RATIO = 0.35

# Насколько похожи bbox по IoU, чтобы считать, что это та же стоящая фура
TRACK_IOU_THRESHOLD = 0.30

# Максимальный пропуск (в sampled-шагов), после которого событие завершаем
MAX_MISSES_IN_EVENT = 2

# Минимальная длительность события стоянки
MIN_EVENT_DURATION_SEC = 60.0

# Минимальное число sampled-кадров в событии
MIN_EVENT_SAMPLES = 8

# Передняя зона фуры внутри bbox:
# x: от 0.20 до 0.80 ширины, y: от 0.45 до 0.95 высоты
# Там чаще всего человек перекрывает номер / переднюю часть
FRONT_X1_REL = 0.20
FRONT_X2_REL = 0.80
FRONT_Y1_REL = 0.45
FRONT_Y2_REL = 0.95

# Максимальное допустимое перекрытие person с передней зоной фуры
MAX_PERSON_FRONT_IOU = 0.02

# Порог резкости по variance of Laplacian
MIN_SHARPNESS = 35.0

# Визуализация отладочных картинок
SAVE_DEBUG_VIS = True


# =========================
# УТИЛИТЫ
# =========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def bbox_area_xyxy(box):
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(box):
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


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


def polygon_centroid(poly: np.ndarray):
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
    # чем ближе, тем лучше
    return 1.0 - min(1.0, d / max(1.0, frame_diag))


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
            cv2.LINE_AA
        )
    return vis


def pick_best_truck_candidate(detections, roi_mask, preferred_point, frame_shape):
    """
    detections: list[dict] with keys: box, conf, cls_name
    Возвращает лучший truck-кандидат или None
    """
    h, w = frame_shape[:2]
    frame_diag = math.hypot(w, h)

    best = None
    best_score = -1e9

    for det in detections:
        if det["cls_name"] not in TARGET_VEHICLE_CLASS_NAMES:
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

        if score > best_score:
            best_score = score
            best = {
                "box": box,
                "conf": det["conf"],
                "cls_name": det["cls_name"],
                "roi_overlap": roi_overlap,
                "raw_score": score,
            }

    return best


def person_front_occlusion(truck_box, person_boxes):
    front_box = front_region_from_truck_box(truck_box)
    max_iou = 0.0
    for pb in person_boxes:
        iou = bbox_iou(front_box, pb)
        if iou > max_iou:
            max_iou = iou
    return max_iou, front_box


def finalize_event(event, cap, out_dir, roi_poly, save_debug=True):
    """
    Выбирает 1 лучший кадр из события и сохраняет его.
    Возвращает dict с метаданными или None.
    """
    samples = event["samples"]
    if len(samples) < MIN_EVENT_SAMPLES:
        return None

    duration = samples[-1]["time_sec"] - samples[0]["time_sec"]
    if duration < MIN_EVENT_DURATION_SEC:
        return None

    center_t = 0.5 * (samples[0]["time_sec"] + samples[-1]["time_sec"])

    valid = []
    for s in samples:
        if s["person_front_iou"] <= MAX_PERSON_FRONT_IOU and s["sharpness"] >= MIN_SHARPNESS:
            valid.append(s)

    if not valid:
        # fallback: если совсем нет хороших кадров, берём лучший по score+близости к центру
        valid = samples

    def rank_fn(s):
        time_penalty = abs(s["time_sec"] - center_t)
        # меньше penalty лучше, больше score лучше
        return (time_penalty, -s["quality_score"])

    best = sorted(valid, key=rank_fn)[0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, best["frame_idx"])
    ok, frame_bgr = cap.read()
    if not ok:
        return None

    event_id = event["event_id"]
    stem = f"event_{event_id:04d}"

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
                f"event={event_id} t={best['time_sec']:.1f}s "
                f"occ={best['person_front_iou']:.3f} "
                f"sharp={best['sharpness']:.1f}"
            )
        )
        dbg_path = out_dir / f"{stem}_debug.jpg"
        cv2.imwrite(str(dbg_path), debug)

    return {
        "event_id": event_id,
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

    roi_mask = build_roi_mask((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), ROI_POLYGON)

    preferred_point = polygon_centroid(ROI_POLYGON) if PREFERRED_POINT is None else np.array(PREFERRED_POINT, dtype=np.float32)
    print(f"Preferred point: {preferred_point.tolist()}")

    model = YOLO(YOLO_MODEL_PATH)

    # классы по именам
    model_names = model.names
    wanted_names = TARGET_VEHICLE_CLASS_NAMES | BLOCKER_CLASS_NAMES
    class_ids = [i for i, name in model_names.items() if name in wanted_names]

    print("Using class ids:", {i: model_names[i] for i in class_ids})

    results_meta = []
    current_event = None
    miss_count = 0
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

        pred = model.predict(
            source=frame_bgr,
            conf=YOLO_CONF,
            classes=class_ids,
            verbose=False
        )[0]

        detections = []
        if pred.boxes is not None and len(pred.boxes) > 0:
            boxes = pred.boxes.xyxy.cpu().numpy()
            confs = pred.boxes.conf.cpu().numpy()
            clss = pred.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, clss):
                detections.append({
                    "box": box.astype(np.float32),
                    "conf": float(conf),
                    "cls_name": model_names[int(cls_id)],
                })

        persons = [d for d in detections if d["cls_name"] in BLOCKER_CLASS_NAMES]
        truck_candidate = pick_best_truck_candidate(
            detections=detections,
            roi_mask=roi_mask,
            preferred_point=preferred_point,
            frame_shape=frame_bgr.shape
        )

        matched = False

        if truck_candidate is not None:
            truck_box = truck_candidate["box"]
            prev_box = current_event["last_box"] if current_event is not None else None

            same_track = False
            if prev_box is not None:
                if bbox_iou(prev_box, truck_box) >= TRACK_IOU_THRESHOLD:
                    same_track = True

            if current_event is None or not same_track:
                # если было старое событие — закрываем
                if current_event is not None:
                    meta = finalize_event(
                        current_event, cap, OUTPUT_DIR, ROI_POLYGON, save_debug=SAVE_DEBUG_VIS
                    )
                    if meta is not None:
                        results_meta.append(meta)

                # старт нового события
                current_event = {
                    "event_id": next_event_id,
                    "samples": [],
                    "last_box": truck_box.copy(),
                }
                next_event_id += 1
                miss_count = 0

            # оцениваем закрытие человеком и резкость
            person_boxes = [p["box"] for p in persons]
            person_front_iou, front_box = person_front_occlusion(truck_box, person_boxes)

            truck_crop = crop_box(frame_bgr, truck_box)
            sharpness = laplacian_sharpness(truck_crop) if truck_crop is not None else 0.0

            quality_score = (
                2.0 * truck_candidate["roi_overlap"] +
                1.0 * truck_candidate["conf"] +
                0.8 * min(1.0, sharpness / 120.0) -
                3.0 * person_front_iou
            )

            current_event["samples"].append({
                "frame_idx": frame_idx,
                "time_sec": time_sec,
                "truck_box": truck_box.copy(),
                "truck_conf": truck_candidate["conf"],
                "roi_overlap": truck_candidate["roi_overlap"],
                "person_front_iou": person_front_iou,
                "front_box": front_box.copy(),
                "persons": persons,
                "sharpness": sharpness,
                "quality_score": quality_score,
            })
            current_event["last_box"] = truck_box.copy()

            matched = True

        if not matched and current_event is not None:
            miss_count += 1
            if miss_count > MAX_MISSES_IN_EVENT:
                meta = finalize_event(
                    current_event, cap, OUTPUT_DIR, ROI_POLYGON, save_debug=SAVE_DEBUG_VIS
                )
                if meta is not None:
                    results_meta.append(meta)
                current_event = None
                miss_count = 0
        elif matched:
            miss_count = 0

        if processed % 50 == 0:
            print(f"Processed sampled frames: {processed}, video time: {time_sec/60:.1f} min")

        frame_idx += sample_step_frames

    # закрыть хвостовое событие
    if current_event is not None:
        meta = finalize_event(
            current_event, cap, OUTPUT_DIR, ROI_POLYGON, save_debug=SAVE_DEBUG_VIS
        )
        if meta is not None:
            results_meta.append(meta)

    cap.release()

    # сохранить CSV
    csv_path = OUTPUT_DIR / "events.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
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