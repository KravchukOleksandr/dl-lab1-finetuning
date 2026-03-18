# ============================================================
# KPP TRUCK EVENT EXTRACTOR (RECALL-ORIENTED)
# ------------------------------------------------------------
# Что делает:
# 1) Берёт видео
# 2) Раз в N секунд запускает YOLO
# 3) Ищет лучшую фуру в ROI
# 4) Делит поток на события:
#       - фура стояла
#       - уехала / сменилась на другую
# 5) Для каждого события сохраняет лучший кадр
#
# Важно:
# - логика заточена под ОДИН КПП
# - приоритет = не пропустить фуру
# - дубли допустимы
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

VIDEO_PATH = r"data/exit.avi"
YOLO_MODEL_PATH = "models/yolo11m.pt"

OUTPUT_DIR = Path("kp1_frames_out_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Раз в сколько секунд смотреть кадр
SAMPLE_EVERY_SEC = 1.0

# YOLO
YOLO_CONF = 0.25

# ROI-полигон: [x, y]
ROI_POLYGON = np.array([
    [350, 850],
    [616,   0],
    [1540,  0],
    [1407, 976],
], dtype=np.int32)

# Если None — берётся центроид ROI
PREFERRED_POINT = None

# Какие классы используем
TARGET_VEHICLE_CLASS_NAMES = {"truck"}
BLOCKER_CLASS_NAMES = {"person"}

# Минимальная площадь bbox фуры
MIN_TRUCK_AREA_PX = 300 * 300

# Минимальная доля bbox внутри ROI
MIN_ROI_OVERLAP_RATIO = 0.35

# Передняя зона фуры внутри bbox
FRONT_X1_REL = 0.20
FRONT_X2_REL = 0.80
FRONT_Y1_REL = 0.45
FRONT_Y2_REL = 0.95

# Резкость
MIN_SHARPNESS = 35.0

# Штраф за человека перед фурой
MAX_PERSON_FRONT_IOU_FOR_CLEAN = 0.02

# Минимальные требования к событию
MIN_EVENT_DURATION_SEC = 8.0
MIN_EVENT_SAMPLES = 3

# Сколько sampled-кадров подряд без фуры нужно, чтобы закрыть событие
EMPTY_CONFIRM_STEPS = 2

# Сколько sampled-кадров подряд должна держаться "новая геометрия",
# чтобы считать, что это уже другая фура
NEW_TRUCK_CONFIRM_STEPS = 2

# Размер истории для "стабильного референса" текущей фуры
REFERENCE_WINDOW = 4

# Пороги разделения событий внутри непрерывной занятости ROI
# Если новая детекция сильно отличается от текущей фуры,
# и это подтвердилось несколько sampled-кадров подряд -> новая фура
SPLIT_IOU_MIN = 0.35
SPLIT_CENTER_DIST_MAX = 110.0
SPLIT_AREA_RATIO_MAX = 1.55   # max(new/old, old/new)

# Сколько кадров сохранять на событие
SAVE_TOP_K_PER_EVENT = 1

# Сохранять debug-версии
SAVE_DEBUG_VIS = True


# =========================
# УТИЛИТЫ
# =========================

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
            cv2.LINE_AA
        )
    return vis


def extract_detections(pred, model_names):
    detections = []

    if pred.boxes is None or len(pred.boxes) == 0:
        return detections

    boxes = pred.boxes.xyxy.cpu().numpy()
    confs = pred.boxes.conf.cpu().numpy()
    clss = pred.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, clss):
        detections.append({
            "box": box.astype(np.float32),
            "conf": float(conf),
            "cls_name": model_names[int(cls_id)],
        })

    return detections


def choose_best_truck_in_roi(detections, roi_mask, preferred_point, frame_shape):
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
            2.7 * roi_overlap +
            1.0 * distance_score(center, preferred_point, frame_diag) +
            0.5 * det["conf"] +
            0.6 * min(1.0, area / 250000.0)
        )

        if score > best_score:
            best_score = score
            best = {
                "box": box,
                "conf": det["conf"],
                "roi_overlap": roi_overlap,
                "center": center,
                "area": area,
                "pick_score": score,
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


def stable_reference_box(samples):
    """
    Стабильный референс текущей фуры:
    медиана по последним REFERENCE_WINDOW committed samples.
    """
    recent = samples[-REFERENCE_WINDOW:] if len(samples) >= REFERENCE_WINDOW else samples
    arr = np.array([s["truck_box"] for s in recent], dtype=np.float32)
    return np.median(arr, axis=0)


def candidate_looks_like_new_truck(candidate_box, ref_box):
    iou = bbox_iou(candidate_box, ref_box)
    center_dist = float(np.linalg.norm(bbox_center(candidate_box) - bbox_center(ref_box)))

    a1 = bbox_area_xyxy(candidate_box)
    a2 = bbox_area_xyxy(ref_box)
    area_ratio = max(a1 / max(a2, 1.0), a2 / max(a1, 1.0))

    # Считаем, что это новая фура, если 2 из 3 признаков указывают на смену
    votes = 0
    if iou < SPLIT_IOU_MIN:
        votes += 1
    if center_dist > SPLIT_CENTER_DIST_MAX:
        votes += 1
    if area_ratio > SPLIT_AREA_RATIO_MAX:
        votes += 1

    return votes >= 2, {
        "iou": iou,
        "center_dist": center_dist,
        "area_ratio": area_ratio,
        "votes": votes,
    }


def sample_quality(sample, center_t=None):
    q = (
        1.8 * sample["roi_overlap"] +
        0.8 * sample["truck_conf"] +
        0.8 * min(1.0, sample["sharpness"] / 120.0) -
        2.5 * sample["person_front_iou"]
    )
    if center_t is not None:
        q -= 0.02 * abs(sample["time_sec"] - center_t)
    return q


def finalize_event(event, cap, out_dir, roi_poly, save_debug=True):
    samples = event["samples"]

    if len(samples) < MIN_EVENT_SAMPLES:
        return None

    duration = samples[-1]["time_sec"] - samples[0]["time_sec"]
    if duration < MIN_EVENT_DURATION_SEC:
        return None

    center_t = 0.5 * (samples[0]["time_sec"] + samples[-1]["time_sec"])

    clean = [
        s for s in samples
        if s["person_front_iou"] <= MAX_PERSON_FRONT_IOU_FOR_CLEAN and s["sharpness"] >= MIN_SHARPNESS
    ]

    pool = clean if clean else samples

    ranked = sorted(
        pool,
        key=lambda s: sample_quality(s, center_t=center_t),
        reverse=True
    )[:SAVE_TOP_K_PER_EVENT]

    saved_paths = []
    debug_paths = []

    for idx, best in enumerate(ranked, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, best["frame_idx"])
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        stem = f"event_{event['event_id']:04d}"
        suffix = f"_k{idx}" if SAVE_TOP_K_PER_EVENT > 1 else ""

        img_path = out_dir / f"{stem}{suffix}.jpg"
        cv2.imwrite(str(img_path), frame_bgr)
        saved_paths.append(str(img_path))

        if save_debug:
            persons = [p["box"] for p in best["persons"]]
            dbg = draw_debug(
                frame_bgr,
                roi_poly,
                truck_box=best["truck_box"],
                front_box=best["front_box"],
                persons=persons,
                text=(
                    f"event={event['event_id']} "
                    f"t={best['time_sec']:.1f}s "
                    f"occ={best['person_front_iou']:.3f} "
                    f"sharp={best['sharpness']:.1f}"
                )
            )
            dbg_path = out_dir / f"{stem}{suffix}_debug.jpg"
            cv2.imwrite(str(dbg_path), dbg)
            debug_paths.append(str(dbg_path))

    if not saved_paths:
        return None

    return {
        "event_id": event["event_id"],
        "image_path": saved_paths[0],
        "saved_count": len(saved_paths),
        "start_sec": samples[0]["time_sec"],
        "end_sec": samples[-1]["time_sec"],
        "duration_sec": duration,
        "sample_count": len(samples),
        "clean_count": len(clean),
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
    print(f"Duration: {duration_sec / 60:.2f} min")

    sample_step_frames = max(1, int(round(SAMPLE_EVERY_SEC * fps)))
    print(f"Sampling every {SAMPLE_EVERY_SEC:.1f}s => every {sample_step_frames} frames")

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    roi_mask = build_roi_mask((frame_h, frame_w, 3), ROI_POLYGON)
    preferred_point = polygon_centroid(ROI_POLYGON) if PREFERRED_POINT is None else np.array(PREFERRED_POINT, dtype=np.float32)

    model = YOLO(YOLO_MODEL_PATH)
    model_names = model.names
    wanted_names = TARGET_VEHICLE_CLASS_NAMES | BLOCKER_CLASS_NAMES
    class_ids = [i for i, name in model_names.items() if name in wanted_names]

    print("Using class ids:", {i: model_names[i] for i in class_ids})
    print("Preferred point:", preferred_point.tolist())

    current_event = None
    pending_new = []          # сюда складываем samples, похожие на "новую фуру"
    empty_count = 0
    next_event_id = 1
    results_meta = []

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

        detections = extract_detections(pred, model_names)
        person_dets = [d for d in detections if d["cls_name"] in BLOCKER_CLASS_NAMES]
        person_boxes = [d["box"] for d in person_dets]

        truck = choose_best_truck_in_roi(
            detections=detections,
            roi_mask=roi_mask,
            preferred_point=preferred_point,
            frame_shape=frame_bgr.shape
        )

        if truck is None:
            # В ROI фуры нет
            if current_event is not None:
                empty_count += 1

                # pending_new сбрасываем, если фура исчезла
                pending_new = []

                if empty_count >= EMPTY_CONFIRM_STEPS:
                    meta = finalize_event(
                        event=current_event,
                        cap=cap,
                        out_dir=OUTPUT_DIR,
                        roi_poly=ROI_POLYGON,
                        save_debug=SAVE_DEBUG_VIS
                    )
                    if meta is not None:
                        results_meta.append(meta)

                    current_event = None
                    empty_count = 0
            frame_idx += sample_step_frames
            continue

        # Есть фура
        empty_count = 0

        person_front_iou, front_box = person_front_occlusion(truck["box"], person_boxes)
        truck_crop = crop_box(frame_bgr, truck["box"])
        sharpness = laplacian_sharpness(truck_crop) if truck_crop is not None else 0.0

        sample = {
            "frame_idx": frame_idx,
            "time_sec": time_sec,
            "truck_box": truck["box"].copy(),
            "truck_conf": truck["conf"],
            "roi_overlap": truck["roi_overlap"],
            "person_front_iou": person_front_iou,
            "front_box": front_box.copy(),
            "persons": person_dets,
            "sharpness": sharpness,
        }

        if current_event is None:
            current_event = {
                "event_id": next_event_id,
                "samples": [sample],
            }
            next_event_id += 1
            pending_new = []
            frame_idx += sample_step_frames
            continue

        # Сравниваем с текущей фурой
        ref_box = stable_reference_box(current_event["samples"])
        looks_new, dbg = candidate_looks_like_new_truck(sample["truck_box"], ref_box)

        if not looks_new:
            # Это всё ещё та же фура.
            # Если до этого была "подозрительная новая", но не подтвердилась — возвращаем её в текущее событие.
            if pending_new:
                current_event["samples"].extend(pending_new)
                pending_new = []

            current_event["samples"].append(sample)
        else:
            # Подозрение на новую фуру
            pending_new.append(sample)

            # Если смена подтвердилась несколько sampled-кадров подряд:
            if len(pending_new) >= NEW_TRUCK_CONFIRM_STEPS:
                # Закрываем старую фуру
                meta = finalize_event(
                    event=current_event,
                    cap=cap,
                    out_dir=OUTPUT_DIR,
                    roi_poly=ROI_POLYGON,
                    save_debug=SAVE_DEBUG_VIS
                )
                if meta is not None:
                    results_meta.append(meta)

                # Открываем новую фуру с накопленного pending_new
                current_event = {
                    "event_id": next_event_id,
                    "samples": pending_new.copy(),
                }
                next_event_id += 1
                pending_new = []

        if processed % 100 == 0:
            print(
                f"sampled={processed} "
                f"time={time_sec/60:.1f} min "
                f"saved_events={len(results_meta)} "
                f"current_samples={0 if current_event is None else len(current_event['samples'])}"
            )

        frame_idx += sample_step_frames

    # Закрываем хвост
    if current_event is not None:
        meta = finalize_event(
            event=current_event,
            cap=cap,
            out_dir=OUTPUT_DIR,
            roi_poly=ROI_POLYGON,
            save_debug=SAVE_DEBUG_VIS
        )
        if meta is not None:
            results_meta.append(meta)

    cap.release()

    csv_path = OUTPUT_DIR / "events.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "image_path",
                "saved_count",
                "start_sec",
                "end_sec",
                "duration_sec",
                "sample_count",
                "clean_count",
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