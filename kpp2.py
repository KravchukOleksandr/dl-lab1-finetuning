# ============================================================
# KPP2 "SCALES" EARLY TRIGGER SNAPSHOT EXTRACTOR
# ------------------------------------------------------------
# Логика:
#   если нижняя центральная точка bbox фуры ВПЕРВЫЕ вошла
#   в маленькую trigger-zone -> сразу сохраняем 1 кадр
#
# Без lane ROI.
# Без argparse.
# Все параметры сверху.
# ============================================================

from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# КОНСТАНТЫ
# =========================

VIDEO_PATH = r"data/kpp2_scales.mp4"
YOLO_MODEL_PATH = "models/yolo11m.pt"

OUTPUT_DIR = Path("kpp2_trigger_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "captures.csv"

# YOLO
YOLO_CONF = 0.25
YOLO_IMGSZ = 640

# Обрабатывать каждый N-й кадр
# 25 fps:
#   4 -> ~6.25 fps
#   3 -> ~8.3 fps
#   2 -> ~12.5 fps
FRAME_STEP = 4

# Целевой класс
TARGET_CLASS_NAMES = {"truck"}

# Минимальная площадь bbox фуры
MIN_TRUCK_AREA_PX = 25000

# Маленькая зона срабатывания для НИЖНЕЙ ЦЕНТРАЛЬНОЙ ТОЧКИ bbox
# Формат: [x, y]
TRIGGER_ZONE_POLYGON = np.array([
    [760, 700],
    [1040, 700],
    [1040, 860],
    [760, 860],
], dtype=np.int32)

# После срабатывания ждём, пока зона освободится N processed-кадров подряд
REARM_EMPTY_STEPS = 3

# Если в кадре несколько truck, выбираем лучший
# Если None -> центроид trigger-zone
PREFERRED_POINT = None

SAVE_DEBUG_VIS = True


# =========================
# УТИЛИТЫ
# =========================

def bbox_area_xyxy(box):
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_bottom_center(box):
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


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


def laplacian_sharpness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def distance_score(center, preferred_point, frame_diag):
    d = np.linalg.norm(center - preferred_point)
    return 1.0 - min(1.0, d / max(frame_diag, 1.0))


def draw_debug(frame_bgr, trigger_zone, truck_box=None, bottom_pt=None, text=None):
    vis = frame_bgr.copy()

    cv2.polylines(vis, [trigger_zone.astype(np.int32)], True, (255, 255, 0), 2)

    if truck_box is not None:
        x1, y1, x2, y2 = map(int, truck_box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if bottom_pt is not None:
        cv2.circle(vis, tuple(map(int, bottom_pt)), 7, (0, 0, 255), -1)

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


def choose_best_truck(detections, frame_shape, preferred_point):
    h, w = frame_shape[:2]
    frame_diag = (w ** 2 + h ** 2) ** 0.5

    best = None
    best_score = -1e9

    for det in detections:
        if det["cls_name"] not in TARGET_CLASS_NAMES:
            continue

        box = det["box"]
        area = bbox_area_xyxy(box)
        if area < MIN_TRUCK_AREA_PX:
            continue

        bottom_pt = bbox_bottom_center(box)

        score = (
            1.0 * det["conf"] +
            0.9 * min(1.0, area / 250000.0) +
            1.0 * distance_score(bottom_pt, preferred_point, frame_diag)
        )

        if score > best_score:
            best_score = score
            best = {
                "box": box,
                "conf": det["conf"],
                "bottom_pt": bottom_pt,
                "area": area,
                "score": score,
            }

    return best


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
    print(f"Frame step: {FRAME_STEP}")
    print(f"YOLO imgsz: {YOLO_IMGSZ}")

    trigger_centroid = polygon_centroid(TRIGGER_ZONE_POLYGON) if PREFERRED_POINT is None else np.array(PREFERRED_POINT, dtype=np.float32)

    model = YOLO(YOLO_MODEL_PATH)
    model_names = model.names
    class_ids = [i for i, name in model_names.items() if name in TARGET_CLASS_NAMES]

    print("Using classes:", {i: model_names[i] for i in class_ids})
    print("Trigger centroid:", trigger_centroid.tolist())

    records = []

    armed = True
    empty_steps = 0
    capture_id = 1

    prev_in_trigger = False

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
            imgsz=YOLO_IMGSZ,
            verbose=False
        )[0]

        detections = extract_detections(pred, model_names)
        truck = choose_best_truck(
            detections=detections,
            frame_shape=frame_bgr.shape,
            preferred_point=trigger_centroid
        )

        current_in_trigger = False

        if truck is not None:
            bottom_pt = truck["bottom_pt"]
            current_in_trigger = point_in_polygon(bottom_pt, TRIGGER_ZONE_POLYGON)

            # КЛЮЧЕВАЯ ЛОГИКА:
            # сохраняем только в момент ВХОДА в зону
            # то есть раньше было False, сейчас стало True
            if armed and (not prev_in_trigger) and current_in_trigger:
                crop = crop_box(frame_bgr, truck["box"])
                sharpness = laplacian_sharpness(crop) if crop is not None else 0.0

                stem = f"capture_{capture_id:05d}"
                img_path = OUTPUT_DIR / f"{stem}.jpg"
                cv2.imwrite(str(img_path), frame_bgr)

                dbg_path = ""
                if SAVE_DEBUG_VIS:
                    dbg = draw_debug(
                        frame_bgr,
                        TRIGGER_ZONE_POLYGON,
                        truck_box=truck["box"],
                        bottom_pt=bottom_pt,
                        text=f"id={capture_id} t={time_sec:.2f}s sharp={sharpness:.1f}"
                    )
                    dbg_file = OUTPUT_DIR / f"{stem}_debug.jpg"
                    cv2.imwrite(str(dbg_file), dbg)
                    dbg_path = str(dbg_file)

                records.append({
                    "capture_id": capture_id,
                    "image_path": str(img_path),
                    "debug_path": dbg_path,
                    "frame_idx": frame_idx,
                    "time_sec": time_sec,
                    "sharpness": sharpness,
                    "bbox_x1": float(truck["box"][0]),
                    "bbox_y1": float(truck["box"][1]),
                    "bbox_x2": float(truck["box"][2]),
                    "bbox_y2": float(truck["box"][3]),
                    "bottom_x": float(bottom_pt[0]),
                    "bottom_y": float(bottom_pt[1]),
                })

                print(f"[SAVE] capture_id={capture_id} frame={frame_idx} time={time_sec:.2f}s")
                capture_id += 1
                armed = False
                empty_steps = 0

        # Rearm: ждём, пока зона освободится несколько processed-кадров подряд
        if not current_in_trigger:
            if not armed:
                empty_steps += 1
                if empty_steps >= REARM_EMPTY_STEPS:
                    armed = True
                    empty_steps = 0
        else:
            empty_steps = 0

        prev_in_trigger = current_in_trigger

        if processed % 100 == 0:
            print(f"processed={processed} time={time_sec/60:.1f} min captures={len(records)} armed={armed}")

        frame_idx += FRAME_STEP

    cap.release()

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "capture_id",
                "image_path",
                "debug_path",
                "frame_idx",
                "time_sec",
                "sharpness",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "bottom_x",
                "bottom_y",
            ]
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print("\nDone.")
    print(f"Saved captures: {len(records)}")
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()