# ============================================================
# KPP2 "SCALES" EARLY TRIGGER SNAPSHOT EXTRACTOR
# MOTION ONLY IN ENLARGED TRIGGER ZONE
# ------------------------------------------------------------
# Логика:
# - есть узкая TRIGGER_ZONE, куда должна войти нижняя средняя точка bbox фуры
# - движение считаем только в слегка увеличенной версии этой зоны
# - если в enlarged zone появилось движение -> включаем YOLO
# - если нижняя средняя точка bbox ВПЕРВЫЕ вошла в TRIGGER_ZONE -> сохраняем 1 кадр
# ============================================================

from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# КОНСТАНТЫ
# =========================

VIDEO_PATH = r"data/vesy.avi"
YOLO_MODEL_PATH = "models/yolo11m.pt"

OUTPUT_DIR = Path("kpp2_trigger_out_motion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "captures.csv"

# Если OpenCV врёт про fps, задай руками
VIDEO_FPS_OVERRIDE = 60.0

# YOLO
YOLO_CONF = 0.25
YOLO_IMGSZ = 640

# Обрабатывать каждый N-й кадр
# при 60 fps:
#   10 -> 6 fps
#   8  -> 7.5 fps
#   6  -> 10 fps
FRAME_STEP = 10

TARGET_CLASS_NAMES = {"truck"}

# Минимальная площадь bbox фуры
MIN_TRUCK_AREA_PX = 25000

# Узкая trigger-zone для НИЖНЕЙ СРЕДНЕЙ ТОЧКИ bbox
# Формат: [x, y]
TRIGGER_ZONE_POLYGON = np.array([
    [1220, 700],
    [1450, 700],
    [1450, 820],
    [1220, 820],
], dtype=np.int32)

# Насколько увеличить trigger-zone для motion detection
# Движение считаем ТОЛЬКО внутри этой увеличенной зоны
MOTION_ZONE_SCALE_X = 1.35
MOTION_ZONE_SCALE_Y = 1.80

# После срабатывания ждём, пока trigger-zone освободится
REARM_EMPTY_STEPS = 3

# Если в кадре несколько truck, выбираем лучший
# Если None -> центроид trigger-zone
PREFERRED_POINT = None

SAVE_DEBUG_VIS = True

# -------------------------
# Motion gating
# -------------------------
MOTION_DOWNSCALE = 0.5
MOTION_DIFF_THRESHOLD = 18
MOTION_ACTIVE_RATIO = 0.0035
YOLO_HOLD_STEPS_AFTER_MOTION = 8
WARMUP_STEPS = 8
MOTION_BLUR_KERNEL = 5


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


def draw_debug(frame_bgr, trigger_zone, motion_zone, truck_box=None, bottom_pt=None, text=None):
    vis = frame_bgr.copy()

    cv2.polylines(vis, [motion_zone.astype(np.int32)], True, (0, 255, 255), 2)
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


def scale_polygon(poly, scale_x=1.0, scale_y=1.0):
    center = polygon_centroid(poly)
    p = poly.astype(np.float32).copy()
    p[:, 0] = center[0] + (p[:, 0] - center[0]) * scale_x
    p[:, 1] = center[1] + (p[:, 1] - center[1]) * scale_y
    return np.round(p).astype(np.int32)


def build_polygon_mask(frame_shape_hw, polygon):
    h, w = frame_shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def preprocess_for_motion(frame_bgr, scale=1.0, blur_kernel=5):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    return gray


def motion_ratio(prev_gray, curr_gray, motion_mask, diff_threshold):
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, bin_img = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    active = np.logical_and(bin_img > 0, motion_mask > 0).sum()
    total = max(1, (motion_mask > 0).sum())
    return float(active) / float(total)


# =========================
# ОСНОВНОЙ КОД
# =========================

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

    fps_cv = cap.get(cv2.CAP_PROP_FPS)
    fps = VIDEO_FPS_OVERRIDE if VIDEO_FPS_OVERRIDE is not None else fps_cv
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0.0

    print(f"Video: {VIDEO_PATH}")
    print(f"FPS from OpenCV: {fps_cv:.3f}")
    print(f"FPS used: {fps:.3f}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration_sec / 60:.2f} min")
    print(f"Frame step: {FRAME_STEP}")
    print(f"Effective inference FPS (max): {fps / FRAME_STEP:.2f}")
    print(f"YOLO imgsz: {YOLO_IMGSZ}")

    trigger_centroid = polygon_centroid(TRIGGER_ZONE_POLYGON) if PREFERRED_POINT is None else np.array(PREFERRED_POINT, dtype=np.float32)
    motion_zone_polygon = scale_polygon(
        TRIGGER_ZONE_POLYGON,
        scale_x=MOTION_ZONE_SCALE_X,
        scale_y=MOTION_ZONE_SCALE_Y
    )

    model = YOLO(YOLO_MODEL_PATH)
    model_names = model.names
    class_ids = [i for i, name in model_names.items() if name in TARGET_CLASS_NAMES]

    print("Using classes:", {i: model_names[i] for i in class_ids})
    print("Trigger centroid:", trigger_centroid.tolist())
    print("Trigger zone:", TRIGGER_ZONE_POLYGON.tolist())
    print("Motion zone:", motion_zone_polygon.tolist())

    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    motion_h = int(round(cap_h * MOTION_DOWNSCALE))
    motion_w = int(round(cap_w * MOTION_DOWNSCALE))

    motion_zone_scaled = scale_polygon(
        TRIGGER_ZONE_POLYGON,
        scale_x=MOTION_ZONE_SCALE_X,
        scale_y=MOTION_ZONE_SCALE_Y
    )
    motion_zone_scaled = np.round(motion_zone_scaled * MOTION_DOWNSCALE).astype(np.int32)
    motion_mask = build_polygon_mask((motion_h, motion_w), motion_zone_scaled)

    records = []

    armed = True
    empty_steps = 0
    capture_id = 1
    prev_in_trigger = False

    prev_motion_gray = None
    yolo_hold_steps = 0

    frame_idx = 0
    processed = 0
    yolo_runs = 0
    motion_hits = 0

    while frame_idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break

        time_sec = frame_idx / fps
        processed += 1

        # -------------------------
        # Motion gating только в enlarged trigger-zone
        # -------------------------
        curr_motion_gray = preprocess_for_motion(
            frame_bgr,
            scale=MOTION_DOWNSCALE,
            blur_kernel=MOTION_BLUR_KERNEL
        )

        run_yolo = False
        motion_r = 0.0

        if prev_motion_gray is None:
            run_yolo = True
        elif processed <= WARMUP_STEPS:
            run_yolo = True
        else:
            motion_r = motion_ratio(
                prev_motion_gray,
                curr_motion_gray,
                motion_mask,
                MOTION_DIFF_THRESHOLD
            )

            if motion_r >= MOTION_ACTIVE_RATIO:
                motion_hits += 1
                yolo_hold_steps = YOLO_HOLD_STEPS_AFTER_MOTION
                run_yolo = True
            elif yolo_hold_steps > 0:
                yolo_hold_steps -= 1
                run_yolo = True
            else:
                run_yolo = False

        prev_motion_gray = curr_motion_gray

        if not run_yolo:
            current_in_trigger = False

            if not armed:
                empty_steps += 1
                if empty_steps >= REARM_EMPTY_STEPS:
                    armed = True
                    empty_steps = 0

            prev_in_trigger = current_in_trigger

            if processed % 100 == 0:
                print(
                    f"processed={processed} time={time_sec/60:.1f} min "
                    f"captures={len(records)} armed={armed} "
                    f"yolo_runs={yolo_runs} motion_hits={motion_hits} "
                    f"motion_ratio={motion_r:.5f}"
                )

            frame_idx += FRAME_STEP
            continue

        # -------------------------
        # YOLO
        # -------------------------
        yolo_runs += 1

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

            # Сохраняем в самый ранний момент ВХОДА в trigger-zone
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
                        motion_zone_polygon,
                        truck_box=truck["box"],
                        bottom_pt=bottom_pt,
                        text=(
                            f"id={capture_id} "
                            f"t={time_sec:.2f}s "
                            f"sharp={sharpness:.1f} "
                            f"motion={motion_r:.5f}"
                        )
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
                    "motion_ratio": motion_r,
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
            print(
                f"processed={processed} time={time_sec/60:.1f} min "
                f"captures={len(records)} armed={armed} "
                f"yolo_runs={yolo_runs} motion_hits={motion_hits} "
                f"motion_ratio={motion_r:.5f}"
            )

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
                "motion_ratio",
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
    print(f"YOLO runs: {yolo_runs} / processed sampled frames: {processed}")
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()