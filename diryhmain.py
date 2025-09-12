# -*- coding: utf-8 -*-
"""
Видео-аналитика на YOLO с трекингом и зонами.
- Классы: 0 male, 1 female, 2 staff(игнор), 3 car(только появление/исчезновение)
- Зоны из JSON (нормированные полигоны)
- CSV логи с entry/exit; аннотированное видео сохраняется локально
"""

# ============================ КОНСТАНТЫ ============================

WEIGHTS_PATH = "weights/best.pt"               # путь к кастомным весам YOLO
SOURCE_VIDEO = "input.mp4"                     # входное видео рядом со скриптом
OUTPUT_VIDEO = "output_annotated.mp4"          # куда писать аннотированное видео
ZONES_JSON_PATH = "zones.json"                 # файл зон
OUTPUT_CSV_PATH = "events.csv"                 # csv с событиями

# Карта индексов классов -> имена (наши бизнес-названия)
CLASS_NAMES = {0: "male", 1: "female", 2: "staff", 3: "car"}

# Порог детекции/IoU YOLO
CONF_THRESH = 0.25
IOU_THRESH = 0.45

# Трекинг
TRACKER_CONFIG = "bytetrack.yaml"              # используем встроенный ByteTrack

# Подтверждения (устойчивость к шуму)
INIT_CONSEC_FRAMES = 3     # сколько подряд кадров нужно увидеть объект, чтобы инициализировать трек «активным»
ENTRY_CONFIRM_FRAMES = 3   # подтверждение входа в зону (меньше = чувствительнее)
EXIT_CONFIRM_FRAMES = 5    # подтверждение выхода из зоны
LOST_FRAMES_TOLERANCE = 10 # сколько кадров терпим пропажу, прежде чем считать трек исчезнувшим

# Поведение по классам
ENABLE_CAR_LOGGING = True  # писать ли события для class 3 (car)
PERSON_CLASSES = {0, 1}    # классы, по которым считаем заходы/выходы по ЗОНАМ
IGNORE_CLASSES = {2}       # staff игнорируем полностью

# Отрисовка
DRAW_ZONES = True          # рисовать полигоны зон
DRAW_TRACKS = True         # рисовать треки, ID и класс

# ==================================================================

import json
import math
from pathlib import Path
from datetime import timedelta

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO


def hhmmss_from_frame(frame_idx: int, fps: float) -> str:
    """Перевод кадра в hh:mm:ss (округление вниз до секунды)."""
    total_seconds = int(frame_idx / max(fps, 1e-6))
    return str(timedelta(seconds=total_seconds))


def scale_polygon(poly_norm, width, height):
    """Масштабировать нормированный полигон [[x,y],...] в пиксели."""
    return np.array([[int(p[0] * width), int(p[1] * height)] for p in poly_norm], dtype=np.int32)


def point_in_polygons(point_xy, polygons_px):
    """
    Проверить принадлежность точки (x,y) хотя бы одному из полигонов.
    Возвращает True/False.
    """
    x, y = int(point_xy[0]), int(point_xy[1])
    for poly in polygons_px:
        # >0 внутри, =0 на границе, <0 снаружи
        if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
            return True
    return False


def find_zone_for_point(point_xy, zones_px):
    """
    Определить ID зоны, в которую попадает точка (центр бокса).
    zones_px: dict(zone_id(str) -> list[np.ndarray(poly_px)])
    Возвращает (zone_id:str | None)
    """
    for zid, polys in zones_px.items():
        if point_in_polygons(point_xy, polys):
            return zid
    return None


def draw_zones(frame, zones_px):
    """Нарисовать зоны на кадре."""
    overlay = frame.copy()
    for zid, polys in zones_px.items():
        color = (0, 255, 255)
        for poly in polys:
            cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=2)
        # подпись зоны (берём первую вершину первого полигона)
        try:
            pt = tuple(polys[0][0])
            cv2.putText(overlay, f"Zone {zid}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        except Exception:
            pass
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    return frame


def main():
    # --- Подготовка видео ---
    cap = cv2.VideoCapture(SOURCE_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {SOURCE_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter (same fps/size)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Не удалось создать выходное видео: {OUTPUT_VIDEO}")

    # --- Загрузка зон ---
    zones_px = {}
    if Path(ZONES_JSON_PATH).exists():
        with open(ZONES_JSON_PATH, "r", encoding="utf-8") as f:
            zones_norm = json.load(f)
        for zid, polys in zones_norm.items():
            zones_px[zid] = [scale_polygon(poly, width, height) for poly in polys]
    else:
        print(f"[ВНИМАНИЕ] Файл зон не найден: {ZONES_JSON_PATH}. Зоны отключены.")
        zones_px = {}

    # --- Модель YOLO ---
    model = YOLO(WEIGHTS_PATH)

    # Для лога событий
    rows = []  # временно складываем строки; в процессе тоже можно сбрасывать в df при желании

    # Состояние треков: по track_id
    # для PERSON (0/1):
    # {
    #   'class_id', 'class_name',
    #   'active' (bool),               # активирован после INIT_CONSEC_FRAMES
    #   'seen_consec',                 # подряд увиденных кадров
    #   'last_seen_frame',
    #   'in_zone' (bool),
    #   'current_zone' (str|None),
    #   'entry_time' (str|None),
    #   'entry_confirm', 'exit_confirm'
    # }
    #
    # для CAR (3):
    #   те же поля, но вместо зон — фиксируем только появление/исчезновение (zone="None")
    track_state = {}

    # Вспомогательный набор для отметки «кто был виден на текущем кадре»
    current_seen_ids = set()

    frame_idx = -1

    # Используем встроенный трекер: генератор по кадрам (YOLO сам читает видео)
    # Но нам нужен свой writer и логика, поэтому будем отправлять кадры YOLO из cap вручную:
    tracker = model.track(
        source=SOURCE_VIDEO,
        stream=True,
        tracker=TRACKER_CONFIG,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        verbose=False,
        persist=True,  # сохранять ID на протяжении видео
    )

    for result in tracker:
        frame_idx += 1
        frame = result.orig_img  # BGR np.ndarray
        if frame is None:
            break

        current_seen_ids.clear()

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)  # track ids
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(ids)):
                tid = int(ids[i])
                cls_id = int(cls[i])
                if cls_id in IGNORE_CLASSES:
                    continue  # staff полностью игнорируем

                if cls_id not in CLASS_NAMES:
                    continue  # на всякий случай — неизвестный класс

                class_name = CLASS_NAMES[cls_id]
                x1, y1, x2, y2 = xyxy[i]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                current_seen_ids.add(tid)

                # Инициализация состояния трека при первом появлении
                st = track_state.get(tid)
                if st is None:
                    st = {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "active": False,
                        "seen_consec": 0,
                        "last_seen_frame": frame_idx,
                        "in_zone": False,
                        "current_zone": None,
                        "entry_time": None,
                        "entry_confirm": 0,
                        "exit_confirm": 0,
                    }
                    track_state[tid] = st

                # Если класс внезапно «переопределился» — обновим на всякий случай
                st["class_id"] = cls_id
                st["class_name"] = class_name

                # Апдейт видимости
                if st["last_seen_frame"] == frame_idx - 1:
                    st["seen_consec"] += 1
                else:
                    st["seen_consec"] = 1
                st["last_seen_frame"] = frame_idx

                # Активация после INIT_CONSEC_FRAMES
                if not st["active"] and st["seen_consec"] >= INIT_CONSEC_FRAMES:
                    st["active"] = True
                    # Для car — запишем «вход в кадр» сразу при активации
                    if cls_id == 3 and ENABLE_CAR_LOGGING:
                        entry_time = hhmmss_from_frame(frame_idx, fps)
                        rows.append({
                            "object_type": "car",
                            "entry_time": entry_time,
                            "exit_time": None,
                            "zone": "None"
                        })

                # ------- Логика по классам -------
                if cls_id in PERSON_CLASSES:
                    # Определим зону по центру бокса
                    zone_id = find_zone_for_point((cx, cy), zones_px) if zones_px else None

                    if st["active"]:
                        # Если входим в новую зону
                        if zone_id is not None and (not st["in_zone"] or st["current_zone"] != zone_id):
                            st["entry_confirm"] += 1
                            st["exit_confirm"] = 0
                            if st["entry_confirm"] >= ENTRY_CONFIRM_FRAMES:
                                # если раньше были в другой зоне — считаем выход из неё
                                if st["in_zone"] and st["current_zone"] is not None and st["entry_time"] is not None:
                                    exit_time = hhmmss_from_frame(frame_idx, fps)
                                    rows.append({
                                        "object_type": st["class_name"],
                                        "entry_time": st["entry_time"],
                                        "exit_time": exit_time,
                                        "zone": st["current_zone"]
                                    })
                                # новый вход
                                st["in_zone"] = True
                                st["current_zone"] = zone_id
                                st["entry_time"] = hhmmss_from_frame(frame_idx, fps)
                                st["entry_confirm"] = 0  # сброс
                        # Если вышли из всех зон
                        elif zone_id is None and st["in_zone"]:
                            st["exit_confirm"] += 1
                            st["entry_confirm"] = 0
                            if st["exit_confirm"] >= EXIT_CONFIRM_FRAMES:
                                exit_time = hhmmss_from_frame(frame_idx, fps)
                                rows.append({
                                    "object_type": st["class_name"],
                                    "entry_time": st["entry_time"],
                                    "exit_time": exit_time,
                                    "zone": st["current_zone"]
                                })
                                # сброс состояния зоны
                                st["in_zone"] = False
                                st["current_zone"] = None
                                st["entry_time"] = None
                                st["exit_confirm"] = 0
                        else:
                            # остаёмся в той же зоне — сбрасываем счетчики подтверждений
                            st["entry_confirm"] = 0
                            st["exit_confirm"] = 0

                # ------- Отрисовка бокса/лейбла -------
                if DRAW_TRACKS:
                    color = (0, 255, 0) if cls_id in PERSON_CLASSES else (255, 0, 0) if cls_id == 3 else (200, 200, 200)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"ID {tid} {class_name}"
                    if cls_id in PERSON_CLASSES and track_state[tid]["current_zone"] is not None and track_state[tid]["in_zone"]:
                        label += f" @Z{track_state[tid]['current_zone']}"
                    cv2.putText(frame, label, (int(x1), max(int(y1)-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # Отрисуем центр
                cv2.circle(frame, (int(cx), int(cy)), 3, (0, 255, 255), -1)

        # Обработка исчезновений: те, кого не видели в текущем кадре
        # — если «активный» и (car) или (person в зоне) — закрываем событие.
        to_delete = []
        for tid, st in track_state.items():
            # пропущенные кадры
            missed = frame_idx - st["last_seen_frame"]
            if missed > LOST_FRAMES_TOLERANCE:
                # для person: если был в зоне — зафиксировать выход «по пропаже»
                if st["class_id"] in PERSON_CLASSES and st["active"] and st["in_zone"] and st["entry_time"] is not None:
                    # exit_time — по последнему виденному кадру (а не текущему)
                    exit_time = hhmmss_from_frame(st["last_seen_frame"], fps)
                    rows.append({
                        "object_type": st["class_name"],
                        "entry_time": st["entry_time"],
                        "exit_time": exit_time,
                        "zone": st["current_zone"]
                    })
                # для car: если активный — закрыть событие появление/исчезновение
                if st["class_id"] == 3 and st["active"] and ENABLE_CAR_LOGGING:
                    exit_time = hhmmss_from_frame(st["last_seen_frame"], fps)
                    # Найти последнюю строку без exit_time для car и закрыть её
                    for r in reversed(rows):
                        if r["object_type"] == "car" and r["exit_time"] is None:
                            r["exit_time"] = exit_time
                            break
                # готов к удалению
                to_delete.append(tid)

        for tid in to_delete:
            track_state.pop(tid, None)

        # Отрисовать зоны
        if DRAW_ZONES and zones_px:
            frame = draw_zones(frame, zones_px)

        # Запись кадра
        out.write(frame)

    # После прохода видео: если какие-то персоны "зависли" в зоне — закрыть по финальному кадру
    last_time = hhmmss_from_frame(frame_idx, fps if frame_idx >= 0 else 30.0)
    for tid, st in list(track_state.items()):
        if st["class_id"] in PERSON_CLASSES and st["active"] and st["in_zone"] and st["entry_time"] is not None:
            rows.append({
                "object_type": st["class_name"],
                "entry_time": st["entry_time"],
                "exit_time": last_time,
                "zone": st["current_zone"]
            })
        if st["class_id"] == 3 and st["active"] and ENABLE_CAR_LOGGING:
            # если у car осталась незакрытая запись — закроем
            for r in reversed(rows):
                if r["object_type"] == "car" and r["exit_time"] is None:
                    r["exit_time"] = last_time
                    break

    # --- Сохранение CSV ---
    # Требуемые столбцы: object_type, entry_time, exit_time, zone
    # Для car zone всегда "None" (строкой).
    # Убедимся, что все car записи имеют zone="None"
    for r in rows:
        if r["object_type"] == "car":
            r["zone"] = "None"

    df = pd.DataFrame(rows, columns=["object_type", "entry_time", "exit_time", "zone"])
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")

    # --- Завершение ---
    cap.release()
    out.release()
    print(f"Готово. CSV: {OUTPUT_CSV_PATH}, Видео: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()