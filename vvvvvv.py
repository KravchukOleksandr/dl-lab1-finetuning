import cv2
import os
import glob
import random
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator

def yolo_video_from_folder(
    folder_path,
    class_probs,
    class_names,
    class_colors,
    output_video="output.mp4",
    fps=20
):
    # находим все картинки
    image_paths = sorted(
        glob.glob(os.path.join(folder_path, "*.jpg"))
        + glob.glob(os.path.join(folder_path, "*.png"))
    )

    if len(image_paths) == 0:
        raise ValueError("Нет изображений в папке!")

    # открываем первый кадр — чтобы узнать размер
    sample = cv2.imread(image_paths[0])
    h, w = sample.shape[:2]

    # создаём видеопоток
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        name = os.path.splitext(img_path)[0]
        label_path = name + ".txt"

        annotator = Annotator(img)

        # если лейбла нет — просто добавляем пустой кадр
        if not os.path.isfile(label_path):
            video.write(img)
            continue

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                x, y, bw, bh = map(float, parts[1:5])

                # генерируем случайный conf из диапазона
                conf_low, conf_high = class_probs[cls]
                conf = random.uniform(conf_low, conf_high)

                # YOLO normalized → пиксели
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                label = f"{class_names[cls]} {conf:.2f}"
                color = class_colors.get(cls, (0, 255, 0))

                annotator.box_label([x1, y1, x2, y2], label, color=color)

        # берём финальный кадр
        frame = annotator.result()
        video.write(frame)

    video.release()
    print(f"Видео сохранено как: {output_video}")


# --------------------------
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# --------------------------

folder = r"C:/path/to/your/folder"

class_probs = {
    0: (0.8, 0.95),
    1: (0.4, 0.7),
    2: (0.2, 0.4),
    3: (0.6, 1.0),
}

class_names = {
    0: "person",
    1: "helmet",
    2: "car",
    3: "object"
}

# цвета BGR
class_colors = {
    0: (255, 0, 0),    # person → синий
    1: (0, 0, 255),    # helmet → красный
    2: (0, 255, 0),    # car → зелёный
    3: (0, 255, 255)   # object → жёлтый
}

yolo_video_from_folder(
    folder_path=folder,
    class_probs=class_probs,
    class_names=class_names,
    class_colors=class_colors,
    output_video="yolo_render.mp4",
    fps=20
)