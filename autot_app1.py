import os
import io
import csv
import tempfile
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk
from azure.storage.blob import BlobServiceClient, ResourceNotFoundError


# ================== НАСТРОЙКИ ==================

CONNECTION_STRING = "ВАШ_CONNECTION_STRING"
CONTAINER_NAME = "ВАШ_CONTAINER_NAME"
META_PREFIX = "meta/"
TMP_DIR = os.path.join(tempfile.gettempdir(), "yolo_labeler_tmp")
os.makedirs(TMP_DIR, exist_ok=True)


# ================== STORAGE LAYER ==================

class BlobClientWrapper:
    def __init__(self, connection_string: str, container_name: str):
        self.service = BlobServiceClient.from_connection_string(connection_string)
        self.container = self.service.get_container_client(container_name)

    def list_cameras(self):
        """
        Ищем все meta/camXX_index.csv и по ним определяем список камер.
        """
        cameras = set()
        for blob in self.container.list_blobs(name_starts_with=META_PREFIX):
            name = blob.name
            if not name.endswith("_index.csv"):
                continue
            # meta/cam01_index.csv -> cam01
            base = name[len(META_PREFIX):]          # cam01_index.csv
            cam = base[:-len("_index.csv")]         # cam01
            cameras.add(cam)
        return sorted(cameras)

    def download_text(self, blob_name: str) -> str | None:
        try:
            bc = self.container.get_blob_client(blob_name)
            data = bc.download_blob().readall()
            return data.decode("utf-8")
        except ResourceNotFoundError:
            return None

    def upload_text(self, blob_name: str, text: str):
        bc = self.container.get_blob_client(blob_name)
        bc.upload_blob(text.encode("utf-8"), overwrite=True)

    def download_blob_to_path(self, blob_name: str, local_path: str):
        bc = self.container.get_blob_client(blob_name)
        with open(local_path, "wb") as f:
            data = bc.download_blob().readall()
            f.write(data)


# ================== DATA LAYER ==================

def parse_csv_text(text: str, fieldnames=None) -> list[dict]:
    if text is None:
        return []
    f = io.StringIO(text)
    if fieldnames is None:
        reader = csv.DictReader(f)
    else:
        reader = csv.DictReader(f, fieldnames=fieldnames)
    return list(reader)


class CameraLabels:
    """
    Представляет camXX_labels.csv в памяти.
    Формат CSV:
        box_blob,human_label,labeled_ts
    """
    def __init__(self, rows: list[dict]):
        self.labels = {}  # box_blob -> row
        for r in rows:
            box_blob = r.get("box_blob")
            if not box_blob:
                continue
            self.labels[box_blob] = r

    def is_labeled(self, box_blob: str) -> bool:
        return box_blob in self.labels

    def set_label(self, box_blob: str, label: str):
        now = datetime.utcnow().isoformat()
        row = self.labels.get(box_blob, {"box_blob": box_blob})
        row["human_label"] = label
        row["labeled_ts"] = now
        self.labels[box_blob] = row

    def to_csv_text(self) -> str:
        fieldnames = ["box_blob", "human_label", "labeled_ts"]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in self.labels.values():
            writer.writerow({
                "box_blob": row.get("box_blob", ""),
                "human_label": row.get("human_label", ""),
                "labeled_ts": row.get("labeled_ts", ""),
            })
        return output.getvalue()


class CameraIndex:
    """
    Представляет camXX_index.csv в памяти, но только неразмеченные строки.
    Формат CSV:
        box_blob,frame_blob,model_score,created_ts
    """
    def __init__(self, index_rows: list[dict], labels: CameraLabels):
        # фильтруем уже размеченные
        self.samples = [
            r for r in index_rows
            if r.get("box_blob") and not labels.is_labeled(r["box_blob"])
        ]
        self.position = 0
        self.history = []  # стек индексов для Back

    def has_current(self) -> bool:
        return 0 <= self.position < len(self.samples)

    def current(self) -> dict | None:
        if not self.has_current():
            return None
        return self.samples[self.position]

    def next(self):
        if self.has_current():
            self.history.append(self.position)
            self.position += 1

    def back(self):
        if self.history:
            self.position = self.history.pop()


class AppState:
    """
    Общее состояние приложения: текущая камера, индекс, разметка.
    """
    def __init__(self, blob_client: BlobClientWrapper):
        self.blob = blob_client
        self.current_camera: str | None = None
        self.index_rows: list[dict] = []
        self.labels: CameraLabels | None = None
        self.index: CameraIndex | None = None

    def load_camera(self, camera: str):
        self.current_camera = camera

        index_blob = f"{META_PREFIX}{camera}_index.csv"
        labels_blob = f"{META_PREFIX}{camera}_labels.csv"

        index_text = self.blob.download_text(index_blob)
        if index_text is None:
            raise RuntimeError(f"Индекс для камеры {camera} не найден: {index_blob}")
        self.index_rows = parse_csv_text(index_text)

        labels_text = self.blob.download_text(labels_blob)
        label_rows = parse_csv_text(labels_text) if labels_text else []
        self.labels = CameraLabels(label_rows)

        self.index = CameraIndex(self.index_rows, self.labels)

    def save_labels(self):
        if not self.current_camera or not self.labels:
            return
        labels_blob = f"{META_PREFIX}{self.current_camera}_labels.csv"
        text = self.labels.to_csv_text()
        self.blob.upload_text(labels_blob, text)


# ================== TKINTER UI ==================

class LabelingApp(tk.Tk):
    def __init__(self, blob_client: BlobClientWrapper):
        super().__init__()
        self.title("YOLO TP/FP Labeler")
        self.geometry("1400x800")

        self.blob_client = blob_client
        self.state = AppState(blob_client)

        # кеш скачанных frame-изображений: frame_blob -> local_path
        self.frame_cache = {}

        # UI элементы
        self._build_ui()

        # переменные для изображений и зума
        self.left_original = None    # PIL.Image для box
        self.right_original = None   # PIL.Image для frame
        self.left_scale = 1.0
        self.right_scale = 1.0

        self.left_photo = None
        self.right_photo = None

        # загрузка списка камер
        self._load_cameras()

        # бинды клавиш
        self._bind_hotkeys()

    # ---------- UI building ----------

    def _build_ui(self):
        # Верхняя панель (камера, кнопки)
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(top_frame, text="Camera:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(top_frame, textvariable=self.camera_var, state="readonly", width=15)
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)

        self.reload_button = ttk.Button(top_frame, text="Reload", command=self.on_reload)
        self.reload_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(top_frame, text="Save", command=self.on_save_clicked)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Центр: два Canvas
        center_frame = ttk.Frame(self)
        center_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # левый (box)
        left_container = ttk.Frame(center_frame)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(left_container, text="BOX").pack(side=tk.TOP)
        self.left_canvas = tk.Canvas(left_container, bg="black")
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # scroll (по желанию можно добавить)
        left_scroll_y = ttk.Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview)
        left_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.configure(yscrollcommand=left_scroll_y.set)

        # правый (frame)
        right_container = ttk.Frame(center_frame)
        right_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(right_container, text="FRAME").pack(side=tk.TOP)
        self.right_canvas = tk.Canvas(right_container, bg="black")
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_scroll_y = ttk.Scrollbar(right_container, orient="vertical", command=self.right_canvas.yview)
        right_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_canvas.configure(yscrollcommand=right_scroll_y.set)

        # Нижняя панель (статус)
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(bottom_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)

        help_text = "←: FP, →: TP, Backspace: Back, Ctrl+S: Save, колёсико+Ctrl или +/-: Zoom"
        ttk.Label(bottom_frame, text=help_text).pack(side=tk.RIGHT)

        # бинды для pan
        self.left_canvas.bind("<ButtonPress-1>", self.on_left_button_press)
        self.left_canvas.bind("<B1-Motion>", self.on_left_mouse_drag)
        self.right_canvas.bind("<ButtonPress-1>", self.on_right_button_press)
        self.right_canvas.bind("<B1-Motion>", self.on_right_mouse_drag)

    def _bind_hotkeys(self):
        self.bind("<Left>", lambda e: self.on_mark("FP"))
        self.bind("<Right>", lambda e: self.on_mark("TP"))
        self.bind("<BackSpace>", lambda e: self.on_back())
        self.bind("<Control-s>", lambda e: self.on_save_clicked())
        self.bind("<plus>", lambda e: self.on_zoom(1.1))
        self.bind("<minus>", lambda e: self.on_zoom(0.9))
        # Ctrl + колесо мыши – zoom
        self.bind("<Control-MouseWheel>", self.on_ctrl_mouse_wheel)

    # ---------- Cameras & loading ----------

    def _load_cameras(self):
        try:
            cameras = self.blob_client.list_cameras()
        except Exception as e:
            messagebox.showerror("Error", f"Ошибка при получении списка камер:\n{e}")
            return

        self.camera_combo["values"] = cameras
        if cameras:
            self.camera_combo.current(0)
            self.load_camera(cameras[0])

    def on_camera_selected(self, event=None):
        camera = self.camera_var.get()
        if camera:
            self.load_camera(camera)

    def on_reload(self):
        camera = self.camera_var.get()
        if camera:
            self.load_camera(camera)

    def load_camera(self, camera: str):
        try:
            self.state.load_camera(camera)
        except Exception as e:
            messagebox.showerror("Error", f"Не удалось загрузить камеру {camera}:\n{e}")
            return

        self.left_original = None
        self.right_original = None
        self.left_scale = 1.0
        self.right_scale = 1.0
        self.left_canvas.delete("all")
        self.right_canvas.delete("all")

        self.update_status()
        self.load_and_show_current_sample()

    # ---------- Status ----------

    def update_status(self):
        if not self.state.index:
            self.status_var.set("Нет данных")
            return

        total = len(self.state.index_rows)
        labeled = len(self.state.labels.labels) if self.state.labels else 0
        remaining = len(self.state.index.samples) if self.state.index else 0

        self.status_var.set(
            f"Camera: {self.state.current_camera} | labeled: {labeled} / {total} | remaining: {remaining}"
        )

    # ---------- Sample loading & drawing ----------

    def load_and_show_current_sample(self):
        idx = self.state.index
        if not idx or not idx.has_current():
            self.left_canvas.delete("all")
            self.right_canvas.delete("all")
            self.status_var.set(f"Camera: {self.state.current_camera} | Размечено всё.")
            return

        row = idx.current()
        box_blob = row["box_blob"]
        frame_blob = row["frame_blob"]
        score = row.get("model_score", "")

        # скачиваем box
        box_local = self.download_to_tmp(box_blob)

        # скачиваем / берём из кеша frame
        if frame_blob in self.frame_cache:
            frame_local = self.frame_cache[frame_blob]
        else:
            frame_local = self.download_to_tmp(frame_blob)
            self.frame_cache[frame_blob] = frame_local

        # загружаем как PIL.Image
        self.left_original = Image.open(box_local)
        self.right_original = Image.open(frame_local)
        self.left_scale = 1.0
        self.right_scale = 1.0

        self.draw_left()
        self.draw_right()

        self.update_status()
        # Добавим в статус score
        if score:
            self.status_var.set(self.status_var.get() + f" | model_score: {score}")

    def download_to_tmp(self, blob_name: str) -> str:
        local_path = os.path.join(TMP_DIR, blob_name.replace("/", "_"))
        if not os.path.exists(local_path):
            self.blob_client.download_blob_to_path(blob_name, local_path)
        return local_path

    def draw_left(self):
        if not self.left_original:
            return
        self._draw_on_canvas(self.left_canvas, self.left_original, "left")

    def draw_right(self):
        if not self.right_original:
            return
        self._draw_on_canvas(self.right_canvas, self.right_original, "right")

    def _draw_on_canvas(self, canvas: tk.Canvas, image: Image.Image, side: str):
        if side == "left":
            scale = self.left_scale
        else:
            scale = self.right_scale

        w, h = image.size
        w2, h2 = int(w * scale), int(h * scale)
        if w2 < 1: w2 = 1
        if h2 < 1: h2 = 1

        resized = image.resize((w2, h2))
        photo = ImageTk.PhotoImage(resized)

        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.image_ref = photo
        canvas.config(scrollregion=canvas.bbox("all"))

    # ---------- Label actions ----------

    def on_mark(self, label: str):
        idx = self.state.index
        labels = self.state.labels
        if not idx or not labels or not idx.has_current():
            return

        row = idx.current()
        box_blob = row["box_blob"]
        labels.set_label(box_blob, label)
        idx.next()
        self.load_and_show_current_sample()

    def on_back(self):
        idx = self.state.index
        if not idx:
            return
        idx.back()
        self.load_and_show_current_sample()

    def on_save_clicked(self):
        try:
            self.state.save_labels()
            messagebox.showinfo("Save", "Разметка сохранена в Blob.")
            self.update_status()
        except Exception as e:
            messagebox.showerror("Error", f"Ошибка при сохранении:\n{e}")

    # ---------- Zoom & Pan ----------

    def on_zoom(self, factor: float):
        # zoom обе картинки одновременно
        if self.left_original:
            self.left_scale *= factor
            self.draw_left()
        if self.right_original:
            self.right_scale *= factor
            self.draw_right()

    def on_ctrl_mouse_wheel(self, event):
        if event.delta > 0:
            self.on_zoom(1.1)
        else:
            self.on_zoom(0.9)

    # pan для левого canvas
    def on_left_button_press(self, event):
        self.left_canvas.scan_mark(event.x, event.y)

    def on_left_mouse_drag(self, event):
        self.left_canvas.scan_dragto(event.x, event.y, gain=1)

    # pan для правого canvas
    def on_right_button_press(self, event):
        self.right_canvas.scan_mark(event.x, event.y)

    def on_right_mouse_drag(self, event):
        self.right_canvas.scan_dragto(event.x, event.y, gain=1)


# ================== MAIN ==================

def main():
    blob_client = BlobClientWrapper(CONNECTION_STRING, CONTAINER_NAME)
    app = LabelingApp(blob_client)
    app.mainloop()


if __name__ == "__main__":
    main()