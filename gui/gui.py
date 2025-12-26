import os
import math
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from .blob_io import AzureBlobIO, BlobCache, Prefetcher
from .state_db import init_state_db, get_labeled_boxes, upsert_label, is_frame_exported, mark_frame_exported, is_box_exported, mark_box_exported
from .dataset_export import export_convnext_crop, export_yolo_frame, should_export_on_tp, should_export_on_fp

KYIV_TZ = ZoneInfo("Europe/Kyiv")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def unix_to_kyiv_str(unix_ts: float) -> str:
    try:
        dt = datetime.fromtimestamp(float(unix_ts), tz=ZoneInfo("UTC")).astimezone(KYIV_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return "N/A"

def compute_split_by_count(df: pd.DataFrame, ratio: float) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.array([], dtype=object)
    created = df["created_ts"].to_numpy(dtype=float)
    order = np.lexsort((np.arange(n), created))  # stable by index
    cut = int(math.ceil(ratio * n))
    split = np.empty(n, dtype=object)
    split[order[:cut]] = "train"
    split[order[cut:]] = "val"
    return split

class ImageCanvas(ttk.Frame):
    """Zoom/pan viewer with optional overlay rectangle in image coords."""
    def __init__(self, master, width=540, height=540):
        super().__init__(master)
        self.canvas = tk.Canvas(self, width=width, height=height, bg="#222222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._img_pil = None
        self._img_tk = None

        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._drag_start = None

        self.overlay_rect = None  # (x1,y1,x2,y2) in image coords or None
        self.overlay_enabled = True

        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)      # Windows
        self.canvas.bind("<Button-4>", self._on_mousewheel)        # Linux up
        self.canvas.bind("<Button-5>", self._on_mousewheel)        # Linux down
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)

    def set_image(self, pil_img: Image.Image):
        self._img_pil = pil_img
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._redraw()

    def set_overlay(self, rect_or_none):
        self.overlay_rect = rect_or_none
        self._redraw()

    def set_overlay_enabled(self, enabled: bool):
        self.overlay_enabled = enabled
        self._redraw()

    def _on_mousewheel(self, event):
        if self._img_pil is None:
            return
        if hasattr(event, "delta") and event.delta != 0:
            factor = 1.1 if event.delta > 0 else 0.9
        else:
            factor = 1.1 if event.num == 4 else 0.9
        self.scale = max(0.1, min(10.0, self.scale * factor))
        self._redraw()

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_drag_move(self, event):
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self.offset_x += dx
        self.offset_y += dy
        self._drag_start = (event.x, event.y)
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        if self._img_pil is None:
            return

        w, h = self._img_pil.size
        new_w = max(1, int(w * self.scale))
        new_h = max(1, int(h * self.scale))
        resized = self._img_pil.resize((new_w, new_h), Image.BILINEAR)
        self._img_tk = ImageTk.PhotoImage(resized)

        cx = self.canvas.winfo_width() // 2 + self.offset_x
        cy = self.canvas.winfo_height() // 2 + self.offset_y
        self.canvas.create_image(cx, cy, image=self._img_tk, anchor="center")

        if self.overlay_enabled and self.overlay_rect is not None:
            x1, y1, x2, y2 = self.overlay_rect
            img_left = cx - (w * self.scale) / 2
            img_top = cy - (h * self.scale) / 2
            X1 = img_left + x1 * self.scale
            Y1 = img_top + y1 * self.scale
            X2 = img_left + x2 * self.scale
            Y2 = img_top + y2 * self.scale
            self.canvas.create_rectangle(X1, Y1, X2, Y2, outline="red", width=2)

@dataclass
class Paths:
    work_dir: str
    filtered_dir: str
    cache_boxes: str
    cache_frames: str
    state_dir: str
    convnext_dir: str
    yolo_dir: str

class LabelGUI(tk.Tk):
    def __init__(self, blobio: AzureBlobIO, paths: Paths, split_ratio: float, yolo_thr: float, cnext_thr: float, tolerance: float):
        super().__init__()
        self.title("Labeler: TP / FP / Skip")

        self.blobio = blobio
        self.paths = paths
        self.split_ratio = float(split_ratio)
        self.yolo_thr = float(yolo_thr)
        self.cnext_thr = float(cnext_thr)
        self.tolerance = float(tolerance)

        ensure_dir(self.paths.filtered_dir)
        ensure_dir(self.paths.cache_boxes)
        ensure_dir(self.paths.cache_frames)
        ensure_dir(self.paths.state_dir)
        ensure_dir(self.paths.convnext_dir)
        ensure_dir(self.paths.yolo_dir)

        for split in ["train", "val"]:
            for cls in ["TP", "FP"]:
                ensure_dir(os.path.join(self.paths.convnext_dir, split, cls))

        self.cache = BlobCache(self.blobio, self.paths.cache_boxes, self.paths.cache_frames)
        self.prefetcher = Prefetcher(self.cache)

        self.con = init_state_db(os.path.join(self.paths.state_dir, "state.db"))

        self.df = None
        self.splits = None
        self.cam = None
        self.ptr = 0
        self.labeled_set = set()

        self.show_bbox_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._load_cameras()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Camera:").pack(side="left")
        self.cam_combo = ttk.Combobox(top, values=[], state="readonly", width=30)
        self.cam_combo.pack(side="left", padx=6)
        self.cam_combo.bind("<<ComboboxSelected>>", lambda e: self.load_camera())

        ttk.Label(top, text=f"split_ratio=").pack(side="left", padx=(10,2))
        ttk.Label(top, text=str(self.split_ratio)).pack(side="left")

        ttk.Label(top, text=f"yolo_thr=").pack(side="left", padx=(10,2))
        ttk.Label(top, text=str(self.yolo_thr)).pack(side="left")

        ttk.Label(top, text=f"cnext_thr=").pack(side="left", padx=(10,2))
        ttk.Label(top, text=str(self.cnext_thr)).pack(side="left")

        ttk.Label(top, text=f"tol=").pack(side="left", padx=(10,2))
        ttk.Label(top, text=str(self.tolerance)).pack(side="left")

        ttk.Checkbutton(top, text="Show bbox on frame", variable=self.show_bbox_var, command=self._toggle_bbox).pack(side="right")

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=8, pady=8)

        self.box_view = ImageCanvas(mid, width=520, height=520)
        self.box_view.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.frame_view = ImageCanvas(mid, width=880, height=520)
        self.frame_view.pack(side="left", fill="both", expand=True, padx=(6, 0))

        meta = ttk.Frame(self)
        meta.pack(fill="x", padx=8, pady=(0, 8))
        self.meta_lbl = ttk.Label(meta, text="(metadata)", justify="left")
        self.meta_lbl.pack(side="left")

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=8, pady=(0, 12))

        ttk.Button(btns, text="TP (человек)", command=lambda: self.on_label("TP")).pack(side="left", padx=6)
        ttk.Button(btns, text="FP (галлюцинация)", command=lambda: self.on_label("FP")).pack(side="left", padx=6)
        ttk.Button(btns, text="Skip", command=lambda: self.on_label("SKIP")).pack(side="left", padx=6)

        # hotkeys
        self.bind("<Right>", lambda e: self.on_label("TP"))
        self.bind("<Left>", lambda e: self.on_label("FP"))
        self.bind("<Down>", lambda e: self.on_label("SKIP"))

    def _toggle_bbox(self):
        self.frame_view.set_overlay_enabled(bool(self.show_bbox_var.get()))

    def _load_cameras(self):
        cams = self.blobio.list_cameras()
        self.cam_combo["values"] = cams
        if cams:
            self.cam_combo.current(0)
            self.load_camera()

    def load_camera(self):
        cam = self.cam_combo.get().strip()
        if not cam:
            return

        local_csv = os.path.join(self.paths.filtered_dir, f"{cam}_filtered.csv")
        if not os.path.exists(local_csv):
            # download from blob
            self.blobio.download_filtered_csv(cam, local_csv)

        self.cam = cam
        self.df = pd.read_csv(local_csv)

        required = ["box_blob","frame_blob","x1","y1","x2","y2","yolo_score","cnext_sscore","created_ts"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            messagebox.showerror("Bad CSV", f"Missing columns: {missing}")
            return

        self.splits = compute_split_by_count(self.df, self.split_ratio)

        self.labeled_set = get_labeled_boxes(self.con, cam)

        # resume: first not labeled
        self.ptr = 0
        while self.ptr < len(self.df) and str(self.df.iloc[self.ptr]["box_blob"]) in self.labeled_set:
            self.ptr += 1

        self.show_current()

    def show_current(self):
        if self.df is None or self.ptr >= len(self.df):
            messagebox.showinfo("Done", "Нет больше неразмеченных боксов в этом filtered.")
            return

        row = self.df.iloc[self.ptr]
        box_blob = str(row["box_blob"])
        frame_blob = str(row["frame_blob"])

        # prefetch next
        nxt = self._next_unlabeled_ptr(self.ptr + 1)
        if nxt is not None:
            nrow = self.df.iloc[nxt]
            self.prefetcher.prefetch(str(nrow["box_blob"]), str(nrow["frame_blob"]))

        # load images
        try:
            box_path = self.cache.get_box(box_blob)
            frame_path = self.cache.get_frame(frame_blob)
        except Exception as e:
            messagebox.showerror("Download error", str(e))
            return

        box_img = Image.open(box_path).convert("RGB")
        frame_img = Image.open(frame_path).convert("RGB")

        self.box_view.set_image(box_img)
        self.frame_view.set_image(frame_img)

        rect = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
        self.frame_view.set_overlay(rect)
        self.frame_view.set_overlay_enabled(bool(self.show_bbox_var.get()))

        split = self.splits[self.ptr]
        created_ts = float(row["created_ts"])
        kyiv_time = unix_to_kyiv_str(created_ts)

        self.meta_lbl.configure(text=
            f"cam: {self.cam} | row: {self.ptr+1}/{len(self.df)} | split: {split}\n"
            f"yolo_score: {row['yolo_score']} | cnext_sscore: {row['cnext_sscore']} | phash: {row.get('phash','')}\n"
            f"created_ts (Kyiv): {kyiv_time}\n"
            f"box: {os.path.basename(box_blob)}\n"
            f"frame: {os.path.basename(frame_blob)}"
        )

    def _next_unlabeled_ptr(self, start: int):
        if self.df is None:
            return None
        i = start
        while i < len(self.df):
            if str(self.df.iloc[i]["box_blob"]) not in self.labeled_set:
                return i
            i += 1
        return None

    def on_label(self, label: str):
        if self.df is None or self.cam is None or self.ptr >= len(self.df):
            return

        row = self.df.iloc[self.ptr]
        box_blob = str(row["box_blob"])
        frame_blob = str(row["frame_blob"])
        created_ts = float(row["created_ts"])
        split = str(self.splits[self.ptr])

        # store label in sqlite
        upsert_label(self.con, self.cam, box_blob, label, created_ts)
        self.labeled_set.add(box_blob)

        # parse scores
        yolo_score = float(row["yolo_score"])
        cnext_score = float(row["cnext_sscore"])

        # Exports per your new rules
        if label == "TP":
            export_frame, export_crop_tp = should_export_on_tp(
                yolo_score, cnext_score, self.yolo_thr, self.cnext_thr, self.tolerance
            )
            if export_crop_tp:
                if not is_box_exported(self.con, box_blob, "TP"):
                    box_path = self.cache.get_box(box_blob)
                    export_convnext_crop(box_path, self.paths.convnext_dir, split, "TP", box_blob)
                    mark_box_exported(self.con, self.cam, box_blob, split, "TP")

            if export_frame:
                if not is_frame_exported(self.con, frame_blob):
                    frame_path = self.cache.get_frame(frame_blob)
                    export_yolo_frame(frame_path, self.paths.yolo_dir, self.cam, split, frame_blob)
                    mark_frame_exported(self.con, self.cam, frame_blob, split)

        elif label == "FP":
            export_frame, export_crop_fp = should_export_on_fp(
                yolo_score, cnext_score, self.yolo_thr, self.cnext_thr, self.tolerance
            )
            if export_crop_fp:
                if not is_box_exported(self.con, box_blob, "FP"):
                    box_path = self.cache.get_box(box_blob)
                    export_convnext_crop(box_path, self.paths.convnext_dir, split, "FP", box_blob)
                    mark_box_exported(self.con, self.cam, box_blob, split, "FP")

            if export_frame:
                if not is_frame_exported(self.con, frame_blob):
                    frame_path = self.cache.get_frame(frame_blob)
                    export_yolo_frame(frame_path, self.paths.yolo_dir, self.cam, split, frame_blob)
                    mark_frame_exported(self.con, self.cam, frame_blob, split)

        # SKIP: nothing exported

        # advance to next unlabeled
        nxt = self._next_unlabeled_ptr(self.ptr + 1)
        if nxt is None:
            messagebox.showinfo("Done", "Дошли до конца (всё размечено или пропущено).")
            return
        self.ptr = nxt
        self.show_current()
