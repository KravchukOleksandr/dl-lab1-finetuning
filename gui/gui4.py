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
from .state_db import (
    init_state_db,
    get_labels_for_cam,
    upsert_label,

    # Frame export + refs (YOLO)
    is_frame_exported,
    get_exported_frame_info,
    mark_frame_exported,
    add_frame_ref,
    remove_frame_ref,
    get_frame_refcount,
    delete_exported_frame_row,

    # Crop export (ConvNeXt)
    is_box_exported,
    get_exported_box_info,
    mark_box_exported,
    delete_exported_box_row,
)
from .dataset_export import (
    export_convnext_crop,
    export_yolo_frame,
    should_export_on_tp,
    should_export_on_fp,
)

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
    """
    Assign "train"/"val" to each row in df (original row order),
    based on chronological order of created_ts (first ratio*N -> train).
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=object)

    created = df["created_ts"].to_numpy(dtype=float)
    order = np.lexsort((np.arange(n), created))  # stable by original index
    cut = int(math.ceil(ratio * n))

    split = np.empty(n, dtype=object)
    split[order[:cut]] = "train"
    split[order[cut:]] = "val"
    return split


def subsample_df(df: pd.DataFrame, k: int, mode: str) -> pd.DataFrame:
    """
    Stable per-camera subsampling.
    mode:
      - first_k: take first k in created_ts order
      - uniform_time: take k roughly uniformly across created_ts
    """
    if k <= 0 or len(df) <= k:
        return df

    if "created_ts" not in df.columns:
        return df.iloc[:k].copy()

    order = np.lexsort((np.arange(len(df)), df["created_ts"].to_numpy(dtype=float)))

    if mode == "first_k":
        chosen = order[:k]
        chosen = np.sort(chosen)  # keep UI browsing order stable
        return df.iloc[chosen].copy()

    # uniform_time
    pos = np.linspace(0, len(order) - 1, k)
    pos = np.round(pos).astype(int)
    pos = np.unique(pos)

    # If unique reduced size, pad deterministically
    t = 0
    while len(pos) < k:
        if len(pos) == 0:
            pos = np.array([0], dtype=int)
            continue
        cand = min(len(order) - 1, pos[-1] + 1) if t % 2 == 0 else max(0, pos[0] - 1)
        pos = np.unique(np.append(pos, cand))
        t += 1

    chosen = order[pos[:k]]
    chosen = np.sort(chosen)
    return df.iloc[chosen].copy()


class ImageCanvas(ttk.Frame):
    """
    Simple zoom/pan image viewer with optional rectangle overlay in image coordinates.
    """
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

        self.overlay_rect = None
        self.overlay_enabled = True

        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)      # Windows/macOS
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
    def __init__(
        self,
        blobio: AzureBlobIO,
        paths: Paths,
        split_ratio: float,
        yolo_thr: float,
        cnext_thr: float,
        tolerance: float,
        sampling_enabled: bool = False,
        max_per_camera: int = 0,
        sampling_mode: str = "uniform_time",
        sampling_seed: int = 123,  # reserved (not used now)
    ):
        super().__init__()
        self.title("Labeler: TP / FP / Skip")

        self.blobio = blobio
        self.paths = paths
        self.split_ratio = float(split_ratio)
        self.yolo_thr = float(yolo_thr)
        self.cnext_thr = float(cnext_thr)
        self.tolerance = float(tolerance)

        self.sampling_enabled = bool(sampling_enabled)
        self.max_per_camera = int(max_per_camera or 0)
        self.sampling_mode = str(sampling_mode or "uniform_time").lower()
        self.sampling_seed = int(sampling_seed)

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

        # box_blob -> label
        self.labels_map: dict[str, str] = {}

        # progress counters for current camera view (after subsampling)
        self.total_in_view = 0
        self.labeled_in_view = 0

        # relabel mode flag (allows overwriting existing labels safely)
        self.relabel_mode = False

        self.show_bbox_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._load_cameras()

    # ---------------- UI ----------------

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Camera:").pack(side="left")
        self.cam_combo = ttk.Combobox(top, values=[], state="readonly", width=30)
        self.cam_combo.pack(side="left", padx=6)
        self.cam_combo.bind("<<ComboboxSelected>>", lambda e: self.load_camera())

        self.progress_lbl = ttk.Label(top, text="Progress: 0/0")
        self.progress_lbl.pack(side="left", padx=14)

        self.mode_lbl = ttk.Label(top, text="", foreground="orange")
        self.mode_lbl.pack(side="left", padx=14)

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

        self.btn_prev = ttk.Button(btns, text="<= (prev)", command=self.go_prev)
        self.btn_prev.pack(side="left", padx=6)

        self.btn_next = ttk.Button(btns, text="=> (next)", command=self.go_next)
        self.btn_next.pack(side="left", padx=6)

        self.btn_tp = ttk.Button(btns, text="TP (person)", command=lambda: self.on_label("TP"))
        self.btn_tp.pack(side="left", padx=10)

        self.btn_fp = ttk.Button(btns, text="FP (hallucination)", command=lambda: self.on_label("FP"))
        self.btn_fp.pack(side="left", padx=6)

        self.btn_skip = ttk.Button(btns, text="Skip", command=lambda: self.on_label("SKIP"))
        self.btn_skip.pack(side="left", padx=6)

        # Relabel button
        self.btn_relabel = ttk.Button(btns, text="Relabel (R)", command=self.enable_relabel_mode)
        self.btn_relabel.pack(side="left", padx=12)

        # Labeling hotkeys
        self.bind("<Right>", lambda e: self.on_label("TP"))
        self.bind("<Left>", lambda e: self.on_label("FP"))
        self.bind("<Down>", lambda e: self.on_label("SKIP"))

        # Navigation hotkeys
        self.bind("<Control-Left>", lambda e: self.go_prev())
        self.bind("<Control-Right>", lambda e: self.go_next())
        self.bind("a", lambda e: self.go_prev())
        self.bind("d", lambda e: self.go_next())

        # Relabel hotkey
        self.bind("r", lambda e: self.enable_relabel_mode())

    def _toggle_bbox(self):
        self.frame_view.set_overlay_enabled(bool(self.show_bbox_var.get()))

    def _load_cameras(self):
        cams = self.blobio.list_cameras()
        self.cam_combo["values"] = cams
        if cams:
            self.cam_combo.current(0)
            self.load_camera()

    def _update_progress_label(self):
        self.progress_lbl.configure(text=f"Progress: {self.labeled_in_view}/{self.total_in_view}")

    def _update_mode_label(self):
        self.mode_lbl.configure(text="RELABEL MODE" if self.relabel_mode else "")

    # ---------------- Camera / data ----------------

    def load_camera(self):
        cam = self.cam_combo.get().strip()
        if not cam:
            return

        local_csv = os.path.join(self.paths.filtered_dir, f"{cam}_filtered.csv")
        if not os.path.exists(local_csv):
            self.blobio.download_filtered_csv(cam, local_csv)

        self.cam = cam
        df = pd.read_csv(local_csv)

        required = ["box_blob", "frame_blob", "x1", "y1", "x2", "y2", "yolo_score", "cnext_sscore", "created_ts"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            messagebox.showerror("Bad CSV", f"Missing columns: {missing}")
            return

        # Optional per-camera subsampling
        if self.sampling_enabled and self.max_per_camera > 0:
            df = subsample_df(df, self.max_per_camera, self.sampling_mode)

        self.df = df.reset_index(drop=True)
        self.splits = compute_split_by_count(self.df, self.split_ratio)

        # Load labels for camera
        self.labels_map = get_labels_for_cam(self.con, cam)

        # Compute progress for this view (after sampling)
        self.total_in_view = len(self.df)
        if self.total_in_view == 0:
            self.labeled_in_view = 0
        else:
            bb = self.df["box_blob"].astype(str)
            self.labeled_in_view = int(bb.isin(set(self.labels_map.keys())).sum())
        self._update_progress_label()

        # Reset relabel mode
        self.relabel_mode = False
        self._update_mode_label()

        # Resume to first unlabeled if possible, else start at 0 (browsing)
        self.ptr = 0
        while self.ptr < len(self.df) and str(self.df.iloc[self.ptr]["box_blob"]) in self.labels_map:
            self.ptr += 1
        if self.ptr >= len(self.df):
            self.ptr = 0

        self.show_current()

    # ---------------- Navigation ----------------

    def go_prev(self):
        if self.df is None:
            return
        if self.ptr > 0:
            self.ptr -= 1
            self.relabel_mode = False
            self._update_mode_label()
            self.show_current()

    def go_next(self):
        if self.df is None:
            return
        if self.ptr < len(self.df) - 1:
            self.ptr += 1
            self.relabel_mode = False
            self._update_mode_label()
            self.show_current()

    # ---------------- Relabel ----------------

    def enable_relabel_mode(self):
        """
        Enable relabel mode for current box if it is already labeled.
        In relabel mode, TP/FP/SKIP buttons become available and overwrite previous label safely.
        """
        if self.df is None or self.cam is None:
            return
        if self.ptr < 0 or self.ptr >= len(self.df):
            return

        row = self.df.iloc[self.ptr]
        box_blob = str(row["box_blob"])

        if box_blob not in self.labels_map:
            # Nothing to relabel
            return

        old_label = self.labels_map[box_blob]
        ok = messagebox.askyesno("Relabel", f"This box is already labeled as {old_label}. Relabel?")
        if not ok:
            return

        self.relabel_mode = True
        self._update_mode_label()

        # Allow labeling buttons while in relabel mode
        self.btn_tp.configure(state="normal")
        self.btn_fp.configure(state="normal")
        self.btn_skip.configure(state="normal")

    # ---------------- UI state ----------------

    def _update_label_buttons_state(self, box_blob: str):
        """
        Disable TP/FP/SKIP if the item is already labeled, unless relabel mode is active.
        """
        labeled = box_blob in self.labels_map
        if labeled and not self.relabel_mode:
            state = "disabled"
        else:
            state = "normal"

        self.btn_tp.configure(state=state)
        self.btn_fp.configure(state=state)
        self.btn_skip.configure(state=state)

        # Relabel button should be enabled only for already labeled items
        self.btn_relabel.configure(state=("normal" if labeled else "disabled"))

    # ---------------- Rendering ----------------

    def show_current(self):
        if self.df is None or self.ptr < 0 or self.ptr >= len(self.df):
            return

        row = self.df.iloc[self.ptr]
        box_blob = str(row["box_blob"])
        frame_blob = str(row["frame_blob"])

        # Prefetch next row for smoother browsing
        if self.ptr + 1 < len(self.df):
            nrow = self.df.iloc[self.ptr + 1]
            self.prefetcher.prefetch(str(nrow["box_blob"]), str(nrow["frame_blob"]))

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

        split = str(self.splits[self.ptr])
        created_ts = float(row["created_ts"])
        kyiv_time = unix_to_kyiv_str(created_ts)

        labeled_status = self.labels_map.get(box_blob, "UNLABELED")
        self._update_label_buttons_state(box_blob)

        self.meta_lbl.configure(text=
            f"cam: {self.cam} | row: {self.ptr+1}/{len(self.df)} | split: {split} | status: {labeled_status}\n"
            f"yolo_score: {row['yolo_score']} | cnext_sscore: {row['cnext_sscore']} | phash: {row.get('phash','')}\n"
            f"created_ts (Kyiv): {kyiv_time}\n"
            f"box: {os.path.basename(box_blob)}\n"
            f"frame: {os.path.basename(frame_blob)}"
        )

    # ---------------- Labeling / exporting ----------------

    def _advance_to_next_unlabeled_from(self, start: int) -> int | None:
        if self.df is None:
            return None
        i = start
        while i < len(self.df):
            bb = str(self.df.iloc[i]["box_blob"])
            if bb not in self.labels_map:
                return i
            i += 1
        return None

    def _undo_previous_exports_for_box(self, box_blob: str, frame_blob: str):
        """
        Safely undo previous exports caused by this box:
        - ConvNeXt crop: remove old TP/FP crop file if it exists and delete exported_boxes row.
        - YOLO frame: remove reference (frame_refs). Delete actual frame file ONLY if no refs remain.
        """
        # Undo ConvNeXt crop(s) for this box: could be TP and/or FP historically
        for cls in ("TP", "FP"):
            info = get_exported_box_info(self.con, box_blob, cls)
            if info is not None:
                _cam, split, _cls = info
                crop_path = os.path.join(self.paths.convnext_dir, split, cls, os.path.basename(box_blob))
                if os.path.exists(crop_path):
                    try:
                        os.remove(crop_path)
                    except Exception:
                        pass
                delete_exported_box_row(self.con, box_blob, cls)

        # Undo YOLO frame reference for this box
        remove_frame_ref(self.con, frame_blob, box_blob)
        refcount = get_frame_refcount(self.con, frame_blob)

        # If no refs remain, delete the exported frame file and its exported_frames row
        if refcount == 0:
            finfo = get_exported_frame_info(self.con, frame_blob)
            if finfo is not None:
                fcam, split = finfo
                frame_path = os.path.join(self.paths.yolo_dir, f"{fcam}-{split}", os.path.basename(frame_blob))
                if os.path.exists(frame_path):
                    try:
                        os.remove(frame_path)
                    except Exception:
                        pass
                delete_exported_frame_row(self.con, frame_blob)

    def on_label(self, label: str):
        if self.df is None or self.cam is None:
            return
        if self.ptr < 0 or self.ptr >= len(self.df):
            return

        row = self.df.iloc[self.ptr]
        box_blob = str(row["box_blob"])
        frame_blob = str(row["frame_blob"])

        already_labeled = box_blob in self.labels_map
        if already_labeled and not self.relabel_mode:
            # Prevent accidental overwrites when not in relabel mode
            return

        created_ts = float(row["created_ts"])
        split = str(self.splits[self.ptr])

        # If we are relabeling, remove previous exports caused by this box before applying new rules
        if already_labeled and self.relabel_mode:
            self._undo_previous_exports_for_box(box_blob, frame_blob)

        # Persist label in DB (insert or overwrite)
        upsert_label(self.con, self.cam, box_blob, label, created_ts)
        self.labels_map[box_blob] = label

        # Update progress counter only if it was previously unlabeled
        if not already_labeled:
            self.labeled_in_view += 1
            self._update_progress_label()

        # Exit relabel mode after applying new label
        self.relabel_mode = False
        self._update_mode_label()

        # Apply export rules according to your tolerance logic
        yolo_score = float(row["yolo_score"])
        cnext_score = float(row["cnext_sscore"])

        if label == "TP":
            export_frame, export_crop_tp = should_export_on_tp(
                yolo_score, cnext_score, self.yolo_thr, self.cnext_thr, self.tolerance
            )

            # ConvNeXt TP crop
            if export_crop_tp and not is_box_exported(self.con, box_blob, "TP"):
                box_path = self.cache.get_box(box_blob)
                export_convnext_crop(box_path, self.paths.convnext_dir, split, "TP", box_blob)
                mark_box_exported(self.con, self.cam, box_blob, split, "TP")

            # YOLO frame (with ref tracking)
            if export_frame:
                if not is_frame_exported(self.con, frame_blob):
                    frame_path = self.cache.get_frame(frame_blob)
                    export_yolo_frame(frame_path, self.paths.yolo_dir, self.cam, split, frame_blob)
                    mark_frame_exported(self.con, self.cam, frame_blob, split)
                add_frame_ref(self.con, frame_blob, box_blob)

        elif label == "FP":
            export_frame, export_crop_fp = should_export_on_fp(
                yolo_score, cnext_score, self.yolo_thr, self.cnext_thr, self.tolerance
            )

            # ConvNeXt FP crop
            if export_crop_fp and not is_box_exported(self.con, box_blob, "FP"):
                box_path = self.cache.get_box(box_blob)
                export_convnext_crop(box_path, self.paths.convnext_dir, split, "FP", box_blob)
                mark_box_exported(self.con, self.cam, box_blob, split, "FP")

            # YOLO frame (with ref tracking)
            if export_frame:
                if not is_frame_exported(self.con, frame_blob):
                    frame_path = self.cache.get_frame(frame_blob)
                    export_yolo_frame(frame_path, self.paths.yolo_dir, self.cam, split, frame_blob)
                    mark_frame_exported(self.con, self.cam, frame_blob, split)
                add_frame_ref(self.con, frame_blob, box_blob)

        else:
            # SKIP: no exports, and also no frame ref should remain
            # If we relabeled from TP/FP to SKIP, undo logic already removed refs.
            pass

        # Jump to next unlabeled for fast workflow, but manual browsing still possible
        nxt = self._advance_to_next_unlabeled_from(self.ptr + 1)
        if nxt is None:
            self.show_current()
            messagebox.showinfo("Done", "No more unlabeled items ahead (you can browse manually).")
            return

        self.ptr = nxt
        self.show_current()
