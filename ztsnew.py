import os
import json
import shutil
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from PIL import Image, ImageTk


# Bright palette (cycled)
PALETTE = [
    "#00E5FF", "#FF1744", "#76FF03", "#FFEA00", "#7C4DFF",
    "#FF9100", "#1DE9B6", "#F500FF", "#00C853", "#2979FF",
    "#FF5252", "#C6FF00", "#FFD600", "#651FFF", "#FF6D00",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_closed(poly: List[List[float]]) -> bool:
    """Check if polygon is closed (last point equals first)."""
    if len(poly) < 3:
        return False
    return poly[0] == poly[-1]


def close_polygon(poly_open: List[List[float]]) -> List[List[float]]:
    """Return a closed polygon. Input assumed open (no duplicated last point)."""
    if len(poly_open) < 3:
        return poly_open[:]  # do not force-close invalid polygons
    return poly_open + [poly_open[0]]


def open_polygon(poly_closed: List[List[float]]) -> List[List[float]]:
    """Return an open polygon (remove duplicate last point if it equals first)."""
    if len(poly_closed) >= 2 and poly_closed[0] == poly_closed[-1]:
        return poly_closed[:-1]
    return poly_closed[:]


def sort_numeric_str_keys(keys: List[str]) -> List[str]:
    """Sort keys like ['0','10','2'] numerically."""
    def key_fn(k: str):
        return (0, int(k)) if k.isdigit() else (1, k)
    return sorted(keys, key=key_fn)


def next_free_numeric_id(existing: List[str]) -> str:
    """Return next free numeric string id starting from 0."""
    used = set(int(k) for k in existing if k.isdigit())
    i = 0
    while i in used:
        i += 1
    return str(i)


@dataclass
class FrameContext:
    frame_path: str
    stem: str
    out_dir: str
    zone_path: str
    plane_path: str
    zone3d_path: str


class ZoomPanCanvas(ttk.Frame):
    """
    Canvas with image + zoom/pan and custom drawing.
    We keep image in canvas center + offsets. All overlay points are in normalized [0..1].
    """
    def __init__(self, master, bg="#1f1f1f"):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._img_pil: Optional[Image.Image] = None
        self._img_tk: Optional[ImageTk.PhotoImage] = None
        self._img_id = None

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._drag_start = None

        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows/macOS
        self.canvas.bind("<Button-4>", self._on_mousewheel)    # Linux up
        self.canvas.bind("<Button-5>", self._on_mousewheel)    # Linux down
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)

        # Expose left click events to parent
        self.on_left_click = None
        self.on_left_drag = None
        self.on_left_release = None

        self.canvas.bind("<ButtonPress-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)

    def set_image(self, img: Image.Image):
        self._img_pil = img
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.redraw()

    def img_size(self) -> Tuple[int, int]:
        if self._img_pil is None:
            return (1, 1)
        return self._img_pil.size

    def canvas_center(self) -> Tuple[float, float]:
        return (self.canvas.winfo_width() / 2.0, self.canvas.winfo_height() / 2.0)

    def norm_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        """Normalized [0..1] -> canvas coordinates."""
        W, H = self.img_size()
        cx, cy = self.canvas_center()
        img_w = W * self.scale
        img_h = H * self.scale
        left = cx - img_w / 2.0 + self.offset_x
        top = cy - img_h / 2.0 + self.offset_y
        return (left + x * img_w, top + y * img_h)

    def canvas_to_norm(self, X: float, Y: float) -> Tuple[float, float]:
        """Canvas coordinates -> normalized [0..1]."""
        W, H = self.img_size()
        cx, cy = self.canvas_center()
        img_w = W * self.scale
        img_h = H * self.scale
        left = cx - img_w / 2.0 + self.offset_x
        top = cy - img_h / 2.0 + self.offset_y
        x = (X - left) / img_w
        y = (Y - top) / img_h
        return (x, y)

    def clamp_norm(self, x: float, y: float) -> Tuple[float, float]:
        return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

    def _on_mousewheel(self, event):
        if self._img_pil is None:
            return
        if hasattr(event, "delta") and event.delta != 0:
            factor = 1.1 if event.delta > 0 else 0.9
        else:
            factor = 1.1 if event.num == 4 else 0.9
        self.scale = max(0.1, min(12.0, self.scale * factor))
        self.redraw()

    def _on_pan_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_pan_move(self, event):
        if not self._drag_start:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self.offset_x += dx
        self.offset_y += dy
        self._drag_start = (event.x, event.y)
        self.redraw()

    def _on_left_down(self, event):
        if self.on_left_click:
            self.on_left_click(event.x, event.y)

    def _on_left_move(self, event):
        if self.on_left_drag:
            self.on_left_drag(event.x, event.y)

    def _on_left_up(self, event):
        if self.on_left_release:
            self.on_left_release(event.x, event.y)

    def redraw(self):
        self.canvas.delete("all")
        if self._img_pil is None:
            return
        W, H = self._img_pil.size
        new_w = max(1, int(W * self.scale))
        new_h = max(1, int(H * self.scale))
        img_resized = self._img_pil.resize((new_w, new_h), Image.BILINEAR)
        self._img_tk = ImageTk.PhotoImage(img_resized)

        cx, cy = self.canvas_center()
        self._img_id = self.canvas.create_image(cx + self.offset_x, cy + self.offset_y, image=self._img_tk, anchor="center")


class ZonePlanesLabeler(tk.Tk):
    """
    Minimal but functional labeler for:
    - zone.json: {"subzone_id": [polygon0, polygon1, ...]}
    - plane.json: {"plane_id": [polygon0, polygon1, ...]}  where plane_id is "0..N-1"
    - zone3d.json: same as zone.json but points [x,y,p], p=int plane id
    """
    def __init__(self):
        super().__init__()
        self.title("Zone / Planes / Zone3D Labeler (Tkinter)")

        # Input dirs
        self.frames_dir = ""
        self.results_dir = ""

        # Frame list
        self.frames: List[str] = []
        self.frame_idx = 0
        self.ctx: Optional[FrameContext] = None

        # Data models (open polygons internally)
        self.zone: Dict[str, List[List[List[float]]]] = {}     # subzone -> list[poly_open] where point=[x,y]
        self.planes: Dict[str, List[List[List[float]]]] = {}   # plane_id -> list[poly_open]
        self.zone3d: Dict[str, List[List[List[float]]]] = {}   # subzone -> list[poly_open] where point=[x,y,p]

        # Current selection (for editor)
        self.mode = "ZONE"  # ZONE | PLANES | BIND
        self.cur_group = None  # subzone_id or plane_id (string)
        self.cur_poly_idx = 0
        self.cur_vertex_idx = None

        # Editor state
        self.tool = "ADD"  # ADD | MOVE | DELETE
        self._dragging_vertex = False

        # UI
        self._build_ui()
        self._bind_hotkeys()

    # ---------------- UI ----------------

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Button(top, text="Open frames folder", command=self.pick_frames_dir).pack(side="left", padx=4)
        ttk.Button(top, text="Open results folder", command=self.pick_results_dir).pack(side="left", padx=4)

        self.lbl_paths = ttk.Label(top, text="frames_dir: - | results_dir: -")
        self.lbl_paths.pack(side="left", padx=10)

        nav = ttk.Frame(self)
        nav.pack(fill="x", padx=8, pady=6)

        ttk.Button(nav, text="<< Prev", command=self.prev_frame).pack(side="left", padx=4)
        ttk.Button(nav, text="Next >>", command=self.next_frame).pack(side="left", padx=4)
        ttk.Button(nav, text="Save (Ctrl+S)", command=self.save_all).pack(side="left", padx=10)

        self.lbl_frame = ttk.Label(nav, text="Frame: -")
        self.lbl_frame.pack(side="left", padx=10)

        self.status = ttk.Label(nav, text="", foreground="orange")
        self.status.pack(side="right")

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=6)

        # Left: list + controls
        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 8))

        ttk.Label(left, text="Frames").pack(anchor="w")
        self.lst_frames = tk.Listbox(left, height=18, width=28)
        self.lst_frames.pack(fill="y", expand=False)
        self.lst_frames.bind("<<ListboxSelect>>", self._on_frame_select)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        # Mode tabs
        tabs = ttk.Frame(left)
        tabs.pack(fill="x")

        ttk.Button(tabs, text="ZONE", command=lambda: self.set_mode("ZONE")).pack(side="left", padx=2)
        ttk.Button(tabs, text="PLANES", command=lambda: self.set_mode("PLANES")).pack(side="left", padx=2)
        ttk.Button(tabs, text="BIND", command=lambda: self.set_mode("BIND")).pack(side="left", padx=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(left, text="Groups (subzones / planes)").pack(anchor="w")
        self.tree = ttk.Treeview(left, show="tree", height=12)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        grp_btns = ttk.Frame(left)
        grp_btns.pack(fill="x", pady=4)

        self.btn_add_group_auto = ttk.Button(grp_btns, text="Add group (auto)", command=self.add_group_auto)
        self.btn_add_group_auto.pack(side="left", padx=2)

        self.btn_add_group_custom = ttk.Button(grp_btns, text="Add group (custom)", command=self.add_group_custom)
        self.btn_add_group_custom.pack(side="left", padx=2)

        poly_btns = ttk.Frame(left)
        poly_btns.pack(fill="x", pady=4)

        ttk.Button(poly_btns, text="Add polygon", command=self.add_polygon).pack(side="left", padx=2)
        ttk.Button(poly_btns, text="Del polygon", command=self.del_polygon).pack(side="left", padx=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        # Tools
        ttk.Label(left, text="Tool").pack(anchor="w")
        tool_row = ttk.Frame(left)
        tool_row.pack(fill="x", pady=2)

        self.tool_var = tk.StringVar(value="ADD")
        ttk.Radiobutton(tool_row, text="Add", variable=self.tool_var, value="ADD", command=self._set_tool).pack(side="left", padx=2)
        ttk.Radiobutton(tool_row, text="Move", variable=self.tool_var, value="MOVE", command=self._set_tool).pack(side="left", padx=2)
        ttk.Radiobutton(tool_row, text="Del", variable=self.tool_var, value="DELETE", command=self._set_tool).pack(side="left", padx=2)

        ttk.Label(left, text="Tips: Left-click to add/move/delete vertices.\n"
                             "Enter = finish polygon. N = next unbound vertex.\n"
                             "In BIND: keys 0..4 assign plane.\n"
                             "Pan with right/middle mouse, zoom with wheel.").pack(anchor="w", pady=6)

        # Right: canvas + bind plane buttons + legend
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        self.canvas = ZoomPanCanvas(right)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.on_left_click = self._canvas_click
        self.canvas.on_left_drag = self._canvas_drag
        self.canvas.on_left_release = self._canvas_release

        bottom = ttk.Frame(right)
        bottom.pack(fill="x", pady=6)

        self.bind_bar = ttk.Frame(bottom)
        self.bind_bar.pack(side="left")

        ttk.Label(self.bind_bar, text="Bind plane:").pack(side="left", padx=(0, 6))
        self.plane_buttons: List[ttk.Button] = []
        for i in range(5):
            b = ttk.Button(self.bind_bar, text=f"{i}", command=lambda p=i: self.assign_plane_to_selected_vertex(p))
            b.pack(side="left", padx=2)
            self.plane_buttons.append(b)

        self.legend = ttk.Label(bottom, text="", justify="left")
        self.legend.pack(side="right", padx=8)

        self._update_ui_state()

    def _bind_hotkeys(self):
        self.bind("<Control-s>", lambda e: self.save_all())
        self.bind("<Return>", lambda e: self.finish_polygon())
        self.bind("<Escape>", lambda e: self.cancel_vertex_selection())
        self.bind("n", lambda e: self.next_unbound_vertex())

        # In BIND: assign plane by keys 0..4
        for k in "01234":
            self.bind(k, lambda e, kk=k: self.assign_plane_to_selected_vertex(int(kk)))

    def _set_tool(self):
        self.tool = self.tool_var.get()

    # ---------------- Folder selection ----------------

    def pick_frames_dir(self):
        d = filedialog.askdirectory(title="Select folder with JPG frames")
        if not d:
            return
        self.frames_dir = d
        self._refresh_frames()

    def pick_results_dir(self):
        d = filedialog.askdirectory(title="Select results folder")
        if not d:
            return
        self.results_dir = d
        self._refresh_frames()

    def _refresh_frames(self):
        self.lbl_paths.config(text=f"frames_dir: {self.frames_dir or '-'} | results_dir: {self.results_dir or '-'}")
        if not self.frames_dir or not os.path.isdir(self.frames_dir):
            return

        files = [f for f in os.listdir(self.frames_dir) if f.lower().endswith(".jpg")]
        files.sort()
        self.frames = files

        self.lst_frames.delete(0, tk.END)
        for f in self.frames:
            self.lst_frames.insert(tk.END, f)

        if self.frames:
            self.frame_idx = 0
            self.lst_frames.selection_set(0)
            self.load_frame_by_index(0)

    # ---------------- Frame navigation ----------------

    def _on_frame_select(self, event):
        sel = self.lst_frames.curselection()
        if not sel:
            return
        idx = int(sel[0])
        self.load_frame_by_index(idx)

    def prev_frame(self):
        if not self.frames:
            return
        self.load_frame_by_index(max(0, self.frame_idx - 1))

    def next_frame(self):
        if not self.frames:
            return
        self.load_frame_by_index(min(len(self.frames) - 1, self.frame_idx + 1))

    def load_frame_by_index(self, idx: int):
        if not self.results_dir:
            messagebox.showwarning("No results dir", "Select results folder first.")
            return
        if idx < 0 or idx >= len(self.frames):
            return

        self.frame_idx = idx
        fname = self.frames[idx]
        frame_path = os.path.join(self.frames_dir, fname)
        stem = os.path.splitext(fname)[0]
        out_dir = os.path.join(self.results_dir, stem)

        ctx = FrameContext(
            frame_path=frame_path,
            stem=stem,
            out_dir=out_dir,
            zone_path=os.path.join(out_dir, f"{stem}_zone.json"),
            plane_path=os.path.join(out_dir, f"{stem}_plane.json"),
            zone3d_path=os.path.join(out_dir, f"{stem}_zone3d.json"),
        )
        self.ctx = ctx

        # Load image
        img = Image.open(frame_path).convert("RGB")
        self.canvas.set_image(img)

        # Load jsons if exist
        self.zone = self._load_group_polys_2d(ctx.zone_path)
        self.planes = self._load_group_polys_2d(ctx.plane_path)
        self.zone3d = self._load_group_polys_3d(ctx.zone3d_path)

        # If zone3d missing, initialize from zone with p=None (-1), later bound
        if not self.zone3d:
            self.zone3d = self._zone_to_zone3d_unbound(self.zone)

        # If exactly one plane exists, auto-assign p=that plane id for all zone3d points
        self._auto_assign_if_single_plane()

        self.lbl_frame.config(text=f"Frame: {fname} ({idx+1}/{len(self.frames)})")
        self.status.config(text="")

        # Reset selection
        self.cur_group = None
        self.cur_poly_idx = 0
        self.cur_vertex_idx = None
        self.rebuild_tree()
        self._update_ui_state()
        self.redraw_overlay()

    # ---------------- JSON IO ----------------

    def _load_group_polys_2d(self, path: str) -> Dict[str, List[List[List[float]]]]:
        """Load {group_id: [closed_poly, ...]} and convert to open polygons internally."""
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out = {}
            for gid, polys in data.items():
                out[gid] = []
                for poly in polys:
                    out[gid].append(open_polygon(poly))
            return out
        except Exception as e:
            messagebox.showerror("JSON load error", f"Failed to load {path}\n{e}")
            return {}

    def _load_group_polys_3d(self, path: str) -> Dict[str, List[List[List[float]]]]:
        """Load {group_id: [closed_poly3d, ...]} and convert to open polygons internally."""
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out = {}
            for gid, polys in data.items():
                out[gid] = []
                for poly in polys:
                    poly_open = open_polygon(poly)
                    out[gid].append(poly_open)
            return out
        except Exception as e:
            messagebox.showerror("JSON load error", f"Failed to load {path}\n{e}")
            return {}

    def _save_group_polys_2d(self, path: str, data: Dict[str, List[List[List[float]]]]) -> None:
        """Save open polygons as closed polygons to JSON."""
        out = {}
        for gid, polys in data.items():
            out[gid] = []
            for poly_open in polys:
                out[gid].append(close_polygon(poly_open))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def _save_group_polys_3d(self, path: str, data: Dict[str, List[List[List[float]]]]) -> None:
        out = {}
        for gid, polys in data.items():
            out[gid] = []
            for poly_open in polys:
                out[gid].append(close_polygon(poly_open))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    # ---------------- Save output ----------------

    def save_all(self):
        if not self.ctx:
            return

        ensure_dir(self.ctx.out_dir)

        # Copy frame into results folder
        dst_frame = os.path.join(self.ctx.out_dir, os.path.basename(self.ctx.frame_path))
        if not os.path.exists(dst_frame):
            shutil.copy2(self.ctx.frame_path, dst_frame)

        # Ensure zone3d exists and (if single plane) auto-assign
        self._auto_assign_if_single_plane()

        # Save zone/plane/zone3d
        try:
            self._save_group_polys_2d(self.ctx.zone_path, self.zone)
            self._save_group_polys_2d(self.ctx.plane_path, self.planes)
            self._save_group_polys_3d(self.ctx.zone3d_path, self.zone3d)
        except Exception as e:
            messagebox.s