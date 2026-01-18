import os
import json
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageTk

# =========================
# CONFIG (EDIT THESE PATHS)
# =========================
FRAMES_DIR = r"./frames"     # folder with *.jpg
RESULTS_DIR = r"./results"   # output root


# =========================
# CONSTANTS / PALETTE
# =========================
PALETTE = [
    "#00E5FF", "#FF1744", "#76FF03", "#FFEA00", "#7C4DFF",
    "#FF9100", "#1DE9B6", "#F500FF", "#00C853", "#2979FF",
    "#FF5252", "#C6FF00", "#FFD600", "#651FFF", "#FF6D00",
]

UNBOUND_COLOR = "#B0B0B0"


# =========================
# JSON HELPERS
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sort_numeric_str(keys: List[str]) -> List[str]:
    def kf(k: str):
        return (0, int(k)) if k.isdigit() else (1, k)
    return sorted(keys, key=kf)


def is_closed(poly: List[List[float]]) -> bool:
    return len(poly) >= 2 and poly[0] == poly[-1]


def open_polygon(poly_closed: List[List[float]]) -> List[List[float]]:
    # Internally keep polygons "open" (no duplicated last point)
    if len(poly_closed) >= 2 and poly_closed[0] == poly_closed[-1]:
        return poly_closed[:-1]
    return poly_closed[:]


def close_polygon(poly_open: List[List[float]]) -> List[List[float]]:
    # On save, always close polygons
    if len(poly_open) >= 3:
        return poly_open + [poly_open[0]]
    return poly_open[:]


def next_free_numeric_id(existing: List[str]) -> str:
    used = set(int(k) for k in existing if k.isdigit())
    i = 0
    while i in used:
        i += 1
    return str(i)


# =========================
# ZOOM/PAN CANVAS
# =========================
class ZoomPanCanvas(ttk.Frame):
    """
    Image viewer with zoom/pan.
    Overlay points are stored in normalized [0..1], converted for drawing.
    """
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg="#1f1f1f", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.img_pil: Optional[Image.Image] = None
        self.img_tk: Optional[ImageTk.PhotoImage] = None

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._pan_start = None

        # mouse bindings
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<MouseWheel>", self._on_wheel)   # win/mac
        self.canvas.bind("<Button-4>", self._on_wheel)     # linux up
        self.canvas.bind("<Button-5>", self._on_wheel)     # linux down

        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)

    def set_image(self, img: Image.Image) -> None:
        self.img_pil = img
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.redraw()

    def _on_wheel(self, event):
        if self.img_pil is None:
            return
        if hasattr(event, "delta") and event.delta != 0:
            factor = 1.1 if event.delta > 0 else 0.9
        else:
            factor = 1.1 if event.num == 4 else 0.9
        self.scale = max(0.1, min(12.0, self.scale * factor))
        self.redraw()

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)

    def _on_pan_move(self, event):
        if not self._pan_start:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self.offset_x += dx
        self.offset_y += dy
        self._pan_start = (event.x, event.y)
        self.redraw()

    def _img_dims(self) -> Tuple[int, int]:
        if self.img_pil is None:
            return (1, 1)
        return self.img_pil.size

    def _canvas_center(self) -> Tuple[float, float]:
        return (self.canvas.winfo_width() / 2.0, self.canvas.winfo_height() / 2.0)

    def norm_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        W, H = self._img_dims()
        cx, cy = self._canvas_center()
        img_w = W * self.scale
        img_h = H * self.scale
        left = cx - img_w / 2.0 + self.offset_x
        top = cy - img_h / 2.0 + self.offset_y
        return (left + x * img_w, top + y * img_h)

    def canvas_to_norm(self, X: float, Y: float) -> Tuple[float, float]:
        W, H = self._img_dims()
        cx, cy = self._canvas_center()
        img_w = W * self.scale
        img_h = H * self.scale
        left = cx - img_w / 2.0 + self.offset_x
        top = cy - img_h / 2.0 + self.offset_y
        x = (X - left) / img_w
        y = (Y - top) / img_h
        return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

    def redraw(self) -> None:
        self.canvas.delete("all")
        if self.img_pil is None:
            return
        W, H = self.img_pil.size
        new_w = max(1, int(W * self.scale))
        new_h = max(1, int(H * self.scale))
        resized = self.img_pil.resize((new_w, new_h), Image.BILINEAR)
        self.img_tk = ImageTk.PhotoImage(resized)

        cx, cy = self._canvas_center()
        self.canvas.create_image(cx + self.offset_x, cy + self.offset_y, image=self.img_tk, anchor="center")


# =========================
# MAIN APP
# =========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Zone / Planes / Zone3D Bind Labeler")
        self.geometry("1400x860")

        # Load frames list
        if not os.path.isdir(FRAMES_DIR):
            messagebox.showerror("Error", f"FRAMES_DIR not found:\n{FRAMES_DIR}")
            self.destroy()
            return

        self.frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(".jpg")])
        if not self.frames:
            messagebox.showerror("Error", f"No .jpg frames in:\n{FRAMES_DIR}")
            self.destroy()
            return

        ensure_dir(RESULTS_DIR)

        # Data: open polygons internally
        self.zone: Dict[str, List[List[List[float]]]] = {}      # subzone -> [poly_open points [x,y], ...]
        self.planes: Dict[str, List[List[List[float]]]] = {}    # plane_id -> [poly_open]
        self.zone3d: Dict[str, List[List[List[float]]]] = {}    # subzone -> [poly_open points [x,y,p], ...]

        # Selection
        self.mode = "ZONE"  # ZONE | PLANES | BIND
        self.cur_group: Optional[str] = None   # subzone id or plane id
        self.cur_poly: int = 0
        self.selected_vertex: Optional[int] = None  # only used in BIND for current zone polygon

        # Current working polygon (when drawing)
        self.draw_poly_open: Optional[List[List[float]]] = None  # reference to current poly in self.zone/self.planes

        # Current frame
        self.frame_idx = 0
        self.cur_frame_name = ""
        self.cur_stem = ""
        self.cur_out_dir = ""

        # UI
        self._build_ui()
        self._bind_hotkeys()

        # Load initial frame
        self.load_frame(0)

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Label(top, text="Mode:").pack(side="left")
        ttk.Button(top, text="ZONE", command=lambda: self.set_mode("ZONE")).pack(side="left", padx=2)
        ttk.Button(top, text="PLANES", command=lambda: self.set_mode("PLANES")).pack(side="left", padx=2)
        ttk.Button(top, text="BIND", command=lambda: self.set_mode("BIND")).pack(side="left", padx=2)

        ttk.Button(top, text="Save (Ctrl+S)", command=self.save_all).pack(side="left", padx=10)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Button(top, text="Prev (←)", command=lambda: self.load_frame(self.frame_idx - 1)).pack(side="left", padx=2)
        ttk.Button(top, text="Next (→)", command=lambda: self.load_frame(self.frame_idx + 1)).pack(side="left", padx=2)

        self.lbl_info = ttk.Label(top, text="")
        self.lbl_info.pack(side="left", padx=12)

        self.lbl_status = ttk.Label(top, text="", foreground="orange")
        self.lbl_status.pack(side="right")

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=6)

        # Left panel
        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 8))

        ttk.Label(left, text="Frames").pack(anchor="w")
        self.lst_frames = tk.Listbox(left, width=30, height=18)
        self.lst_frames.pack(fill="y")
        for f in self.frames:
            self.lst_frames.insert(tk.END, f)
        self.lst_frames.bind("<<ListboxSelect>>", self._on_frame_select)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        # Groups
        self.lbl_groups = ttk.Label(left, text="Groups")
        self.lbl_groups.pack(anchor="w")

        self.lst_groups = tk.Listbox(left, width=30, height=10)
        self.lst_groups.pack(fill="y")
        self.lst_groups.bind("<<ListboxSelect>>", self._on_group_select)

        grp_btns = ttk.Frame(left)
        grp_btns.pack(fill="x", pady=4)
        ttk.Button(grp_btns, text="Add group (auto)", command=self.add_group_auto).pack(side="left", padx=2)
        ttk.Button(grp_btns, text="Add group (custom)", command=self.add_group_custom).pack(side="left", padx=2)

        poly_btns = ttk.Frame(left)
        poly_btns.pack(fill="x", pady=4)
        ttk.Button(poly_btns, text="New polygon", command=self.new_polygon).pack(side="left", padx=2)
        ttk.Button(poly_btns, text="Del polygon", command=self.del_polygon).pack(side="left", padx=2)
        ttk.Button(poly_btns, text="Finish polygon (Enter)", command=self.finish_polygon).pack(side="left", padx=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(left, text="Legend").pack(anchor="w")
        self.txt_legend = tk.Text(left, width=30, height=10, wrap="none")
        self.txt_legend.pack(fill="both", expand=True)
        self.txt_legend.configure(state="disabled")

        # Right panel (canvas)
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        self.viewer = ZoomPanCanvas(right)
        self.viewer.pack(fill="both", expand=True)

        # Click on canvas
        self.viewer.canvas.bind("<Button-1>", self.on_canvas_click)

        # Bottom bind bar
        bottom = ttk.Frame(right)
        bottom.pack(fill="x", pady=6)

        self.bind_bar = ttk.Frame(bottom)
        self.bind_bar.pack(side="left")

        ttk.Label(self.bind_bar, text="Bind plane (0..4):").pack(side="left", padx=(0, 6))
        self.plane_btns: List[ttk.Button] = []
        for i in range(5):
            b = ttk.Button(self.bind_bar, text=str(i), command=lambda p=i: self.assign_plane(p))
            b.pack(side="left", padx=2)
            self.plane_btns.append(b)

        ttk.Label(bottom, text="Tips: Left click adds vertex (ZONE/PLANES). In BIND: click a vertex then press 0..4, N=next unbound. Zoom=wheel, Pan=RMB drag.").pack(side="right", padx=8)

    def _bind_hotkeys(self):
        self.bind("<Control-s>", lambda e: self.save_all())
        self.bind("<Left>", lambda e: self.load_frame(self.frame_idx - 1))
        self.bind("<Right>", lambda e: self.load_frame(self.frame_idx + 1))
        self.bind("<Return>", lambda e: self.finish_polygon())
        self.bind("n", lambda e: self.next_unbound_vertex())

        for k in "01234":
            self.bind(k, lambda e, kk=k: self.assign_plane(int(kk)))

    # ---------------- Frame loading / IO ----------------
    def _frame_paths(self, stem: str, frame_name: str) -> Tuple[str, str, str, str, str]:
        out_dir = os.path.join(RESULTS_DIR, stem)
        zone_path = os.path.join(out_dir, f"{stem}_zone.json")
        plane_path = os.path.join(out_dir, f"{stem}_plane.json")
        zone3d_path = os.path.join(out_dir, f"{stem}_zone3d.json")
        out_frame_path = os.path.join(out_dir, frame_name)  # copy of jpg
        return out_dir, out_frame_path, zone_path, plane_path, zone3d_path

    def load_frame(self, idx: int):
        if idx < 0:
            idx = 0
        if idx >= len(self.frames):
            idx = len(self.frames) - 1
        self.frame_idx = idx

        self.cur_frame_name = self.frames[idx]
        self.cur_stem = os.path.splitext(self.cur_frame_name)[0]

        self.cur_out_dir, self.cur_out_frame_path, self.cur_zone_path, self.cur_plane_path, self.cur_zone3d_path = \
            self._frame_paths(self.cur_stem, self.cur_frame_name)

        frame_path = os.path.join(FRAMES_DIR, self.cur_frame_name)
        img = Image.open(frame_path).convert("RGB")
        self.viewer.set_image(img)

        # Load JSONs (if exist)
        self.zone = self._load_2d(self.cur_zone_path)
        self.planes = self._load_2d(self.cur_plane_path)
        self.zone3d = self._load_3d(self.cur_zone3d_path)

        # Initialize zone3d from zone if empty
        if not self.zone3d:
            self.zone3d = self._init_zone3d_from_zone()

        # Auto-assign if single plane
        self._auto_assign_if_single_plane()

        # Reset selection
        self.selected_vertex = None
        self.cur_group = None
        self.cur_poly = 0
        self.draw_poly_open = None

        # Update lists and redraw
        self.lst_frames.selection_clear(0, tk.END)
        self.lst_frames.selection_set(idx)
        self.lst_frames.see(idx)

        self._refresh_groups_list()
        self._update_labels()
        self.redraw()

    def _load_2d(self, path: str) -> Dict[str, List[List[List[float]]]]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out: Dict[str, List[List[List[float]]]] = {}
            for gid, polys in data.items():
                out[gid] = [open_polygon(poly) for poly in polys]
            return out
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load:\n{path}\n\n{e}")
            return {}

    def _load_3d(self, path: str) -> Dict[str, List[List[List[float]]]]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out: Dict[str, List[List[List[float]]]] = {}
            for gid, polys in data.items():
                out[gid] = [open_polygon(poly) for poly in polys]
            return out
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load:\n{path}\n\n{e}")
            return {}

    def _save_2d(self, path: str, data: Dict[str, List[List[List[float]]]]) -> None:
        out = {}
        for gid, polys in data.items():
            out[gid] = [close_polygon(poly) for poly in polys]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def _save_3d(self, path: str, data: Dict[str, List[List[List[float]]]]) -> None:
        out = {}
        for gid, polys in data.items():
            out[gid] = [close_polygon(poly) for poly in polys]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def save_all(self):
        ensure_dir(self.cur_out_dir)

        # Copy frame as-is (original name, jpg)
        if not os.path.exists(self.cur_out_frame_path):
            src = os.path.join(FRAMES_DIR, self.cur_frame_name)
            shutil.copy2(src, self.cur_out_frame_path)

        # Ensure zone3d exists and auto-assign for single plane
        if not self.zone3d:
            self.zone3d = self._init_zone3d_from_zone()
        self._auto_assign_if_single_plane()

        try:
            self._save_2d(self.cur_zone_path, self.zone)
            self._save_2d(self.cur_plane_path, self.planes)
            self._save_3d(self.cur_zone3d_path, self.zone3d)
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            return

        self.lbl_status.config(text="Saved ✔")
        self.redraw()

    # ---------------- Mode / Groups / Polygons ----------------
    def set_mode(self, mode: str):
        self.mode = mode
        self.selected_vertex = None
        self.draw_poly_open = None
        self._refresh_groups_list()
        self._update_labels()
        self.redraw()

    def _refresh_groups_list(self):
        self.lst_groups.delete(0, tk.END)

        if self.mode == "ZONE" or self.mode == "BIND":
            keys = sort_numeric_str(list(self.zone.keys()))
            self.lbl_groups.config(text="Subzones (zone)")
        else:
            keys = sort_numeric_str(list(self.planes.keys()))
            self.lbl_groups.config(text="Planes")

        for k in keys:
            self.lst_groups.insert(tk.END, k)

        # Auto select first if exists
        if keys:
            self.lst_groups.selection_set(0)
            self.cur_group = keys[0]
            self.cur_poly = 0
        else:
            self.cur_group = None
            self.cur_poly = 0

        self._sync_draw_poly_ref()

    def _sync_draw_poly_ref(self):
        """Update self.draw_poly_open to point at current polygon list (or None)."""
        self.draw_poly_open = None
        if not self.cur_group:
            return
        if self.mode == "PLANES":
            grp = self.planes.get(self.cur_group, [])
        else:
            grp = self.zone.get(self.cur_group, [])
        if not grp:
            return
        if self.cur_poly < 0:
            self.cur_poly = 0
        if self.cur_poly >= len(grp):
            self.cur_poly = len(grp) - 1
        self.draw_poly_open = grp[self.cur_poly]

    def add_group_auto(self):
        if self.mode == "PLANES":
            gid = next_free_numeric_id(list(self.planes.keys()))
            self.planes[gid] = []
        else:
            gid = next_free_numeric_id(list(self.zone.keys()))
            self.zone[gid] = []
            if gid not in self.zone3d:
                self.zone3d[gid] = []
        self.cur_group = gid
        self.cur_poly = 0
        self._refresh_groups_list()
        self.redraw()

    def add_group_custom(self):
        if self.mode == "PLANES":
            messagebox.showinfo("Not allowed", "Custom ids are disabled for planes (planes are 0..N-1).")
            return
        gid = simpledialog.askstring("Custom subzone id", "Enter subzone id (numeric string), e.g. 5:")
        if not gid:
            return
        gid = gid.strip()
        if not gid.isdigit():
            messagebox.showerror("Invalid", "Subzone id must be a numeric string, e.g. '5'.")
            return
        if gid in self.zone:
            messagebox.showerror("Exists", f"Subzone '{gid}' already exists.")
            return
        self.zone[gid] = []
        if gid not in self.zone3d:
            self.zone3d[gid] = []
        self.cur_group = gid
        self.cur_poly = 0
        self._refresh_groups_list()
        self.redraw()

    def new_polygon(self):
        if not self.cur_group:
            messagebox.showwarning("No group", "Create/select a group first.")
            return
        if self.mode == "PLANES":
            self.planes.setdefault(self.cur_group, []).append([])
        else:
            self.zone.setdefault(self.cur_group, []).append([])
            self.zone3d.setdefault(self.cur_group, []).append([])
        # select newest polygon
        self.cur_poly = self._current_group_poly_count() - 1
        self._sync_draw_poly_ref()
        self.redraw()

    def del_polygon(self):
        if not self.cur_group:
            return
        if self.mode == "PLANES":
            grp = self.planes.get(self.cur_group, [])
            if not grp:
                return
            if 0 <= self.cur_poly < len(grp):
                grp.pop(self.cur_poly)
        else:
            grp = self.zone.get(self.cur_group, [])
            if not grp:
                return
     