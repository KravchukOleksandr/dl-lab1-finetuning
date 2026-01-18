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
# COLORS / CONSTANTS
# =========================
PALETTE = [
    "#00E5FF", "#FF1744", "#76FF03", "#FFEA00", "#7C4DFF",
    "#FF9100", "#1DE9B6", "#F500FF", "#00C853", "#2979FF",
    "#FF5252", "#C6FF00", "#FFD600", "#651FFF", "#FF6D00",
]
UNBOUND_COLOR = "#B0B0B0"

MODE_ZONE = "ZONE"
MODE_PLANES = "PLANES"
MODE_BIND = "BIND"

EDIT_IDLE = "IDLE"
EDIT_DRAWING = "DRAWING"


# =========================
# HELPERS
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sort_numeric_str(keys: List[str]) -> List[str]:
    def kf(k: str):
        return (0, int(k)) if k.isdigit() else (1, k)
    return sorted(keys, key=kf)


def next_free_numeric_id(existing: List[str]) -> str:
    used = set(int(k) for k in existing if k.isdigit())
    i = 0
    while i in used:
        i += 1
    return str(i)


def open_polygon(poly_closed: List[List[float]]) -> List[List[float]]:
    # Internally store polygons open (no duplicated last point)
    if len(poly_closed) >= 2 and poly_closed[0] == poly_closed[-1]:
        return poly_closed[:-1]
    return poly_closed[:]


def close_polygon(poly_open: List[List[float]]) -> List[List[float]]:
    # On save, always close if polygon has >=3 points
    if len(poly_open) >= 3:
        return poly_open + [poly_open[0]]
    return poly_open[:]


def load_json_2d(path: str) -> Dict[str, List[List[List[float]]]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[List[List[float]]]] = {}
    for gid, polys in data.items():
        out[gid] = [open_polygon(poly) for poly in polys]
    return out


def load_json_3d(path: str) -> Dict[str, List[List[List[float]]]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[List[List[float]]]] = {}
    for gid, polys in data.items():
        out[gid] = [open_polygon(poly) for poly in polys]
    return out


def save_json_2d(path: str, data: Dict[str, List[List[List[float]]]]) -> None:
    out = {}
    for gid, polys in data.items():
        out[gid] = []
        for poly_open in polys:
            if len(poly_open) >= 3:
                out[gid].append(close_polygon(poly_open))
            # skip draft polygons (<3 pts)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def save_json_3d(path: str, data: Dict[str, List[List[List[float]]]]) -> None:
    out = {}
    for gid, polys in data.items():
        out[gid] = []
        for poly_open in polys:
            if len(poly_open) >= 3:
                out[gid].append(close_polygon(poly_open))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# =========================
# ZOOM/PAN IMAGE CANVAS
# =========================
class ZoomPanCanvas(ttk.Frame):
    """
    Image viewer with zoom/pan.
    Overlay points are stored in normalized [0..1] and mapped to canvas coordinates.
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
        self.geometry("1500x880")

        # Validate folders
        if not os.path.isdir(FRAMES_DIR):
            messagebox.showerror("Error", f"FRAMES_DIR not found:\n{FRAMES_DIR}")
            self.destroy()
            return
        ensure_dir(RESULTS_DIR)

        self.frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(".jpg")])
        if not self.frames:
            messagebox.showerror("Error", f"No .jpg frames in:\n{FRAMES_DIR}")
            self.destroy()
            return

        # Data (open polygons internally)
        self.zone: Dict[str, List[List[List[float]]]] = {}
        self.planes: Dict[str, List[List[List[float]]]] = {}
        self.zone3d: Dict[str, List[List[List[float]]]] = {}

        # State
        self.mode = MODE_ZONE
        self.edit_state = EDIT_IDLE
        self.cur_group: Optional[str] = None
        self.cur_poly_idx: int = 0

        # Vertex selection & dragging
        self.selected_vertex: Optional[int] = None
        self._dragging_vertex = False

        # Current frame context
        self.frame_idx = 0
        self.frame_name = ""
        self.stem = ""
        self.out_dir = ""
        self.zone_path = ""
        self.plane_path = ""
        self.zone3d_path = ""
        self.out_frame_path = ""

        # UI
        self._build_ui()
        self._bind_hotkeys()

        # Load first frame
        self._set_frame_by_index(0)

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Label(top, text="Frame:").pack(side="left", padx=(0, 6))
        self.cmb_frame = ttk.Combobox(top, values=self.frames, state="readonly", width=48)
        self.cmb_frame.pack(side="left")
        self.cmb_frame.bind("<<ComboboxSelected>>", self._on_frame_changed)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(top, text="Mode:").pack(side="left", padx=(0, 6))
        ttk.Button(top, text="ZONE", command=lambda: self.set_mode(MODE_ZONE)).pack(side="left", padx=2)
        ttk.Button(top, text="PLANES", command=lambda: self.set_mode(MODE_PLANES)).pack(side="left", padx=2)
        ttk.Button(top, text="BIND", command=lambda: self.set_mode(MODE_BIND)).pack(side="left", padx=2)

        ttk.Button(top, text="Save (Ctrl+S)", command=self.save_all).pack(side="left", padx=10)

        self.lbl_status = ttk.Label(top, text="", foreground="orange")
        self.lbl_status.pack(side="right")

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=6)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 10))

        # Groups
        self.lbl_groups = ttk.Label(left, text="Groups")
        self.lbl_groups.pack(anchor="w")
        self.lst_groups = tk.Listbox(left, width=28, height=12)
        self.lst_groups.pack(fill="y")
        self.lst_groups.bind("<<ListboxSelect>>", self._on_group_selected)

        # Polygons
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(left, text="Polygons").pack(anchor="w")
        self.lst_polys = tk.Listbox(left, width=28, height=10)
        self.lst_polys.pack(fill="y")
        self.lst_polys.bind("<<ListboxSelect>>", self._on_poly_selected)

        # Buttons
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        btns1 = ttk.Frame(left)
        btns1.pack(fill="x", pady=2)
        ttk.Button(btns1, text="New group", command=self.new_group).pack(side="left", padx=2)
        ttk.Button(btns1, text="Delete group", command=self.delete_group).pack(side="left", padx=2)

        btns2 = ttk.Frame(left)
        btns2.pack(fill="x", pady=2)
        ttk.Button(btns2, text="New polygon", command=self.new_polygon).pack(side="left", padx=2)
        ttk.Button(btns2, text="Delete polygon", command=self.delete_polygon).pack(side="left", padx=2)

        btns3 = ttk.Frame(left)
        btns3.pack(fill="x", pady=2)
        ttk.Button(btns3, text="Finish polygon (Enter)", command=self.finish_polygon).pack(side="left", padx=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(left, text="Legend").pack(anchor="w")
        self.txt_legend = tk.Text(left, width=28, height=14, wrap="none")
        self.txt_legend.pack(fill="both", expand=True)
        self.txt_legend.configure(state="disabled")

        # Right: image viewer
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        self.viewer = ZoomPanCanvas(right)
        self.viewer.pack(fill="both", expand=True)

        # Click & drag for vertex move
        self.viewer.canvas.bind("<Button-1>", self._on_canvas_click)
        self.viewer.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.viewer.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

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

        ttk.Label(
            bottom,
            text="Tips: DRAWING adds points. IDLE lets you drag vertices. BIND: click vertex then 0..4, N=next unbound. Zoom=wheel, Pan=RMB drag."
        ).pack(side="right", padx=8)

    def _bind_hotkeys(self):
        self.bind("<Control-s>", lambda e: self.save_all())
        self.bind("<Return>", lambda e: self.finish_polygon())
        self.bind("<Left>", lambda e: self._set_frame_by_index(self.frame_idx - 1))
        self.bind("<Right>", lambda e: self._set_frame_by_index(self.frame_idx + 1))
        self.bind("n", lambda e: self.next_unbound_vertex())

        for k in "01234":
            self.bind(k, lambda e, kk=k: self.assign_plane(int(kk)))

    # ---------------- Frame load/save ----------------
    def _frame_paths(self, stem: str, frame_name: str) -> Tuple[str, str, str, str, str]:
        out_dir = os.path.join(RESULTS_DIR, stem)
        zone_path = os.path.join(out_dir, f"{stem}_zone.json")
        plane_path = os.path.join(out_dir, f"{stem}_plane.json")
        zone3d_path = os.path.join(out_dir, f"{stem}_zone3d.json")
        out_frame_path = os.path.join(out_dir, frame_name)
        return out_dir, out_frame_path, zone_path, plane_path, zone3d_path

    def _set_frame_by_index(self, idx: int):
        if idx < 0:
            idx = 0
        if idx >= len(self.frames):
            idx = len(self.frames) - 1
        self.frame_idx = idx
        self.frame_name = self.frames[idx]
        self.stem = os.path.splitext(self.frame_name)[0]

        self.out_dir, self.out_frame_path, self.zone_path, self.plane_path, self.zone3d_path = self._frame_paths(self.stem, self.frame_name)

        # Set combobox selection
        self.cmb_frame.set(self.frame_name)

        # Load image
        frame_path = os.path.join(FRAMES_DIR, self.frame_name)
        img = Image.open(frame_path).convert("RGB")
        self.viewer.set_image(img)

        # Load json data
        try:
            self.zone = load_json_2d(self.zone_path)
            self.planes = load_json_2d(self.plane_path)
            self.zone3d = load_json_3d(self.zone3d_path)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self.zone = {}
            self.planes = {}
            self.zone3d = {}

        # Init zone3d if missing
        if not self.zone3d:
            self.zone3d = self._init_zone3d_from_zone()

        # Auto assign p if single plane "0"
        self._auto_assign_if_single_plane()

        # Reset selection
        self.cur_group = None
        self.cur_poly_idx = 0
        self.selected_vertex = None
        self._dragging_vertex = False
        self.edit_state = EDIT_IDLE

        # Refresh UI lists
        self._refresh_group_list()
        self._refresh_poly_list()
        self._refresh_legend()

        self.lbl_status.config(text="")
        self.redraw()

    def _on_frame_changed(self, event):
        val = self.cmb_frame.get()
        if val in self.frames:
            self._set_frame_by_index(self.frames.index(val))

    def save_all(self):
        ensure_dir(self.out_dir)

        # Copy frame if not present
        if not os.path.exists(self.out_frame_path):
            shutil.copy2(os.path.join(FRAMES_DIR, self.frame_name), self.out_frame_path)

        # Keep zone3d synced and auto-assign if single plane
        if not self.zone3d:
            self.zone3d = self._init_zone3d_from_zone()
        self._sync_zone3d_full()
        self._auto_assign_if_single_plane()

        try:
            # Save (draft polygons <3 points are skipped)
            save_json_2d(self.zone_path, self.zone)
            save_json_2d(self.plane_path, self.planes)
            save_json_3d(self.zone3d_path, self.zone3d)
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            return

        self.lbl_status.config(text="Saved ✔")
        self.redraw()

    # ---------------- Mode ----------------
    def set_mode(self, mode: str):
        self.mode = mode
        self.selected_vertex = None
        self._dragging_vertex = False
        self.edit_state = EDIT_IDLE

        self._refresh_group_list()
        self._refresh_poly_list()
        self._refresh_legend()
        self.lbl_status.config(text="")
        self.redraw()

    # ---------------- Lists ----------------
    def _active_groups_dict(self) -> Dict[str, List[List[List[float]]]]:
        if self.mode == MODE_PLANES:
            return self.planes
        return self.zone  # ZONE and BIND operate on zone groups/polys

    def _refresh_group_list(self):
        self.lst_groups.delete(0, tk.END)

        if self.mode == MODE_PLANES:
            self.lbl_groups.config(text="Planes (0..N-1)")
            keys = sort_numeric_str(list(self.planes.keys()))
        else:
            self.lbl_groups.config(text="Subzones (string ids)")
            keys = sort_numeric_str(list(self.zone.keys()))

        for k in keys:
            self.lst_groups.insert(tk.END, k)

        if keys:
            self.cur_group = keys[0]
            self.lst_groups.selection_set(0)
        else:
            self.cur_group = None

    def _refresh_poly_list(self):
        self.lst_polys.delete(0, tk.END)
        self.cur_poly_idx = 0

        if not self.cur_group:
            return

        groups = self._active_groups_dict()
        polys = groups.get(self.cur_group, [])
        for i in range(len(polys)):
            self.lst_polys.insert(tk.END, f"polygon_{i}")

        if polys:
            self.cur_poly_idx = 0
            self.lst_polys.selection_set(0)

    def _on_group_selected(self, event):
        sel = self.lst_groups.curselection()
        if not sel:
            return
        self.cur_group = self.lst_groups.get(sel[0])
        self.cur_poly_idx = 0
        self.selected_vertex = None
        self.edit_state = EDIT_IDLE
        self._refresh_poly_list()
        self.redraw()

    def _on_poly_selected(self, event):
        sel = self.lst_polys.curselection()
        if not sel:
            return
        self.cur_poly_idx = int(sel[0])
        self.selected_vertex = None
        self.edit_state = EDIT_IDLE
        self.redraw()

    # ---------------- Buttons: group/polygon ----------------
    def new_group(self):
        if self.mode == MODE_PLANES:
            # planes: fixed next sequential id
            pid = next_free_numeric_id(list(self.planes.keys()))
            # show dialog but locked: just info
            messagebox.showinfo("New plane", f"Creating plane with id = {pid}")
            self.planes[pid] = []
            self.cur_group = pid
        else:
            # zones: allow custom id, default next free
            default_id = next_free_numeric_id(list(self.zone.keys()))
            gid = simpledialog.askstring("New subzone", f"Enter subzone id (numeric string). Default: {default_id}", initialvalue=default_id)
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
            self.zone3d.setdefault(gid, [])
            self.cur_group = gid

        self.selected_vertex = None
        self.edit_state = EDIT_IDLE
        self._refresh_group_list()
        # select current group in listbox
        self._select_group_in_list(self.cur_group)
        self._refresh_poly_list()
        self.redraw()

    def delete_group(self):
        if not self.cur_group:
            return

        if self.mode == MODE_PLANES:
            # Only allow deleting the last plane to keep ids stable
            keys = sort_numeric_str(list(self.planes.keys()))
            if not keys:
                return
            last = keys[-1]
            if self.cur_group != last:
                messagebox.showwarning("Not allowed", f"You can delete only the last plane ({last}).")
                return
            if not messagebox.askyesno("Delete plane", f"Delete plane {self.cur_group}?"):
                return
            del self.planes[self.cur_group]
        else:
            if not messagebox.askyesno("Delete subzone", f"Delete subzone {self.cur_group}?"):
                return
            if self.cur_group in self.zone:
                del self.zone[self.cur_group]
            if self.cur_group in self.zone3d:
                del self.zone3d[self.cur_group]

        self.cur_group = None
        self.cur_poly_idx = 0
        self.selected_vertex = None
        self.edit_state = EDIT_IDLE

        self._refresh_group_list()
        self._refresh_poly_list()
        self.redraw()

    def new_polygon(self):
        if not self.cur_group:
            messagebox.showwarning("No group", "Create/select a group first.")
            return

        if self.mode == MODE_PLANES:
            self.planes.setdefault(self.cur_group, []).append([])
        else:
            self.zone.setdefault(self.cur_group, []).append([])
            self.zone3d.setdefault(self.cur_group, []).append([])

        # select newest polygon and enter DRAWING state
        self._refresh_poly_list()
        groups = self._active_groups_dict()
        self.cur_poly_idx = max(0, len(groups.get(self.cur_group, [])) - 1)
        self._select_poly_in_list(self.cur_poly_idx)

        self.edit_state = EDIT_DRAWING
        self.selected_vertex = None
        self.lbl_status.config(text="DRAWING")
        self.redraw()

    def delete_polygon(self):
        if not self.cur_group:
            return
        groups = self._active_groups_dict()
        polys = groups.get(self.cur_group, [])
        if not polys:
            return
        if self.cur_poly_idx < 0 or self.cur_poly_idx >= len(polys):
            return

        if not messagebox.askyesno("Delete polygon", f"Delete polygon_{self.cur_poly_idx}?"):
            return

        polys.pop(self.cur_poly_idx)

        if self.mode != MODE_PLANES:
            # keep zone3d aligned
            z3 = self.zone3d.get(self.cur_group, [])
            if 0 <= self.cur_poly_idx < len(z3):
                z3.pop(self.cur_poly_idx)

        self.cur_poly_idx = max(0, self.cur_poly_idx - 1)
        self.edit_state = EDIT_IDLE
        self.selected_vertex = None

        self._refresh_poly_list()
        self._select_poly_in_list(self.cur_poly_idx)
        self.redraw()

    def finish_polygon(self):
        # Leave drawing mode
        self.edit_state = EDIT_IDLE
        self._dragging_vertex = False
        self.lbl_status.config(text="")
        self.redraw()

    def _select_group_in_list(self, gid: Optional[str]):
        if gid is None:
            return
        for i in range(self.lst_groups.size()):
            if self.lst_groups.get(i) == gid:
                self.lst_groups.selection_clear(0, tk.END)
                self.lst_groups.selection_set(i)
                self.lst_groups.see(i)
                return

    def _select_poly_in_list(self, idx: int):
        if idx < 0:
            return
        if idx >= self.lst_polys.size():
            return
        self.lst_polys.selection_clear(0, tk.END)
        self.lst_polys.selection_set(idx)
        self.lst_polys.see(idx)

    # ---------------- Drawing & editing ----------------
    def _get_active_poly2d(self) -> Optional[List[List[float]]]:
        groups = self._active_groups_dict()
        if not self.cur_group:
            return None
        polys = groups.get(self.cur_group, [])
        if not polys:
            return None
        if self.cur_poly_idx < 0 or self.cur_poly_idx >= len(polys):
            return None
        return polys[self.cur_poly_idx]

    def _on_canvas_click(self, event):
        x, y = self.viewer.canvas_to_norm(event.x, event.y)
        poly = self._get_active_poly2d()

        if self.mode == MODE_BIND:
            # select nearest vertex for binding
            self.selected_vertex = self._pick_nearest_vertex(event.x, event.y)
            self.lbl_status.config(text="")
            self.redraw()
            return

        # ZONE / PLANES
        if poly is None:
            return

        if self.edit_state == EDIT_DRAWING:
            # add vertex
            poly.append([x, y])
            if self.mode != MODE_PLANES:
                self._ensure_zone3d_sync_current()
                p = self._default_plane_for_new_point()
                self.zone3d[self.cur_group][self.cur_poly_idx].append([x, y, p])
            self.lbl_status.config(text="DRAWING")
            self.redraw()
            return

        # IDLE: pick vertex for dragging
        vidx = self._pick_nearest_vertex(event.x, event.y)
        self.selected_vertex = vidx
        self._dragging_vertex = (vidx is not None)
        self.lbl_status.config(text="")
        self.redraw()

    def _on_canvas_drag(self, event):
        if self.mode == MODE_BIND:
            return
        if self.edit_state != EDIT_IDLE:
            return
        if not self._dragging_vertex:
            return
        if self.selected_vertex is None:
            return

        poly = self._get_active_poly2d()
        if poly is None:
            return

        x, y = self.viewer.canvas_to_norm(event.x, event.y)
        if 0 <= self.selected_vertex < len(poly):
            poly[self.selected_vertex] = [x, y]

        # sync zone3d positions
        if self.mode != MODE_PLANES:
            self._ensure_zone3d_sync_current()
            poly3 = self.zone3d[self.cur_group][self.cur_poly_idx]
            if 0 <= self.selected_vertex < len(poly3):
                p = poly3[self.selected_vertex][2]
                poly3[self.selected_vertex] = [x, y, p]

        self.redraw()

    def _on_canvas_release(self, event):
        self._dragging_vertex = False

    def _pick_nearest_vertex(self, X: float, Y: float) -> Optional[int]:
        poly = self._get_active_poly2d()
        if poly is None or not poly:
            return None
        best_i = None
        best_d2 = 1e18
        r2 = 12.0 * 12.0
        for i, (x, y) in enumerate(poly):
            px, py = self.viewer.norm_to_canvas(x, y)
            d2 = (px - X) ** 2 + (py - Y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        if best_i is not None and best_d2 <= r2:
            return best_i
        return None

    # ---------------- Zone3D binding ----------------
    def _init_zone3d_from_zone(self) -> Dict[str, List[List[List[float]]]]:
        out: Dict[str, List[List[List[float]]]] = {}
        for gid, polys in self.zone.items():
            out[gid] = []
            for poly in polys:
                out[gid].append([[pt[0], pt[1], -1] for pt in poly])
        return out

    def _sync_zone3d_full(self):
        # Ensure zone3d structure matches zone structure (by indices)
        for gid, polys2 in self.zone.items():
            self.zone3d.setdefault(gid, [])
            while len(self.zone3d[gid]) < len(polys2):
                self.zone3d[gid].append([])
            for pi, poly2 in enumerate(polys2):
                poly3 = self.zone3d[gid][pi]
                # shrink if needed
                if len(poly3) > len(poly2):
                    del poly3[len(poly2):]
                # extend if needed
                while len(poly3) < len(poly2):
                    p = self._default_plane_for_new_point()
                    poly3.append([poly2[len(poly3)][0], poly2[len(poly3)][1], p])
                # sync x,y
                for vi in range(len(poly2)):
                    poly3[vi][0] = poly2[vi][0]
                    poly3[vi][1] = poly2[vi][1]

        # Remove zone3d groups that no longer exist in zone
        for gid in list(self.zone3d.keys()):
            if gid not in self.zone:
                del self.zone3d[gid]

    def _ensure_zone3d_sync_current(self):
        if self.cur_group is None:
            return
        self.zone3d.setdefault(self.cur_group, [])
        z2 = self.zone.get(self.cur_group, [])
        while len(self.zone3d[self.cur_group]) < len(z2):
            self.zone3d[self.cur_group].append([])
        # ensure current polygon exists and aligned in length
        if self.cur_poly_idx >= len(self.zone3d[self.cur_group]):
            return
        poly2 = z2[self.cur_poly_idx]
        poly3 = self.zone3d[self.cur_group][self.cur_poly_idx]
        # shrink
        if len(poly3) > len(poly2):
            del poly3[len(poly2):]
        # extend
        while len(poly3) < len(poly2):
            p = self._default_plane_for_new_point()
            poly3.append([poly2[len(poly3)][0], poly2[len(poly3)][1], p])

    def _auto_assign_if_single_plane(self):
        # If there is exactly one plane "0", auto-assign p=0 for all points
        keys = sort_numeric_str(list(self.planes.keys()))
        if len(keys) == 1 and keys[0] == "0":
            for gid, polys in self.zone3d.items():
                for poly in polys:
                    for pt in poly:
                        if len(pt) == 3:
                            pt[2] = 0

    def _default_plane_for_new_point(self) -> int:
        keys = sort_numeric_str(list(self.planes.keys()))
        if len(keys) == 1 and keys[0] == "0":
            return 0
        return -1

    def assign_plane(self, p: int):
        if self.mode != MODE_BIND:
            return
        if self.selected_vertex is None:
            return
        if self.cur_group is None:
            return
        self._sync_zone3d_full()
        if self.cur_group not in self.zone3d:
            return
        if self.cur_poly_idx < 0 or self.cur_poly_idx >= len(self.zone3d[self.cur_group]):
            return
        poly3 = self.zone3d[self.cur_group][self.cur_poly_idx]
        if 0 <= self.selected_vertex < len(poly3):
            poly3[self.selected_vertex][2] = int(p)
        self.redraw()

    def next_unbound_vertex(self):
        if self.mode != MODE_BIND:
            return
        if self.cur_group is None:
            return
        self._sync_zone3d_full()
        if self.cur_group not in self.zone3d:
            return
        if self.cur_poly_idx < 0 or self.cur_poly_idx >= len(self.zone3d[self.cur_group]):
            return
        poly3 = self.zone3d[self.cur_group][self.cur_poly_idx]
        if not poly3:
            return
        start = (self.selected_vertex + 1) if self.selected_vertex is not None else 0
        n = len(poly3)
        for step in range(n):
            i = (start + step) % n
            if poly3[i][2] == -1:
                self.selected_vertex = i
                self.redraw()
                return
        messagebox.showinfo("Bind", "No unbound vertices in this polygon.")

    # ---------------- Rendering ----------------
    def _color_zone_group(self, gid: str) -> str:
        keys = sort_numeric_str(list(self.zone.keys()))
        idx = keys.index(gid) if gid in keys else 0
        return PALETTE[idx % len(PALETTE)]

    def _color_plane(self, pid: str) -> str:
        idx = int(pid) if pid.isdigit() else 0
        return PALETTE[(idx + 5) % len(PALETTE)]

    def _refresh_legend(self):
        zone_keys = sort_numeric_str(list(self.zone.keys()))
        plane_keys = sort_numeric_str(list(self.planes.keys()))
        lines = []
        if zone_keys:
            lines.append("ZONE subzones (dashed):")
            for k in zone_keys[:10]:
                lines.append(f"  {k}: {self._color_zone_group(k)}")
            if len(zone_keys) > 10:
                lines.append("  ...")
        if plane_keys:
            lines.append("")
            lines.append("PLANES (solid):")
            for k in plane_keys[:10]:
                lines.append(f"  {k}: {self._color_plane(k)}")
            if len(plane_keys) > 10:
                lines.append("  ...")
        self.txt_legend.configure(state="normal")
        self.txt_legend.delete("1.0", tk.END)
        self.txt_legend.insert("1.0", "\n".join(lines))
        self.txt_legend.configure(state="disabled")

    def redraw(self):
        self.viewer.redraw()
        c = self.viewer.canvas

        # Bind bar enabled only in BIND
        st = "normal" if self.mode == MODE_BIND else "disabled"
        for b in self.plane_btns:
            b.config(state=st)

        # Draw planes in PLANES and BIND
        if self.mode in (MODE_PLANES, MODE_BIND):
            for pid in sort_numeric_str(list(self.planes.keys())):
                col = self._color_plane(pid)
                for poly in self.planes[pid]:
                    self._draw_poly(poly, col, width=2, dashed=False, draw_vertices=False)

        # Draw zone always
        for gid in sort_numeric_str(list(self.zone.keys())):
            col = self._color_zone_group(gid)
            polys = self.zone[gid]
            for pi, poly in enumerate(polys):
                active = (gid == self.cur_group and pi == self.cur_poly_idx and self.mode != MODE_PLANES)
                w = 4 if active else 3
                draw_v = active and (self.mode == MODE_ZONE)  # show editable vertices in ZONE only
                self._draw_poly(poly, col, width=w, dashed=True, draw_vertices=draw_v)

        # Draw BIND vertices (colored by plane assignment)
        if self.mode == MODE_BIND:
            self._sync_zone3d_full()
            self._draw_bind_vertices()

    def _draw_poly(self, poly_open: List[List[float]], color: str, width: int, dashed: bool, draw_vertices: bool):
        if not poly_open:
            return
        dash = (6, 3) if dashed else None

        # poly lines
        if len(poly_open) >= 2:
            pts = []
            for x, y in poly_open:
                X, Y = self.viewer.norm_to_canvas(x, y)
                pts.extend([X, Y])
            self.viewer.canvas.create_line(*pts, fill=color, width=width, dash=dash)

            # closing segment (visual)
            if len(poly_open) >= 3:
                x0, y0 = poly_open[0]
                xN, yN = poly_open[-1]
                X0, Y0 = self.viewer.norm_to_canvas(x0, y0)
                XN, YN = self.viewer.norm_to_canvas(xN, yN)
                self.viewer.canvas.create_line(XN, YN, X0, Y0, fill=color, width=width, dash=dash)

        if draw_vertices:
            for i, (x, y) in enumerate(poly_open):
                X, Y = self.viewer.norm_to_canvas(x, y)
                outline = "white" if (self.selected_vertex == i and self.edit_state == EDIT_IDLE) else "black"
                self._circle(X, Y, 6, fill=color, outline=outline)

        # Status
        if self.edit_state == EDIT_DRAWING:
            self.lbl_status.config(text="DRAWING")
        else:
            if self.lbl_status.cget("text") == "DRAWING":
                self.lbl_status.config(text="")

    def _draw_bind_vertices(self):
        if self.cur_group is None:
            return
        if self.cur_group not in self.zone3d:
            return
        if self.cur_poly_idx < 0 or self.cur_poly_idx >= len(self.zone3d[self.cur_group]):
            return
        pts3 = self.zone3d[self.cur_group][self.cur_poly_idx]
        if not pts3:
            return

        # unbound count
        unbound = sum(1 for pt in pts3 if pt[2] == -1)
        if len(self.planes.keys()) == 1 and "0" in self.planes:
            self.lbl_status.config(text="Only one plane → auto p=0")
        else:
            self.lbl_status.config(text=f"BIND: unbound={unbound}")

        for i, (x, y, p) in enumerate(pts3):
            X, Y = self.viewer.norm_to_canvas(x, y)
            if p == -1:
                fill = UNBOUND_COLOR
            else:
                fill = self._color_plane(str(int(p)))
            outline = "white" if self.selected_vertex == i else "black"
            self._circle(X, Y, 8, fill=fill, outline=outline)

    def _circle(self, X: float, Y: float, r: float, fill: str, outline: str):
        self.viewer.canvas.create_oval(X - r, Y - r, X + r, Y + r, fill=fill, outline=outline, width=2)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
