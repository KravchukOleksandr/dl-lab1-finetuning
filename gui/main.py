import os
import tkinter as tk
from tkinter import filedialog

from .config import load_config
from .blob_io import AzureBlobIO
from .gui import LabelGUI, Paths

def main():
    cfg = load_config()  # читает ../configs.yaml

    # choose work_dir
    root = tk.Tk()
    root.withdraw()
    work_dir = filedialog.askdirectory(title="Choose work_dir (project folder)")
    root.destroy()
    if not work_dir:
        return

    paths = Paths(
        work_dir=work_dir,
        filtered_dir=os.path.join(work_dir, "filtered"),
        cache_boxes=os.path.join(work_dir, "cache", "boxes"),
        cache_frames=os.path.join(work_dir, "cache", "frames"),
        state_dir=os.path.join(work_dir, "_state"),
        convnext_dir=os.path.join(work_dir, "convnext_data"),
        yolo_dir=os.path.join(work_dir, "yolo_data"),
    )

    blobio = AzureBlobIO.from_connection_string(cfg.conn_str, cfg.container, cfg.filtered_prefix)

    app = LabelGUI(
        blobio=blobio,
        paths=paths,
        split_ratio=cfg.split_ratio,
        yolo_thr=cfg.yolo_thr,
        cnext_thr=cfg.cnext_thr,
        tolerance=cfg.tolerance,
    )
    app.geometry(f"{cfg.window_width}x{cfg.window_height}")
    app.mainloop()

if __name__ == "__main__":
    main()
