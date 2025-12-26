import os
import shutil
from datetime import datetime

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def auto_prefix_ddmmyyyy() -> str:
    return datetime.now().strftime("auto%d%m%Y-")

def base_name(blob_path: str) -> str:
    return os.path.basename(blob_path)

def export_convnext_crop(
    cache_box_path: str,
    out_convnext_dir: str,
    split: str,
    cls: str,   # "TP" or "FP"
    original_box_blob: str
):
    dst_dir = os.path.join(out_convnext_dir, split, cls)
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, auto_prefix_ddmmyyyy() + base_name(original_box_blob))
    if not os.path.exists(dst):
        shutil.copy2(cache_box_path, dst)

def export_yolo_frame(
    cache_frame_path: str,
    out_yolo_dir: str,
    cam: str,
    split: str,
    original_frame_blob: str
):
    dst_dir = os.path.join(out_yolo_dir, f"{cam}-{split}")
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, auto_prefix_ddmmyyyy() + base_name(original_frame_blob))
    if not os.path.exists(dst):
        shutil.copy2(cache_frame_path, dst)

def should_export_on_tp(yolo_score: float, cnext_score: float, yolo_thr: float, cnext_thr: float, tol: float):
    """
    TP pressed:
    - YOLO: if yolo_score < yolo_thr + tol => export frame (FN + prophylaxis)
    - ConvNeXt: if cnext_score < cnext_thr + tol => export crop to TP (FN + prophylaxis)
    """
    export_frame = (yolo_score < (yolo_thr + tol))
    export_crop_tp = (cnext_score < (cnext_thr + tol))
    return export_frame, export_crop_tp

def should_export_on_fp(yolo_score: float, cnext_score: float, yolo_thr: float, cnext_thr: float, tol: float):
    """
    FP pressed:
    - YOLO: if yolo_score >= yolo_thr => export frame (hallucinations)
    - ConvNeXt: if cnext_score > cnext_thr - tol => export crop to FP (FP + prophylaxis)
    """
    export_frame = (yolo_score >= yolo_thr)
    export_crop_fp = (cnext_score > (cnext_thr - tol))
    return export_frame, export_crop_fp
