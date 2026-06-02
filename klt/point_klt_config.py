from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class PointKLTConfig:
    # Paths live near these scripts for now. Later you can move them to your project config.
    student_model_path: str = str(Path(__file__).resolve().parent / "student_mask_best.pt")

    # YOLO refresh policy
    yolo_interval_seconds: float = 3.0
    yolo_imgsz: int = 640
    yolo_conf: float = 0.25
    yolo_person_class_id: int = 0
    yolo_device: Optional[int | str] = None  # None = let Ultralytics decide
    yolo_half: bool = True

    # Student crop/mask settings
    box_scale: float = 1.25
    input_w: int = 64
    input_h: int = 128

    # Point selector settings
    mask_threshold: float = 0.80
    torso_y_min: float = 0.03
    torso_y_max: float = 0.60
    target_points: int = 5

    # KLT settings
    klt_win_size: Tuple[int, int] = (21, 21)
    klt_max_level: int = 3
    fb_max_error: float = 1.5

    # Motion consistency
    min_points_for_klt: int = 3
    motion_residual_min: float = 2.0
    motion_residual_rel: float = 0.04
    motion_residual_max: float = 8.0

    # Safe refill
    enable_safe_refill: bool = True
    refill_only_when_points_equal: int = 3
    min_points_inside_mask_for_refill: int = 3
    min_points_after_refill: int = 4
    min_distance_new_point: float = 5.0

    # YOLO re-association
    match_iou_min: float = 0.25
    match_center_norm_max: float = 0.55
    max_yolo_misses: int = 2

    # Debug/annotation
    draw_debug_points: bool = False
    draw_debug_status: bool = False
    log_every_n_frames: int = 300
