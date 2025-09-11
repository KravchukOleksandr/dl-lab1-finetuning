import os, shutil, torch
import timm
from timm.models.hub import download_cached_file
from ultralytics.utils.downloads import attempt_download_asset

def save_timm_model(model_name: str, dst_dir: str):
    """Скачивает timm-веса и сохраняет их в dst_dir"""
    cfg = timm.models.get_pretrained_cfg(model_name)
    if not cfg.url:
        print(f"[timm] {model_name}: нет pretrained URL")
        return
    cache_path = download_cached_file(cfg.url, check_hash=True)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, os.path.basename(cache_path))
    shutil.copy2(cache_path, dst_path)
    print(f"[timm] {model_name} сохранена: {dst_path}")

def save_yolo_model(model_name: str, dst_dir: str):
    """Скачивает ultralytics-веса и сохраняет их в dst_dir"""
    os.makedirs(dst_dir, exist_ok=True)
    # ultralytics >8.2.0: все веса доступны через attempt_download_asset
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}.pt"
    local_path = attempt_download_asset(url)
    dst_path = os.path.join(dst_dir, os.path.basename(local_path))
    shutil.copy2(local_path, dst_path)
    print(f"[yolo] {model_name} сохранена: {dst_path}")

if __name__ == "__main__":
    # укажите папку назначения
    target_dir = "/home/you/models"

    # YOLOv8n-cls
    save_yolo_model("yolov8n-cls", target_dir)

    # ConvNeXtV2-Atto
    save_timm_model("convnextv2_atto", target_dir)

    # MobileNetV4-Conv-Small
    save_timm_model("mobilenetv4_conv_small", target_dir)