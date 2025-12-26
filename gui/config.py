from dataclasses import dataclass
import os
import yaml

@dataclass
class AppConfig:
    conn_str: str
    container: str
    filtered_prefix: str
    split_ratio: float
    yolo_thr: float
    cnext_thr: float
    tolerance: float
    window_width: int
    window_height: int

def load_config(config_path: str | None = None) -> AppConfig:
    # по умолчанию: configs.yaml в корне проекта
    if config_path is None:
        # app/config.py -> ../configs.yaml
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, ".."))
        config_path = os.path.join(root, "configs.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    azure = cfg.get("azure", {})
    splits = cfg.get("splits", {})
    thr = cfg.get("thresholds", {})
    ui = cfg.get("ui", {})

    conn_str = str(azure.get("connection_string", "")).strip()
    container = str(azure.get("container", "")).strip()
    filtered_prefix = str(azure.get("filtered_prefix", "meta/")).strip()

    if not conn_str or not container:
        raise RuntimeError("configs.yaml: azure.connection_string and azure.container are required.")

    if not filtered_prefix.endswith("/"):
        filtered_prefix += "/"

    split_ratio = float(splits.get("train_ratio", 0.8))
    yolo_thr = float(thr.get("yolo_thr", 0.25))
    cnext_thr = float(thr.get("cnext_thr", 0.5))
    tolerance = float(thr.get("tolerance", 0.1))

    window_width = int(ui.get("window_width", 1500))
    window_height = int(ui.get("window_height", 780))

    return AppConfig(
        conn_str=conn_str,
        container=container,
        filtered_prefix=filtered_prefix,
        split_ratio=split_ratio,
        yolo_thr=yolo_thr,
        cnext_thr=cnext_thr,
        tolerance=tolerance,
        window_width=window_width,
        window_height=window_height,
    )
