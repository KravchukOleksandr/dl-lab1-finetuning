from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn


# =========================
# MODEL
# =========================

class GRN(nn.Module):
    """
    Global Response Normalization, ConvNeXtV2-style.
    Работает в NCHW.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()

        padding = kernel // 2

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MicroNeXtBlock(nn.Module):
    """
    Маленький ConvNeXtV2-inspired блок:
    depthwise spatial mixing + pointwise expansion + GELU + GRN + projection.
    """
    def __init__(self, channels, kernel=7, expansion=2):
        super().__init__()

        hidden = channels * expansion
        padding = kernel // 2

        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)

        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.grn = GRN(hidden)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        shortcut = x

        x = self.dw(x)
        x = self.bn(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)

        return x + shortcut


class MicroNeXtMaskNet(nn.Module):
    """
    Input:  B×3×128×64
    Output: B×1×32×16 logits
    """
    def __init__(self):
        super().__init__()

        self.stem = ConvBNAct(3, 16, kernel=3, stride=2)   # 128×64 -> 64×32

        self.stage1 = nn.Sequential(
            MicroNeXtBlock(16, kernel=7, expansion=2),
        )

        self.down = ConvBNAct(16, 24, kernel=3, stride=2)   # 64×32 -> 32×16

        self.stage2 = nn.Sequential(
            MicroNeXtBlock(24, kernel=7, expansion=2),
            MicroNeXtBlock(24, kernel=7, expansion=2),
            MicroNeXtBlock(24, kernel=7, expansion=2),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )

        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        x = self.proj(x)
        x = self.head(x)
        return x


# =========================
# POINT SELECTOR
# =========================

@dataclass
class PointSelectorConfig:
    mask_threshold: float = 0.35

    torso_y_min: float = 0.25
    torso_y_max: float = 0.70

    fat_row_ratio: float = 0.75

    tri_top_y_frac: float = 0.18
    tri_low_y_frac: float = 0.16
    tri_side_x_frac: float = 0.22

    micro_square_frac: float = 0.28
    min_micro_half: int = 3

    score_mask_power: float = 1.3
    min_triangle_area: float = 18.0
    min_point_distance: float = 4.0


def largest_component(binary_mask):
    mask_u8 = binary_mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if num_labels <= 1:
        return binary_mask.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1

    return labels == best_label


def triangle_area(points):
    p0 = np.asarray(points[0], dtype=np.float32)
    p1 = np.asarray(points[1], dtype=np.float32)
    p2 = np.asarray(points[2], dtype=np.float32)

    return float(abs(np.cross(p1 - p0, p2 - p0)) * 0.5)


def min_pairwise_distance(points):
    pts = [np.asarray(p, dtype=np.float32) for p in points]

    d01 = np.linalg.norm(pts[0] - pts[1])
    d02 = np.linalg.norm(pts[0] - pts[2])
    d12 = np.linalg.norm(pts[1] - pts[2])

    return float(min(d01, d02, d12))


def compute_cornerness(image_rgb):
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb

    gray = gray.astype(np.float32) / 255.0

    corner = cv2.cornerMinEigenVal(gray, blockSize=3, ksize=3)

    max_val = float(corner.max())
    if max_val > 1e-12:
        corner = corner / max_val

    return corner.astype(np.float32)


def find_body_anchor(mask, cfg):
    h, w = mask.shape

    y0 = int(round(cfg.torso_y_min * h))
    y1 = int(round(cfg.torso_y_max * h))
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))

    body_mask = np.zeros_like(mask, dtype=bool)
    body_mask[y0:y1] = mask[y0:y1]

    if body_mask.sum() == 0:
        return None

    row_widths = body_mask.sum(axis=1)
    max_width = int(row_widths.max())

    if max_width <= 1:
        return None

    fat_rows = np.where(row_widths >= cfg.fat_row_ratio * max_width)[0]

    if len(fat_rows) == 0:
        return None

    y_center = int(np.median(fat_rows))

    xs = np.where(body_mask[y_center])[0]

    if len(xs) == 0:
        y_center = int(fat_rows[np.argmax(row_widths[fat_rows])])
        xs = np.where(body_mask[y_center])[0]

    if len(xs) < 2:
        return None

    x_left = int(xs.min())
    x_right = int(xs.max())

    body_width = float(x_right - x_left)

    if body_width < 4:
        return None

    x_center = 0.5 * (x_left + x_right)

    return {
        "body_mask": body_mask,
        "x_center": x_center,
        "y_center": float(y_center),
        "body_width": body_width,
    }


def pick_point_in_micro_region(center, body_mask, score_map, body_width, cfg):
    h, w = body_mask.shape

    cx, cy = center

    half = max(cfg.min_micro_half, int(round(0.5 * cfg.micro_square_frac * body_width)))

    def try_pick(scale):
        hh = int(round(half * scale))

        x0 = max(0, int(round(cx - hh)))
        x1 = min(w, int(round(cx + hh + 1)))
        y0 = max(0, int(round(cy - hh)))
        y1 = min(h, int(round(cy + hh + 1)))

        if x1 <= x0 or y1 <= y0:
            return None

        region = np.zeros_like(body_mask, dtype=bool)
        region[y0:y1, x0:x1] = True

        candidate = region & body_mask

        if candidate.sum() == 0:
            return None

        local_score = score_map.copy()
        local_score[~candidate] = -1.0

        flat_idx = int(np.argmax(local_score))
        yy, xx = np.unravel_index(flat_idx, local_score.shape)

        if local_score[yy, xx] < 0:
            return None

        return float(xx), float(yy)

    point = try_pick(1.0)

    if point is None:
        point = try_pick(1.8)

    return point


def select_triangle_points(image_rgb, pred_mask, cfg=None):
    """
    image_rgb: H×W×3 uint8 RGB
    pred_mask: H×W float, [0, 1]
    returns: list of 3 (x, y), or [] if failed
    """
    if cfg is None:
        cfg = PointSelectorConfig()

    pred_mask = np.asarray(pred_mask, dtype=np.float32)
    pred_mask = np.clip(pred_mask, 0.0, 1.0)

    binary = pred_mask >= cfg.mask_threshold

    if binary.sum() == 0:
        return []

    binary = largest_component(binary)

    anchor = find_body_anchor(binary, cfg)

    if anchor is None:
        return []

    body_mask = anchor["body_mask"]
    x_center = anchor["x_center"]
    y_center = anchor["y_center"]
    body_width = anchor["body_width"]

    ideal_top = (
        x_center,
        y_center - cfg.tri_top_y_frac * body_width,
    )
    ideal_left = (
        x_center - cfg.tri_side_x_frac * body_width,
        y_center + cfg.tri_low_y_frac * body_width,
    )
    ideal_right = (
        x_center + cfg.tri_side_x_frac * body_width,
        y_center + cfg.tri_low_y_frac * body_width,
    )

    cornerness = compute_cornerness(image_rgb)
    score_map = cornerness * np.power(pred_mask, cfg.score_mask_power)

    points = []

    for center in [ideal_top, ideal_left, ideal_right]:
        point = pick_point_in_micro_region(
            center=center,
            body_mask=body_mask,
            score_map=score_map,
            body_width=body_width,
            cfg=cfg,
        )

        if point is None:
            return []

        points.append(point)

    if triangle_area(points) < cfg.min_triangle_area:
        return []

    if min_pairwise_distance(points) < cfg.min_point_distance:
        return []

    return points