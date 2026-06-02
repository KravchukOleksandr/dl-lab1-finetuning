from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class GRN(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MicroNeXtBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 7, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        padding = kernel // 2

        self.dw = nn.Conv2d(channels, channels, kernel_size=kernel, padding=padding, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.grn = GRN(hidden)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw(x)
        x = self.bn(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)
        return x + shortcut


class MicroNeXtMaskNet(nn.Module):
    """Input: Bx3x128x64. Output: Bx1x32x16 logits."""
    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(3, 16, kernel=3, stride=2)
        self.stage1 = nn.Sequential(MicroNeXtBlock(16, kernel=7, expansion=2))
        self.down = ConvBNAct(16, 24, kernel=3, stride=2)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        x = self.proj(x)
        return self.head(x)


@dataclass
class SimplePointSelectorConfig:
    mask_threshold: float = 0.80
    torso_y_min: float = 0.03
    torso_y_max: float = 0.60
    max_points: int = 5

    # dist_threshold = min(max_abs_px, rel_from_dmax * dmax)
    attempts: Tuple[Tuple[float, float, float, float], ...] = (
        (10.0, 0.75, 6.0, 0.010),
        (8.0, 0.65, 6.0, 0.010),
        (6.0, 0.55, 6.0, 0.008),
        (4.0, 0.45, 5.0, 0.006),
        (3.0, 0.35, 5.0, 0.005),
        (2.0, 0.25, 4.0, 0.004),
        (1.0, 0.15, 4.0, 0.003),
    )

    min_component_area: int = 12
    block_size: int = 3


def largest_component(binary_mask: np.ndarray) -> np.ndarray:
    mask_u8 = binary_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return binary_mask.astype(bool)
    best_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return labels == best_label


def select_points_simple(
    image_rgb: np.ndarray,
    pred_mask: np.ndarray,
    cfg: Optional[SimplePointSelectorConfig] = None,
) -> List[Tuple[float, float]]:
    if cfg is None:
        cfg = SimplePointSelectorConfig()

    pred_mask = np.clip(np.asarray(pred_mask, dtype=np.float32), 0.0, 1.0)
    h, w = pred_mask.shape

    binary = pred_mask >= cfg.mask_threshold
    if binary.sum() < cfg.min_component_area:
        return []

    binary = largest_component(binary)
    ys, xs = np.where(binary)
    if len(xs) == 0 or len(ys) == 0:
        return []

    my1 = int(ys.min())
    my2 = int(ys.max()) + 1
    mask_h = max(1, my2 - my1)

    y0 = int(round(my1 + cfg.torso_y_min * mask_h))
    y1 = int(round(my1 + cfg.torso_y_max * mask_h))
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))

    torso_mask = np.zeros_like(binary, dtype=bool)
    torso_mask[y0:y1] = binary[y0:y1]
    if torso_mask.sum() < cfg.min_component_area:
        return []

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb.copy()
    gray = gray.astype(np.uint8)

    dist = cv2.distanceTransform(torso_mask.astype(np.uint8), cv2.DIST_L2, 3)
    dmax = float(dist.max())
    if dmax <= 1e-6:
        return []

    best_points: List[Tuple[float, float]] = []

    for max_abs_dist, rel_dist, min_dist, quality in cfg.attempts:
        dist_thr = min(float(max_abs_dist), float(rel_dist) * dmax)
        safe_mask = (dist >= dist_thr).astype(np.uint8) * 255
        if safe_mask.sum() == 0:
            continue

        pts = cv2.goodFeaturesToTrack(
            image=gray,
            maxCorners=cfg.max_points,
            qualityLevel=float(quality),
            minDistance=float(min_dist),
            mask=safe_mask,
            blockSize=cfg.block_size,
            useHarrisDetector=False,
        )
        if pts is None:
            continue

        points = [(float(p[0][0]), float(p[0][1])) for p in pts]
        if len(points) > len(best_points):
            best_points = points
        if len(points) >= cfg.max_points:
            return points[:cfg.max_points]

    return best_points[:cfg.max_points]
