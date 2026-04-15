from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
import colour


@dataclass
class SupportColor:
    oklch: np.ndarray   # [L, C, h_deg]
    oklab: np.ndarray   # [L, a, b]
    srgb: np.ndarray    # [r, g, b] in [0, 1]


class OklabColorFootprint:
    """
    Fixed-bin color footprint in Oklab.

    Bin centers are generated in Oklch:
        - uniform levels in L
        - uniform levels in C
        - angular sectors per chroma ring

    Distances are computed in Oklab with Euclidean metric.
    """

    def __init__(
        self,
        n_L: int = 6,
        n_C: int = 5,
        c_max: float = 0.32,
        sigma: float = 0.10,
        min_fraction: float = 0.10,
        use_bin_centers_for_L: bool = True,
    ) -> None:
        """
        Args:
            n_L:
                Number of lightness levels.
            n_C:
                Number of chroma levels, including C=0.
            c_max:
                Maximum Oklch chroma before gamut filtering.
            sigma:
                Common Gaussian sigma in Oklab.
            min_fraction:
                Minimal histogram mass to keep in directional comparison.
            use_bin_centers_for_L:
                If True, L levels are taken at bin centers.
        """
        self.n_L = n_L
        self.n_C = n_C
        self.c_max = c_max
        self.sigma = sigma
        self.min_fraction = min_fraction
        self.use_bin_centers_for_L = use_bin_centers_for_L

        self.support_colors = self._generate_support_colors()
        self.centers_oklab = np.stack([c.oklab for c in self.support_colors], axis=0).astype(np.float32)
        self.centers_srgb = np.stack([c.srgb for c in self.support_colors], axis=0).astype(np.float32)

    def _oklab_to_srgb(self, oklab: np.ndarray) -> np.ndarray | None:
        """
        Convert one Oklab color to sRGB and reject out-of-gamut colors.
        """
        XYZ = colour.Oklab_to_XYZ(oklab.astype(np.float64))
        rgb = colour.XYZ_to_sRGB(XYZ)

        if np.any(~np.isfinite(rgb)):
            return None
        if np.any(rgb < 0.0) or np.any(rgb > 1.0):
            return None

        return np.clip(rgb, 0.0, 1.0).astype(np.float32)

    def _generate_support_colors(self) -> list[SupportColor]:
        """
        Generate fixed support colors in Oklch and keep only valid sRGB colors.
        """
        if self.n_L < 1:
            raise ValueError("n_L must be >= 1")
        if self.n_C < 2:
            raise ValueError("n_C must be >= 2")

        if self.use_bin_centers_for_L:
            L_values = (np.arange(self.n_L, dtype=np.float64) + 0.5) / self.n_L
        else:
            if self.n_L == 1:
                L_values = np.array([0.5], dtype=np.float64)
            else:
                L_values = np.linspace(0.0, 1.0, self.n_L, dtype=np.float64)

        C_values = np.linspace(0.0, self.c_max, self.n_C, dtype=np.float64)

        centers: list[SupportColor] = []

        for L in L_values:
            for j, C in enumerate(C_values):
                if j == 0:
                    oklch = np.array([L, 0.0, 0.0], dtype=np.float64)
                    oklab = colour.Oklch_to_Oklab(oklch)
                    srgb = self._oklab_to_srgb(oklab)

                    if srgb is not None:
                        centers.append(SupportColor(oklch=oklch, oklab=oklab.astype(np.float32), srgb=srgb))
                else:
                    n_h = 4 * j

                    for k in range(n_h):
                        h_deg = 360.0 * k / n_h
                        oklch = np.array([L, C, h_deg], dtype=np.float64)
                        oklab = colour.Oklch_to_Oklab(oklch)
                        srgb = self._oklab_to_srgb(oklab)

                        if srgb is not None:
                            centers.append(SupportColor(oklch=oklch, oklab=oklab.astype(np.float32), srgb=srgb))

        if not centers:
            raise RuntimeError("No valid support colors remained after gamut filtering.")

        return centers

    def _bgr_to_oklab(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Convert BGR uint8 image to Oklab float32.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        XYZ = colour.sRGB_to_XYZ(image_rgb)
        oklab = colour.XYZ_to_Oklab(XYZ)
        return oklab.astype(np.float32)

    def _compute_soft_histogram(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute soft histogram over fixed Oklab support colors.

        Histogram is normalized by the full mask area.
        """
        valid = mask > 0
        total_area = int(valid.sum())

        if total_area == 0:
            return np.zeros(len(self.support_colors), dtype=np.float32)

        oklab = self._bgr_to_oklab(image_bgr)
        pixels = oklab[valid].reshape(-1, 3).astype(np.float32)

        # distances: [num_pixels, num_centers]
        diff = pixels[:, None, :] - self.centers_oklab[None, :, :]
        d2 = np.sum(diff * diff, axis=2)

        raw_w = np.exp(-d2 / (2.0 * self.sigma * self.sigma)).astype(np.float32)
        raw_w /= raw_w.sum(axis=1, keepdims=True) + 1e-12

        hist = raw_w.sum(axis=0).astype(np.float32)
        hist /= float(total_area)

        return hist

    def _directional_ratio(
        self,
        hist1: np.ndarray,
        hist2: np.ndarray,
    ) -> float:
        """
        Compute worst retained ratio from hist1 to hist2.

        Only bins with hist1 >= min_fraction are checked.
        If there are no such bins, returns 1.0.
        """
        keep = hist1 >= self.min_fraction
        if not np.any(keep):
            return 1.0

        ratios = hist2[keep] / (hist1[keep] + 1e-12)
        return float(ratios.min())

    def compare_color_ratio(
        self,
        image1_bgr: np.ndarray,
        mask1: np.ndarray,
        image2_bgr: np.ndarray,
        mask2: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        """
        Compare two truck crops by Oklab fixed-bin histograms.

        Returns:
            {
                "ratio_12": float,
                "ratio_21": float,
                "ratio": float,
                "hist1": np.ndarray,
                "hist2": np.ndarray,
            }
        """
        hist1 = self._compute_soft_histogram(image1_bgr, mask1)
        hist2 = self._compute_soft_histogram(image2_bgr, mask2)

        ratio_12 = self._directional_ratio(hist1, hist2)
        ratio_21 = self._directional_ratio(hist2, hist1)
        ratio = min(ratio_12, ratio_21)

        return {
            "ratio_12": ratio_12,
            "ratio_21": ratio_21,
            "ratio": ratio,
            "hist1": hist1,
            "hist2": hist2,
        }

    def show_support_palette(
        self,
        max_cols: int = 16,
        patch_size: int = 60,
    ) -> None:
        """
        Visualize all support colors in one palette.
        """
        import matplotlib.pyplot as plt

        centers = sorted(
            self.support_colors,
            key=lambda c: (float(c.oklch[0]), float(c.oklch[1]), float(c.oklch[2]))
        )

        n = len(centers)
        cols = min(max_cols, n)
        rows = math.ceil(n / cols)

        canvas = np.ones((rows * patch_size, cols * patch_size, 3), dtype=np.float32)

        for i, c in enumerate(centers):
            r = i // cols
            col = i % cols

            y1 = r * patch_size
            y2 = (r + 1) * patch_size
            x1 = col * patch_size
            x2 = (col + 1) * patch_size

            canvas[y1:y2, x1:x2] = c.srgb

        plt.figure(figsize=(cols * 0.6, rows * 0.6))
        plt.imshow(canvas)
        plt.title(f"Total support colors: {len(centers)}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


footprint = OklabColorFootprint(
    n_L=6,
    n_C=5,
    c_max=0.32,
    sigma=0.10,
    min_fraction=0.10,
    use_bin_centers_for_L=True,
)

print("Number of support colors:", len(footprint.support_colors))

footprint.show_support_palette()

result = footprint.compare_color_ratio(
    image1_bgr=img1,
    mask1=work_mask1,
    image2_bgr=img2,
    mask2=work_mask2,
)

print("ratio 1->2:", result["ratio_12"])
print("ratio 2->1:", result["ratio_21"])
print("final ratio:", result["ratio"])
print("hist1 shape:", result["hist1"].shape)
print("hist2 shape:", result["hist2"].shape)
