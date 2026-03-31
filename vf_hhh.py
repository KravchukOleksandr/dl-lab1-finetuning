from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class TruckView:
    """
    One truck crop.
    """
    image: np.ndarray
    mask: np.ndarray


def truck_spatial_coverage_stats(
    points: np.ndarray,
    image_height: int,
) -> dict[str, int]:
    """
    Count matched points globally and across key truck regions.

    Regions:
        - top:    0%..25%
        - middle: 50%..75%
        - bottom: 75%..100%

    The 25%..50% band is intentionally skipped as a rough approximation of the
    windshield area.

    Returns:
        {
            "num_points_total": int,
            "num_points_region_min": int,
        }
    """
    num_points_total = len(points)
    if num_points_total == 0:
        return {
            "num_points_total": 0,
            "num_points_region_min": 0,
        }

    y = points[:, 1]

    top_end = 0.25 * image_height
    mid_start = 0.50 * image_height
    mid_end = 0.75 * image_height
    bottom_start = 0.75 * image_height

    num_top = int(((y >= 0) & (y < top_end)).sum())
    num_middle = int(((y >= mid_start) & (y < mid_end)).sum())
    num_bottom = int(((y >= bottom_start) & (y <= image_height)).sum())

    return {
        "num_points_total": int(num_points_total),
        "num_points_region_min": min(num_top, num_middle, num_bottom),
    }


class KeypointsFootprint:
    """
    Keypoint-only footprint.

    This class is responsible only for:
        - geometric matching
        - spatial coverage statistics
    """

    def compare(
        self,
        matcher: Any,
        feats1: Any,
        feats2: Any,
        truck1: TruckView,
    ) -> dict[str, Any]:
        """
        Match keypoints and return geometric and spatial statistics.
        """
        matched, src_pts, dst_pts = match_images(
            matcher=matcher,
            feats0=feats1,
            feats1=feats2,
        )

        stats = truck_spatial_coverage_stats(
            points=src_pts,
            image_height=truck1.image.shape[0],
        )

        return {
            "matched": matched,
            "num_points_total": stats["num_points_total"],
            "num_points_region_min": stats["num_points_region_min"],
            "src_pts": src_pts,
            "dst_pts": dst_pts,
        }


class ColorFootprint:
    """
    Aggregator of strong color, weak color and brightness footprints.

    This class assumes that incoming TruckView objects already contain
    prepared masks.
    """

    def __init__(
        self,
        strong_num_bins: int = 12,
        strong_sigma_bins: float = 1.05,
        weak_num_bins: int = 6,
        weak_sigma_bins: float = 1.05,
        brightness_num_bins: int = 6,
        brightness_sigma_bins: float = 1.25,
        strong_low_chroma_center: float = 15.0,
        strong_low_chroma_sharpness: float = 3.0,
        weak_low_chroma_center: float = 5.0,
        weak_low_chroma_sharpness: float = 0.8,
        weak_high_chroma_center: float = 15.0,
        weak_high_chroma_sharpness: float = 3.0,
        brightness_high_chroma_center: float = 5.0,
        brightness_high_chroma_sharpness: float = 0.8,
        strong_min_fraction: float = 0.10,
        weak_min_fraction: float = 0.10,
        brightness_min_fraction: float = 0.15,
    ) -> None:
        self.strong_num_bins = strong_num_bins
        self.strong_sigma_bins = strong_sigma_bins

        self.weak_num_bins = weak_num_bins
        self.weak_sigma_bins = weak_sigma_bins

        self.brightness_num_bins = brightness_num_bins
        self.brightness_sigma_bins = brightness_sigma_bins

        self.strong_low_chroma_center = strong_low_chroma_center
        self.strong_low_chroma_sharpness = strong_low_chroma_sharpness

        self.weak_low_chroma_center = weak_low_chroma_center
        self.weak_low_chroma_sharpness = weak_low_chroma_sharpness
        self.weak_high_chroma_center = weak_high_chroma_center
        self.weak_high_chroma_sharpness = weak_high_chroma_sharpness

        self.brightness_high_chroma_center = brightness_high_chroma_center
        self.brightness_high_chroma_sharpness = brightness_high_chroma_sharpness

        self.strong_min_fraction = strong_min_fraction
        self.weak_min_fraction = weak_min_fraction
        self.brightness_min_fraction = brightness_min_fraction

    def prepare(self, truck: TruckView, points: np.ndarray, top_fraction: float) -> TruckView:
        """
        Return a new TruckView with a prepared mask.
        """
        return TruckView(
            image=truck.image,
            mask=prepare_work_mask(truck.mask, points, top_fraction=top_fraction),
        )

    def strong_color_compare(
        self,
        truck1: TruckView,
        truck2: TruckView,
    ) -> dict[str, Any]:
        """
        Compare strongly saturated colors on already prepared masks.
        """
        ratio_12, ratio_21, ratio, hist1, hist2 = compare_truck_color_ratio(
            image1=truck1.image,
            mask1=truck1.mask,
            image2=truck2.image,
            mask2=truck2.mask,
            num_bins=self.strong_num_bins,
            sigma_bins=self.strong_sigma_bins,
            low_chroma_center=self.strong_low_chroma_center,
            low_chroma_sharpness=self.strong_low_chroma_sharpness,
            high_chroma_center=None,
            high_chroma_sharpness=None,
            min_fraction=self.strong_min_fraction,
        )

        return {
            "ratio_12": ratio_12,
            "ratio_21": ratio_21,
            "ratio": ratio,
            "hist1": hist1,
            "hist2": hist2,
        }

    def weak_color_compare(
        self,
        truck1: TruckView,
        truck2: TruckView,
    ) -> dict[str, Any]:
        """
        Compare weakly saturated colors on already prepared masks.
        """
        ratio_12, ratio_21, ratio, hist1, hist2 = compare_truck_color_ratio(
            image1=truck1.image,
            mask1=truck1.mask,
            image2=truck2.image,
            mask2=truck2.mask,
            num_bins=self.weak_num_bins,
            sigma_bins=self.weak_sigma_bins,
            low_chroma_center=self.weak_low_chroma_center,
            low_chroma_sharpness=self.weak_low_chroma_sharpness,
            high_chroma_center=self.weak_high_chroma_center,
            high_chroma_sharpness=self.weak_high_chroma_sharpness,
            min_fraction=self.weak_min_fraction,
        )

        return {
            "ratio_12": ratio_12,
            "ratio_21": ratio_21,
            "ratio": ratio,
            "hist1": hist1,
            "hist2": hist2,
        }

    def brightness_compare(
        self,
        truck1: TruckView,
        truck2: TruckView,
    ) -> dict[str, Any]:
        """
        Compare brightness structure on already prepared masks.
        """
        ratio_12, ratio_21, ratio, hist1, hist2 = compare_truck_brightness_ratio(
            image1=truck1.image,
            mask1=truck1.mask,
            image2=truck2.image,
            mask2=truck2.mask,
            num_bins=self.brightness_num_bins,
            sigma_bins=self.brightness_sigma_bins,
            low_chroma_center=None,
            low_chroma_sharpness=None,
            high_chroma_center=self.brightness_high_chroma_center,
            high_chroma_sharpness=self.brightness_high_chroma_sharpness,
            min_fraction=self.brightness_min_fraction,
        )

        return {
            "ratio_12": ratio_12,
            "ratio_21": ratio_21,
            "ratio": ratio,
            "hist1": hist1,
            "hist2": hist2,
        }

    def compare(
        self,
        truck1: TruckView,
        truck2: TruckView,
    ) -> dict[str, Any]:
        """
        Run all color-related channels on already prepared masks.
        """
        return {
            "strong_color": self.strong_color_compare(truck1, truck2),
            "weak_color": self.weak_color_compare(truck1, truck2),
            "brightness": self.brightness_compare(truck1, truck2),
        }


truck1 = TruckView(image=img1, mask=mask1)
truck2 = TruckView(image=img2, mask=mask2)

kp_fp = KeypointsFootprint()

color_fp = ColorFootprint(
    strong_num_bins=12,
    strong_sigma_bins=1.05,
    weak_num_bins=6,
    weak_sigma_bins=1.05,
    brightness_num_bins=6,
    brightness_sigma_bins=1.25,
    strong_low_chroma_center=15.0,
    strong_low_chroma_sharpness=3.0,
    weak_low_chroma_center=5.0,
    weak_low_chroma_sharpness=0.8,
    weak_high_chroma_center=15.0,
    weak_high_chroma_sharpness=3.0,
    brightness_high_chroma_center=5.0,
    brightness_high_chroma_sharpness=0.8,
    strong_min_fraction=0.10,
    weak_min_fraction=0.10,
    brightness_min_fraction=0.15,
)

kp_result = kp_fp.compare(
    matcher=matcher,
    feats1=feats1,
    feats2=feats2,
    truck1=truck1,
)

print(kp_result)

if kp_result["matched"]:
    truck1_prepared = color_fp.prepare(
        truck=truck1,
        points=kp_result["src_pts"],
        top_fraction=0.45,
    )
    truck2_prepared = color_fp.prepare(
        truck=truck2,
        points=kp_result["dst_pts"],
        top_fraction=0.45,
    )

    color_result = color_fp.compare(truck1_prepared, truck2_prepared)
    print(color_result["strong_color"]["ratio"])
    print(color_result["weak_color"]["ratio"])
    print(color_result["brightness"]["ratio"])