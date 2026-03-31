from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TruckView:
    image: np.ndarray
    mask: np.ndarray
    points: np.ndarray


class TruckColorFootprint:
    """
    Color-only visual footprint for truck comparison.

    The class does not implement low-level math itself.
    It wraps existing color-related functions and keeps all color
    hyperparameters in one place.
    """

    def __init__(
        self,
        top_fraction: float = 0.45,
        strong_num_bins: int = 12,
        strong_sigma_bins: float = 0.9,
        weak_num_bins: int = 6,
        weak_sigma_bins: float = 1.2,
        brightness_num_bins: int = 6,
        brightness_sigma_bins: float = 1.5,
        strong_low_chroma_center: float = 15.0,
        strong_low_chroma_sharpness: float = 3.0,
        weak_low_chroma_center: float = 5.0,
        weak_low_chroma_sharpness: float = 0.8,
        weak_high_chroma_center: float = 15.0,
        weak_high_chroma_sharpness: float = 3.0,
        brightness_high_chroma_center: float = 5.0,
        brightness_high_chroma_sharpness: float = 0.6,
        min_fraction: float = 0.20,
    ) -> None:
        self.top_fraction = top_fraction

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

        self.min_fraction = min_fraction

    def prepare_mask(self, truck: TruckView) -> np.ndarray:
        """
        Prepare working mask for color analysis.
        """
        return prepare_work_mask(
            truck.mask,
            truck.points,
            top_fraction=self.top_fraction,
        )

    def strong_color_ratio(self, truck1: TruckView, truck2: TruckView) -> float:
        """
        Compare strongly saturated colors.
        """
        ratio_12, ratio_21, ratio_min, _, _ = compare_truck_color_ratio(
            image1=truck1.image,
            mask1=truck1.mask,
            src_pts=truck1.points,
            image2=truck2.image,
            mask2=truck2.mask,
            dst_pts=truck2.points,
            top_fraction=self.top_fraction,
            num_bins=self.strong_num_bins,
            sigma_bins=self.strong_sigma_bins,
            low_chroma_center=self.strong_low_chroma_center,
            low_chroma_sharpness=self.strong_low_chroma_sharpness,
            high_chroma_center=None,
            high_chroma_sharpness=None,
            min_fraction=self.min_fraction,
        )
        return ratio_min

    def weak_color_ratio(self, truck1: TruckView, truck2: TruckView) -> float:
        """
        Compare weakly saturated colors.
        """
        ratio_12, ratio_21, ratio_min, _, _ = compare_truck_color_ratio(
            image1=truck1.image,
            mask1=truck1.mask,
            src_pts=truck1.points,
            image2=truck2.image,
            mask2=truck2.mask,
            dst_pts=truck2.points,
            top_fraction=self.top_fraction,
            num_bins=self.weak_num_bins,
            sigma_bins=self.weak_sigma_bins,
            low_chroma_center=self.weak_low_chroma_center,
            low_chroma_sharpness=self.weak_low_chroma_sharpness,
            high_chroma_center=self.weak_high_chroma_center,
            high_chroma_sharpness=self.weak_high_chroma_sharpness,
            min_fraction=self.min_fraction,
        )
        return ratio_min

    def brightness_ratio(self, truck1: TruckView, truck2: TruckView) -> float:
        """
        Compare low-chroma brightness structure.
        """
        ratio_12, ratio_21, ratio_min, _, _ = compare_truck_brightness_ratio(
            image1=truck1.image,
            mask1=truck1.mask,
            src_pts=truck1.points,
            image2=truck2.image,
            mask2=truck2.mask,
            dst_pts=truck2.points,
            top_fraction=self.top_fraction,
            num_bins=self.brightness_num_bins,
            sigma_bins=self.brightness_sigma_bins,
            low_chroma_center=None,
            low_chroma_sharpness=None,
            high_chroma_center=self.brightness_high_chroma_center,
            high_chroma_sharpness=self.brightness_high_chroma_sharpness,
            min_fraction=self.min_fraction,
        )
        return ratio_min

    def compare(self, truck1: TruckView, truck2: TruckView) -> dict[str, float]:
        """
        Run all color-related comparison channels.
        """
        return {
            "strong_color_ratio": self.strong_color_ratio(truck1, truck2),
            "weak_color_ratio": self.weak_color_ratio(truck1, truck2),
            "brightness_ratio": self.brightness_ratio(truck1, truck2),
        }