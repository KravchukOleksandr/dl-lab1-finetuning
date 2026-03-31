from __future__ import annotations

from typing import Any


class KeypointsFootprint:
    """
    Keypoint-only footprint.

    The class owns:
        - feature extractor
        - matcher

    And is responsible for:
        - extracting image features
        - matching keypoints
        - computing spatial coverage statistics
    """

    def __init__(
        self,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.extractor, self.matcher = init_models(device=device)

    def extract_features(self, image: Any) -> Any:
        """
        Extract local features from one image.
        """
        return extract_image_features(
            extractor=self.extractor,
            frame=image,
            device=self.device,
        )

    def compare(
        self,
        truck1: TruckView,
        truck2: TruckView,
    ) -> dict[str, Any]:
        """
        Compare two truck crops using keypoints only.

        Returns:
            {
                "matched": bool,
                "num_points_total": int,
                "num_points_region_min": int,
                "src_pts": np.ndarray,
                "dst_pts": np.ndarray,
            }
        """
        feats1 = self.extract_features(truck1.image)
        feats2 = self.extract_features(truck2.image)

        matched, src_pts, dst_pts = match_images(
            matcher=self.matcher,
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


kp_fp = KeypointsFootprint(device="cpu")

truck1 = TruckView(
    image=img1,
    mask=mask1,
)

truck2 = TruckView(
    image=img2,
    mask=mask2,
)

kp_result = kp_fp.compare(truck1, truck2)

print(kp_result["matched"])
print(kp_result["num_points_total"])
print(kp_result["num_points_region_min"])
print(kp_result["src_pts"].shape)
print(kp_result["dst_pts"].shape)