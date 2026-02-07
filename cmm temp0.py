import json


def merge_zone_and_plane_indices_to_zone3d_points(
    zone_json_path: str,
    plane_indices_json_path: str,
) -> dict[int, list[list[list[float | int]]]]:
    """
    zone.json:
      { idx: [ [[x,y], ...], ... ] }

    plane_indices.json:
      { idx: [ [plane_idx, ...], ... ] }

    returns:
      { idx: [ [ [x,y,plane_idx], ... ], ... ] }
    """
    with open(zone_json_path, "r", encoding="utf-8") as f:
        zone = json.load(f)

    with open(plane_indices_json_path, "r", encoding="utf-8") as f:
        plane_indices = json.load(f)

    zone3d = {}

    for k, polygons in zone.items():
        idx = int(k)
        zone3d[idx] = [
            [[pt[0], pt[1], pi] for pt, pi in zip(polygon, plane_indices[str(k)][i])]
            for i, polygon in enumerate(polygons)
        ]

    return zone3d