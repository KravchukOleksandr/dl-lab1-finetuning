def load_plane_indices(path: str) -> Dict[str, List[List[int]]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure ints
    out: Dict[str, List[List[int]]] = {}
    for gid, polys in data.items():
        out[gid] = []
        for arr in polys:
            out[gid].append([int(v) for v in arr])
    return out


def save_plane_indices(path: str, data: Dict[str, List[List[int]]], zone: Dict[str, List[List[List[float]]]]) -> int:
    """
    Save plane indices, aligned to zone polygons.
    Draft zone polygons (<3 vertices) are skipped, same as zone.json logic.
    Returns number of skipped polygons (for message only).
    """
    out = {}
    skipped = 0
    for gid, zone_polys in zone.items():
        idx_polys = data.get(gid, [])
        out[gid] = []
        for pi, poly2 in enumerate(zone_polys):
            if len(poly2) < 3:
                skipped += 1
                continue
            arr = idx_polys[pi] if pi < len(idx_polys) else [-1] * len(poly2)
            # force correct length
            if len(arr) != len(poly2):
                if len(arr) > len(poly2):
                    arr = arr[:len(poly2)]
                else:
                    arr = arr + [-1] * (len(poly2) - len(arr))
            out[gid].append([int(v) for v in arr])

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return skipped



