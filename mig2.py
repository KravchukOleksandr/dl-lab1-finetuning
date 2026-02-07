def open_indices(arr_closed: List[int]) -> List[int]:
    # Internally store indices open (no duplicated last index)
    if len(arr_closed) >= 2 and arr_closed[0] == arr_closed[-1]:
        return arr_closed[:-1]
    return arr_closed[:]


def close_indices(arr_open: List[int]) -> List[int]:
    # On save, close if there is at least 3 vertices (same rule as polygons)
    if len(arr_open) >= 3:
        return arr_open + [arr_open[0]]
    return arr_open[:]


def load_plane_indices(path: str) -> Dict[str, List[List[int]]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, List[List[int]]] = {}
    for gid, polys in data.items():
        out[gid] = []
        for arr in polys:
            arr_int = [int(v) for v in arr]
            out[gid].append(open_indices(arr_int))
    return out


def save_plane_indices(path: str, data: Dict[str, List[List[int]]], zone: Dict[str, List[List[List[float]]]]) -> int:
    out = {}
    skipped = 0

    for gid, zone_polys in zone.items():
        idx_polys = data.get(gid, [])
        out[gid] = []

        for pi, poly2_open in enumerate(zone_polys):
            if len(poly2_open) < 3:
                skipped += 1
                continue

            arr_open = idx_polys[pi] if pi < len(idx_polys) else [-1] * len(poly2_open)

            # force open-length match (N)
            if len(arr_open) > len(poly2_open):
                arr_open = arr_open[:len(poly2_open)]
            elif len(arr_open) < len(poly2_open):
                arr_open = arr_open + [-1] * (len(poly2_open) - len(arr_open))

            # NOW close indices to match zone.json closed polygon length (N+1)
            out[gid].append(close_indices([int(v) for v in arr_open]))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return skipped