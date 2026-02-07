def _has_unbound_plane_indices(self) -> bool:
    """
    Returns True if there is at least one unbound vertex (plane index == -1)
    in polygons that are valid (>=3 vertices) and will be saved.
    """
    for gid, zone_polys in self.zone.items():
        idx_polys = self.plane_indices.get(gid, [])
        for pi, poly2 in enumerate(zone_polys):
            if len(poly2) < 3:
                continue  # drafts are skipped on save anyway
            if pi >= len(idx_polys):
                return True  # missing indices => unbound by definition
            arr = idx_polys[pi]
            # enforce length safety
            n = len(poly2)
            if len(arr) < n:
                return True
            # check unbound
            for i in range(n):
                if int(arr[i]) == -1:
                    return True
    return False



def _has_valid_zone_polygons(self) -> bool:
    """
    Returns True if zone contains at least one valid polygon
    (polygon with >= 3 vertices).
    """
    for polys in self.zone.values():
        for poly in polys:
            if len(poly) >= 3:
                return True
    return False