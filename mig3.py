def _init_plane_indices_from_zone(self) -> Dict[str, List[List[int]]]:
    out: Dict[str, List[List[int]]] = {}
    default_p = self._default_plane_for_new_point()
    for gid, polys in self.zone.items():
        out[gid] = []
        for poly in polys:
            out[gid].append([default_p] * len(poly))
    return out


def _sync_plane_indices_full(self) -> None:
    default_p = self._default_plane_for_new_point()

    for gid, zone_polys in self.zone.items():
        self.plane_indices.setdefault(gid, [])

        # Match polygon count
        while len(self.plane_indices[gid]) < len(zone_polys):
            self.plane_indices[gid].append([])
        if len(self.plane_indices[gid]) > len(zone_polys):
            self.plane_indices[gid] = self.plane_indices[gid][:len(zone_polys)]

        # Match vertex count per polygon
        for pi, poly2 in enumerate(zone_polys):
            arr = self.plane_indices[gid][pi]
            if len(arr) > len(poly2):
                self.plane_indices[gid][pi] = arr[:len(poly2)]
            elif len(arr) < len(poly2):
                self.plane_indices[gid][pi] = arr + [default_p] * (len(poly2) - len(arr))

    # Remove groups that don't exist in zone
    for gid in list(self.plane_indices.keys()):
        if gid not in self.zone:
            del self.plane_indices[gid]


def _ensure_plane_indices_sync_current(self) -> None:
    if self.cur_group is None:
        return

    default_p = self._default_plane_for_new_point()
    self.plane_indices.setdefault(self.cur_group, [])

    zone_polys = self.zone.get(self.cur_group, [])

    while len(self.plane_indices[self.cur_group]) < len(zone_polys):
        self.plane_indices[self.cur_group].append([])

    if self.cur_poly_idx >= len(zone_polys):
        return

    poly2 = zone_polys[self.cur_poly_idx]
    arr = self.plane_indices[self.cur_group][self.cur_poly_idx]

    if len(arr) > len(poly2):
        self.plane_indices[self.cur_group][self.cur_poly_idx] = arr[:len(poly2)]
    elif len(arr) < len(poly2):
        self.plane_indices[self.cur_group][self.cur_poly_idx] = arr + [default_p] * (len(poly2) - len(arr))


def _sanitize_plane_indices(self) -> None:
    existing = self._existing_plane_ids_set()
    for gid, polys in self.plane_indices.items():
        for arr in polys:
            for i, p in enumerate(arr):
                p = int(p)
                if p != -1 and p not in existing:
                    arr[i] = -1