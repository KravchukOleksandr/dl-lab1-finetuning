def _auto_assign_if_single_plane(self):
    keys = sort_numeric_str(list(self.planes.keys()))
    if len(keys) == 1 and keys[0] == "0":
        for gid, polys in self.plane_indices.items():
            for arr in polys:
                for i in range(len(arr)):
                    arr[i] = 0