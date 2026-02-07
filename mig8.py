if self.mode != MODE_PLANES:
    arr_polys = self.plane_indices.get(self.cur_group, [])
    if 0 <= self.cur_poly_idx < len(arr_polys):
        arr_polys.pop(self.cur_poly_idx)