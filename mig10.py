poly2 = self.zone[self.cur_group][self.cur_poly_idx]                  # [[x,y], ...] (OPEN)
arr = self.plane_indices[self.cur_group][self.cur_poly_idx]          # [p0,p1,...]  (OPEN)

for i, (x, y) in enumerate(poly2):
    p = arr[i] if i < len(arr) else -1
    X, Y = self.viewer.norm_to_canvas(x, y)
    fill = UNBOUND_COLOR if p == -1 else self._color_plane(str(int(p)))
    outline = "white" if self.selected_vertex == i else "black"
    self._circle(X, Y, 8, fill=fill, outline=outline)