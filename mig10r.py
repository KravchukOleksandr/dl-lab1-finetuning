def _draw_bind_vertices(self) -> None:
    """Draw bindable vertices: x,y from zone; plane assignment from plane_indices."""
    if self.cur_group is None:
        return
    if self.cur_group not in self.zone:
        return
    if not (0 <= self.cur_poly_idx < len(self.zone[self.cur_group])):
        return

    # Ensure indices are aligned and valid
    self._sync_plane_indices_full()
    self._auto_assign_if_single_plane()
    self._sanitize_plane_indices_plane_ids()

    if self.cur_group not in self.plane_indices:
        return
    if not (0 <= self.cur_poly_idx < len(self.plane_indices[self.cur_group])):
        return

    poly2 = self.zone[self.cur_group][self.cur_poly_idx]                  # [[x,y], ...] (OPEN)
    arr = self.plane_indices[self.cur_group][self.cur_poly_idx]           # [p0,p1,...]  (OPEN)

    if not poly2:
        return

    # Safety: if something went out of sync, enforce lengths now (shouldn't happen if sync works)
    if len(arr) != len(poly2):
        default_p = self._default_plane_for_new_point()
        if len(arr) > len(poly2):
            arr[:] = arr[:len(poly2)]
        else:
            arr.extend([default_p] * (len(poly2) - len(arr)))

    unbound = sum(1 for p in arr if int(p) == -1)

    # Status text
    keys = sort_numeric_str(list(self.planes.keys()))
    if len(keys) == 1 and keys[0] == "0":
        self.lbl_status.config(text="Only one plane → auto p=0")
    else:
        # Don't overwrite recent error message like "Plane X doesn't exist"
        cur_text = self.lbl_status.cget("text") if hasattr(self, "lbl_status") else ""
        if not (isinstance(cur_text, str) and cur_text.startswith("Plane ")):
            self.lbl_status.config(text=f"BIND: unbound={unbound}")

    # Draw circles at zone vertices, color by plane assignment
    for i, (x, y) in enumerate(poly2):
        p = int(arr[i])
        X, Y = self.viewer.norm_to_canvas(x, y)

        if p == -1:
            fill = UNBOUND_COLOR
        else:
            fill = self._color_plane(str(p))

        outline = "white" if (self.selected_vertex == i) else "black"
        self._circle(X, Y, 8, fill=fill, outline=outline)