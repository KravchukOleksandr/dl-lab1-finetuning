def assign_plane(self, p: int) -> None:
    """Assign plane index `p` to the currently selected zone vertex (BIND mode)."""
    if self.mode != MODE_BIND:
        return

    # Safety: prevent assigning a non-existing plane
    existing = self._existing_plane_ids_set()
    if p not in existing:
        # If you have flash_msg() use it; otherwise fall back to lbl_status
        if hasattr(self, "flash_msg"):
            self.flash_msg(f"Plane {p} doesn't exist", ms=1200)
        else:
            self.lbl_status.config(text=f"Plane {p} doesn't exist")
        return

    # Need a selected vertex and a valid active polygon
    if self.cur_group is None or self.selected_vertex is None:
        return
    if self.cur_group not in self.zone:
        return
    if not (0 <= self.cur_poly_idx < len(self.zone[self.cur_group])):
        return

    # Ensure plane_indices structure matches zone structure (OPEN in memory)
    self._sync_plane_indices_full()
    self._sanitize_plane_indices_plane_ids()

    if self.cur_group not in self.plane_indices:
        return
    if not (0 <= self.cur_poly_idx < len(self.plane_indices[self.cur_group])):
        return

    arr = self.plane_indices[self.cur_group][self.cur_poly_idx]

    if not (0 <= self.selected_vertex < len(arr)):
        return

    arr[self.selected_vertex] = int(p)
    self.redraw()



def next_unbound_vertex(self) -> None:
    """Select next vertex with plane index == -1 in the current zone polygon (BIND mode)."""
    if self.mode != MODE_BIND:
        return
    if self.cur_group is None:
        return
    if self.cur_group not in self.zone:
        return
    if not (0 <= self.cur_poly_idx < len(self.zone[self.cur_group])):
        return

    self._sync_plane_indices_full()
    self._sanitize_plane_indices_plane_ids()

    if self.cur_group not in self.plane_indices:
        return
    if not (0 <= self.cur_poly_idx < len(self.plane_indices[self.cur_group])):
        return

    arr = self.plane_indices[self.cur_group][self.cur_poly_idx]
    if not arr:
        return

    start = (self.selected_vertex + 1) if self.selected_vertex is not None else 0
    n = len(arr)

    for step in range(n):
        i = (start + step) % n
        if int(arr[i]) == -1:
            self.selected_vertex = i
            self.redraw()
            return

    messagebox.showinfo("Bind", "No unbound vertices in this polygon.")