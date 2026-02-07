self._sync_plane_indices_full()
self._sanitize_plane_indices()

arr = self.plane_indices[self.cur_group][self.cur_poly_idx]
if 0 <= self.selected_vertex < len(arr):
    arr[self.selected_vertex] = int(p)
self.redraw()


self._sync_plane_indices_full()
self._sanitize_plane_indices()

arr = self.plane_indices[self.cur_group][self.cur_poly_idx]
start = (self.selected_vertex + 1) if self.selected_vertex is not None else 0
n = len(arr)
for step in range(n):
    i = (start + step) % n
    if arr[i] == -1:
        self.selected_vertex = i
        self.redraw()
        return
messagebox.showinfo("Bind", "No unbound vertices in this polygon.")