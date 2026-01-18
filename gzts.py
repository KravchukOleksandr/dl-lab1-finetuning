def save_zone_only(self):
    """Save only frame copy + 2D zone json. Do NOT touch planes/zone3d."""
    ensure_dir(self.out_dir)

    # Copy frame if not present
    if not os.path.exists(self.out_frame_path):
        shutil.copy2(os.path.join(FRAMES_DIR, self.frame_name), self.out_frame_path)

    try:
        skipped = save_json_2d(self.zone_path, self.zone)
    except Exception as e:
        messagebox.showerror("Save error", str(e))
        return

    msg = "Saved zone+frame ✔" if skipped == 0 else f"Saved zone+frame ✔ (skipped drafts: {skipped})"
    # If you already implemented flash_msg, use it; otherwise use lbl_status directly
    if hasattr(self, "flash_msg"):
        self.flash_msg(msg, ms=1400)
    else:
        self.lbl_status.config(text=msg)

    self.redraw()



ttk.Button(top, text="Save zone+frame", command=self.save_zone_only).pack(side="left", padx=2)


