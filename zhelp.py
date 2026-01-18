# Message label for transient notifications like "Saved"
self.lbl_msg = ttk.Label(top, text="", foreground="#7CFC00")
self.lbl_msg.pack(side="right", padx=(10, 0))

# Mode/status label (DRAWING / BIND info, etc.)
self.lbl_status = ttk.Label(top, text="", foreground="orange")
self.lbl_status.pack(side="right")


def flash_msg(self, text: str, ms: int = 1200) -> None:
    self.lbl_msg.config(text=text)
    self.after(ms, lambda: self.lbl_msg.config(text=""))


msg = "Saved ✔" if skipped_total == 0 else f"Saved ✔ (skipped drafts: {skipped_total})"
self.flash_msg(msg, ms=1400)
self.redraw()


self.flash_msg(f"Plane {p} doesn't exist", ms=1200)
return
