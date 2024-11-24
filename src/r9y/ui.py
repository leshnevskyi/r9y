from tkinter import Tk, ttk
from typing import Optional


def root_geometry(root: Tk, width: int, height: int):
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")


def frame_entry(
    frame: ttk.Frame,
    label_text: str,
    row: int,
    def_val: Optional[float] = None,
) -> ttk.Entry:
    ttk.Label(frame, text=label_text).grid(column=0, row=row, sticky="W")
    entry = ttk.Entry(frame)
    entry.grid(column=1, row=row, sticky="EW")

    if def_val is not None:
        entry.insert(0, str(def_val))

    return entry
