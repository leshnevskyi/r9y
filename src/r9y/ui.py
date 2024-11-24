import tkinter as tk
from tkinter import Tk, ttk
from typing import Optional

from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from r9y.core import IvpSolution


def root_geometry(root: Tk, width: int, height: int):
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")


def num_entry(
    frame: ttk.Frame,
    label_text: str,
    row: int,
    def_val: Optional[float] = None,
) -> ttk.Entry:
    label = ttk.Label(frame, text=label_text)
    label.grid(column=0, row=row, sticky="E")
    entry = ttk.Entry(frame)
    entry.grid(column=1, row=row, sticky="EW")

    if def_val is not None:
        entry.insert(0, str(def_val))

    return entry


def setup_plot(frame: ttk.Frame):
    fig = Figure(figsize=(8, 4))

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    ax = fig.add_subplot(111)
    ax.grid()

    return canvas, ax


def draw_plot(canvas: FigureCanvasTkAgg, ax: Axes, sol: IvpSolution):
    ax.clear()
    ax.set_title("State Probabilities")
    ax.set_xlabel("Time, t")
    ax.set_ylabel("State Probability, P(t)")
    ax.grid()

    for i in range(sol.y.shape[0]):
        ax.plot(sol.t, sol.y[i], label=f"P{i+1}(t)")

    ax.legend()
    canvas.draw()
