import tkinter as tk
from tkinter import Tk, ttk
from typing import List, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from r9y.core import IvpSolution, SystemType, solve_r9y_sys_nonrec, solve_r9y_sys_rec


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

    ax = fig.add_subplot(1, 1, 1)
    ax.grid()

    return canvas, ax


def draw_plot(
    canvas: FigureCanvasTkAgg, ax: Axes, sol: IvpSolution, green_states: List[int]
):
    ax.clear()
    ax.set_title("State Probabilities")
    ax.set_xlabel("Time, t")
    ax.set_ylabel("State Probability, P(t)")
    ax.grid()

    ax.plot(
        sol.t,
        np.sum(sol.y[green_states], axis=0),
        label="P(t)",
        linestyle="--",
        linewidth=1,
        color="black",
    )

    for i in range(sol.y.shape[0]):
        ax.plot(sol.t, sol.y[i], label=f"P{i+1}(t)", linewidth=0.75)

    ax.legend()
    canvas.draw()


def add_tab(notebook: ttk.Notebook, tab_name: str) -> ttk.Frame:
    tab = ttk.Frame(notebook)
    notebook.add(tab, text=tab_name)

    entry_frame = ttk.Frame(tab, padding="20 0")
    entry_frame.grid(row=0, column=0, sticky="NS")
    entry_frame.columnconfigure(1, weight=1)

    error_label = ttk.Label(entry_frame, text="", foreground="red")
    error_label.grid(column=0, row=0, columnspan=2)

    entry_t_start = num_entry(entry_frame, "t1", row=1, def_val=0)
    entry_t_end = num_entry(entry_frame, "t2", row=2, def_val=3000)

    entry_row_offset = len([error_label, entry_t_start, entry_t_end])
    lam_h_len = 3
    lam_h_entries = [
        num_entry(
            entry_frame,
            f"λ{i + 1}h",
            row=(
                i + entry_row_offset
                if tab_name == SystemType.NONRECOVERABLE.value
                else i * 2 + entry_row_offset
            ),
            def_val=round((5 - i) * 1e-4, 4),
        )
        for i in range(lam_h_len)
    ]
    mu_h_entries = (
        []
        if tab_name == SystemType.NONRECOVERABLE.value
        else [
            num_entry(
                entry_frame,
                f"μ{i + 1}h",
                row=(
                    i
                    if tab_name == SystemType.NONRECOVERABLE.value
                    else i * 2 + entry_row_offset + 1
                ),
                def_val=round((7 - i) * 1e-4, 4),
            )
            for i in range(lam_h_len)
        ]
    )
    lam_s_entries = (
        []
        if tab_name == SystemType.NONRECOVERABLE.value
        else [
            num_entry(
                entry_frame,
                f"λ1s",
                row=entry_row_offset + lam_h_len * 2,
                def_val=5e-4,
            )
        ]
    )
    mu_s_entries = (
        []
        if tab_name == SystemType.NONRECOVERABLE.value
        else [
            num_entry(
                entry_frame,
                f"μ1s",
                row=entry_row_offset + lam_h_len * 2 + 1,
                def_val=6e-4,
            )
        ]
    )

    plot_frame = ttk.Frame(tab, padding="20 22")
    plot_frame.grid(row=0, column=1, sticky="NSEW")
    tab.columnconfigure(1, weight=1)
    tab.rowconfigure(0, weight=1)

    canvas, ax = setup_plot(plot_frame)

    def draw_plot_by_entries(*_):
        try:
            error_label.config(text="")

            t_start = float(entry_t_start.get())
            t_end = float(entry_t_end.get())
            t_span = (t_start, t_end)
            t_eval = np.linspace(t_start, t_end, 200)
            lam_h = [float(entry.get()) for entry in lam_h_entries]

            sol = (
                solve_r9y_sys_nonrec(
                    t_span=t_span,
                    y0=[1.0] + [0.0] * 6,
                    lam=lam_h,
                    t_eval=t_eval,
                )
                if tab_name == SystemType.NONRECOVERABLE.value
                else solve_r9y_sys_rec(
                    t_span=t_span,
                    y0=[1.0] + [0.0] * 29,
                    lam_h=lam_h,
                    mu_h=[float(entry.get()) for entry in mu_h_entries],
                    lam_s=[float(entry.get()) for entry in lam_s_entries],
                    mu_s=[float(entry.get()) for entry in mu_s_entries],
                    t_eval=t_eval,
                )
            )
            green_states = (
                [0, 1, 2]
                if tab_name == SystemType.NONRECOVERABLE.value
                else [0, 1, 2, 3, 5, 11, 12, 13, 17, 21, 22, 23]
            )
            draw_plot(canvas, ax, sol, green_states=green_states)

        except ValueError:
            error_label.config(text="Value must be a number")

        except Exception as err:
            error_label.config(text=f"Error: {err}")

    for entry in [
        *lam_h_entries,
        *mu_h_entries,
        *lam_s_entries,
        *mu_s_entries,
        entry_t_start,
        entry_t_end,
    ]:
        entry.bind("<KeyRelease>", draw_plot_by_entries)

    draw_plot_by_entries()

    return tab
