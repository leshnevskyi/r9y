import tkinter as tk
from tkinter import ttk

import numpy as np

from r9y.core import solve_r9y_sys_nonrec
from r9y.ui import draw_plot, num_entry, root_geometry, setup_plot


def main() -> None:
    root = tk.Tk()
    root_geometry(root, 1000, 500)
    root.title("Failure Rate Impact on System State Probabilities")
    root.columnconfigure(0, weight=1)

    entry_frame = ttk.Frame(root, padding="20 30")
    entry_frame.grid(row=0, column=0, sticky="NS")
    entry_frame.columnconfigure(1, weight=1)

    entry_lam1 = num_entry(entry_frame, "λ1", row=0, def_val=5e-4)
    entry_lam2 = num_entry(entry_frame, "λ2", row=1, def_val=4e-4)
    entry_lam3 = num_entry(entry_frame, "λ3", row=2, def_val=3e-4)
    entry_t_start = num_entry(entry_frame, "t1", row=3, def_val=0)
    entry_t_end = num_entry(entry_frame, "t2", row=4, def_val=3000)

    error_label = ttk.Label(entry_frame, text="", foreground="red")
    error_label.grid(column=0, row=5, columnspan=2)

    plot_frame = ttk.Frame(root)
    plot_frame.grid(row=0, column=1, sticky="NSEW")
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    canvas, ax = setup_plot(plot_frame)

    def draw_plot_by_entries(*_):
        try:
            error_label.config(text="")
            lam1 = float(entry_lam1.get())
            lam2 = float(entry_lam2.get())
            lam3 = float(entry_lam3.get())
            t_start = float(entry_t_start.get())
            t_end = float(entry_t_end.get())
            sol = solve_r9y_sys_nonrec(
                t_span=(t_start, t_end),
                y0=[1.0] + [0.0] * 6,
                lam=(lam1, lam2, lam3),
                t_eval=np.linspace(t_start, t_end, 200),
            )
            draw_plot(canvas, ax, sol)
        except ValueError:
            error_label.config(text="Value must be a number")
        except Exception as err:
            error_label.config(text=f"Error: {err}")

    for entry in [entry_lam1, entry_lam2, entry_lam3, entry_t_start, entry_t_end]:
        entry.bind("<KeyRelease>", draw_plot_by_entries)

    draw_plot_by_entries()
    root.mainloop()
