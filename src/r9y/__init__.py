import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib import pyplot as plt

from r9y.core import solve_r9y_sys_nonrec
from r9y.ui import frame_entry, root_geometry


def main() -> None:
    root = tk.Tk()
    root_geometry(root, 500, 250)
    root.title("Failure Rate Impact on System State Probabilities")
    root.columnconfigure(0, weight=1)

    frame = ttk.Frame(root, padding="50")
    frame.columnconfigure(1, weight=1)
    frame.grid(sticky="EW")

    entry_lam1 = frame_entry(frame, "λ1:", row=0, def_val=5e-4)
    entry_lam2 = frame_entry(frame, "λ2:", row=1, def_val=4e-4)
    entry_lam3 = frame_entry(frame, "λ3:", row=2, def_val=3e-4)
    entry_t_start = frame_entry(frame, "t1:", row=3, def_val=0)
    entry_t_end = frame_entry(frame, "t2:", row=4, def_val=2500)

    error_label = ttk.Label(frame, text="", foreground="red")
    error_label.grid(column=0, row=5, columnspan=2)

    def show_plot() -> None:
        try:
            lam1 = float(entry_lam1.get())
            lam2 = float(entry_lam2.get())
            lam3 = float(entry_lam3.get())
            t_start = float(entry_t_start.get())
            t_end = float(entry_t_end.get())
            t_span = (t_start, t_end)
            lam = (lam1, lam2, lam3)

            sol = solve_r9y_sys_nonrec(
                t_span=t_span,
                y0=[1.0] + [0.0] * 6,
                lam=lam,
                t_eval=np.linspace(t_start, t_end, 200),
            )

            t = sol.t
            P = sol.y
            plt.figure(figsize=(10, 6))

            for i in range(P.shape[0]):
                plt.plot(t, P[i], label=f"P{i+1}(t)")

            plt.title("State Probabilities")
            plt.xlabel("Time, t")
            plt.ylabel("State Probability, P(t)")
            plt.legend()
            plt.grid()
            plt.show()

        except ValueError:
            error_label.config(text="Input values must be numbers")

        except Exception as e:
            error_label.config(text=f"Error: {e}")

    calc_button = ttk.Button(frame, text="Plot", command=show_plot)
    calc_button.grid(column=0, row=6, columnspan=2)

    root.mainloop()
