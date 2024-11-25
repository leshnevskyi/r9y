import tkinter as tk
from tkinter import ttk

from r9y.core import SystemType
from r9y.ui import add_tab, root_geometry


def main() -> None:
    root = tk.Tk()
    root_geometry(root, 1200, 500)
    root.title("Failure Rate Impact on System State Probabilities")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    add_tab(notebook, SystemType.NONRECOVERABLE.value)
    add_tab(notebook, SystemType.RECOVERABLE.value)

    root.mainloop()
