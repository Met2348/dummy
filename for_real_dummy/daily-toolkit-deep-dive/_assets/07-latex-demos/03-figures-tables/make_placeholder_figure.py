"""Generate one placeholder training-curve PNG for the LaTeX figures+tables demo.

Not a real experiment result -- a synthetic decaying curve, used only to prove
that \\includegraphics{} + a real image file produces a real figure in the PDF.
Run with the repo .venv: .venv/Scripts/python.exe make_placeholder_figure.py
"""
import matplotlib
matplotlib.use("Agg")  # no display available in this shell, render straight to file
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
steps = np.arange(0, 200)
loss = 2.5 * np.exp(-steps / 60) + 0.05 * rng.standard_normal(len(steps)) + 0.3

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(steps, loss, color="#1f77b4", linewidth=1.5)
ax.set_xlabel("Training step")
ax.set_ylabel("DPO loss")
ax.set_title("Placeholder training curve")
fig.tight_layout()
fig.savefig("placeholder-figure.png", dpi=200)
print("saved placeholder-figure.png")
