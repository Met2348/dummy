"""可视化对照实验结果."""
from __future__ import annotations

import math
from typing import Optional


def loss_curve_text(steps: list, curves: dict) -> str:
    """文本版 loss curve (ASCII)."""
    lines = ["loss curve (text):"]
    for v, ys in curves.items():
        bar = ""
        for y in ys[::4]:
            n = int(max(0, 11 - y) * 5)
            bar += "#" if n > 5 else "."
        lines.append(f"  {v}: {bar} (final {ys[-1]:.2f})")
    return "\n".join(lines)


def spider_chart_text(results: dict, metrics: list) -> str:
    """文本版 spider chart (列每 ckpt 各 metric)."""
    out = ["Spider (text approximation):"]
    out.append("var | " + " | ".join(f"{m:>10}" for m in metrics))
    out.append("-" * (5 + 13 * len(metrics)))
    for v in ["A", "B", "C", "D", "E"]:
        row = results.get(v, {})
        cells = [f"{row.get(m, 0):>10.3f}" if isinstance(
            row.get(m, 0), float) else f"{row.get(m, ''):>10}"
                  for m in metrics]
        out.append(f"  {v} | " + " | ".join(cells))
    return "\n".join(out)


def matplotlib_curves_script() -> str:
    return """
import matplotlib.pyplot as plt
import json

results = json.load(open("bench.csv"))

# 1. loss curves
plt.figure(figsize=(8,5))
for v in ["A","B","C","D","E"]:
    losses = load_train_losses(f"ckpt_{v}.pt")
    plt.plot(losses, label=v)
plt.xlabel("step"); plt.ylabel("loss"); plt.legend()
plt.savefig("report/curve.png", dpi=120)

# 2. bar per metric
for m in ["hellaswag","tinymmlu","gsm8k"]:
    plt.figure(figsize=(4,4))
    plt.bar(["A","B","C","D","E"], [results[v][m] for v in "ABCDE"])
    plt.title(m); plt.ylabel("acc")
    plt.savefig(f"report/{m}.png", dpi=120)

# 3. spider chart
import numpy as np
N = 6
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
fig, ax = plt.subplots(subplot_kw={"polar": True})
for v in "ABCDE":
    vals = [results[v][m] for m in METRICS]
    vals.append(vals[0])  # close loop
    ax.plot(angles, vals, label=v)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(METRICS)
ax.legend()
plt.savefig("report/spider.png", dpi=120)

# 4. NIAH heatmap (for ckpt D vs E)
ctx_lens = [1024, 2048, 4096, 8192]
depths = [10, 30, 50, 70, 90]
for v in ["D", "E"]:
    acc = run_niah_grid(ckpt=f"ckpt_{v}", lens=ctx_lens, depths=depths)
    plt.figure(figsize=(6,4))
    plt.imshow(acc, cmap="viridis", vmin=0, vmax=1)
    plt.xticks(range(len(ctx_lens)), ctx_lens)
    plt.yticks(range(len(depths)), depths)
    plt.colorbar(label="acc")
    plt.title(f"NIAH ckpt {v}")
    plt.savefig(f"report/niah_{v}.png", dpi=120)
"""


if __name__ == "__main__":
    from bench_matrix import EXPECTED, METRICS, make_loss_curve_data
    print(loss_curve_text(**make_loss_curve_data()))
    print()
    print(spider_chart_text(EXPECTED, METRICS))
    print()
    print("=== matplotlib script (save to plots.py) ===")
    print(matplotlib_curves_script()[:500] + "...")
