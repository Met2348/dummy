# L08 · 可视化与报告

> 10 slides | 30 min ⭐⭐⭐

## Slide 1 · 4 张图

```
fig1: loss curve overlay (5 ckpt)
fig2: bar chart per metric
fig3: spider chart (5 metric × 5 ckpt)
fig4: NIAH heatmap (ctx × depth) for ckpt D 与 E
```

## Slide 2 · loss curve

```python
import matplotlib.pyplot as plt
for v in ["A","B","C","D","E"]:
    losses = load_losses(f"ckpt_{v}.pt")
    plt.plot(losses, label=v)
plt.legend(); plt.xlabel("step"); plt.ylabel("loss")
plt.savefig("loss_curve.png")
```

## Slide 3 · bar chart

```python
metrics = ["HellaSwag", "MMLU", "GSM8K"]
for m in metrics:
    plt.bar(["A","B","C","D","E"], [r[v][m] for v in ...])
    plt.title(m)
    plt.savefig(f"{m}.png")
```

## Slide 4 · spider/radar

```python
import numpy as np
N = 6  # 6 metrics
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
fig, ax = plt.subplots(subplot_kw={"polar": True})
for v in VARIANTS:
    ax.plot(angles, results[v], label=v)
ax.legend()
```

5 条曲线, 一眼看 E 远胜.

## Slide 5 · NIAH heatmap

```python
ctx = [1k, 2k, 4k, 8k]
depth = [10, 30, 50, 70, 90]
acc = np.zeros((5, 4))
for i, d in enumerate(depth):
    for j, c in enumerate(ctx):
        acc[i,j] = niah(ckpt, c, d)
plt.imshow(acc, cmap="viridis")
plt.colorbar()
```

## Slide 6 · report markdown 自动生成

```python
def make_report(results):
    md = ["# Report", "## Setup", "[table]", "## Results", ...]
    Path("report.md").write_text("\n".join(md))
```

## Slide 7 · 多 ckpt 同 prompt 输出

```python
prompt = "The quick brown fox"
for v in ["A","B","C","E"]:
    m = load(f"ckpt_{v}.pt")
    out = generate(m, prompt, max_new_tokens=50)
    print(f"--- {v} ---\n{out}\n")
```

## Slide 8 · 文档结构

```
report/
  curve.png
  hellaswag.png mmlu.png ...
  spider.png
  niah_heatmap.png
  generations.md
  report.md
  bench.csv
```

## Slide 9 · 提交时所有图齐

```
git add report/
git commit -m "report: 五部曲对照 final"
git tag 造改-graduation
```

## Slide 10 · 总结

```
图表 + 数据 + 文字三件齐
报告本身是 capstone 的成品
```

## 参考
- matplotlib gallery
- seaborn (heatmap, radar)
