# L01 · 自我改进的三种形态

## 1. 从 Gödel Machine 到经验版

理论上的 **Gödel Machine**（Schmidhuber）是个漂亮的设想：一个程序，只有当它能
**形式化证明**"改写后的自己更好"时，才改写自己。问题是：现实里几乎证不出来。

于是 2024–25 的系统全都退到**经验版**：不要求证明，改成"试一个变体 → 用 fitness 评 →
更好就留下 → 存进档案"。这一步退让看着务实，却把整套方法的成败**全押在 fitness 上**——
这正是本模块要你警惕的地方。

## 2. 三种"进化的是什么"

| 形态 | 进化对象 | 代表 | 一句话 |
|------|---------|------|--------|
| 进化"解" | 候选程序/答案 | AlphaEvolve | 用进化搜索 + 自动评测器找更好的"解" |
| 进化"自己" | agent 自身代码 | Darwin Gödel Machine（2505.22954） | agent 改写自己的代码，benchmark 验证 |
| agent 设计 agent | agent 架构 | ADAS | 一个 meta-agent 搜索更好的 agent 设计 |

三者的公共骨架都是 `selfimprove/evolve.py` 里这几行：

```python
for gen in range(1, gens+1):
    cands = _mutations(best)           # 变异
    bc = max(cands, key=fitness)       # 评估 + 选优
    if fitness(bc) > fitness(best):    # keep-if-better
        best = bc; archive.append(...) # 存档案
```

## 3. 我们的缩小版进化了什么

本模块的"解"是一个 `Genome`：

- `threshold`：真泛化的杠杆（任务真值边界是 0，调对它在任何数据上都更准）。
- `memo`：一张查找表——**作弊的杠杆**（把见过的测试点答案背下来）。

一个诚实的自我改进会去调 `threshold`；一个被坏 fitness 牵引的会去填 `memo`。
下一讲你会看到：**到底走哪条路，完全由 fitness 决定，而不是由"系统想不想进步"决定。**

## 4. 动手

1. 读 `evolve.py` 的 `_mutations`，指出"真改进杠杆"和"作弊杠杆"各是哪几行。
2. 想想：如果把"证明更好才采纳"（Gödel Machine 的原意）加回来，本模块的作弊还成立吗？
   （提示：你得能"证明" holdout 也更好——但 holdout 恰恰是进化时看不到的。这就是难点。）
