# L01 · novelty ≠ feasibility（创意的照妖镜）

## 1. 为什么"创意"是 litmus test

很多人对 auto-research 的第一个期待是"让 AI 帮我想 idea"。
[Si, Hashimoto, Yang (2409.04109)](https://arxiv.org/abs/2409.04109) 做了大规模人类对照，
发现 LLM 生成的研究 idea **被专家评得比人类更新颖**——一个很提气的结果。

但同一批作者的追问 [The Ideation-Execution Gap (2506.20803)](https://arxiv.org/abs/2506.20803)
把那些点子**真去执行**，结论翻转：AI 点子的实际表现**反而不如**人类点子。

> **新颖是一种事前的感觉，可行是一种事后的事实。** ideation 阶段能看到的只有前者。

## 2. 我们用真训练把这一刀复现出来

本模块每个 idea 不是一句空话，而是一组**真训练配置**（`task.py` 的 numpy 逻辑回归 GD）。
它的 `feasibility` = 真跑出来的准确率，评委**看不到**。跑 `python src/run.py`：

```
idea           src   novelty  judge   elo  feasibility  feas#
exotic-init    self     1.00   1.30  1322       0.795      6   ← 评委冠军，真实垫底
fancy-sched    self     1.00   1.30  ...        0.818      5
...
more-steps     other    0.00   0.00   ...       0.932      1   ← 真实冠军，评委倒数
```

评委冠军 `exotic-init`（文案堆满"新颖/exotic/重参数化/突破"）真实可行性**全场第 6/6**；
真正最好的 `more-steps`（文案平淡到"多训练一会儿"）被评委排到倒数。

## 3. 为什么会这样

`judge.py` 的 `novelty()` 只做一件事：数文案里的热词。

```python
hits = sum(1 for w in BUZZWORDS if w in idea.text)   # 新颖/exotic/cutting-edge...
```

热词和真实价值**正交**。一个把学习率"大胆激进"拉到 50 的点子（真跑发散），
文案听着就是比"加个动量"带劲。评委被语言骗了，因为它在事前，没有执行的地面真值可依。

## 4. 动手

1. 跑 `python src/run.py`，盯住 `novelty` 列和 `feasibility` 列——它们几乎反着来。
   这种"评分轴与价值轴正交甚至反向"的局面，就是 ideation gap 的几何形状。
2. 把 `ideas.py` 里 `exotic-init` 的**文案**改平淡（去掉所有热词），但**配置不动**。
   评委还把它排第一吗？这说明评委排序依赖的到底是 idea 的什么——内容还是包装？
