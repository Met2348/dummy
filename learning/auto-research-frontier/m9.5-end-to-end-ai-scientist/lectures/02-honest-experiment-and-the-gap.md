# L02 · 真实验逼出的"想 ≠ 做得出"（ideation-execution gap）

## 1. 核心论文：The Ideation-Execution Gap

[Si, Hashimoto, Yang (Stanford), 2025](https://arxiv.org/abs/2506.20803) 是 [Can LLMs Generate Novel Research Ideas?](https://arxiv.org/abs/2409.04109) 的致命追问：

- 前作发现：LLM 的点子被专家评为**比人类更新颖**。
- 后作把那些点子**真去执行**：结果 AI 点子的实际表现**反而不如**人类点子。

**结论**：新颖 ≠ 可行。ideation 的高分是一种幻觉，只有 execution 才能证伪。

## 2. 在我们的模块里亲手看到它

跑 `--idea all`，看 `self_nov`（事前自评）与 `verdict`（事后真做）的错位：

```
  idea           self_nov   base→treat        Δ       verdict
  crank-lr         0.40  0.987→0.843     -0.143       refuted    ← 听着合理，做出来更差
  relu-vs-tanh     0.45  0.980→0.987     +0.007  inconclusive    ← 听着该赢，其实没差
  add-depth        0.30  0.863→0.987     +0.123     supported    ← 自评"不新"，反而真涨点
```

- `crank-lr`："学习率调大就训得更好"听起来很合理（自评 novelty 0.40）。真做：lr=2.0 把精度从 0.987 砸到 0.843，**refuted**。
- `add-depth`：自评只有 0.30（"加层有什么新的"），却是全场最大真涨点。**novelty 与真实价值脱钩**。

> 这正是 gap 的本质：**评估一个 idea 的唯一可靠方式是去执行它。** 任何只在 ideation 阶段打分的系统都在自欺。

## 3. `verdict()`：什么算"真有效果"

`analysis.py` 的判定规则：效果要**同时超过噪声（combined_std）和最小实际意义（threshold）**才算 supported；比 baseline 显著更差则 refuted；否则 inconclusive。

为什么不是"Δ>0 就算赢"？因为：

- 单跑一次的 Δ>0 可能纯是种子运气（所以要跨种子 std）；
- 0.001 的提升即使"真"，也没有实际意义（所以要 threshold）。

这条规则就是科研诚信的具体化。**它被你调松一点，模块就开始把噪声当成果。**（README 留了把它升级成 t-检验的练习。）

## 4. 动手

1. 改 `experiment.py` 的 `noise`（make_moons 噪声）到 0.4，再跑 `--idea all`：哪些 verdict 翻了？为什么 std 变大后更多 idea 变 inconclusive？
2. 给 IDEA_BANK 加一个你自己的假设（如 `dropout` 有没有用），写出 baseline/treatment override，跑出它的诚实 verdict。
