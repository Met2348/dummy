# M9.5 · 端到端 AI Scientist（诚实缩小版）⭐⭐⭐⭐⭐

> auto-research 教学系列（见 [`../CURRICULUM.md`](../CURRICULUM.md)）的**系列脊梁** —— 在 **RTX 3080 Ti 上真训练**跑通
> The AI Scientist 的五阶段闭环：**ideation → experiment → analysis → writeup → review**。
> 指标全部来自真实训练，**不是 mock 分数**；并把两个最重要的批判点焊进代码。

## 这个模块教什么

1. **把"AI 自己做研究"拆成五阶段并真跑一遍**——你会亲手看到 idea 怎么变成实验、实验怎么变成报告。
2. **ideation-execution gap**（[2506.20803](https://arxiv.org/abs/2506.20803)）：新点子可能真的不涨点。模块诚实返回 `refuted`/`inconclusive`，不美化。
3. **grading-its-own-homework**（[2502.14297](https://arxiv.org/abs/2502.14297) / [2509.08713](https://arxiv.org/abs/2509.08713)）：自动评审可被刷——模块用代码把这个缺陷做出来给你看。

## 五阶段映射（src/mini_ai_scientist/）

| 阶段 | 文件 | 真做了什么 |
|------|------|-----------|
| 1 Ideation | `ideation.py` | 模板 idea 库（可插真 LLM），每个带"事前自评 novelty" |
| 2 Experiment | `experiment.py` | **真训练** torch MLP on make_moons；任务种子与模型种子分离；确定性可复现 |
| 3 Analysis | `analysis.py` | 跨种子 mean±std 对照 + `verdict()` 诚实判定 + matplotlib 画图 |
| 4 Writeup | `writeup.py` | 自动写 1 页 markdown 报告；**数字源自真实指标，test 锁死不幻觉** |
| 5 Review | `review.py` | mock 评审，**故意可被刷**（只看宣称效果大小），演示 grading-own-homework |
| 编排 | `pipeline.py` / `run.py` | 串起五阶段 + 打印诚实光谱 + 两个教学演示 |

## 诚实光谱（真实示例输出，CPU/seeds=3/epochs=60）

```
  idea           self_nov   base→treat        Δ       verdict  review
  add-depth        0.30  0.863→0.987     +0.123     supported   8.2
  go-deeper        0.55  0.987→1.000     +0.013     supported   6.0
  crank-lr         0.40  0.987→0.843     -0.143       refuted   8.6   ← 被推翻，评审分却最高！
  widen            0.50  0.920→1.000     +0.080     supported   7.3
  relu-vs-tanh     0.45  0.980→0.987     +0.007  inconclusive   5.9
```

> **看出门道了吗**：被推翻的 `crank-lr`（学习率飙到 2.0，把精度从 0.987 打到 0.843）拿了**全场最高评审分 8.6**——
> 因为 mock 评审只奖励 |Δ| 的大小，不看正负与真伪。这就是"自己给自己改作业"最有力的现场实证。

## 运行验证（Runbook）

> 本模块 2 个直跑入口已登记在 [`runbook.yaml`](runbook.yaml)，在 ERIC-3080Ti（3080 Ti 16GB）V0+V1 验证通过；device 默认 `auto→cuda`，真用 GPU 训练（秒级）。

```powershell
# 一键复验（注意 --modules 传完整嵌套路径）
python scripts/eric_3080ti_env_audit.py --runbook --modules auto-research-frontier/m9.5-end-to-end-ai-scientist

# 直跑：单 idea / 全光谱
python learning/auto-research-frontier/m9.5-end-to-end-ai-scientist/src/run.py --idea add-depth
python learning/auto-research-frontier/m9.5-end-to-end-ai-scientist/src/run.py --idea all --device cpu
```

**测试（V2）**：7 个"诚实性"测试（真 exec / 真阳真阴 / 不幻觉数字 / 评审可被刷）：

```powershell
python -m pytest learning/auto-research-frontier/m9.5-end-to-end-ai-scientist/src/tests/ -q
# 或经 harness：python scripts/eric_3080ti_env_audit.py --modules auto-research-frontier/m9.5-end-to-end-ai-scientist --tests
```

## ✍️ 留给你打磨的地方（5–10 行，关乎科研诚信）

`analysis.py` 的 **`verdict()`** 决定了"什么算真效果"——这是整个模块最核心的科研判断。
默认用"Δ 是否超过 combined_std 与阈值"的粗略启发式。**你来把它换成更严谨的判据**：

- 对两组 `test_accs` 做 **Welch t-检验**（`scipy.stats.ttest_ind(equal_var=False)`），或 bootstrap 置信区间；
- 想清楚：阈值多大才**既不放过真效果、又不被噪声忽悠**？

这一处的判据松一点，模块就会开始自欺——正是 AI Scientist 翻车的地方。改完跑 `--tests` 看是否仍守得住 `crank-lr != supported`。

## 绑定文献（见 [`../papers/INDEX.md`](../papers/INDEX.md)）

- 主线：[The AI Scientist v1](https://arxiv.org/abs/2408.06292) / [v2 树搜索](https://arxiv.org/abs/2504.08066)（五阶段、实验经理、自评审）
- 批判：[Ideation-Execution Gap](https://arxiv.org/abs/2506.20803) · [Wishful Thinking/ARI](https://arxiv.org/abs/2502.14297) · [Hidden Pitfalls](https://arxiv.org/abs/2509.08713)

## 与已学模块的桥

| 这里用到 | 来自你学过的 |
|----------|------------|
| ReAct/规划的放大 | agent-foundations |
| 真训练件（MLP/优化器/指标） | pretraining-recipe · rl-foundations · small-model-graduation |
| "评审=LLM judge"的可被刷 | llm-judge-arena |
| verdict 的"效果是否真"判据 | agent-code-eval（真 exec 评测同源） |

## 毕业 checklist

- [x] 五阶段闭环真跑通（idea→实验→报告→评审）
- [x] 真训练（3080Ti torch on make_moons），确定性可复现
- [x] 诚实光谱（supported/refuted/inconclusive 都出现）
- [x] 不幻觉数字（test 锁死"报告数字==实验数字"）
- [x] grading-own-homework 可被刷（test + 光谱现场实证）
- [x] runbook.yaml（V0+V1）+ 7 个 V2 测试
- [ ] **接下 9.6**：把 `verdict()` 升级成真 t-检验；把 `review` 接真 LLM；给 mini scientist 加复现守卫（→ 9.8 红队）

## 一句话总结

> 这是你对"research 被 Agent 接管到什么程度"的**亲手答案**：五阶段能跑通，但**真做出来的远比想出来的少**，而且**自评分根本不能信**。
