# 9.4 experiment-design — 实验设计: 把 idea 做成站得住的结果

> **Module 9「科研技能」· 阶段: 执行 (execution)** · 本专题是整个 Module 9 对博0**最核心**的一块。
> 从 9.3 拿到一张 idea 卡之后, 怎么把它变成一套**严谨、可信、经得起审稿人追问**的实验? 这就是实验设计。

---

## 这个专题要解决的真问题

你会复现 48 个模型 (工程), 但「复现」和「做实验验证一个假设」是两回事:

- **复现**: 把别人的方法跑出别人的数。目标是「一致」。
- **做实验**: 用受控对照检验**你自己的**一个假设, 产出前人没有的新结论。目标是「可信地回答一个问题」。

新手做实验的典型翻车: 没写假设就乱跑 → baseline 偷偷不公平 → 只看主效应错过交互 → 单种子下结论。结果是一堆「看起来有提升」但**经不起一问**、复现不出来的数。

> **会做实验 = 你的每个结论都能扛住这四问**: ①假设是什么、能被什么推翻 (L1)? ②最便宜怎么先验证 (L2)? ③对照公平吗、差距会不会是混淆变量 (L3)? ④是哪个组件、在什么条件下起作用 (L4)? ⑤差距真吗、大吗、几个种子 (L5)?

```
   idea 卡 (9.3 来)
        │
   L1 可证伪假设 → L2 最小验证(MVE) → L3 公平对照 → L4 消融矩阵+交互 → L5 方差+显著性
        │
   一个站得住的结论 (交棒 9.5 可靠执行 / 9.7 写成论文)
```

---

## 学习路径 (5 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-falsifiable-hypothesis.md` | 把模糊 idea 变成实验能杀死的可证伪假设 (含 H0/H1) | 假设卡 |
| L2 | `lectures/L2-minimal-viable-experiment.md` | 最小验证实验: 一周拿 go/no-go, 别在错方向烧三个月 | MVE 方案 |
| L3 | `lectures/L3-baselines-and-controls.md` | 公平 baseline + 控混淆变量, 让差距经得起追问 | 对照设计 |
| L4 | `lectures/L4-ablation-matrix.md` | 消融矩阵 (全因子) + **交互效应** (新手盲区) | 消融矩阵 |
| L5 | `lectures/L5-variance-and-significance.md` | 多种子/误差棒/置信区间/p 值/效应量, 戳穿单种子自欺 | 带统计的结论 |

## 动手 (2 个 notebook — 在确定性模拟器上把实验设计循环走通)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-design-an-ablation.ipynb` | 用 `src/experiment.py` 跑 method×noise **全因子矩阵**, 亲手算**交互效应**, 并对比「如果只用 OFAT 会得出什么错误结论」 |
| `notebooks/N2-variance-and-significance.ipynb` | 用 `src/stats.py` 画 error bar、算 bootstrap CI / p 值 / Cohen's d, 看「同一对方法高噪声下显著、低噪声下不显著」, 再亲手做一次「挑种子作弊」看差距怎么被吹大 |

> 模拟器埋了一个**真实的交互效应** (Robust-DPO 的好处随噪声增大) + 真实的种子方差。
> 你用正确的实验设计**把它检测出来** —— 整个例子对应你的 `learning/dpo-family` 复现。

## 可复用模板 (`templates/`)

- `hypothesis-card.md` — 可证伪假设卡 (H0/H1 / 变量 / 幅度 / 什么结果推翻它)
- `experiment-protocol.md` — MVE 实验方案 (砍了什么 / go-no-go 阈值)
- `ablation-matrix.md` — 消融矩阵设计表 (因子 / 水平 / 关键对比 / 预期形状)

## 工具 (`src/`)

- `experiment.py` — 确定性模拟实验引擎 (内埋交互效应 + 种子方差, 跨进程可复现)
- `stats.py` — mean±std / bootstrap CI / Welch t 检验 / Cohen's d / 对比裁决

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native。两个 notebook 零算力、秒级、**跨天可复现** (确定性模拟器)。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 把任意 idea 写成可证伪假设, 并说出「什么结果会推翻它」
- [ ] 设计一个 ≤10 次运行的 MVE, 跑之前定好 go/no-go 阈值
- [ ] 列出并控制一个对比里的混淆变量, 把 baseline 调到最强再声称超越
- [ ] 排一个全因子消融矩阵, 算出并解读交互效应
- [ ] 报 mean±std + error bar + p 值 + 效应量, 判断该跑几个种子
- [ ] 解释 p 值的三个误区, 不 p-hack

---

## 在 Module 9 中的位置

```
Module 9 科研技能
  地基   9.1 research-knowledge-mgmt   ✅
  输入   9.2 literature-mapping        ✅
        9.3 critical-reading-gap       ✅ (idea 卡从这来)
  执行   9.4 experiment-design         ◄── 你在这里 (Module 9 核心)
        9.5 experiment-ops-repro       (可靠执行 9.4 设计的实验)
  输出   9.6 / 9.7 / 9.8
  ...
```
> 9.3→9.4→9.5 是研究的执行主轴: 9.3 找到值得做的 idea, **9.4 设计出能可信回答它的实验**, 9.5 把实验可靠地跑出来并留痕。
>
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`
