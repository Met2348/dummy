# L02 · 科研生命周期七环 与 二维地图

## 1. 把"做研究"拆成七环

阶梯（L01）回答"自主到哪一级"，但同一级别的两个系统可能做的是**完全不同的环**。
所以第二个坐标轴是**科研生命周期**：

```
ideation → hypothesis → design → experiment → analysis → writeup → review
  创意       假设         设计       实验          分析       写作       评审
```

`systems.py` 的 `STAGES` 就是这七环。每个系统的 `automates` 字段记录它**真正自动化了哪几环**
（是子集，不是全有）。两个轴一交叉，就得到一张地图：**纵轴=自主性级别，横轴=生命周期环**。

## 2. 读这张地图

`python src/run.py --show map`：

```
  level     | ide hyp des exp ana wri rev | systems
  scientist |   4   0   0   4   4   4   2 | 4: AI Scientist v1/v2, AI-Researcher, NovelSeek
  analyst   |   1   1   7   6   6   1   0 | 7: co-scientist, Agent Lab, AlphaEvolve, DGM, ...
  tool      |   3   1   0   0   0   3   1 | 4: ResearchAgent, STORM, GPT-Researcher, AgentRxiv
```

格子里的数 = 该级别里有几个系统自动化了该环。几个能立刻读出来的结构：

- **Tool 行**集中在 `ide/wri`（创意、写作），`exp/ana` 全是 0——工具类不碰真实验。
- **Analyst 行**在 `des/exp/ana`（设计/实验/分析）最密——这正是"自动跑实验"的主场。
- **Scientist 行** `exp/ana/wri` 全满，还点亮了 `rev`（评审）——它们**自己写论文还自己评审**。
  这个 `rev` 列的 4，是 L03 整章的引线。

## 3. 覆盖广 ≠ 级别高

`Agent Laboratory` 自动化了 `des/exp/ana/wri` 四环，比某些 Scientist 还多；
但它的 idea 是人给的（`human_sets_problem=True`），所以稳稳停在 **Analyst**。

> **环数多是"勤奋"，问题自定才是"自主"。** 地图把这两件事分到了两个轴上，
> 就是为了不让"它做了好多事"冒充"它是个科学家"。

## 4. 动手

1. 改 `systems.py`：给 `Agent Laboratory` 的 `automates` 再加 `ideation`、把 `human_sets_problem`
   改成 `False`，重画地图——它升到 Scientist 了吗？这说明**升级的钥匙是哪个字段**？
2. 数一下：哪一环被自动化得最少？（提示：看 `hyp`/`rev` 列。）想想为什么"假设"和"评审"
   是最难真正自动化、也最容易造假的两环——这通向 9.3（假设）和 9.8（评审造假）。
