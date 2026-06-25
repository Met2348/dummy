# 9.1 · 自主性阶梯与全景（Autonomy Ladder & Map）

> M9 系列的**坐标系**模块。学完你能把任意一个"AI 科研系统"准确归到
> **Tool / Analyst / Scientist** 哪一级，说清它在科研生命周期七环里覆盖了哪几环，
> 并一眼识破"营销级别 vs 证据级别"的落差。

## 这个模块教什么

整个 auto-research 领域在 2025–26 爆炸式增长，论文人人自称"AI Scientist"。
不建立坐标系，你会被话术带着走。本模块把综述
[A Survey on LLM-based Agents for Science (2505.13259)](https://arxiv.org/abs/2505.13259)
的 **Tool→Analyst→Scientist** 三级阶梯，做成一个**可跑、可测**的分级器：

- **按证据分级，不按自称分级**——`classifier.py` 只看「自动化了哪几环 / 问题谁定 / 有没有独立验证」，
  完全无视系统宣称的级别。
- **claimed vs evidenced 的差 = hype gap**，被算成一个数、画进地图。
- 地图还会**程序化地**揭示一个不舒服的事实（见下）。

## 跑起来

```powershell
# 全景地图 + 逐系统对照 + 领域阅读法洞见
python src/run.py --show all

# 只看地图 / 只看对照表
python src/run.py --show map
python src/run.py --show table

# 单个系统的归类与理由
python src/run.py --system "AI Scientist v2"
python src/run.py --system "ResearchAgent"   # 看 hype gap +1
```

官方验证（V0 文档静态 / V1 冒烟 / V2 测试）：

```powershell
python scripts/eric_3080ti_env_audit.py --runbook --tests `
  --modules auto-research-frontier/m9.1-autonomy-ladder-and-map `
  --json-out $env:TEMP/m9.json --md-out $env:TEMP/m9.md
```

## 地图读出的事实（不是我写死的，是算出来的）

跑 `--show all` 末尾会打印：

> 证据级别为 **Scientist** 的系统，结果**全部仅靠自评**（无独立验证）；
> 而经过独立验证的系统，证据级别最高只到 **Analyst**。
> → **自称越自主，结果越是自己说了算；真正可独立验证的，反而都还只是 Analyst。**

这条洞见被测试 `test_every_scientist_is_self_verified` /
`test_no_independently_verified_system_reaches_scientist` 锁死在数据上——
你改了目录的证据标注，要么这俩测试挂、要么洞见自动更新，不会留一句过时的漂亮话。

## 目录

```
m9.1-autonomy-ladder-and-map/
├── README.md                     ← 你在这
├── runbook.yaml                  ← V0/V1/V2 入口（harness 读）
├── lectures/
│   ├── 01-the-three-level-ladder.md       三级阶梯：怎么界定
│   ├── 02-lifecycle-seven-stages.md       生命周期七环 + 二维地图
│   └── 03-hype-vs-evidence-reading-the-field.md  领域阅读法（核心技能）
└── src/
    ├── run.py                    入口（argparse；sys.path 引导）
    ├── autonomy_map/
    │   ├── systems.py            七环定义 + 15 个真实系统的诚实证据标注
    │   ├── classifier.py         evidenced_level()：只用证据分级
    │   └── mapping.py            二维地图 + 对照表渲染
    └── tests/test_taxonomy.py    8 个测试：锁死分级规则 + 两条洞见
```

## Hands-on（轮到你）

`classifier.py` 的 `evidenced_level()` 里有一条 **★ 决策线**：
到底「几环、闭不闭环、问题谁定」才算升一级？现在的规则刻意简单。试试：

1. 把 Scientist 的门槛收紧（比如要求**连 review 也自动**才算），重跑 `--show map`——
   哪些系统掉级了？地图洞见变了吗？
2. 给目录加一个你读到的新系统（编辑 `systems.py` 的 `SYSTEM_CATALOG`），
   诚实标注它的证据，看它落在哪、有没有 hype gap。
3. 跑 `test_taxonomy.py`：你的改动让哪条洞见测试挂了？那正是「结论必须跟着数据走」的体感。

## 桥接

- **agent-foundations**：你注释过的 ReAct 就是 Tool→Analyst 的最小内核（单步工具调用 → 多步分析）。
- 下一站 **9.2**：把"什么是研究 agent"从分类，深入到"它内部由什么零件构成"。
