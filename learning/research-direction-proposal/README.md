# 9.0 research-direction-proposal — 研究方向选择与开题 (从零选方向 → 立项)

> **Module 9「科研技能」第10个专题 · 阶段: 起点 + 立项检查点**
> 老师在"科研技能"9个专题之外,进一步指出用户需要练习"怎么发掘/产生一个研究idea",并要求参照一套"20环节科研全流程方法论"系统学习。调研确认仓库里9个专题从来没有被显式串成一张编号地图,而且它们全部假设"研究方向已经给定"——本专题补上地图本身(L0),以及地图上此前完全空白的4个环节:方向从零选择、项目级可行性评估、开题报告写作、开题答辩(L1-L4)。

---

## 这个专题要解决的真问题

`literature-mapping`的开篇例子是"导师扔给你一个方向,去看看这块"——教你怎么摸清一个**已经确定**的方向。`critical-reading-gap`的idea生成同样是gap驱动的:你必须先在某个方向里读够论文,才能长出idea。这意味着一个真实的场景完全没人教:**入学半年,导师说"你自己想个方向",该怎么办?** 以及更晚一点:**idea已经过了三筛,该怎么把它正式变成一个被认可的研究项目?**

```
                没有本专题                            有本专题
方向: 导师给什么做什么, 从没独立选过        方向: 四维打分框架, 能自己列候选并比较
可行性: 只看单个idea能不能跑              可行性: 项目/多年尺度的资源-时间线判断
立项: 口头汇报, 没人追问细节             立项: 开题报告七块骨架 + 答辩追问预判
      ↓                                        ↓
真到独立带方向那天才第一次练              提前把这条肌肉练熟
```

---

## L0:20环节全流程地图

`lectures/L0-research-lifecycle-map.md` 是这个专题真正的起点:把9个已有专题的16个环节,和本专题新写的4个环节,按科研生命周期顺序重组成一张编号20的地图。**建议先读L0,再读L1-L4**——L0会告诉你为什么这4讲长在这两个此前空白的位置上。

## 学习路径 (5 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L0 | `lectures/L0-research-lifecycle-map.md` | 20环节全流程地图: 哪16个已讲过, 哪4个是本专题新写 | 定位"我现在在哪个环节" |
| L1 | `lectures/L1-choosing-a-research-direction.md` | 从零选研究方向: 兴趣/实验室积累/资助趋势/职业规划四维打分 | 候选方向打分卡 |
| L2 | `lectures/L2-project-level-feasibility.md` | 项目级可行性评估: 区别于单个idea的tractability, 是几年尺度的资源/时间线判断 | 可行性评估表 |
| L3 | `lectures/L3-writing-the-proposal.md` | 开题报告写作: 把过三筛的idea展开成七块骨架的正式文档 | 开题报告草稿 |
| L4 | `lectures/L4-proposal-defense.md` | 开题答辩: 预判质询维度, 诚实亮出风险预案而非回避 | 答辩追问预判清单 |

> 读法: L0→L4 顺序; L0是地图,L1-L4是地图上4块新拼图的具体展开。每讲读完立刻去对应模板/notebook上手一次。

## 动手 (2 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-score-a-direction.ipynb` | 用 `src/direction_scorer.py` 给你脑子里真实存在的候选方向打分对比,不需要虚构材料 |
| `notebooks/N2-audit-your-proposal.ipynb` | 用 `src/proposal_audit.py` 自检一版开题报告草稿的完整性,并把薄弱环节转成答辩追问预判 |

## 可复用模板 (`templates/`)

- `direction-scorecard.md` — 候选方向四维打分卡(配L1)
- `feasibility-worksheet.md` — 项目级可行性评估表(配L2)
- `proposal-outline.md` — 开题报告七块骨架大纲(配L3)
- `proposal-defense-prep.md` — 开题答辩准备清单(配L4)

## 工具 (`src/`)

- `direction_scorer.py` — 候选方向四维打分 + 完整性自检 + 排序对比(纯stdlib)
- `proposal_audit.py` — 开题报告骨架自检 + 把薄弱环节转成答辩追问预判(纯stdlib)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native 即可, 无需 WSL2。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 说清"选方向"(几年尺度)和"找idea"(几周尺度)的本质区别
- [ ] 用四维框架给至少2个候选方向打分并写出依据,而不是空口拍板
- [ ] 说清"项目级可行性"和"单个idea是否tractable"是两个不同粒度的判断
- [ ] 写出一版包含全部七块骨架的开题报告草稿,并用工具自检通过
- [ ] 预判一场开题答辩最可能被追问的3-5个问题,并准备好诚实的风险预案
- [ ] 能画出20环节地图,说出自己现在在哪个环节

---

## 在 Module 9 中的位置

```
Module 9 科研技能 (按研究项目生命周期, 现为10个专题)
  起点   9.0 research-direction-proposal L1-L2 (方向选择+可行性)  ◄── 你在这里(新增①)
  地基   9.1 research-knowledge-mgmt
  输入   9.2 literature-mapping
        9.3 critical-reading-gap
  立项   9.0 research-direction-proposal L3-L4 (开题写作+答辩)   ◄── 同一专题, 第二次出现(新增②)
  执行   9.4 experiment-design
        9.5 experiment-ops-repro
  输出   9.6 research-figures
        9.7 paper-writing-submission
        9.8 research-presentation
  科研生活 9.9 research-life
```
> 本专题编号"9.0"、但在生命周期图里出现两次: L1-L2在最前端(比9.1更早,因为方向选择先于知识管理基础设施搭建),L3-L4在9.3之后、9.4之前(因为开题必须在idea过三筛之后、实验正式开始之前)。这不是编号错误,是这个专题本身就横跨两个不相邻的插入点。
>
> 完整20环节地图见 `lectures/L0-research-lifecycle-map.md`。设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`(原9模块设计文档,本专题为该设计的后续补全,未修改原文档)。
