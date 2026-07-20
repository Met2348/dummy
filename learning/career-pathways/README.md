# 9.10 career-pathways — 学术界/工业界/博后职业路径选择 (求职 → 转换 → 博后 → 教职早期)

> **Module 9「科研技能」第11个专题 · 阶段: 科研生涯周期(与"20环节项目周期"并列的新分支)**
> `research-direction-proposal`(9.0)教你怎么**在一个项目里**从零选研究方向; `research-life`(9.9)教你怎么在学术共同体里长久立足的软技能(审稿/导师沟通/署名伦理/可持续)。但两者都没回答一个跨越多个项目、以年为尺度反复出现的问题: **我这条职业路径该怎么走——学术界还是工业界?现在这条路走得不顺,要不要换?要不要读博后、读哪个组?拿到教职之后前三年怎么不把自己拖垮?** 本专题补上这一整条"科研生涯周期"的第一块拼图。

---

## 这个专题要解决的真问题

`research-direction-proposal` L1教的是"这个研究领域我该不该长期投入"(几年尺度的领域选择); `interview-prep`教的是"45分钟coding面试怎么不翻车"(面试技巧层)。两者中间空出一大块完全没人教的内容: **学术界和工业界求职的材料/流程本质不同, 什么时候该考虑从一条赛道转到另一条, 读不读博后、选哪个组, 以及教职拿到手之后前三年怎么不把自己耗干**——这些问题被默认成"走一步看一步"或者"导师/前辈说什么就是什么", 从来没有一套可核查的决策框架。

```
                没有本专题                              有本专题
求职           学术界/工业界材料混为一谈, 直接照搬        四阶段漏斗(学术,L1) vs research portfolio(工业,L2)
               会议报告slides去讲job talk                 各自独立的评审标准和材料结构
转换           被动等到deadline(非升即走)才慌乱转行      信号识别(市场/资源/价值观/技能)+主动叙事重塑(L3)
博后           "导师推荐哪个我就去哪个"                   要不要读的决策图 + 选人和组织环境的四维度(L4)
教职早期       什么都想做好, 结果什么都做不深              Boice式nihil nimus优先级排序(L5)
      ↓                                                    ↓
职业路径选择全靠直觉、从众和运气                     每个关键节点都有可核查、可复盘的决策依据
```

---

## 学习路径 (5 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-academic-job-market.md` | 学术界求职四阶段漏斗: 材料初筛→电话初面→校园面试邀请→校园面试, 和`research-presentation`会议报告的修辞目标区别 | research statement三段式草稿 |
| L2 | `lectures/L2-industry-research-job-market.md` | 工业界research岗求职: portfolio包装/和hiring manager谈/case study/reference, 和`interview-prep`coding面试互补而非重复 | research portfolio雏形 |
| L3 | `lectures/L3-academic-industry-transition.md` | 学界⇄业界转换: 什么信号该考虑转换、简历故事怎么重新包装、双向偏见怎么应对 | 转换信号自查清单 |
| L4 | `lectures/L4-choosing-a-postdoc.md` | 要不要读博后的决策框架 + 选组四维度(区别于`research-direction-proposal` L1的"选方向") | 博后邀约对比表 |
| L5 | `lectures/L5-early-faculty-survival.md` | 教职前三年生存: Boice式"nihil nimus"优先级排序(教学/经费/招生/服务性工作) | 前三年优先级排序草案 |

> 读法: L1→L5 顺序; L1-L2讲两条赛道各自的求职规则, L3讲两条赛道之间的转换, L4-L5讲学术界内部两个更细的阶段性决策(博后→教职早期)。每讲读完立刻去 `templates/career-path-scorecard.md` 或 notebook 上手一次。

## 动手 (1 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-score-a-career-path.ipynb` | 用 `src/career_path_scorer.py` 给你脑子里真实存在的2个候选职业路径打分对比(如"工业界research scientist" vs "学术界tenure-track"),不需要虚构材料 |

## 可复用模板 (`templates/`)

- `career-path-scorecard.md` — 候选职业路径四维打分卡(配L1-L5, 每个维度留2个以上候选的空白栏供手填分数+依据)

## 工具 (`src/`)

- `career_path_scorer.py` — 候选职业路径四维打分(技能匹配度/入行门槛与准备度/长期稳定性与成长空间/当前市场窗口期) + 完整性自检 + 排序对比(纯stdlib)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native 即可, 无需 WSL2。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 说清学术界job talk和普通会议报告(`research-presentation`)的修辞目标区别
- [ ] 说清工业界research岗求职中portfolio/hiring manager对话/case study/reference四个环节各自在评估什么, 以及和`interview-prep`coding面试的关系
- [ ] 识别至少两类转换信号(市场时机/资源自主权/价值观错位/技能边际收益)叠加出现, 而不是凭单一信号做转换决定
- [ ] 用选组四维度(带教风格/资源网络/往届去向/个人生活因素)对比至少两个博后邀约, 而不是只看方向匹配度
- [ ] 说出Boice"nihil nimus"优先级原则在教学/经费/招生/服务性工作四方面的具体应用
- [ ] 用四维打分框架给至少2个真实的候选职业路径打分并写出依据, 并理解总分高不等于终局决定

---

## 在 Module 9 中的位置

```
Module 9 科研技能 (现11个专题, 两套并列体系)

体系① 20环节项目周期 (单个研究项目从选方向到发表答辩的生命周期, 见 research-direction-proposal/L0)
  起点   9.0 research-direction-proposal L1-L2 (方向选择+可行性)
  地基   9.1 research-knowledge-mgmt
  输入   9.2 literature-mapping / 9.3 critical-reading-gap
  立项   9.0 research-direction-proposal L3-L4 (开题写作+答辩)
  执行   9.4 experiment-design / 9.5 experiment-ops-repro
  输出   9.6 research-figures / 9.7 paper-writing-submission / 9.8 research-presentation

体系② 科研生涯周期 (跨越多个项目、以年为尺度的个人发展轨迹, 20环节地图容不下, 新分支)
  科研生活 9.9 research-life (审稿/导师沟通/署名伦理/可持续 —— 贯穿全程的通用软技能)
  职业路径 9.10 career-pathways (求职/转换/博后/教职早期 —— 关键节点的路径选择)  ◄── 你在这里(新增)
  (9.11-9.17 计划中: 科研品牌与谈判/团队带教/团队运营/经费资源/诚信合规/学术共同体参与/开放科学, 尚未建成)
```
> 两套体系的区别: 体系①回答"这一个项目该怎么做", 体系②回答"我的科研生涯接下来往哪走"——不强行把9.10塞进20环节地图的编号序列, 它是并列的"生涯周期"分支, 不是项目周期的第21个环节。9.10紧接9.9之后, 因为"怎么在共同体里长久立足"(9.9, 贯穿全程)和"关键节点该往哪条路走"(9.10, 求职/转换/博后/教职)是同一个人生阶段里连续追问的两个问题。
>
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`(原9模块设计文档)、`docs/superpowers/specs/2026-07-14-research-career-lifecycle-design.md`(9.10-9.17科研生涯与共同体扩展设计文档, 本专题为该扩展的第一个专题)。
