# 9.7 paper-writing-submission — 科研写作 + 投稿 (升级复用 how_to_write_a_paper)

> **Module 9「科研技能」· 阶段: 输出 (output)**
> 把整个研究**写成论文并投出去**。**写作方法论复用你已有的 `how_to_write_a_paper` 技能包** (叙事先行 + 逐部分细则); 本专题补你完全没有的下半场: **装配 / 投稿流程 / 同行评审 / rebuttal / 录用拒稿应对**。

---

## 这个专题要解决的真问题

你已经有 `how_to_write_a_paper` (怎么写好每一节)。但「会写」和「能让论文录用」之间, 还隔着:

- 怎么把 9.1-9.6 的零散产物**装配**成一篇有闭合证据链的论文?
- 投哪个会、怎么卡 deadline、双盲怎么匿名、OpenReview 怎么用?
- 收到审稿意见 (大概率有 negative) 怎么读、怎么写一份**能翻盘的 rebuttal**?
- 被拒了 (大概率) 怎么把意见当燃料、高效改投?

> **论文的命运常常不在写作时、而在 rebuttal 时决定。** 这些是博0 几乎零准备、却每篇都要走一遍的技能。本专题给你投稿循环的完整地图 + 两个把流程自动化的工具。

L1-L4 是投稿循环的主干, 但论文的生命周期不止"投出去"这一段——还有几件同样容易被博0忽略、L1-L4 没展开的现实事:

- 一个实验没跑出预期效果, 这个**负结果**到底该悄悄归入 file drawer, 还是值得单独写一篇?
- 手头的大项目, 该**攒成一篇大论文**还是**切成几篇小论文**? 切太碎和攒太久分别踩什么坑?
- 会议和期刊(TACL/TMLR)到底该怎么选? dual submission 的红线具体划在哪?
- 论文 accept 之后, camera-ready 那几天到底要按什么顺序做完哪些事?
- 论文发出去之后, 长期的影响力和 follow-up 工作该怎么规划, 而不是灌水引用了事?
- 英语不是母语, 写作里那些系统性的"中式学术英语"陷阱要怎么识别、怎么用工具又不丢掉自己的论证逻辑?

本专题用 L5-L10 补上这几件事。

```
   研究(9.1-9.6) → L1装配(+技能包写细则) → L2投稿流程 → L3评审+rebuttal → L4录用/拒稿
                                                          ↑ 论文命运常在这决定
                                                          ↓
        L5负结果分级 · L6论文切分策略 · L7venue深层选择 · L8camera-ready · L9长期影响力经营 · L10非母语写作
                        (贯穿论文全生命周期、L1-L4 未展开的六件现实事)
```

---

## 学习路径 (10 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-from-research-to-paper.md` | 把 9.1-9.6 产物装配成骨架 + 叙事链自检 (桥接技能包) | 论文骨架 |
| L2 | `lectures/L2-submission-process.md` | 投稿全流程: venue / 时间线 / 格式匿名 / 提交机制 | 投稿 checklist |
| L3 | `lectures/L3-peer-review-rebuttal.md` | 同行评审 + **rebuttal 写作** (决定生死的下半场) | rebuttal 骨架 |
| L4 | `lectures/L4-after-decision.md` | camera-ready / 改投 / arxiv / 学术诚信 | 改投复盘 |
| L5 | `lectures/L5-negative-results.md` | file drawer problem + 负结果何时值得单独发表 + 诚实报告失败分支 | 负结果分级判断 |
| L6 | `lectures/L6-paper-series-strategy.md` | 一个大项目怎么切成几篇论文: 切太碎 (salami slicing) vs 攒太久两种反模式 | 论文切分判断线 |
| L7 | `lectures/L7-venue-selection.md` | 会议 vs 期刊 (TACL/TMLR) 深层选择 + dual submission 红线与误解 | venue 选择备忘 |
| L8 | `lectures/L8-camera-ready-and-artifacts.md` | camera-ready 按天时间表 + artifact evaluation + 代码/数据发布时点 | camera-ready checklist |
| L9 | `lectures/L9-building-long-term-impact.md` | 真实传播 vs 灌水引用 + follow-up 工作的规划来源 | follow-up 候选清单 |
| L10 | `lectures/L10-writing-as-non-native-speaker.md` | 中式学术英语三类系统性陷阱 + 写作辅助工具的正确用法 | 逐句自检清单 |

> **写每一节的细则** (标题/摘要/引言/方法/...)请打开 `how_to_write_a_paper` 技能包 —— 本专题不重复它。
> **L5-L10 补充说明**: L1-L4 是投稿循环主干 (装配→投稿→评审rebuttal→录用拒稿后); L5-L10 是贯穿论文全生命周期、但 L1-L4 未展开的现实议题——负结果处理、多篇论文的项目级规划、venue 的深层机制、录用后的运营细节、发表之后的长期经营、以及非母语写作本身的策略。

## 动手 (2 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-paper-skeleton-assembler.ipynb` | 用 `src/paper_assembler.py` 把你的研究产物装配成论文骨架, 跑**叙事链审计** (查节覆盖 + claim→evidence, 抓出无证据的过度宣称) |
| `notebooks/N2-rebuttal-builder.ipynb` | 用 `src/rebuttal_kit.py` 对一组审稿意见做分类 + 优先级 + 字数预算, 生成结构化 rebuttal 骨架, 亲手写一段有证据的回应 |

## 可复用模板 (`templates/`)

- `paper-skeleton.md` — 论文骨架 (对应技能包 sections, 标注每节用 9.x 的什么产物)
- `submission-checklist.md` — 投稿前 checklist (格式 / 双盲匿名 / 强制声明 / 提交)
- `rebuttal-template.md` — rebuttal 结构模板 (分类 / 优先级 / 语气)

## 工具 (`src/`)

- `paper_assembler.py` — 产物→骨架装配 + 叙事链审计 (claim→evidence 完整性)
- `rebuttal_kit.py` — 审稿意见分类 (6 类) + 优先级 + 字数预算 + rebuttal 骨架生成

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native。两个 notebook 纯文本处理, 零算力。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 把研究产物装配成论文骨架, 用叙事链审计抓出无证据的 claim
- [ ] 选对会 (匹配度 > 名气), 从 deadline 倒排研究节奏
- [ ] 过双盲匿名检查 (自引中性化 / 匿名仓库), 避免 desk reject
- [ ] 系统分类审稿意见, 按对录用影响分配 rebuttal 火力
- [ ] 写出礼貌、用证据、逐条回应的 rebuttal; 知道补实验是最强武器
- [ ] 把拒稿当数据, 区分真问题 vs 噪声, 高效改投; 守住学术诚信红线
- [ ] 能判断一个负结果是否值得独立发表, 而不是简单归入 file drawer
- [ ] 能把一个大项目合理切分成几篇论文, 不切太碎(salami slicing)也不攒太久
- [ ] 能说清会议 vs 期刊(TACL/TMLR)的深层选择权衡, 守住 dual submission 红线
- [ ] 知道 accept 之后到 camera-ready 之间要按顺序做完哪些事、artifact evaluation 怎么应对
- [ ] 能规划一篇论文发表后的长期影响力经营, 而不是靠灌水引用撑门面
- [ ] 能识别并纠正中式学术英语写作里最系统性的几类陷阱, 正确使用写作辅助工具而不丢掉自己的论证逻辑

---

## 在 Module 9 中的位置

```
Module 9 科研技能
  输出   9.6 research-figures      ✅ (图)
        9.7 paper-writing-submission  ◄── 你在这里 (写成论文 + 投出去)
        9.8 research-presentation     (讲给人听)
  科研生活 9.9 research-life          (审稿/导师/伦理)
```
> 复用资产: `how_to_write_a_paper` 技能包 (写作细则) + 9.1-9.6 全部产物 (论文素材)。
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`
