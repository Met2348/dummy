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

```
   研究(9.1-9.6) → L1装配(+技能包写细则) → L2投稿流程 → L3评审+rebuttal → L4录用/拒稿
                                                          ↑ 论文命运常在这决定
```

---

## 学习路径 (4 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-from-research-to-paper.md` | 把 9.1-9.6 产物装配成骨架 + 叙事链自检 (桥接技能包) | 论文骨架 |
| L2 | `lectures/L2-submission-process.md` | 投稿全流程: venue / 时间线 / 格式匿名 / 提交机制 | 投稿 checklist |
| L3 | `lectures/L3-peer-review-rebuttal.md` | 同行评审 + **rebuttal 写作** (决定生死的下半场) | rebuttal 骨架 |
| L4 | `lectures/L4-after-decision.md` | camera-ready / 改投 / arxiv / 学术诚信 | 改投复盘 |

> **写每一节的细则** (标题/摘要/引言/方法/...)请打开 `how_to_write_a_paper` 技能包 —— 本专题不重复它。

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
