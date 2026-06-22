# 9.1 research-knowledge-mgmt — 研究知识管理 (你的「第二大脑」)

> **Module 9「科研技能」· 阶段: 地基 (foundation)**
> 在你开始读论文、找 gap、做实验之前, 先建一套**能撑五年博士、知识会复利**的管理系统。
> 这是整个 Module 9 的地基: 后面 9.2 文献综述、9.3 找 gap、9.4 实验设计产生的所有卡片/笔记/idea, 都落进这套系统里。

---

## 这个专题要解决的真问题

博士五年, 你会读上千篇论文、跑上百个实验、闪过几百个 idea。如果没有系统:

- **读了就忘** — 三个月前读过的论文, 现在只记得「好像有篇做过类似的」, 找不回来。
- **idea 漏光** — 洗澡时想到的好点子, 第二天就忘了; 一年后看到别人发出来才拍大腿。
- **重复劳动** — 同一个 BibTeX 引用手敲三遍; 同一篇论文的笔记散在五个地方。
- **被信息流淹没** — arxiv 每天上百篇, 不看怕错过, 看了又没时间, 最后焦虑性囤积。

> **核心信念: 研究是一项「知识复利」的事业。** 今天读的每篇论文、记的每条笔记、捕获的每个 idea, 如果进了一个**能被检索、能被链接、能被复用**的系统, 它的价值会随时间累积; 否则就是一次性消费, 读完归零。
>
> 你已经有 48 个工程专题的「输出复利」(代码可复用)。这个专题给你「输入复利」(知识可复用)。

```
              没有系统                          有系统 (第二大脑)
   读论文 → 当时懂了 → 三个月后归零        读论文 → 文献卡 → 双链 → 半年后一搜即回
   想 idea → 没记 → 忘了                   想 idea → inbox → 孵化 → 立项
   追 arxiv → 焦虑囤积 → 没读              追 arxiv → triage → 本周清单 → 读完入库
        ↓                                        ↓
   五年后: 一盘散沙                        五年后: 一座可检索的私人知识库
```

---

## 学习路径 (4 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-why-second-brain.md` | 为什么要第二大脑 + 文献管理工具选型 (Zotero 工作流) | 一个配好的 Zotero + BibTeX 流水线 |
| L2 | `lectures/L2-note-system-zettelkasten.md` | 文献笔记系统: 文献笔记 vs 永久笔记 (Zettelkasten), 防笔记墓地 | 原子化永久笔记 + 双链 |
| L3 | `lectures/L3-idea-pipeline.md` | idea 流水线: 从灵感闪现到立项的 4 道工序 | idea inbox + 孵化流程 |
| L4 | `lectures/L4-staying-current.md` | 信息流系统: 如何追 arxiv 而不被淹没 | 可持续的 triage + weekly review |

> 读法: L1→L4 顺序; 每讲读完立刻去对应 notebook / 模板搭一次。知识系统是「搭起来用」才有价值的, 光读会等于没读。

## 动手 (2 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-build-knowledge-base.ipynb` | 用 `src/bib_to_cards.py` 解析一个真实 BibTeX 文件, 自动生成一座 markdown 文献卡库, 建立你「第二大脑」的最小骨架 |
| `notebooks/N2-arxiv-triage.ipynb` | 用 `src/arxiv_triage.py` 对一批论文按你的研究关键词打分排序, 自动产出「本周阅读清单」, 把 arxiv 洪流变成可执行队列 |

## 可复用模板 (`templates/`)

- `literature-note.md` — 原子化永久笔记 (一条笔记一个想法, Zettelkasten 风格)
- `idea-inbox.md` — idea 捕获 inbox (灵感先落地, 不评判)
- `weekly-review.md` — 每周回顾 (清 inbox / 分诊 arxiv / 推进 idea)

## 工具 (`src/`)

- `bib_to_cards.py` — 解析 BibTeX → 批量生成 markdown 文献卡 (纯 stdlib, 无外部依赖)
- `arxiv_triage.py` — 给一批论文按关键词权重打分排序, 输出本周阅读清单 (纯 stdlib)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native 即可, 无需 WSL2。两个 notebook 离线可跑 (内置样例数据)。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 配好 Zotero + better-bibtex, 浏览器一键抓论文, 一键导出 `.bib`
- [ ] 说清「文献笔记」和「永久笔记」的区别, 并写出第一条原子化永久笔记
- [ ] 有一个 idea inbox, 任何时刻闪过的点子都有地方落
- [ ] 有一条 arxiv triage 流程, 每周 30 分钟把洪流变成 3-5 篇精读清单
- [ ] 这套系统能接住后面 9.2-9.9 产生的所有卡片

---

## 在 Module 9 中的位置

```
Module 9 科研技能 (按研究项目生命周期)
  地基   9.1 research-knowledge-mgmt   ◄── 你在这里 (一切的容器)
  输入   9.2 literature-mapping
        9.3 critical-reading-gap   ✅
  执行   9.4 experiment-design
        9.5 experiment-ops-repro
  输出   9.6 research-figures
        9.7 paper-writing-submission
        9.8 research-presentation
  科研生活 9.9 research-life
```
> 9.1 是容器, 9.3 是往容器里装的第一批货 (论文卡/gap 卡/idea 卡)。两者互补: 先有系统 (9.1), 系统才接得住批判式阅读的产出 (9.3)。
>
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`
