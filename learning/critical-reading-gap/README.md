# 9.3 critical-reading-gap — 批判式读论文 + 找问题

> **Module 9「科研技能」首个模板专题** · 阶段: 输入 (intake)
> 把「读懂一篇论文做了什么」(学习式阅读) 升级为「看出它真假、强弱、漏洞, 并据此找出可做的研究问题」(批判式阅读 → 找问题).

---

## 这个专题要解决的真问题

你过去读论文是为了**学技术**: 它做了什么、怎么实现、我能不能复现。
做研究读论文是另一回事: **这篇的真正贡献是什么? baseline 公不公平? 附录藏了什么? 下一个该做却没做的实验是什么?**

`I can replicate R1-Zero` 和 `I found why R1-Zero fails on X, and here's the fix nobody published` 之间, 隔的就是这层技能。本专题就是补这层。

```
学习式阅读  ──────────────►  批判式阅读  ──────────────►  找问题
"它做了什么/怎么实现"        "它哪里站不住/没测到"        "哪个缝隙值得我做"
   (你已经会)                  (L1-L2)                    (L3-L4-L5)
```

---

## 学习路径 (5 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-three-pass-reading.md` | 三遍读法: 用 3 次有结构的扫描代替 1 次从头读到尾 | 论文解剖卡 |
| L2 | `lectures/L2-adversarial-reading.md` | 攻击式阅读: 切换成「想拒掉这篇」的审稿人 | 攻击清单 (10 问) |
| L3 | `lectures/L3-gap-taxonomy.md` | gap 分类学: 6 类研究缝隙 + 每类的嗅探信号 | gap 记录卡 |
| L4 | `lectures/L4-idea-generation.md` | 从 gap 到 idea 的 5 种手法 + idea 三筛 | idea 卡 |
| L5 | `lectures/L5-from-reading-to-research.md` | 把前四讲串成可执行 SOP | 个人 gap 库雏形 |

> 读法建议: L1→L5 顺序读; 每读完一讲, 立刻去对应 notebook / 模板上手一次。研究技能是练会的, 不是读会的。

## 动手 (2 个 notebook — 真实科研动作, 不是训模型)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-dissect-a-paper.ipynb` | 拿一篇真实 NLP 论文, 跑完整三遍读法 + 攻击清单, 用 `src/citation_graph.py` 真的拉一次它的引用关系, 产出一张论文解剖卡 |
| `notebooks/N2-find-gaps-in-own-work.ipynb` | **把你自己的 `learning/reasoning-r1` 复现当作一篇待审稿的论文**, 攻击式自审, 列 gap → 收敛出 2-3 个候选 idea |

> N2 是这个专题的灵魂: 你不需要造练习材料, 你已经有 48 个真实复现。每一个都离「一篇论文」只差一个新实验。

## 可复用模板 (`templates/`)

- `paper-note-card.md` — 三遍读法笔记卡 (以后每读一篇论文复制一份)
- `gap-record-card.md` — gap 记录卡 (类型 / 证据 / 可做性 / 重要性)
- `idea-card.md` — idea 卡 (来源 gap / 假设 / 最小验证实验 / 风险)

## 工具 (`src/`)

- `make_cards.py` — 从模板批量生成空白卡片 (开新论文/新 idea 时一键起卡)
- `citation_graph.py` — 用 Semantic Scholar 公开 API 拉一篇论文的引用/被引, networkx 画迷你图谱 (与 9.2 文献图谱衔接)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native 即可, 无需 WSL2。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 对任意论文 30 分钟内完成三遍读法, 产出解剖卡
- [ ] 用 10 问攻击清单找出一篇论文至少 3 个可攻击点
- [ ] 说清 6 类 gap 各是什么, 并在一篇真实论文里指认出来
- [ ] 把一个 gap 用 5 种手法之一变成一个具体 idea, 并通过三筛
- [ ] 在自己的 R1-Zero 复现上, 独立产出 2-3 张 idea 卡
- [ ] 开始维护一个持续增长的个人 gap 库

---

## 在 Module 9 中的位置

```
Module 9 科研技能 (按研究项目生命周期)
  地基   9.1 research-knowledge-mgmt
  输入   9.2 literature-mapping
        9.3 critical-reading-gap   ◄── 你在这里 (首个模板专题)
  执行   9.4 experiment-design
        9.5 experiment-ops-repro
  输出   9.6 research-figures
        9.7 paper-writing-submission  (升级复用 how_to_write_a_paper)
        9.8 research-presentation
  科研生活 9.9 research-life
```
设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`
