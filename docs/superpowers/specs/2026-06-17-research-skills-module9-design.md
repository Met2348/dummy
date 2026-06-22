# Module 9「科研技能」设计 spec

> Date: 2026-06-17 · Brainstormed with user (博0, EE 本硕, 主做 NLP/LLM, VLM 未来延伸)
> 终态: 在现有 8 个工程 Module 之上新增 Module 9「科研技能」, 课件式专题为主, 按研究项目生命周期编排.

## 1. 背景与动机

用户已建成 48 专题的 LLM **工程**自学体系 (portfolio v3, Module 1-8), 工程广度和完成度极高.
但其 portfolio 通篇是**工程师叙事** ("What I Can Do" / Career Paths / salary), 衡量的是「能复现/实现什么」.
PhD 衡量的是另一件事: **产出前人没有的新知识**. `I can replicate R1-Zero` ≠ `I can do research`.

Module 9 的目标: 在已有工程地基之上, 长出薄薄但关键的一层「从消费知识到生产知识」的研究技能.

### 关键洞察 (驱动整个设计)
1. **研究技能不是「讲」会的, 是「练」会的.** 不能靠读一篇「如何批判式读论文」的课件就学会.
   → 课件为主 (用户偏好), 但 notebook 内核必须是**真实科研动作**, 而非跑模型.
2. **写作这块已有半成品.** `how_to_write_a_paper/` 已是成熟的「叙事先行」操作手册.
   → Module 9 不重做写作, 而是升级复用它, 并补「投稿/rebuttal」下半场.
3. **用户有 48 个真实复现当原料.** 每个复现离「一篇论文」只差一个新实验.
   → 练「找 gap / 设计消融 / 写成 paper」时, 直接拿用户自己的 R1-Zero / DPO 复现当材料, 不需造玩具数据.

## 2. 已确认的设计决策 (brainstorming 四问)

| 决策点 | 选择 |
|---|---|
| PhD 方向 | NLP / LLM (VLM 未来延伸, 暂不纳入) |
| 学习形态 | 课件式专题为主 (沿用 learning/ 套路), 练习作为辅线编进 notebook |
| 范围与节奏 | 先定全套蓝图 (本 spec), 再深做第 1 个专题当模板, 之后逐个推进 |
| 组织原则 | 按研究项目生命周期编排 (输入→执行→输出→科研生活) |
| 执行授权 | 用户明确: 后面一口气做完, 无需继续审批 |

## 3. 专题蓝图 (生命周期排序)

| # | slug | 阶段 | 覆盖技能 | 核心产出 |
|---|---|---|---|---|
| 9.1 | `research-knowledge-mgmt` | 地基 | 知识管理 | Zotero 文献库 + markdown 笔记系统 + idea pipeline + arxiv 跟进 |
| 9.2 | `literature-mapping` | 输入 | 文献综述+领域地图 | 2 周摸清子领域 SOP + 一张真实领域图谱 |
| 9.3 | `critical-reading-gap` ★ | 输入 | 批判式读论文+找问题 | 三遍读法 + 攻击式阅读 + gap taxonomy + idea 生成 |
| 9.4 | `experiment-design` | 执行 | 实验设计+baseline/消融 | 可证伪假设→最小实验→消融矩阵→方差/显著性 |
| 9.5 | `experiment-ops-repro` | 执行 | 实验管理与可复现 | wandb + config + repo 卫生 + 复现 checklist |
| 9.6 | `research-figures` | 输出 | 科研绘图 | matplotlib 出版级图 + 架构示意图 + 一张好图的语法 |
| 9.7 | `paper-writing-submission` | 输出 | 科研写作+投稿 | 升级复用 how_to_write_a_paper + 补投稿/rebuttal/OpenReview |
| 9.8 | `research-presentation` | 输出 | presentation | 会议 talk + poster + slides + 答辩 + elevator pitch |
| 9.9 | `research-life` | 科研生活 | 审稿+导师沟通+伦理 | 给别人审稿反哺写作 + 周报/meeting + 署名/伦理 |

★ = 首个模板专题.

## 4. 专题统一内部结构

```
learning/<topic>/
├── README.md      专题导览 + 学习路径 + 产出 checklist
├── papers/        ① 真实 NLP/LLM 案例论文  ② 「怎么做研究」经典参考
├── lectures/      研究生课程级课件 md (ASCII/mermaid 图 + 必要公式, 反复交代每一项定义)
├── notebooks/     研究动作 notebook (做真实科研动作, 非训模型)
├── templates/     可复用模板 (读论文笔记卡 / gap 记录卡 / idea 卡)
├── src/           可复用小工具
└── environment/   环境检查与依赖 (确保 notebook 可跑)
```

与技术专题的唯一内核差异: notebook 做的是真实科研动作 (拉引用图 / 画论文级图 / 在自己复现数据上设计消融), 而非训练/推理模型.

## 5. 首个模板专题详细设计: `9.3 critical-reading-gap`

### 5.1 学习目标
从「读懂一篇论文做了什么/怎么实现」(学习式阅读) 升级到「看出贡献的真伪、对比是否公平、附录藏了什么、下一个该做却没做的实验是什么」(批判式阅读), 并据此**找出可做的研究问题**.

### 5.2 lectures (课件, 研究生课程级)
- `L1-three-pass-reading.md` — 三遍读法 (Keshav): 鸟瞰 / 抓骨架 / 复现级精读; 每遍的产出物.
- `L2-adversarial-reading.md` — 攻击式阅读: 切换成「想拒掉这篇」的审稿人; baseline 公平性 / 数据泄露 / cherry-picking / 附录陷阱 / 过度宣称的识别清单.
- `L3-gap-taxonomy.md` — gap 分类学: 方法 gap / 评测 gap / 假设 gap / 泛化 gap / 复现 gap / 理论 gap; 每类的「嗅探信号」.
- `L4-idea-generation.md` — 从 gap 到 idea: 组合法 / 迁移法 / 极限法 / 反向法 / "what they didn't test"; idea 的三筛 (没解决 / 可做 / 重要).
- `L5-from-reading-to-research.md` — 把前四讲串成 SOP: 一周读 N 篇 → 维护 gap 记录 → 收敛出 2-3 个候选 idea.

每讲含: ASCII/mermaid 示意图; 必要处给出公式并逐项交代 (考虑用户易遗忘前文定义, 每次重新交代).

### 5.3 notebooks (研究动作)
- `N1-dissect-a-paper.ipynb` — 拿一篇真实 NLP 论文 (案例: 用户已复现的 R1-Zero 对应原论文 DeepSeek-R1 / 或 DPO 原论文), 跑完整三遍读法, 逐格填批判清单, 产出一张「论文解剖卡」.
- `N2-find-gaps-in-own-work.ipynb` — 把用户自己的 `learning/reasoning-r1` 复现当作「一篇待审稿的论文」, 用攻击式阅读自审, 列出 gap 记录, 收敛出候选 idea.

### 5.4 papers/
- 案例论文: 与用户已有复现对应的 1-2 篇原论文 (DPO / R1 系列), 便于「拿自己工作练」.
- 方法参考: Keshav "How to Read a Paper"; (可选) S. Keshav / "The Task of the Referee" 等经典短文.

### 5.5 templates/
- `paper-note-card.md` — 三遍读法笔记卡模板.
- `gap-record-card.md` — gap 记录卡 (类型 / 证据 / 可做性 / 重要性).
- `idea-card.md` — idea 卡 (来源 gap / 假设 / 最小验证实验 / 风险).

### 5.6 src/
- 轻量脚本: 从模板生成空白笔记卡; (可选) 用 Semantic Scholar API 拉一篇论文的引用/被引做迷你图谱 (与 9.2 衔接).

### 5.7 environment/
- 检查本机 Python / jupyter; requirements (jupyter, requests, 可选 matplotlib/networkx).
- 确保两个 notebook 端到端可跑.

## 6. 与现有资产整合

- **复用写作技能**: `how_to_write_a_paper/` 在 9.7 升级为细则层并补投稿环节; `how_to_write/` 合并去重 (留一份).
- **案例原料**: 全程引用 `learning/` 下既有 48 专题 (尤其 `reasoning-r1` / `dpo-family`) 作真实练习材料.
- **收口**: 完成首专题后更新 `portfolio` → v4, 新增「第 8 大画像: 会做研究的人」, Career Paths 增 Research/PhD track.
- **工作流**: 沿用 `docs/superpowers/specs` + `docs/superpowers/plans` 既有规范, 每专题一份 plan.

## 7. 本次交付范围 (一口气做完)

1. 本 spec (蓝图) — 完成即 commit.
2. 一份实施 plan (`docs/superpowers/plans/2026-06-17-research-skills-module9.md`).
3. **完整建出 `9.3 critical-reading-gap` 专题** (README + 5 讲课件 + 2 notebook + 3 模板 + src + environment), 环境检查通过, notebook 可跑.
4. 收尾: 更新 portfolio 提及 Module 9 已开张 + 首专题完成.

> 后续 8 个专题 (9.1/9.2/9.4-9.9) 留待下一轮, 以 9.3 为模板逐个推进.

## 8. 成功标准

- [x] Module 9 蓝图清晰可执行, 9 个专题各有明确产出.
- [x] 9.3 专题完整落地, 目录结构成为可复制模板.
- [x] 两个 notebook 端到端跑通 (环境已配).
- [x] 课件达到研究生课程级: 有图、有清单、公式逐项交代.
- [x] 至少一个 notebook **真的拿用户自己的复现工作**当材料练「找 gap」.
- [x] portfolio 更新, 体现「工程 → 研究」第二条腿.

## 9. 完成记录 (2026-06-22 收官) ✅ 9/9

全部 9 专题完成 (9.3 首专题 commit 31ad189; 其余 8 专题 2026-06-22 一口气建完):

| # | slug | 讲 | notebook | 关键 src 工具 | commit |
|---|---|---|---|---|---|
| 9.1 | research-knowledge-mgmt | 4 | 2 | bib_to_cards / arxiv_triage | b7e7d63 |
| 9.2 | literature-mapping | 4 | 2 | snowball / field_map | (2026-06-22) |
| 9.3 | critical-reading-gap | 5 | 2 | make_cards / citation_graph | 31ad189 |
| 9.4 | experiment-design | 5 | 2 | experiment / stats | (2026-06-22) |
| 9.5 | experiment-ops-repro | 4 | 2 | exp_tracker / repro_check | (2026-06-22) |
| 9.6 | research-figures | 4 | 2 | plotstyle / schematic | (2026-06-22) |
| 9.7 | paper-writing-submission | 4 | 2 | paper_assembler / rebuttal_kit | (2026-06-22) |
| 9.8 | research-presentation | 4 | 2 | talk_planner / pitch_kit | (2026-06-22) |
| 9.9 | research-life | 4 | 2 | review_kit / meeting_prep | (2026-06-22) |

合计: **38 讲课件 + 18 notebook (全 nbconvert 0 报错) + 18 src 工具 + 多份卡模板**, 每专题 verify_env 全过。

**实现中的设计亮点**:
- **活数据流**: 9.4 确定性模拟器 (埋真实交互效应) → 9.5 留痕 jsonl → 9.6 出版级图, 同一份「Robust-DPO 噪声鲁棒性」数据 (对应用户 dpo-family 复现) 逐级加工, 走完研究结果「跑出来→印进论文」全生命周期。
- **三身份批判视角闭环**: 9.3 攻击者 (找 gap) → 9.7 被告 (rebuttal) → 9.9 审判者 (审稿)。
- **桥接而非重做**: 9.7 桥接已有 `how_to_write_a_paper` 技能包 (写作细则), 只补投稿/评审/rebuttal 下半场。
- **一个真 bug 变教材**: 9.4 `experiment.py` 用 `hash(str)` 派生种子 (带 PYTHONHASHSEED 随机盐, 跨进程不可复现), 当场修掉并写进 9.4-L5 / 9.5-L1 当活案例。

收尾: portfolio 升 v4 (`portfolio_v4.md`), 第 8 大画像「会做研究的人」转正, 新增 PhD/Research Scientist 职业轨道。
