# Module 9 科研技能 Implementation Plan ⭐ (首专题: critical-reading-gap)

**Goal**: 新开 Module 9「科研技能」(课件式专题为主, 按研究项目生命周期编排, 共 9 专题); 本轮完整建出首个模板专题 `critical-reading-gap` (批判式读论文 + 找问题).

**Architecture**: 9 专题 = 地基(9.1) + 输入(9.2/9.3) + 执行(9.4/9.5) + 输出(9.6/9.7/9.8) + 科研生活(9.9). 每专题统一外壳 `README + papers + lectures + notebooks + templates + src + environment`; 内核差异: notebook 做真实科研动作而非训模型. 首专题 9.3 完整落地后作为后续 8 个的复制模板.

**Tech Stack**: Python 3.13 (Windows native) / jupyter + nbformat / requests / matplotlib / networkx / pandas. 全部已验证可用, 无需 WSL2.

**Design 文档**: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`

**案例原料**: 复用用户已有 `learning/reasoning-r1` (R1-Zero 复现) 与 `learning/dpo-family` (DPO 复现) 当真实练习材料.

---

## Phase 1: 专题骨架 + 环境

### Task 1.1: 目录骨架
- `learning/critical-reading-gap/{papers,lectures,notebooks,templates,src,environment}/`

### Task 1.2: environment/
- `requirements.txt` (jupyter, nbformat, requests, matplotlib, networkx, pandas)
- `verify_env.py` (检查 import + nbformat 可写 + 输出 OK)

### Task 1.3: README.md
- 专题导览 + 5 讲学习路径 + 产出 checklist + 在 Module 9 中的位置

## Phase 2: lectures (研究生课程级课件, 含 ASCII/mermaid 图 + 公式逐项交代)

### Task 2.1: `L1-three-pass-reading.md`
- Keshav 三遍读法; 每遍的时间盒、目标、产出物; 何时弃读.

### Task 2.2: `L2-adversarial-reading.md`
- 攻击式阅读: 审稿人视角; baseline 公平性 / 数据泄露 / cherry-picking / 附录陷阱 / 过度宣称识别清单.

### Task 2.3: `L3-gap-taxonomy.md`
- 6 类 gap (方法/评测/假设/泛化/复现/理论) + 每类嗅探信号 + 真实 NLP 例子.

### Task 2.4: `L4-idea-generation.md`
- 从 gap 到 idea 的 5 法 (组合/迁移/极限/反向/未测) + idea 三筛 (未解决/可做/重要).

### Task 2.5: `L5-from-reading-to-research.md`
- 串成 SOP: 周读 N 篇 → 维护 gap 库 → 收敛 2-3 候选 idea; 与 9.1/9.2 衔接.

## Phase 3: templates (可复用模板)

### Task 3.1: `paper-note-card.md` — 三遍读法笔记卡
### Task 3.2: `gap-record-card.md` — gap 记录卡 (类型/证据/可做性/重要性)
### Task 3.3: `idea-card.md` — idea 卡 (来源 gap/假设/最小验证实验/风险)

## Phase 4: src (轻量工具)

### Task 4.1: `make_cards.py` — 从模板批量生成空白笔记卡/gap 卡/idea 卡
### Task 4.2: `citation_graph.py` — 用 Semantic Scholar API 拉一篇论文的引用/被引, networkx 出迷你图谱 (衔接 9.2)

## Phase 5: notebooks (真实科研动作, nbformat 生成)

### Task 5.1: `N1-dissect-a-paper.ipynb`
- 对一篇真实 NLP 论文跑完整三遍读法 + 攻击清单, 产出「论文解剖卡」; 含一段可跑的 citation_graph 调用.

### Task 5.2: `N2-find-gaps-in-own-work.ipynb`
- 把用户自己的 `reasoning-r1` 复现当「待审稿论文」自审, 列 gap → 收敛 idea, 落成 gap 卡 + idea 卡.

## Phase 6: papers/ + 收尾

### Task 6.1: papers/README.md + 下载 Keshav "How to Read a Paper" (公开 PDF); 列出案例原论文获取方式.
### Task 6.2: 跑 verify_env.py + nbconvert 执行两个 notebook 确认端到端可跑.
### Task 6.3: 更新 portfolio (新增 Module 9 开张 + 第 8 大画像「会做研究的人」).
### Task 6.4: commit.

## 成功标准
- [x] 9.3 专题目录完整, 可作模板复制.
- [x] 5 讲课件研究生级 (图 + 清单 + 公式逐项).
- [x] 2 notebook nbconvert 跑通.
- [x] N2 真的拿用户自己的复现练「找 gap」.
- [x] portfolio 体现「工程 → 研究」第二条腿.

---

## 后续 (2026-06-22): 9.1/9.2/9.4-9.9 全部完成 ✅ — Module 9 9/9 收官

以 9.3 为模板, 一口气建完其余 8 专题 (各 4-5 讲 + 2 notebook + src 工具 + 卡模板 + environment, 全 nbconvert 0 报错、verify_env 全过)。逐专题 commit 见 git log (feat(research-knowledge-mgmt) ... feat(research-life))。
portfolio 升 v4 (`portfolio_v4.md`): 第 8 大画像「会做研究的人」转正 + 新增 PhD/Research Scientist 轨道。
完整完成记录见 spec 第 9 节。
