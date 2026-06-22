# 9.5 experiment-ops-repro — 实验管理与可复现

> **Module 9「科研技能」· 阶段: 执行 (execution)**
> 9.4 教你把实验**设计**对; 9.5 教你把实验**可靠地执行、留痕、可复现** —— 从「能跑出一个数」到「能负责任地报告一个数」。

---

## 这个专题要解决的真问题

你按 9.4 设计了一个漂亮的消融矩阵, 真跑起来是几十个种子 × 几十个配置。三个月后导师说「再跑一次」, 你发现:

- 数对不上了 (没固定 seed / 换了包版本)。
- 不知道当时用的什么超参 (硬编码散在代码里, 改过好几次)。
- 不知道哪张图是哪版代码跑的 (没记 git sha)。

这不是你菜, 是 ML 的**可复现性危机** + 缺少工程纪律。9.5 用四件套堵住它: **seed everything (确定性) + config as code (配置) + 实验追踪 (留痕) + repo 卫生 (一键复现)**。

```
   9.4 设计对的实验
        │
   L1 确定性(seed)  →  L2 配置(config)  →  L3 追踪(留痕)  →  L4 repo卫生+复现checklist
        │
   一个半年后还站得住、别人能复现的结果 (交棒 9.6 出图 / 9.7 写论文)
```

---

## 学习路径 (4 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-reproducibility-crisis.md` | 可复现危机 + 三层级 + 不确定性来源 + seed everything | 确定性清单 |
| L2 | `lectures/L2-config-management.md` | config as code: 一个 run = 一份 config + 一个 seed | config 模板 |
| L3 | `lectures/L3-experiment-tracking.md` | 实验追踪: config+metrics+指纹, 几十个 run 可查可对比 | 实验记录库 |
| L4 | `lectures/L4-repo-hygiene-checklist.md` | repo 卫生 + 复现 checklist (投稿硬关卡) | 复现 checklist |

## 动手 (2 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-config-and-tracking.ipynb` | 用 `src/exp_tracker.py` 把 9.4 的消融矩阵完整跑一遍并自动留痕 (config+metrics+git/env 指纹), 再用 pandas 把 jsonl 变成「可查询的实验数据库」 |
| `notebooks/N2-reproducibility-audit.ipynb` | 用 `src/repro_check.py` 给规范/潦草两条记录做可复现体检; 亲手验证「同 config+seed 两次一致 / 不固定 seed 则不一致」, 再拿 checklist 审自己的复现专题 |

> 复用 9.4 的 `experiment.py` 当被追踪的实验 —— 9.4 设计 + 9.5 留痕, 在 notebook 里真的接上。

## 可复用模板 (`templates/`)

- `experiment-config.yaml` — config as code 模板 (含 seed / data 版本)
- `run-card.md` — 单次实验记录卡 (config+metrics+指纹)
- `repro-checklist.md` — 复现 checklist (NeurIPS 风格, 投稿前必过)

## 工具 (`src/`)

- `exp_tracker.py` — 轻量本地实验追踪器 (mimics wandb: init/log/finish, 自动盖 git+env 指纹, 存 jsonl)
- `repro_check.py` — 可复现性体检: 给一条记录逐项打分 (seed/git/config/env/data/metrics)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native。两个 notebook 零算力、可复现 (复用 9.4 确定性模拟器)。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 写出完整的 seed_everything (含 PYTHONHASHSEED), 区分三层可复现
- [ ] 把硬编码超参重构成 config as code, 一个 run = 一份 config
- [ ] 用追踪器自动记录每个 run 的 config+metrics+git/env 指纹
- [ ] 把几十个 run 的 jsonl 变成 pandas 可查询的实验库
- [ ] 用复现 checklist 审一个项目, 指出缺哪几项、怎么补
- [ ] 投稿前出 lockfile、记数据版本、写一键复现脚本

---

## 在 Module 9 中的位置

```
Module 9 科研技能
  执行   9.4 experiment-design     ✅ (设计实验)
        9.5 experiment-ops-repro   ◄── 你在这里 (可靠执行 + 留痕 + 复现)
  输出   9.6 research-figures       (把留痕的结果画成图)
  ...
```
> 9.4→9.5 是一对: 设计 (9.4) 保证「实验问对了问题」, 执行+复现 (9.5) 保证「答案可信且可重现」。两者缺一, 结果都不可靠。
>
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`
