# 9.13 research-team-operations — 团队运营实务 (多项目时间管理 → 招募筛选 → 异步协作 → onboarding → 跨专业协作)

> **Module 9「科研技能」第14个专题 · 阶段: 科研生涯周期(继`team-leadership-for-researchers` 9.12之后, 第4块拼图)**
> `team-leadership-for-researchers`(9.12)教你怎么向下带人——任务分配、建设性反馈、组会主持、冲突处理、团队健康诊断, 对象始终是"团队里的人"。但一个团队每天真正运转起来, 还依赖一整套更基础的操作性问题: 你自己的时间怎么在几个并行项目之间分配、怎么招到合适的人、跨时区的人怎么协作、新人进来第一周该干什么、和非ML背景的合作者怎么对齐术语——这些是"日常怎么把团队跑起来"的运营逻辑, 而不是"怎么带人"本身。本专题补上这块此前完全空白的操作性拼图。

---

## 这个专题要解决的真问题

大多数博士生/年轻研究者第一次需要同时运营好几件事(自己的课题+合作项目+带的人+跨时区协作), 完全没有练过任何一件事该怎么结构化处理——时间分配靠"谁催得紧先做谁", 招人靠"谁先联系就要谁", 远程协作靠反复约同步会议, 新人靠口头传统带, 和工程师/PM协作靠连蒙带猜地互相解释。

```
                没有本专题                              有本专题
时间            谁催得紧先做谁, 主线课题永远被挤              时间块分配法, 主线课题深度块优先保护
招募            谁先联系就要谁, 或标准空洞到没人敢投          必须项/可现学项拆分 + 有边界的trial task
远程协作        反复约同步会议, 消息发出去石沉大海             响应时限约定 + 书面文档替代部分会议
onboarding      口头传统带教, 新人卡在等人腾时间               第一周阅读集+golden path文档, 新人独立跑通
跨专业协作      各自用各自的黑话, 靠上下文硬猜                 双向术语对照表 + 明确的决策边界
      ↓                                                    ↓
团队运营全靠临场发挥、口头传统、隐性协调                每个日常操作动作都有可核查、可复用的具体方法
```

---

## 学习路径 (5 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-multi-project-time-management.md` | 多项目并行时间管理: 时间块分配法, 先保护主线课题深度块再分配救火时间 | 每周时间块骨架 |
| L2 | `lectures/L2-recruiting-collaborators.md` | 招募筛选合作者/实习生: 必须项vs可现学项标准 + 有边界的trial task设计 | 招募标准+试做任务清单 |
| L3 | `lectures/L3-async-remote-collaboration.md` | 远程/异步跨时区协作: 响应时限约定 + 用书面文档替代部分同步会议 | 响应时限约定+异步替代方案 |
| L4 | `lectures/L4-onboarding-new-members.md` | 团队知识传承与onboarding文档: 第一周该看什么、该跑通什么 | onboarding文档草稿 |
| L5 | `lectures/L5-cross-disciplinary-teamwork.md` | 跨专业背景团队协作: 和工程师/PM/设计师的术语对齐与协作边界 | 术语对照表+决策边界清单 |

> 读法: L1→L5 顺序; L1-L2讲你自己怎么分配时间、怎么招人(团队构建的起点), L3-L4讲团队组建之后怎么运转(远程协作/新人上手), L5讲和团队之外、不同专业背景的协作者对齐。每讲读完立刻去 `templates/team-ops-plan.md` 对应小节上手一次。

## 动手 (1 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-audit-team-ops.ipynb` | 用 `src/team_ops_audit.py` 给你现在(或未来可能)真实要运营的一个团队/合作关系写一版五块骨架, 自动检查漏项/敷衍项, 并把每个缺口对应回具体该重读哪一讲 |

## 可复用模板 (`templates/`)

- `team-ops-plan.md` — 团队运营计划五块骨架模板(配L1-L5, 对应 `SECTIONS` 的五个key)

## 工具 (`src/`)

- `team_ops_audit.py` — 团队运营计划骨架自检(五块: 时间分配/招募标准/异步协作/onboarding文档/跨专业桥接, 纯stdlib)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native 即可, 无需 WSL2。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 用时间块分配法画出一张真实的每周时间骨架, 主线课题的深度块被明确保护而不是被救火任务挤占
- [ ] 为一个真实(或未来)的招募场景写出必须项/可现学项两张清单, 并设计一个2-4小时的trial task
- [ ] 为一次跨时区/远程协作写出按紧急程度分层的响应时限约定, 并把至少一项日常事务从同步会议改成异步文档
- [ ] 写出一份真实的onboarding文档草稿(第一周阅读集+可验证的golden path), 并用"新人独立跑通"检验它
- [ ] 为一次跨专业协作写出至少5组术语对照和2-3条决策边界划分
- [ ] 用 `team_ops_audit.py` 给自己的运营计划做一次完整性自检, 五块骨架全部通过而不是敷衍带过

---

## 在 Module 9 中的位置

```
Module 9 科研技能 (现14个专题, 两套并列体系)

体系① 20环节项目周期 (单个研究项目从选方向到发表答辩的生命周期, 见 research-direction-proposal/L0)
  起点   9.0 research-direction-proposal L1-L2 (方向选择+可行性)
  地基   9.1 research-knowledge-mgmt
  输入   9.2 literature-mapping / 9.3 critical-reading-gap
  立项   9.0 research-direction-proposal L3-L4 (开题写作+答辩)
  执行   9.4 experiment-design / 9.5 experiment-ops-repro
  输出   9.6 research-figures / 9.7 paper-writing-submission / 9.8 research-presentation

体系② 科研生涯周期 (跨越多个项目、以年为尺度的个人发展轨迹, 20环节地图容不下, 新分支)
  科研生活 9.9 research-life (审稿/导师沟通/署名伦理/可持续 —— 贯穿全程的通用软技能)
  职业路径 9.10 career-pathways (求职/转换/博后/教职早期 —— 关键节点的路径选择)
  品牌谈判 9.11 research-visibility-negotiation (可见度/推荐信/job talk/谈判/CV管理)
  带教管理 9.12 team-leadership-for-researchers (指导新手/向下反馈/组会主持/冲突处理/团队健康诊断)
  运营实务 9.13 research-team-operations (时间管理/招募筛选/异步协作/onboarding/跨专业协作)  ◄── 你在这里(新增)
  (9.14-9.17 计划中: 经费与资源规划/科研诚信与合规深水/学术共同体参与/开放科学与科学传播, 尚未建成)
```
> 9.12和9.13是同一条"向下管理"分支上紧密衔接的两块拼图, 但关注对象不同: 9.12回答"团队里的人该怎么被带、被反馈、被诊断"(以人为对象), 9.13回答"团队本身的日常运转该怎么被结构化地安排"(以流程/操作为对象)——一个团队既需要9.12教的人际技能, 也需要9.13教的操作性骨架, 两者缺一不可但不能互相替代, 9.13不重复9.12已经讲过的任务阶梯/反馈/会议主持/冲突处理/健康诊断内容, 只补它没覆盖的时间分配、招募、异步协作、onboarding、跨专业协作这五块操作性问题。
>
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`(原9模块设计文档)、`docs/superpowers/specs/2026-07-14-research-career-lifecycle-design.md`(9.10-9.17科研生涯与共同体扩展设计文档, 本专题为该扩展的第4个专题)。
