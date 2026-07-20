# 9.14 research-funding-and-resources — 经费与资源规划 (经费申请全流程 → 算力规划 → 数据管理 → 多机构合作 → 供应商合规)

> **Module 9「科研技能」第15个专题 · 阶段: 科研生涯周期(继`research-team-operations` 9.13之后, 第5块拼图)**
> `research-team-operations`(9.13)教的是团队日常怎么运转(时间分配/招募/异步协作/onboarding/跨专业协作), 但一个团队/项目要真正跑起来, 还依赖一整套此前完全空白的"钱和资源"实务: 经费怎么申请、算力怎么规划、数据怎么按合规要求管理、多机构合作的责任和资源怎么划分、用到的第三方供应商/API是否合规。本专题补上这块操作性拼图——它和9.13的关系是"人怎么运转团队" vs "钱和资源怎么被规划和申请", 对象不同, 互补而非重复。

---

## 这个专题要解决的真问题

大多数博士生/年轻研究者第一次真正接触"经费和资源"这件事, 往往是被动的: 导师说"预算表填一下这几项", 或者"这次算力不够, 你去申请一下"——从没有人系统教过"经费申请的完整生命周期是什么""算力资源有哪些来源、该怎么谈判""数据管理规划(DMP)为什么很多资助机构强制要求""多机构合作怎么提前把责任和署名写清楚""用到的第三方API/数据集是否真的允许你这样用"。

```
                没有本专题                              有本专题
经费            budget填个大概数字, 随手写"研究经费"        每项花费对应具体产出, 知道执行/结题阶段怎么衔接
算力            "需要更多算力"就结束了                    分清三种来源+成本估算+有据可依的谈判技巧
数据            DMP当走过场附件随手填"按需共享"             按FAIR原则+四块内容写清楚, 用DMPTool辅助
多机构合作      口头约定"到时候看贡献"                     责任/资源/署名提前写成书面协议, 含退出条款
第三方合规      用了再说, 上线/投稿前才想起查条款            立项阶段就核查API/数据集使用许可是否覆盖用途
      ↓                                                    ↓
经费和资源全靠"缺什么现申请"的被动应对                每个环节都有可核查、可复用的具体方法, 不再临场手足无措
```

---

## 学习路径 (5 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-grant-application-lifecycle.md` | 经费申请全流程: 撰写申请材料→预算论证→执行→结题报告, 区别于`research-direction-proposal`L3聚焦"钱"而非研究内容 | 预算论证草稿 |
| L2 | `lectures/L2-compute-resource-planning.md` | 算力资源规划与申请: 集群quota/云计算预算估算/GPU资源battle谈判技巧 | 算力规划+备选方案 |
| L3 | `lectures/L3-data-management-plans.md` | 数据管理规划(DMP): FAIR原则、存储/共享/留存策略, 很多资助机构强制要求 | DMP草稿 |
| L4 | `lectures/L4-multi-institution-collaboration.md` | 大型多机构合作项目管理: 责任划分/资源分摊/多方署名协议 | 合作协议核对清单 |
| L5 | `lectures/L5-vendor-and-api-compliance.md` | 供应商/API/第三方合规评估: 数据处理协议/模型使用许可核查, 工业界研究常见场景 | 合规核查清单 |

> 读法: L1→L5 顺序; L1-L2讲经费和算力这两类最基础的资源怎么申请, L3讲数据这一常被忽略的强制合规环节, L4-L5讲对外(机构间/供应商)的责任与合规关系。每讲读完立刻去 `templates/funding-plan-worksheet.md` 对应小节上手一次。

## 动手 (1 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-audit-funding-plan.ipynb` | 用 `src/funding_plan_audit.py` 给你现在(或未来可能)真实要申请的一份经费/资源计划写一版五块骨架, 自动检查漏项, 并用 `reviewer_focus()` 把薄弱环节转成评审最可能追问的问题清单 |

## 可复用模板 (`templates/`)

- `funding-plan-worksheet.md` — 经费与资源计划五块骨架模板(配L1-L5, 对应 `SECTIONS` 的五个key)

## 工具 (`src/`)

- `funding_plan_audit.py` — 经费与资源计划骨架自检 + 评审追问预判(五块: 预算论证/算力规划/数据管理/多机构协议/供应商合规, 纯stdlib)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native 即可, 无需 WSL2。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 说清"经费申请全流程"和`research-direction-proposal`L3「开题报告写作」的区别: 前者聚焦钱怎么申请和执行, 后者聚焦研究内容本身的论证
- [ ] 写出一版预算论证草稿, 每一项花费都对应具体产出, 而不是笼统的"研究经费"
- [ ] 判断一次真实(或设想中)的算力需求该走哪种来源(组内集群/外部大型资源/云计算), 并写出至少一条具体备选方案
- [ ] 按FAIR原则和四块内容(来源规模/存储备份/权限隐私/共享留存)写出一版DMP草稿
- [ ] 为一次多机构合作写出责任划分/资源分摊/署名协议/退出条款的书面核对清单
- [ ] 对一次真实用到的API或数据集, 按核查清单确认使用许可是否覆盖你打算的发表或商用用途
- [ ] 用 `funding_plan_audit.py` 给自己的经费/资源计划做一次完整性自检并跑一遍 `reviewer_focus()`

---

## 在 Module 9 中的位置

```
Module 9 科研技能 (现15个专题, 两套并列体系)

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
  运营实务 9.13 research-team-operations (时间管理/招募筛选/异步协作/onboarding/跨专业协作)
  经费资源 9.14 research-funding-and-resources (经费申请/算力规划/数据管理/多机构合作/供应商合规)  ◄── 你在这里(新增)
  诚信合规 9.15 research-integrity-and-compliance (不端调查/authorship仲裁/IRB伦理/IP成果转化/国际合作合规/负责任披露)
  共同体参与 9.16 academic-community-engagement (审稿之外的学术服务/组织workshop/会议社交/长期合作/跨机构评审网络)
  开放传播 9.17 open-science-and-communication (跨学科合作/公众沟通/预注册与开源发布/竞赛组织参与/社交媒体边界)
```
> 9.13和9.14是"科研生涯周期"体系里紧密衔接的两块操作性拼图, 但关注对象不同: 9.13回答"团队本身的日常运转(人/时间/协作)该怎么被结构化地安排", 9.14回答"团队/项目依赖的钱和资源(经费/算力/数据/多机构关系/第三方合规)该怎么被规划和申请"——一个真实运转的研究团队既需要9.13教的操作骨架, 也需要9.14教的资源规划骨架, 两者缺一不可但不能互相替代, 9.14不重复9.13已经讲过的时间管理/招募/协作内容, 只补它没覆盖的这五块"钱和资源"问题。
>
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`(原9模块设计文档)、`docs/superpowers/specs/2026-07-14-research-career-lifecycle-design.md`(9.10-9.17科研生涯与共同体扩展设计文档, 本专题为该扩展的第5个专题)。
