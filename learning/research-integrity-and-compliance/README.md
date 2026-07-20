# 9.15 research-integrity-and-compliance — 科研诚信与合规深水 (不端调查/authorship仲裁 → IRB伦理审查 → 知识产权与成果转化 → 国际合作合规 → 安全与负责任披露)

> **Module 9「科研技能」第16个专题 · 阶段: 科研生涯周期(继`research-funding-and-resources` 9.14之后, 第6块拼图)**
> `research-life`(9.9)的L3已经讲过署名规则和伦理红线本身(怎么不踩线), `research-funding-and-resources`(9.14)讲的是经费/算力/数据/合作/供应商这五类"资源怎么被规划和申请"。但还有一整块此前完全空白的合规深水区: 红线真的被踩(或被怀疑被踩)之后走什么正式调查程序、涉及人类被试的项目该怎么申请伦理审查、研究成果本身作为知识产权该怎么被保护、国际合作会不会踩到出口管制或跨境数据的法律红线、发现真实安全漏洞后该怎么负责任地披露。本专题补上这块"没人提醒就压根没想到要处理"的盲区拼图。

---

## 这个专题要解决的真问题

大多数博士生/年轻研究者对"科研诚信"的认知停留在"别造假、别抄袭"这句正确但不够用的口号上——从没有人系统教过"真的被指控不端会走什么程序""涉及人类被试的项目该怎么申请审查、常见在哪一步被打回""我的研究成果本身有没有可能悄悄放弃了专利权""国际合作会不会因为出口管制或数据跨境规则而踩坑""发现一个真实的安全漏洞该怎么负责任地公开"。

```
                没有本专题                              有本专题
不端处理        以为出事私下道歉/撤稿就能了结              知道调查是四阶段正式流程(举报→初步排查→正式调查→
                                                        裁决), 程序一旦启动不会因撤稿而终止
伦理审查        "我只是跑模型, 跟人类被试没关系"           意识到众包标注/人类评测本身就是人类被试研究, 需要
                                                        判断走豁免/快速/全审哪一种
知识产权        "发论文=保护了我的成果"                    知道发表可能悄悄放弃很多国家的可专利性, IDF必须先
                                                        于公开发表提交
国际合作        "都是搞学术的, 互相帮忙很正常"              合作前核实受限清单、基础研究豁免是否成立、数据跨境
                                                        传输是否有具体合规安排
安全披露        发现漏洞冲动立刻公开分享                    知道协调披露(先私下报告、给合理修复窗口)是默认选择,
                                                        而非全公开或完全不披露两个极端
      ↓                                                    ↓
诚信与合规全靠"出了事才第一次想起来查"的被动应对         每个环节都有可核查、可复用的具体方法, 不再临场手足无措
```

---

## 学习路径 (5 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-research-misconduct-deep-dive.md` | 学术诚信深水: 不端调查四阶段流程、authorship纠纷仲裁阶梯、图像/数据造假识别技术, 区别于`research-life`L3聚焦"红线被踩之后的正式程序"而非"怎么不踩线" | 署名纠纷应对预案草稿 |
| L2 | `lectures/L2-irb-and-ethics-review.md` | IRB/伦理审查全流程: 三种审查类别判断法、申请全流程、最常见的驳回原因, 尤重NLP方向众包标注/人类评测阶段常被忽视的审查义务 | IRB审查类别判断草稿 |
| L3 | `lectures/L3-ip-and-tech-transfer.md` | 知识产权与成果转化: 专利disclosure的时间点、技术转移办公室(TTO)角色、创业spin-off基本路径 | IP披露时间线草稿 |
| L4 | `lectures/L4-international-collaboration-compliance.md` | 国际合作合规: 出口管制基础概念(基础研究豁免/视同出口)、跨境数据传输限制, 区别于`research-funding-and-resources`L5聚焦"国家层面强制合规"而非"和供应商的合同性合规" | 国际合作合规核对清单 |
| L5 | `lectures/L5-responsible-disclosure.md` | 安全与负责任披露: AI安全/red-teaming研究发现真实漏洞后的协调披露流程与时间线, 呼应`red-team-jailbreak`但聚焦"披露流程"而非"攻击技术本身" | 披露时间线与保留细节清单 |

> 读法: L1→L5 顺序; L1讲"红线被踩之后"的处理程序, L2讲立项阶段的强制前置审查, L3-L4讲研究成果和国际合作两类此前空白的法律/资产合规, L5讲发现真实问题之后的负责任披露。每讲读完立刻去 `templates/compliance-checklist.md` 对应小节上手一次。

## 动手 (1 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-audit-compliance.ipynb` | 用 `src/compliance_checklist.py` 给你现在(或未来可能)真实要处理的一个研究项目写一版五块骨架, 自动检查漏项 |

## 可复用模板 (`templates/`)

- `compliance-checklist.md` — 科研合规五块骨架模板(配L1-L5, 对应 `SECTIONS` 的五个key)

## 工具 (`src/`)

- `compliance_checklist.py` — 科研合规骨架自检(五块: 署名协议/IRB状态/知识产权披露/出口管制/负责任披露计划, 纯stdlib)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native 即可, 无需 WSL2。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 说清"红线被踩之后走什么正式程序"和`research-life`L3「怎么不踩红线」的区别, 知道不端调查是四阶段流程且不会因论文撤回而自动终止
- [ ] 判断一个项目(含众包标注/人类评测阶段)是否需要IRB审查、该走豁免/快速/全审哪一种类别
- [ ] 说出专利disclosure(IDF)为什么必须先于公开发表提交, 以及技术转移办公室(TTO)在成果商业化中扮演的角色
- [ ] 对一次真实(或设想中)的国际合作, 判断是否需要核实受限清单、基础研究豁免是否成立、跨境数据传输是否合规
- [ ] 说清协调披露(coordinated disclosure)相比完全公开/完全不披露两个极端的优势, 以及公开发表时该保留哪些技术细节
- [ ] 用 `compliance_checklist.py` 给自己的项目做一次完整性自检

---

## 在 Module 9 中的位置

```
Module 9 科研技能 (现16个专题, 两套并列体系)

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
  经费资源 9.14 research-funding-and-resources (经费申请/算力规划/数据管理/多机构合作/供应商合规)
  诚信合规 9.15 research-integrity-and-compliance (不端调查/authorship仲裁/IRB伦理/IP成果转化/国际合作合规/负责任披露)  ◄── 你在这里(新增)
  共同体参与 9.16 academic-community-engagement (审稿之外的学术服务/组织workshop/会议社交/长期合作/跨机构评审网络)
  开放传播 9.17 open-science-and-communication (跨学科合作/公众沟通/预注册与开源发布/竞赛组织参与/社交媒体边界)
```
> 9.14和9.15是"科研生涯周期"体系里紧密衔接的两块操作性拼图, 但关注对象不同: 9.14回答"钱和资源(经费/算力/数据/多机构关系/第三方合规)该怎么被规划和申请", 9.15回答"诚信和合规红线(不端调查/伦理审查/知识产权/国际合规/安全披露)该怎么被处理和核查"——一个真实运转的研究项目既需要9.14教的资源规划骨架, 也需要9.15教的合规核查骨架, 两者缺一不可但不能互相替代, 9.15不重复9.9/9.14已经讲过的伦理红线/供应商合规内容, 只补它们没覆盖的这五块"诚信与合规深水"问题。
>
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`(原9模块设计文档)、`docs/superpowers/specs/2026-07-14-research-career-lifecycle-design.md`(9.10-9.17科研生涯与共同体扩展设计文档, 本专题为该扩展的第6个专题)。
