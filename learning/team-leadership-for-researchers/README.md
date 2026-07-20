# 9.12 team-leadership-for-researchers — 带教与向下管理 (指导新手 → 建设性反馈 → 组会主持 → 冲突处理 → 团队健康诊断)

> **Module 9「科研技能」第13个专题 · 阶段: 科研生涯周期(继`research-visibility-negotiation` 9.11之后, 第3块拼图)**
> `research-life`(9.9)教你怎么向上管理导师这段关系; `career-pathways`(9.10)和`research-visibility-negotiation`(9.11)教你怎么走完求职、经营可见度、谈判这条路径。但一旦你自己开始带本科生intern、师弟师妹, 甚至独立带一个小团队, 这些专题全都没有回答一个方向完全相反的问题: **怎么向下带人?** 怎么分配任务难度、怎么给建设性反馈、怎么主持组会、怎么处理authorship这类敏感冲突、怎么在问题变得不可收拾之前诊断出团队已经出了状况。本专题补上"科研生涯周期"里这一块此前完全空白的拼图。

---

## 这个专题要解决的真问题

大多数博士生第一次带人, 唯一的参照系是"自己当年是怎么被带的"——如果这个参照系本身就是坏样本(放养到失联, 或者微观管理到窒息), 新手带教者会不自觉地把同一套复制给自己带的人。同样, 大多数人第一次主持组会、第一次面对authorship分歧, 都是现场现学, 没有任何系统训练过。

```
                没有本专题                              有本专题
带教            凭直觉复刻自己被带的方式(常是坏样本)      任务难度阶梯+新手向code review, 可核查的过关标准
反馈            "继续努力""再仔细点"式空话                四维度自检(具体/可执行/平衡/时效), 具体到可执行
组会            声音最大的人主导, 安静的人全程不说话      议程分类+轮流发言/先写后说, 让安静的人也能发言
冲突            authorship/功劳分歧被回避, 私下嘀咕       公开化处理三原则+四步流程, 摊开谈而不是憋着
团队健康        等到有人离职才回头看哪里出了问题          四信号定期诊断, 早期发现早期干预
      ↓                                                    ↓
向下带人全靠直觉、复制坏样本、临场发挥                每个关键动作都有可核查、可复盘的具体方法
```

---

## 学习路径 (5 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-mentoring-junior-researchers.md` | 指导低年级学生/实习生: 任务难度阶梯+新手向code review+避免自己变成瓶颈 | 任务难度评估+一次code review示范记录 |
| L2 | `lectures/L2-giving-downward-feedback.md` | 向下管理与建设性反馈: 和`research-life` L2"向上管理"相对, 四维度让反馈具体可执行 | 反馈质量自检表 |
| L3 | `lectures/L3-running-effective-meetings.md` | 高效组会/brainstorm会主持: 议程设计+避免一言堂+让安静成员发言 | 组会议程分类记录 |
| L4 | `lectures/L4-handling-team-conflict.md` | 团队协作冲突处理: authorship纠纷、功劳分配分歧的公开化处理原则 | 冲突处理四步流程记录 |
| L5 | `lectures/L5-diagnosing-team-health.md` | 团队健康度诊断: 心理安全感/工作量/冲突可见度/成长可见度四信号早期识别 | 团队健康度诊断表 |

> 读法: L1→L5 顺序; L1-L2讲"一对一"层面的带教(任务分配/反馈), L3讲"一对多"层面的组会主持, L4-L5讲团队层面更棘手的冲突处理与健康诊断。每讲读完立刻去对应模板/notebook上手一次。

## 动手 (2 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-give-feedback.ipynb` | 用 `src/feedback_quality.py` 给你自己真实给出过(或收到过)的一次反馈打质量分, 并把敷衍版本改写成具体可执行的版本 |
| `notebooks/N2-diagnose-team.ipynb` | 用 `src/team_health_check.py` 给你现在(或曾经)所在的真实团队做一次四信号诊断, 并把偏低信号转成具体干预动作 |

## 可复用模板 (`templates/`)

- `feedback-quality-checklist.md` — 向下反馈四维度自检表(配L2)
- `team-health-checkup.md` — 团队健康度四信号诊断表(配L5)

## 工具 (`src/`)

- `feedback_quality.py` — 向下反馈四维度打分自检(具体性/可执行性/平衡性/时效性, 纯stdlib)
- `team_health_check.py` — 团队健康度四信号诊断(心理安全感/工作量均衡/冲突可见度/成长可见度, 纯stdlib)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native 即可, 无需 WSL2。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 用四级任务难度阶梯给新手分配一个匹配当前水平的任务, 并写出明确的过关标准
- [ ] 做一次新手向code review, 少而精、解释"为什么", 而不是一次性列一堆问题
- [ ] 用四维度(具体性/可执行性/平衡性/时效性)给自己一次真实的反馈打分, 并改写敷衍版本
- [ ] 设计一份分类清楚(通报/决策/brainstorm)的组会议程, 并用至少一种手法主动让安静成员发言
- [ ] 用公开化处理三原则和四步流程, 对一次authorship/功劳分配分歧写出具体的处理记录
- [ ] 用四信号诊断表给一个真实团队做一次健康度诊断, 并为偏低信号写出具体干预动作

---

## 在 Module 9 中的位置

```
Module 9 科研技能 (现13个专题, 两套并列体系)

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
  带教管理 9.12 team-leadership-for-researchers (指导新手/向下反馈/组会主持/冲突处理/团队健康诊断)  ◄── 你在这里(新增)
  (9.13-9.17 计划中: 团队运营实务/经费与资源规划/科研诚信与合规深水/学术共同体参与/开放科学与科学传播, 尚未建成)
```
> 两套体系的区别不变: 体系①回答"这一个项目该怎么做", 体系②回答"我的科研生涯接下来往哪走"。9.9-9.11教的都是**你自己**怎么在共同体里立足、走职业路径、经营品牌——对象始终是"你自己"; 9.12第一次把对象从"你自己"换成"你带的人": 从怎么带一个新手, 到怎么诊断一整个团队, 是"科研生涯周期"体系里第一个真正面向"向下管理"的专题, 不重复9.9-9.11已经讲过的内容, 也不提前展开9.13团队运营实务里更宏观的团队组建/时间管理问题。
>
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`(原9模块设计文档)、`docs/superpowers/specs/2026-07-14-research-career-lifecycle-design.md`(9.10-9.17科研生涯与共同体扩展设计文档, 本专题为该扩展的第3个专题)。
