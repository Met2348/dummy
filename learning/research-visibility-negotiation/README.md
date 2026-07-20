# 9.11 research-visibility-negotiation — 科研品牌/推荐信/job talk/谈判 (个人可见度经营 → 求职材料包 → 谈判)

> **Module 9「科研技能」第12个专题 · 阶段: 科研生涯周期(继`career-pathways` 9.10之后, 第2块拼图)**
> `career-pathways`(9.10)教你按赛道规则走完求职流程的宏观框架(学术界四阶段漏斗/工业界求职/转换/博后/教职早期); 但那些讲义假设你已经会做几件具体的事——推荐信会有人主动帮你写好、job talk临场发挥就够、offer直接接受就是了。**这几个假设都不成立**, 每一件都是需要单独练习的具体技能。本专题把它们拎出来单独精修: 个人科研品牌怎么经营可见度、推荐信怎么请、job talk怎么设计演练、offer/经费/合作条件怎么谈判、多版本CV怎么系统管理。

---

## 这个专题要解决的真问题

`career-pathways`L1提过, 学术求职材料初筛阶段"推荐信的分量往往比你自己写的所有材料加起来还重"——但那一讲没有展开"怎么才能拿到一封有分量的推荐信"这件具体的事。同样, `career-pathways`L1也指出job talk是校园面试最容易翻车的环节, 但没有展开具体的结构设计和演练方法。这些"知道原理但不知道怎么落实"的空白, 正是本专题要补上的:

```
                没有本专题                              有本专题
可见度         等着被动被发现,论文发了没人知道           主动经营可见度: 让对的人知道你在做什么
推荐信         临时抱佛脚,推荐人现写现想                提前维护关系+给素材包,信写得具体有力
job talk       照搬会议报告slides去讲                    专门为"说服录用"设计+演练的15-45分钟
谈判           被动接受第一次offer,怕谈判显得贪心         带着锚定/底线/BATNA框架主动谈
CV管理         一份CV到处投,或多个版本互相打架            按受众版本化管理,叙事保持跨材料一致
      ↓                                                    ↓
求职这件事本身走完了流程,但每个具体动作都是临场发挥      career-pathways(9.10)教的每个阶段,都有对应的扎实技能支撑
```

---

## 学习路径 (5 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-building-research-visibility.md` | 个人科研品牌: 学术社群可见度经营(不是运营网红号), 区别于9.16`academic-community-engagement`的共同体角色参与 | 可见度输出习惯清单 |
| L2 | `lectures/L2-recommendation-letter-strategy.md` | 推荐信策略: 怎么请、提前给推荐人什么素材、怎么维护长期关系, 避免"临时抱佛脚" | 推荐人素材包 |
| L3 | `lectures/L3-designing-a-job-talk.md` | job talk设计与演练, 区别于`research-presentation` L1-L3的常规会议报告 | job talk大纲 |
| L4 | `lectures/L4-negotiation-for-researchers.md` | 谈判技巧: offer谈判/startup package/合作条件的锚定/底线/BATNA框架 | 谈判准备清单 |
| L5 | `lectures/L5-versioning-your-cv.md` | CV/简历版本化管理: 学术版/工业版/grant申请版, 避免版本混乱 | 版本化CV管理方案 |

> 读法: L1→L5 顺序; L1-L2讲"被人发现和记住"(可见度/推荐信), L3-L4讲"求职流程里的临场硬技能"(job talk/谈判), L5收束到"怎么系统管理呈现材料本身"。每讲读完立刻去`templates/application-kit-worksheet.md`对应小节或notebook上手一次。

## 动手 (1 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-audit-application-kit.ipynb` | 用 `src/application_kit_audit.py` 给你自己现在(或未来)真实的求职材料包做五块骨架自检(CV定制/推荐人策略/job talk主线/谈判准备/叙事一致性), 并把薄弱环节转成谈判/面试前的准备清单 |

## 可复用模板 (`templates/`)

- `application-kit-worksheet.md` — 求职材料包五块骨架填空worksheet(配L2-L5, 对应`application_kit_audit.SECTIONS`每一节)

## 工具 (`src/`)

- `application_kit_audit.py` — 求职材料包五块骨架自检 + 完整性检查(含空泛套话检测) + 谈判/面试准备清单生成(纯stdlib)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native 即可, 无需 WSL2。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 说清"经营可见度"和"运营网红号"的区别: 目标是让对的几十到几百人准确知道你在做什么, 不是追求转发量
- [ ] 说清推荐信质量90%取决于"请求之前做了什么", 并能列出至少2-3位推荐人的具体喂料计划
- [ ] 说清job talk和常规会议报告(`research-presentation`)听众/评判标准的区别, 并设计出一份"5分钟建框架+代表作+未来计划"结构的talk大纲
- [ ] 用锚定/底线/BATNA框架, 对一场真实或假想的谈判(offer/经费/合作条件)写出具体的准备材料
- [ ] 说清学术CV和工业界简历不是详略版本关系, 是穷举逻辑和聚焦逻辑两种不同文体, 并能用"单一信源+按受众派生"管理多个版本
- [ ] 用`application_kit_audit.py`给至少1份真实(或雏形)求职材料包做五块骨架自检并理解结果

---

## 在 Module 9 中的位置

```
Module 9 科研技能 (现12个专题, 两套并列体系)

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
  品牌谈判 9.11 research-visibility-negotiation (可见度/推荐信/job talk/谈判/CV管理 —— 支撑求职流程的具体硬技能)  ◄── 你在这里(新增)
  带教管理 9.12 team-leadership-for-researchers (指导新手/向下反馈/组会主持/冲突处理/团队健康诊断)
  运营实务 9.13 research-team-operations (时间管理/招募筛选/异步协作/onboarding/跨专业协作)
  经费资源 9.14 research-funding-and-resources (经费申请/算力规划/数据管理/多机构合作/供应商合规)
  诚信合规 9.15 research-integrity-and-compliance (不端调查/authorship仲裁/IRB伦理/IP成果转化/国际合作合规/负责任披露)
  共同体参与 9.16 academic-community-engagement (审稿之外的学术服务/组织workshop/会议社交/长期合作/跨机构评审网络)
  开放传播 9.17 open-science-and-communication (跨学科合作/公众沟通/预注册与开源发布/竞赛组织参与/社交媒体边界)
```
> 两套体系的区别不变: 体系①回答"这一个项目该怎么做", 体系②回答"我的科研生涯接下来往哪走"。9.10和9.11的关系: 9.10画出求职这条路径上各阶段"该做什么、按什么规则"的宏观地图, 9.11把地图上每个阶段里最容易被假设"自然而然会做好"、实际上需要单独练习的具体技能(可见度/推荐信/job talk/谈判/CV管理)拎出来精修——9.10不重复讲这些技能的具体操作, 9.11也不重复讲9.10已经讲过的阶段划分逻辑, 两者互补。
>
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`(原9模块设计文档)、`docs/superpowers/specs/2026-07-14-research-career-lifecycle-design.md`(9.10-9.17科研生涯与共同体扩展设计文档, 本专题为该扩展的第2个专题)。
