# 9.16 academic-community-engagement — 学术共同体参与 (审稿之外的学术服务 → 组织workshop → 会议社交 → 长期合作 → 跨机构评审网络)

> **Module 9「科研技能」第17个专题 · 阶段: 科研生涯周期(继`research-integrity-and-compliance` 9.15之后, 第7块拼图)**
> `research-life`(9.9)L1教的是"接了审稿邀请之后怎么把这一篇稿子审好"; `career-pathways`(9.10)教的是求职/转换/博后/教职这几个关键节点的路径选择; `research-visibility-negotiation`(9.11)教的是持续性的个人科研品牌经营。但还有一整块此前空白的内容: **面对一份PC/AC邀约该不该接、接了之后角色怎么定位, 要不要自己从零发起一场workshop, 在会议现场怎么主动认识人而不是"社恐式陪跑", 怎么把一次会议偶遇变成持续多年的合作, 以及怎么让整个细分领域的评审圈子反过来主动找你**。本专题补上这块"学术共同体参与"的完整拼图。

---

## 这个专题要解决的真问题

`research-life` L1讲的是"怎么审好一篇稿子"(执行层面), `research-visibility-negotiation`讲的是"怎么经营个人可见度"(持续性经营)。两者中间空出一大块完全没人教的内容: **学术共同体里那些"角色性"和"关系性"的参与动作——PC/AC该不该接、workshop该不该自己办、会议现场怎么破冰、认识的人怎么变成长期合作者、怎么进入评审圈子**——这些问题被默认成"资历到了自然就会"或"性格外向的人才擅长", 从来没有一套可核查的决策框架和可执行的动作清单。

```
                没有本专题                              有本专题
学术服务       来者不拒(照单全收累垮自己)或全部推掉        四维打分框架(时间成本/可见度/人脉/回馈, L1)判断
               (错过低成本高回报的机会), 全凭情绪化直觉      该不该接, 接了之后AC/PC角色定位分清楚
组织活动       "会议当天租个房间"式的低估, 临时抱佛脚        提案六要点+完整生命周期时间线(L2), 提前规避
                                                        常见组织失误
会议社交       "社恐式陪跑"坐在角落全程不说话                会前目标名单+破冰脚本+可量化小目标(L3)
长期合作       加了联系方式之后就再也没联系过                分阶段升级路径: 偶遇→48小时内follow-up→低风险
                                                        合作动作→长期维持(L4)
评审网络       审稿邀请多不多全靠运气或人脉的玄学             可主动触发的正反馈循环: 第一次被看见→表现可靠→
                                                        被记住优先邀请→跨机构扩散(L5)
      ↓                                                    ↓
学术共同体参与全靠性格外向或资历熬到了才自然发生         每个环节都有可核查的打分依据和可执行的具体动作
```

---

## 学习路径 (5 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-academic-service-beyond-reviewing.md` | 审稿之外的学术服务: PC/AC该不该接、接了之后角色怎么定位, 区别于`research-life`L1讲"接了之后怎么审好" | PC/AC邀约打分记录 |
| L2 | `lectures/L2-organizing-workshops.md` | 组织workshop/研讨会: 从提案到执行的完整流程、常见组织失误 | mini workshop proposal骨架 |
| L3 | `lectures/L3-conference-networking.md` | 学术会议社交与人脉网络建设: 怎么主动搭话、poster session社交技巧、避免"社恐式陪跑" | 会前目标名单+开场问题清单 |
| L4 | `lectures/L4-building-long-term-collaborations.md` | 建立长期合作关系: 怎么把一次会议偶遇变成持续多年的合作 | follow-up邮件草稿 |
| L5 | `lectures/L5-entering-review-networks.md` | 跨机构review/评审网络建设: 怎么进入某个细分领域的评审圈子、被邀请审稿的正反馈循环 | 入圈动作自查清单 |

> 读法: L1→L5 顺序; L1讲加入别人已搭好的桌子当裁判(PC/AC), L2讲自己从零搭一张新桌子(workshop), L3-L4讲个体层面的社交动作(认识人→维持关系), L5把视角拉高到群体层面的正反馈循环。每讲读完立刻去 `templates/engagement-scorecard.md` 或 notebook 上手一次。

## 动手 (1 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-score-an-engagement.ipynb` | 用 `src/engagement_scorer.py` 给你实际收到过(或可以合理设想)的2个学术共同体参与邀约打分对比(如"顶会workshop PC邀请" vs "不知名期刊单篇审稿"), 不需要虚构材料 |

## 可复用模板 (`templates/`)

- `engagement-scorecard.md` — 候选邀约四维打分卡(配L1-L5, 每个维度留2个以上候选的空白栏供手填分数+依据)

## 工具 (`src/`)

- `engagement_scorer.py` — 学术共同体参与邀约四维打分(时间成本/可见度增益/人脉价值/回馈价值) + 完整性自检 + 排序对比(纯stdlib)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native 即可, 无需 WSL2。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 说清PC和AC两个角色的时间投入量级差异(15-25小时 vs 40-60小时), 以及AC"协调与综合"而非"审稿加强版"的角色定位
- [ ] 用一份mini workshop proposal骨架(动机/为什么是现在/差异化/组织者分工/意向嘉宾/审稿规划)自查, 而不是临场才想起这几块
- [ ] 会前列出至少3个目标人物和对应的具体开场问题, 而不是"社恐式陪跑"全程不说话
- [ ] 对至少一位最近认识但还没follow-up的人, 写出48小时内、提到具体对话细节、带一个低成本下一步的follow-up
- [ ] 说清评审圈子"第一次被看见→表现可靠→被记住优先邀请→跨机构扩散"这套正反馈循环的每一步具体动作
- [ ] 用四维打分框架给至少2个真实的候选邀约打分并写出依据, 并理解总分高不等于必须接受

---

## 在 Module 9 中的位置

```
Module 9 科研技能 (现17个专题, 两套并列体系)

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
  诚信合规 9.15 research-integrity-and-compliance (不端调查/authorship仲裁/IRB伦理/IP成果转化/国际合作合规/负责任披露)
  共同体参与 9.16 academic-community-engagement (审稿之外的学术服务/组织workshop/会议社交/长期合作/跨机构评审网络)  ◄── 你在这里(新增)
  开放传播 9.17 open-science-and-communication (跨学科合作/公众沟通/预注册与开源发布/竞赛组织参与/社交媒体边界)
```
> 9.16和9.9的区别: 9.9教的是"怎么在共同体里不踩雷、可持续地做事"(审稿/导师沟通/署名伦理), 是贯穿全程的通用软技能; 9.16教的是"怎么主动参与、主动经营共同体关系"(该不该接PC/AC、怎么组织活动、怎么社交、怎么维持合作、怎么进圈子), 是更主动、更具体的角色和关系动作, 两者互补但不重复。9.16和9.11的区别: 9.11教的是持续性的个人品牌可见度经营(社交媒体/推荐信/job talk), 9.16教的是具体场景下的共同体角色参与和人际连接执行动作, 关注对象不同。
>
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`(原9模块设计文档)、`docs/superpowers/specs/2026-07-14-research-career-lifecycle-design.md`(9.10-9.17科研生涯与共同体扩展设计文档, 本专题为该扩展的第7个专题)。
