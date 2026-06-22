# 9.9 research-life — 科研生活: 审稿 / 导师沟通 / 署名伦理 / 可持续

> **Module 9「科研技能」· 阶段: 科研生活 (research life)** · Module 9 收官专题。
> 你已会**做**研究 (9.4/9.5)、**写**研究 (9.7)、**讲**研究 (9.8)。最后一块是怎么在学术共同体里**长久立足**: 给别人审稿、和导师高效协作、守住署名与伦理红线、可持续地把博士跑成马拉松。

---

## 这个专题要解决的真问题

技术能力决定你能做多好的研究; **科研生活的软技能决定你能做多久、走多远。** 博0 几乎零准备的:

- 收到审稿邀请, 怎么做一个**建设性、有伦理**的审稿人 (还能反哺自己写作)?
- 导师一周只给你 30 分钟, 怎么开**有明确 ask** 的高效 meeting?
- 合作出成果了, **谁一作、谁通讯**? 怎么不撕破脸?
- 拒稿、迷茫、burnout —— 怎么把博士**可持续**地跑完?

> Module 9 让你在学术桌子边坐过三个座位: **9.3 攻击者** (找别人 gap) → **9.7 被告** (应对攻击) → **9.9 审判者** (审别人)。同一套批判视角的三个身份, 让你理解学术共同体怎么自我纠错。

```
   9.9: L1 审稿(反哺写作) → L2 导师/合作沟通 → L3 署名+伦理红线 → L4 可持续科研
```

---

## 学习路径 (4 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-being-a-reviewer.md` | 建设性审稿 + 调转枪口反哺自己 | 一份结构化评审 |
| L2 | `lectures/L2-mentor-collaboration.md` | 向上管理: 有 ask 的 meeting / 周报 / 提问 / 适配导师 | meeting 议程 |
| L3 | `lectures/L3-authorship-ethics.md` | 署名规则 + 学术不端红线 + AI 工具诚信 | 署名伦理 checklist |
| L4 | `lectures/L4-sustainable-research.md` | 应对失败/拒稿/迷茫/burnout, 把博士跑成马拉松 | 可持续节奏 |

## 动手 (2 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-review-a-paper.ipynb` | 用 `src/review_kit.py` 给一篇工作写一份结构化评审 (rubric: 扎实/新颖/清晰/可复现/重要), 跑建设性自检, 再**调转枪口审自己** |
| `notebooks/N2-mentor-meeting-prep.ipynb` | 用 `src/meeting_prep.py` 给导师 meeting 准备带**明确 ask** 的议程, 自检"纯汇报无 ask"通病 |

## 可复用模板 (`templates/`)

- `review-form.md` — 结构化审稿表 (复用 9.3 攻击清单维度)
- `mentor-meeting.md` — meeting 议程 / 周报模板 (进展/卡点/需决策/下一步)
- `authorship-ethics.md` — 署名 + 伦理 checklist (谁一作 / 贡献 / 数据/AI 披露)

## 工具 (`src/`)

- `review_kit.py` — 结构化审稿 rubric + 建设性自检 + 审别人维度调转枪口审自己
- `meeting_prep.py` — 导师 meeting 议程 + "有没有明确 ask" 自检

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native。两个 notebook 纯文本处理, 零算力。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 写一份结构化、建设性 (复述+证据+建议)、有伦理的评审
- [ ] 把审稿维度调转枪口审自己的工作
- [ ] 开有明确 ask 的导师 meeting; 决策项带选项+倾向
- [ ] 说清署名规则 (一作/通讯/共一作), 提前谈署名预期
- [ ] 辨识学术不端红线 + AI 工具使用的诚信边界
- [ ] 用可持续节奏、可控过程指标管理拒稿/迷茫/burnout

---

## 在 Module 9 中的位置 (收官)

```
Module 9 科研技能 (9/9 完成!)
  地基   9.1 research-knowledge-mgmt ✅
  输入   9.2 literature-mapping ✅ / 9.3 critical-reading-gap ✅
  执行   9.4 experiment-design ✅ / 9.5 experiment-ops-repro ✅
  输出   9.6 research-figures ✅ / 9.7 paper-writing-submission ✅ / 9.8 research-presentation ✅
  科研生活 9.9 research-life     ◄── 你在这里 (收官)
```
> 一条从「消费知识」到「生产知识」的完整链路。设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`
