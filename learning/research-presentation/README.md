# 9.8 research-presentation — 把研究讲给人听

> **Module 9「科研技能」· 阶段: 输出 (output)**
> 你 9.4/9.5 把研究**做对**, 9.6/9.7 把它**写清/投出**, 9.8 把它**讲明白** —— talk / poster / 电梯演讲 / 答辩。

---

## 这个专题要解决的真问题

大量好工作因为讲不清被低估, 一些平庸工作因为讲得漂亮被高估。讲研究是一门**独立技能**, 贯穿你整个学术生涯: 组会、开题、资格考、毕业答辩、会议 talk/poster、job talk、招待会上被随口问「你做啥的」。

新手的通病:
- 把 talk 当**论文有声版**照着念 (听众不能倒回, 直接跟丢)
- 一张 slide 塞满字 (听众在读字就没在听你)
- **知识的诅咒**: 自己太懂了, 忘了「不懂的感觉」, 跳过听众需要的铺垫
- 被问到没准备的问题就慌, 或不懂装懂被当场拆穿

> **核心: presentation 是「把你脑子里的东西装进别人脑子里」, 不是自我展示。** 每次只塞得进**一个 takeaway** —— 这和 9.6 一图一信息、9.7 叙事先行是同一原则的不同媒介版本。

```
   9.8: L1 为听众建模/一个takeaway → L2 talk+slides → L3 poster+pitch → L4 答辩+Q&A
```

---

## 学习路径 (4 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-curse-of-knowledge.md` | 知识的诅咒 + 听众建模 + 一个 takeaway | takeaway 一句话 |
| L2 | `lectures/L2-talk-and-slides.md` | talk 结构 + 一图一点 slide + 卡时间 + 临场 | talk 分镜 |
| L3 | `lectures/L3-poster-and-pitch.md` | poster 视觉动线 + 电梯演讲三档 + 按听众调术语 | pitch 阶梯 |
| L4 | `lectures/L4-defense-and-qa.md` | 答辩准备 + Q&A 框架 + 三种难题应对 (诚实最优) | Q&A 预案 |

## 动手 (2 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-talk-storyboard.ipynb` | 用 `src/talk_planner.py` 给你的研究排一个有时间预算的 talk 分镜, 跑结构自检 (唯一 takeaway? motivation 过长? slide 过密?) |
| `notebooks/N2-pitch-ladder.ipynb` | 用 `src/pitch_kit.py` 把研究压成 10秒/1分钟/5分钟三档, 写「外行版」vs「专家版」, 看时长估算 + 黑话检测 |

## 可复用模板 (`templates/`)

- `talk-storyboard.md` — talk 分镜模板 (时间预算 + 每页一个点)
- `poster-layout.md` — poster 视觉动线布局模板
- `pitch-ladder.md` — 10秒/1分钟/5分钟三档 + 听众适配
- `qa-prep.md` — 答辩 Q&A 预案 (预测问题 + 应对)

## 工具 (`src/`)

- `talk_planner.py` — talk 分镜 + 时间预算 + 结构自检 (takeaway/motivation/slide密度)
- `pitch_kit.py` — 电梯演讲三档字数/时长预算 + 听众术语密度自检 (黑话检测)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native。两个 notebook 纯文本处理, 零算力。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 为不同听众建模, 把研究压成一个可复述的 takeaway
- [ ] 排一个卡死时间、一图一点、motivation 不超长的 talk 分镜
- [ ] 设计有视觉动线的 poster (而非论文贴墙)
- [ ] 流利给出 10秒/1分钟/5分钟三档 pitch, 并按听众换术语密度
- [ ] 用回答框架应对 Q&A; 对没准备/敌意/不会的问题有招
- [ ] 答辩前预测 15-20 个问题并备好 backup slides

---

## 在 Module 9 中的位置

```
Module 9 科研技能
  输出   9.6 research-figures ✅ / 9.7 paper-writing-submission ✅
        9.8 research-presentation     ◄── 你在这里 (讲给人听)
  科研生活 9.9 research-life            (审稿/导师/伦理)
```
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`
