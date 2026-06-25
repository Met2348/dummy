# 9.3 · 创意与假设生成（Ideation & Tournament）

> 9.2 里"起草 idea"只是一步；9.3 把它放大成**生成一堆 idea → 排序 → 选 top-k**，
> 并正面撞上这个领域最清醒的一课：**novelty ≠ feasibility（听着新 ≠ 做得出）**。

## 一句话结论（本模块跑出来的真实数字）

```
评委冠军 = 'exotic-init'（真实可行性排第 6/6，acc=0.795）
真实冠军 = 'more-steps'（acc=0.932），评委把它排在第 5 位
self 点子：评委 Elo 均 1161 | 真实 acc 均 0.820
other点子：评委 Elo 均 839  | 真实 acc 均 0.901
```

评委（只看文案）选出的"最佳点子"，真跑出来是**全场最差**；而真正最好的朴素点子被排到末尾。
关键：这里的 `feasibility` 不是我编的数——每个 idea 是一组**真训练配置**，
acc 是 numpy 逻辑回归真跑出来的（`task.py`）。**gap 长在真实测量上。**

## 两个被演示的偏差

1. **novelty 偏差**：评委从文案数"新颖/exotic/cutting-edge"等热词打分，
   而这些词和真实价值无关。高热词文案（`exotic-init`）顶上去，朴素文案（`more-steps`）沉底。
2. **自偏好（self-preference）**：评委对"自家生成的点子"（`source=self`）凭空加 0.3 分。
   于是真实更差的 self 点子，Elo 反而更高——这是 grading-its-own-homework 在 ideation 阶段的脸。

`--no-self-bias` 关掉第 2 种，你会看到**第 1 种偏差仍在**：光靠热词，评委照样选错。

## 跑起来

```powershell
python src/run.py                  # 评委榜 vs 真实可行性榜，并排
python src/run.py --no-self-bias   # 关自偏好，看 novelty 偏差仍把烂点子顶上去

python scripts/eric_3080ti_env_audit.py --runbook --tests `
  --modules auto-research-frontier/m9.3-ideation-and-tournament `
  --json-out $env:TEMP/m9.json --md-out $env:TEMP/m9.md
```

## 目录

```
m9.3-ideation-and-tournament/
├── runbook.yaml
├── lectures/
│   ├── 01-novelty-is-not-feasibility.md   核心：ideation-execution gap
│   ├── 02-the-judge-tournament.md         Elo 锦标赛（复用 llm-judge-arena）
│   └── 03-self-preference-and-guards.md    自偏好 + 怎么防
└── src/
    ├── run.py
    ├── ideation/
    │   ├── task.py        numpy 逻辑回归真训练（病态条件，feasibility 真测）
    │   ├── ideas.py       点子库：文案 + 真配置 + 来源(self/other)
    │   ├── judge.py       评委：只看文案的 novelty + 自偏好
    │   └── tournament.py  确定性 Elo 排序 + rank_by_feasibility
    └── tests/test_tournament.py  6 测试：训练真确定性 + 评委冠军垫底 + 自偏好可测
```

## Hands-on（轮到你）

`judge.py` 现在只用"热词计数"估 novelty——这正是它看走眼的根源。试试：

1. **给评委补上可行性信号**：让 `judge_score` 偷看一点 `feasibility`（现实里就是"先跑个小实验再评"），
   重排锦标赛——评委冠军还垫底吗？这说明**对抗 ideation gap 的唯一解药就是执行**。
2. **加你自己的点子**：往 `IDEA_BANK` 加一个你觉得"听着很普通但其实有效"的配置（如特征标准化），
   看它真实排第几、评委排第几。
3. 把 `SELF_BIAS` 调到 0，再调到 0.8，观察 self 点子的 Elo 怎么随之漂移——
   体会"裁判=选手"时，偏好分一点点就能颠倒排名。

## 桥接

- 复用 **llm-judge-arena**（成对评判 → Elo）· **rag-essential**（真实里 idea 要先检索去重）。
- 往前：**9.6 评测**正面回答"怎么让评判可信"；**9.8 红队**把"刷评分"做成攻击靶子。
- 呼应 **9.5**：那里 `crank-lr` 真跑变差；这里把这种"想≠做"做成一整个排序现象。
