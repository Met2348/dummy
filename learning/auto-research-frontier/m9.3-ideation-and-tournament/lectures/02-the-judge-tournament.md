# L02 · 评委锦标赛：成对评判 → Elo 排序

## 1. 为什么用锦标赛而不是直接打分

让评委对每个 idea 直接打"绝对分"很不稳——分数会漂、缺乏可比性。
更稳的做法（你在 **llm-judge-arena** 学过）是**成对评判 + Elo**：每次只问"A 和 B 哪个好"，
再把一堆成对结果聚合成排名。co-scientist 的 tournament evolution 就是这个思路。

`tournament.py` 的核心是一个确定性 Elo：

```python
for _ in range(passes):                       # 多轮循环赛
    for i, j in 所有无序对:
        winner = prefer(a, b, use_self_bias)  # 成对评判
        # 标准 Elo 更新：赢家涨分、输家掉分，幅度看分差期望
        elo[a] += k * (sa - ea); elo[b] += k * ((1-sa) - (1-ea))
```

`prefer()` 判 `judge_score` 高者胜、平手按 id（确定性，可测试）。
跑多轮后，Elo 把"谁常赢"压成一个连续排名。

## 2. 评委看什么、不看什么

致命点全在 `judge_score`：

```python
s = novelty(idea)                              # 只看文案热词
if use_self_bias and idea.source == "self":
    s += SELF_BIAS                             # 还偏爱自家点子
# 注意：从头到尾没有 feasibility——评委够不着执行结果
```

所以这个 Elo 排的是"**听起来**有多好"，不是"**做出来**有多好"。
两份榜（`run_tournament` vs `rank_by_feasibility`）一对照，gap 就量化了。

## 3. 确定性 = 可测试

整条链没有随机：数据 seed 固定、训练确定、评判平手有确定 tie-break、Elo 更新顺序固定。
于是 `test_judge_champion_is_low_feasibility` 能稳稳断言"评委冠军真实排名 ≥ 后半段"。

> **教学工具的诚实，首先是确定性。** 一个每次跑结果都不一样的 demo，
> 你没法用测试钉住它的结论，也就没法证明它不是碰巧。

## 4. 动手

1. 把 `passes` 从 8 调到 1，再调到 50，看 Elo 排名稳不稳。多少轮以后排名不再变？
   （体会 Elo 需要足够多对局才收敛——这也是真实 arena 要打很多场的原因。）
2. 给 `prefer` 加一点"评委噪声"（比如 5% 概率选错），但要保持确定性
   （提示：用 idea id 的哈希而不是随机数）。排名的稳定性怎么变？
