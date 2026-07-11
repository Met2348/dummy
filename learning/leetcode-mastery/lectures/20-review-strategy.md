# L20 · 用 tracker 做间隔复习

100 题一次刷完不代表记住了——两周后重新看会发现忘了一半。`src/tracker.py` 用简化版 SM-2 算法把"什么时候该复习哪道题"这件事交给算法，而不是凭感觉重刷。

## 核心用法

```python
from catalog import PROBLEMS
from tracker import ReviewTracker, seed_from_catalog

t = ReviewTracker()
seed_from_catalog(t, PROBLEMS)      # 100 题全部种入，初始 due 于第 0 天

today = 0
for pid in t.due(today):            # 今天该复习的题
    ...                              # 自己重写一遍这道题
    quality = 5                      # 0-5 自评：完全没思路=0/1，卡壳看提示才会=2/3，独立想出但慢=4，closed-book流畅=5
    t.review(pid, quality, today)

# 存盘，明天继续
open("progress.json", "w").write(t.to_json())
```

## quality 怎么打分（诚实，别automatically 打 5）

| quality | 含义 |
|---|---|
| 0-1 | 完全没思路 / 看了答案才会 → 间隔打回到 1 天 |
| 2 | 想到了 pattern 但代码写错 |
| 3 | 磕磕绊绊做出来，超时严重 |
| 4 | 独立做出来，但比预期慢 |
| 5 | closed-book 流畅写出且复杂度最优 |

`quality < 3` 会把这道题的间隔打回 1 天（明天再见），`quality >= 5` 且连续答对时间隔会指数拉长（3 天 → 6 天 → 十几天……）——**你越不熟的题，tracker 会让你见得越频繁**，这正是间隔复习的意义。

## 建议节奏

- 每天先跑 `t.due(today)`，把到期的题过一遍（closed-book 重写，不是重读答案）。
- 学新题（`01`→`19` 推进）和复习到期题分开算：新题量控制在每天 3-5 道，复习到期的题不设上限（到期就得清）。
- 每周留一天"只复习不学新题"，防止到期队列堆积。
