# 99 · 间隔复习策略（SM-2）

八股文最大的敌人不是"不会"，是"背过又忘"。`src/tracker.py` 用简化版 SM-2 算法把复习节奏交给算法，而不是凭感觉"再看一遍"。

## 用法

```python
from tracker import ReviewTracker, seed_from_qa
from ai_qa import ALL_QA as AI_QA
from backend_qa import ALL_QA as BE_QA

t = ReviewTracker()
seed_from_qa(t, AI_QA + BE_QA)   # 全部 ~300 张卡片，初始都在第 0 天到期

today = 0
for qid in t.due(today):
    ...                          # 口头/书面作答，跑 grade() 自查
    quality = 5                  # 0-5：能脱口而出关键词+扛住追问=4-5，
                                 # 磕磕巴巴但答对=3，答错/答不出=0-2
    t.review(qid, quality, today)
```

- **quality >= 3**（答对）：间隔按 SM-2 规则拉长（1 天 -> 6 天 -> 此后按 ease 倍数递增），下次复习时间往后推。
- **quality < 3**（答错/答不出）：间隔打回 1 天，明天必须再见到这张卡片。
- `ease` 会跟着答题质量微调——持续答得好，间隔拉长得更快；持续答得差，卡片会更频繁地出现。

## 建议节奏

- 每天固定跑一次 `t.due(today)`，只练"今天到期"的卡片，不要贪多刷全部 300 题——间隔复习的意义就是让你不用每天重复看已经很熟的内容。
- `today` 用整数递增（比如"第几天开始练"），不要绑定真实日历日期，方便离线补练也不会打乱统计。
- 序列化：`t.to_json()` / `ReviewTracker.from_json()` 落盘保存进度，下次启动直接恢复。

## 和 leetcode-mastery 的关系

两个 track 的 `ReviewTracker` 算法完全一致（同一套 SM-2 简化实现），但各自独立成册、互不 import——`leetcode-mastery` 追踪的是"能不能独立写出代码"，本 track 追踪的是"能不能在被追问时脱口而出标准答案"，是两种不同的记忆目标，混在一起复习反而会互相干扰节奏感。
