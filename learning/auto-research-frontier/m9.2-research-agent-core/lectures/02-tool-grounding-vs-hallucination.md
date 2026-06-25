# L02 · 工具接地 vs 幻觉引用（本模块核心）

## 1. 研究 agent 最体面的谎言：幻觉引用

LLM 会用流畅、专业、格式完美的语气，引用一篇**根本不存在**的论文。
[Hidden Pitfalls (2509.08713)](https://arxiv.org/abs/2509.08713) 把这列为 AI 研究 agent
的典型陷阱之一。它比"答错"更危险，因为它**看起来更可信**。

工具（检索）的意义就在于**接地（grounding）**：agent 的每一个事实性引用，
都应该能回查到它**真检索到**的某篇论文。引用集 ⊆ 检索集——这是一条硬约束。

## 2. 我们把这个失败做成确定性复现

`agent.py` 的 `MockLLM.draft_idea` 故意这样写：

```python
ids = tuple(p.arxiv_id for p in retrieved)        # 真·检索到的
cited = ids + (self.HALLUCINATED_ID,)             # ★ 末尾混入一个库里没有的 id
```

`HALLUCINATED_ID = "2599.00001"` 看着像 arXiv 号，但 `corpus` 里根本没有。
这不是"为黑而黑"——真 LLM 就是这么幻觉的，我们只是把它变成**每次都必现、可被测试**的形式。

## 3. grounding 检查：一行集合差

`critic.review` 和 `agent.ungrounded_in` 的核心都是一句话：

```python
ungrounded = [c for c in cited_ids if c not in retrieved_ids]
```

引用集减去检索集，剩下的就是凭空捏造的。跑 `python src/run.py`：

```
A. 无 critic：幻觉引用残留 1   ['2599.00001']
B. 有 critic：幻觉引用残留 0
```

`test_critic_removes_hallucinated_citation`（==0）和
`test_without_critic_hallucination_survives`（>=1）把这对结果钉死。

## 4. 教训

> **能接地的检索工具，加上一个会做集合差的检查，就能挡住最体面的那种谎言。**
> 注意：这只挡住了"引用是否存在"。引用**存在但不支持你的论断**（断章取义）是更深的坑——
> 那是 9.4 的"引用忠实度"。本讲先守住第一关：别引不存在的东西。

## 5. 动手

1. 把 `HALLUCINATED_ID` 改成一个**真在 corpus 里**的 id，重跑——critic 还报幻觉吗？为什么？
   （体会"接地检查"判的是"在不在检索集"，不是"id 长得像不像真的"。）
2. 给 `draft_idea` 再加一种坑：引用一篇**确实检索到、但和子问题无关**的论文。
   现在的 critic 抓得到吗？抓不到的话，这指向哪一关的检查？（提示：忠实度，9.4。）
