# L03 · 领域阅读法：自称 vs 证据（核心技能）

## 1. 把"话术"和"证据"拆成两栏

这是整个 9.1、乃至整个 M9 系列最该带走的一招。`run.py --show table`：

```
  system                     claimed  evidenced  flags
  ResearchAgent              analyst       tool   [!]hype+1 (self)self-verified
  AI Scientist v2          scientist  scientist   (self)self-verified
  AlphaEvolve                analyst    analyst
  ...
```

- `claimed`：系统**自称**的级别（论文标题、摘要、宣传里的口径）。
- `evidenced`：分级器**只看证据**推出来的级别。
- `[!]hype+N`：自称比证据高 N 级——**话术溢价**。
- `(self)self-verified`：结果只经过**自评**，没有独立验证——**可信度警示**。

`ResearchAgent` 自称 Analyst，证据只够 Tool（它只生成 idea 不跑实验）→ `hype+1`。
这不是说它差，而是提醒你读它的结论时**自动打个折**。

## 2. 地图算出的那条不舒服的事实

`run.py` 末尾的洞见**不是我写死的句子**，是从目录里现算的：

```python
scientists = [证据级别==scientist 的系统]
都自评     = all(c.self_verified_only for c in scientists)     # → True
独立验证的最高级别 = max(level for c if 有独立验证)             # → analyst
```

结论：

> **2026 年，证据级别到 Scientist 的系统，结果全部仅靠自评；
> 而真正经过独立验证（held-out 测试 / 湿实验 / 排行榜）的，最高只到 Analyst。**

换句话说，**"自主性"和"可信度"在当前是反相关的**：越敢自称科学家的，越是自己给自己打分。
这条洞见被两个测试钉在数据上——你更新了证据标注，洞见要么自动跟着变、要么测试报错，
**绝不会留一句过期的漂亮话**。这本身就是对"科研诚信"的代码级示范（通向 9.8）。

## 3. 这套阅读法怎么用在你读论文时

读任何一篇 auto-research 论文，机械地填这张小表：

1. 它**自称**哪一级？（标题/摘要的口气）
2. 它**真自动化**了七环里的哪几环？（看 method/experiment，不看 intro）
3. **问题是谁定的**？idea 是人喂的还是自生成的？
4. 结果**谁验的**？自评 / held-out / 第三方复现？
5. 算 hype gap 和可信度旗标。

填完，你对这篇的"实际位置"就有了不被标题绑架的判断。这正是老师说的
**"research 被 Agent 接管到什么程度"——的清醒读法**。

## 4. 动手

1. 挑 `papers/INDEX.md` 里任意一篇你没标注过的系统，按上面五问填表，
   再把它加进 `systems.py`，看分级器是否同意你的判断。
2. 思考题（通往 9.6/9.8）：如果有人把"独立验证"也自动化了（让一个 AI 去复现另一个 AI 的结果），
   可信度就解决了吗？还是只是把"自评"换成了"AI 互评"？——这正是 9.6 评测要正面回答的。
