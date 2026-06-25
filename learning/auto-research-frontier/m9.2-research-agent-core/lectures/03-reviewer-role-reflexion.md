# L03 · Reviewer 角色：批判为什么要"职责分离"

## 1. 自己批判自己 ≠ 批判

如果让生成 idea 的同一个模型、在同一段上下文里"顺手反思一下"，它倾向于**替自己圆场**——
这正是 9.5 演示过的 grading-its-own-homework，也是 co-scientist 为什么要把
Generation 和 Reflection 拆成**不同 agent**的原因。

本模块用最小形式体现"职责分离"：

- **Researcher**：`MockLLM`，只管生成草稿（还会幻觉引用）。
- **Reviewer**：`critic.review`，**不生成、只挑错**——查幻觉引用、查缺 baseline、查吹新颖度。

`critic.py` 里 Reviewer 的判据是可检查的、对作者不留情面的：

```python
if ungrounded:              flag("幻觉引用")
if not draft.has_baseline:  flag("缺 baseline：没有对照，结论无法判强弱")
if novelty>=0.7 and grounded<2:  flag("新颖度自评过高，接地支撑薄弱")  # ← 预告 9.3
```

## 2. Reflexion 回路：critique 要能改变结果，才算数

一个只会说"看起来不错"的 reviewer 是装饰品。检验它是否真在工作的唯一标准是：
**它的批判改变了最终产物吗？** `test_critic_changes_the_plan` 就断言：

```python
plan_no_critic["citations"] != plan_with_critic["citations"]   # 引用集必须不同
```

`revise` 拿到 critique 后会**真的删掉**幻觉引用、补上 baseline、下调虚高新颖度。
没有这一步闭合，reviewer 写得再漂亮也等于没存在。

## 3. 第三条 flag 是给 9.3 埋的线

注意 Reviewer 那条"新颖度自评 0.85 过高、但只有薄弱接地支撑"的批评——
这就是 **ideation-execution gap** 的前兆：一个 idea 自我感觉很新，证据却撑不起来。
9.2 在这里只是**标记**这个风险；9.3 会把它放大成一个完整的 idea 锦标赛，
并让你亲眼看到"自评最高的 idea 真做出来反而最差"。

## 4. 动手

1. 把 `critic.review` 整个换成一个"永远说 OK"的假 reviewer（返回空 flags），
   重跑 `test_critic_changes_the_plan`——它挂了吗？这个测试在保护什么？
2. 思考：本模块的 Reviewer 和 Researcher 用的是同一个进程里的不同函数。
   如果它们其实调用**同一个真 LLM**（只是 prompt 不同），"职责分离"还成立吗？
   真正的独立，要到什么程度才算独立？（带着这个问题去 9.6 评测 / 9.8 红队。）
