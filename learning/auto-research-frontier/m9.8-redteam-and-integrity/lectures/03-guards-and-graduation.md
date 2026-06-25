# L03 · 四个守卫、可复现性，与系列收口

## 1. 四个守卫，每个都是前面某一课

`guards.py` 的四个守卫，不是临时想的，而是把整个 M9 的教训具体化：

```python
guard_provenance        # 消融每行的 config 必须在 run_log（真跑过）
guard_dataset           # 实际数据指纹 == 声称数据集的官方指纹
guard_metric            # 从保存的 predictions 独立复算 acc，必须等于报告值
guard_independent_review# 无视自评，用独立逻辑从真实指标打分
```

| 守卫 | 抓哪种攻击 | 它就是哪一课 |
|------|-----------|------------|
| provenance | 幻觉消融表 | **9.2 接地**：数字要能回查到真实运行 |
| dataset | 偷换数据集 | **9.4 忠实**：声称的对象要名副其实 |
| metric | 硬编码指标 | **9.6 独立复算 / 9.7 别信代理** |
| independent_review | 刷自评 | **9.3 自偏好 / 9.5 grading-own-homework** |

跑 `python src/run.py`：诚实报告四个守卫全过，四种攻击各被对应守卫戳穿
（硬编码还被 metric + independent_review 双杀）。`test_graduation_property` 把
"诚实可信 AND 全部攻击被抓"钉成毕业判定。

## 2. 可复现性守卫的三件套（Hidden Pitfalls 的建议）

把上面四个守卫归纳成你做真研究时该执行的纪律：

1. **artifact + 执行日志**：提交代码、数据、随机种子、**被丢弃的实验**——provenance 的来源。
2. **数据/对象指纹**：给数据集、checkpoint 打 hash，别让"我用的是 X"无法核验。
3. **独立验证**：让一个**不参与生成**的 verifier 重算指标、重判质量——别让作者给自己打分。

这三件套合起来，就是"别信它自己说做出了什么，去独立复算它"。

## 3. 整个 M9 在这里收口

```
9.1 自主性≠可信度  ┐
9.2 接地(grounding) │
9.3 自偏好          ├─►  9.8 四个守卫 = 把这些教训变成"会拒绝造假"的代码
9.4 引用忠实        │
9.6 独立验证        │
9.7 reward hacking ┘
```

> **一句话**：所有自动科研的可信度，最终都压在"独立验证"这一环。
> 系统能 ideation、能 execution、能 writeup、能 self-review——但只有一个**够不到自己**的
> 独立验证，能决定那些产出算不算数。这就是老师那句"research 是下一个前沿"里，
> 真正难、也真正有价值的部分。

## 4. 毕业答辩（capstone）

1. **把守卫接到 9.5**：给 `m9.5` 的 `mini_ai_scientist` 挂上本模块的 `guard_metric` /
   `guard_independent_review`——让它报告里的数字被独立复算，让那个可被刷的 `review` 不再算数。
2. **写诚实报告**：用一段话回答"research 被 Agent 接管到什么程度？"，
   用你这 7 个模块的真实证据（自主性反相关可信度、ideation-execution gap、reward hacking、
   引用存在≠忠实……）支撑，并诚实标注"它会怎么骗你"。
3. **毕业标准**：你的 mini-AI-Scientist 既能跑通（9.5），又扛得住自己的红队（9.8）。
   到这一步，你对这个领域的判断力，已经不靠别人的标题，而靠你亲手跑过、亲手戳穿的证据。
