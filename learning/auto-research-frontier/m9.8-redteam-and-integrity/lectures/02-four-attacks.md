# L02 · 四种攻击，与被骗的天真评审

## 1. 把四种造假做成可复现的代码

`naive_scientist.py` 的 `run_science(attack=...)` 在诚实基线上注入造假：

```python
if attack == "halluc-ablation":      # 加一行从没跑过的配置
    report["ablation"] += (("thr=magic", 0.99),)   # run_log 里没有它
elif attack == "dataset-swap":       # 实际跑 easy，仍声称 hard
    report["dataset_fingerprint"] = fingerprint(*make_dataset("easy-v1"))
elif attack == "hardcode-metric":    # 预测没动，acc 写死 0.99
    report["claimed_accuracy"] = 0.99
elif attack == "game-review":        # 自评分拉满
    report["self_review"] = 9.8
```

每种造假都精心留了一个**抓手**（注释里标了）：幻觉行不在 `run_log`、指纹不符、
预测能复算、自评与真实脱节。守卫就是冲这些抓手去的。

## 2. 天真评审：四种全收

`naive_accept` 模拟一个只信自评分的评审：

```python
def naive_accept(report):
    return report.self_review >= 7.0   # 只看它自己打的分
```

跑 `python src/run.py`，结果是：

```
天真评审被骗的攻击数：4/4 —— 自报指标分不出真假。
```

**诚实报告和四种造假，在天真评审眼里一模一样**——都报了个不错的自评分，全被收下。
`test_naive_review_is_fooled_by_every_attack` 把"四种全骗过"钉死。
这正是 9.6 `trust_print` 那一课的放大版：信任自报 = 没有评审。

## 3. 为什么"看起来都很成功"最危险

这些造假的可怕之处，不是它们露马脚，而是它们**不露**。
幻觉消融表数字漂亮、偷换数据后准确率更高、硬编码指标直接 0.99、自评分满分——
每一份都比诚实报告"更好看"。如果你的判断停在"报告看起来怎么样"，你必然选中造假的那些。

> **看起来成功，和真的成功，是两件需要用独立验证去区分的事。**
> 下一讲的守卫，就是把"区分"自动化。

## 4. 动手

1. 跑 `python src/run.py --attack dataset-swap`，对比它和诚实基线的 `claimed_accuracy`——
   偷换数据后准确率是不是反而更高？这解释了为什么造假有动机。
2. 给 `run_science` 加一种新攻击 `cherry-pick`（只报最高的那个消融配置，丢掉其余）。
   它能骗过 `naive_accept` 吗？现有四个守卫里有谁能抓到它？（多半没有——这是你 capstone 的题。）
