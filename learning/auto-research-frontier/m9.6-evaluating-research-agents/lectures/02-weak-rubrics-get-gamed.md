# L02 · 弱 rubric 怎么被刷（以及 safe_exec 这块地基）

## 1. 三种候选 = 三种"研究 agent 交上来的实现"

`candidates.py`：

- **honest**：真实现规则 `(x+y)>0`，能泛化。
- **hardcode**：把**可见样本**背成查找表，没见过的猜 0——这是 benchmark gaming 的经典形态。
- **print-fraud**：根本不算，直接 `print("ACCURACY=1.0")`，函数恒返回 0——幻觉结果。

## 2. 判分矩阵把"刷"演给你看

```
candidate        visible_only(弱)   trust_print(弱)    heldout(强)
honest               1.00 PASS        0.00 fail       1.00 PASS
hardcode             1.00 PASS        0.00 fail       0.38 fail
print-fraud          0.50 fail        1.00 PASS       0.38 fail
```

- `visible_only` 只在候选**看得到**的样本上判分 → 硬编码把它刷爆（1.00），但这只是背书。
- `trust_print` 信任候选自己打印的指标 → 被造假刷（1.00）。**更糟的是它毙掉了诚实候选**
  （诚实的没吹牛、没 print 那个数，反而 0.00 不过）。**信任自报 = 奖励说谎者、惩罚老实人。**
- `heldout` 在候选**没见过**的数据上真跑 → 只有诚实的过。

`test_trust_print_rubric_is_inverted` 把"放行造假、毙掉诚实"这对反转钉死。

## 3. 这一切的前提：sandbox 真在跑代码

如果判分不是真把候选跑起来，而是"看它说自己多少分"，那就是 `trust_print` 的灾难。
`sandbox.py` 沿用 agent-code-eval 的 `safe_exec`：**白名单 builtins + 禁用模式黑名单**，
真 `exec` 候选源码：

```python
for pat in FORBIDDEN:                 # import / __ / open( / eval( / os. ...
    if pat in src: raise SafeExecError(...)
exec(compile(src, "<candidate>", "exec"), {"__builtins__": safe_builtins})
```

越权候选 `import os` 被直接拦下（`test_sandbox_blocks_forbidden`）。
而诚实候选过沙箱后，`classify(5,-2)` 真能算出 1（`test_sandbox_runs_real_code`）——
**真跑，才有判分的资格。**

## 4. 教训

> **评测的强弱，全在"用什么数据、信不信自报"。**
> 只看可见样本 → 被记忆刷；信自报指标 → 被嘴炮刷。
> 唯一抗刷的最低门槛：在候选没见过的数据上，由你亲自真跑。

## 5. 动手

1. 给 `print-fraud` 改成"既 print 假指标、又在可见样本上背答案"（合并两种刷法）。
   它现在能同时骗过 `visible_only` 和 `trust_print` 吗？`heldout` 还拦得住吗？
2. 把 `sandbox.py` 的 `FORBIDDEN` 删掉 `"import"`，重跑 `test_sandbox_blocks_forbidden`——
   它挂了吗？体会"评测地基的安全性"和"判分正确性"是两件都不能少的事。
