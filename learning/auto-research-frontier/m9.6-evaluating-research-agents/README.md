# 9.6 · 评测 Research Agent（Evaluation as the Moat）

> 怎么知道一个 research agent"**真做出了研究**"而不是**刷了分**？这一关是整个领域的方法论护城河。
> 本模块给同一份任务配三种 rubric，让你亲眼看到**弱 rubric 被刷、强 rubric 抗刷**，
> 且评测地基是你验证过的 `agent-code-eval` 真沙箱。

## 一句话结论（本模块跑出来的判分矩阵）

```
candidate        visible_only(弱)   trust_print(弱)    heldout(强)
honest               1.00 PASS        0.00 fail       1.00 PASS
hardcode             1.00 PASS        0.00 fail       0.38 fail   ← 背下可见样本，held-out 露馅
print-fraud          0.50 fail        1.00 PASS       0.38 fail   ← 只 print 一个假指标
```

- **硬编码**候选把可见样本背成查找表 → `visible_only` 给它满分，`heldout` 一上没见过的就 0.38。
- **print 造假**候选根本不算、只 `print("ACCURACY=1.0")` → `trust_print` 信了它。
- **只有 `heldout`（强）在三个候选里只放行诚实**。
- 越权候选（`import os`）被沙箱直接拦下——**评测的第一性原理是先能安全地真跑候选**。

## 三类基准（领域全景）

真实的 research-agent 基准分三类，越往后越难刷：

| 类 | 代表 | 判什么 |
|----|------|--------|
| 复现类 | PaperBench / CORE-Bench | 能不能复现一篇论文的结果 |
| 工程类 | MLE-bench / **RE-Bench** | 在 Kaggle/ML 工程任务上的真实得分（排行榜独立评） |
| 开放研究类 | MLR-Bench / MLGym / ScienceAgentBench | 开放问题，最难自动判 |

[RE-Bench](https://arxiv.org/abs/2411.15114) 的关键发现：**时间预算决定人机胜负**——
2 小时 AI 强，32 小时人类反超 2 倍。评测设计里"给多久"本身就是变量。

## 跑起来

```powershell
python src/run.py        # 3 候选 × 3 rubric 判分矩阵 + 沙箱拦越权

python scripts/eric_3080ti_env_audit.py --runbook --tests `
  --modules auto-research-frontier/m9.6-evaluating-research-agents `
  --json-out $env:TEMP/m9.json --md-out $env:TEMP/m9.md
```

## 目录

```
m9.6-evaluating-research-agents/
├── runbook.yaml
├── lectures/
│   ├── 01-eval-is-the-moat.md          为何评测是护城河 + 三类基准 + RE-Bench
│   ├── 02-weak-rubrics-get-gamed.md    刷分演示 + safe_exec 地基
│   └── 03-designing-resistant-benchmarks.md  怎么设计抗刷基准
└── src/
    ├── run.py
    ├── mini_eval/
    │   ├── sandbox.py      真 exec 沙箱(白名单 builtins + 禁用黑名单，接 agent-code-eval)
    │   ├── task.py         method spec：VISIBLE 给看 / HELDOUT 私藏
    │   ├── candidates.py   3 候选(诚实/硬编码/print造假) + 1 越权候选
    │   ├── rubrics.py      visible_only / trust_print(弱) · heldout(强)
    │   └── evaluate.py     沙箱 + rubric 接起来
    └── tests/tests_eval... 7 测试：沙箱真跑/拦越权、弱被刷、强只放行诚实
```

## Hands-on（轮到你）

`rubrics.py` 里 `heldout` 是目前最强的，但它够强吗？

1. **写一个能骗过 heldout 的候选**：有没有办法在不真正实现规则的情况下，
   也在 held-out 上拿高分？（提示：如果候选能猜出规则结构……这暴露了"任务太简单"也是评测漏洞。）
2. **加一条行为检查**：除了准确率，再查"classify 对 (x,y) 和 (y,x) 是否一致"等不变量——
   行为约束往往比单一指标更难刷。
3. **加独立 verifier**：让一个**和出题方不同**的逻辑重算分数（呼应 9.2 角色分离 / 9.4 忠实度核查）。

## 桥接

- **直接扩展 agent-code-eval**（SWE-bench/HumanEval 的 `safe_exec` 就是这里的地基）· eval-foundations · reasoning-eval。
- 呼应 9.3（评判被刷）/ 9.4（忠实度核查）；为 **9.8 红队**提供"刷分攻击"的靶子与守卫。
