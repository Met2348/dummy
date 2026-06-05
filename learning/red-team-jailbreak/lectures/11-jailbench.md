# L11 · JailbreakBench / HarmBench — 标准化评测

## JailbreakBench (Chao 2024)

- **200 harmful prompts** × 6 类（CWE-style）
  - Harmful (illegal advice)
  - Sexual / Violent
  - Cybercrime
  - Misinformation
  - Privacy
  - Toxicity
- **judge**：Llama Guard / GPT-4 二选一
- **公开 leaderboard**

## HarmBench (Mazeika 2024, CMU)

- **510 prompts** + 18 攻击方法 baseline
- 6 类同 JailbreakBench
- 强调**复现性**：每攻击有官方 implementation
- 含 **HarmBench-Open** 子集（不含 minor 风险内容）

## 评测公平性原则

```
For each (attack_method, target_model):
  total_attempts = 200
  success_count = ?
  ASR = success_count / total_attempts
```

**公平**需要：
- 同 query 集
- 同 judge
- 同 max_attempts 预算
- 同 random seed (where applicable)

## 防御 leaderboard

报告**两个 ASR**：
1. **No defense ASR**：原模型
2. **With defense ASR**：加 Llama Guard / Constitutional Classifier 后

差值 = 防御有效性。

## 2025 状态

| Target | Avg ASR (no def) | + Llama Guard 3 | + Constit. Cls |
|--------|------------------|------------------|----------------|
| Vicuna-7B | 80% | 30% | 10% |
| Llama 3.1 70B | 25% | 12% | 4% |
| GPT-4o | 18% | 8% | — |
| **Claude 3.7** | **5%** | — | **0.4%** |

## 我们的 bench (toy)

src/jailbench_runner.py：
- 4 methods (gcg/pair/autodan/crescendo)
- 5 harmful queries
- 输出 markdown 表格

```python
from jailbench_runner import run_jailbench, to_md
from common import make_safe_target

vuln = make_safe_target("vuln", jb_keys=["{!}"])
asr = run_jailbench(vuln)
print(to_md("vuln", asr))
```

## 工程注意

- **数据隔离**：jailbench 题不能进训练集
- **复现 randomness**：seed 固定
- **Cost 控制**：4 method × 200 q × N attempts = 大量 API 调用
- **judge 一致**：换 judge 数字变

## 一句话

> JailbreakBench / HarmBench = 红队报告卡的标准化模板。
