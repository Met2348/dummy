# L12 · Capstone — 4-bench 联跑 pipeline

## 目标

把 Topic 1 全部内容串起来：MMLU + BBH + TruthfulQA + Commonsense 联跑，
输出 markdown 报告 + ASCII 雷达。

## 设计

```
src/eval_pipeline.py
├── run_all_benches(model)       # 调 4 个 runner
├── ascii_bar(value)             # 横条可视化
└── to_md(results, label)        # markdown 报告
```

## 跑 random baseline

```python
from common import make_random_model
from eval_pipeline import run_all_benches, to_md

m = make_random_model(seed=0)
res = run_all_benches(m)
print(to_md(res, "random_baseline"))
```

输出（精简）：
```
# Eval pipeline report — `random_baseline`

| benchmark   | accuracy | bar |
|---|---:|---|
| mmlu        | 0.083    | `[##                  ]  8.3%` |
| bbh         | 0.333    | `[#######             ] 33.3%` |
| truthfulqa  | 0.500    | `[##########          ] 50.0%` |
| commonsense | 0.333    | `[#######             ] 33.3%` |

**Overall average:** 0.313 (31.3%)
```

(注：seed 影响 random baseline 数值；预期都接近 chance level)

## 跑 oracle model

```python
from common import make_mock_model
from mmlu_runner import build_samples as mmlu_samples
# ... 凑齐 4 个 runner 的 gold 字典 ...

oracle = make_mock_model(merged_gold_map)
res = run_all_benches(oracle)
# Expected: 100% / 100% / 100% / 100%
```

## 可以替换的位置

注意 model 是 `ModelFn = Callable[[str, int], str]`。
要替换为真模型：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
mdl = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").cuda()

def real_model(prompt: str, max_new_tokens: int) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    out = mdl.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

res = run_all_benches(real_model)
```

## 退出条件

- 4 个 bench 都跑出非零结果
- to_md 含 4 行 + Overall average
- random baseline ≈ chance（24-35% 区间）
- oracle = 100%

## 一句话

> 4 个 runner + 1 个 pipeline = 一份评测报告 — 这就是 lm-eval-harness 的玩具版。
