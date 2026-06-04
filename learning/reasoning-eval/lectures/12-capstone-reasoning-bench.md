# L12 · Capstone — 推理 bench 2-model 对照

## 目标

用 2 个 mock model（baseline vs R1-tiny）跑 5 个推理 bench，
生成 2 × 5 markdown 表 → 体会"训练投入 → bench 收益"映射。

## 设计

```
src/capstone_reasoning_compare.py
├── make_gpt2_baseline()  → dummy "0" answer
├── make_r1_tiny()        → gold-map mock
├── run_all(model)        → 5 bench acc
└── run_capstone()        → 完整 2x5 表
```

## 跑

```bash
python -c "import sys; sys.path.insert(0,'src'); \
  from capstone_reasoning_compare import run_capstone; \
  print(run_capstone()['table'])"
```

预期输出（精简）：
```
| ckpt | gsm8k | math | aime | gpqa | zebra | avg |
|---|---|---|---|---|---|---|
| gpt2_base (vanilla)  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| r1_tiny (RL-trained) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
```

## 想表达什么

1. **vanilla LM 在推理 bench 上 ≈ 0**：未 RL 之前完全做不了
2. **RL 训练 +100pp**：当然是 mock，但映射到真 R1 也是 +20-50pp
3. **5 bench 覆盖 4 类**：数学（gsm8k/math/aime）+ 科学（gpqa）+ 逻辑（zebra）

## 推广到真模型

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
mdl = AutoModelForCausalLM.from_pretrained(...).cuda()

def real_r1(prompt: str, max_new_tokens: int) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    out = mdl.generate(ids, max_new_tokens=max_new_tokens, temperature=0.7)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

# 跑
res = run_all(real_r1)
```

## 退出条件

- 5 个 bench 都返回 0.0-1.0 之间
- 2x5 表生成
- baseline avg < 0.2，oracle avg = 1.0
- 测试 PASS

## 一句话

> 5 bench 联跑 = "推理高考成绩单"，能区分基线模型与 R1。
