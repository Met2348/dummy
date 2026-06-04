# L13 · Capstone — 量化动物园

## 1 · 目标
同基座 (Qwen-1.5B mock) 跑 6 个版本，输出 6 列对照表。

## 2 · 6 个版本
1. fp16 (baseline)
2. int8 per-channel
3. GPTQ-4bit (minimal 实现)
4. AWQ-4bit (minimal 实现)
5. FP8 (E4M3)
6. W4A8 (smooth + GPTQ-W + int8-A)

## 3 · 指标
- PPL（mock data）
- 模拟 accuracy
- 显存（理论估算）
- 模拟 tok/s

## 4 · 输出表
```
| variant     | PPL  | acc  | mem(GB) | tok/s |
|-------------|------|------|---------|-------|
| fp16        | 5.68 | 0.45 | 14.0    | 130   |
| int8        | 5.72 | 0.45 | 7.0     | 160   |
| GPTQ-4bit   | 5.85 | 0.44 | 3.5     | 180   |
| AWQ-4bit    | 5.81 | 0.45 | 3.5     | 200   |
| FP8         | 5.70 | 0.45 | 7.0     | 220   |
| W4A8        | 5.95 | 0.43 | 3.5     | 280   |
```

## 5 · 实现：[capstone_quant_zoo.py](../src/capstone_quant_zoo.py)
- 6 个 mock 量化 wrapper
- `run_all()` 输出 markdown + json
- 与真实数字相符（取自 paper）

## 6 · 真模型验证（可选）
```bash
# 量 Qwen-1.5B 4bit
python -c "
from awq import AutoAWQForCausalLM
m = AutoAWQForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B')
m.quantize(tokenizer, {'w_bit':4, 'q_group_size':128})
m.save_quantized('./qwen-awq')
"
# 部署
python -m vllm.entrypoints.openai.api_server \
    --model ./qwen-awq --quantization awq
```

## 7 · 退出条件
- 6 版本表完成
- 至少 1 个版本能在 vLLM 跑通（可选）
- README 中包含 6 列表
