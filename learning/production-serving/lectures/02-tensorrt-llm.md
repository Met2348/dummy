# L02 · TensorRT-LLM 概览

## 1 · NVIDIA 官方推理栈
- 编译模型 → engine.plan 二进制
- 集成 FlashAttention v3 + FP8 + W4A16 + in-flight batching
- 比 vLLM 在 H100 上常快 1.5-2x

## 2 · 工作流
```
HF model
  ↓ convert
TRT-LLM checkpoint
  ↓ build (trtllm-build)
engine.plan (二进制 GPU 专属)
  ↓ run
TRT-LLM runtime / Triton
```

## 3 · build 参数
```bash
trtllm-build \
    --checkpoint_dir ./qwen-trtllm \
    --output_dir ./qwen-engine \
    --gemm_plugin float16 \
    --max_batch_size 32 \
    --max_input_len 4096 \
    --max_output_len 1024 \
    --use_paged_context_fmha enable
```

build 时间：7B ≈ 10 min / 70B ≈ 1 h

## 4 · in-flight batching
TRT-LLM 内置的 continuous batching，与 vLLM 等价但 kernel 更优。

## 5 · 性能数字（H100, 7B fp16）
| 引擎 | tok/s |
|------|------|
| vLLM 0.5 | 4200 |
| **TRT-LLM** | **5800** |

H100 FP8：TRT-LLM 9000 tok/s。

## 6 · 缺点
- engine binary 与 GPU 绑定（H100 build 在 A100 跑不了）
- 量化 build 复杂
- 不支持新 SOTA 算法（如 EAGLE-2）

## 7 · 何时用
- H100/H200/B100 大规模 production
- 需要绝对最高 throughput
- 模型架构稳定（少更新）

## 8 · 实现
本课无 minimal 代码；提供 build script 模板。
