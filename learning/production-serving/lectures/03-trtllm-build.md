# L03 · TRT-LLM Build 实战

## 1 · 完整流程
```bash
# 1. 安装
pip install tensorrt-llm

# 2. convert HF model → TRT-LLM checkpoint
python convert_checkpoint.py \
    --model_dir Qwen/Qwen2.5-7B \
    --output_dir ./qwen-ckpt \
    --dtype float16

# 3. build engine
trtllm-build \
    --checkpoint_dir ./qwen-ckpt \
    --output_dir ./qwen-engine \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 32 \
    --max_input_len 4096 \
    --max_output_len 1024 \
    --max_num_tokens 8192 \
    --use_paged_context_fmha enable \
    --use_fp8_context_fmha enable

# 4. run
mpirun -n 1 python ../run.py \
    --engine_dir ./qwen-engine \
    --tokenizer_dir Qwen/Qwen2.5-7B \
    --input_text "Hello"
```

## 2 · 量化 build
```bash
# AWQ
python ../quantization/quantize.py \
    --model_dir Qwen/Qwen2.5-7B \
    --output_dir ./qwen-awq-ckpt \
    --qformat int4_awq \
    --calib_size 32

trtllm-build --checkpoint_dir ./qwen-awq-ckpt \
    --output_dir ./qwen-awq-engine \
    --use_weight_only ...
```

## 3 · TP build
```bash
trtllm-build --tp_size 8 ...
```
engine 会输出 8 个 `rank0.engine` ... `rank7.engine`。

## 4 · build cache
- `~/.trtllm/cache/` 缓存编译产物
- 同模型不同参数复用 nvbuild
- 7B 重 build 30s（缓存命中）

## 5 · 常见错误
- shape 超出 `max_input_len` → OOM
- 显存不够 → 减 `max_batch_size`
- 不同卡 build 不通用 → 在目标卡上 build

## 6 · 实现：[trtllm_build.py](../src/trtllm_build.py)
- build script 模板
- 不依赖 TRT-LLM 库（教学版）
