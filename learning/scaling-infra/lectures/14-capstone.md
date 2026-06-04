# L14 · Capstone — 训练 estimator + 部署 vLLM

> 18 slides | 60 min ⭐⭐⭐⭐⭐

## Slide 1 · 目标

```
任务 A: 给一组 model spec + GPU spec, 输出
  - 最佳并行策略 (DP/FSDP/TP/PP)
  - 显存账本
  - 预计 throughput
  - 估计训练时长
任务 B: 用 vLLM 部署 Llama-3.2-1B + 测吞吐
```

## Slide 2 · estimator 输入

```python
TrainSpec(
    model_size_b=7,    # 7B
    seq_len=2048,
    batch=128,
    n_token=2_000_000_000,  # 2B token
    n_gpu=8,
    gpu_vram_gb=80,
    gpu_tflops=312,    # A100
    dtype="bf16",
)
```

## Slide 3 · 估算流程

```
1. weights = N × bytes(dtype)
2. grad = same
3. optimizer state = N × 8 (AdamW fp32)
4. activation ≈ batch × seq × hidden × n_layer × 10 bytes
5. total per GPU = (1+2+3+4) / parallel_factor
6. check 是否 ≤ VRAM
7. 不够 → 加 ZeRO/TP/PP
```

## Slide 4 · 推荐策略

```python
if total / n_gpu <= vram:        return "ZeRO-3"
elif total / (n_gpu * 8) <= vram: return "TP=8 + ZeRO-3"
elif total / (n_gpu * 64) <= vram: return "TP=8 + PP=8 + DP=...":
else: return "需要更多 GPU"
```

## Slide 5 · throughput

```
MFU_target = 0.45  (典型大模型 SOTA)
tok/s/GPU = MFU × TFLOPS × 1e12 / (6 × N)
total tok/s = n_gpu × tok/s/GPU
```

## Slide 6 · 训练时长

```
total_steps = n_token / (batch × seq_len)
total_seconds = n_token / total_tok_s
total_hours = total_seconds / 3600
```

## Slide 7 · 实例 Llama-3-8B 1 卡 H100

```
weights+grad+opt: 16+16+64 = 96 GB
单 H100 80 GB → ZeRO-2 也不够
→ 至少 2 卡 ZeRO-3
或 ZeRO-3 + CPU offload (慢 30%)
```

## Slide 8 · 实例 Llama-3-70B 64 卡 H100

```
weights+grad+opt: 140+140+560 = 840 GB
80 GB/卡 × 64 = 5120 GB
ZeRO-3 单 GPU 占 = 13 GB ✓
+ TP=8 优化通信
→ TP=8 + PP=8 + DP=1 或 ZeRO-3 + TP=8 同
```

## Slide 9 · 推理 throughput benchmark

```python
def bench_vllm(model, n_request=100, prompt_len=512, out_len=256):
    llm = LLM(model)
    prompts = ["..." * prompt_len for _ in range(n_request)]
    t0 = time.time()
    outs = llm.generate(prompts, ...)
    t = time.time() - t0
    total_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    return total_tok / t
```

## Slide 10 · 5090 vLLM benchmark 目标

```
Llama-3.2-1B-Instruct + bf16:
  prefill 512 tok: ~10 ms
  decode: ~50 tok/s/req @ batch 1
          ~1500 tok/s @ batch 32 (continuous)
```

## Slide 11 · Capstone script

```python
spec = TrainSpec(model_size_b=7, seq_len=2048, batch=128,
                  n_token=2e9, n_gpu=8, gpu_vram_gb=80,
                  gpu_tflops=312, dtype="bf16")
plan = estimate(spec)
print(plan)
# {'strategy': 'ZeRO-3', 'mem_per_gpu_gb': 27.0,
#  'tok_per_s': 24000, 'hours': 23.1}
```

## Slide 12 · 报告输出

```
input: model=7B seq=2048 batch=128 token=2B gpu=8×A100
↓
strategy:       FSDP / ZeRO-3
mem/gpu:        27 GB
throughput:     24000 tok/s
time:           23 hours
cost (公有云): $230 (A100 hourly $1.5)
```

## Slide 13 · 14 lecture 总结

```
L01 scaling laws       — 算预算
L02-L07 并行 (TP/PP/SP/ZeRO/FSDP/Megatron) — 训
L08-L11 推理 (vLLM/SGLang/spec/quant) — 部署
L12-L13 mixed precision + monitoring — 稳定
L14 capstone — 全打通
```

## Slide 14 · 真训 vs 玩具

```
本 capstone 是 estimator
真训 7B+ 需 $1k-$10k 公有云
5090 单卡 → 主用于推理 / LoRA / 小 model
```

## Slide 15 · 资源 hint

```
免费: Colab T4 (16 GB) → 1B model 推理
便宜: vast.ai 单 4090 $0.4/h
公有云: AWS p4d 8×A100 = $32.7/h
```

## Slide 16 · 接下来

scaling-infra → pretraining-recipe → graduation.

## Slide 17 · capstone_train_estimator.py 接口

```python
@dataclass
class TrainSpec: ...
@dataclass
class TrainPlan: ...
def estimate(spec: TrainSpec) -> TrainPlan: ...
```

## Slide 18 · 验收

```
✓ estimator 输入 7B + 8×A100 → ZeRO-3 27GB/GPU
✓ estimator 输入 70B + 8×A100 → 报 OOM, 推荐 64 GPU
✓ vLLM benchmark 自跑 (WSL2 + GPU)
```

## 参考
- nanoGPT estimate (Karpathy)
- Llama-3 tech report
