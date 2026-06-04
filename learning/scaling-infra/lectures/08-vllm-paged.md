# L08 · vLLM + PagedAttention

> 14 slides | 45 min ⭐⭐⭐⭐⭐

## Slide 1 · vLLM = SOTA LLM 推理

```
Berkeley 2023, 现 1k+ star/月
3-24× throughput vs HF transformers
SGLang / TensorRT-LLM 主要竞品
```

## Slide 2 · 痛点：KV cache 碎片

```
naive: 每 request 预分 max_len KV
  实际用 200 / 32768 → 99% 浪费
  外碎片 + 内碎片
```

## Slide 3 · PagedAttention idea

```
内存按 block (16 token 一块) 分配
逻辑地址 ↔ 物理 block 映射 (类似 OS 页表)
↓
0 浪费 + 共享 block 复用
```

## Slide 4 · 数据结构

```python
class KvBlock:
    physical_id: int
    n_used: int  # 0..16

class Sequence:
    logical_blocks: list[int]  # virtual → physical mapping
```

## Slide 5 · prefix 共享

```
prompt: "Hello, how are you?" → blocks [0, 1, 2]
另一 request 同 prompt → 直接复用同 block (refcount++)
节省 prefix KV 计算 + 显存
```

## Slide 6 · copy-on-write

```
beam search: 一个 prefix 分叉
分叉前共享 blocks
分叉后修改时 copy
```

## Slide 7 · Continuous Batching

```
传统: 等齐 batch 一起跑
↓ slow request 拖整 batch
vLLM: 一旦 request done 立刻 schedule 新的
吞吐 3-10×
```

## Slide 8 · scheduler

```
每 step:
  从 waiting queue 拉新 request
  与 running batch 合并
  跑一个 decoder step
  done request 出队
```

## Slide 9 · 启动

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct",
           gpu_memory_utilization=0.9,
           max_model_len=4096)
out = llm.generate(["Hello"], SamplingParams(max_tokens=100))
```

## Slide 10 · OpenAI-compatible server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --port 8000
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1")
c.chat.completions.create(...)
```

## Slide 11 · TP/PP

```bash
vllm serve model --tensor-parallel-size 4
```

支持 TP/PP/DP，多卡推理。

## Slide 12 · 量化

```python
LLM(model="meta-llama/Llama-3-8B", quantization="awq")
LLM(model="meta-llama/Llama-3-8B", quantization="fp8")
LLM(model="meta-llama/Llama-3-8B", quantization="gptq")
```

显存减半，速度持平/略快。

## Slide 13 · 5090 部署

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --quantization fp8
```

5090 24G ✓ Qwen-7B + 8k ctx + fp8.

## Slide 14 · 总结

```
PagedAttention 解决 KV 碎片
Continuous Batching 解决调度
0.6+ 工业标准 LLM 推理引擎
```

## 参考
- vLLM 2023 (Kwon)
- github.com/vllm-project/vllm
