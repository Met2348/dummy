# L11 · 服务工程 5 原则

## 1 · KV 是金
所有优化围绕 KV cache：
- PagedAttention 解决碎片
- Prefix caching 解决重复
- 量化解决空间
- 跨节点解决容量

## 2 · 调度即性能
- continuous batching > static batch
- chunked prefill > naive
- Cuda Graph 减 overhead
- 调度比 kernel 更影响 throughput

## 3 · 选 backend 看场景
- 单 chat → vLLM
- agent → SGLang
- 极致速度 → TRT-LLM
- 端侧 → llama.cpp/Ollama

## 4 · 量化 + 投机 是免费午餐
- AWQ 4bit 几乎无损精度
- EAGLE-2 加速 2-4x
- 不上就是浪费

## 5 · 监控 + 成本是 hidden game
- 没监控不知道 dropping
- 没成本意识每月烧几万
- 这两条决定能不能持续运营

## 6 · 一句话
> 推理服务 = **KV 大小 + 调度方法 + backend 选型 + 量化 + 监控成本**。五件套缺一不可。
