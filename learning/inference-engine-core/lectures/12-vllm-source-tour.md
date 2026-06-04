# L12 · vLLM 源码导读

## 1 · 顶层文件
```
vllm/
├── engine/
│   ├── llm_engine.py          # AsyncLLMEngine + sync wrapper
│   └── async_llm_engine.py
├── core/
│   ├── scheduler.py            # ⭐ 调度核心
│   ├── block_manager.py        # ⭐ PagedKV 分配
│   └── policy.py               # FCFS / priority
├── attention/
│   └── backends/               # FA / FlashInfer / xformers
├── model_executor/
│   └── layers/
│       ├── sampler.py          # ⭐ sampling fused kernel
│       └── attention.py
├── worker/
│   └── model_runner.py         # 单卡 / TP worker
└── csrc/                       # CUDA / C++ kernels
    └── attention/
        └── paged_attention_kernel.cu
```

## 2 · 关键调用路径
```
LLM.generate()
  └→ engine.add_request()
      └→ scheduler.add_seq_group()
         └→ block_manager.allocate()  ← 分 KV block
  └→ engine.step()                    ← 一个 iter
      ├→ scheduler._schedule()        ← 选 running + swap + prefill
      ├→ model_runner.execute_model() ← forward
      │    └→ attention backend
      └→ sampler.sample()             ← top-k/top-p/repetition
```

## 3 · `_schedule()` 三阶段
1. `_schedule_running()`: 检查 KV ok 的 running，全 forward
2. `_schedule_swapped()`: swapped-out 的 KV ok 时 swap-in
3. `_schedule_prefills()`: 新 admit + chunked prefill

总 token 不超 `max_num_batched_tokens`（默认 8192）

## 4 · `BlockManager`
- `allocator`: free_block_ids 池
- `block_tables[req_id]`: per-request block list
- `can_allocate(seq_group)` 检查 free 是否够
- prefix caching：`compute_hash` + `_cached_blocks`

## 5 · `Sampler`
- `_apply_temperature` / `_apply_top_k_top_p` / `_apply_penalties`
- batched logits → batched sampled token

## 6 · 调试入口
```bash
VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B --enforce-eager
```

`--enforce-eager` 关 CUDA Graph，方便单步 debug。

## 7 · 推荐阅读顺序
1. `engine/llm_engine.py:200-260` `add_request` 入口
2. `core/scheduler.py:_schedule()` 主循环
3. `core/block_manager.py:allocate()` paged 分配
4. `attention/backends/flashinfer.py` 实际 kernel 调用
5. `model_executor/layers/sampler.py` 采样

## 8 · 一句话
> vLLM = **engine + scheduler + block_manager + 3 套 backend + 4 套 sampler**，骨架其实可以 1000 行写完（我们的 capstone 就做这个）。
