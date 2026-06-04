# L06 · Batch vs Online 推理

## 1 · 两类负载
| | Batch | Online |
|---|------|--------|
| latency | 慢 (分钟-小时) | 快 (亚秒) |
| 用户 | API 任务 / 离线分析 | 实时 chat |
| 优化 | 吞吐 max | latency min |
| 价格 | **便宜 50%** (OpenAI batch API) | 正常 |

## 2 · Batch 优化
- 大 batch (256+)
- 全 prefill 一起，全 decode 一起
- 无流式输出
- 后台慢慢跑

## 3 · Online 优化
- 小 batch (1-8)
- chunked prefill
- CUDA graph
- SSE 流式

## 4 · 实际配置
```python
# Batch
llm = LLM(max_num_seqs=256, enable_chunked_prefill=False)

# Online
llm = LLM(max_num_seqs=16, enable_chunked_prefill=True, enforce_eager=False)
```

## 5 · OpenAI batch API
- 50% 价格
- 24h 完成保证
- 适合：数据标注、大规模 evaluation、内容生成
