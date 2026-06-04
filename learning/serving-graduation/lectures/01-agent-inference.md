# L01 · Agent 推理特化

## 1 · agent 推理 vs 单轮 chat
| 维度 | chat | agent |
|------|------|-------|
| 轮次 | 1 | 5-20 |
| prefix 共享 | 系统 prompt | 系统 + 历史 + tool 输出 |
| latency 敏感 | 中 | **极高**（人等不动）|
| reasoning | 短 | 长（thinking）|

## 2 · 优化 lever
| lever | 效果 |
|-------|-----|
| prefix cache (radix) | 历史 token 不重 prefill |
| KV cache 跨轮持久 | 每轮 0 prefill (除了新 user input) |
| tool routing 异步 | tool 执行不阻塞下一 LLM call |
| reasoning cache | 同问题 thinking 复用 |

## 3 · 经验数字
- naive multi-turn: 5 轮 = 5 个独立请求
- 优化后: 1 个会话 + 渐进式 KV
- speedup: 3-5x

## 4 · 实现：[agent_inference_demo.py](../src/agent_inference_demo.py)
- 模拟 5 轮 agent
- 对比 naive vs radix-cached
