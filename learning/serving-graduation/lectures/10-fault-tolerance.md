# L10 · 容错

## 1 · 常见故障
- GPU OOM (KV 太满)
- worker 死 (CUDA error / OOM)
- 网络抖动 (LB 路由错)
- model output 异常 (重复 / NaN)

## 2 · 处理
| 故障 | 处理 |
|------|------|
| OOM | preempt + recompute / retry |
| worker 死 | 健康检查 + 自动重启 |
| 网络 | retry with backoff |
| 异常 output | safety filter + retry |

## 3 · circuit breaker
- 错误率超阈值 → 暂停该 worker
- 半开测试 → 恢复

## 4 · fallback
- 主路径失败 → 次模型回退
- vLLM OOM → 用 ollama 单卡
- 商业 API down → 切开源

## 5 · 重试策略
- 立刻 retry（瞬态）
- 指数 backoff（持久）
- circuit breaker（防雪崩）
- 最多 N 次（防永久重试）

## 6 · 实现示意
- nginx upstream + retry
- FastAPI middleware
