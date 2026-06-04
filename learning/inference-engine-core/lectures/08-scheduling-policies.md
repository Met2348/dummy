# L08 · 调度策略（scheduling policies）

## 1 · 决策点
进入 `step()` 时：哪些 pending 进 running？哪些 running 暂停？哪些 prefill 切多大 chunk？

## 2 · 策略矩阵
| 策略 | 入队 | 抢占 | 适用 |
|------|------|-----|-----|
| FCFS (FIFO) | 先到先入 | 不抢 | 公平 / 默认 |
| SJF | 短请求优先 | 不抢 | 降平均延迟 |
| Priority | 高 priority 优先 | 可抢低 | 多租户 |
| EDF (deadline) | 截止前优先 | 紧迫抢 | SLA |
| max-throughput | 估时累计最大 | 不抢 | 离线批 |

## 3 · vLLM 的 `_schedule()`
- 先 `_schedule_running()`：已 admitted 全 forward
- 再 `_schedule_swapped()`：之前 OOM 换出的尝试 swap-in
- 最后 `_schedule_prefills()`：admit new
- 三阶段都不出超过 `max_num_batched_tokens` 总预算

## 4 · 抢占（preemption）
- vLLM 两种：**recompute** 或 **swap to CPU**
- recompute：丢 KV，下次重新 prefill (省 IO 多 compute)
- swap：KV 写 CPU，回来再读 (省 compute 多 IO)
- 默认 swap，CPU 显存 4x GPU 时切 recompute

## 5 · starvation 防御
- max wait time → 强制 promote
- guaranteed slot：fairness ratio

## 6 · 长 prompt SLO
- chunk prefill 自然让长 prompt 不阻塞
- 但 TTFT 累积变长 → 监控 p99 TTFT

## 7 · 实现：[scheduling_policies.py](../src/scheduling_policies.py)
- FCFS / SJF / priority 三种 picker
