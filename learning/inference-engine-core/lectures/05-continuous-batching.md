# L05 · Continuous Batching（Orca OSDI'22）

## 1 · 痛点
传统 batch：所有请求一起 prefill，一起 decode，最长的没结束 → 其他全空转。

```
req A: ████████░░░░░  (8 tok done, 5 remaining)
req B: ████░░░░░░░░░  (4 tok done, 9 remaining)
req C: ███████████░░  (11 tok done, 2 remaining)
            ↑
       short reqs 等 C
```

## 2 · 解：iteration-level scheduling
每个 **iteration** 重新看请求池：
- 完成的请求立刻出队
- 新请求立刻插入
- 在 iter 边界统一 forward

```
iter 1: [A, B, C]
iter 2: [A, B, C]
iter 3: [A, B, C, D]      <- D 入队
iter 4: [A,    C, D]      <- B 完成出队
iter 5: [   E, C, D]      <- A 完成、E 入队
```

## 3 · 关键代码骨架
```python
while running or pending:
    # admission
    while pending and can_admit():
        running.append(pending.popleft())
    # iter
    logits = model.forward(make_batch(running))
    for r in running:
        tok = sample(logits[r.idx])
        r.append(tok)
        if r.is_done():
            finished.append(r)
    running = [r for r in running if not r.is_done()]
```

## 4 · 显存约束
`can_admit()` 关键：检查 paged KV 剩余 block。
公式：`needed = ceil((prompt + max_new) / block_size)`，`free_blocks >= needed`。

## 5 · 性能数字（Orca/vLLM 论文）
| 方案 | throughput |
|------|-----------|
| static batch | 1.0× |
| continuous + paged | **24×** |

## 6 · 与 paged 的协同
- continuous batching 解决"时间"碎片
- paged 解决"空间"碎片
- 一起用才有 24×；任一独立只有 2-3×

## 7 · 失败模式
- **starvation**: 长请求一直占着，新请求饿死 → 加 priority / FIFO
- **OOM mid-iter**: 准入后 KV 不够 → preempt + swap (vLLM swap to CPU)

## 8 · vLLM 实现
- `vllm/core/scheduler.py`
- `_schedule_running` + `_schedule_swapped` + `_schedule_prefills`

## 9 · 实现：[continuous_batching.py](../src/continuous_batching.py)
- `Engine.add_request`
- `Engine.step()` 一个 iter
- 度量：iter 时间 / waiting queue 长度
