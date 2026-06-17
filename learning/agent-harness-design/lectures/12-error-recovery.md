# L12 · 错误恢复与防失控

## 两类错误,两种处置

| 类型 | 例子 | 处置 |
|------|------|------|
| **Transient(瞬时)** | 限流、超时、网络抖动 | **重试**(退避) |
| **Terminal(终态)** | 参数错、鉴权失败、逻辑 bug | **立即 surface**,别白重试 |

分错类 = 白白重试一个永远会失败的请求,或放弃一个本该重试的。

## src 走读

[errors.py](../src/harness/errors.py):

```python
def with_retry(fn, max_attempts=3):
    while attempts < max_attempts:
        try: return {"ok": True, "value": fn(), "attempts": attempts}
        except TransientError: continue              # 重试
        except Exception as e:                        # 终态:立即返回
            return {"ok": False, "error": ..., "attempts": attempts}
    return {"ok": False, "error": "gave up...", "attempts": attempts}
```

`_self_test` 覆盖:瞬时错重试后成功、终态错一次就返回(不重试)、持续瞬时错耗尽次数。

## 防失控:Loop Guard

开放循环最危险的失败是**原地打转**烧光预算。`LoopGuard` 检测"最近 N 个动作签名相同":

```python
@dataclass
class LoopGuard:
    no_progress_limit: int = 3
    def record(self, action_signature) -> bool:     # True = 该停了
        self.history.append(action_signature)
        window = self.history[-self.no_progress_limit:]
        return len(window) >= self.no_progress_limit and len(set(window)) == 1
```

[loop.py](../src/harness/loop.py) 每回合用本回合 tool calls 的签名喂给 guard,触发就返回 `None`。`loop._self_test` 里有个"顽固 brain 反复调同一工具"被 guard 拦停的用例。

## 三道防失控闸(回顾 L02)

1. **max_turns**:硬上限,底线。
2. **loop guard**:检测无进度(动作不变)。
3. **cost ceiling**:token/$ 预算超了硬停(可加在 tracker 上)。

## 设计要点

- **退避真实现要带 sleep/jitter**;本仓库用确定性"次数"代替(无 sleep,便于测试)。
- **重试要幂等**:重试一个"转账"工具是灾难——只对安全/读类自动重试,写类要谨慎。
- **错误始终 surface**:无论重试成功失败,结果都进 context + trace,模型与人都看得见(根治 silent failure)。

## 退出条件
- [ ] 区分 transient / terminal 及对应处置
- [ ] 说清 loop guard 怎么检测无进度
- [ ] 记住三道防失控闸 + "重试要幂等"
