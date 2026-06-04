# L06 · Jump-Forward Decoding

## 1 · 观察
当 grammar 只剩**唯一合法 token**（或唯一合法字符串），不需要 sample：
- JSON `"name":` 后必跟 `"`
- 时间格式 `2026-` 后必是 4 位数字

朴素：每个强制 token 也 forward 一次。
**优化**：检测后直接 "appending without forward"。

## 2 · 收益
- prefix 中确定性段越长，省得越多
- JSON schema 中固定字段名（如 `"name":`、`"age":`）占总输出 30-50%
- 实测：SGLang JSON 任务比 outlines 快 3-5x

## 3 · 算法
```python
def jump_forward(grammar_state, max_lookahead=64):
    forced_tokens = []
    state = grammar_state
    for _ in range(max_lookahead):
        next_set = state.legal_chars()
        if len(next_set) != 1:
            break
        char = next_set.pop()
        state = state.advance(char)
        forced_tokens.append(char)
    return forced_tokens, state
```

## 4 · KV cache 处理
- 强制 token 没经过 forward → KV 没生成
- 解：批量补 forward 一次（chunked prefill 风格）
- SGLang 与 chunked prefill 共享代码

## 5 · 应用场景
| 场景 | jump-forward 占比 |
|------|------------------|
| JSON schema | 40% |
| SQL | 20% |
| 自由 chat | 0% |
| 数学步骤 | 5% |

## 6 · 与 radix cache 协同
- 强制 token 序列 hash 后存 radix → 下次同 schema 直接命中
- "JSON schema cache" — SGLang 内置

## 7 · 实现：[jump_forward.py](../src/jump_forward.py)
- `JumpForward.advance(state)` 返回 (forced_str, new_state)
- 与 grammar_fsm 接口
