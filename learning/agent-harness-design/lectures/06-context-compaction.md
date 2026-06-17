# L06 · Context Compaction 实战

## 问题

长会话迟早撑爆窗口。硬截断会丢掉早期关键信息(用户最初的目标!)。compaction = **把旧轮压成摘要,保住信号、腾出空间**。

## src 走读

[context.py](../src/harness/context.py) 的 `compact`:

```python
def compact(self, keep_recent=3, summarizer=None) -> int:
    system = [m for m in self.messages if m["role"] == "system"]   # 永远保留
    rest   = [m for m in self.messages if m["role"] != "system"]
    if len(rest) <= keep_recent:
        return 0                                                     # 没必要压
    old, recent = rest[:-keep_recent], rest[-keep_recent:]
    note = (summarizer or _default_summarize)(old)
    self.messages = system + [{"role":"system","content":f"[summary] {note}"}] + recent
    return before - self.used()                                      # 返回省下的 token
```

`_self_test` 验证:压缩后 token 下降、**最近一轮逐字保留**、system 消息保留、出现 `[summary]` 标记。

## 三条原则

1. **保近因,压远因**:最近几轮往往最相关,逐字留;久远的压。
2. **system 不动**:角色、工具说明是地基,压了 agent 会"失忆人格"。
3. **保留目标**:用户最初的目标 / 关键约束要进摘要,别压没了——这是 compaction 最容易出的事故。

## 触发时机

| 策略 | 说明 | 本仓库 |
|------|------|--------|
| 被动 | `over_budget()` 才压 | loop 默认 |
| 主动阈值 | 用量 ≥ `compact_at` 就压 | `run_loop(compact_at=...)` |
| 按轮数 | 每 N 轮压一次 | 可自定义 summarizer 触发 |

太早压会丢有用信息,太晚压会撞墙——逼近上限再压是稳妥默认。

## summarizer 在真 harness 里是一次 LLM 调用

本仓库的 `_default_summarize` 是确定性 mock(列角色 + token 数)。真 harness 用一次模型调用把旧对话压成自然语言摘要——这本身要花 token,所以 compaction 不是免费的,要权衡频率。

## handoff:compaction 的极端形式

会话结束/换 agent 时,把整段历史压成一份"交接摘要"传给下一个 context——这就是本仓库 `.remember/` handoff、以及长会话续接的原理。

## 退出条件
- [ ] 说清 compaction 保什么压什么
- [ ] 记住 system 与"用户目标"必须保留
- [ ] 理解触发时机的权衡 + compaction 本身有成本
