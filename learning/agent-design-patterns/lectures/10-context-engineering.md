# L10 · Context Engineering — 把 context window 当预算

## 2025 的视角转变

长跑 agent 的瓶颈,往往**不是"模型会不会推理"**,而是**"该看的东西能不能塞进窗口"**。这门手艺叫 context engineering——是 prompt engineering 的"长程版"。

核心动作四件:

| 动作 | 干什么 | src |
|------|--------|-----|
| **预算核算** | 知道每条消息花多少 token | `ContextWindow.used()` |
| **compaction** | 把旧轮次压成摘要 | `compact()` |
| **工具输出裁剪** | 大结果进窗口前先截 | `trim_tool_output()` |
| **offload** | 细节甩给 memory/sub-agent | 见 Topic 9 |

## src 走读

[context_engineering.py](../src/context_engineering.py):

```python
@dataclass
class ContextWindow:
    budget: int
    messages: list
    def used(self):  return sum(est_tokens(t) for _, t in self.messages)
    def over_budget(self): return self.used() > self.budget

def compact(window, keep_recent=2, summarizer=None):
    old, recent = window.messages[:-keep_recent], window.messages[-keep_recent:]
    note = summarizer(old)                      # 旧轮 → 一条摘要
    # 新窗口 = [摘要] + 最近 N 轮原文
```

`_self_test` 验证:compaction 后 token 用量下降,且**最近一轮原文逐字保留**(摘要只压旧的)。

## 设计要点

1. **保留近因,压缩远因**:最近几轮往往最相关,逐字留;久远的压成摘要。
2. **裁剪保头尾、丢中段**:`trim_tool_output` 对超大文件读/命令输出留头+尾,中间标注 `[trimmed ~N tokens]`——这是真 harness 的常用手法。
3. **compaction 时机**:逼近 budget 才压(过早压会丢有用信息)。Topic 9 [context-compaction](../../agent-harness-design/lectures/06-context-compaction.md) 讲触发策略。
4. **offload 而非硬塞**:能放进文件/memory 的细节,别一直占着窗口。

## 和"造大模型"专题的呼应

`long-context` 专题讲的是**让窗口变大**(YaRN/RoPE/Ring Attention);context engineering 讲的是**在给定窗口里精打细算**。两者互补:窗口再大也有上限,且越满越贵越慢。

## 退出条件
- [ ] 能解释为什么"窗口够大"不等于"不用管 context"
- [ ] 说清 compaction 保近因压远因的原则
- [ ] 知道工具输出裁剪的头尾保留手法
