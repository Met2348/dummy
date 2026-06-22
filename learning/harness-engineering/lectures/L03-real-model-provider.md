# L03 · 接真模型：provider 抽象层

> Part II · 40-min lecture · 配套代码 `src/provider.py` · 目标: 把 harness 和具体模型 API 解耦——一个窄接口, 换 Anthropic/OpenAI/开源/Mock 主体不改。

---

## 0. 问题：别把 harness 和某个 API 焊死

Module 7 的 mini-harness 用 `MockModel`。要上生产, 你得接真模型。但**最糟的做法**是把 `anthropic.messages.create(...)` 直接写进 loop——那样换模型、做 A/B、离线测试全得改 loop。

正确做法: 在模型和 harness 之间插一个**窄接口** `Provider`。模型只是一个「无状态 token 预测器」, harness 通过这个接口和它对话。

```
        harness loop  ──呼叫──►  Provider 接口  ──实现──►  ┌ MockProvider   (确定性, 测试/教学)
                                  (stream)               ├ AnthropicProvider
        harness 不知道、也不关心                            ├ OpenAIProvider
        背后是哪个模型                                      └ vLLMProvider (开源自托管)
```

> 这是软件工程的**依赖倒置**: 高层 (harness) 不依赖低层 (具体 API), 两者都依赖抽象 (Provider 接口)。换底层模型, 高层零改动。

---

## 1. 接口设计：流式 + tool-call 是最小公分母

不同 provider 的 API 各不相同, 但**共同点**是: 都能流式吐出「文本增量」和「工具调用」。所以接口就锚定这两件事。看 `src/provider.py`:

```python
@dataclass
class Chunk:
    kind: str            # "text" | "tool_call" | "done"
    text: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    stop: bool = False   # done 时, 模型是否认为可结束

class Provider(ABC):
    @abstractmethod
    def stream(self, messages, tools=None) -> Iterator[Chunk]: ...
    def count_tokens(self, messages) -> int: ...
```

要点:
- **`stream` 返回 Chunk 流**, 而不是一整段字符串。流式是生产刚需 (用户要立刻看到输出; 长任务要能中途打断, 见 Module 7 的 steering)。
- **三种 Chunk** 把异构 API 归一: 文本增量 / 工具调用 / 结束信号。任何 provider 的原生事件, 适配器负责翻译成这三种。
- **`stop`** 让模型表达「我觉得任务完成了」——这是 L05 long-horizon 里 hook 要拦截的关键信号。

---

## 2. MockProvider：为什么默认用它

生产 harness 也要能**离线、确定性地测试**——你不会想每跑一次单测就烧 API 钱、还得忍受模型的随机性。`MockProvider` 就是干这个的:

```python
prov = MockProvider(script=[
    Turn(text="先调搜索", tool_calls=[{"name":"search","args":{"q":"x"}}]),
    Turn(text="完成", stop=True),
])
```

- 给一个 `script` (一串 `Turn`), 它就**确定性地**逐回合吐 chunk, 驱动多步 loop。
- 不给 script, 它有个默认策略 (看到工具名就发 echo 调用, 否则纯文本)。
- 它模拟了真 provider 的流式行为 (文本拆成增量), 所以**用 Mock 写的 loop, 换成真 provider 一行不改**。

> 这正是本专题所有 notebook「在你 Windows 机上无需 API key 就能跑」的原因。教学/测试用 Mock, 上线时换真 provider——harness 主体不动。这是「可替换性」带来的直接好处。

---

## 3. 接真 provider 长什么样（骨架）

`src/provider.py` 里 `AnthropicProvider` 是教学骨架。真实实现只做一件事: **把该家 API 的原生流式事件, 翻译成我们的三种 Chunk**:

```
Anthropic 原生事件                →   我们的 Chunk
content_block_delta (text)        →   Chunk(kind="text", text=...)
content_block_start (tool_use)    →   Chunk(kind="tool_call", tool_name=..., tool_args=...)
message_stop                      →   Chunk(kind="done", stop=...)
```

OpenAI、vLLM 同理, 各写一个适配器。**harness 的 loop 看到的永远是 Chunk 流, 永远不知道背后是谁。** 这就是抽象的回报。

> 实操提醒 (上生产时):
> - **重试/超时/限流**: 真 provider 会 5xx、会限流。重试逻辑应放在 provider 适配器层, 对 harness 透明 (回忆 Module 7 的 error recovery)。
> - **token 计数**: `count_tokens` 各家算法不同; 生产里用各家的 tokenizer, 教学里用 `approx_tokens` (~4 char/token) 够了。
> - **tool schema 格式**: 各家 function-calling 的 JSON schema 略有差异, 也由适配器吸收。

---

## 4. 本讲小结 + 通往 L04

- 在模型和 harness 之间插一个窄 `Provider` 接口 (依赖倒置)。
- 接口锚定最小公分母: **流式 + tool-call + stop 信号** (三种 Chunk)。
- `MockProvider` 给确定性离线测试; 换真 provider 时 harness 主体零改动。

> **下一讲 L04**: 模型接上了, 真实任务会很快把上下文窗口塞满。怎么在不丢关键信息的前提下管理上下文? Claude Code 被逆向出的 **5 阶段渐进式 compaction** ——我们已经在 `src/compaction.py` 实现了它, L04 拆开讲, 并在 N1 notebook 里看它逐级触发。
