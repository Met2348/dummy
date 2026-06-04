# L04 · Constrained Decoding（受约束解码）

## 1 · 痛点
LLM 自由生成 ≠ 工业要的"结构化输出"：
- JSON API: `{"name":"...","age":...}` 模型 5% 概率漏引号
- SQL: 必须合法 SQL grammar
- 正则: `\d{4}-\d{2}-\d{2}`

后处理 / regex 重试 → 慢、不可靠。

## 2 · 解：在 sampling 阶段加 mask
每步 `softmax(logits)` 前，按 grammar 当前状态计算"合法 token mask"，把非法位置置 `-inf`。

```python
logits = model.forward(ids)
mask = grammar_fsm.next_legal_tokens()
logits = logits.masked_fill(~mask, -inf)
tok = sample(logits)
grammar_fsm.step(tok)
```

## 3 · 三层
| 层 | 输入 | 输出 mask 用 |
|----|------|------------|
| regex | `\d+` | char-level → token-level 上投 |
| CFG  | json schema / SQL | LL/LR FSM |
| Lark grammar | 自定义 DSL | parser state |

## 4 · 库
- **outlines** (2023 最早) — Python，慢
- **xgrammar** (NVIDIA 2024) — C++，比 outlines 快 10-100x
- **lm-format-enforcer** — 较通用
- vLLM 默认走 outlines；SGLang 默认走 xgrammar

## 5 · token vs char 的鸿沟
- grammar 是 char-level，sampler 是 token-level
- 一个 token = 几 char，可能跨越 grammar 边界
- 解：构造 "token→可达状态" 的预编译表

## 6 · 效率：static FSM compile
- 启动时把 JSON schema 编译成 FSM
- 每个 state → 对每个 vocab token 预算"接受?"
- runtime 直接查表 → O(1) mask

## 7 · vLLM/SGLang 接口
```python
# SGLang
@sgl.function
def gen(s):
    s += "JSON: "
    s += sgl.gen("answer", regex=r"\{\"name\":\"[A-Z][a-z]+\"\}")

# vLLM (OpenAI 兼容)
{"response_format": {"type": "json_schema", "json_schema": {...}}}
```

## 8 · 实现：[constrained_sampler.py](../src/constrained_sampler.py)
- `regex_to_mask(regex, vocab)` 预编译
- `ConstrainedSampler.step(logits)`
