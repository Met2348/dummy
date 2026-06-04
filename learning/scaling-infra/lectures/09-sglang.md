# L09 · SGLang (RadixAttention)

> 12 slides | 40 min ⭐⭐⭐⭐

## Slide 1 · SGLang 定位

```
LMSYS (Chatbot Arena) 2024 出品
vLLM 平替, 推理速度更快 (尤其 prefix 共享场景)
开源
```

## Slide 2 · RadixAttention

```
基于 radix tree (前缀树) 管 KV cache
全局 prefix 共享 (不止单 batch)
LRU evict
```

## Slide 3 · 适用场景

```
multi-turn chat: system prompt + history 全共享 ✓
agent / few-shot: 模板共享 ✓
batch reranking: 同 query 多 candidate ✓
```

提升 2-5×.

## Slide 4 · SGLang DSL

```python
import sglang as sgl

@sgl.function
def chat(s, q):
    s += sgl.system("You are a helpful AI.")
    s += sgl.user(q)
    s += sgl.assistant(sgl.gen("answer", max_tokens=200))

state = chat.run(q="Hi")
print(state["answer"])
```

## Slide 5 · 多分支

```python
@sgl.function
def multi_choice(s, q, choices):
    s += sgl.user(q)
    forks = s.fork(len(choices))
    for i, c in enumerate(choices):
        forks[i] += sgl.assistant(c + sgl.gen("score", max_tokens=10))
    return forks
```

## Slide 6 · 启动

```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --port 30000 \
  --mem-fraction-static 0.85
```

## Slide 7 · vs vLLM 性能

```
multi-turn (10 turn):
  vLLM:    35 tok/s/req
  SGLang:  85 tok/s/req  (prefix 复用)
single shot:
  vLLM/SGLang 接近
```

## Slide 8 · OpenAI API

兼容 OpenAI:
```bash
python -m sglang.launch_server --model ... --api-key XX
```

## Slide 9 · 结构化输出

```python
schema = {"type": "object", "properties": {"name": {"type": "string"}}}
s += sgl.gen("data", schema=schema)
```

JSON 约束生成。

## Slide 10 · 部署 trick

```
--enable-torch-compile     # +20% 速度
--quantization fp8         # H100+ 推荐
--tp-size 4                # tensor parallel
```

## Slide 11 · 与 vLLM 何时选

```
SGLang:
  ✓ 多轮 chat
  ✓ Agent
  ✓ RLHF rollout (TRL/verl 已集成)

vLLM:
  ✓ single-shot batch
  ✓ 老牌, 生态成熟
```

## Slide 12 · 总结

```
RadixAttention 是 PagedAttention 的进化
prefix 共享场景碾压 vLLM
单 shot 接近
```

## 参考
- SGLang (LMSYS 2024)
- github.com/sgl-project/sglang
