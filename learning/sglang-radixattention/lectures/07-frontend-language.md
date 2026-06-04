# L07 · SGLang Frontend Language（DSL）

## 1 · 痛点
agent / 多步生成在 OpenAI API 下要手动多轮：
```python
r1 = openai.chat.completions.create(messages=...)
r2 = openai.chat.completions.create(messages=... + r1.choices[0])
```
- 多轮 API call，每次都 prefill 全部历史 → 浪费
- 难表达 fork / select / parallel

## 2 · 解：DSL with primitives
```python
@sgl.function
def tree_of_thought(s, question):
    s += f"Q: {question}\nThought 1: "
    forks = s.fork(3)                 # 3 路 ToT
    for f in forks:
        f += sgl.gen("thought", max_tokens=50)
    # collect + judge
    s += "Best thought: "
    s += sgl.select("choice", choices=[f["thought"] for f in forks])
    s += "\nAnswer: "
    s += sgl.gen("answer", max_tokens=20)
```

## 3 · primitive 列表
| primitive | 行为 |
|----------|------|
| `gen(name, ...)` | 自由生成，名字索引结果 |
| `select(name, choices)` | 选择题（logprob argmax）|
| `fork(k)` | 复制当前状态 k 次（COW）|
| `+=` | append 字面 / gen 结果到 prompt |
| `image(path)` | 多模态输入 |

## 4 · radix tree 自动共享
fork 时所有子 stream **共享父 prefix**：
- 不重新 prefill 父 prompt
- KV cache radix 节点 refcount++
- decode 路径独立

→ 3 路 ToT，prefix 1024 token，**只 prefill 一次**

## 5 · vs LangChain / LlamaIndex
- 它们：编排框架（链 chain / 图 graph）
- SGLang：runtime 级原生支持，无中间序列化

## 6 · 编译为图
- `@sgl.function` 装饰器把 Python AST 转成 SGLang IR
- runtime 优化：调度 fork / 共享 prefix

## 7 · 实现：[frontend_lang.py](../src/frontend_lang.py)
教学 mock 版本，不依赖 sglang lib：
- `Stream` 类（保存 prompt + meta）
- `gen` / `select` / `fork` 接 mock generator
