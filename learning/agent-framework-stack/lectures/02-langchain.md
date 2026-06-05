# L02 · LangChain

## 30 秒核心

> LangChain = **Runnable 抽象 + LCEL** (LangChain Expression Language) — 用 `|` 串接 chain。

2022 项目，90k+ stars，OG of LLM framework。

## LCEL 例

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert"),
    ("human", "{question}"),
])
llm = ChatAnthropic(model="claude-sonnet-4")

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "What is ReAct?"})
```

`|` 把 Runnable 串成 chain。

## Runnable 抽象

每个组件实现 `Runnable` 接口：
- `.invoke(input)` — sync
- `.ainvoke(input)` — async
- `.stream(input)` — stream
- `.batch([inputs])` — batch

## RunnableParallel / RunnablePassthrough

```python
chain = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain,
)
# 3 sub-chain 并行
```

## Tool calling

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for city."""
    return f"Sunny in {city}"

llm_with_tools = llm.bind_tools([get_weather])
result = llm_with_tools.invoke("Weather in Tokyo?")
```

## RAG with LangChain

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## 强弱

| 强 | 弱 |
|----|----|
| 90k stars 社区 | API 频繁 break |
| 集成多 (200+ vector DB / LLM) | Abstraction 太重 |
| LCEL 优雅 | 学习曲线陡 |
| LangSmith / LangGraph 配套 | 老 chain API 难弃 |

## v0.3 (2024) 重大重构

LangChain 拆成：
- `langchain-core` — 抽象
- `langchain` — 主功能
- `langchain-community` — 社区
- `langchain-<vendor>` — 厂商 (langchain-anthropic, etc.)
- `langchain-experimental` — 不稳

## 我们 mock 版（`langchain_style.py` 预告）

```python
class Runnable:
    def __or__(self, other): return RunnableSequence([self, other])
    def invoke(self, input): raise NotImplementedError

class PromptTemplate(Runnable):
    def __init__(self, template): self.template = template
    def invoke(self, input): return self.template.format(**input)

class MockLLM(Runnable):
    def invoke(self, prompt): return f"LLM says: {prompt[:50]}"
```

## 退出条件

- 能默写 LCEL `|` 语法
- 知道 Runnable 4 方法
- 知道 v0.3 拆分

## 一句话

> LangChain = Runnable + LCEL 串接 `|` — 90k stars 综合 framework，学曲线陡但 ecosystem 最大。
