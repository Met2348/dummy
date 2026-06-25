# L01 · Deep Research 范式：多视角 + outline-driven RAG

## 1. "自动写综述"不是"把检索结果拼起来"

[STORM (2402.14207)](https://arxiv.org/abs/2402.14207) 的洞见是：写一篇好综述，
难点不在检索，而在**先想清楚要从哪些角度问**。它的流程：

1. **多视角提问（perspectives）**：对同一主题，模拟不同立场/角色各自提问，覆盖盲区。
2. **outline-driven 检索**：按提纲分头检索，每个子问题各自找证据。
3. **带 inline 引用的合成**：把检索到的事实组织成段落，每句话挂上出处。

`storm.py` 把这三步缩到最小：

```python
def perspectives(topic):       # 多视角：同一主题三个角度
    return [f"{topic} retrieval outline perspective",
            f"{topic} pitfalls hallucination integrity",
            f"{topic} autonomous agent tree-search experiment"]

def synthesize(topic):         # 各视角检索 → 合成带引用的句子
    for p in perspectives(topic):
        for d in retrieve(p, k=2): ...
```

## 2. 为什么多视角重要

如果只用一个 query 检索，你会系统性漏掉某些角度的证据。多视角等于强迫自己问
"从可信度看呢？从失败模式看呢？"——这也是为什么本模块的第二个视角专门问 `pitfalls`，
于是 Hidden Pitfalls 那篇会被检索进来。**视角的多样性，决定综述的盲区大小。**

## 3. 这一步天然可靠吗？不

合成（synthesize）这一环是整个范式最容易出问题的地方：模型把检索到的若干事实
"组织成流畅段落"时，会**顺手加戏**——写出检索结果里并不存在的论断，还煞有介事地挂个引用。
本模块的 `synthesize` 故意保留了两句这样的"加戏"，留给下一讲的忠实度核查去抓。

> Deep Research 让你**当天就能**对一个新领域产出一份带引用的综述初稿——这是它的真价值；
> 但初稿的每一句引用都得过忠实度这一关，否则你只是更快地生产了看着可信的错误。

## 4. 动手

1. 跑 `python src/run.py`，看 `[检索到文档]` 列表——三个视角合起来覆盖了哪几篇？
   把 `perspectives` 删到只剩第一个视角，重跑，哪些文档（和它们的事实）消失了？
2. 想想：如果让一个**真 LLM** 来写 `perspectives`，它会不会漏掉"批判/失败模式"这个视角？
   （乐观系统往往不爱给自己设批判视角——这是 9.8 的伏笔。）
