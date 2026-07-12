"""BM25 打分公式，闭卷从零手写。

面试高频度 ***。搜索/RAG 岗位必考的"经典 IR baseline"，经常和"为什么 RAG 的
检索器不能只用 dense embedding，还要混一路 BM25（混合检索）"连起来问。这里只
要求单个 query 对单个 doc 的打分（工程上真正的检索系统会对全部候选 doc 批量算），
纯 stdlib（只用 math.log）。

接口约定
--------
    bm25_score(query_terms, doc_terms, corpus_doc_freqs, corpus_size, avg_doc_len,
               k1=1.5, b=0.75) -> float

    query_terms      : list[str]，query 的分词结果（如果同一个词出现多次，按出现
                        次数原样累加贡献，不做去重——这是本题为了简化 query-side
                        term frequency 做的约定，实践中 BM25 完整公式在 query 侧
                        还有一个 k3 饱和项，这里简化掉了）
    doc_terms        : list[str]，**这一篇**要打分的文档的分词结果
    corpus_doc_freqs : dict[str, int]，词 -> 整个语料库里包含这个词的文档数 n(t)
                        （这是提前在整个语料库上统计好、作为参数传进来的，本函数
                        本身不负责扫描语料库）
    corpus_size      : 语料库里的文档总数 N
    avg_doc_len      : 语料库里所有文档的平均长度（token 数），是一个**全局常量**，
                        不是当前这篇文档自己的长度
    k1, b            : BM25 超参数，默认值 k1=1.5, b=0.75（Lucene/Elasticsearch
                        的经典默认值）

公式
----
    对 query 里每个词 t，累加：
        IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) )
        TF饱和项(t, D) = f(t,D) * (k1+1) / ( f(t,D) + k1 * (1 - b + b * |D| / avgdl) )
        score(D, Q) = sum_t IDF(t) * TF饱和项(t, D)
    其中 f(t,D) 是词 t 在文档 D 里的出现次数，|D| 是文档 D 的长度（token 数）。

面试常问
--------
- b 参数的作用方向？—— `1 - b + b*(|D|/avgdl)` 这一项：b=0 时恒等于 1，**完全
  不做长度归一化**（长文档不受惩罚）；b=1 时是**完全按长度线性归一化**
  （`|D|/avgdl`，长文档在同样词频下被显著压低）。b=0.75 是这两者之间的折中。
  一个常见的反直觉点：b 越大，"长文档惩罚"越强，不是越弱。
- 为什么 TF 项要"饱和"（不是词频越高线性加分）？—— 分母里的 `f(t,D)` 让这一项
  是关于 f 的一个饱和曲线（f 趋于无穷时趋于 (k1+1)，不会线性发散），防止简单的
  "关键词堆砌"就能无限刷高分数，也更符合"同一个词出现第 10 次带来的信息增益
  远小于第 1 次"的直觉。
- IDF 为什么会出现负数？—— 经典 Robertson-Sparck-Jones 形式在 n(t) > N/2
  （这个词在超过一半的文档里都出现）时，`(N-n+0.5)/(n+0.5) < 1`，log 出来是负数，
  意味着"太常见的词反而拉低分数"。这是这个公式本身的已知行为，不是 bug——
  但生产系统（Lucene/Elasticsearch）常用 `ln(1 + (N-n+0.5)/(n+0.5))` 这个
  "+1" 变体来避免负分，这属于工程上的稳健性改进，面试官经常拿这个点追问，
  你需要清楚自己实现的是哪一种、以及为什么生产系统要改。

常见实现陷阱
------------
1. **avg_doc_len 用错**：必须是整个语料库的平均长度（全局常量），不能错用
   当前文档自己的长度 `|D|`（那样 `|D|/avgdl` 恒为 1，长度归一化完全失效）。
2. **b 的方向搞反**：写成 `b + (1-b)*avgdl/|D|` 之类的变体，符号/方向反了会
   导致文档越长分数反而越高。
3. **IDF 公式漏掉 +0.5 平滑项**：`n(t)=0` 或 `n(t)=N` 时不加 +0.5 会导致
   分母为 0 或 log(0)。
4. **corpus_doc_freqs 里查不到某个 query 词**：应该按 `n(t)=0` 处理（用
   `dict.get(t, 0)`），不要直接 KeyError 崩掉。
"""
from __future__ import annotations


def bm25_score(
    query_terms: list[str],
    doc_terms: list[str],
    corpus_doc_freqs: dict[str, int],
    corpus_size: int,
    avg_doc_len: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """见模块 docstring：对 query_terms 逐词累加 IDF(t) * TF饱和项(t,D)。"""
    raise NotImplementedError("闭卷手写：删除这行 raise，实现 BM25 打分公式")
