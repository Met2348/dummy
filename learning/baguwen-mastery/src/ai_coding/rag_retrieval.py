"""简化版 BM25 检索（纯 stdlib，不依赖 numpy），对一个手写的小语料库做检索验证。

分词简化说明：真实中文分词需要 jieba 等第三方库，这里为了保持纯 stdlib，
语料库和查询都预先按"词/短语"用空格分隔好（相当于已经分好词），
tokenize() 只做最基础的空白切分，不影响 BM25 打分逻辑本身的正确性验证。
"""
from __future__ import annotations

import math
from collections import Counter

# 一个手写的 8 篇短文档小语料库，覆盖本专栏几个不同类别的知识点，
# 每篇主题足够区分，方便验证检索能命中"手工确认过的最相关文档"。
CORPUS: list[str] = [
    # 0
    "PagedAttention 借鉴 操作系统 分页 思想 管理 KV cache 减少 显存 碎片 浪费",
    # 1
    "ZeRO 优化器状态 分片 减少 数据并行 下 每张卡 冗余 显存 占用",
    # 2
    "BM25 是 一种 基于 词频 和 逆文档频率 的 经典 稀疏 检索 排序 算法",
    # 3
    "Transformer 自注意力 机制 通过 Query Key Value 计算 注意力 权重 加权求和",
    # 4
    "MCP 协议 标准化 大模型 Agent 和 外部 工具 数据源 之间 的 连接 方式",
    # 5
    "梯度检查点 gradient checkpointing 通过 重新计算 前向 激活值 来 节省 显存",
    # 6
    "AWQ 量化 通过 激活感知 的 方式 找出 并 保护 显著 权重 通道 提升 int4 精度",
    # 7
    "continuous batching 连续批处理 在 每次 解码 迭代 动态 替换 已完成 的 请求",
]


def tokenize(text: str) -> list[str]:
    return text.split()


class BM25:
    """经典 BM25 打分（Okapi BM25），纯 Python 实现。"""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("documents 不能为空")
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(d) for d in documents]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / len(self.docs)
        self.doc_term_freqs = [Counter(d) for d in self.docs]
        self.n_docs = len(self.docs)

        df: Counter = Counter()
        for doc in self.docs:
            for term in set(doc):
                df[term] += 1
        # BM25 的 idf 用 +1 平滑，保证任何词的 idf 都不会是负数
        self.idf = {
            term: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def score(self, query: str, doc_index: int) -> float:
        freqs = self.doc_term_freqs[doc_index]
        dl = self.doc_len[doc_index]
        total = 0.0
        for term in tokenize(query):
            if term not in freqs:
                continue
            idf = self.idf.get(term, 0.0)
            f = freqs[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += idf * (f * (self.k1 + 1)) / denom
        return total

    def rank(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k] if top_k else scores


def rerank_by_keyword_coverage(
    query: str, documents: list[str], candidates: list[tuple[int, float]],
) -> list[tuple[int, float, int]]:
    """简单的二次重排序：按"查询词在文档里的覆盖数"降序排列，覆盖数相同再按原始 BM25 分数排。"""
    q_terms = set(tokenize(query))
    rescored = []
    for idx, bm25_score in candidates:
        doc_terms = set(tokenize(documents[idx]))
        coverage = len(q_terms & doc_terms)
        rescored.append((idx, bm25_score, coverage))
    rescored.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return rescored


def _self_test() -> None:
    bm25 = BM25(CORPUS)

    # 查询1：明确指向 PagedAttention（文档0），不应命中 ZeRO(文档1)等其它显存优化话题
    ranked = bm25.rank("KV cache 显存 碎片 怎么 管理")
    assert ranked[0][0] == 0, f"top-1 应为文档0(PagedAttention)，实际排名={ranked[:3]}"
    assert ranked[0][1] > 0

    # 查询2：明确指向 MCP 协议（文档4）
    ranked = bm25.rank("MCP 协议 解决 工具 数据源 集成 问题")
    assert ranked[0][0] == 4, f"top-1 应为文档4(MCP)，实际排名={ranked[:3]}"

    # 查询3：明确指向 AWQ 量化（文档6）
    ranked = bm25.rank("AWQ 量化 显著 权重 通道")
    assert ranked[0][0] == 6, f"top-1 应为文档6(AWQ)，实际排名={ranked[:3]}"

    # 查询4：明确指向 continuous batching（文档7）
    ranked = bm25.rank("连续批处理 解码 迭代 动态 替换 请求")
    assert ranked[0][0] == 7, f"top-1 应为文档7(continuous batching)，实际排名={ranked[:3]}"

    # 完全不相关的查询：分数应明显低于强相关查询命中的分数
    unrelated = bm25.score("今天 天气 怎么样 适合 出门 吗", 0)
    related = bm25.score("KV cache 显存 碎片 怎么 管理", 0)
    assert unrelated < related

    # 二次重排序：覆盖数最高的应该排到最前面
    query = "梯度检查点 重新计算 前向 激活值 节省 显存"
    candidates = bm25.rank(query, top_k=5)
    reranked = rerank_by_keyword_coverage(query, CORPUS, candidates)
    assert reranked[0][0] == 5, f"重排序后 top-1 应为文档5(gradient checkpointing)，实际={reranked}"

    print(f"[PASS] rag_retrieval: BM25 检索 {len(CORPUS)}篇语料 4/4 查询命中top-1 + 关键词覆盖二次重排序")


if __name__ == "__main__":
    _self_test()
