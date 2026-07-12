"""bm25_spec 的纯断言检验：不 import solutions/。

用一个 5 篇文档的确定性小语料手算/独立复算做对拍，另外专门构造"同样词频、
文档长度不同"的一对文档来隔离验证 b 参数的长度惩罚效果。
"""
from __future__ import annotations

import math
from collections import Counter


def _ref_idf(n: int, corpus_size: int) -> float:
    """独立复算的 IDF 参考实现（经典 Robertson-Sparck-Jones 形式，不调用被测代码）。"""
    return math.log((corpus_size - n + 0.5) / (n + 0.5))


def _ref_bm25(query_terms, doc_terms, doc_freqs, corpus_size, avg_doc_len, k1, b) -> float:
    counts = Counter(doc_terms)
    doc_len = len(doc_terms)
    length_norm = 1.0 - b + b * (doc_len / avg_doc_len)
    total = 0.0
    for t in query_terms:
        n_t = doc_freqs.get(t, 0)
        idf = _ref_idf(n_t, corpus_size)
        f_td = counts.get(t, 0)
        total += idf * (f_td * (k1 + 1)) / (f_td + k1 * length_norm)
    return total


def check(target) -> None:
    # ---- 5 篇文档的确定性小语料 ----
    # D0/D4 特意让 "cat","sat" 的原始词频完全一样(各1次)，D4 只是塞了一堆无关的
    # filler 词把文档拉长——这样才能干净地隔离出"文档越长同样词频分数被惩罚"
    # 这条只由 b 参数决定的效果，而不是被词频差异干扰。
    d0 = ["cat", "sat", "mat"]
    d1 = ["dog", "runs", "fast"]
    d2 = ["bird", "flies", "high"]
    d3 = ["fish", "swims", "deep"]
    d4 = ["cat", "sat", "mat", "filler1", "filler2", "filler3", "filler4", "filler5", "filler6"]

    docs = [d0, d1, d2, d3, d4]
    corpus_size = len(docs)
    avg_doc_len = sum(len(d) for d in docs) / corpus_size  # (3+3+3+3+9)/5 = 4.2

    corpus_doc_freqs = {
        "cat": 2,  # 出现在 d0, d4
        "sat": 2,  # 出现在 d0, d4
        "mat": 2,  # 出现在 d0, d4
        "dog": 1, "runs": 1, "fast": 1,
        "bird": 1, "flies": 1, "high": 1,
        "fish": 1, "swims": 1, "deep": 1,
    }
    query = ["cat", "sat"]

    # ---- 1) 精确数值对拍（默认 k1=1.5, b=0.75）----
    want_d0 = _ref_bm25(query, d0, corpus_doc_freqs, corpus_size, avg_doc_len, 1.5, 0.75)
    got_d0 = target(query, d0, corpus_doc_freqs, corpus_size, avg_doc_len)
    assert math.isclose(got_d0, want_d0, rel_tol=1e-6), (
        f"D0 打分对不上手算/独立复算的结果: got={got_d0}, want={want_d0}"
    )

    # ---- 2) 文档越长(同样词频) 分数应该被合理惩罚：D0(len=3) 应该显著高于 D4(len=9) ----
    got_d4 = target(query, d4, corpus_doc_freqs, corpus_size, avg_doc_len)
    want_d4 = _ref_bm25(query, d4, corpus_doc_freqs, corpus_size, avg_doc_len, 1.5, 0.75)
    assert math.isclose(got_d4, want_d4, rel_tol=1e-6), (
        f"D4 打分对不上手算/独立复算的结果: got={got_d4}, want={want_d4}"
    )
    assert got_d0 > got_d4, (
        f"D0 和 D4 的 query 词频完全一样，D4 只是被 filler 词拉长了，"
        f"D0 的分数应该显著更高(长文档被 b 参数惩罚)，实得 D0={got_d0}, D4={got_d4}"
    )

    # ---- 3) b=0 时应该完全关闭长度归一化：词频一样的 D0/D4 分数必须精确相等 ----
    got_d0_b0 = target(query, d0, corpus_doc_freqs, corpus_size, avg_doc_len, 1.5, 0.0)
    got_d4_b0 = target(query, d4, corpus_doc_freqs, corpus_size, avg_doc_len, 1.5, 0.0)
    assert math.isclose(got_d0_b0, got_d4_b0, rel_tol=1e-6), (
        f"b=0 时应该完全不做长度归一化，词频相同的 D0/D4 分数必须精确相等，"
        f"实得 D0={got_d0_b0}, D4={got_d4_b0}（b 参数的方向是不是反了？）"
    )

    # ---- 4) query 里包含语料库里从没出现过的词：应按 n(t)=0 处理，不应该 KeyError ----
    got_oov = target(["nonexistent_term"], d0, corpus_doc_freqs, corpus_size, avg_doc_len)
    want_oov = _ref_bm25(["nonexistent_term"], d0, corpus_doc_freqs, corpus_size, avg_doc_len, 1.5, 0.75)
    assert math.isclose(got_oov, want_oov, rel_tol=1e-6), f"OOV 词处理不对: got={got_oov}, want={want_oov}"
