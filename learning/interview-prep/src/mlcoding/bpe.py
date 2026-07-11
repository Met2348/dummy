"""最小 BPE 分词器，从零手写：训练 merges + encode/decode 往返。

面试高频度 ★★★（"tokenizer 怎么来的"常问）。得分点：BPE 贪心地反复合并语料中最高频的
相邻符号对；encode 按学到的 merge 顺序贪心应用；decode 只是把符号接起来。
纯确定性（频率并列时取字典序最小），无随机。
"""
from __future__ import annotations

from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.merges: list[tuple[str, str]] = []     # 学到的合并顺序
        self.vocab: set[str] = set()

    @staticmethod
    def _pairs(symbols: list[str]) -> Counter:
        c = Counter()
        for a, b in zip(symbols, symbols[1:]):
            c[(a, b)] += 1
        return c

    def train(self, corpus: list[str], num_merges: int) -> "BPETokenizer":
        # 词表示为字符序列的列表（词内合并，不跨词）
        words = [list(w) for w in corpus]
        self.vocab = {ch for w in words for ch in w}
        for _ in range(num_merges):
            pair_counts = Counter()
            for w in words:
                pair_counts.update(self._pairs(w))
            if not pair_counts:
                break
            # 取最高频；并列取字典序最小 → 完全确定性（无随机、无平台差异）
            max_count = max(pair_counts.values())
            best_pair = min(p for p, c in pair_counts.items() if c == max_count)
            merged = "".join(best_pair)
            self.merges.append(best_pair)
            self.vocab.add(merged)
            words = [self._merge_word(w, best_pair) for w in words]
        return self

    @staticmethod
    def _merge_word(word: list[str], pair: tuple[str, str]) -> list[str]:
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(word[i] + word[i + 1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        return out

    def encode(self, word: str) -> list[str]:
        symbols = list(word)
        for pair in self.merges:                    # 按学到的顺序贪心应用
            symbols = self._merge_word(symbols, pair)
        return symbols

    def decode(self, symbols: list[str]) -> str:
        return "".join(symbols)


def _self_test() -> None:
    corpus = ["low", "low", "low", "lower", "newest", "newest", "widest"]
    tok = BPETokenizer().train(corpus, num_merges=10)

    # 1) roundtrip：encode 后 decode 还原
    for w in ["low", "lower", "newest", "widest", "slowest"]:
        assert tok.decode(tok.encode(w)) == w, w

    # 2) 高频对 'e','s' 或 'l','o' 应被合并进词表
    assert any(len(v) >= 2 for v in tok.vocab), "应学到多字符 token"

    # 3) 高频词 'low' 出现 3 次，'lo' 应成为一个 merge
    assert ("l", "o") in tok.merges or "lo" in tok.vocab

    # 4) encode('low') 的 token 数应少于字符数（发生了合并）
    assert len(tok.encode("low")) < 3
    print("[PASS] bpe: 往返还原 + 学到多字符 token + 高频对被合并")


if __name__ == "__main__":
    _self_test()
