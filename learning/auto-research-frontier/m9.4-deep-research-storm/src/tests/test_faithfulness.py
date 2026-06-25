"""V2 测试：锁死 9.4 的诚实性——合成确定性、忠实句通过、植入的不忠实句被抓、
且不忠实句能骗过"只查存在性"的 naive 检查（引用存在 ≠ 引用忠实）。
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mini_storm import (
    BY_ID, Sentence, audit, check_sentence, existence_only, synthesize,
)

TOPIC = "how AI research agents do retrieval and avoid pitfalls"


def test_synthesis_deterministic():
    a = synthesize(TOPIC)
    b = synthesize(TOPIC)
    assert a == b
    assert len(a.sentences) >= 4


def test_faithful_sentences_pass():
    """每篇检索到的文档配的那句（claim ⊆ supports）应判 faithful。"""
    report = synthesize(TOPIC)
    grounded = [s for s in report.sentences
                if s.cited_doc in report.retrieved_ids
                and s.claim_tokens <= BY_ID[s.cited_doc].supports]
    assert grounded, "应有一批忠实句"
    assert all(check_sentence(s) == "faithful" for s in grounded)


def test_planted_unfaithful_is_caught():
    """两句植入的不忠实引用被忠实度检查抓到。"""
    report = synthesize(TOPIC)
    a = audit(report)
    assert len(a["unfaithful"]) >= 2
    texts = " ".join(s.text for s in a["unfaithful"])
    assert "湿实验" in texts and "引用准确率" in texts


def test_existence_check_is_fooled_by_unfaithful():
    """核心：不忠实句引的都是**真存在**的 id → naive 存在性检查全部放行。"""
    report = synthesize(TOPIC)
    a = audit(report)
    for s in a["unfaithful"]:
        assert existence_only(s) is True            # 存在性：通过（被骗）
        assert check_sentence(s) == "unfaithful"    # 忠实度：抓住
    assert a["existence_pass"] == a["total"]         # naive 检查认为"全合规"


def test_dangling_citation_is_distinct_from_unfaithful():
    """引一个不存在的 id → dangling，且连存在性检查也过不了（这是 9.2 那一关）。"""
    s = Sentence("某句乱引 [9999.99999]", frozenset({"x"}), "9999.99999")
    assert check_sentence(s) == "dangling"
    assert existence_only(s) is False


def test_faithfulness_is_claim_specific():
    """同一篇 v2，既有忠实句也有不忠实句——忠实与否取决于 claim，不取决于 doc。"""
    report = synthesize(TOPIC)
    v2 = [s for s in report.sentences if s.cited_doc == "2504.08066"]
    verdicts = {check_sentence(s) for s in v2}
    assert "faithful" in verdicts and "unfaithful" in verdicts


if __name__ == "__main__":   # 直跑兜底
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{'OK' if fails == 0 else f'{fails} FAILED'}")
    raise SystemExit(1 if fails else 0)
