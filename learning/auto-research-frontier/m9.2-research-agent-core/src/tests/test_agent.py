"""V2 测试：锁死 9.2 的诚实性——检索确定性、agent 闭环终止、
以及核心对比"有 critic 清掉幻觉引用、无 critic 留着"。
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research_agent import (
    CORPUS_IDS, MockLLM, ResearchAgent, review, search, ungrounded_in,
)

Q = "autonomous research agent for ideation and evaluation"


def test_search_is_deterministic_and_grounded():
    a = [p.arxiv_id for p in search("ideation evaluation pitfalls", k=3)]
    b = [p.arxiv_id for p in search("ideation evaluation pitfalls", k=3)]
    assert a == b and len(a) > 0
    assert all(pid in CORPUS_IDS for pid in a)


def test_agent_loop_terminates_and_retrieves():
    r = ResearchAgent().run(Q)
    assert r["retrieved_ids"], "应检索到至少一篇"
    assert all(pid in CORPUS_IDS for pid in r["retrieved_ids"])
    assert r["transcript"].count("\n") >= 4   # 多步 ReAct 轨迹


def test_draft_contains_a_hallucinated_citation():
    """前提：mock 草稿确实混入了一个检索集外的引用（否则后面的对比没意义）。"""
    r = ResearchAgent().run(Q, use_critic=False)
    assert MockLLM.HALLUCINATED_ID in r["draft"].cited_ids
    assert MockLLM.HALLUCINATED_ID not in CORPUS_IDS


def test_critic_removes_hallucinated_citation():
    """有 critic：最终计划里幻觉引用被清空（grounding 守住）。"""
    r = ResearchAgent().run(Q, use_critic=True)
    assert ungrounded_in(r["plan"], r["retrieved_ids"]) == []
    assert MockLLM.HALLUCINATED_ID not in r["plan"]["citations"]


def test_without_critic_hallucination_survives():
    """无 critic：没人查，幻觉引用留在最终计划里——这正是要对比出的失败。"""
    r = ResearchAgent().run(Q, use_critic=False)
    assert len(ungrounded_in(r["plan"], r["retrieved_ids"])) >= 1


def test_critic_changes_the_plan():
    """同生成、同检索，有/无 critic 的最终计划必须不同（证明 critic 真在起作用）。"""
    a = ResearchAgent().run(Q, use_critic=False)["plan"]
    b = ResearchAgent().run(Q, use_critic=True)["plan"]
    assert a["citations"] != b["citations"] or a["baseline"] != b["baseline"]


def test_critic_flags_missing_baseline():
    r = ResearchAgent().run(Q, use_critic=True)
    assert any("baseline" in f for f in r["critique"].flags)


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
