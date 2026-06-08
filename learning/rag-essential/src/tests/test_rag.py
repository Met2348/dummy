"""Aggregate test runner for Topic 2 rag-essential."""
import sys
import os

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

MODULES = [
    "common",
    "chunker",
    "naive_rag",
    "bm25_minimal",
    "hybrid",
    "reranker_mock",
    "colbert_minimal",
    "hyde_demo",
    "graph_rag",
    "hipporag",
    "rag_fusion",
    "self_rag",
    "ragas_metrics",
    "rag_original_minimal",
    "capstone_rag_compare",
]


def main() -> int:
    failed = []
    for name in MODULES:
        try:
            mod = __import__(name, fromlist=["_self_test"])
            mod._self_test()
        except Exception as e:  # noqa: BLE001
            failed.append((name, str(e)))
            print(f"[FAIL] {name}: {e}")

    print()
    print(f"=== {len(MODULES) - len(failed)}/{len(MODULES)} modules passed ===")
    if failed:
        for n, m in failed:
            print(f"  FAIL: {n} -> {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
