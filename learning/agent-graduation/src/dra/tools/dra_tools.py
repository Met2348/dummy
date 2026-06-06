"""5 mock tools for DRA."""
from __future__ import annotations


KB = {
    "vllm": {
        "title": "vLLM PagedAttention",
        "url": "https://vllm.ai/paged-attention",
        "text": "vLLM uses PagedAttention to manage KV cache as virtual memory pages, achieving 24x throughput improvement.",
    },
    "sglang": {
        "title": "SGLang RadixAttention",
        "url": "https://sglang.ai/radix",
        "text": "SGLang uses RadixAttention to share KV cache across requests via a radix tree, ideal for multi-turn conversations.",
    },
    "speculative": {
        "title": "Speculative Decoding",
        "url": "https://arxiv.org/abs/2302.01318",
        "text": "Speculative decoding uses a small draft model to propose tokens that a larger target model verifies, yielding 2-3x speedup.",
    },
    "quantization": {
        "title": "GPTQ and AWQ",
        "url": "https://arxiv.org/abs/2210.17323",
        "text": "GPTQ and AWQ enable INT4 quantization with minimal quality loss, reducing memory by 4x.",
    },
    "flash": {
        "title": "FlashAttention v3",
        "url": "https://arxiv.org/abs/2407.08608",
        "text": "FlashAttention v3 (2024) achieves 75% of theoretical FLOP utilization on H100, 2x speed over v2.",
    },
    "mla": {
        "title": "DeepSeek MLA",
        "url": "https://arxiv.org/abs/2405.04434",
        "text": "DeepSeek-V2 introduced MLA (Multi-head Latent Attention), reducing KV cache 93% via low-rank projection.",
    },
    "moe": {
        "title": "Mixture-of-Experts",
        "url": "https://arxiv.org/abs/2401.06066",
        "text": "MoE models like Mixtral 8x7B activate only 13B parameters per token, achieving 70B-level quality at 13B inference cost.",
    },
}


def search_tool(args: dict) -> list[dict]:
    q = args.get("query", "").lower()
    hits = []
    for key, doc in KB.items():
        if key in q or any(t in q for t in doc["text"].lower().split()[:5]):
            hits.append({"doc_id": key, **doc})
    if not hits:
        hits = list({"doc_id": k, **v} for k, v in list(KB.items())[:3])
    return hits[:3]


def fetch_tool(args: dict) -> dict:
    doc_id = args.get("doc_id", "")
    if doc_id in KB:
        return {"doc_id": doc_id, "full_text": KB[doc_id]["text"], "url": KB[doc_id]["url"]}
    return {"doc_id": doc_id, "error": "not found"}


def cite_tool(args: dict) -> str:
    doc_id = args.get("doc_id", "")
    n = args.get("n", 1)
    if doc_id not in KB:
        return f"[{n}] unknown"
    doc = KB[doc_id]
    return f"[{n}] {doc['title']}. {doc['url']}"


_IN_MEMORY_FS: dict[str, str] = {}


def file_write_tool(args: dict) -> dict:
    path = args.get("path", "")
    content = args.get("content", "")
    if not path:
        return {"error": "missing path"}
    _IN_MEMORY_FS[path] = content
    return {"path": path, "bytes": len(content)}


def get_fs() -> dict[str, str]:
    return dict(_IN_MEMORY_FS)


def python_tool(args: dict) -> dict:
    code = args.get("code", "")
    import ast
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return {"error": "import forbidden"}
        out = []
        builtins = {"print": lambda *a, **k: out.append(" ".join(str(x) for x in a)),
                    "range": range, "len": len, "sum": sum, "abs": abs,
                    "int": int, "float": float, "str": str, "list": list}
        ns: dict = {"__builtins__": builtins}
        exec(compile(tree, "<sandbox>", "exec"), ns, ns)  # noqa: S102
        return {"stdout": "\n".join(out)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _self_test() -> None:
    hits = search_tool({"query": "vllm paged attention"})
    assert len(hits) >= 1
    assert any(h["doc_id"] == "vllm" for h in hits)

    fetched = fetch_tool({"doc_id": "vllm"})
    assert "PagedAttention" in fetched["full_text"]

    cite = cite_tool({"doc_id": "vllm", "n": 1})
    assert "[1]" in cite and "vllm.ai" in cite

    r = file_write_tool({"path": "report.md", "content": "# Report"})
    assert r["bytes"] == 8
    assert "report.md" in get_fs()

    r = python_tool({"code": "print(2+3)"})
    assert r.get("stdout") == "5"

    bad = python_tool({"code": "import os"})
    assert "forbidden" in bad.get("error", "")
    print("[OK] dra_tools._self_test passed")


if __name__ == "__main__":
    _self_test()
