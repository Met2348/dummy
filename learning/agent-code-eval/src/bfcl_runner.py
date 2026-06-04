"""BFCL (Berkeley Function Calling Leaderboard, 2024) — mock.

Real bench: ~2k cases across function-calling categories (simple/
parallel/multiple/relevance). Evaluates: did the model pick right
function + emit correct args?
"""
from __future__ import annotations

import json
from typing import Dict, List

from common import ModelFn, make_mock_model


# === Tool definitions ===

TOOLS = [
    {"name": "get_weather", "args": ["city"]},
    {"name": "send_email", "args": ["to", "subject", "body"]},
    {"name": "search_web", "args": ["query"]},
]


_TASKS: List[Dict] = [
    {"qid": "bfcl_1",
     "q": "What's the weather in Tokyo?",
     "expect": {"name": "get_weather", "args": {"city": "Tokyo"}}},
    {"qid": "bfcl_2",
     "q": "Email alice@x.com with subject 'Hi' and body 'Hello'.",
     "expect": {"name": "send_email",
                "args": {"to": "alice@x.com", "subject": "Hi", "body": "Hello"}}},
    {"qid": "bfcl_3",
     "q": "Look up python decorator on the web.",
     "expect": {"name": "search_web", "args": {"query": "python decorator"}}},
]


def build_prompts() -> List[Dict]:
    out = []
    for t in _TASKS:
        prompt = (f"[qid={t['qid']}]\n"
                  f"Available tools: {json.dumps(TOOLS)}\n"
                  f"User says: {t['q']}\n"
                  "Respond with a JSON: "
                  "{\"name\": <tool>, \"args\": {<args>}}")
        out.append({"qid": t["qid"], "prompt": prompt, "expect": t["expect"]})
    return out


def parse_call(text: str) -> Dict:
    # find first {...} block
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


def run_bfcl(model: ModelFn) -> List[Dict]:
    rs = []
    for d in build_prompts():
        text = model(d["prompt"], 128)
        call = parse_call(text)
        ok = (call.get("name") == d["expect"]["name"]
              and call.get("args") == d["expect"]["args"])
        rs.append({"qid": d["qid"], "got": call,
                    "expect": d["expect"], "passed": ok})
    return rs


def _self_test() -> int:
    # Empty → 0%
    rs = run_bfcl(make_mock_model({}, default=""))
    assert not any(r["passed"] for r in rs)
    # Mock correct JSON answers
    correct = {t["qid"]: json.dumps(t["expect"]) for t in _TASKS}
    rs2 = run_bfcl(make_mock_model(correct))
    assert all(r["passed"] for r in rs2), rs2
    # parse_call edge case
    assert parse_call("garbage") == {}
    assert parse_call("text {\"a\": 1} text").get("a") == 1
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"bfcl_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
