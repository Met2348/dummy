"""ZebraLogic / Logic puzzles — multi-constraint inference.

Real ZebraLogic: 1000+ multi-house multi-attribute puzzles. We ship 3
simplified puzzles. Top models score 30-60% (small grid only).
"""
from __future__ import annotations

from typing import Dict, List

from common import ModelFn, ReasoningResult, accuracy, make_dummy_model, make_mock_model


_PUZZLES: List[Dict] = [
    {"qid": "zb_1", "size": "2x2",
     "q": "Two people (Alice, Bob) own two pets (dog, cat). "
          "Alice doesn't have the dog. Who owns the cat?",
     "gold": "Alice"},
    {"qid": "zb_2", "size": "3x3",
     "q": "Three houses in a row. Red, Blue, Green. "
          "Red is to the left of Blue. Green is at position 3. "
          "What color is house 1?",
     "gold": "Red"},
    {"qid": "zb_3", "size": "3x3",
     "q": "Anna, Ben, Cam each like one of: pizza, sushi, salad. "
          "Anna doesn't like pizza. Ben likes salad. What does Cam like?",
     "gold": "pizza"},
]


def build_prompts() -> List[Dict]:
    out = []
    for r in _PUZZLES:
        prompt = (f"[qid={r['qid']}] (Grid: {r['size']})\n"
                  f"Puzzle: {r['q']}\n"
                  f"Answer concisely.\n"
                  f"Answer:")
        out.append({"qid": r["qid"], "prompt": prompt, "gold": r["gold"],
                    "meta": {"size": r["size"]}})
    return out


def run_zebra(model: ModelFn) -> List[ReasoningResult]:
    rs = []
    for d in build_prompts():
        text = model(d["prompt"], 64).strip().lower()
        gold = d["gold"].lower()
        rs.append(ReasoningResult(
            qid=d["qid"], pred=text, gold=d["gold"],
            correct=(gold in text),
            meta=d["meta"],
        ))
    return rs


def _self_test() -> int:
    rs = run_zebra(make_dummy_model("nothing"))
    assert accuracy(rs) == 0.0
    gold = {r["qid"]: r["gold"] for r in _PUZZLES}
    rs2 = run_zebra(make_mock_model(gold))
    assert accuracy(rs2) == 1.0
    return 0


def _demo() -> None:
    """Visible demo: run the real scorer on mock models, print live accuracy.

    NOTE: this is a benchmark *runner* (scores a model's answer), not a
    constraint solver. The mock model supplies the answer; we check it.
    """
    print(f"ZebraLogic micro-set: {len(_PUZZLES)} constraint puzzles "
          f"(sizes: {', '.join(p['size'] for p in _PUZZLES)})")
    base = run_zebra(make_dummy_model("nothing"))
    gold = {r["qid"]: r["gold"] for r in _PUZZLES}
    oracle = run_zebra(make_mock_model(gold))
    print(f"  dummy        accuracy = {accuracy(base):.2f}")
    print(f"  oracle       accuracy = {accuracy(oracle):.2f}")
    print(f"  e.g. {_PUZZLES[0]['qid']}: gold = {_PUZZLES[0]['gold']!r}")
    print("  -> accuracy is computed live: case-folded gold-substring containment.")


if __name__ == "__main__":
    f = _self_test()
    print(f"zebra_logic.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    _demo()
