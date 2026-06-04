"""MMLU runner — handwritten, no external deps.

MMLU = 57 subjects × ~100 questions each, 4-option MCQ, 5-shot is standard.
Here we ship 12 micro-samples across 4 subjects for teaching.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from common import (
    EvalResult,
    EvalSample,
    ModelFn,
    accuracy,
    extract_letter,
    format_multiple_choice,
    group_accuracy,
    make_mock_model,
    make_random_model,
)


# === Micro-MMLU dataset (12 q, 4 subjects) ===

_MICRO_MMLU: List[Dict] = [
    # Subject: high_school_math
    {"qid": "math_1", "subject": "high_school_math",
     "q": "What is the value of 5 factorial?",
     "options": ["100", "120", "150", "240"], "gold": "B"},
    {"qid": "math_2", "subject": "high_school_math",
     "q": "Solve for x: 2x + 4 = 12",
     "options": ["2", "4", "6", "8"], "gold": "B"},
    {"qid": "math_3", "subject": "high_school_math",
     "q": "What is the area of a circle with radius 3?",
     "options": ["3pi", "6pi", "9pi", "12pi"], "gold": "C"},
    # Subject: world_history
    {"qid": "hist_1", "subject": "world_history",
     "q": "Which year did WWII end?",
     "options": ["1942", "1944", "1945", "1947"], "gold": "C"},
    {"qid": "hist_2", "subject": "world_history",
     "q": "Who was the first emperor of unified China?",
     "options": ["Qin Shi Huang", "Han Wudi", "Tang Taizong", "Kublai Khan"], "gold": "A"},
    {"qid": "hist_3", "subject": "world_history",
     "q": "The Magna Carta was signed in which century?",
     "options": ["11th", "12th", "13th", "14th"], "gold": "C"},
    # Subject: computer_science
    {"qid": "cs_1", "subject": "computer_science",
     "q": "Which sorting algorithm has O(n log n) average time?",
     "options": ["Bubble", "Insertion", "Quicksort", "Selection"], "gold": "C"},
    {"qid": "cs_2", "subject": "computer_science",
     "q": "What does HTTP stand for?",
     "options": ["HyperText Transfer Protocol", "High Transfer Text Protocol",
                 "HyperText Tunneling Protocol", "Host Transfer Text Procedure"], "gold": "A"},
    {"qid": "cs_3", "subject": "computer_science",
     "q": "What is the time complexity of binary search?",
     "options": ["O(n)", "O(log n)", "O(n log n)", "O(n^2)"], "gold": "B"},
    # Subject: biology
    {"qid": "bio_1", "subject": "biology",
     "q": "The powerhouse of the cell is the:",
     "options": ["Nucleus", "Mitochondrion", "Ribosome", "Lysosome"], "gold": "B"},
    {"qid": "bio_2", "subject": "biology",
     "q": "DNA is composed of how many nucleotide types?",
     "options": ["2", "3", "4", "5"], "gold": "C"},
    {"qid": "bio_3", "subject": "biology",
     "q": "Photosynthesis primarily occurs in which organelle?",
     "options": ["Mitochondria", "Chloroplast", "Nucleus", "Ribosome"], "gold": "B"},
]


# === 5-shot demos (3 hand-picked per subject as example) ===

_FIVE_SHOT_DEMOS: Dict[str, List[Tuple[str, List[str], str]]] = {
    "high_school_math": [
        ("What is 2 + 2?", ["3", "4", "5", "6"], "B"),
        ("What is 10 - 7?", ["1", "2", "3", "4"], "C"),
    ],
    "world_history": [
        ("In what year did the French Revolution begin?",
         ["1689", "1789", "1889", "1989"], "B"),
    ],
    "computer_science": [
        ("Which is a relational database?",
         ["MongoDB", "PostgreSQL", "Redis", "Cassandra"], "B"),
    ],
    "biology": [
        ("Which is a prokaryote?", ["Plant", "Bacteria", "Fungi", "Animal"], "B"),
    ],
}


def build_samples() -> List[EvalSample]:
    samples = []
    for row in _MICRO_MMLU:
        demos = _FIVE_SHOT_DEMOS.get(row["subject"], [])
        prompt = (
            f"[qid={row['qid']}] (Subject: {row['subject']})\n"
            + format_multiple_choice(row["q"], row["options"], k_shot=demos)
        )
        samples.append(EvalSample(
            qid=row["qid"], prompt=prompt, gold=row["gold"],
            meta={"subject": row["subject"]},
        ))
    return samples


def run_mmlu(model: ModelFn, max_new_tokens: int = 4) -> List[EvalResult]:
    samples = build_samples()
    results = []
    for s in samples:
        text = model(s.prompt, max_new_tokens)
        pred = extract_letter(text) or ""
        results.append(EvalResult(
            qid=s.qid, pred=pred, gold=s.gold,
            correct=(pred == s.gold), meta=s.meta,
        ))
    return results


def summarize(results: List[EvalResult]) -> Dict:
    return {
        "n": len(results),
        "accuracy": accuracy(results),
        "by_subject": group_accuracy(results, by="subject"),
    }


def _self_test() -> int:
    failed = 0
    samples = build_samples()
    assert len(samples) == 12
    assert all(s.gold in "ABCD" for s in samples)
    subjects = {s.meta["subject"] for s in samples}
    assert subjects == {"high_school_math", "world_history",
                        "computer_science", "biology"}
    # Random baseline ≈ 25% (allow wide band on 12 samples)
    rand = make_random_model(seed=0)
    rs = run_mmlu(rand)
    acc = accuracy(rs)
    assert 0.0 <= acc <= 1.0
    # Oracle = 100%
    gold = {s.qid: s.gold for s in samples}
    oracle = make_mock_model(gold)
    rs2 = run_mmlu(oracle)
    assert accuracy(rs2) == 1.0
    # Summary structure
    summ = summarize(rs2)
    assert summ["n"] == 12
    assert summ["accuracy"] == 1.0
    assert len(summ["by_subject"]) == 4
    return failed


if __name__ == "__main__":
    f = _self_test()
    print(f"mmlu_runner.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    # Demo run
    rand = make_random_model(seed=42)
    rs = run_mmlu(rand)
    print("random baseline:", summarize(rs))
