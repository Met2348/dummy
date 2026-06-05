"""Capstone — 3-agent hierarchical coding crew (PM + Engineer + Reviewer)."""
from __future__ import annotations
from dataclasses import dataclass, field
from cost_analyzer import CostReport


SPEC_TEMPLATE = """Spec for: {task}
- Inputs: int n (1..N)
- Rules:
  - i % 3 == 0 -> 'Fizz'
  - i % 5 == 0 -> 'Buzz'
  - i % 15 == 0 -> 'FizzBuzz'
  - else -> str(i)
- Output: list[str]
- Edge cases: n=0 returns [], n<0 returns []
"""

CODE_TEMPLATE = """def fizzbuzz(n):
    out = []
    for i in range(1, n+1):
        if i % 15 == 0: out.append('FizzBuzz')
        elif i % 3 == 0: out.append('Fizz')
        elif i % 5 == 0: out.append('Buzz')
        else: out.append(str(i))
    return out
"""


@dataclass
class CrewArtifact:
    spec: str = ""
    code: str = ""
    review: str = ""
    verdict: str = ""
    test_count: tuple[int, int] = (0, 0)
    cost: CostReport = field(default_factory=CostReport)
    rounds: int = 0


def _pm(task: str, cost: CostReport) -> str:
    cost.add_call("PM", tin=80, tout=120)
    return SPEC_TEMPLATE.format(task=task)


def _engineer(spec: str, cost: CostReport) -> str:
    cost.add_call("Engineer", tin=200, tout=180)
    return CODE_TEMPLATE


def _reviewer(spec: str, code: str, cost: CostReport) -> tuple[str, str, tuple[int, int]]:
    cost.add_call("Reviewer", tin=400, tout=100)
    ns: dict = {}
    try:
        exec(code, ns, ns)  # noqa: S102
    except Exception as e:
        return "PARSE_FAIL", f"code did not parse: {e}", (0, 5)
    fb = ns.get("fizzbuzz")
    if not callable(fb):
        return "NO_FN", "no fizzbuzz function", (0, 5)
    tests = [
        ("fb(3)", fb(3) == ["1", "2", "Fizz"]),
        ("fb(5)", fb(5) == ["1", "2", "Fizz", "4", "Buzz"]),
        ("fb(15) ends FizzBuzz", fb(15)[-1] == "FizzBuzz"),
        ("fb(0)", fb(0) == []),
        ("fb(1)", fb(1) == ["1"]),
    ]
    passed = sum(1 for _, ok in tests if ok)
    review = "Reviewer findings:\n" + "\n".join(
        f"- {t}: {'PASS' if ok else 'FAIL'}" for t, ok in tests
    )
    verdict = "PASS" if passed == len(tests) else f"FAIL ({passed}/{len(tests)})"
    return verdict, review, (passed, len(tests))


def run_capstone() -> CrewArtifact:
    art = CrewArtifact()
    art.spec = _pm("Implement FizzBuzz with tests", art.cost)
    art.code = _engineer(art.spec, art.cost)
    verdict, review, counts = _reviewer(art.spec, art.code, art.cost)
    art.verdict = verdict
    art.review = review
    art.test_count = counts
    art.rounds = 1
    return art


def to_md(art: CrewArtifact) -> str:
    passed, total = art.test_count
    icon = "[PASS]" if art.verdict == "PASS" else "[FAIL]"
    return (
        "# 3-Agent Coding Crew Capstone\n\n"
        "## PM spec\n"
        f"```\n{art.spec}```\n\n"
        "## Engineer code\n"
        f"```python\n{art.code}```\n\n"
        "## Reviewer\n"
        f"{art.review}\n\n"
        f"## Verdict: {icon} ({passed}/{total} tests)\n"
        f"## Cost\n"
        f"- tokens_in: {art.cost.total_tokens_in}\n"
        f"- tokens_out: {art.cost.total_tokens_out}\n"
        f"- llm_calls: {art.cost.n_llm_calls}\n"
        f"- ~cost_usd: {art.cost.usd()}\n"
    )


def _self_test() -> None:
    art = run_capstone()
    assert art.verdict == "PASS", art.verdict
    assert art.test_count == (5, 5), art.test_count
    assert "fizzbuzz" in art.code
    assert "Spec" in art.spec
    assert art.cost.n_llm_calls == 3
    assert "PM" in art.cost.by_agent
    assert "Engineer" in art.cost.by_agent
    assert "Reviewer" in art.cost.by_agent
    print("[OK] capstone_coding_crew._self_test passed (3 agents, 5/5 tests)")


if __name__ == "__main__":
    _self_test()
    print()
    print(to_md(run_capstone()))
