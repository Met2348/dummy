"""MetaGPT-style SOP pipeline: PM to Architect to Engineer to QA."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Role:
    name: str
    profile: str
    actions: list[Callable] = field(default_factory=list)


@dataclass
class SOPArtifact:
    prd: str = ""
    design: str = ""
    task_list: list[str] = field(default_factory=list)
    code: str = ""
    test_report: str = ""


class MetaGPTPipeline:
    def __init__(self, requirement: str):
        self.req = requirement
        self.artifact = SOPArtifact()

    def run_pm(self) -> None:
        self.artifact.prd = (
            f"PRD for: {self.req}\n"
            f"- Inputs: int n from 1 to N\n"
            f"- Output: list following FizzBuzz rules\n"
            f"- Rules: %3==0 -> Fizz, %5==0 -> Buzz, both -> FizzBuzz"
        )

    def run_architect(self) -> None:
        self.artifact.design = (
            f"Design:\n- module: fizzbuzz.py\n- fn fizzbuzz(n) -> list[str]\n"
            f"- algorithm: iterate 1..n, apply rule"
        )

    def run_pm_task_split(self) -> None:
        self.artifact.task_list = [
            "implement fizzbuzz function",
            "add 5 unit tests",
            "handle edge case n=0",
        ]

    def run_engineer(self) -> None:
        self.artifact.code = (
            "def fizzbuzz(n):\n"
            "    out = []\n"
            "    for i in range(1, n+1):\n"
            "        if i % 15 == 0: out.append('FizzBuzz')\n"
            "        elif i % 3 == 0: out.append('Fizz')\n"
            "        elif i % 5 == 0: out.append('Buzz')\n"
            "        else: out.append(str(i))\n"
            "    return out"
        )

    def run_qa(self) -> None:
        # mock test execution by literally running the generated code
        ns: dict = {}
        exec(self.artifact.code, ns, ns)  # noqa: S102
        fb = ns["fizzbuzz"]
        tests = [
            fb(3) == ["1", "2", "Fizz"],
            fb(5) == ["1", "2", "Fizz", "4", "Buzz"],
            fb(15)[-1] == "FizzBuzz",
            fb(0) == [],
            fb(1) == ["1"],
        ]
        passed = sum(tests)
        self.artifact.test_report = f"Tests: {passed}/{len(tests)} passed"

    def kickoff(self) -> SOPArtifact:
        self.run_pm()
        self.run_architect()
        self.run_pm_task_split()
        self.run_engineer()
        self.run_qa()
        return self.artifact


def _self_test() -> None:
    pipe = MetaGPTPipeline("Build FizzBuzz")
    art = pipe.kickoff()
    assert "PRD" in art.prd
    assert "Design" in art.design
    assert len(art.task_list) == 3
    assert "fizzbuzz" in art.code
    assert "5/5" in art.test_report
    print("[OK] metagpt_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
