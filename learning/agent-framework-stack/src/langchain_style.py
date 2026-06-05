"""LangChain LCEL `|` style mock."""
from __future__ import annotations
from typing import Any, Callable


class Runnable:
    def __or__(self, other: "Runnable") -> "RunnableSequence":
        return RunnableSequence([self, other])

    def invoke(self, input: Any) -> Any:
        raise NotImplementedError


class RunnableSequence(Runnable):
    def __init__(self, steps: list[Runnable]):
        self.steps = []
        for s in steps:
            if isinstance(s, RunnableSequence):
                self.steps.extend(s.steps)
            else:
                self.steps.append(s)

    def invoke(self, input: Any) -> Any:
        out = input
        for s in self.steps:
            out = s.invoke(out)
        return out


class RunnableLambda(Runnable):
    def __init__(self, fn: Callable[[Any], Any]):
        self.fn = fn

    def invoke(self, input: Any) -> Any:
        return self.fn(input)


class RunnableParallel(Runnable):
    def __init__(self, **branches: Runnable):
        self.branches = branches

    def invoke(self, input: Any) -> dict:
        return {name: r.invoke(input) for name, r in self.branches.items()}


class PromptTemplate(Runnable):
    def __init__(self, template: str):
        self.template = template

    def invoke(self, input: Any) -> str:
        if isinstance(input, dict):
            return self.template.format(**input)
        return self.template.format(input=input)


class MockLLM(Runnable):
    def invoke(self, prompt: Any) -> str:
        return f"LLM[{str(prompt)[:80]}]"


def _self_test() -> None:
    prompt = PromptTemplate("Q: {q}")
    llm = MockLLM()
    chain = prompt | llm
    result = chain.invoke({"q": "What is ReAct?"})
    assert "ReAct" in result, result
    assert "LLM[" in result, result

    parallel = RunnableParallel(
        upper=RunnableLambda(lambda x: x.upper()),
        len=RunnableLambda(lambda x: len(x)),
    )
    out = parallel.invoke("hello")
    assert out["upper"] == "HELLO" and out["len"] == 5
    print("[OK] langchain_style._self_test passed")


if __name__ == "__main__":
    _self_test()
