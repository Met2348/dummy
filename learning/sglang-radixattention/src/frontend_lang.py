"""Educational SGLang-frontend mock — supports gen/select/fork primitives.

A Stream owns a prompt buffer and a `vars` dict keyed by gen-name.  fork()
shallow-copies the stream so multiple branches inherit the same prefix.

No real LLM is invoked; a pluggable `generator(prompt, name) -> str` lets
tests assert on call patterns without dependency on a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


def default_generator(prompt: str, name: str, max_tokens: int = 16) -> str:
    return f"<gen:{name}@{len(prompt)}>"


@dataclass
class Stream:
    prompt: str = ""
    vars: Dict[str, str] = field(default_factory=dict)
    generator: Callable[[str, str, int], str] = default_generator
    parent: Optional["Stream"] = None
    n_forwards: int = 0   # how many model invocations this stream caused

    def __iadd__(self, other) -> "Stream":
        if isinstance(other, str):
            self.prompt += other
        elif isinstance(other, Gen):
            text = self.generator(self.prompt, other.name, other.max_tokens)
            self.vars[other.name] = text
            self.prompt += text
            self.n_forwards += 1
        elif isinstance(other, Select):
            best = max(other.choices, key=lambda c: len(c))   # mock judge
            self.vars[other.name] = best
            self.prompt += best
            self.n_forwards += 1
        else:
            raise TypeError(f"unsupported operand: {type(other)}")
        return self

    def fork(self, k: int) -> List["Stream"]:
        return [
            Stream(prompt=self.prompt, vars=dict(self.vars), generator=self.generator, parent=self)
            for _ in range(k)
        ]


@dataclass
class Gen:
    name: str
    max_tokens: int = 16
    stop: Optional[List[str]] = None


@dataclass
class Select:
    name: str
    choices: List[str]


def function(fn):
    """Decorator analogous to @sgl.function — creates a Stream and runs fn."""
    def wrapper(*args, generator=None, **kwargs):
        s = Stream(generator=generator) if generator else Stream()
        fn(s, *args, **kwargs)
        return s
    return wrapper


if __name__ == "__main__":
    @function
    def tot(s, q):
        s += f"Q: {q}\nThought 1: "
        forks = s.fork(3)
        for f in forks:
            f += Gen("thought", max_tokens=30)
        s += Select("choice", choices=[f.vars["thought"] for f in forks])
        s += "\nAnswer: "
        s += Gen("answer")
    out = tot("1+1?")
    print(out.prompt)
    print("vars:", out.vars)
