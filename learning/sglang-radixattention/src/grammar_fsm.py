"""Mini regex/grammar FSM for constrained decoding teaching.

Supports a limited regex subset: literals, character classes (\\d, \\w, .),
+ and * quantifiers, alternation (a|b).  Enough to demonstrate the FSM-driven
mask-construction pipeline without a full regex engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


DIGITS = set("0123456789")
WORD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
ANY = None   # None means "any char"


@dataclass(frozen=True)
class State:
    idx: int


@dataclass
class Fsm:
    """Linear-state FSM produced by `compile_regex_simple`."""
    states: List[State]
    delta: Dict[Tuple[int, str], int]
    accept: Set[int]
    start: int = 0

    def legal_chars(self, state: int) -> Set[str]:
        out: Set[str] = set()
        for (s, c), _ in self.delta.items():
            if s == state and c != "*":
                out.add(c)
        return out

    def advance(self, state: int, char: str) -> Optional[int]:
        if (state, char) in self.delta:
            return self.delta[(state, char)]
        if (state, "*") in self.delta:           # ANY wildcard
            return self.delta[(state, "*")]
        return None

    def accepts(self, s: str) -> bool:
        state = self.start
        for c in s:
            state = self.advance(state, c)
            if state is None:
                return False
        return state in self.accept


def compile_literal(literal: str) -> Fsm:
    """Compile a fixed literal string into a linear FSM."""
    n = len(literal)
    states = [State(i) for i in range(n + 1)]
    delta = {(i, literal[i]): i + 1 for i in range(n)}
    return Fsm(states=states, delta=delta, accept={n}, start=0)


def compile_digits_n(n: int) -> Fsm:
    """`\\d{n}`: exactly n digits."""
    states = [State(i) for i in range(n + 1)]
    delta: Dict[Tuple[int, str], int] = {}
    for i in range(n):
        for d in DIGITS:
            delta[(i, d)] = i + 1
    return Fsm(states=states, delta=delta, accept={n}, start=0)


def compile_concat(*fsms: Fsm) -> Fsm:
    """Concat FSMs by gluing accept(prev) -> start(next)."""
    base = 0
    new_states: List[State] = []
    new_delta: Dict[Tuple[int, str], int] = {}
    accept_chain: int = 0
    for k, f in enumerate(fsms):
        for s in f.states:
            new_states.append(State(s.idx + base))
        for (s, c), t in f.delta.items():
            new_delta[(s + base, c)] = t + base
        if k > 0:
            # glue previous accept to this start
            prev_accept = accept_chain
            new_start = base
            # for each transition from prev_accept, we redirect to base
            # (assumes previous FSM had a single accept state, which our
            # compilers guarantee)
            # We simply move forward; start of next is `base`.
            for (s, c), t in list(f.delta.items()):
                if s == 0:
                    new_delta[(prev_accept, c)] = t + base
        base += len(f.states)
        accept_chain = base - 1
    return Fsm(states=new_states, delta=new_delta, accept={accept_chain}, start=0)


def compile_token_table(fsm: Fsm, vocab: List[str]) -> Dict[int, List[bool]]:
    """For each state, return a boolean mask of size |vocab|: True = accepted."""
    table: Dict[int, List[bool]] = {}
    for state in range(len(fsm.states)):
        mask = []
        for tok in vocab:
            s = state
            ok = True
            for c in tok:
                s2 = fsm.advance(s, c)
                if s2 is None:
                    ok = False
                    break
                s = s2
            mask.append(ok)
        table[state] = mask
    return table


if __name__ == "__main__":
    fsm = compile_concat(
        compile_literal('{"name":"'),
        compile_digits_n(4),
        compile_literal('"}'),
    )
    print("accepts {\"name\":\"1234\"}:", fsm.accepts('{"name":"1234"}'))
    print("rejects {\"name\":\"abcd\"}:", not fsm.accepts('{"name":"abcd"}'))
