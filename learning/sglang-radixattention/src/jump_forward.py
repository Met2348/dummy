"""Jump-Forward Decoding — skip steps when grammar forces a unique path."""
from __future__ import annotations

from typing import List, Tuple
from grammar_fsm import Fsm


def jump_forward(fsm: Fsm, state: int, max_lookahead: int = 64) -> Tuple[str, int]:
    """Greedily walk while exactly one character is legal.

    Returns (forced_string, new_state).  Empty string when no jump possible.
    """
    forced: List[str] = []
    cur = state
    for _ in range(max_lookahead):
        legal = fsm.legal_chars(cur)
        if len(legal) != 1:
            break
        ch = next(iter(legal))
        nxt = fsm.advance(cur, ch)
        if nxt is None:
            break
        forced.append(ch)
        cur = nxt
    return "".join(forced), cur


if __name__ == "__main__":
    from grammar_fsm import compile_literal, compile_digits_n, compile_concat
    fsm = compile_concat(
        compile_literal('{"name":"'),
        compile_digits_n(4),
        compile_literal('"}'),
    )
    forced, new_state = jump_forward(fsm, state=0)
    print(f"forced: {forced!r}  new_state={new_state}")
