"""Constrained sampler: apply grammar mask to logits before sampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class ConstrainedSampler:
    fsm: object              # any FSM exposing advance(state, char) -> int|None
    vocab: List[str]
    state: int = 0

    def legal_token_mask(self) -> List[bool]:
        """Return |V| bool list — True for tokens that keep us legal."""
        mask: List[bool] = []
        for tok in self.vocab:
            s = self.state
            ok = True
            for c in tok:
                s2 = self.fsm.advance(s, c)
                if s2 is None:
                    ok = False
                    break
                s = s2
            mask.append(ok)
        return mask

    def apply(self, logits: List[float]) -> List[float]:
        mask = self.legal_token_mask()
        out = list(logits)
        for i, ok in enumerate(mask):
            if not ok:
                out[i] = -math.inf
        return out

    def commit(self, token_id: int) -> bool:
        """Advance state with the picked token's chars; return False if invalid."""
        tok = self.vocab[token_id]
        s = self.state
        for c in tok:
            s2 = self.fsm.advance(s, c)
            if s2 is None:
                return False
            s = s2
        self.state = s
        return True

    def at_accept(self) -> bool:
        return self.state in self.fsm.accept


if __name__ == "__main__":
    from grammar_fsm import compile_literal, compile_digits_n, compile_concat
    fsm = compile_concat(compile_literal("v"), compile_digits_n(2))
    # toy vocab
    vocab = ["a", "v", "1", "2", "12", "v12"]
    sampler = ConstrainedSampler(fsm=fsm, vocab=vocab)
    print("legal at start:", sampler.legal_token_mask())
    sampler.commit(vocab.index("v"))
    print("legal after 'v':", sampler.legal_token_mask())
    sampler.commit(vocab.index("12"))
    print("at accept after 'v','12':", sampler.at_accept())
