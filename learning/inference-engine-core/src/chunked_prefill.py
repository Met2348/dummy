"""Chunked prefill scheduler — mix decode with sliced prefill in one batch.

Each scheduling iteration assembles a token budget `max_tokens_per_iter` that
mixes decode tokens (1 per running req) with a slice of prefill from one
"prefilling" request.  We track prefill progress per request via `prefill_pos`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from common import Request, now


@dataclass
class PrefillState:
    req: Request
    prefill_pos: int = 0      # number of prompt tokens already prefilled

    @property
    def done(self) -> bool:
        return self.prefill_pos >= len(self.req.prompt_ids)


@dataclass
class ChunkedPrefillEngine:
    max_tokens_per_iter: int = 512
    decoding: List[Request] = field(default_factory=list)
    prefilling: List[PrefillState] = field(default_factory=list)
    finished: List[Request] = field(default_factory=list)
    iter_idx: int = 0

    def add_request(self, req: Request) -> None:
        self.prefilling.append(PrefillState(req))

    def step(self) -> dict:
        """One iteration: produce per-req chunks, return info dict."""
        budget = self.max_tokens_per_iter
        info = {"iter": self.iter_idx, "decode": 0, "prefill_chunks": []}

        # Reserve 1 token per decoding req
        decode_n = min(len(self.decoding), budget)
        budget -= decode_n
        info["decode"] = decode_n

        # Pump prefill slices while budget remains
        for ps in list(self.prefilling):
            if budget <= 0:
                break
            remaining = len(ps.req.prompt_ids) - ps.prefill_pos
            take = min(remaining, budget)
            ps.prefill_pos += take
            budget -= take
            info["prefill_chunks"].append((ps.req.rid, take, ps.prefill_pos))
            if ps.done:
                self.prefilling.remove(ps)
                self.decoding.append(ps.req)

        # advance decoders by 1 token each
        for r in list(self.decoding[:decode_n]):
            r.output_ids.append(0)
            if len(r.output_ids) >= r.max_new_tokens:
                r.finished = True
                self.finished.append(r)
                self.decoding.remove(r)

        self.iter_idx += 1
        return info

    def is_idle(self) -> bool:
        return not (self.decoding or self.prefilling)

    def run_until_idle(self) -> List[dict]:
        log = []
        while not self.is_idle():
            log.append(self.step())
        return log


if __name__ == "__main__":
    from common import Request as R
    eng = ChunkedPrefillEngine(max_tokens_per_iter=64)
    eng.add_request(R(rid=0, prompt_ids=list(range(200)), max_new_tokens=5))
    eng.add_request(R(rid=1, prompt_ids=list(range(20)), max_new_tokens=10))
    for step in eng.run_until_idle():
        print(step)
