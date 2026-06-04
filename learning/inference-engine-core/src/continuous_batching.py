"""Continuous batching scheduler skeleton.

Mirrors Orca/vLLM's iteration-level scheduling without owning a real model.
A `mock_forward` is wired in for tests; swap with HF model.generate logits for
real runs.

Loop:
  1. Admit pending requests if KV budget allows.
  2. Run one forward over `running`.
  3. Append the sampled token to each request, retire finished ones.
  4. Repeat until `pending` and `running` are both empty.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional
from common import Request, EngineMetrics, now, stub_token_step


@dataclass
class Engine:
    max_running: int = 8
    kv_budget: int = 1024                     # total tokens of headroom
    forward_fn: Callable[[List[Request]], List[int]] = stub_token_step
    eos_id: int = -1

    pending: Deque[Request] = field(default_factory=deque)
    running: List[Request] = field(default_factory=list)
    finished: List[Request] = field(default_factory=list)
    metrics: EngineMetrics = field(default_factory=EngineMetrics)

    def add_request(self, req: Request) -> None:
        req.t_admit = now()
        self.pending.append(req)
        self.metrics.n_requests += 1
        self.metrics.n_tokens_in += len(req.prompt_ids)

    def _kv_used(self) -> int:
        return sum(r.total_len for r in self.running)

    def _can_admit(self, req: Request) -> bool:
        need = len(req.prompt_ids) + req.max_new_tokens
        return (
            len(self.running) < self.max_running
            and self._kv_used() + need <= self.kv_budget
        )

    def step(self) -> bool:
        """Run one scheduling iteration.  Returns True if anything happened."""
        # admission
        admitted = 0
        while self.pending and self._can_admit(self.pending[0]):
            self.running.append(self.pending.popleft())
            admitted += 1
        if not self.running:
            return admitted > 0

        # forward (callable returns one token per running req)
        if callable(self.forward_fn):
            # By default each request samples its own token via stub
            if self.forward_fn is stub_token_step:
                tokens = [stub_token_step(r) for r in self.running]
            else:
                tokens = self.forward_fn(self.running)
        else:
            tokens = [stub_token_step(r) for r in self.running]

        # record + retire
        still_running: List[Request] = []
        for r, tok in zip(self.running, tokens):
            if not r.output_ids:
                r.t_first_token = now()
            r.output_ids.append(int(tok))
            self.metrics.n_tokens_out += 1
            done = (int(tok) == self.eos_id) or (len(r.output_ids) >= r.max_new_tokens)
            if done:
                r.finished = True
                r.t_finish = now()
                self.finished.append(r)
            else:
                still_running.append(r)
        self.running = still_running
        return True

    def run(self) -> EngineMetrics:
        self.metrics.t_start = now()
        while self.pending or self.running:
            self.step()
        self.metrics.t_end = now()
        return self.metrics


def make_request(rid: int, prompt_len: int = 10, max_new: int = 20) -> Request:
    return Request(rid=rid, prompt_ids=list(range(prompt_len)), max_new_tokens=max_new)


if __name__ == "__main__":
    eng = Engine(max_running=4, kv_budget=200)
    for i in range(10):
        eng.add_request(make_request(i, prompt_len=8, max_new=12))
    m = eng.run()
    print(m.report())
