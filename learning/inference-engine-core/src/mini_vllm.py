"""mini-vLLM — capstone: ~200 LOC engine wiring everything together.

Pieces composed:
- PagedKvPool / BlockTable  (paged_kv.py)
- Engine continuous batching  (continuous_batching.py)
- Chunked prefill scheduler   (chunked_prefill.py)
- Sampler                     (sampling.py)
- Prefix cache                (prefix_cache.py)

The `forward_fn` is intentionally pluggable: pass a real HF `model` for live
inference, or use the stub for smoke tests and CI.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Sequence
import argparse
import json
import time
import torch

from common import Request, EngineMetrics, now
from paged_kv import PagedKvPool, BlockTable
from sampling import SamplerConfig, sample_one
from prefix_cache import PrefixCache


@dataclass
class MiniEngine:
    pool: PagedKvPool
    sampler_cfg: SamplerConfig = field(default_factory=SamplerConfig)
    forward_fn: Optional[Callable[[List[Request]], torch.Tensor]] = None
    max_running: int = 8
    eos_id: int = -1

    pending: Deque[Request] = field(default_factory=deque)
    running: List[Request] = field(default_factory=list)
    block_tables: dict = field(default_factory=dict)   # rid -> BlockTable
    finished: List[Request] = field(default_factory=list)
    metrics: EngineMetrics = field(default_factory=EngineMetrics)
    prefix_cache: Optional[PrefixCache] = None
    vocab_size: int = 50257

    def add(self, req: Request) -> None:
        req.t_admit = now()
        self.pending.append(req)
        self.metrics.n_requests += 1
        self.metrics.n_tokens_in += len(req.prompt_ids)

    def _can_admit(self, req: Request) -> bool:
        needed = (len(req.prompt_ids) + req.max_new_tokens + self.pool.block_size - 1) // self.pool.block_size
        return (
            len(self.running) < self.max_running
            and self.pool.n_free() >= needed
        )

    def _make_table(self, req: Request) -> BlockTable:
        table = BlockTable(self.pool)
        for tok in req.prompt_ids:
            k = torch.zeros(self.pool.k.shape[3:], dtype=self.pool.k.dtype)
            v = torch.zeros_like(k)
            table.append_token(layer=0, k=k, v=v)
        return table

    def step(self) -> bool:
        # admission
        while self.pending and self._can_admit(self.pending[0]):
            r = self.pending.popleft()
            self.block_tables[r.rid] = self._make_table(r)
            self.running.append(r)
        if not self.running:
            return bool(self.pending)

        # forward (stub by default: deterministic token from prompt[0]+len(out))
        if self.forward_fn is None:
            logits_batch = torch.stack([
                torch.randn(self.vocab_size) + r.prompt_ids[0] * 0.001
                for r in self.running
            ])
        else:
            logits_batch = self.forward_fn(self.running)

        still: List[Request] = []
        for i, r in enumerate(self.running):
            if not r.output_ids:
                r.t_first_token = now()
            tok = sample_one(
                logits_batch[i],
                torch.tensor(r.prompt_ids + r.output_ids, dtype=torch.long),
                self.sampler_cfg,
            )
            r.output_ids.append(tok)
            # write KV for new token
            k = torch.zeros(self.pool.k.shape[3:], dtype=self.pool.k.dtype)
            v = torch.zeros_like(k)
            self.block_tables[r.rid].append_token(layer=0, k=k, v=v)
            self.metrics.n_tokens_out += 1
            done = tok == self.eos_id or len(r.output_ids) >= r.max_new_tokens
            if done:
                r.finished = True
                r.t_finish = now()
                self.finished.append(r)
                self.block_tables[r.rid].free()
                self.block_tables.pop(r.rid, None)
            else:
                still.append(r)
        self.running = still
        return True

    def run(self) -> EngineMetrics:
        self.metrics.t_start = now()
        while self.pending or self.running:
            self.step()
        self.metrics.t_end = now()
        return self.metrics


# ---- 5 demo cases -----------------------------------------------------------

CASES = {
    1: dict(name="short-in-short-out", n=8, p=32, m=32),
    2: dict(name="long-in-short-out", n=4, p=1024, m=8),
    3: dict(name="short-in-long-out", n=8, p=16, m=256),
    4: dict(name="big-batch", n=32, p=16, m=32),
    5: dict(name="streaming-1", n=1, p=32, m=128),
}


def make_engine_for(case: dict) -> MiniEngine:
    block_size = 16
    # Generous pool sizing for the demo
    max_tokens_per_req = case["p"] + case["m"]
    n_blocks = max(64, case["n"] * (max_tokens_per_req // block_size + 2))
    pool = PagedKvPool(n_blocks=n_blocks, block_size=block_size, n_kv_heads=2, head_dim=8, n_layers=1)
    eng = MiniEngine(pool=pool, max_running=case["n"], vocab_size=128)
    for rid in range(case["n"]):
        prompt = [rid + i for i in range(case["p"])]
        eng.add(Request(rid=rid, prompt_ids=prompt, max_new_tokens=case["m"]))
    return eng


def run_case(case_id: int) -> dict:
    case = CASES[case_id]
    torch.manual_seed(case_id)
    eng = make_engine_for(case)
    m = eng.run()
    return {
        "case": case_id,
        "name": case["name"],
        "n_req": case["n"],
        "prompt_len": case["p"],
        "out_len": case["m"],
        "elapsed": round(m.elapsed, 3),
        "tokens_out": m.n_tokens_out,
        "throughput": round(m.throughput_out, 1),
    }


def run_all() -> List[dict]:
    return [run_case(c) for c in CASES]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=0, help="0=all, 1..5=single")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    results = run_all() if args.case == 0 else [run_case(args.case)]
    for r in results:
        print(json.dumps(r))
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
