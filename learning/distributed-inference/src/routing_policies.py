"""Prefix-aware routing for multi-replica serving."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import hashlib


def prompt_hash(prefix: List[int]) -> str:
    return hashlib.sha1(bytes(repr(prefix), "utf-8")).hexdigest()[:12]


@dataclass
class RouterStats:
    hits: int = 0
    misses: int = 0
    loads: Dict[int, int] = field(default_factory=dict)


class Router:
    def __init__(self, n_replicas: int):
        self.n = n_replicas
        self.stats = RouterStats(loads={i: 0 for i in range(n_replicas)})
        self.affinity: Dict[str, int] = {}

    def route(self, prompt_ids: List[int], policy: str) -> int:
        if policy == "round_robin":
            r = sum(self.stats.loads.values()) % self.n
        elif policy == "random":
            r = hash(tuple(prompt_ids)) % self.n
        elif policy == "prefix_hash":
            h = prompt_hash(prompt_ids[: max(1, len(prompt_ids) // 4)])
            r = int(h, 16) % self.n
        elif policy == "consistent":
            h = prompt_hash(prompt_ids[: max(1, len(prompt_ids) // 4)])
            r = (int(h, 16) % (self.n * 16)) // 16
        elif policy == "load_aware_prefix":
            key = prompt_hash(prompt_ids[: max(1, len(prompt_ids) // 4)])
            if key in self.affinity:
                r = self.affinity[key]
            else:
                # min-load replica
                r = min(self.stats.loads, key=lambda k: self.stats.loads[k])
                self.affinity[key] = r
        else:
            raise ValueError(policy)
        self.stats.loads[r] += 1
        return r


def evaluate(policy: str, prompts: List[List[int]], n_replicas: int) -> Dict:
    router = Router(n_replicas)
    seen_per_replica: Dict[int, set] = {i: set() for i in range(n_replicas)}
    hits = 0
    for p in prompts:
        key = prompt_hash(p[: max(1, len(p) // 4)])
        r = router.route(p, policy)
        if key in seen_per_replica[r]:
            hits += 1
        seen_per_replica[r].add(key)
    loads = list(router.stats.loads.values())
    return {
        "policy": policy,
        "hit_rate": hits / len(prompts),
        "load_max": max(loads),
        "load_min": min(loads),
        "imbalance": (max(loads) - min(loads)) / max(sum(loads) / len(loads), 1e-9),
    }


def demo() -> None:
    """Compare routing policies on prefix cache hit-rate and load balance."""
    import random

    rng = random.Random(0)
    # A few hot shared prefixes (e.g. system prompts) reused across many requests,
    # plus a unique suffix per request -> prefix-aware routing should hit the cache.
    shared_prefixes = [[1000 + p] * 16 for p in range(8)]
    prompts: List[List[int]] = []
    for _ in range(2000):
        pref = rng.choice(shared_prefixes)
        prompts.append(pref + [rng.randint(0, 9999)])

    print("=== Prefix-aware routing (4 replicas, 2000 reqs, 8 hot prefixes) ===")
    print(f"{'policy':>18}{'hit_rate':>10}{'imbalance':>11}")
    for policy in ("round_robin", "random", "prefix_hash", "consistent", "load_aware_prefix"):
        r = evaluate(policy, prompts, n_replicas=4)
        print(f"{policy:>18}{r['hit_rate']:>10.2%}{r['imbalance']:>11.3f}")
    print("\n=> prefix-aware policies route a shared prefix to the same replica,"
          "\n   so its KV cache is reused (high hit_rate); round_robin/random scatter it.")


if __name__ == "__main__":
    demo()
