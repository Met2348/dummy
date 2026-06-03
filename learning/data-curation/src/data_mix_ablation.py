"""数据配比 ablation — 在玩具配比下看 ppl 变化.

教学目标：
    1. MixSampler 抽样器
    2. 不同配比下 toy LM 的 perplexity 变化
    3. 演示 ablation 的方法论（不真训）

运行：
    python data_mix_ablation.py --demo
"""
from __future__ import annotations

import argparse
import math
import random
from typing import Iterable


class MixSampler:
    """按权重从 domain 抽样的迭代器."""
    def __init__(self, weights: dict[str, float], domains: dict[str, list[str]],
                 seed: int = 42):
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        self.domains = domains
        self.rng = random.Random(seed)

    def sample(self, n: int) -> Iterable[tuple[str, str]]:
        keys = list(self.weights.keys())
        probs = list(self.weights.values())
        for _ in range(n):
            d = self.rng.choices(keys, probs)[0]
            yield d, self.rng.choice(self.domains[d])


def mock_ppl(samples: list[tuple[str, str]], domain_difficulty: dict[str, float]) -> float:
    """假装 ppl 来自 domain 难度（玩具用，不真训）."""
    if not samples:
        return float("inf")
    nll = 0.0
    for d, _ in samples:
        nll += domain_difficulty.get(d, 5.0)
    return math.exp(nll / len(samples))


def run_demo() -> None:
    domains = {
        "web":  ["sample web doc " + str(i) for i in range(100)],
        "code": ["def f(): return " + str(i) for i in range(100)],
        "math": ["x^2 + y^2 = " + str(i) for i in range(100)],
        "wiki": ["Wikipedia article " + str(i) for i in range(100)],
    }
    # 玩具难度：web 平均（基线 2.5），code 较难（3.0），math 难（3.5），wiki 易（2.0）
    difficulty = {"web": 2.5, "code": 3.0, "math": 3.5, "wiki": 2.0}

    configs = [
        ("uniform",     {"web": 0.25, "code": 0.25, "math": 0.25, "wiki": 0.25}),
        ("Llama-like",  {"web": 0.67, "code": 0.05, "math": 0.03, "wiki": 0.25}),
        ("Code-heavy",  {"web": 0.30, "code": 0.50, "math": 0.10, "wiki": 0.10}),
        ("Math-heavy",  {"web": 0.30, "code": 0.10, "math": 0.50, "wiki": 0.10}),
    ]

    print(f"\n{'config':<14}  | {'web':>5} | {'code':>5} | {'math':>5} | {'wiki':>5} | {'mock_ppl':>10}")
    print("-" * 75)
    for name, w in configs:
        sampler = MixSampler(w, domains, seed=42)
        samples = list(sampler.sample(1000))
        ppl = mock_ppl(samples, difficulty)
        actual_freq = {k: sum(1 for d, _ in samples if d == k) / len(samples)
                       for k in w}
        print(f"{name:<14}  | {actual_freq['web']:>5.2f} | "
              f"{actual_freq['code']:>5.2f} | {actual_freq['math']:>5.2f} | "
              f"{actual_freq['wiki']:>5.2f} | {ppl:>10.2f}")

    print("\n(玩具 ppl 数值；真实 ablation 需训 1B 代理模型。)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        run_demo()


if __name__ == "__main__":
    main()
