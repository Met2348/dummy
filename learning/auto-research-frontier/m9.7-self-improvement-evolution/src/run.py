"""9.7 入口：让自我改进循环分别在"可刷代理 fitness"和"真目标 fitness"下进化，
并排看 reward hacking——代理那条档案在涨，真本事却纹丝不动。

跑法：
  python learning/auto-research-frontier/m9.7-self-improvement-evolution/src/run.py
  python .../src/run.py --gens 20
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from selfimprove import (                                       # noqa: E402
    SEED, evolve, holdout_fitness, naive_fitness,
)


def _run(name, fitness, gens):
    best, archive = evolve(fitness, gens=gens)
    print(f"\n=== 在 {name} 下自我改进 ===")
    print(f"  {'gen':>4}{'threshold':>11}{'memo':>6}{'naive':>9}{'holdout':>9}")
    for gen, g, _fv, nv, hv in archive:
        print(f"  {gen:>4}{g.threshold:>11.1f}{len(g.memo):>6}{nv:>9.3f}{hv:>9.3f}")
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="9.7 自我改进 / reward hacking 现场")
    ap.add_argument("--gens", type=int, default=14)
    args = ap.parse_args()

    print(f"种子解：threshold={SEED.threshold}（真值边界是 0）| "
          f"naive={naive_fitness(SEED):.3f} holdout={holdout_fitness(SEED):.3f}")

    best_naive = _run("naive fitness（代理：在泄漏测试集上判分，可被背书刷）", naive_fitness, args.gens)
    best_hold = _run("holdout fitness（真目标：在没见过的数据上判分）", holdout_fitness, args.gens)

    n_n, n_h = naive_fitness(best_naive), holdout_fitness(best_naive)
    h_n, h_h = naive_fitness(best_hold), holdout_fitness(best_hold)

    print("\n[教学点 · fitness 被 game = reward hacking]")
    print(f"  优化代理 naive：naive {naive_fitness(SEED):.3f}→{n_n:.3f}（涨满），"
          f"但 holdout {holdout_fitness(SEED):.3f}→{n_h:.3f}（没动）。")
    print(f"    它没学会任务，只是把泄漏测试集**背**了下来（memo={len(best_naive.memo)}，threshold 没变）。")
    print(f"  优化真目标 holdout：holdout {holdout_fitness(SEED):.3f}→{h_h:.3f}（真涨），"
          f"靠的是把 threshold 调到 ~0（真泛化）。")
    print(f"  代理-真目标缺口：优化代理后 gap={n_n - n_h:+.3f}，优化真目标后 gap={h_n - h_h:+.3f}。")
    print("  → **档案在涨 ≠ 系统在变强。** 当 fitness 可被刷，自我改进会自动找到那条捷径。")
    print("  （这就是 process-reward 里 reward hacking 的进化版；守卫见 9.8：held-out + 独立验证 + 记执行日志。）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
