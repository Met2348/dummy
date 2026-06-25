"""mini-AI-Scientist 入口：端到端跑五阶段，打印诚实光谱 + 两个批判演示。

两种跑法都支持：
- 脚本：python learning/auto-research-frontier/m9.5-end-to-end-ai-scientist/src/run.py --idea all
- 包：  （从 src/ 目录）python -m mini_ai_scientist ...（见 __main__.py 可自行加）
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

# —— sys.path 引导：让 `python <repo>/.../src/run.py` 也能 import 同目录的包 ——
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mini_ai_scientist.pipeline import run_all          # noqa: E402
from mini_ai_scientist.review import demonstrate_gaming  # noqa: E402
from mini_ai_scientist.ideation import IDEA_BANK         # noqa: E402


def _resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description="端到端 mini-AI-Scientist（真训练闭环）")
    ap.add_argument("--idea", default="all",
                    help="idea id（见 IDEA_BANK）或 'all'")
    ap.add_argument("--out", default=None, help="输出目录；缺省用 tempdir")
    ap.add_argument("--seeds", type=int, default=5, help="重复的模型种子数")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = _resolve_device(args.device)
    out_dir = args.out or tempfile.mkdtemp(prefix="mini_ai_scientist_")
    idea_ids = None if args.idea == "all" else [args.idea]

    print(f"=== mini-AI-Scientist | device={device} | seeds={args.seeds} "
          f"epochs={args.epochs} n={args.n_samples} ===")
    results = run_all(out_dir, n_seeds=args.seeds, epochs=args.epochs,
                      n_samples=args.n_samples, device=device, idea_ids=idea_ids)

    # —— 诚实光谱：事前自评 novelty vs 事后真实 verdict ——
    print("\n[五阶段闭环 · 诚实结果光谱]")
    print(f"  {'idea':14}{'self_nov':>9}{'base→treat':>16}{'Δ':>9}{'verdict':>14}{'review':>8}")
    gap_cases = []
    for r in results:
        print(f"  {r['idea_id']:14}{r['self_novelty']:>9.2f}"
              f"{r['baseline_mean']:>7.3f}→{r['treatment_mean']:<7.3f}"
              f"{r['delta']:>+9.3f}{r['verdict']:>14}{r['review_overall']:>8.1f}")
        gap_cases.append((r['idea_id'], r['self_novelty'], r['verdict'], r['delta']))

    # —— 教学点 1：ideation-execution gap ——
    print("\n[教学点 1 · ideation-execution gap]")
    for iid, nov, v, d in gap_cases:
        if v != "supported":
            print(f"  '{iid}'：自评 novelty {nov:.2f}（觉得是个想法），真做出来却 **{v}**（Δ={d:+.3f}）"
                  f" —— 想 ≠ 做得出")

    # —— 教学点 2：grading-its-own-homework（评审可被刷）——
    g = demonstrate_gaming()
    print("\n[教学点 2 · grading-its-own-homework：自动评审可被刷]")
    print(f"  诚实小真效果 overall={g['honest_small_real']['overall']} | "
          f"注水大假效果 overall={g['rigged_big_fake']['overall']} | "
          f"被刷={g['gamed']} → 别信评审分，看实验本身")
    # 有机证据：评分最高的 idea 若不是 supported，就是"评审只看效果大小不看真伪"的现场
    if results:
        top = max(results, key=lambda r: r["review_overall"])
        if top["verdict"] != "supported":
            print(f"  ⚠ 现场实证：评分最高的是 '{top['idea_id']}'（overall={top['review_overall']}），"
                  f"它却被判 **{top['verdict']}**！评审奖励的是|Δ|大小而非对错。")

    print(f"\n报告与图已写入：{out_dir}")
    print(f"（共 {len(results)} 个 idea，{len(IDEA_BANK)} 个在库）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
