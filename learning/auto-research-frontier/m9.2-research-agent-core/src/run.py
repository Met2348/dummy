"""9.2 入口：跑研究 agent 的 ReAct 闭环，并排对比"带 critic vs 不带 critic"，
现场看 Reviewer 角色怎么把幻觉引用清掉。

跑法：
  python learning/auto-research-frontier/m9.2-research-agent-core/src/run.py
  python .../src/run.py --question "autonomous research agent for ideation"
  python .../src/run.py --no-critic     # 只看裸 ReAct（保留幻觉引用）
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from research_agent import ResearchAgent, ungrounded_in   # noqa: E402


def _show(result, title):
    print(f"\n=== {title} ===")
    print("[ReAct transcript]")
    print(result["transcript"])
    plan = result["plan"]
    ung = ungrounded_in(plan, result["retrieved_ids"])
    print("[最终研究计划]")
    print(f"  idea     : {plan['idea']}")
    print(f"  引用     : {list(plan['citations'])}")
    print(f"  baseline : {plan['baseline']}")
    print(f"  新颖度   : {plan['novelty']}")
    print(f"  幻觉引用残留: {len(ung)}  {ung if ung else ''}")
    return len(ung)


def main() -> int:
    ap = argparse.ArgumentParser(description="9.2 研究 agent 内核（ReAct + 工具 + 记忆 + 角色）")
    ap.add_argument("--question", default="autonomous research agent for ideation and evaluation")
    ap.add_argument("--no-critic", action="store_true",
                    help="关掉 Reviewer 角色，只跑裸 ReAct（用于对比）")
    args = ap.parse_args()

    agent = ResearchAgent()

    if args.no_critic:
        n = _show(agent.run(args.question, use_critic=False), "裸 ReAct（无 critic）")
        print(f"\n[结论] 无 critic：最终计划残留 {n} 处幻觉引用——没人查，它就留在论文里。")
        return 0

    # 默认：并排对比
    n_no = _show(agent.run(args.question, use_critic=False), "A. 无 critic（裸 ReAct）")
    n_yes = _show(agent.run(args.question, use_critic=True), "B. 有 critic（Reviewer 角色）")

    print("\n[教学点 · 内核里哪个零件在起作用]")
    print(f"  无 critic 残留幻觉引用: {n_no}  →  有 critic 残留: {n_yes}")
    print("  同一个生成模型、同一份检索，差别只在**是否有一个职责分离的 Reviewer 去查引用**。")
    print("  这就是'研究 agent 内核'里 Reflexion/多角色 这块零件的真实价值：")
    print("  规划+工具+记忆让它能跑，独立批判才让它**不自欺**（接 9.3 ideation gap / 9.8 红队）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
