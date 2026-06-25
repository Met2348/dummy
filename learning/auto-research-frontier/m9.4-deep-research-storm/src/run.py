"""9.4 入口：跑 mini-STORM 综述，再用两种检查审它的引用，
现场看"引用存在 ≠ 引用忠实"。

跑法：
  python learning/auto-research-frontier/m9.4-deep-research-storm/src/run.py
  python .../src/run.py --topic "autonomous research agents"
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mini_storm import audit, check_sentence, synthesize   # noqa: E402

MARK = {"faithful": "[OK ]", "unfaithful": "[!! ]", "dangling": "[DNG]", "uncited": "[---]"}


def main() -> int:
    ap = argparse.ArgumentParser(description="9.4 mini-STORM + 引用忠实度核查")
    ap.add_argument("--topic", default="how AI research agents do retrieval and avoid pitfalls")
    args = ap.parse_args()

    report = synthesize(args.topic)
    print(f"=== 9.4 mini-STORM | 主题：{args.topic} ===")
    print(f"[多视角提问] {len(report.perspectives)} 个视角")
    print(f"[检索到文档] {list(report.retrieved_ids)}\n")

    print("[合成综述 · 逐句忠实度]")
    for s in report.sentences:
        v = check_sentence(s)
        print(f"  {MARK[v]} {s.text}")

    a = audit(report)
    print(f"\n[审计] 共 {a['total']} 句 | 忠实 {a['faithful']} | "
          f"不忠实 {len(a['unfaithful'])} | 悬空 {len(a['dangling'])}")
    print("\n[教学点 · 引用存在 ≠ 引用忠实]")
    print(f"  naive 检查（只问 id 在不在库）：{a['existence_pass']}/{a['total']} 句通过 —— 看起来全合规。")
    print(f"  忠实度检查（问被引文献真支持吗）：揪出 {len(a['unfaithful'])} 句不忠实：")
    for s in a["unfaithful"]:
        cited = s.cited_doc
        print(f"    · {s.text}")
        print(f"      → 引的 [{cited}] 真实存在，但它并不支持 '{sorted(s.claim_tokens)}' 这个论断。")
    print("  → 只核对'引用是否存在'会被骗过；必须核对'被引内容是否真支持本句'。")
    print("  （这正是 STORM/OpenScholar 反复强调的 citation faithfulness；通往 9.6 评测 / 9.8 红队。）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
