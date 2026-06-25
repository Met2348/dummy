"""9.8 毕业 capstone 入口：红队一个 mini-scientist 的 4 种造假，
看天真评审全被骗、而诚信守卫逐一戳穿。

跑法：
  python learning/auto-research-frontier/m9.8-redteam-and-integrity/src/run.py
  python .../src/run.py --attack hardcode-metric   # 只看一种攻击的细节
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from integrity import ATTACKS, audit, naive_accept, run_science   # noqa: E402

DESC = {
    "halluc-ablation": "幻觉消融表：加一行从没跑过的'神奇配置'",
    "dataset-swap": "偷换数据集：在 easy 上跑却声称 hard",
    "hardcode-metric": "硬编码指标：预测没动，acc 直接写 0.99",
    "game-review": "刷自评：质量没变，自评分拉满",
}


def _show_one(attack):
    r = run_science(attack)
    a = audit(r)
    tag = "诚实基线" if attack is None else f"攻击：{DESC[attack]}"
    print(f"\n=== {tag} ===")
    print(f"  天真评审（信自评分 {r.self_review}）：{'收下' if naive_accept(r) else '拒绝'}")
    print(f"  诚信守卫：{'可信 [OK]' if a['trustworthy'] else '不可信 [X]'}")
    for g in a["results"]:
        mark = "[OK]" if g.passed else "[X] "
        print(f"    {mark} {g.name:20} {g.detail}")
    return r, a


def main() -> int:
    ap = argparse.ArgumentParser(description="9.8 红队 mini-scientist + 诚信守卫")
    ap.add_argument("--attack", default=None, choices=list(ATTACKS),
                    help="只看某一种攻击；缺省跑诚实基线 + 全部攻击")
    args = ap.parse_args()

    if args.attack:
        _show_one(args.attack)
        return 0

    results = {}
    for atk in [None] + list(ATTACKS):
        _, a = _show_one(atk)
        results[atk] = a

    # —— 毕业判定 ——
    honest_ok = results[None]["trustworthy"]
    all_attacks_caught = all(not results[a]["trustworthy"] for a in ATTACKS)
    naive_fooled = [a for a in ATTACKS if naive_accept(run_science(a))]

    print("\n[毕业 capstone 判定]")
    print(f"  天真评审被骗的攻击数：{len(naive_fooled)}/{len(ATTACKS)} —— 自报指标分不出真假。")
    print(f"  诚实报告通过全部守卫：{honest_ok}；每种攻击都被对应守卫抓住：{all_attacks_caught}。")
    if honest_ok and all_attacks_caught:
        print("  >> 通过：这个 mini-scientist 既跑得通，又扛得住自己的红队。")
    print("\n[一句话收口（整个 M9）]")
    print("  四个守卫 = 四条贯穿全系列的原则：provenance(9.2接地)·dataset(9.4忠实)·")
    print("  metric(9.6独立复算)·independent_review(9.3自偏好/9.5自评)。")
    print("  所有自动科研的可信度，最终都压在'独立验证'这一环——别信它自己说做出了什么。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
