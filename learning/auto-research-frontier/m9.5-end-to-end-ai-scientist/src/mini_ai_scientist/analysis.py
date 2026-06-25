"""阶段 4：结果分析 —— 对照比较 + 判定 + 画图。

核心是 `verdict()`：从"带误差棒的对照"里**诚实地**判定假设是否成立。
这是整个模块最关乎"科研诚信"的一函数——它决定了什么叫"真有效果"。
**这也是留给你打磨的地方**（见函数内 TODO：默认用 std 当粗略 CI，你可以换成真正的 t-检验）。
"""
from __future__ import annotations

import math
from typing import Dict


def compare(baseline: Dict, treatment: Dict) -> Dict:
    """比较两个 run_repeated 的输出，给出 delta 与合并标准差（粗略 CI）。"""
    bm, bs = baseline["test_acc_mean"], baseline["test_acc_std"]
    tm, ts = treatment["test_acc_mean"], treatment["test_acc_std"]
    delta = round(tm - bm, 4)
    combined_std = round(math.sqrt(bs ** 2 + ts ** 2), 4)
    return {
        "baseline_mean": bm, "baseline_std": bs,
        "treatment_mean": tm, "treatment_std": ts,
        "delta": delta, "combined_std": combined_std,
    }


def verdict(comparison: Dict, threshold: float = 0.01) -> str:
    """从对照里判定：supported / refuted / inconclusive。

    判定规则（默认）：效果要同时**超过噪声**（combined_std）和**超过最小实际意义**
    （threshold），才算数。否则 inconclusive。这条规则就是"什么算真效果"的科研判断——
    它直接决定模块会不会自欺。

    >>> TODO（留给你打磨，5–10 行）：把下面的"delta vs std"启发式换成更严谨的判据，
    >>> 例如对两组 test_accs 做 Welch t-检验、或用 bootstrap 置信区间。
    >>> 想一想：阈值设多大才既不放过真效果、又不被噪声忽悠？这正是科研诚信的核心抉择。
    """
    delta = comparison["delta"]
    margin = max(threshold, comparison["combined_std"])
    if delta > margin:
        return "supported"
    if delta < -margin:
        return "refuted"          # 比 baseline 更差（诚实的真阴）
    return "inconclusive"


def make_figure(idea_title: str, comparison: Dict, out_path: str) -> bool:
    """画 baseline vs treatment 的带误差棒柱状图。matplotlib 缺失则跳过（返回 False）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")        # 无显示后端
        import matplotlib.pyplot as plt
    except Exception:
        return False
    means = [comparison["baseline_mean"], comparison["treatment_mean"]]
    stds = [comparison["baseline_std"], comparison["treatment_std"]]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["baseline", "treatment"], means, yerr=stds, capsize=6,
           color=["#888", "#3b7"])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(idea_title, fontsize=9)
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True
