"""
stats.py — 实验结果的统计处理: 均值±方差 / bootstrap 置信区间 / 显著性检验 / 效应量.

为什么需要它 (9.4-L5 的核心): ML 论文里最常见的造假/自欺不是编数据, 而是
**用单个种子的一次结果当结论** —— 而方法间 0.3% 的差距, 可能完全淹没在种子噪声里。
会做研究 = 永远报方差、永远问「这个差距统计上站得住吗」。本模块给你三件武器:
  1. mean±std + 误差棒  —— 最低限度的诚实。
  2. bootstrap 置信区间 —— 不假设分布, 直接从数据重采样估计不确定性 (教学手写)。
  3. 显著性检验 + 效应量 —— 「差距是真的吗 (p)」+「差距有多大 (Cohen's d)」。

依赖: numpy (必需) + scipy (t 检验, 没有则回退到正态近似)。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def mean_std(xs) -> tuple[float, float]:
    """返回 (均值, 样本标准差 ddof=1). 报告 win_rate = mean ± std 是最低门槛."""
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def sem(xs) -> float:
    """标准误 (standard error of the mean) = std / sqrt(n). 误差棒常用它而非 std."""
    a = np.asarray(xs, dtype=float)
    return float(a.std(ddof=1) / np.sqrt(len(a)))


def bootstrap_ci(xs, n_boot: int = 10000, ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """bootstrap 置信区间: 从样本有放回重采样 n_boot 次, 取每次均值的分位数.

    为什么用它: 不假设数据服从正态分布 (ML 指标常常不正态), 直接用数据本身估不确定性。
    思想: 「如果重做很多次实验, 均值会落在哪个区间」—— 用重采样模拟「重做很多次」。
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(xs, dtype=float)
    boot_means = np.array([rng.choice(a, size=len(a), replace=True).mean()
                           for _ in range(n_boot)])
    lo = float(np.quantile(boot_means, (1 - ci) / 2))
    hi = float(np.quantile(boot_means, 1 - (1 - ci) / 2))
    return lo, hi


def welch_t_test(a, b) -> tuple[float, float]:
    """Welch t 检验 (不假设两组方差相等), 返回 (t 统计量, 双尾 p 值).

    问的问题: 「a 和 b 的均值差, 有多大可能纯属种子噪声的巧合?」 p 小 → 不像巧合。
    有 scipy 用 scipy; 没有则用正态近似 (大样本下够用, 教学透明)。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        from scipy import stats
        t, p = stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    except Exception:
        # 正态近似回退
        ma, mb = a.mean(), b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = (ma - mb) / se if se > 0 else 0.0
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        return float(t), float(p)


def cohens_d(a, b) -> float:
    """效应量 Cohen's d = 均值差 / 合并标准差. 回答「差距有多大」(p 只回答「是不是真的」).

    经验阈值: |d|≈0.2 小, 0.5 中, 0.8 大。一个 p<0.05 但 d=0.05 的结果"统计显著但无关紧要"。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def summarize_comparison(a, b, label_a="A", label_b="B") -> str:
    """把两组结果的对比打成一段人话: 均值±std, 差距, p, d, 以及一句裁决."""
    ma, sa = mean_std(a)
    mb, sb = mean_std(b)
    t, p = welch_t_test(a, b)
    d = cohens_d(a, b)
    lo, hi = bootstrap_ci(np.asarray(a) - np.asarray(b)) if len(a) == len(b) else (float("nan"),) * 2
    verdict = "显著且非平凡" if (p < 0.05 and abs(d) >= 0.5) else \
              "显著但效应小" if p < 0.05 else "不显著 (差距可能只是种子噪声)"
    lines = [
        f"{label_a}: {ma:.3f} ± {sa:.3f}   {label_b}: {mb:.3f} ± {sb:.3f}",
        f"差距 (A-B): {ma - mb:+.3f}",
        f"Welch t={t:.2f}, p={p:.4f}   Cohen's d={d:.2f}",
        f"裁决: {verdict}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import experiment as ex
    runs = ex.ablation_grid(seeds=range(8))
    # 在高噪声下比 DPO vs Robust-DPO
    dpo = ex.runs_for(runs, method="DPO", noise=0.4)
    rob = ex.runs_for(runs, method="Robust-DPO", noise=0.4)
    print("=== noise=0.4 下 Robust-DPO vs DPO ===")
    print(summarize_comparison(rob, dpo, "Robust-DPO", "DPO"))
    print("\n=== noise=0.0 下 (预期不显著) ===")
    dpo0 = ex.runs_for(runs, method="DPO", noise=0.0)
    rob0 = ex.runs_for(runs, method="Robust-DPO", noise=0.0)
    print(summarize_comparison(rob0, dpo0, "Robust-DPO", "DPO"))
