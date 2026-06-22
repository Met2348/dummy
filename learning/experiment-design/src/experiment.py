"""
experiment.py — 一个**确定性的模拟实验引擎**, 用来练实验设计 (而不真去烧 GPU).

为什么用模拟而不真训模型: 本专题教的是「实验设计」这门手艺 —— 怎么提假设、排消融、控
变量、报方差。这些技能与「具体跑什么模型」无关。用一个**可复现、秒级、零算力**的模拟器,
你能在一个下午里把「设计 → 跑 → 读结果」的循环走几十遍, 而真训模型一次要几小时。
模拟器内部埋了一个**真实的交互效应** (见下), 你的任务是用正确的实验设计**把它检测出来**。

运行示例 (贯穿整个 9.4): **Robust-DPO 在噪声偏好标签下是否真的更鲁棒?**
  - 自变量①: method ∈ {DPO, Robust-DPO}
  - 自变量②: noise (偏好标签噪声比例) ∈ {0.0, 0.2, 0.4}
  - 因变量: win_rate (对齐质量, 0~1)
  - 埋进去的真相 (你设计实验要去发现它):
      * 无噪声时 DPO ≈ Robust-DPO (Robust 的额外机制没用武之地)
      * 噪声越大, DPO 掉得越狠, 而 Robust-DPO 扛得住
      * → 这是一个 **interaction (交互效应)**: method 的好处依赖 noise 水平
  - 每次测量还有 **种子噪声** (seed variance), 逼你学会报方差、别被单种子骗。

纯 numpy, 完全确定性 (固定 seed → 固定结果)。
"""
from __future__ import annotations

import sys
from itertools import product

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

METHODS = ["DPO", "Robust-DPO"]
NOISES = [0.0, 0.2, 0.4]
SEED_STD = 0.018  # 种子间测量噪声的标准差 (现实里来自数据 shuffle / init / dropout 等)


def _true_win_rate(method: str, noise: float) -> float:
    """埋在模拟器里的"真相"(ground truth). 真实研究中你不知道它, 要靠实验估计.

    构造: 基线 0.62; 噪声对两个方法都有惩罚, 但 DPO 的惩罚斜率更陡 →
    Robust-DPO 的相对优势随 noise 增大 = 交互效应。
    """
    base = 0.62
    if method == "DPO":
        penalty = 0.55 * noise          # DPO: 噪声惩罚陡
        bonus = 0.0
    else:  # Robust-DPO
        penalty = 0.20 * noise          # Robust-DPO: 噪声惩罚缓 (这就是它的卖点)
        bonus = 0.005                   # 无噪声时几乎没差 (额外机制的微小开销/收益)
    return base + bonus - penalty


def _stable_seed(method: str, noise: float, seed: int) -> int:
    """跨进程确定性的 seed 派生. 注意: 不能用 Python 内置 hash(str) ——
    它带进程级随机盐 (PYTHONHASHSEED), 会让你**每次重跑 notebook 得到不同数字**,
    这正是本专题 L5 要警示的那类可复现陷阱。所以这里手工编码, 完全确定性。"""
    m = METHODS.index(method) if method in METHODS else sum(map(ord, method))
    nz = int(round(noise * 1000))
    return seed * 1_000_003 + m * 7919 + nz


def run_experiment(method: str, noise: float, seed: int) -> float:
    """跑一次实验, 返回带种子噪声的 win_rate. 确定性: 同 (method,noise,seed) → 同结果
    (跨进程、跨天都一致 —— 见 _stable_seed 为什么不用内置 hash)."""
    rng = np.random.default_rng(_stable_seed(method, noise, seed))
    measured = _true_win_rate(method, noise) + rng.normal(0, SEED_STD)
    return float(np.clip(measured, 0, 1))


def ablation_grid(methods=METHODS, noises=NOISES, seeds=range(5)) -> list[dict]:
    """跑一个完整的 factorial (全因子) 消融网格: 每个 (method × noise × seed) 组合各一次."""
    runs = []
    for method, noise, seed in product(methods, noises, list(seeds)):
        runs.append({
            "method": method,
            "noise": noise,
            "seed": seed,
            "win_rate": run_experiment(method, noise, seed),
        })
    return runs


def aggregate(runs: list[dict], by=("method", "noise")) -> list[dict]:
    """把多种子的 runs 按 by 分组, 算 mean / std / n. 这是"报方差"的最小动作."""
    groups: dict[tuple, list[float]] = {}
    for r in runs:
        key = tuple(r[k] for k in by)
        groups.setdefault(key, []).append(r["win_rate"])
    out = []
    for key, vals in sorted(groups.items()):
        arr = np.array(vals)
        row = dict(zip(by, key))
        row.update({"mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
                    "n": len(arr)})
        out.append(row)
    return out


def runs_for(runs: list[dict], **filt) -> list[float]:
    """取出满足条件的 win_rate 列表, 给统计检验用. 例: runs_for(runs, method='DPO', noise=0.4)."""
    return [r["win_rate"] for r in runs
            if all(r[k] == v for k, v in filt.items())]


if __name__ == "__main__":
    runs = ablation_grid(seeds=range(8))
    print(f"{'method':12} {'noise':>6} {'mean':>7} {'std':>7}  n")
    for row in aggregate(runs):
        print(f"{row['method']:12} {row['noise']:>6} {row['mean']:>7.3f} "
              f"{row['std']:>7.3f}  {row['n']}")
    print("\n注意: noise=0 时两法接近, noise=0.4 时 Robust-DPO 明显更高 → 这就是交互效应。")
