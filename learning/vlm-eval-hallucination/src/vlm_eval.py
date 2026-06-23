"""
vlm_eval.py — VLM 评测 + 视觉幻觉探测 (POPE 式), 把评测严谨性扩到多模态.

为什么需要它 (M10.6): VLM 会读会画了 (M10.1-10.5), 但它**说得对吗**? VLM 有一个独特病:
**视觉幻觉 (visual hallucination)** —— 图里根本没有的东西, 它却说有。原因常是「语言先验太强」
(M10.3-L4 模态坍缩的近亲): 模型靠常识猜 (「图里通常有人」) 而非真看图。

怎么测幻觉? **POPE (Polling-based Object Probing)** 的思想: 问一堆「图里有 X 吗?」的是非题,
一半 X 真的在图里 (正), 一半不在 (负)。如果模型对不在的 X 也大量说「有」, 就是在幻觉
(yes-bias)。本文件实现这套评测方法 + 指标 (准确率/yes率/精确率/召回/F1)。

为离线可跑, 这里用一个**可调幻觉率的模拟 VLM** (真实使用时换成真 VLM 的 yes/no 输出)。
模拟让你专注学**评测方法**, 而非依赖一个真模型。纯 numpy/stdlib。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def make_probe_set(n: int = 200, seed: int = 0) -> list[dict]:
    """造一组 POPE 式探测题: 每题 (图里有没有某物体, 真相). 正负各半 (平衡)。"""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        present = bool(i % 2)   # 一半物体真在图里, 一半不在 (平衡正负)
        items.append({"object": f"obj_{rng.integers(0, 50)}", "truly_present": present})
    return items


def simulated_vlm_answer(item: dict, hallucination: float = 0.3,
                         miss_rate: float = 0.1, rng=None) -> bool:
    """模拟 VLM 对「图里有 X 吗」的 yes/no 回答, 带可调幻觉率.
    - 物体真在 (present): 以 1-miss_rate 概率正确说 yes (miss_rate = 漏检)。
    - 物体不在: 以 hallucination 概率错误说 yes (这就是幻觉/yes-bias)。
    hallucination 越高, 模型越爱「无中生有」。"""
    rng = rng or np.random.default_rng(0)
    if item["truly_present"]:
        return rng.random() > miss_rate         # 大概率说有
    else:
        return rng.random() < hallucination     # 不在却说有 = 幻觉


def evaluate(items: list[dict], answers: list[bool]) -> dict:
    """POPE 式评测指标. answers[i] = 模型对 items[i] 的 yes(True)/no(False)。
    返回 accuracy / yes_rate / precision / recall / f1 + 混淆矩阵。"""
    y_true = np.array([it["truly_present"] for it in items])
    y_pred = np.array(answers)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())   # 幻觉: 说有但没有
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / len(items)
    yes_rate = (y_pred == 1).mean()                   # 模型说「有」的比例
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": round(acc, 3), "yes_rate": round(float(yes_rate), 3),
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "confusion": {"tp": tp, "fp_幻觉": fp, "tn": tn, "fn_漏检": fn}}


def run_pope(hallucination: float, n: int = 200, seed: int = 0) -> dict:
    """对一个给定幻觉率的模拟 VLM 跑完整 POPE 评测。"""
    rng = np.random.default_rng(seed)
    items = make_probe_set(n, seed=seed)
    answers = [simulated_vlm_answer(it, hallucination=hallucination, rng=rng) for it in items]
    return evaluate(items, answers)


def yes_bias_signal(report: dict) -> str:
    """从 yes_rate 读幻觉信号: 远高于 0.5 (真相一半一半) = 模型有 yes-bias = 在幻觉。"""
    yr = report["yes_rate"]
    if yr > 0.65:
        return f"⚠ yes_rate={yr} >> 0.5: 强 yes-bias, 模型在大量幻觉 (说'有'但实无)"
    if yr > 0.55:
        return f"yes_rate={yr}: 轻微 yes-bias, 有幻觉倾向"
    return f"yes_rate={yr} ≈ 0.5: 无明显 yes-bias (真相正负各半)"


if __name__ == "__main__":
    print("幻觉率 → POPE 指标 (真相正负各半, 理想 yes_rate=0.5):")
    print(f"{'幻觉率':>6} {'准确率':>7} {'yes率':>7} {'F1':>6}  信号")
    for h in [0.0, 0.2, 0.4, 0.6, 0.8]:
        r = run_pope(hallucination=h, n=400, seed=1)
        print(f"{h:>6} {r['accuracy']:>7} {r['yes_rate']:>7} {r['f1']:>6}  {yes_bias_signal(r)}")
    print("\n→ 幻觉率越高, yes率越偏离 0.5, 准确率越低。POPE 用 yes-bias 量化视觉幻觉。")
