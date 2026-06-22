"""
pitch_kit.py — 电梯演讲 (elevator pitch) 工具箱: 三档长度 + 时长估算 + 听众适配自检.

为什么需要它: 你会无数次被问「你研究啥的?」—— 在电梯里、招待会上、面试时。能不能用
**10 秒 / 1 分钟 / 5 分钟**三个版本流利回答, 直接影响别人对你的印象和合作机会。而且不同
听众 (外行亲戚 / 隔壁方向同行 / 你领域专家) 要用不同的术语密度。这个工具帮你:
  1. 按目标时长**估算字数预算** (中文演讲约 220 字/分钟)。
  2. **检查**你写的 pitch 是否超/欠时长。
  3. **术语密度自检**: 给外行的版本里有没有混进黑话。

纯 stdlib。
"""
from __future__ import annotations

import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CHARS_PER_MIN = 220   # 中文演讲速度经验值 (字/分钟; 偏慢更稳)

# 三档 pitch
LADDER = {
    "10s": (10 / 60, "一句话钩子: 我研究 X, 让 Y 变好。给完全陌生的人/电梯里"),
    "1min": (1.0, "问题→我做了什么→为什么重要。给招待会/networking"),
    "5min": (5.0, "完整故事: 背景→gap→方法→结果→意义。给面试/小组会"),
}

# 常见黑话 (给外行版时应避免/解释)
JARGON = ["DPO", "RLHF", "ablation", "baseline", "SOTA", "token", "logits",
          "梯度", "消融", "对齐", "偏好优化", "鲁棒", "loss", "fine-tune", "benchmark"]


def char_budget(level: str) -> tuple[int, int]:
    """返回某档的 (建议字数下限, 上限). 用时长 × 语速估."""
    minutes = LADDER[level][0]
    target = minutes * CHARS_PER_MIN
    return int(target * 0.75), int(target * 1.15)


def _char_count(text: str) -> int:
    """中文按字符计 (去掉空白/标点对计数影响小, 这里粗略去空白)。"""
    return len(re.sub(r"\s", "", text))


def check_pitch(text: str, level: str, audience: str = "peer") -> dict:
    """检查一段 pitch: 字数是否落在该档预算内 + (若 audience='layperson') 黑话密度."""
    n = _char_count(text)
    lo, hi = char_budget(level)
    fit = "偏短" if n < lo else "偏长" if n > hi else "合适"
    jargon_hits = [j for j in JARGON if j.lower() in text.lower()]
    issues = []
    if fit != "合适":
        mins = LADDER[level][0]
        issues.append(f"字数 {n} {fit} (该档建议 {lo}-{hi} 字 ≈ {mins*60:.0f}秒)")
    if audience == "layperson" and jargon_hits:
        issues.append(f"给外行却含黑话: {jargon_hits} —— 换成大白话或解释")
    return {"chars": n, "budget": (lo, hi), "fit": fit,
            "seconds_est": round(n / CHARS_PER_MIN * 60), "jargon": jargon_hits,
            "issues": issues, "ok": not issues}


def render_check(text: str, level: str, audience: str = "peer") -> str:
    r = check_pitch(text, level, audience)
    lines = [f"[{level} / 听众={audience}] {r['chars']} 字 ≈ {r['seconds_est']} 秒  ({r['fit']})"]
    if r["ok"]:
        lines.append("  ✅ 长度合适" + ("、无黑话" if audience == "layperson" else ""))
    else:
        for i in r["issues"]:
            lines.append(f"  ⚠ {i}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("各档字数预算:")
    for lv in LADDER:
        lo, hi = char_budget(lv)
        print(f"  {lv:<5} {lo}-{hi} 字  ({LADDER[lv][1][:20]}...)")

    print("\n--- 10秒钩子 (外行版, 无黑话) ---")
    hook = "我教 AI 在人给的反馈有噪声、不靠谱的时候, 还能学好怎么回答问题。"
    print(hook)
    print(render_check(hook, "10s", audience="layperson"))

    print("\n--- 10秒钩子 (混了黑话, 给外行就翻车) ---")
    bad = "我做带噪声标签下的 DPO 鲁棒性, 提升对齐 win-rate。"
    print(render_check(bad, "10s", audience="layperson"))
