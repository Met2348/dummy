"""
team_health_check.py — 团队健康度诊断: 识别团队功能失调的早期信号, 而不是等到有人
离职才发现问题。

四个信号 (对应 L5):
  psychological_safety 心理安全感 —— 成员敢不敢说"我不知道"/"我错了"
  workload_balance      工作量分配均衡
  conflict_visibility   冲突是否被公开处理还是被回避/憋着
  growth_visibility     成员能否看到自己的成长路径

纯 stdlib。
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SIGNALS = [
    ("psychological_safety", "心理安全感", "成员敢在组会上说'我不知道'或承认失败吗?"),
    ("workload_balance", "工作量分配均衡", "是否有人长期超负荷而有人明显清闲?"),
    ("conflict_visibility", "冲突是否被公开处理", "上次意见分歧是摊开谈了还是不了了之?"),
    ("growth_visibility", "成长路径可见度", "成员知道自己半年后会成长成什么样子吗?"),
]


def blank_checkup() -> dict:
    return {key: {"score": 0, "note": ""} for key, _, _ in SIGNALS}


def diagnose(checkup: dict) -> dict:
    risks = []
    for key, name, hint in SIGNALS:
        d = checkup.get(key, {})
        score = d.get("score", 0)
        if score and score <= 2:
            risks.append(f"「{name}」偏低({score}分) —— {hint}")
    return {"risks": risks, "healthy": not risks}


def render(checkup: dict) -> str:
    lines = ["=== 团队健康度诊断 ==="]
    for key, name, hint in SIGNALS:
        d = checkup.get(key, {"score": 0, "note": ""})
        lines.append(f"{name}: {d['score']}分 —— {d['note'] or '(未填)'}")
    diag = diagnose(checkup)
    lines.append("\n" + ("✅ 暂无明显风险信号" if diag["healthy"]
                          else "⚠ 风险信号:\n  " + "\n  ".join(diag["risks"])))
    return "\n".join(lines)


if __name__ == "__main__":
    c = blank_checkup()
    c["psychological_safety"] = {"score": 2, "note": "组会上很少有人主动说自己实验失败了"}
    c["workload_balance"] = {"score": 4, "note": "目前3个人任务量分布均衡"}
    c["conflict_visibility"] = {"score": 2, "note": "上次authorship意见不合, 私下嘀咕但没摊开谈"}
    c["growth_visibility"] = {"score": 3, "note": "有semi-annual review但反馈比较笼统"}
    print(render(c))
