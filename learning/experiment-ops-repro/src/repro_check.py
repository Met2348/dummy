"""
repro_check.py — 给一条实验记录打「可复现性体检」分, 缺哪个要素就指出来.

为什么需要它: 「可复现」不是一个含糊的美德, 而是一张**具体的清单** (NeurIPS/ML 社区有
正式的 reproducibility checklist)。一个结果能不能被半年后的你 (或别人) 重现, 取决于当时
有没有留下足够的信息。这个工具把那张清单**代码化**: 喂它一条 run 记录, 它逐项检查并打分。

体检的 6 个要素 (每个都是「复现失败」的常见单点故障):
  1. seed        —— 没固定随机种子, 结果每次都不同
  2. git_sha     —— 不知道是哪版代码跑的 (且代码有未提交改动 = -dirty 更糟)
  3. config      —— 超参没记全, 无法重建当时的设置
  4. env         —— 包版本/python 版本未记, 换环境结果漂移
  5. data_ref    —— 数据集版本/哈希未记, 数据一变结论就变
  6. metrics     —— 连结果都没结构化记录, 只在某张截图里

纯 stdlib。配合 exp_tracker 的记录格式, 也能审计任何 dict。
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 每个要素: (取值函数, 通过判据, 失败时的整改建议)
CHECKS = {
    "seed": ("种子已固定", "config 里没有 seed —— 加上并在代码里 seed everything"),
    "git_sha": ("代码版本已记 (git sha)", "没记 git sha —— 用 exp_tracker.init 自动盖章"),
    "config": ("超参配置完整", "config 为空/过简 —— 把所有超参显式写进 config"),
    "env": ("环境指纹已记", "没记 python/包版本 —— 记录 env 或附 lockfile"),
    "data_ref": ("数据版本已记", "没记数据集版本/哈希 —— 记下 data 名称+版本+哈希"),
    "metrics": ("结果已结构化记录", "metrics 为空 —— 别只截图, 把指标写进记录"),
}


def audit(record: dict) -> dict:
    """对一条记录逐项体检, 返回 {item: {ok, advice}} + 总分."""
    cfg = record.get("config", {}) or {}
    result = {}

    def present(key) -> bool:
        if key == "seed":
            return "seed" in cfg and cfg["seed"] is not None
        if key == "git_sha":
            sha = record.get("git_sha", "")
            return bool(sha) and sha != "no-git"   # 注意: -dirty 仍算"有记录但有警告"
        if key == "config":
            return len(cfg) >= 2
        if key == "env":
            return bool(record.get("env"))
        if key == "data_ref":
            return any(k in cfg for k in ("data", "dataset", "data_ref", "data_hash"))
        if key == "metrics":
            return bool(record.get("metrics"))
        return False

    for key, (label, advice) in CHECKS.items():
        ok = present(key)
        warn = ""
        if key == "git_sha" and ok and record.get("git_sha", "").endswith("-dirty"):
            warn = " ⚠ 代码有未提交改动 (-dirty), 复现性打折"
        result[key] = {"ok": ok, "label": label, "advice": advice, "warn": warn}

    score = sum(1 for v in result.values() if v["ok"])
    return {"items": result, "score": score, "total": len(CHECKS)}


def render(report: dict) -> str:
    lines = [f"可复现性体检: {report['score']}/{report['total']}\n"]
    for key, v in report["items"].items():
        mark = "✅" if v["ok"] else "❌"
        line = f"  {mark} {v['label']}{v['warn']}"
        if not v["ok"]:
            line += f"\n        → {v['advice']}"
        lines.append(line)
    s = report["score"]
    verdict = ("可复现性良好" if s == report["total"]
               else "基本可复现, 补齐缺项更稳" if s >= 4
               else "复现风险高: 半年后很可能重现不出来")
    lines.append(f"\n裁决: {verdict}")
    return "\n".join(lines)


if __name__ == "__main__":
    good = {"config": {"method": "DPO", "noise": 0.4, "seed": 3, "dataset": "hh-rlhf@v2"},
            "git_sha": "a1b2c3d", "env": {"python": "3.13"}, "metrics": {"win_rate": 0.40}}
    bad = {"config": {"method": "DPO"}, "git_sha": "no-git", "metrics": {}}
    print("=== 一条规范记录 ===")
    print(render(audit(good)))
    print("\n=== 一条糟糕记录 ===")
    print(render(audit(bad)))
