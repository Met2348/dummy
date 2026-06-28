"""
embodied_capstone.py — Module 11 capstone: 装配全栈 + mini-benchmark + 研究 gap 雷达.

M11 走完了具身全栈: 地基/tokens-as-actions(11.1) → VLA架构(11.2) → 扩散动作头(11.3)
→ 模仿学习/数据(11.4) → 世界模型/规划(11.5) → sim2real/DR(11.6)。本文件做 capstone:
  1. assembly_check(): 跨专题 import 全部 M11 src, 各跑最小烟测, 证明它们组合成一个栈。
  2. 研究 gap 雷达 + idea 卡: 把具身/VLA 的研究 gap 结构化 (接 M9 找 gap 框架)。

跨专题 src 复用 (同 M10/M13 capstone)。纯 toy CPU 确定性。
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LEARNING = Path(__file__).resolve().parents[2]   # learning/

# M11 全栈: (专题目录, src 模块, 一句话)
STACK = [
    ("embodied-foundations",        "toy_env",         "11.1 玩具控制环境 (M11 共享地基)"),
    ("embodied-foundations",        "action_serialize","11.1 tokens-as-actions 序列化"),
    ("vla-architectures",           "mini_vla",        "11.2 VLA = backbone + 动作头"),
    ("action-heads-diffusion-policy","diffusion_policy","11.3 扩散动作头 (多峰)"),
    ("robot-data-imitation",        "bc_train",        "11.4 行为克隆 + 数据 scaling"),
    ("world-action-models",         "world_model",     "11.5 世界模型 + MPC 规划"),
    ("sim2real-isaaclab",           "domain_rand",     "11.6 域随机化 (sim2real)"),
]


def add_paths():
    for topic, _, _ in STACK:
        p = LEARNING / topic / "src"
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def assembly_check() -> list:
    """跨专题 import + 最小烟测, 证明 M11 全栈组合。返回 [(label, ok, detail)]。"""
    add_paths()
    results = []
    has_torch = importlib_ok("torch")

    try:
        import toy_env as env
        sr = env.eval_policy(env.expert_action, n_episodes=50)
        results.append(("11.1 toy_env", sr > 0.95, f"专家成功率 {sr:.2f}"))
    except Exception as e:
        results.append(("11.1 toy_env", False, repr(e)))

    try:
        import action_serialize as ser
        import toy_env as env
        _, A = env.make_demos(n=20, seed=0)
        err = ser.roundtrip_error(A)
        results.append(("11.1 action_serialize", ser.N_ACTION_TOKENS == 9, f"{ser.N_ACTION_TOKENS}动作token, 离散损失{err:.2f}"))
    except Exception as e:
        results.append(("11.1 action_serialize", False, repr(e)))

    try:
        import mini_vla as vla
        m = vla.build_mini_vla(head="discrete")
        results.append(("11.2 mini_vla", m is not None or not has_torch, "VLA backbone+动作头 build"))
    except Exception as e:
        results.append(("11.2 mini_vla", False, repr(e)))

    try:
        import diffusion_policy as dp
        S, A = dp.make_obstacle_demos(n=40, chunk=1, seed=0)
        results.append(("11.3 diffusion_policy", S.shape[1] == 2, f"双峰绕障 demo {S.shape}"))
    except Exception as e:
        results.append(("11.3 diffusion_policy", False, repr(e)))

    try:
        import bc_train as bc
        m = bc.build_bc_policy()
        results.append(("11.4 bc_train", m is not None or not has_torch, "BC 策略 build"))
    except Exception as e:
        results.append(("11.4 bc_train", False, repr(e)))

    try:
        import world_model as wm
        S, A, D = wm.make_random_transitions(n=100, seed=0)
        results.append(("11.5 world_model", S.shape[0] == 100, f"随机转移 {S.shape}"))
    except Exception as e:
        results.append(("11.5 world_model", False, repr(e)))

    try:
        import domain_rand as dr
        S, A = dr.collect_demos(n=20, region="wide", seed=0)
        results.append(("11.6 domain_rand", len(S) > 0, "DR 配置随机化 demo"))
    except Exception as e:
        results.append(("11.6 domain_rand", False, repr(e)))

    return results


def importlib_ok(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


# ───────────────────────── 研究 gap 雷达 (接 M9) ─────────────────────────
GAPS = [
    dict(area="数据高效具身学习",
         gap="真机 demo 极贵 (M11.4); 怎么用少 demo / 人类视频 / 仿真泛化到新任务?",
         why_hard="embodiment gap (人≠机器人), 分布漂移, 标签缺失。",
         min_exp="toy 上比较: 少 demo BC vs 世界模型(随机数据) 的样本效率 (你 N1/M11.5-N2)。",
         connects="M11.4 数据 + M11.5 世界模型 + M11.1-L4 视频"),
    dict(area="多峰动作 + 实时 (扩散策略加速)",
         gap="扩散动作头多峰好 (M11.3) 但采样慢; 怎么少步高质量满足高频控制?",
         why_hard="少步采样 vs 多峰保真权衡; chunking 反应性代价 (M11.3-N2)。",
         min_exp="toy 绕障上对扩散动作头做 flow/reflow (M13.2) 少步采样, 看多峰是否保持。",
         connects="M11.3 扩散动作头 + M13.2 少步采样"),
    dict(area="VLM 知识迁移到控制 (co-train 防遗忘)",
         gap="VLA 微调易遗忘 VLM 知识 (M11.4-L3); 最优 机器人:VLM 数据配比?",
         why_hard="机器人数据少, 纯微调灾难性遗忘; 配比是关键超参。",
         min_exp="mini-VLA (M11.2) 上消融 co-train 比例, 看控制 vs 保留知识权衡。",
         connects="M11.2 VLA + M11.4 co-train + M10.3 冻结"),
    dict(area="世界模型物理正确性 (具身版)",
         gap="世界模型/视频模拟器物理不准 → 规划歪 (M11.5/M13.5); 怎么评/保证物理?",
         why_hard="物理合理性难形式化; 误差累积 (M11.5-N1)。",
         min_exp="toy 世界模型上测多步误差累积, 设计一个守恒量检验 (接 M13.5)。",
         connects="M11.5 世界模型 + M13.5 + 评估"),
    dict(area="sim2real 自动 DR 范围",
         gap="DR 范围太窄漏 gap、太宽训不动 (M11.6-L3); 怎么自动定范围?",
         why_hard="范围是关键超参; 需真机反馈校准 (系统辨识/ADR)。",
         min_exp="toy DR 上扫随机化范围 vs real 成功率, 找最优范围 (接 9.4 + M11.6-N1)。",
         connects="M11.6 DR + 9.4 实验设计"),
    dict(area="统一评测 (具身 benchmark)",
         gap="具身评测碎片化; LIBERO/CALVIN 等怎么标准化 + 抓真泛化 (非过拟合 benchmark)?",
         why_hard="真实任务多样, 成功率指标抓不全鲁棒性/泛化。",
         min_exp="toy 上建 mini-benchmark (多方法 × 多配置成功率表), 体会评测设计 (你 N1)。",
         connects="M11.7 capstone + 评估模块"),
]


def make_idea_card(gap: dict) -> str:
    return (f"┌─ idea 卡: {gap['area']} " + "─" * max(2, 36 - len(gap['area'])) + "\n"
            f"│ 问题 (gap): {gap['gap']}\n"
            f"│ 为什么难:   {gap['why_hard']}\n"
            f"│ 最小实验:   {gap['min_exp']}\n"
            f"│ 连接:       {gap['connects']}\n"
            f"└" + "─" * 50)


def gap_radar() -> str:
    return "\n".join(f"  [{i+1}] {g['area']}  ←  {g['connects']}" for i, g in enumerate(GAPS))


if __name__ == "__main__":
    print("== M11 全栈装配检查 ==")
    for label, ok, detail in assembly_check():
        print(f"  [{'OK ' if ok else 'FAIL'}] {label:24} {detail}")
    print(f"\n== 具身/VLA 研究 gap 雷达 ({len(GAPS)} 个) ==")
    print(gap_radar())
    print("\n== 示例 idea 卡 ==")
    print(make_idea_card(GAPS[0]))
