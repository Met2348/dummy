"""
generative_capstone.py — Module 13 capstone: 装配全链 + 研究 gap 雷达.

M13 走完了生成式媒体全链: 扩散(13.1)→流匹配(13.2)→DiT/CFG/latent(13.3)→视频(13.4)
→世界模型(13.5)→dLLM(13.6)。本文件做两件 capstone 的事:
  1. assembly_check(): 跨专题 import 全部 6 个 src, 各跑一个最小烟测, 证明它们组合成一个栈
     —— 你真的能驱动每一种生成方法 (像 M10.7 的装配检查)。
  2. gap 雷达 + idea 卡: 把 M13 的研究 gap 结构化, 用 M9 的找 gap 框架产出 idea 卡。

跨专题 src 复用: 把 6 个兄弟专题的 src 目录加进 sys.path (同 M10 的做法)。
纯 toy CPU 确定性。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LEARNING = Path(__file__).resolve().parents[2]   # learning/

# M13 全链: (专题目录, src 模块名, 一句话)
STACK = [
    ("diffusion-foundations",      "diffusion",      "13.1 扩散地基 (前向加噪/反向去噪)"),
    ("flow-matching-sota",         "flow_matching",  "13.2 流匹配 (直路径/少步采样)"),
    ("dit-latent-diffusion",       "dit",            "13.3 DiT+CFG (transformer 去噪+条件控制)"),
    ("video-generation",           "video_diffusion","13.4 视频 (时空扩散/时序连贯)"),
    ("world-models",               "world_model",    "13.5 世界模型 (动作条件预测/想象)"),
    ("diffusion-language-models",  "diffusion_lm",   "13.6 dLLM (文本扩散/双向并行解码)"),
]


def add_paths():
    """把 6 个兄弟专题的 src 加入 sys.path (跨专题复用)。"""
    for topic, _, _ in STACK:
        p = LEARNING / topic / "src"
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def assembly_check() -> list:
    """跨专题 import + 最小烟测, 证明全链组合。返回 [(topic, ok, detail)]。"""
    add_paths()
    results = []
    try:
        import torch  # noqa
        has_torch = True
    except Exception:
        has_torch = False

    # 13.1 diffusion: 数据 + 去噪器 build
    try:
        import diffusion as d
        x = d.make_two_moons(64, seed=0); m = d.build_denoiser(); ok = x.shape == (64, 2) and (m is not None or not has_torch)
        results.append(("13.1 diffusion", ok, f"two_moons {x.shape} + 去噪器 build"))
    except Exception as e:
        results.append(("13.1 diffusion", False, repr(e)))

    # 13.2 flow_matching: 速度场 build
    try:
        import flow_matching as fm
        x = fm.make_two_moons(64, seed=0); m = fm.build_velocity_field()
        results.append(("13.2 flow_matching", x.shape == (64, 2), f"velocity field build, data {x.shape}"))
    except Exception as e:
        results.append(("13.2 flow_matching", False, repr(e)))

    # 13.3 dit: 类数据 + DiT build
    try:
        import dit
        x, y = dit.make_class_blobs(n_per=30, seed=0); m = dit.build_dit()
        results.append(("13.3 dit", x.shape[0] == 120 and set(y.tolist()) == {0,1,2,3}, f"4类blobs {x.shape} + DiT build"))
    except Exception as e:
        results.append(("13.3 dit", False, repr(e)))

    # 13.4 video: 轨迹 + 连贯度量
    try:
        import video_diffusion as vd
        v = vd.make_trajectories(20, seed=0); c = vd.temporal_coherence(v)
        results.append(("13.4 video", v.shape == (20, vd.T_FRAMES, 2) and c > 0, f"轨迹 {v.shape}, 连贯 {c:.3f}"))
    except Exception as e:
        results.append(("13.4 video", False, repr(e)))

    # 13.5 world model: 转移 + build
    try:
        import world_model as wm
        data = wm.make_transitions(100, seed=0); m = wm.build_world_model()
        results.append(("13.5 world_model", data[0].shape == (100, 2), f"转移 {data[0].shape} + WM build"))
    except Exception as e:
        results.append(("13.5 world_model", False, repr(e)))

    # 13.6 dLLM: 回文 + dLLM build
    try:
        import diffusion_lm as dl
        s = dl.make_sequences(50, seed=0); m = dl.build_dlm()
        results.append(("13.6 dLLM", s.shape == (50, dl.L) and dl.is_palindrome(s) == 1.0, f"回文 {s.shape}"))
    except Exception as e:
        results.append(("13.6 dLLM", False, repr(e)))

    return results


# ───────────────────────── 研究 gap 雷达 (接 M9) ─────────────────────────
GAPS = [
    dict(area="少步高质量 dLLM 解码",
         gap="dLLM 并行解码轮数少则质量掉 (M13.6-N1); 能否像扩散一致性模型 (M13.2) 那样几轮高质量?",
         why_hard="文本离散, 一致性蒸馏/直路径思想怎么迁到 masked diffusion 不平凡。",
         min_exp="在玩具 dLLM 上试: 用一个'教师多轮解码'蒸馏'学生少轮解码', 看少轮质量能否追上。",
         connects="M13.2 一致性模型 + M13.6 dLLM"),
    dict(area="视频长程一致 / 持久记忆",
         gap="扩散续帧无显式记忆, 几百帧后场景漂移 (M13.4-L3); 怎么给视频/世界模型加持久状态?",
         why_hard="注意力范围 ≠ 视频长度; 显式记忆与扩散框架怎么结合是 open。",
         min_exp="玩具世界模型 (M13.5) 加一个显式状态向量, 看长程 rollout 一致性是否改善。",
         connects="M13.4 视频 + M13.5 世界模型"),
    dict(area="世界模型物理正确性评测",
         gap="没有好指标判断世界模型/视频'懂物理 vs 拟合像素' (M13.5-L4)。",
         why_hard="物理合理性难形式化; 现有指标 (FID/帧间一致) 抓不到。",
         min_exp="设计一个'守恒量'玩具检验 (如总动量), 看世界模型 rollout 是否守恒。",
         connects="M13.5 世界模型 + 评估模块"),
    dict(area="dLLM 的对齐迁移 (RLHF→dLLM)",
         gap="AR 的对齐方法 (RLHF/DPO, 你的 RL 模块) 怎么迁到 dLLM 的并行解码范式?",
         why_hard="dLLM 无逐 token 似然分解, 策略梯度/偏好优化的对应物不显然。",
         min_exp="在玩具 dLLM 上定义一个偏好信号 (如更短/含某 token), 试一步 DPO 式更新。",
         connects="M13.6 dLLM + RL/DPO 模块"),
    dict(area="统一生成 (跨模态一个扩散模型)",
         gap="图/视频/文本现在各训各的; 能否一个扩散模型统一处理 (时空 patch + 离散 token 统一表示)?",
         why_hard="连续 (图/视频) 与离散 (文本) 扩散的统一表示/目标不平凡。",
         min_exp="玩具: 把 2D 点 (连续) 和回文 (离散) 喂同一个去噪 transformer, 看能否都学。",
         connects="M13.3 DiT + M13.6 dLLM + M10.5 时空 token"),
    dict(area="实时视频 (少步 + 蒸馏)",
         gap="视频推理极贵 (M13.4-L4); 一致性模型/蒸馏 (M13.2) 用到视频能否实时?",
         why_hard="视频 token 多 + 时序一致约束, 少步更易崩。",
         min_exp="玩具视频扩散 (M13.4) 上做 reflow (M13.2), 看少步采样质量保持。",
         connects="M13.2 少步采样 + M13.4 视频"),
    dict(area="世界模型 × 具身 sim2real",
         gap="世界模型当机器人模拟器, 怎么跨越想象与真实的差距 (误差累积 M13.5-N2)?",
         why_hard="多步误差累积 + 真实物理复杂; sim2real gap 大。",
         min_exp="玩具世界模型加噪训练 (域随机化), 看对'真'环境 rollout 的鲁棒性。",
         connects="M13.5 世界模型 + M11 具身 sim2real"),
]


def make_idea_card(gap: dict) -> str:
    """把一个 gap 格式化成 idea 卡 (接 M9 找 gap 框架)。"""
    return (f"┌─ idea 卡: {gap['area']} " + "─" * max(2, 40 - len(gap['area'])) + "\n"
            f"│ 问题 (gap): {gap['gap']}\n"
            f"│ 为什么难:   {gap['why_hard']}\n"
            f"│ 最小实验:   {gap['min_exp']}\n"
            f"│ 连接:       {gap['connects']}\n"
            f"└" + "─" * 50)


def gap_radar() -> str:
    return "\n".join(f"  [{i+1}] {g['area']}  ←  {g['connects']}" for i, g in enumerate(GAPS))


if __name__ == "__main__":
    print("== M13 全链装配检查 ==")
    for topic, ok, detail in assembly_check():
        print(f"  [{'OK ' if ok else 'FAIL'}] {topic:22} {detail}")
    print(f"\n== 研究 gap 雷达 ({len(GAPS)} 个) ==")
    print(gap_radar())
    print("\n== 示例 idea 卡 ==")
    print(make_idea_card(GAPS[0]))
