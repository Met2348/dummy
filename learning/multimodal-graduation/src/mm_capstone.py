"""
mm_capstone.py — Module 10 收官: 装配完整 mini-VLM 流水线检查 + 多模态研究 gap 雷达.

两个功能:
  1. assemble_pipeline_check(): 验证 M10.1-10.6 的 src 组件都在、能串成一条完整流水线
     (视觉塔 → 连接器 → VLM 训练 → 视觉 token 化 → 时序模态 → 评测)。
  2. multimodal_gaps(): 用 critical-reading-gap (9.3) 的 6 类 gap 雷达 + 优先级公式
     (Importance×Tractability/Cost), 给「多模态/VLM」扫出候选研究题目, 供 Capstone 起 idea 卡。

把 M10 从「学了一堆技术」变成「能装配 + 能找研究 gap」—— PhD 研究入口。
纯 stdlib (gap 部分) + 跨专题 import 检查。
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# M10 各专题的 src 模块 (用于装配检查)
M10_TOPICS = [
    ("vision-encoders", "tiny_vit", "视觉塔 (ViT/CLIP/SigLIP)"),
    ("vl-fusion-architectures", "connectors", "VL 连接器 (投影/cross-attn/early)"),
    ("vlm-training-recipe", "mini_vlm", "VLM 训练 (两阶段配方)"),
    ("visual-tokenization-generation", "vq_tokenizer", "视觉 token 化 (VQ, 让画图)"),
    ("video-audio-omni", "temporal_tokens", "时序模态 (视频/音频)"),
    ("vlm-eval-hallucination", "vlm_eval", "VLM 评测 (POPE 幻觉)"),
]


def assemble_pipeline_check(learning_root: Path) -> list[dict]:
    """检查 M10 各专题 src 是否就位 (能装配成完整流水线). 返回每个专题的状态。"""
    out = []
    for topic, module, desc in M10_TOPICS:
        src = learning_root / topic / "src"
        present = (src / f"{module}.py").exists()
        out.append({"专题": topic, "模块": module, "作用": desc, "就位": present})
    return out


# 多模态研究 gap 雷达 (6 类, 接 9.3) + 优先级 (Importance×Tractability/Cost, 1-5 分)
MULTIMODAL_GAPS = [
    {"gap": "VLM 视觉幻觉的抗噪偏好优化 (用 DPO 教 VLM 不幻觉)",
     "类型": "方法", "I": 5, "T": 4, "C": 2,
     "复用": "dpo-family + 10.6 vlm_eval", "为什么友好": "你的 DPO 复现 + POPE 直接接上"},
    {"gap": "用哪一层视觉特征喂 LLM 最好 (跨任务系统消融)",
     "类型": "复现", "I": 3, "T": 5, "C": 1,
     "复用": "10.1 + 10.3 + 9.4", "为什么友好": "纯消融, 低成本, 你的实验设计强项"},
    {"gap": "跨模态迁移到底有多少 (any-to-any 的核心假设)",
     "类型": "假设", "I": 5, "T": 2, "C": 4,
     "复用": "10.4 + 10.5", "为什么友好": "重要但贵, 需大模型; 适合中期"},
    {"gap": "视觉 token 压缩 vs 性能的帕累托前沿 (resampler/帧采样)",
     "类型": "评测", "I": 4, "T": 4, "C": 2,
     "复用": "10.2 + 10.5 + long-context", "为什么友好": "接你 long-context, 工程友好"},
    {"gap": "VLM 评测的 prompt 敏感性有多严重 (分数有多少是 prompt 工程)",
     "类型": "复现", "I": 4, "T": 5, "C": 1,
     "复用": "10.6 + 9.3/9.4", "为什么友好": "复现类, 现成 VLM 就能做, 极低成本"},
    {"gap": "reasoning VLM: 视觉链式推理的忠实性 (CoT 在视觉上真吗)",
     "类型": "理论", "I": 5, "T": 3, "C": 3,
     "复用": "reasoning-r1 + 10.3 + M12 可解释性", "为什么友好": "接你推理复现 + 即将建的 M12"},
]


def score_gaps() -> list[dict]:
    """给每个 gap 算优先级 = I×T/C (9.3 公式), 降序。高分 = 重要×可做÷成本。"""
    scored = []
    for g in MULTIMODAL_GAPS:
        pri = round(g["I"] * g["T"] / g["C"], 1)
        scored.append({**g, "优先级": pri})
    scored.sort(key=lambda x: x["优先级"], reverse=True)
    return scored


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]   # learning/
    print("=== M10 流水线装配检查 ===")
    for s in assemble_pipeline_check(root):
        mark = "✅" if s["就位"] else "❌"
        print(f"  {mark} {s['专题']:32} {s['作用']}")
    print("\n=== 多模态研究 gap 雷达 (优先级 = 重要×可做÷成本, 9.3) ===")
    for g in score_gaps():
        print(f"  [{g['优先级']:>4}] [{g['类型']}] {g['gap']}")
        print(f"         复用: {g['复用']} | {g['为什么友好']}")
