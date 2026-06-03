"""⭐⭐⭐ 毕业 Capstone — 五线综合在同一道 GSM8K 题上对照.

5 个 model:
    1. Vanilla GPT-2 (base)
    2. LoRA 微调版 (lora-family L01 ckpt)
    3. Adapter 版 (adapter-tuning-family L01 ckpt)
    4. DPO 版 (dpo-family L01 ckpt)
    5. R1-Zero 版 (reasoning-r1 capstone-A ckpt)

同一道题 + 5 路径 + 5 响应对照 = 整个学习历程的成果可视化.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# 五种 mock response 演示（真实需加载 5 个 ckpt）
GSM8K_PROBLEM = """Janet has 16 eggs. She eats 3 and sells 6 for $2 each.
How much money does she make from selling eggs?"""

GROUND_TRUTH = "12"

MOCK_RESPONSES = {
    "1_vanilla": {
        "response": "16 eggs minus some equals 12",
        "explanation": "base 模型只是模糊给数字，无可靠推理",
    },
    "2_lora": {
        "response": "Janet eats 3 eggs. She has 16-3=13. She sells 6 at $2.\nAnswer: 12",
        "explanation": "LoRA SFT 后能格式化但推理浅",
    },
    "3_adapter": {
        "response": "Step 1: 16 - 3 = 13 eggs left.\nStep 2: She sells 6 at $2 = $12.\nAnswer: $12",
        "explanation": "Adapter 加层，结构化推理但同 LoRA",
    },
    "4_dpo": {
        "response": "To find Janet's earnings:\n1. Start: 16 eggs\n2. She eats 3, but eating doesn't matter for sales\n3. She sells 6 eggs at $2 each\n4. Earnings = 6 × $2 = $12\nAnswer: $12",
        "explanation": "DPO 对齐后回答更人类偏好（解释清晰）",
    },
    "5_r1_zero": {
        "response": """<think>
Let me analyze. Janet has 16 eggs. She eats 3 (these are gone, not sold).
She sells 6 of the remaining.
Wait, the question asks about money from selling. So I only need to count sales.
6 eggs × $2/egg = $12.
Let me verify: she has 16-3-6 = 7 eggs left over, sold 6, ate 3. Total 16. ✓
Money = 6 × 2 = 12.
</think>
<answer>#### 12</answer>""",
        "explanation": "R1-Zero 涌现自检 (Wait, verify) + 严格 format",
    },
}


def correctness(response: str, gt: str) -> bool:
    return gt in response


def print_capstone_comparison():
    print(f"五线综合 — 毕业作品\n{'=' * 70}")
    print(f"\n题目:\n  {GSM8K_PROBLEM}")
    print(f"\nGround truth: {GROUND_TRUTH}")
    print(f"\n{'─' * 70}")

    for key, data in MOCK_RESPONSES.items():
        ok = correctness(data["response"], GROUND_TRUTH)
        mark = "✓" if ok else "✗"
        print(f"\n[{key}]  {mark} correct")
        print(f"Response: {data['response'][:200]}{'...' if len(data['response']) > 200 else ''}")
        print(f"Explanation: {data['explanation']}")
        print(f"─" * 70)

    print("\n观察 (五线对照):")
    print("  1. vanilla:   无格式，无推理")
    print("  2. LoRA:      格式 OK，推理浅 (weight 扰动)")
    print("  3. Adapter:   类似 LoRA (structure 扰动)")
    print("  4. DPO:       人类偏好风格强 (distribution 改)")
    print("  5. R1-Zero:   长 CoT + aha + 严格 format (trajectory 改) ⭐")


def export_for_notebook():
    """返回 jupyter 友好的 dict 用于绘表."""
    return {
        "problem": GSM8K_PROBLEM,
        "ground_truth": GROUND_TRUTH,
        "results": [
            {
                "method": k,
                "response": v["response"],
                "correct": correctness(v["response"], GROUND_TRUTH),
                "response_len": len(v["response"].split()),
                "has_think": "<think>" in v["response"],
                "has_format": "<answer>" in v["response"],
            }
            for k, v in MOCK_RESPONSES.items()
        ],
    }


if __name__ == "__main__":
    print_capstone_comparison()
    print(f"\n{'=' * 70}")
    print("🎓 毕业作品完成 — 整个 RL+对齐+推理系列收官 🎓")
    print("\n88 方法 / 90 lecture / ~101h 学完")
    print("PEFT (Prompt/LoRA/Adapter) + RLHF + DPO + Process Reward + R1 + SOTA + 多模态/Agent")
    print(f"{'=' * 70}")
