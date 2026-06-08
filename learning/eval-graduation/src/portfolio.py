"""Capstone-3: Portfolio README generator.

Builds a comprehensive markdown report compiling:
- 25-topic timeline
- 5-ckpt comparison
- mini-HELM table
- mini-Arena ranking
- red-team matrix + defense effect
- decision tree (场景 → ckpt 选型)
- "What can I do" 画像
"""
from __future__ import annotations

from typing import Dict, List

from ckpt_zoo import CKPT_METADATA, load_all
from mini_helm import run_mini_helm, to_md as helm_md, ascii_radar
from mini_arena import run_capstone_arena, to_md as arena_md
from mini_red_team import run_red_team, to_md as redteam_md
from mini_defense import compare_defense, to_md as defense_md


MODULE_TIMELINE = [
    ("Module 1 PEFT", ["prompt-tuning", "lora", "adapter-tuning"]),
    ("Module 3 造大模型", ["data-curation", "transformer-deep", "moe-architecture",
                          "ssm-hybrid", "long-context", "scaling-infra",
                          "pretraining-recipe", "small-model-graduation"]),
    ("Module 4 改大模型 (RL)", ["rl-foundations", "rlhf-classic", "dpo-family",
                              "process-reward", "reasoning-r1", "rl-sota-2026",
                              "multimodal-agent"]),
    ("Module 5 用大模型", ["inference-engine-core", "sglang-radixattention",
                          "speculative-decoding", "quantization-deploy",
                          "distributed-inference", "production-serving",
                          "serving-graduation"]),
    ("Module 6 评测/安全", ["eval-foundations", "reasoning-eval",
                          "agent-code-eval", "llm-judge-arena",
                          "red-team-jailbreak", "safety-defense",
                          "eval-graduation"]),
]


DECISION_TREE = """
question: 我有这个场景，选哪个 ckpt？

- 知识 QA (轻量) -> lora 或 phi_tiny
- 推理 / 数学 -> r1_tiny (思考链) 或 phi_tiny (简洁)
- 安全敏感 (客服 / 政务) -> dpo + 4-layer defense
- 多模态 -> 需 VLM 训 ckpt (本系列未覆盖, 见 Module 4 multimodal-agent)
- 长 ctx (RAG) -> 需长 ctx 训 ckpt (见 Module 3 long-context)
- 紧凑端侧 -> phi_tiny (270M 强于 vanilla GPT-2 124M)
"""


WHAT_I_CAN_DO = """
**25 专题学完后你能做：**

1. **造模型**：从 0 开始预训 GPT-2 / Phi-tiny 风格小模型
2. **改模型**：LoRA / Adapter / DPO / R1-Zero 调任意 ckpt
3. **用模型**：mini-vLLM / SGLang / speculative / 量化 / TP-PP
4. **评模型**：MMLU / HumanEval / SWE-Bench / Arena / 红队评测
5. **守模型**：4 层防御 + Constitutional Classifiers + 监控

**画像** = 2026 年的 LLM 全栈工程师
- 训练侧: PEFT + RL + R1
- 部署侧: vLLM + quantization + cost engineering
- 安全侧: 红队 + 防御 + 合规
- 评测侧: bench x judge x Arena
"""


def gen_portfolio() -> str:
    helm = run_mini_helm()
    arena = run_capstone_arena()
    rt = run_red_team()
    defense = compare_defense()

    parts: List[str] = ["# 25-topic LLM Learning Portfolio",
                        "",
                        "> 2026-06-05 - Module 6 收官，全系列完结",
                        "",
                        "## 25 专题时间线", ""]
    n = 0
    for module, topics in MODULE_TIMELINE:
        parts.append(f"### {module}")
        for t in topics:
            n += 1
            parts.append(f"  {n:2d}. `{t}`")
        parts.append("")

    parts.extend(["## 5-ckpt 元数据表", "",
                  "| key | name | params | latency | 推理 | 安全 |",
                  "|---|---|---:|---:|---|---|"])
    for m in CKPT_METADATA:
        parts.append(f"| `{m.key}` | {m.name} | {m.params_M}M | {m.latency_ms}ms | "
                     f"{m.reasoning_quality} | {m.safety_level} |")
    parts.append("")

    parts.extend(["## Capstone-1: mini-HELM", "", helm_md(helm), ""])
    parts.extend(["## mini-HELM 雷达 (r1_tiny)", "", "```",
                  ascii_radar(helm["r1_tiny"]), "```", ""])
    parts.extend(["## Capstone-2A: mini-Arena", "", arena_md(arena), ""])
    parts.extend(["## Capstone-2B: 红队 ASR", "", redteam_md(rt), ""])
    parts.extend(["## Capstone-2C: 防御加 ASR 降低", "", defense_md(defense), ""])
    parts.extend(["## 选型决策树", "", "```", DECISION_TREE.strip(), "```", ""])
    parts.extend(["## 我能做什么", "", WHAT_I_CAN_DO.strip()])
    return "\n".join(parts)


def write_portfolio(out_path: str = "portfolio.md") -> str:
    md = gen_portfolio()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    return out_path


def _self_test() -> int:
    md = gen_portfolio()
    assert "25-topic" in md
    assert "Module 1 PEFT" in md
    assert "Capstone-1" in md
    assert "Capstone-2A" in md
    assert "Capstone-2B" in md
    assert "Capstone-2C" in md
    assert "选型决策树" in md
    assert "r1_tiny" in md and "phi_tiny" in md
    # Count topics enumerated (lines containing `topic-name` backticks under timeline)
    # Module 1+3+4+5 = 25 prior topics; Module 6 = 7 new (including this graduation) -> 32 total
    topic_lines = [line for line in md.split("\n")
                    if line.strip() and line.strip()[0].isdigit() and "`" in line
                    and line.strip().split(".", 1)[0].isdigit()]
    assert len(topic_lines) == 32, f"Expected 32 topics, got {len(topic_lines)}"
    return 0


if __name__ == "__main__":
    f = _self_test()
    print(f"portfolio.py self-test: {'OK' if f == 0 else f'FAILED ({f})'}")
    print(gen_portfolio()[:1500])
