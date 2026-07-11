"""Capstone-3 — 39-topic Portfolio v2 generator."""
from __future__ import annotations

import os


HEADER = """# 39-Topic LLM Learning Portfolio (v2)

> Generated 2026-06-05 · Module 7 收官 · 整个学习马拉松完成 ⭐⭐⭐⭐⭐⭐⭐
"""


TIMELINE = """## Section 1: 39-topic Timeline

### Module 1 PEFT (3 专题)
  1. `prompt-tuning-family` — soft prompt / P-tuning / IPT
  2. `lora-family` — LoRA / QLoRA / DoRA / LoRA+
  3. `adapter-tuning-family` — Pfeiffer / Houlsby / Parallel / Compacter

### Module 3 造大模型 (8 专题)
  4. `data-curation` — RefinedWeb / FineWeb / DCLM
  5. `transformer-deep` — Attention variants / FlashAttn / RoPE / RMSNorm
  6. `moe-architecture` — Mixtral / DeepSeek-V3 / GShard
  7. `ssm-hybrid` — Mamba / RWKV / Jamba
  8. `long-context` — YaRN / NTK / position interpolation
  9. `scaling-infra` — Megatron / DeepSpeed / FSDP
 10. `pretraining-recipe` — Phi-3 / TinyLlama recipe
 11. `small-model-graduation` — 270M Phi-tiny from scratch

### Module 4 改大模型 (7 专题)
 12. `rl-foundations` — PPO / CartPole / GAE
 13. `rlhf-classic` — InstructGPT 三段管线
 14. `dpo-family` — DPO / KTO / ORPO / SimPO / CPO + RainbowPO
 15. `process-reward` — PRM / Math-Shepherd / PRIME / MCTS
 16. `reasoning-r1` — GRPO + R1-Zero 双轨复现
 17. `rl-sota-2026` — DAPO / VAPO / GenRM / Skywork-RM
 18. `multimodal-agent` — VLM-R1 / WebRL / SWE-Gym / 五线综合

### Module 5 用大模型 (7 专题)
 19. `inference-engine-core` — vLLM PagedAttention
 20. `sglang-radixattention` — SGLang RadixTree
 21. `speculative-decoding` — Medusa / Eagle / draft model
 22. `quantization-deploy` — GPTQ / AWQ / FP8 / INT4
 23. `distributed-inference` — TP / PP / EP
 24. `production-serving` — autoscale / cost / monitoring
 25. `serving-graduation` — mini-vLLM serving

### Module 6 评测/安全 (7 专题)
 26. `eval-foundations` — MMLU / HELM / BBH / TruthfulQA
 27. `reasoning-eval` — GSM8K / MATH / AIME / GPQA
 28. `agent-code-eval` — HumanEval / SWE-Bench / WebArena / OSWorld
 29. `llm-judge-arena` — MT-Bench / Arena-Hard / BT-Elo
 30. `red-team-jailbreak` — GCG / PAIR / AutoDAN / 5 攻击
 31. `safety-defense` — Llama Guard / Constitutional Cls
 32. `eval-graduation` — mini-HELM / mini-Arena / Portfolio v1

### Module 7 Agent 应用层 (7 专题)
 33. `agent-foundations` — ReAct / Reflexion / Plan-Execute / Router / StateGraph
 34. `rag-essential` — naive / hybrid / rerank / HyDE / GraphRAG / HippoRAG / RAGAS
 35. `tool-use-mcp` — MCP / A2A / Computer Use / Sandbox
 36. `multi-agent-orchestration` — AutoGen / CrewAI / LangGraph / Magentic-One
 37. `agent-memory-context` — Letta / Mem0 / episodic / semantic / prompt cache
 38. `agent-framework-stack` — LangChain / LangGraph / LlamaIndex / Pydantic AI / Vercel / Claude SDK
 39. `agent-graduation` — DRA + τ-bench + Portfolio v2 ⭐⭐⭐⭐⭐⭐⭐
"""


CKPT_ZOO = """## Section 2: 6-ckpt Zoo + DRA

| Ckpt | Source | Purpose |
|------|--------|---------|
| `vanilla` | Module 3 baseline | GPT-2 base, no fine-tune |
| `lora` | Module 1 PEFT | + LoRA on instruction data |
| `dpo` | Module 4 alignment | + DPO on Anthropic-HH |
| `r1_tiny` | Module 4 reasoning | + R1-Zero style GRPO |
| `phi_tiny` | Module 3 pretrain | 270M from-scratch |
| `dra_v1` | Module 7 (this) | Deep research agent |
"""


CAPSTONES = """## Section 3: All Capstones

| Capstone | Module | Output |
|----------|--------|--------|
| Phi-tiny 270M | Module 3 | from-scratch GPT-2 small |
| 五线综合 (5-line) | Module 4 | 5 ckpt × 1 GSM8K 题对照 |
| mini-vLLM serving | Module 5 | FastAPI + KV cache + batch |
| mini-HELM (4 dim) | Module 6 | 5 ckpt × 4 维评分 |
| mini-Arena (BT-Elo) | Module 6 | 5 ckpt round-robin |
| 红队 ASR matrix | Module 6 | 3 attack × 5 ckpt |
| 防御 4 层 pipeline | Module 6 | Llama Guard mock + Constitutional |
| Portfolio v1 | Module 6 | 32-topic ID 卡 |
| **DRA from scratch** | Module 7 | planner + retriever + writer + verifier ⭐ |
| **τ-bench eval pack** | Module 7 | 5 task × 5 dim 评分 |
| **Portfolio v2** | Module 7 | 39-topic ID 卡 v2 ⭐⭐⭐⭐⭐⭐⭐ |
"""


SELECTION_TREES = """## Section 4: Selection Trees

### Bench 选型 (from Module 6)
- 通识 → MMLU-Pro / HELM
- 推理 → AIME / MATH / GPQA
- Code → SWE-Bench / HumanEval / LiveCodeBench
- Agent → WebArena / GAIA / OSWorld / τ-bench
- 多模态 → MMMU / MathVista

### Inference engine 选型 (from Module 5)
- 多 vendor 通用 → vLLM
- 多 turn prefix share → SGLang
- Edge / 极低延迟 → TensorRT-LLM
- 训练嵌入 → DeepSpeed-MII

### RL 算法选型 (from Module 4)
- 标准 RLHF → PPO
- 无需 reward model → DPO / SimPO
- 推理任务 → GRPO / DAPO
- 多 reward → Safe-RLHF

### Agent framework 选型 (from Module 7) ⭐ NEW
- Quick PoC → CrewAI / Vercel AI SDK
- Complex state → LangGraph
- RAG heavy → LlamaIndex
- Type-safe → Pydantic AI
- Anthropic stack → Claude Agent SDK
- C# / .NET → Semantic Kernel
"""


SIX_PROFILES = """## Section 5: 6 大画像

```
你已具备：

1. 造模型 — 从 0 训 GPT-2 / Phi-tiny (Module 3)
2. 改模型 — LoRA / Adapter / DPO / R1-Zero (Modules 1 + 4)
3. 用模型 — vLLM / SGLang / 量化 / 分布式 (Module 5)
4. 评模型 — 25 bench × judge × Arena (Module 6)
5. 守模型 — 红队 + 4 层防御 + Constitutional Cls (Module 6)
6. 造 agent 产品 — ReAct / RAG / MCP / multi-agent / memory (Module 7) ⭐ NEW

= 2026 年 LLM 全栈工程师 ID 卡 v2
```
"""


WHAT_I_CAN_DO = """## Section 6: What I Can Do (cover letter snippets)

- "I can train a 270M Phi-tiny from scratch (Module 3 small-model-graduation)."
- "I can replicate R1-Zero style RL on GPT-2-M, observing aha moment (Module 4 reasoning-r1)."
- "I can serve LLM at scale with vLLM/SGLang + 4-bit quant (Module 5 serving-graduation)."
- "I can run mini-HELM 4-dim eval + mini-Arena BT-Elo (Module 6 eval-graduation)."
- "I can build a deep research agent with planner/retriever/writer/verifier (Module 7 graduation)."
- "I can do red-team + 4-layer defense including Constitutional Classifiers (Module 6 safety)."
- "I can compare 6 agent frameworks and pick one with a decision tree (Module 7 framework)."
"""


CAREER_PATHS = """## Section 7: 5 Career Paths

| Path | Salary (2025 SF) | Key topics |
|------|-----------------:|------------|
| LLM Infra Engineer | $250k-$500k | Modules 3 + 5 |
| AI Application Engineer | $150k-$300k | Modules 6 + 7 |
| ML Research Engineer | $300k-$1M+ | Modules 3 + 4 |
| AI Safety Engineer | $200k-$500k | Module 6 |
| AI Product Manager | $150k-$400k | All modules |
"""


def write_portfolio_v2(path: str) -> str:
    sections = [
        HEADER,
        TIMELINE,
        CKPT_ZOO,
        CAPSTONES,
        SELECTION_TREES,
        SIX_PROFILES,
        WHAT_I_CAN_DO,
        CAREER_PATHS,
    ]
    content = "\n".join(sections)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _self_test() -> None:
    out_path = "tmp_portfolio_v2.md"
    actual_path = write_portfolio_v2(out_path)
    assert actual_path == out_path

    with open(out_path, encoding="utf-8") as f:
        content = f.read()

    n_topics_listed = sum(
        1 for line in content.split("\n")
        if line.strip() and line.strip()[0].isdigit() and "`" in line
    )
    assert n_topics_listed >= 39, n_topics_listed

    assert "Module 1 PEFT" in content
    assert "Module 3 造大模型" in content
    assert "Module 4 改大模型" in content
    assert "Module 5 用大模型" in content
    assert "Module 6 评测/安全" in content
    assert "Module 7 Agent 应用层" in content

    assert "agent-graduation" in content
    assert "Portfolio v2" in content
    assert "6 大画像" in content
    assert "5 Career Paths" in content

    os.remove(out_path)
    print(f"[OK] portfolio_v2._self_test passed ({n_topics_listed} topics enumerated)")


if __name__ == "__main__":
    _self_test()
    # Demonstrate the real Capstone-3 deliverable (a pushable portfolio_v2.md) without
    # littering the repo: write to the OS temp dir, not the current working directory.
    import tempfile

    tmp_out = os.path.join(tempfile.gettempdir(), "agent_graduation_portfolio_v2.md")
    write_portfolio_v2(tmp_out)
    with open(tmp_out, encoding="utf-8") as f:
        preview = f.read()[:1500]
    print(preview)
    print(f"\n[full portfolio_v2.md written to {tmp_out}]")
