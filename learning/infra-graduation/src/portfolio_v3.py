"""Capstone-3 — 46-topic Portfolio v3 generator."""
from __future__ import annotations

import os


HEADER = """# 46-Topic LLM Learning Portfolio (v3)

> Generated 2026-06-06 · Module 8 收官 · 整个学习马拉松 + 硬件层完成 ⭐⭐⭐⭐⭐⭐⭐⭐
"""


TIMELINE = """## Section 1: 46-topic Timeline

### Module 1 PEFT (3 专题)
  1. `prompt-tuning-family` — soft prompt / P-tuning / IPT
  2. `lora-family` — LoRA / QLoRA / DoRA / LoRA+
  3. `adapter-tuning-family` — Pfeiffer / Houlsby / Parallel / Compacter

### Module 3 造大模型 (8 专题)
  4. `data-curation`
  5. `transformer-deep`
  6. `moe-architecture`
  7. `ssm-hybrid`
  8. `long-context`
  9. `scaling-infra`
 10. `pretraining-recipe`
 11. `small-model-graduation`

### Module 4 改大模型 (7 专题)
 12. `rl-foundations`
 13. `rlhf-classic`
 14. `dpo-family`
 15. `process-reward`
 16. `reasoning-r1`
 17. `rl-sota-2026`
 18. `multimodal-agent`

### Module 5 用大模型 (7 专题)
 19. `inference-engine-core`
 20. `sglang-radixattention`
 21. `speculative-decoding`
 22. `quantization-deploy`
 23. `distributed-inference`
 24. `production-serving`
 25. `serving-graduation`

### Module 6 评测/安全 (7 专题)
 26. `eval-foundations`
 27. `reasoning-eval`
 28. `agent-code-eval`
 29. `llm-judge-arena`
 30. `red-team-jailbreak`
 31. `safety-defense`
 32. `eval-graduation`

### Module 7 Agent 应用层 (7 专题)
 33. `agent-foundations`
 34. `rag-essential`
 35. `tool-use-mcp`
 36. `multi-agent-orchestration`
 37. `agent-memory-context`
 38. `agent-framework-stack`
 39. `agent-graduation`

### Module 8 Infra/硬件层 (7 专题) ⭐ NEW
 40. `gpu-architecture` — H100/B200/Tensor Core/HBM/NVLink/Roofline
 41. `cuda-essentials` — kernel/thread/warp/SMEM/coalesce/online softmax
 42. `kernel-engineering` — Triton/CUTLASS/FlashAttn/fused MLP
 43. `cluster-networking` — NVLink/IB/NCCL/SHARP/fat-tree
 44. `storage-dataops` — Lustre/dataloader/sharding/DCP-async ckpt
 45. `training-orchestration` — Slurm/Ray/Young's formula/elastic
 46. `infra-graduation` — Mini-cluster sim + topology selector + Portfolio v3 ⭐⭐⭐⭐⭐⭐⭐⭐
"""


CKPT_ZOO = """## Section 2: 6-ckpt Zoo + DRA + Cluster blueprints

| Asset | Source | Purpose |
|-------|--------|---------|
| `vanilla` | Module 3 baseline | GPT-2 base |
| `lora` | Module 1 PEFT | + LoRA |
| `dpo` | Module 4 alignment | + DPO |
| `r1_tiny` | Module 4 reasoning | + GRPO R1-Zero |
| `phi_tiny` | Module 3 pretrain | 270M from-scratch |
| `dra_v1` | Module 7 | Deep research agent |
| `cluster_blueprints` ⭐ | Module 8 | Mini-cluster sim recipes |
"""


CAPSTONES = """## Section 3: All Capstones (Modules 1-8)

| Capstone | Module | Output |
|----------|--------|--------|
| Phi-tiny 270M | M3 | from-scratch GPT-2 small |
| 五线综合 | M4 | 5 ckpt × 1 GSM8K |
| mini-vLLM serving | M5 | FastAPI + KV cache |
| mini-HELM | M6 | 5 ckpt × 4 dim |
| mini-Arena BT-Elo | M6 | 5 ckpt round-robin |
| Red-team ASR matrix | M6 | 3 attack × 5 ckpt |
| Portfolio v1 | M6 | 32-topic |
| DRA from scratch | M7 | planner+retriever+writer+verifier |
| τ-bench eval | M7 | 5 task × 5 dim |
| Portfolio v2 | M7 | 39-topic |
| Roofline zoo ⭐ | M8 | 10 op × 4 GPU |
| Online softmax ⭐ | M8 | 1-pass FlashAttn kernel |
| Attention HBM zoo ⭐ | M8 | 1025× HBM saved at 128k |
| Fabric zoo ⭐ | M8 | 17500× SHARP gain |
| 7-day ckpt economics ⭐ | M8 | DCP-async vs full |
| 24h Slurm sim ⭐ | M8 | 8 jobs × 512 GPU |
| **Mini-cluster simulator** ⭐⭐⭐ | M8 | 18 scenario time-to-train + TCO |
| **Topology selector** ⭐⭐⭐ | M8 | (model, budget, deadline) → blueprint |
| **Portfolio v3** | M8 | 46-topic ⭐⭐⭐⭐⭐⭐⭐⭐ |
"""


SELECTION_TREES = """## Section 4: Selection Trees (cumulative)

### GPU 选型 (M8 新) ⭐
- 推理预算敏感 → H100 + FP8
- 推理极致吞吐 → B200 + FP4 (4× FP8 throughput)
- 训练 < 70B → 64x H100 即可
- 训练 70-405B → 512-4096x H100/B200
- 训练 > 1T → 必须 B200 + GB200 NVL72 + IB XDR

### 互连 fabric 选型 (M8 新) ⭐
- 单节点 → NVSwitch
- 8-64 节点 → IB NDR + NCCL
- 256+ 节点 → IB XDR + SHARP (必须)
- 推理集群 → RoCEv2 性价比

### 存储选型 (M8 新) ⭐
- 训练数据 → Lustre 大集群 / GPFS HPC 老牌
- 流式 → WebDataset/Mosaic Streaming tar shard
- ckpt → DCP-async (PyTorch 2.4+) + 双 tier (Lustre 热 + S3 冷)

### 调度选型 (M8 新) ⭐
- 共享 HPC → Slurm (FIFO + backfill + fairshare)
- 专属云 / RLHF → Ray (actor + placement group)
- K8s 云原生 → Volcano gang scheduler
- 弹性 spot → torchrun --nnodes=N:M

### Bench / Inference engine / RL / Agent 选型 (M5-M7 之前已涵盖)
- 见 portfolio_v2 + module 5/6/7 各 selection tree
"""


SIX_PROFILES = """## Section 5: 6 大画像 v3

```
你已具备：

1. 造模型      — 从 0 训 GPT-2 / Phi-tiny (M3)
2. 改模型      — LoRA / DPO / R1-Zero (M1+M4)
3. 用模型      — vLLM / SGLang / 量化 / 分布式 (M5)
4. 评模型      — 25 bench / Arena / 红队 (M6)
5. 守模型      — 4 层防御 + Constitutional Cls (M6)
6. 造 agent   — ReAct / RAG / MCP / multi-agent / memory (M7)
7. 造 infra ⭐ — GPU / CUDA / kernel / fabric / Slurm (M8 NEW)

= 2026 年 LLM 全栈工程师 ID 卡 v3
```

> 6 大画像 → 7 大画像 ⭐：M8 把"硬件层"加入工具箱
"""


WHAT_I_CAN_DO = """## Section 6: What I Can Do (cover letter snippets v3)

- "I can train a 270M Phi-tiny from scratch (M3)."
- "I can replicate R1-Zero RL on GPT-2, observing aha moment (M4)."
- "I can serve LLM at scale with vLLM/SGLang + 4-bit quant (M5)."
- "I can run mini-HELM 4-dim eval + mini-Arena BT-Elo (M6)."
- "I can build a deep research agent with planner/retriever/writer/verifier (M7)."
- "I can do red-team + 4-layer defense including Constitutional Classifiers (M6)."
- "I can compare 6 agent frameworks and pick one with a decision tree (M7)."
- "I can roofline-analyze any LLM op against H100/B200, identifying memory vs compute bound (M8)." ⭐
- "I can write FlashAttn-style online softmax + tiled GEMM kernels in Triton (M8)." ⭐
- "I can size a training cluster for an N-param model under a budget + deadline (M8)." ⭐
- "I can compute Young's formula ckpt frequency for 1k+ GPU cluster (M8)." ⭐
"""


CAREER_PATHS = """## Section 7: Career Paths v3 (2025 SF salary band)

| Path | Salary | Key topics |
|------|-------:|-----------|
| **LLM Infra Engineer** ⭐ | $300k-$700k | M3 + M5 + **M8** |
| AI Application Engineer | $150k-$300k | M6 + M7 |
| ML Research Engineer | $300k-$1M+ | M3 + M4 |
| AI Safety Engineer | $200k-$500k | M6 |
| AI Product Manager | $150k-$400k | All |
| **GPU/CUDA Engineer** ⭐ NEW | $250k-$600k | **M8 deep** + M3 (kernel) |
| **HPC/Cluster Engineer** ⭐ NEW | $200k-$500k | **M8 sys** + M5 |

> M8 直接打开两条新职业路径（GPU 内核 + HPC 集群）
"""


def write_portfolio_v3(path: str) -> str:
    sections = [HEADER, TIMELINE, CKPT_ZOO, CAPSTONES, SELECTION_TREES,
                SIX_PROFILES, WHAT_I_CAN_DO, CAREER_PATHS]
    content = "\n".join(sections)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _self_test() -> None:
    p = "tmp_portfolio_v3.md"
    write_portfolio_v3(p)
    with open(p, encoding="utf-8") as f:
        content = f.read()

    n_topics = sum(
        1 for line in content.split("\n")
        if line.strip() and line.strip()[0].isdigit() and "`" in line
    )
    assert n_topics >= 46, n_topics

    for m in ["Module 1 PEFT", "Module 3 造大模型", "Module 4 改大模型",
              "Module 5 用大模型", "Module 6 评测/安全", "Module 7 Agent 应用层",
              "Module 8 Infra/硬件层"]:
        assert m in content, m

    assert "Portfolio v3" in content
    assert "infra-graduation" in content
    assert "7 大画像" in content
    assert "GPU/CUDA Engineer" in content
    os.remove(p)
    print(f"[OK] portfolio_v3 ({n_topics} topics enumerated)")


if __name__ == "__main__":
    _self_test()
    # Demonstrate the real Capstone-3 deliverable (a pushable portfolio_v3.md) without
    # littering the repo: write to the OS temp dir, not the current working directory.
    import sys
    import tempfile

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tmp_out = os.path.join(tempfile.gettempdir(), "infra_graduation_portfolio_v3.md")
    write_portfolio_v3(tmp_out)
    with open(tmp_out, encoding="utf-8") as f:
        preview = f.read()[:1500]
    print(preview)
    print(f"\n[full portfolio_v3.md written to {tmp_out}]")
