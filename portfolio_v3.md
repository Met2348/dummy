# 48-Topic LLM Learning Portfolio (v3.1)

> Generated 2026-06-06 · Module 8 收官 ⭐⭐⭐⭐⭐⭐⭐⭐
> Extended 2026-06-17 · Module 7 +2 (agent-design-patterns / agent-harness-design) → 48 专题
> Extended 2026-06-17 · **Module 9「科研技能」开张** ⭐ — 首专题 `critical-reading-gap` 完成 (工程 → 研究, 第二条腿); 详见文末

## Section 1: 48-topic Timeline

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

### Module 7 Agent 应用层 (9 专题)
 33. `agent-foundations`
 34. `rag-essential`
 35. `tool-use-mcp`
 36. `multi-agent-orchestration`
 37. `agent-memory-context`
 38. `agent-framework-stack`
 39. `agent-design-patterns` ⭐ NEW — 5 workflow 模式 + workflow-vs-agent + context eng
 40. `agent-harness-design` ⭐ NEW — agentic loop + tool/context/permission/trace 引擎
 41. `agent-graduation`

### Module 8 Infra/硬件层 (7 专题)
 42. `gpu-architecture` — H100/B200/Tensor Core/HBM/NVLink/Roofline
 43. `cuda-essentials` — kernel/thread/warp/SMEM/coalesce/online softmax
 44. `kernel-engineering` — Triton/CUTLASS/FlashAttn/fused MLP
 45. `cluster-networking` — NVLink/IB/NCCL/SHARP/fat-tree
 46. `storage-dataops` — Lustre/dataloader/sharding/DCP-async ckpt
 47. `training-orchestration` — Slurm/Ray/Young's formula/elastic
 48. `infra-graduation` — Mini-cluster sim + topology selector + Portfolio v3 ⭐⭐⭐⭐⭐⭐⭐⭐

## Section 2: 6-ckpt Zoo + DRA + Cluster blueprints

| Asset | Source | Purpose |
|-------|--------|---------|
| `vanilla` | Module 3 baseline | GPT-2 base |
| `lora` | Module 1 PEFT | + LoRA |
| `dpo` | Module 4 alignment | + DPO |
| `r1_tiny` | Module 4 reasoning | + GRPO R1-Zero |
| `phi_tiny` | Module 3 pretrain | 270M from-scratch |
| `dra_v1` | Module 7 | Deep research agent |
| `cluster_blueprints` ⭐ | Module 8 | Mini-cluster sim recipes |

## Section 3: All Capstones (Modules 1-8)

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
| Pattern Zoo ⭐ | M7 | 1 task × 6 designs 成本对照 |
| Mini-Harness ⭐ | M7 | agentic loop + 权限拦截 + trace |
| Roofline zoo ⭐ | M8 | 10 op × 4 GPU |
| Online softmax ⭐ | M8 | 1-pass FlashAttn kernel |
| Attention HBM zoo ⭐ | M8 | 1025× HBM saved at 128k |
| Fabric zoo ⭐ | M8 | 17500× SHARP gain |
| 7-day ckpt economics ⭐ | M8 | DCP-async vs full |
| 24h Slurm sim ⭐ | M8 | 8 jobs × 512 GPU |
| **Mini-cluster simulator** ⭐⭐⭐ | M8 | 18 scenario time-to-train + TCO |
| **Topology selector** ⭐⭐⭐ | M8 | (model, budget, deadline) → blueprint |
| **Portfolio v3** | M8 | 48-topic (v3.1) ⭐⭐⭐⭐⭐⭐⭐⭐ |

## Section 4: Selection Trees (cumulative)

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

## Section 5: 6 大画像 v3

```
你已具备：

1. 造模型      — 从 0 训 GPT-2 / Phi-tiny (M3)
2. 改模型      — LoRA / DPO / R1-Zero (M1+M4)
3. 用模型      — vLLM / SGLang / 量化 / 分布式 (M5)
4. 评模型      — 25 bench / Arena / 红队 (M6)
5. 守模型      — 4 层防御 + Constitutional Cls (M6)
6. 造 agent   — ReAct / RAG / MCP / multi-agent / memory
              + 设计选型(5 workflow 模式) + 手搭 harness 引擎 (M7, +2)
7. 造 infra ⭐ — GPU / CUDA / kernel / fabric / Slurm (M8 NEW)

= 2026 年 LLM 全栈工程师 ID 卡 v3
```

> 6 大画像 → 7 大画像 ⭐：M8 把"硬件层"加入工具箱

## Section 6: What I Can Do (cover letter snippets v3)

- "I can train a 270M Phi-tiny from scratch (M3)."
- "I can replicate R1-Zero RL on GPT-2, observing aha moment (M4)."
- "I can serve LLM at scale with vLLM/SGLang + 4-bit quant (M5)."
- "I can run mini-HELM 4-dim eval + mini-Arena BT-Elo (M6)."
- "I can build a deep research agent with planner/retriever/writer/verifier (M7)."
- "I can do red-team + 4-layer defense including Constitutional Classifiers (M6)."
- "I can compare 6 agent frameworks and pick one with a decision tree (M7)."
- "I can pick the cheapest workflow design for a task (5 patterns) instead of over-building an agent (M7)." ⭐
- "I can build an agent harness from scratch: agentic loop, tool dispatch, context mgmt, permissions, tracing (M7)." ⭐
- "I can roofline-analyze any LLM op against H100/B200, identifying memory vs compute bound (M8)." ⭐
- "I can write FlashAttn-style online softmax + tiled GEMM kernels in Triton (M8)." ⭐
- "I can size a training cluster for an N-param model under a budget + deadline (M8)." ⭐
- "I can compute Young's formula ckpt frequency for 1k+ GPU cluster (M8)." ⭐

## Section 7: Career Paths v3 (2025 SF salary band)

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

---

## Module 9 科研技能 — 工程之外的第二条腿 (在建 ⭐ NEW, 2026-06-17 起)

> **动因**: Module 1-8 练的是**工程**(复现/实现已有技术), 衡量标准是 "What I Can Do"。
> PhD 衡量的是另一件事: **产出前人没有的新知识**。`I can replicate R1-Zero` ≠ `I can do research`。
> Module 9 在工程地基之上, 补「从消费知识到生产知识」这一层。**课件式专题为主, 按研究项目生命周期编排。**

### 9 专题蓝图 (生命周期)
```
地基    9.1 research-knowledge-mgmt   知识管理 (Zotero/笔记/idea pipeline)
输入    9.2 literature-mapping        文献综述 + 领域地图
       9.3 critical-reading-gap  ✅   批判式读论文 + 找问题   ← 首专题, 已完成
执行    9.4 experiment-design         实验设计 + baseline + 消融
       9.5 experiment-ops-repro      实验管理 + 可复现
输出    9.6 research-figures          科研绘图 (出版级图)
       9.7 paper-writing-submission  科研写作 + 投稿 (升级复用 how_to_write_a_paper)
       9.8 research-presentation     会议 talk / poster / 答辩
科研生活 9.9 research-life            审稿 + 导师沟通 + 伦理
```

### 首专题 `critical-reading-gap` 产出 (已完成)
- 5 讲课件: 三遍读法 → 攻击式阅读(10问) → gap 分类学(6类) → idea 生成(5法+三筛) → 周流水线 SOP
- 2 notebook (已 nbconvert 跑通): N1 解剖一篇真论文 + 拉引用邻域图; **N2 把自己的 R1-Zero 复现当待审稿论文挖 gap**
- 3 模板 + 2 工具 (make_cards 一键起卡 / citation_graph 拉引用图)
- 设计/计划: `docs/superpowers/specs|plans/2026-06-17-research-skills-module9*`

### 第 8 大画像 (在建)
```
1-7 大画像 (工程): 造/改/用/评/守模型 + 造 agent + 造 infra
8. 会做研究的人 ⭐ — 读论文找 gap → 设计实验 → 写成 paper → 投稿/presentation
   (Module 9 完成后, portfolio 将正式升 v4)
```

> 诚实标注: Module 9 仅完成 1/9 专题, 故**暂不升 v4**; 待 9 专题收齐再正式发 portfolio v4「工程 + 研究」双 ID 卡。
