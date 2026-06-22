# LLM Learning Portfolio (v4) — 工程 + 研究 双腿

> Generated 2026-06-06 · Module 8 收官 ⭐⭐⭐⭐⭐⭐⭐⭐
> Extended 2026-06-17 · Module 7 +2 (agent-design-patterns / agent-harness-design) → 48 工程专题
> Extended 2026-06-22 · **`harness-engineering`** ⭐ — 首个「工程 ⨯ 研究」双栖专题 (14 讲 + 生产级 src + 研究桥)
> **升级 2026-06-22 · v4 ⭐⭐ — Module 9「科研技能」9/9 全完成**: 从「消费知识」到「生产知识」的完整研究链路落地, 正式长出第二条腿 (第 8 大画像「会做研究的人」转正)。详见文末。

> **两条腿一句话**: 工程腿 (Module 1-8, 48 专题) = 「我能复现/实现已有技术」; 研究腿 (Module 9, 9 专题) = 「我能产出前人没有的新知识」。前者是 PhD 的地基, 后者是 PhD 的目的。

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

## Section 5: 8 大画像 v4 (7 工程 + 1 研究)

```
你已具备：

—— 工程腿 (Module 1-8, 48 专题) ——
1. 造模型      — 从 0 训 GPT-2 / Phi-tiny (M3)
2. 改模型      — LoRA / DPO / R1-Zero (M1+M4)
3. 用模型      — vLLM / SGLang / 量化 / 分布式 (M5)
4. 评模型      — 25 bench / Arena / 红队 (M6)
5. 守模型      — 4 层防御 + Constitutional Cls (M6)
6. 造 agent   — ReAct / RAG / MCP / multi-agent / memory
              + 设计选型(5 workflow 模式) + 手搭 harness 引擎 (M7)
7. 造 infra   — GPU / CUDA / kernel / fabric / Slurm (M8)

—— 研究腿 (Module 9, 9 专题) ⭐⭐ NEW ——
8. 会做研究的人 — 第二大脑(9.1) → 摸领域(9.2) → 找 gap(9.3) → 设计实验(9.4)
              → 可复现(9.5) → 出版级图(9.6) → 写+投+rebuttal(9.7)
              → 讲/答辩(9.8) → 审稿/导师/伦理/可持续(9.9)
              = 一条从「消费知识」到「生产知识」的完整链路

= 2026 年 LLM 全栈工程师 + 准研究者 ID 卡 v4
```

> 7 大画像 (工程) → 8 大画像 ⭐⭐：Module 9 把"研究能力"加入 ID 卡, portfolio 正式双腿站立。
> **这一步的意义**: 之前 portfolio 通篇是工程师叙事 (能复现什么); v4 加上的是 PhD 真正衡量的东西 (能发现什么)。

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
- "I can take any of my 48 reproductions, find a research gap, design a falsifiable ablation, and write it up (M9)." ⭐⭐
- "I can read a paper adversarially, snowball a subfield into a map, and run a minimal viable experiment with proper variance reporting (M9)." ⭐⭐
- "I can write a rebuttal, review a paper constructively, and run a reproducible experiment pipeline (M9)." ⭐⭐

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
| **PhD / Research Scientist** ⭐⭐ NEW | (学术轨/工业研究院) | **M9** (研究全流程) + M3/M4 (硬核底子) |

> M8 打开两条工程职业路径（GPU 内核 + HPC 集群）; **M9 打开研究轨道 (PhD → Research Scientist), 这正是本仓库的初衷。**

---

## Module 9 科研技能 — 工程之外的第二条腿 ✅ 9/9 全完成 (2026-06-17 起 → 2026-06-22 收官) ⭐⭐

> **动因**: Module 1-8 练的是**工程**(复现/实现已有技术), 衡量标准是 "What I Can Do"。
> PhD 衡量的是另一件事: **产出前人没有的新知识**。`I can replicate R1-Zero` ≠ `I can do research`。
> Module 9 在工程地基之上, 补「从消费知识到生产知识」这一层。**课件式专题为主, 按研究项目生命周期编排。**

### 9 专题 (生命周期) — 全部完成, 每个统一外壳 README+lectures+notebooks+templates+src+environment
```
地基    9.1 research-knowledge-mgmt   ✅  第二大脑: Zotero/Zettelkasten/idea pipeline/arxiv triage
输入    9.2 literature-mapping        ✅  系统摸领域: 滚雪球+引用网中心度+领域地图
       9.3 critical-reading-gap      ✅  批判式读论文+找问题 (首模板专题)
执行    9.4 experiment-design         ✅  可证伪假设→MVE→公平对照→消融矩阵+交互→方差/显著性 (核心)
       9.5 experiment-ops-repro      ✅  seed/config as code/实验追踪/复现 checklist
输出    9.6 research-figures          ✅  出版级图(data-ink/色盲安全)+方法示意图+图的诚实性
       9.7 paper-writing-submission  ✅  装配(桥接 how_to_write_a_paper)+投稿+评审+rebuttal+诚信
       9.8 research-presentation     ✅  知识的诅咒/talk/poster/电梯演讲/答辩 Q&A
科研生活 9.9 research-life            ✅  审稿(反哺写作)+导师沟通+署名伦理+可持续科研
```
> 全部 9 专题: **36 讲研究生级课件 + 18 notebook (全 nbconvert 跑通, 0 报错) + 多份卡模板 + 18 个纯 stdlib/轻量 src 工具**。
> 设计/计划: `docs/superpowers/specs|plans/2026-06-17-research-skills-module9*`

### 一条贯穿全模块的「活数据流」(设计亮点)
```
9.4 确定性模拟器 (埋真实交互效应) → 9.5 留痕 jsonl → 9.6 出版级图
   ↑ 同一份实验数据 (Robust-DPO 噪声鲁棒性, 对应你的 dpo-family 复现) 逐级加工
   = 一个研究结果从「跑出来」到「能印进论文」的完整生命周期, 在代码层面兑现
```

### 可跑工具亮点 (每个专题都有真能用的 src, 不是玩具)
- 9.1 `bib_to_cards` / `arxiv_triage` — BibTeX→文献卡库 / arxiv 关键词打分分诊
- 9.2 `snowball` / `field_map` — 引用网 PageRank 找奠基作 + taxonomy/timeline 出图 (含「中心度相对采样集」caveat)
- 9.4 `experiment` / `stats` — 确定性模拟实验 (修过一个 `hash(str)` 进程级随机盐的真可复现 bug) + bootstrap/Welch t/Cohen's d
- 9.5 `exp_tracker` / `repro_check` — 80 行本地版 wandb (git/env 指纹) + 6 项复现体检
- 9.6 `plotstyle` / `schematic` — 出版级样式包 (Okabe-Ito 色盲安全 + PDF 导出, 写真论文可直接用)
- 9.7 `paper_assembler` / `rebuttal_kit` — 叙事链审计 (抓无证据 claim) + 审稿意见分类/优先级/字数预算
- 9.8 `talk_planner` / `pitch_kit` — talk 时间预算自检 + 电梯演讲三档时长/听众黑话检测
- 9.9 `review_kit` / `meeting_prep` — 结构化审稿+调转枪口审自己 + 导师 meeting「有没有 ask」自检

### 三个身份的批判视角闭环 (Module 9 的暗线)
> **9.3 攻击者** (找别人 gap) → **9.7 被告** (rebuttal 应对攻击) → **9.9 审判者** (给别人审稿)
> 同一套批判视角的三个座位, 让学员理解学术共同体如何自我纠错。

---

## `harness-engineering` — 首个「工程 ⨯ 研究」双栖专题 (2026-06-22) ⭐

> **动因**: 2026 最火的工程方向。模型在商品化, 护城河转移到模型外那层 **harness** (88% 企业 agent 落不了地, 65% 失败源于 harness 缺陷)。是 Module 7 `agent-harness-design` (理解层) 的进阶续作。
> **定位**: 第一个同时长在工程 (Module 7) 和研究 (Module 9) 上的专题 —— 既造生产级 harness, 又用 `critical-reading-gap` 的 gap 雷达把它变成 PhD 研究入口。

### 14 讲 (4 Part)
```
Part I  护城河与定义    L01 模型商品化→护城河 · L02 本构定义(4要素 inclusion test)
Part II 升级到生产级    L03 provider抽象 · L04 5阶段compaction · L05 long-horizon(loop-with-hook)
                       · L06 subagent firewall · L07 tool/MCP控制平面 · L08 安全与控制
Part III 成熟度三件套   L09 OTel可观测 · L10 harness eval · L11 五大架构模式(70系统) · L12 portable harness
Part IV 研究入口        L13 6类gap雷达扫harness开放问题 · L14 Capstone
```

### 可跑的生产级 src (stdlib, MockProvider 默认无需 API key, 11 单测全绿)
- `provider.py` 模型抽象 (流式+tool-call, 换 Anthropic/OpenAI 主体不改)
- `compaction.py` 5 阶段渐进式压缩 (逆向 Claude Code)
- `long_horizon.py` loop-with-hook + 文件系统 state (跨窗口长任务)
- `otel_trace.py` OpenTelemetry 式 span 树 · `harness_eval.py` 同模型换 harness 对照
- 4 notebook (nbconvert 跑通): compaction 实况 / long-horizon 救回 early-stop / harness eval 对照 / Capstone

### 研究桥 (Part IV)
用 6 类 gap 雷达扫出 12 个 harness 候选研究题目 (G1-G12), ⑤复现类 (G9/G10) 对博0 最友好; Capstone 产出 harness 方向 idea 卡。
设计/计划: `docs/superpowers/{specs,plans}/2026-06-22-harness-engineering*`

> 这个专题示范了一种新模式: **同一个热点, 既当工程能力练 (Module 7), 又当研究入口挖 (Module 9)**。
