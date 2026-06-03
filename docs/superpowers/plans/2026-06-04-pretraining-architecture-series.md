# Module 3 · 造大模型 — 预训练 + 架构 8 专题学习系列完整规划

> 设计日期：2026-06-04
> 学习仓库：`c:\Workspace\dummy`
> 系列定位：与 PEFT (Module 1) / RL+对齐 (Module 2) 形成 **造-改对偶**
> 模板来源：已完成的 RL 七专题（rl-foundations → multimodal-agent）

---

## Context — 为什么开此系列

### 学习者当前坐标（2026-06-04）

- ✅ Module 1 **PEFT**（28 方法 / ~29h）：prompt-tuning / lora / adapter 三专题完成
- ✅ Module 2 **RL+对齐+推理**（88 方法 / ~101h）：rl-foundations → rlhf → dpo → process-reward → reasoning-r1 → rl-sota-2026 → multimodal-agent 七专题完成
- ⏳ Module 3 **造大模型**：本次规划

### 为什么这是必修

PEFT 和 RL 解决"如何**调整**一个已有大模型"。但 2024-2026 的开源 SOTA 转折点几乎全在**架构创新**和**预训练 recipe** 上：
- DeepSeek-V3 (2024.12)：MLA + DeepSeekMoE + Aux-Loss-Free 路由
- Llama-3.3 (2024.12)：405B 训练 + 长上下文 + GQA
- Mamba-3 / Hybrid Jamba (2025)：替代 Transformer 的 SSM 路线
- Phi-4 (2024.12) / SmolLM2 / Qwen3-0.6B：小模型时代「数据 > 参数」
- YaRN / RoPE-scaling：长上下文方法论的稳定化

**不学这条线 → 看不懂 2025+ 任何前沿 PR**。Module 1+2 把你训练成"会改的工程师"，Module 3 训练你成"懂造的工程师"。

### 系列设计原则

1. **教学 minimal + 真训放云**：每个专题的算法骨架可在 5090 24GB 跑通；真预训练 capstone（专题 7）需要明确"租云或缩规模"
2. **三轨代码（继承 Module 2）**：minimal 手写 / HuggingFace 库 / Megatron-Core 或同等生产级框架
3. **加一条 naive-kernel 轨**：仅在 `transformer-deep` 和 `long-context` 出现，1-2 个 CUDA / Triton 微 kernel 演示，帮助理解 FlashAttention / RoPE 等底层
4. **承上启下**：第 8 专题的 capstone 把 Module 1+2+3 串起来——五部曲毕业（造 → PEFT → DPO → R1 → 量化部署）

### 输出物

- 本文件：8 专题整体蓝图（执行参考）
- 后续：8 份 `docs/superpowers/specs/2026-MM-DD-<topic>-design.md` + 8 份 `docs/superpowers/plans/2026-MM-DD-<topic>.md`
- 实施：8 个 `learning/<topic>/` 目录（lectures/ + src/ + notebooks/ + environment/ + tests/ + papers/）

---

## 一、8 专题总览

| # | 专题代号 | 一句话定位 | 方法数 | Lec | 时长 | 环境 | git tag |
|---|---------|----------|--------|-----|------|------|---------|
| 1 | `data-curation` | 数据爬清去重 + tokenizer 全套 | 14 | 12 | 14h | Win | `data-curation` |
| 2 | `transformer-deep` | Transformer 架构骨架完整版 | 16 | 14 | 16h | Win | `transformer-deep` |
| 3 | `moe-architecture` | MoE 路线（Switch → DeepSeek-V3）⭐ | 14 | 13 | 16h | **WSL2** | `moe-arch` |
| 4 | `ssm-hybrid` | Mamba / RWKV / 混合架构 | 12 | 11 | 12h | **WSL2** | `ssm-hybrid` |
| 5 | `long-context` | 长上下文主线（RoPE/YaRN/Ring）⭐ | 14 | 13 | 14h | **WSL2** | `long-context` |
| 6 | `scaling-infra` | 训练 infra (ZeRO/FSDP/Megatron/3D) | 16 | 14 | 18h | **WSL2/多卡** | `scaling-infra` |
| 7 | `pretraining-recipe` | 从 0 训 Phi-tiny 完整 capstone ⭐⭐⭐ | 18 | 16 | 20h | **WSL2/云** | `pretraining-recipe` |
| 8 | `small-model-graduation` | 小模型时代 + 五部曲毕业 ⭐⭐⭐⭐⭐ | 14 | 14 | 12h | **WSL2** | `造改-graduation` |
| | **合计** | | **118** | **107** | **122h** | | |

对照 RL 系列 (88 方法/101h)，本系列体量约 **1.2×**，因架构/infra 内容硬核度高。

### 依赖关系图

```
专题 1: 数据 + tokenizer  (Win, 14h)
        |
        ↓
专题 2: Transformer 架构  (Win, 16h)
        |
        ├──→ 专题 3: MoE 架构 (WSL2, 16h)    \
        ├──→ 专题 4: SSM/Mamba (WSL2, 12h)    |  并行可能
        └──→ 专题 5: 长上下文 (WSL2, 14h)    /
                              ↓
        专题 6: 训练 infra (WSL2/多卡, 18h)
                              ↓
        专题 7: 预训练 capstone (WSL2/云, 20h, 系列高峰)
                              ↓
        专题 8: 小模型 + 五部曲毕业 (WSL2, 12h, 系列收官)
```

### 实施排期建议（接续 Module 2 后）

| 月份 | 专题 | 是否切环境 |
|------|------|----------|
| 2026-11 全月 | 专题 1 数据 + 专题 2 Transformer | Win |
| 2026-12 前半 | 专题 3 MoE | **WSL2 切换** |
| 2026-12 后半 | 专题 4 SSM | — |
| 2027-01 前半 | 专题 5 长上下文 | — |
| 2027-01 后半 | 专题 6 训练 infra | 多卡云租用建议 |
| 2027-02 全月 | 专题 7 预训练 capstone | 云 24h+ 训练 |
| 2027-03 前半 | 专题 8 毕业 | — |

共 4 个月。如压 3 个月可削专题 4（SSM 短期热度低于 MoE）。

---

## 二、专题 1：数据 curation + tokenizer（`data-curation`）

### 定位
LLM 第一步从来不是模型——是**数据**。本专题覆盖数据获取、清洗、去重、质量过滤、分词的全套，2024-2026 「数据为王」的工程基线。

### 章节规划（12 lectures）

| Lec | 主题 | 核心 idea |
|-----|------|---------|
| 01-data-overview.md | LLM 数据全景 (2017-2026) | C4 / The Pile / RedPajama / FineWeb / DCLM 演化 |
| 02-commoncrawl.md | CommonCrawl 抽取 | WARC 格式 / trafilatura / language detection |
| 03-dedup.md | 去重策略 | MinHash / SimHash / suffix array / SemDeDup |
| 04-quality-filter.md | 质量过滤 | C4 启发式 / Wiki-classifier / FineWeb-Edu 分类器 |
| 05-toxicity-pii.md | 毒性 / PII 过滤 | KenLM / Detoxify / 正则 / Presidio |
| 06-tokenizer-bpe.md | BPE 算法 | tiktoken / 训练 / merge rule |
| 07-tokenizer-spm.md | SentencePiece + Unigram | 与 BPE 对比 / 多语言场景 |
| 08-tokenizer-byte.md | Byte-level / 多语言 tokenizer | GPT-4 cl100k / Llama-3 128k 词表 |
| 09-data-mix.md | 数据配比 | DataComp / Doremi / 自动权重学习 |
| 10-instruction-data.md | 指令数据合成 | Alpaca / Self-Instruct / Magpie 系列 |
| 11-curation-pitfalls.md | 数据陷阱合集 | 重复 / 污染 / benchmark 泄漏 |
| 12-capstone-mini-corpus.md | Capstone：自制 1B-token 微语料 | CommonCrawl → 去重 → 过滤 → 训 tokenizer |

### src/ 三轨规划

| 文件 | 实现 |
|------|------|
| `cc_extract.py` | CommonCrawl WARC 抽取 + trafilatura HTML 清洗 |
| `minhash_dedup.py` | datasketch MinHash + LSH 去重 |
| `simhash_dedup.py` | 手写 SimHash 算法 |
| `quality_filter.py` | FineWeb-Edu 风格分类器训练 |
| `bpe_trainer.py` | 手写 BPE 训练（参考 Karpathy minbpe）|
| `bpe_tiktoken.py` | tiktoken 对照 |
| `spm_trainer.py` | sentencepiece 训练 |
| `vocab_compare.py` | 不同 tokenizer 在多语言上压缩率对照 |
| `data_mix_ablation.py` | 不同配比下小模型 perplexity 对照 |
| `tests/test_dedup_recall.py` | 注入重复，验证 MinHash 召回 |
| `tests/test_bpe_consistency.py` | 自训 vs tiktoken merge sequence 对照 |

### Capstone：自制 1B-token 微语料
- **流程**：CommonCrawl 1 dump → 抽取 200GB → trafilatura → MinHash 去重（保留 30%）→ FineWeb-Edu 过滤（保留 10%）→ 训 32k 词表 SentencePiece
- **输出**：~1B-token jsonl + 自训 tokenizer
- **指标**：去重前后大小 / 过滤前后 quality 分布 / tokenizer 压缩率
- **耗时**：5090 + 高速 SSD ~10h（IO bound）

### 环境
```
trafilatura warcio datasketch simhash sentencepiece tiktoken
fasttext langdetect presidio detoxify
```

### 风险
- CommonCrawl 单 dump ~80TB → 用 1 个 segment 即可（~200GB）
- 法律合规 → 注明 dataset license / 不公开下载脚本输出
- 玩具 quality classifier 准确率有限 → 期望管理 70%

### 退出条件
- [ ] 12 lecture + notebook 全跑通
- [ ] capstone：1B-token jsonl + 32k SP tokenizer 输出
- [ ] BPE 自训 vs tiktoken 一致性 PASS
- [ ] tag `data-curation`

---

## 三、专题 2：Transformer 架构骨架完整版（`transformer-deep`）

### 定位
GPT-2 → Llama-3 → DeepSeek-V3 的架构演化全图。每个组件**单文件实现**，便于横向对比。

### 章节规划（14 lectures）

| Lec | 主题 | 核心 idea |
|-----|------|---------|
| 01-transformer-recap.md | Vanilla Transformer 复习 | Attention is All You Need 原文 |
| 02-positional-encoding.md | 位置编码全套 | Sinusoidal / Learned / RoPE / ALiBi / NoPE |
| 03-rope-deep.md | RoPE 数学推导 ⭐ | 复数旋转 / 相对位置 / 推外能力 |
| 04-attention-variants.md | Attention 变体 | MHA / MQA / GQA / **MLA (DeepSeek-V2)** |
| 05-normalization.md | LayerNorm 家族 | LayerNorm / RMSNorm / Pre vs Post / DeepNorm |
| 06-activation.md | 激活函数演化 | ReLU / GELU / SwiGLU / GeGLU |
| 07-flash-attention-v1.md | FlashAttention 算法 ⭐ | tiling / online softmax / 内存层级 |
| 08-flash-attention-v2-v3.md | FA2 / FA3 改进 | warp specialization / TMA / FP8 |
| 09-paged-attention.md | PagedAttention (vLLM) | KV cache 内存碎片管理 |
| 10-sliding-window.md | Sliding Window Attention | Mistral / Gemma 长上下文方案 |
| 11-architecture-search.md | 现代 LLM 架构搜索 | μP / MUP / hyperparam transfer |
| 12-deepseek-v3-walkthrough.md | DeepSeek-V3 架构精读 ⭐ | MLA + DeepSeekMoE + Aux-Free 路由 |
| 13-llama3-walkthrough.md | Llama-3 / Llama-3.3 精读 | GQA 8/64 头 / 128k 上下文 |
| 14-capstone-build-gpt-mini.md | Capstone：从 0 搭 80M GPT-mini | 用所有现代组件（RoPE+RMSNorm+SwiGLU+GQA）|

### src/ 三轨规划

| 文件 | 实现 |
|------|------|
| `rope.py` | 标准 RoPE + position interpolation |
| `mha.py` / `mqa.py` / `gqa.py` / `mla.py` | 4 种 attention 独立文件 |
| `rmsnorm.py` | 手写 RMSNorm + 与 torch.nn.RMSNorm 对照 |
| `swiglu.py` | SwiGLU / GeGLU / 与 GELU MLP 对照 |
| `flash_attn_naive.py` | naive triton 实现 FA1（教学版）|
| `flash_attn_lib.py` | flash-attn 库调用 + benchmark |
| `paged_attn_demo.py` | KV cache 分页演示 |
| `sliding_window.py` | Mistral 风格 sliding window |
| `gpt_mini.py` | 集成所有组件的 80M GPT |
| `tests/test_rope_consistency.py` | naive vs flash RoPE 一致性 |
| `tests/test_flash_attn_correctness.py` | naive triton vs flash-attn 库 1e-4 |
| `tests/test_gpt_mini_forward.py` | 模型可前向 + 反向 + 显存预测对 |

### Capstone：从 0 搭 80M GPT-mini
- **架构**：12 层 / hidden 768 / 12 GQA 头（kv=2）/ RoPE / RMSNorm / SwiGLU
- **训练**：在 OpenWebText 1B-token 上训 1 epoch（用专题 1 capstone 输出）
- **指标**：train loss 6.0 → 3.5；val ppl < 30
- **耗时**：5090 单卡 ~6h

### 环境
```
torch>=2.5+cu130
flash-attn>=2.6
triton  # naive kernel 教学
einops
```

### 风险
- FA3 需 H100 → 只放代码 + 性能图，不强制运行
- MLA 数学复杂 → L04 单独 30 分钟推导
- naive triton kernel 与 flash-attn 精度差 → 测试用 1e-4 容差

### 退出条件
- [ ] 14 lecture + notebook 全跑通
- [ ] Capstone 80M GPT val ppl < 30
- [ ] flash-attn 库 vs naive 一致性 PASS
- [ ] tag `transformer-deep`

---

## 四、专题 3：MoE 架构（`moe-architecture`）⭐

### 定位
2024-2026 开源大模型走向 MoE 已成定局（DeepSeek-V3 671B / Mixtral / Phi-MoE / Qwen3-MoE 全是 MoE）。本专题覆盖路由算法演化 + 训练稳定化 + 推理效率全套。

### 章节规划（13 lectures）

| Lec | 主方法 | 团队/时间 | 核心 idea |
|-----|--------|----------|---------|
| 01-moe-intro.md | MoE 概念起源 | Shazeer 2017 | gating + experts，sparse activation |
| 02-gshard.md | GShard | Google 2020 | top-2 routing / expert parallel |
| 03-switch.md | **Switch Transformer** | Google 2021 | top-1 / 简化 / 1.6T 参数 |
| 04-expert-choice.md | Expert Choice | Google 2022 | 反向路由（experts 选 token）|
| 05-mixtral.md | **Mixtral 8x7B** | Mistral 2023.12 | 开源 MoE 起点 |
| 06-deepseekmoe.md | **DeepSeekMoE** | DeepSeek 2024 | 细粒度 + 共享专家 |
| 07-aux-loss-free.md | **Aux-Loss-Free 路由** | DeepSeek-V3 2024.12 ⭐ | 偏置项替代 aux loss |
| 08-phi-moe.md | Phi-3.5-MoE / Phi-4-MoE | Microsoft 2024-2025 | 小 MoE 路线 |
| 09-qwen3-moe.md | Qwen3-MoE | Alibaba 2025 | A3B / 235B 系列 |
| 10-mor.md | **MoR (Mixture of Recursions)** 2025 | — | MoE × Recurrence 新方向 |
| 11-moe-training.md | MoE 训练稳定化 | — | router z-loss / capacity factor / load balance |
| 12-moe-inference.md | MoE 推理优化 | — | expert offloading / sparse compile |
| 13-capstone-mini-moe.md | Capstone：自训 4-expert mini-MoE | — | 在专题 2 GPT-mini 上加 MoE 层 |

### src/ 三轨规划

| 文件 | 实现 |
|------|------|
| `moe_layer_naive.py` | 手写 4 expert + top-2 gating |
| `gshard_router.py` | top-2 + aux loss + capacity factor |
| `switch_router.py` | top-1 简化 |
| `expert_choice.py` | reversed routing |
| `aux_loss_free.py` | DeepSeek-V3 偏置项实现 ⭐ |
| `moe_megablocks.py` | megablocks 库对照 |
| `moe_grouped_gemm.py` | grouped GEMM 高效实现 |
| `router_z_loss.py` | 训练稳定 trick |
| `expert_offload.py` | 推理时 CPU offload 演示 |
| `mini_moe.py` | 4 expert × 80M base 集成 |
| `tests/test_router_load_balance.py` | 各 expert 利用率均匀性 |
| `tests/test_aux_free_stability.py` | aux-free 收敛性 |

### Capstone：自训 4-expert mini-MoE
- **架构**：专题 2 GPT-mini 的 MLP 替换成 4 expert top-2 MoE
- **训练**：同样 1B token，对比 dense 80M 与 MoE-80M(active)
- **指标**：相同 train compute 下 MoE val ppl < dense val ppl ≥ 1.5
- **耗时**：5090 + WSL2 ~8h

### 环境
```
megablocks  # MoE 高效 kernel
torch>=2.5+cu130 flash-attn>=2.6
deepspeed  # MoE expert parallel
```

### 风险
- expert parallel 需 ≥2 卡 → capstone 用单卡 4 expert（不分布式）
- Aux-Free 数学复杂 → L07 完整推导 + 30min 视频
- megablocks 安装坑 → Dockerfile 备份

### 退出条件
- [ ] 13 lecture + notebook 全跑通
- [ ] Capstone MoE vs dense 收益 ≥ 1.5 ppl
- [ ] 路由负载均衡测试 PASS
- [ ] tag `moe-arch`

---

## 五、专题 4：SSM / Hybrid 架构（`ssm-hybrid`）

### 定位
非 Transformer 路线全图。Mamba 系列 + RWKV + Jamba/Zamba 混合架构。这是 2024-2026 的"另一条路"。

### 章节规划（11 lectures）

| Lec | 主方法 | 核心 idea |
|-----|--------|---------|
| 01-ssm-intro.md | SSM 数学背景 | 状态空间方程 / HiPPO / S4 |
| 02-s4-s5.md | S4 / S5 | 卷积形式 / 频域 / 离散化 |
| 03-mamba.md | **Mamba** (Gu 2023) | Selective SSM / 硬件感知 scan |
| 04-mamba2.md | **Mamba-2** (2024) | SSD / 矩阵分解 / 与 attention 等价性 |
| 05-mamba3.md | **Mamba-3** (2025) | 长上下文优化 |
| 06-rwkv-7.md | **RWKV-7** (2025) | linear attention 路线 |
| 07-retnet.md | RetNet / RWKV-6 | retention mechanism |
| 08-jamba.md | **Jamba** (AI21 2024) | Mamba + MoE + attention 混合 |
| 09-zamba.md | Zamba / Zamba-2 | Mamba + shared attention |
| 10-hybrid-design.md | 混合架构设计 | 何时混 / 比例 / 选 layer |
| 11-capstone-mamba-mini.md | Capstone：mini-Mamba 130M | OpenWebText 1B + 与 GPT-mini 对照 |

### src/ 三轨规划

| 文件 | 实现 |
|------|------|
| `s4_naive.py` | 手写 S4（卷积形式）|
| `mamba_block.py` | 手写 Mamba block (无 selective scan kernel) |
| `mamba_lib.py` | mamba_ssm 库调用 |
| `mamba2_block.py` | Mamba-2 SSD 形式 |
| `rwkv_block.py` | RWKV-7 简化 |
| `jamba_block.py` | hybrid layer 配比 |
| `mini_mamba.py` | 130M Mamba 完整模型 |
| `compare_dense_vs_mamba.py` | 同算力下 ppl + 长上下文对比 |
| `tests/test_mamba_correctness.py` | naive vs mamba_ssm 库 |
| `tests/test_long_context_extrapolation.py` | 训 1k 推 4k 的 perplexity |

### Capstone：130M Mamba mini
- **架构**：24 层 Mamba (d_model=512)
- **训练**：1B-token OpenWebText
- **指标**：与同算力 80M GPT-mini 对比，长上下文外推优势
- **耗时**：5090 ~8h

### 环境
```
mamba-ssm causal-conv1d>=1.4
rwkv  # RWKV 库
```

### 风险
- mamba_ssm 安装常坑（cuda 版本敏感）→ 提供 known-good combo
- 130M 在 1B token 上 underfit → 用 reduced loss 比较
- 教学价值 > 实际性能 → 明确"理解为主"

### 退出条件
- [ ] 11 lecture + notebook 全跑通
- [ ] Capstone Mamba train loss 下降合理
- [ ] 长上下文外推测试 PASS
- [ ] tag `ssm-hybrid`

---

## 六、专题 5：长上下文（`long-context`）⭐

### 定位
2024-2026 长上下文从 4k → 1M → 10M。本专题覆盖 RoPE 扩展方法、注意力分布式实现、infini-attention 系列。

### 章节规划（13 lectures）

| Lec | 主方法 | 核心 idea |
|-----|--------|---------|
| 01-long-context-overview.md | 2024-2026 长上下文全景 | Gemini 2M / Claude 200k / GPT-4 128k |
| 02-rope-scaling-basic.md | 简单 RoPE scaling | Position Interpolation (Meta 2023) |
| 03-ntk-aware.md | **NTK-aware scaling** | LocalLlama 社区 / base scaling |
| 04-yarn.md | **YaRN** (Peng 2023) ⭐ | NTK by parts / attention temperature |
| 05-rope-3d.md | 3D RoPE / Q-RoPE | 多模态 + 视频位置 |
| 06-ring-attention.md | **Ring Attention** (Liu 2023) | sequence parallel 1M context |
| 07-striped-attention.md | Striped Attention | Ring 改进 |
| 08-infini-attention.md | Infini-Attention (Google 2024) | compressive memory |
| 09-flash-attention-causal.md | FA 长上下文优化 | block-sparse / window |
| 10-needle-haystack.md | 评测：Needle in Haystack | NIAH / LongBench / RULER |
| 11-long-context-data.md | 长上下文训练数据 | book-length / repo-level / 拼接策略 |
| 12-long-context-pitfalls.md | 长上下文陷阱 | lost in middle / 注意力稀释 |
| 13-capstone-yarn-extension.md | Capstone：YaRN 把 Llama-3.2-1B 从 8k 扩到 32k | NIAH 验证 |

### src/ 三轨规划

| 文件 | 实现 |
|------|------|
| `rope_pi.py` | Position Interpolation 实现 |
| `rope_ntk.py` | NTK-aware scaling |
| `rope_yarn.py` | YaRN 完整实现 ⭐ |
| `ring_attention_naive.py` | naive ring attention 教学版 |
| `ring_attention_lib.py` | ring-flash-attention 库调用 |
| `infini_attention.py` | compressive memory 简化 |
| `niah_eval.py` | Needle in Haystack 评测 |
| `ruler_eval.py` | RULER benchmark 子集 |
| `long_data_packing.py` | book + code 长 sample 拼接 |
| `tests/test_rope_extrapolation.py` | 训 4k 推 16k 的 ppl |
| `tests/test_niah_pass_rate.py` | NIAH 通过率 ≥ 90% |

### Capstone：YaRN 把 Llama-3.2-1B 从 8k 扩到 32k
- **基座**：Llama-3.2-1B (8k 原生)
- **方法**：YaRN scaling + 32k 数据微调（LoRA）
- **数据**：long-context book / pile-narrative / repo
- **指标**：NIAH 32k pass rate > 80%；perplexity 不退化
- **耗时**：5090 ~6h

### 环境
```
ring-flash-attention
flash-attn>=2.6
RULER  # benchmark
```

### 风险
- 32k 训练显存 → 用 LoRA + GQA
- Ring attention 真正分布式需 4 卡 → naive 实现单卡演示
- NIAH 通过率要求 → 期望管理 80%

### 退出条件
- [ ] 13 lecture + notebook 全跑通
- [ ] Capstone NIAH 32k pass rate > 80%
- [ ] RoPE 一致性 PASS
- [ ] tag `long-context`

---

## 七、专题 6：训练 infra（`scaling-infra`）

### 定位
ZeRO / FSDP / Megatron / 3D 并行 / 通信原语全套。这是真训百亿模型的工程地基。

### 章节规划（14 lectures）

| Lec | 主题 | 核心 idea |
|-----|------|---------|
| 01-scaling-laws.md | Scaling laws | Kaplan / Chinchilla / Hoffmann |
| 02-chinchilla.md | Chinchilla 教训 | 数据/参数 平衡 |
| 03-mup.md | μP (Maximal Update Param) | 超参 transfer |
| 04-mixed-precision.md | 混合精度 | FP16 / BF16 / FP8 (H100) |
| 05-gradient-accum.md | 梯度累积 + checkpointing | 显存换算力 |
| 06-zero-1-2-3.md | ZeRO 1/2/3 | optimizer / grad / param 分片 |
| 07-fsdp.md | **PyTorch FSDP** | DDP+ZeRO 集大成 |
| 08-tensor-parallel.md | Tensor 并行 (Megatron) | column / row split |
| 09-pipeline-parallel.md | Pipeline 并行 | GPipe / 1F1B / interleaved |
| 10-3d-parallel.md | 3D 并行 | DP × TP × PP 组合 |
| 11-deepspeed.md | DeepSpeed 完整栈 | ZeRO + offload + infinity |
| 12-megatron-core.md | Megatron-Core | 工业级 LLM 训练框架 |
| 13-comm-primitives.md | 通信原语 | all-reduce / all-gather / NCCL |
| 14-capstone-fsdp-train.md | Capstone：FSDP 训 350M 模型 | 单卡 vs 模拟多卡对比 |

### src/ 三轨规划

| 文件 | 实现 |
|------|------|
| `mixed_precision_demo.py` | FP16 vs BF16 train loss 稳定性 |
| `grad_accum_demo.py` | 梯度累积等价于大 batch 验证 |
| `zero_naive.py` | 手写 ZeRO-1 (optimizer 分片) 玩具 |
| `fsdp_train.py` | FSDP wrap + 训练完整 example |
| `tp_megatron.py` | tensor parallel column-row 拆分 |
| `pp_gpipe.py` | pipeline 微批 调度 |
| `deepspeed_zero3.py` | DeepSpeed ZeRO-3 配置 |
| `megatron_core_minimal.py` | Megatron-Core 启动 |
| `flops_calculator.py` | 给定模型估 FLOPs / 显存 |
| `tests/test_grad_accum_equiv.py` | 累积等价于大 batch |
| `tests/test_fsdp_loss_match.py` | FSDP vs DDP 单步 loss 一致 |

### Capstone：FSDP 训 350M 模型
- **架构**：GPT-style 350M (24 层 / hidden 1024)
- **训练**：100M-token 子集，FSDP wrap
- **对比**：与无 FSDP 单卡训练显存对比
- **耗时**：5090 单卡 ~10h；建议租 2-4 卡云 ~3h

### 环境
```
deepspeed>=0.15 megatron-core>=0.9
torch>=2.5+cu130
nvidia-nccl
```

### 风险
- 多卡云租用 → 提供云预算建议（vast.ai 4×A100 ~$10/h）
- Megatron-Core 安装坑 → Docker 备份
- 单卡模拟 multi-GPU → 用 torchrun --nproc-per-node=4 + CPU shard 演示

### 退出条件
- [ ] 14 lecture + notebook 全跑通
- [ ] Capstone FSDP 350M 训练 100M token 完成
- [ ] FSDP vs DDP loss 一致性 PASS
- [ ] tag `scaling-infra`

---

## 八、专题 7：预训练 capstone（`pretraining-recipe`）⭐⭐⭐

### 定位
**系列高峰**。从 0 训一个完整 270M 参数 Phi-style 模型，覆盖完整 recipe（warmup / cosine / muP / clip / EMA / restart 等）。

### 章节规划（16 lectures）

| Lec | 主题 | 核心 idea |
|-----|------|---------|
| 01-pretraining-overview.md | 现代预训练全流程 | data → tokenizer → model → train → eval |
| 02-warmup-cosine.md | learning rate schedule | linear / cosine / warmup-stable-decay |
| 03-optimizer-choice.md | optimizer 选型 | AdamW / Lion / Shampoo / Sophia |
| 04-grad-clip-loss-spike.md | 梯度裁剪 + loss spike 处理 | rollback / skip / restart |
| 05-init-strategy.md | 初始化 | Xavier / He / μP init |
| 06-dropout-strategy.md | dropout / weight decay | 早期 / 后期策略 |
| 07-data-loading.md | 数据加载 | StreamingDataset / WebDataset / Mosaic |
| 08-loss-monitor.md | loss 监控 | wandb / tensorboard / 早期警报 |
| 09-eval-during-train.md | 训练时评测 | val loss / downstream tasks |
| 10-checkpoint-strategy.md | checkpoint 管理 | resume / EMA / averaging |
| 11-phi-recipe.md | **Phi 系列 recipe** | 教科书数据 + 合成数据 |
| 12-llama3-recipe.md | **Llama-3 recipe** | 15T token + 长上下文 + annealing |
| 13-data-annealing.md | 数据 annealing | 末段切高质量 |
| 14-mid-training.md | mid-training / continued pretraining | 引入新能力 |
| 15-capstone-phi-tiny-train.md | Capstone：从 0 训 270M Phi-tiny ⭐⭐⭐ | 完整 20h 训练 |
| 16-capstone-evaluation.md | Capstone 评测 | MMLU / GSM8K / HumanEval 小子集 |

### src/ 规划（重头戏）

| 文件 | 实现 |
|------|------|
| `pretrain_main.py` | 主入口（模型+数据+optim+train loop）|
| `model_phi_tiny.py` | 270M Phi-style 模型定义 |
| `data_streaming.py` | StreamingDataset 多文件迭代 |
| `optimizer_lion.py` | Lion 优化器实现 |
| `optimizer_sophia.py` | Sophia 优化器（Hessian-aware）|
| `scheduler_wsd.py` | warmup-stable-decay |
| `loss_spike_handler.py` | 自动 rollback |
| `checkpoint_avg.py` | model averaging / EMA |
| `mup_init.py` | μP init + lr 转移 |
| `eval_harness.py` | 训中评测 MMLU/GSM8K 子集 |
| `data_annealing.py` | 末段切换高质量数据 |
| `tests/test_resume_consistency.py` | resume 后 loss 连续 |
| `tests/test_ema_correctness.py` | EMA 数学正确 |

### Capstone：从 0 训 270M Phi-tiny
- **架构**：Phi-style 270M (24 层 / hidden 1024 / GQA 16/4 / SwiGLU / RMSNorm / RoPE)
- **数据**：用专题 1 capstone 输出 + Phi-合成数据子集 ~5B token
- **训练**：
  - 5090 单卡 ~20h（如可接受）
  - 或租 4×A100 ~4h（推荐，省时）
- **评测**：
  - val ppl < 25
  - GSM8K 0-shot ≥ 5% (270M 极限)
  - MMLU 5-shot ≥ 30%
- **退出**：达到上面 3 个指标即视为"造模型成功"

### 环境
```
deepspeed megatron-core
streaming  # mosaic
lm-eval-harness>=0.4
wandb
```

### 风险
- 270M × 5B token ≈ 1.4 PFLOPS hours → 5090 24h 紧
- 真训失败概率 ≥ 30% → 提供预训练 checkpoint 下载
- 评测 underperform → 期望管理（270M 类似 GPT-2-small 性能）

### 退出条件
- [ ] 16 lecture + notebook 全跑通
- [ ] Capstone val ppl < 25
- [ ] GSM8K 0-shot ≥ 5%
- [ ] tag `pretraining-recipe`

---

## 九、专题 8：小模型 + 五部曲毕业（`small-model-graduation`）⭐⭐⭐⭐⭐

### 定位
**系列终点**。覆盖 2024-2026 小模型潮 + 五部曲毕业作品（Module 1+2+3 完整串联）。

### 章节规划（14 lectures）

| Lec | 主方法 | 核心 idea |
|-----|--------|---------|
| 01-small-model-era.md | 小模型时代背景 | 边缘 + 隐私 + 成本 |
| 02-phi-1-2-3-4.md | **Phi 全系列** | 教科书数据 + 合成数据 |
| 03-llama-3.2-edge.md | Llama-3.2 1B/3B | 边缘部署 |
| 04-qwen3-small.md | **Qwen3-0.6B/1.7B/4B** | 多语言小模型 |
| 05-smollm2.md | **SmolLM2** | HF 完全开源小模型 |
| 06-tinyllama.md | TinyLlama | 1B 长训练 |
| 07-gemma-3.md | Gemma-3 | Google 开源小模型 |
| 08-h20-distillation.md | 蒸馏 | KD / TinyStories / sequence-level |
| 09-soft-distill.md | logit / hidden 蒸馏 | DistilBERT 风格 |
| 10-bitnet.md | **BitNet b1.58** | 1-bit/三值 LLM |
| 11-quantization-aware-pretrain.md | 量化感知预训练 | 训练阶段 QAT |
| 12-capstone-distill-phi-tiny.md | Capstone-1：把专题 7 270M 蒸馏到 80M | logit distill |
| 13-five-line-unification-revisit.md | 五部曲综合理论 ⭐⭐⭐ | 造 → 改 → 用 → 扩 → 包 |
| 14-capstone-graduation.md | Capstone-2：五部曲毕业作品 ⭐⭐⭐⭐⭐ | 同一 prompt 跨 5 路径对照 |

### src/ 规划

| 文件 | 实现 |
|------|------|
| `phi_4_load_test.py` | Phi-4 加载 + 推理 |
| `smollm2_load_test.py` | SmolLM2 加载 |
| `distill_logit.py` | logit-level 蒸馏 (KL)  |
| `distill_hidden.py` | hidden-state 蒸馏 |
| `bitnet_demo.py` | BitNet b1.58 forward 演示 |
| `qat_pretrain.py` | 量化感知训练 |
| `mini_distill_phi.py` | 把 270M Phi-tiny 蒸馏到 80M |
| `capstone_five_module_graduation.py` | ⭐⭐⭐⭐⭐ 五部曲毕业作品 |
| `tests/test_distill_loss.py` | 蒸馏后 student loss 接近 teacher |
| `tests/test_bitnet_forward.py` | BitNet 数值正确 |

### Capstone-1：把 270M Phi-tiny 蒸馏到 80M
- **Teacher**：专题 7 capstone 输出（270M）
- **Student**：专题 2 GPT-mini 架构（80M）
- **方法**：logit + hidden 双蒸馏
- **指标**：student val ppl < teacher × 1.3
- **耗时**：5090 ~6h

### Capstone-2：⭐⭐⭐⭐⭐ 五部曲毕业作品

**同一道 GSM8K 题** + **5 个 model 演化路径**（Module 1+2+3 完整复盘）：

| # | 路径 | 来源 | 说明 |
|---|------|------|------|
| 1 | **Vanilla GPT-2** | 公开 ckpt | base |
| 2 | **LoRA 微调** | Module 1 ckpt | 改 weight |
| 3 | **DPO 对齐** | Module 2 ckpt | 改 distribution |
| 4 | **R1-Zero 推理** | Module 2 ckpt | 改 trajectory |
| 5 | **自训 Phi-tiny** | Module 3 ckpt ⭐ | **造** model |

**对照可视化**：
- 同一道 "Janet has 16 eggs..." 题
- 5 种 response + 5 种推理深度 + 5 种 latency / 显存
- 雷达图：格式 / 准确 / 推理深度 / 自检 / 响应速度
- **核心观察**：第 5 路径 (自训 270M) 不一定最好——但展示了从 0 造的能力

### 五部曲统一理论（L13 详解 32 slides）

```
        ┌──────────────────────────────────────────┐
        │  p(y | x ; θ_data, θ_arch, θ_weight, φ)  │
        └──────────────────────────────────────────┘
        造 (Module 3): θ_data + θ_arch + θ_weight
        改 (Module 1): φ_LoRA / φ_Adapter / φ_Prompt
        改 (Module 2): φ_RLHF / φ_R1（分布/轨迹）
        用 (Module 4 后续): 量化 / 部署 / 推理
        扩 (Module 5 后续): 多模态 / Diffusion
        包 (Module 6 后续): Agent / Orchestration
```

→ **整个 LLM 工程学的统一框架**。Capstone-2 是这个框架的实证。

### 环境
继承前面所有专题。

### 风险
- 五个 ckpt 全加载显存 → 提供加载脚本+下载链接
- 自训 ckpt 可能太弱 → 期望管理（"展示造的能力 != 性能")
- Capstone-2 需要前面所有 Module 的 ckpt → 提供 fallback 下载

### 退出条件
- [ ] 14 lecture + notebook 全跑通
- [ ] Capstone-1 蒸馏后 ppl 退化 < 30%
- [ ] Capstone-2 五路径对照可视化完成
- [ ] tag `造改-graduation` ⭐ 系列收官

---

## 十、跨专题工程策略

### 三轨代码策略（汇总）

| 专题 | minimal | 库 | 工业级 |
|------|---------|-----|--------|
| 1 数据 | 手写 BPE / MinHash | tiktoken / sentencepiece | datasets / Mosaic |
| 2 Transformer | 手写 RoPE/RMSNorm/GQA | flash-attn 库 | (无,本身是基础) |
| 3 MoE | 手写 top-2 router | megablocks | DeepSpeed-MoE |
| 4 SSM | 手写 S4/Mamba block | mamba-ssm | (本身)|
| 5 长上下文 | 手写 YaRN/Ring | ring-flash-attention | (Megatron 内置) |
| 6 Infra | 手写 ZeRO-1 玩具 | torch FSDP | Megatron-Core / DeepSpeed |
| 7 预训练 | 全配方组合 | lm-eval-harness | Megatron-Core + Mosaic |
| 8 毕业 | (无新算法) | transformers | (无) |

**新增 naive-kernel 轨**：
- `transformer-deep` L07-08：Triton 手写 naive FA1 / FA2
- `long-context` L06：naive ring attention 单卡演示

### 环境策略

**前 2 专题（Windows native）**：
- 复用 Module 1+2 cu130 nightly torch
- 主用 datasets / tiktoken / sentencepiece
- 每专题独立 venv

**专题 3-8（WSL2）**：
- 沿用 Module 2 末期已切换的 WSL2
- 加 deepspeed + megatron-core + flash-attn
- 专题 6/7 推荐租云多卡

### 一致性测试新定义（区别于 RL）

| 测试类型 | 标准 | 示例 |
|---------|------|------|
| **数值正确** | naive vs 库实现 < 1e-4 | flash-attn / Mamba |
| **去重召回** | 注入重复，召回率 > 95% | MinHash dedup |
| **训练稳定** | 500 step 内 loss 单调下降 | GPT-mini / MoE-mini |
| **资源准确** | 显存预测 vs 实测 < 10% | FLOPs calculator |
| **性能基线** | 标准任务达到论文报告值 ±20% | 270M Phi-tiny GSM8K |

### Git 里程碑

| Tag | 时机 | 内容 |
|-----|------|------|
| `data-curation` | 专题 1 末 | 1B-token 自制语料 + 32k SP tokenizer |
| `transformer-deep` | 专题 2 末 | 80M GPT-mini ppl < 30 |
| `moe-arch` | 专题 3 末 | 4-expert mini-MoE ≥ dense 1.5 ppl |
| `ssm-hybrid` | 专题 4 末 | 130M Mamba 长上下文外推 |
| `long-context` | 专题 5 末 | Llama-3.2-1B YaRN 32k NIAH > 80% |
| `scaling-infra` | 专题 6 末 | 350M FSDP 训练完成 |
| `pretraining-recipe` | 专题 7 末 | ⭐⭐⭐ 270M Phi-tiny GSM8K ≥ 5% |
| `造改-graduation` | 专题 8 末 | ⭐⭐⭐⭐⭐ 五部曲毕业作品 |

---

## 十一、2025-2026 高影响力方法补充清单

### 架构层
| 方法 | 团队/时间 | 影响 | 一句话 |
|------|----------|------|--------|
| **DeepSeek-V3 (MLA+DeepSeekMoE+Aux-Free)** | DeepSeek 2024.12 | ⭐⭐⭐⭐⭐ | 671B 开源 MoE 巅峰 |
| **Llama-3.3 405B** | Meta 2024.12 | ⭐⭐⭐⭐⭐ | 开源 dense 巅峰 |
| **Mamba-2 / Mamba-3** | Gu 2024-2025 | ⭐⭐⭐⭐ | SSM 主线 |
| **RWKV-7** | RWKV 2025 | ⭐⭐⭐⭐ | linear attention 路线 |
| **Jamba-1.5** | AI21 2024 | ⭐⭐⭐⭐ | hybrid Mamba+attn+MoE |
| **MoR (Mixture of Recursions)** | 2025 | ⭐⭐⭐ | MoE 新方向 |

### 长上下文
| 方法 | 团队 | 影响 | 一句话 |
|------|------|------|--------|
| **YaRN** | Peng 2023 | ⭐⭐⭐⭐⭐ | RoPE scaling 事实标准 |
| **Ring Attention** | Liu 2023 | ⭐⭐⭐⭐⭐ | 1M context 基础 |
| **Infini-Attention** | Google 2024 | ⭐⭐⭐⭐ | compressive memory |
| Striped Attention | 2024 | ⭐⭐⭐ | Ring 改进 |
| **Gemini 2 (2M context)** | Google 2024 | ⭐⭐⭐⭐⭐ | 商业 long context 标杆 |

### 小模型
| 方法 | 团队 | 影响 | 一句话 |
|------|------|------|--------|
| **Phi-4 14B** | Microsoft 2024.12 | ⭐⭐⭐⭐⭐ | 教科书数据 |
| **Llama-3.2 1B/3B** | Meta 2024 | ⭐⭐⭐⭐⭐ | 边缘 SOTA |
| **Qwen3-0.6B/1.7B/4B** | Alibaba 2025 | ⭐⭐⭐⭐⭐ | 多语言小模型 |
| **SmolLM2** | HuggingFace 2024 | ⭐⭐⭐⭐ | 全开源 |
| **Gemma-3 4B** | Google 2025 | ⭐⭐⭐⭐ | Google 小模型 |
| **BitNet b1.58** | Microsoft 2024 | ⭐⭐⭐⭐⭐ | 1-bit 范式 |

### 训练 infra / 配方
| 方法 | 团队 | 影响 | 一句话 |
|------|------|------|--------|
| **Megatron-Core** | NVIDIA 2024 | ⭐⭐⭐⭐⭐ | 工业级框架 |
| **DeepSpeed Ulysses** | Microsoft | ⭐⭐⭐⭐ | sequence parallel |
| **MUP / μP** | Yang 2022 | ⭐⭐⭐⭐ | 超参 transfer |
| **WSD scheduler** | DeepSeek | ⭐⭐⭐⭐ | warmup-stable-decay |
| **Data annealing** | Llama-3 / Phi | ⭐⭐⭐⭐ | 末段高质量数据 |
| **Sophia optimizer** | Stanford 2023 | ⭐⭐⭐ | Hessian-aware |

### 数据 curation
| 方法 | 团队 | 影响 | 一句话 |
|------|------|------|--------|
| **FineWeb / FineWeb-Edu** | HuggingFace 2024 | ⭐⭐⭐⭐⭐ | 15T 开源数据 |
| **DCLM** | DataComp 2024 | ⭐⭐⭐⭐ | competition-driven |
| **Magpie** | 2024 | ⭐⭐⭐⭐ | 指令数据合成 |
| **SemDeDup** | Meta 2024 | ⭐⭐⭐⭐ | embedding 去重 |
| **MinHash + LSH** | 经典 | ⭐⭐⭐⭐⭐ | dedup 事实标准 |

---

## 十二、风险总览（系列级）

| 风险类别 | 具体风险 | 缓解策略 |
|---------|---------|---------|
| **计算资源** | 270M × 5B 预训需 24h+ | 提供 ckpt 下载，或租云 4×A100 ~$40 |
| | 多卡 infra 演示需 ≥ 2 卡 | 单卡 torchrun 模拟 + 云租用建议 |
| | MoE 真正 expert parallel 需 ≥ 4 卡 | 单卡 4 expert 演示，理论加 lecture |
| **数据准备** | CommonCrawl 一 dump 80TB | 用 1 segment ~200GB |
| | FineWeb 全集 15T | 子集 1B-token |
| | 法律 license | 注明 + 不提交输出 |
| **环境** | flash-attn 安装坑 | 提供 known-good combo |
| | Megatron-Core Windows 装不动 | 强制 WSL2（专题 3 起）|
| | mamba-ssm cuda 版本敏感 | Dockerfile 备份 |
| **学习负担** | 118 方法 / 122h 巨大 | 4 个月排期，每月一专题，可暂停 |
| | 专题 6 infra 抽象度高 | 提供云演示视频替代真跑 |
| | 专题 7 capstone 失败概率 | 提供 fallback ckpt |
| **算法陷阱** | MoE 路由崩塌 | Aux-Free / z-loss 单 lecture 讲 |
| | Loss spike 训挂 | 自动 rollback 实现 |
| | μP 超参 transfer 失败 | 配合 lecture 11 验证脚本 |

---

## 十三、复用模板说明

每专题统一目录结构（继承 Module 1+2）：

```
learning/<topic>/
├── README.md                # 标准模板
├── environment/
│   ├── requirements.txt
│   └── verify_env.py        # 三段式 (基础 / GPU / 库特定)
├── papers/
├── lectures/                # NN-method-name.md
├── src/
│   ├── common.py
│   ├── <method>_naive.py    # 手写
│   ├── <method>_lib.py      # 库对照
│   ├── <method>_industry.py # 工业级（Megatron/DeepSpeed）
│   └── tests/test_<method>_<type>.py
├── notebooks/
└── （docs/superpowers/specs/plans/ 放设计 + 计划）
```

文件命名规范同 RL 系列（NN-method-name.md / `<method>_{naive|lib|industry}.py`）。

---

## 十四、验证方法

每专题完成后验证清单：

```powershell
# 1. 环境验证
python learning/<topic>/environment/verify_env.py

# 2. 全部测试
python -m pytest learning/<topic>/src/tests/ -v

# 3. 全部 notebook
jupyter nbconvert --execute --inplace learning/<topic>/notebooks/*.ipynb

# 4. Capstone 单独验证
python learning/<topic>/src/capstone_*.py
```

系列级验证（专题 8 结束后）：

```powershell
# 五部曲毕业作品验证
python learning/small-model-graduation/src/capstone_five_module_graduation.py
# 预期：5 个 ckpt 全部加载，同一 GSM8K 题生成 5 个 response + 雷达图
```

---

## 十五、关键文件路径速查

### 现有模板参考
- `c:\Workspace\dummy\learning\multimodal-agent\README.md` — Module 2 收尾 README 模板
- `c:\Workspace\dummy\learning\reasoning-r1\README.md` — 高密度 R1 专题模板
- `c:\Workspace\dummy\docs\superpowers\specs\2026-06-03-rl-foundations-design.md` — Spec 起点参考
- `c:\Workspace\dummy\docs\superpowers\plans\2026-06-03-rl-foundations.md` — Plan 起点参考

### 即将新建（按专题）
- `c:\Workspace\dummy\learning\data-curation\` （专题 1）
- `c:\Workspace\dummy\learning\transformer-deep\` （专题 2）
- `c:\Workspace\dummy\learning\moe-architecture\` （专题 3）
- `c:\Workspace\dummy\learning\ssm-hybrid\` （专题 4）
- `c:\Workspace\dummy\learning\long-context\` （专题 5）
- `c:\Workspace\dummy\learning\scaling-infra\` （专题 6）
- `c:\Workspace\dummy\learning\pretraining-recipe\` （专题 7）
- `c:\Workspace\dummy\learning\small-model-graduation\` （专题 8）

### 即将新建（spec + plan，每专题各一对）
- `c:\Workspace\dummy\docs\superpowers\specs\2026-06-XX-data-curation-design.md` + plan
- （以此类推，共 16 个文档）

---

## 十六、本次任务的"下一步"

本规划完成后，下一轮可选：

### 选项 1：一口气完成式（同 Module 2 风格）
基于本规划文件，直接进入 8 专题快速实施：
- 每个专题：scaffold + 关键 src + capstone + 3-5 lectures + tests
- 完成后 tag → 进入下一专题
- 体量 ≈ Module 2（一次性 6-8 个 commit）

### 选项 2：spec/plan 全写式
先并行写完 8 份 spec design + 8 份 plan 作为完整蓝图，再实施。
- 体量 ≈ ~1500 行 spec + ~3000 行 plan

### 选项 3：单专题深入式
只挑 1-2 个最高优先级专题（推荐 `transformer-deep` + `moe-architecture`）做完整 spec + plan + 实施。

---

## 设计签字

- **设计日期**：2026-06-04
- **设计者**：Claude Opus 4.7
- **审阅者**：用户（待）
- **依赖前置**：Module 1 (PEFT 完成 ✓) + Module 2 (RL/对齐 完成 ✓)
- **后续依赖**：Module 4 (推理部署) / Module 5 (多模态生成) / Module 6 (Agent) 三选一作为收尾

🏗️ Module 3 蓝图就绪。
