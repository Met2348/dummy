# Transformer Deep 学习专题 — 设计文档

> **承接**: 专题 1 data-curation（提供 1B-token 自制语料 + 32k SP tokenizer）
> **本专题**: Module 3 第 2 站 — Transformer 架构骨架完整版
> **战略地位**: 后续 MoE / SSM / 长上下文 / 预训练 全部依赖本专题"现代 Transformer"基础
> **总体规划**: `docs/superpowers/plans/2026-06-04-pretraining-architecture-series.md`

---

## 1. 专题定位

GPT-2 → Llama-3 → DeepSeek-V3 的架构演化全图。每个组件**单文件实现**，便于横向对比和组合。结尾用所有现代组件搭一个 80M GPT-mini。

### 1.1 为什么这是 Module 3 第 2 站

- 数据 ready 后下一步必须造模型骨架
- 现代 LLM = vanilla Transformer + N 个小改动（RoPE / RMSNorm / GQA / SwiGLU / ...）
- 后续专题（MoE / SSM / 长上下文）全部在本架构上"加层"或"换块"

### 1.2 本专题的"硬核度"

- **数学 40%**: RoPE 复数旋转推导、attention 数值稳定性、Flash 算法
- **工程 40%**: kernel / 显存 / KV cache
- **架构 20%**: MLA / MoE 等 SOTA 组件横向对比

---

## 2. 方法清单（16 种）

| # | 方法 | 年份 | 论文/出处 | 核心 idea |
|---|------|------|---------|----------|
| 1 | **Vanilla Transformer** | 2017 | Vaswani | Attention is All You Need |
| 2 | **Sinusoidal PE** | 2017 | 同上 | 正弦位置编码 |
| 3 | **Learned PE** | 2018 | BERT | 可学习位置编码 |
| 4 | **RoPE** | 2021 | Su (RoFormer) | 复数旋转相对位置 ⭐ |
| 5 | **ALiBi** | 2021 | Press | attention bias 线性衰减 |
| 6 | **MHA** | 2017 | — | multi-head attention |
| 7 | **MQA** | 2019 | Shazeer | 单 KV 头 |
| 8 | **GQA** | 2023 | Ainslie | group KV head（Llama-3 用） |
| 9 | **MLA** | 2024 | DeepSeek-V2 | low-rank KV 压缩 ⭐ |
| 10 | **LayerNorm** | 2016 | Ba | 标准 |
| 11 | **RMSNorm** | 2019 | Zhang | 简化 LN（Llama 用）|
| 12 | **Pre vs Post Norm** | 2020 | — | Pre-Norm 训练稳定 |
| 13 | **SwiGLU** | 2020 | Shazeer | gated GLU（Llama 用）|
| 14 | **FlashAttention v1** | 2022 | Dao | tiling + online softmax ⭐ |
| 15 | **FlashAttention v2/v3** | 2023-2024 | Dao | warp / TMA / FP8 |
| 16 | **PagedAttention** | 2023 | Kwon (vLLM) | KV cache 分页 |

---

## 3. Lecture 结构（14 篇 = 13 主线 + 1 capstone）

| Lecture | 主题 | 主方法 | 时长 |
|---------|------|--------|------|
| **L01** Vanilla Transformer 复习 | 2017 原论文 | 60 min |
| **L02** 位置编码全套 | Sinusoidal / Learned / RoPE / ALiBi / NoPE | 90 min |
| **L03** RoPE 数学推导 ⭐ | 复数旋转 + 推外能力 | 90 min |
| **L04** Attention 变体 | MHA / MQA / GQA / MLA | 90 min |
| **L05** Normalization 家族 | LN / RMSNorm / Pre vs Post / DeepNorm | 60 min |
| **L06** 激活函数演化 | ReLU / GELU / SwiGLU / GeGLU | 60 min |
| **L07** FlashAttention v1 ⭐ | tiling + online softmax | 120 min |
| **L08** FA v2/v3 改进 | warp specialization / TMA / FP8 | 75 min |
| **L09** PagedAttention | KV cache 分页 (vLLM) | 60 min |
| **L10** Sliding Window | Mistral / Gemma 长上下文方案 | 60 min |
| **L11** 架构搜索 | μP / hyperparam transfer | 60 min |
| **L12** DeepSeek-V3 精读 ⭐ | MLA + DeepSeekMoE + Aux-Free | 90 min |
| **L13** Llama-3 精读 | GQA 8/64 + 128k | 75 min |
| **L14** Capstone：80M GPT-mini | 现代组件全集成 | 120 min |

**总学时**: 14 lecture × 平均 80 min + 7h notebook ≈ 16 hours

---

## 4. Lecture 模板

```markdown
# Lecture N: {组件名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 动机（这个组件解决什么）
## Slide 3-6: 核心公式 / 算法
## Slide 7-10: 直觉解释 + 图解
## Slide 11-15: 与前代对比（vanilla vs 该方法）
## Slide 16-20: 代码逐行（minimal vs 库 vs 工业）
## Slide 21-24: 实验：显存 / 速度 / ppl 对比
## Slide 25-28: 陷阱与思考题
```

---

## 5. 代码三轨策略

| 方法 | 手写 minimal | 库 (flash-attn / triton) | 工业级 |
|------|-------------|------------------------|--------|
| RoPE | ✅ | ✅ flash_attn.rope | — |
| MHA / MQA / GQA / MLA | ✅ 四独立文件 | — | (Llama 内置) |
| RMSNorm | ✅ | ✅ torch.nn.RMSNorm | — |
| SwiGLU | ✅ | — | (Llama 内置) |
| FA v1 naive | ✅ Triton | ✅ flash-attn | — |
| FA v2 / v3 | — | ✅ flash-attn | (H100 only) |
| Paged Attention | ✅ 教学版 | — | ✅ vLLM |
| Sliding Window | ✅ | ✅ flash-attn | — |
| GPT-mini 集成 | ✅ | — | — |

---

## 6. 一致性测试

```python
def test_rope_naive_vs_flash():        # < 1e-4 numerical
def test_attention_variants_equiv():    # MHA / GQA / MQA 同 input 输出一致 with proper kv_dup
def test_rmsnorm_vs_torch():            # < 1e-5
def test_flash_naive_vs_lib():          # FA1 naive Triton vs flash-attn 库 < 1e-4
def test_gpt_mini_forward_shape():       # batch / seq / vocab 维度正确
def test_gpt_mini_grad_flow():          # 反向梯度无 NaN
def test_kv_cache_correctness():        # incremental 解码 vs 一次性 一致
```

---

## 7. Notebook 结构（14 个）

每个 lecture 一个 ipynb：
1. import + 模型组件初始化
2. 公式可视化（RoPE 旋转图 / softmax 数值稳定性）
3. minimal 实现 step-by-step
4. mini benchmark（128/512/2048 序列长度）
5. 库对照
6. 显存 / 速度对比
7. 思考题 + 下节预告

---

## 8. 环境配置

```
torch>=2.5+cu130
flash-attn>=2.6
triton>=3.0  # naive kernel 教学
einops>=0.8
transformers>=5.0  # 加载 Llama / DeepSeek 对照
```

**verify_env.py 三段式**:
- Part A: 基础（torch + flash-attn + triton）
- Part B: GPU + sm_120
- Part C: GPT-mini 单次 forward smoke

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `trans-pe` | L01-L03: PE 全套 + RoPE 推导 | 4 |
| `trans-attn-variants` | L04: MHA/MQA/GQA/MLA | 4 |
| `trans-norm-act` | L05-L06: Norm + Act | 2 |
| `trans-flash` | L07-L08: FA1 naive + FA2/3 | 4 |
| `trans-paged` | L09-L10: Paged + SWA | 2 |
| `trans-modern` | L11-L13: μP + DeepSeek-V3 + Llama-3 | 3 |
| `transformer-deep` | L14: 80M GPT-mini + README | 3 |

---

## 10. 跨专题衔接

### 上游
- 专题 1 data-curation：提供 1B-token 语料 + 32k SP tokenizer

### 下游
- 专题 3 MoE：把 GPT-mini MLP 换成 MoE
- 专题 4 SSM：替换 attention 为 Mamba block
- 专题 5 长上下文：扩 RoPE base
- 专题 6 infra：用 GPT-mini 作 FSDP demo
- 专题 7 预训练：基于 GPT-mini 扩到 270M Phi-tiny

### 跨专题对照表预留位
| 架构组件 | 引入年份 | 用途 |
|---------|---------|------|
| RoPE | 2021 | 所有现代 LLM |
| GQA | 2023 | Llama-3 / DeepSeek |
| MLA | 2024 | DeepSeek-V2/V3 |
| SwiGLU | 2020 | Llama / PaLM |
| FA1/2/3 | 2022+ | 全行业 |

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| flash-attn 安装坑（cuda 版本） | 高 | 高 | 提供 known-good combo + Dockerfile 备份 |
| FA3 需 H100 | 高 | 中 | 仅放代码 + 性能图，不强制运行 |
| naive Triton FA 精度 | 中 | 中 | 测试用 1e-4 容差 |
| MLA 数学复杂 | 中 | 低 | L04 单独 30 min 推导视频 |
| 80M GPT-mini 训练时长 | 中 | 中 | 1 epoch ~6h，提供训好 ckpt 下载 |
| KV cache 实现易错 | 中 | 高 | 增量 vs 一次性输出严格测试 |

---

## 12. 论文 / 资料占位

```
papers/
├── 01-vaswani-2017-attention.md
├── 02-su-2021-rope.md                # RoPE 原论文
├── 03-press-2021-alibi.md
├── 04-shazeer-2019-mqa.md
├── 05-ainslie-2023-gqa.md
├── 06-deepseek-2024-mla.md           # DeepSeek-V2 MLA
├── 07-zhang-2019-rmsnorm.md
├── 08-xiong-2020-prenorm.md
├── 09-shazeer-2020-swiglu.md
├── 10-dao-2022-flash-v1.md
├── 11-dao-2023-flash-v2.md
├── 12-shah-2024-flash-v3.md
├── 13-kwon-2023-vllm-paged.md
├── 14-yang-2022-mup.md
├── 15-deepseek-v3-2024.md            # 完整架构精读
├── 16-llama3-2024.md
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-04-transformer-deep.md` 的 7 个 Phase 推进：

- Phase 1: 基础设施
- Phase 2: L01-L03 PE 全套 + RoPE 推导（tag `trans-pe`）
- Phase 3: L04 Attention 变体（tag `trans-attn-variants`）
- Phase 4: L05-L06 Norm + Act（tag `trans-norm-act`）
- Phase 5: L07-L08 FlashAttention（tag `trans-flash`）
- Phase 6: L09-L11 Paged + SWA + μP（tag `trans-paged` + `trans-modern`）
- Phase 7: L12-L14 DeepSeek-V3 + Llama-3 + Capstone（tag `transformer-deep`）

---

## 设计签字

- **设计日期**: 2026-06-04
- **设计者**: Claude Opus 4.7
- **审阅者**: 用户（待）
