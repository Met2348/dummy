# LoRA 家族学习专题

> 配套书籍：《大模型算法：强化学习、微调与对齐》（ISBN 9787121500725），2.x 章 LoRA 系列
> 设计文档：[`../../docs/superpowers/specs/2026-06-03-lora-family-design.md`](../../docs/superpowers/specs/2026-06-03-lora-family-design.md)
> 实施计划：[`../../docs/superpowers/plans/2026-06-03-lora-family.md`](../../docs/superpowers/plans/2026-06-03-lora-family.md)
> 承接专题：[`../prompt-tuning-family/README.md`](../prompt-tuning-family/README.md)

---

## 专题概览

本专题覆盖 **12 种 LoRA 家族微调方法**，按"主线 + 附录"的混合粒度组织为 **8 个 lecture**：

| Lecture | 主方法 | 附录方法 | 核心 idea |
|---------|--------|----------|----------|
| [01 LoRA](lectures/01-lora.md) | LoRA | rsLoRA, LoRA+ | $\Delta W = BA$ 低秩适配 |
| [02 AdaLoRA](lectures/02-adalora.md) | AdaLoRA | — | SVD 形式 + 重要性自适应分秩 |
| [03 PiSSA](lectures/03-pissa.md) | PiSSA | OLoRA | 用 $W_0$ 的 SVD 主成分初始化 |
| [04 VeRA](lectures/04-vera.md) | VeRA | — | 共享冻结 A/B，只学对角向量 |
| [05 LoHa+LoKr](lectures/05-loha-lokr.md) | LoHa, LoKr | — | Hadamard / Kronecker 积分解 |
| [06 QLoRA](lectures/06-qlora.md) | QLoRA | — | NF4 量化 + LoRA |
| [07 LoftQ](lectures/07-loftq.md) | LoftQ | — | 量化感知的迭代 SVD 初始化 |
| [08 DoRA](lectures/08-dora.md) | DoRA | — | 权重分解为 magnitude × direction |

## 学习路径（推荐）

```
01 LoRA → 02 AdaLoRA → 03 PiSSA → 04 VeRA
基础      [SVD 三剑客: 同一视角下的秩控制 / SVD 初始化 / 极致压缩]

       → 05 LoHa+LoKr → 06 QLoRA → 07 LoftQ → 08 DoRA
         分解形式        [量化二人组]            权重分解 (压轴)
```

每个 lecture 约 40 分钟阅读 + 30 分钟跑 notebook。完整专题 ~10 小时。

## 目录结构

```
learning/lora-family/
├── README.md                          # 本文件
├── environment/
│   ├── requirements.txt
│   └── verify_env.py                  # CPU + GPU + bitsandbytes 三段式自检
├── papers/                            # 12 篇原论文 PDF
├── lectures/                          # 8 篇 PPT-style 中文 md
├── src/                               # minimal + peft 代码
│   ├── common.py
│   ├── nf4_quant.py                   # ⭐ NF4 fake-quant 共享模块
│   ├── lora_minimal.py / lora_peft.py / lora_extensions.py
│   ├── adalora_minimal.py / adalora_peft.py
│   ├── pissa_minimal.py / pissa_peft.py / pissa_olora_extension.py
│   ├── vera_minimal.py / vera_peft.py
│   ├── loha_minimal.py / lokr_minimal.py / loha_lokr_peft.py
│   ├── qlora_minimal.py / qlora_peft.py
│   ├── loftq_minimal.py / loftq_peft.py
│   ├── dora_minimal.py / dora_peft.py
│   └── tests/                         # 一致性测试
└── notebooks/                         # 8 个 ipynb
```

## 环境配置

> **重要**：本仓库已在 RTX 5090 Laptop（Blackwell sm_120）验证。Blackwell 需要 PyTorch 编译时支持 sm_120，**cu126 不行**，必须用 cu130 或 nightly：

```powershell
# 1. 卸载旧 torch
pip uninstall -y torch torchvision torchaudio

# 2. 装 nightly cu130（Blackwell sm_120 兼容）
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# 3. 装其它依赖
pip install --upgrade peft bitsandbytes accelerate datasets scipy matplotlib

# 4. 验证
python learning/lora-family/environment/verify_env.py
```

预期输出（截至 2026-06-03）：

```
Part A (基础):           PASS    # torch 2.13.0.dev+cu130, transformers 5.9, peft 0.19, ...
Part B (GPU):            PASS    # RTX 5090, 25.7 GB, sm_120, GEMM ✅
Part C (bitsandbytes):   PASS    # bnb 0.49.2, NF4 quant-dequant ✅
```

非 Blackwell GPU（A100、H100、RTX 4090）用 stable cu126 即可：

```powershell
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
```

无 GPU：跳过 torch 重装，主线代码仍可在 CPU 上跑（QLoRA/LoftQ 的 GPU 选做 cell 会自动 SKIP）。

## 横向对比表（完整版）

### 1. 参数量与主战场

| 方法 | 年份 | $\Delta W$ 形式 | 参数 (GPT-2, $r=8$) | 占 base | 典型 $r$ | 主战场 |
|------|------|----------------|---------------------|---------|---------|--------|
| Full FT | — | $W \mathrel{+}= \Delta W$ | 124,439,808 | 100% | — | 通用（昂贵）|
| **LoRA** | 2021 | $BA$ | 294,912 | 0.236% | 4-64 | 通用 |
| **rsLoRA** | 2023 | $BA$ ($\alpha/\sqrt r$) | 294,912 | 0.236% | 64-256 | 大 $r$ 稳定 |
| **LoRA+** | 2024 | $BA$（$\eta_B = 16 \eta_A$） | 294,912 | 0.236% | 4-64 | 加速收敛 |
| **AdaLoRA** | 2023 | $P \Lambda Q^T$ + 重要性 | 442,512 | 0.354% | 12→8 | 自适应分秩 |
| **PiSSA** | 2024 | $BA$ (SVD top-r 初始化) | 294,912 | 0.236% | 8-128 | 加速收敛 |
| **OLoRA** | 2024 | $BA$ (QR top-r 初始化) | 294,912 | 0.236% | 8-128 | 正交初始化 |
| **VeRA** | 2024 | $\Lambda_d \odot B \Lambda_b A$（A、B 冻结共享）| 30,720 ($r=256$) | 0.025% | 256-1024 | 极致压缩 |
| **LoHa** | 2021 | $(B_1 A_1) \odot (B_2 A_2)$ | 589,824 | 0.472% | 4-8 | 高等效秩（$r^2$）|
| **LoKr** | 2023 | $B \otimes A$ | 23,808 (factor=32, $r=4$) | 0.019% | factor=8-48 | Stable Diffusion |
| **QLoRA** | 2023 | NF4($W$) + $BA$ | 294,912 | 0.236% | 64 | 大模型省显存 |
| **LoftQ** | 2023 | NF4($W - BA^*$) + $BA^*$ | 294,912 | 0.236% | 8-64 | 量化精度 |
| **DoRA** | 2024 | $\boldsymbol{m} \cdot \frac{W_0 + BA}{\|W_0+BA\|_c}$ | 304,128 | 0.244% | 8-64 | 接近全参 FT |

### 2. 训练特性

| 方法 | 学习率 | 训练复杂度 | 收敛速度 | 推理时延 | peft 支持 |
|------|--------|------------|---------|----------|-----------|
| LoRA | 1e-4 | 低 | 慢 | 0（合并）| ✅ `LoraConfig` |
| rsLoRA | 1e-4 | 低 | 慢 | 0 | ✅ `use_rslora` |
| LoRA+ | $\eta_B = 16 \eta_A$ | 低 | **快**（2× LoRA）| 0 | 需手写 optimizer |
| AdaLoRA | 5e-4 | 中（正交正则） | 中 | 0（剪枝后）| ✅ `AdaLoraConfig` |
| PiSSA | 1e-4 | 中（SVD 预处理）| **快**（2× LoRA） | 0 | ✅ `init_lora_weights="pissa"` |
| OLoRA | 1e-4 | 中（QR 预处理） | 中 | 0 | ✅ `init_lora_weights="olora"` |
| VeRA | 1e-2 | 低 | 中 | 0 | ✅ `VeraConfig` |
| LoHa | 1e-4 | 中（双路径） | 中 | 0 | ✅ `LoHaConfig`（不支持 GPT-2 Conv1D）|
| LoKr | 1e-4 | 中（Kronecker） | 中 | 0 | ✅ `LoKrConfig`（不支持 GPT-2 Conv1D）|
| QLoRA | 2e-4 | 中（运行时反量化） | 慢 | 反量化开销 5-10% | ✅ + bitsandbytes |
| LoftQ | 2e-4 | 中（T 次 SVD 预处理）| **快** | 反量化开销 | ✅ `init_lora_weights="loftq"` |
| DoRA | 1e-4 | 中（norm + detach） | **快** | 0 | ✅ `use_dora=True` |

### 3. 一致性测试矩阵（minimal vs peft）

| 方法 | 测试类型 | logits 误差 | 状态 |
|------|---------|-------------|------|
| LoRA | 强一致 | **0.00e+00** | ✅ bit 精确 |
| rsLoRA | 单元（scaling 公式） | — | ✅ |
| LoRA+ | 单元（学习率比例） | — | ✅ |
| AdaLoRA | 弱一致（实现细节差异） | — | ✅ 6 单元测试 |
| PiSSA | 强一致 (重建 W_0) | **1.19e-07** | ✅ 浮点 round-off |
| OLoRA | 强一致 (重建 W_0) | **1.19e-07** | ✅ |
| VeRA | 强一致 (共享 A/B) | — | ✅ data_ptr 共享验证 |
| LoHa | 强一致 (rank ≤ r²) | **rank=16 = r²** | ✅ |
| LoKr | 强一致 (Kronecker shape) | — | ✅ |
| NF4 fake-quant | GPU 对比 bitsandbytes | **0.0** | ✅ 完美一致 |
| QLoRA | 强一致 (训练时 base 不变) | **0.0** | ✅ |
| LoftQ | 单调收敛 | 25.49 → 23.95 | ✅ |
| DoRA | 强一致 (初始 W = W_0) | **0.0** | ✅ |

### 4. 工程选型决策树

```
你的问题是什么？
│
├─ 65B 大模型 + 24GB GPU？        → QLoRA ⭐⭐⭐
├─ 极致省参数 (< 1K per layer)？  → VeRA / Prompt Tuning
├─ Stable Diffusion 风格微调？    → LoKr ⭐⭐⭐
├─ 追求接近全参 FT 质量？         → DoRA ⭐⭐⭐
├─ 自适应秩分配？                 → AdaLoRA
├─ 快速收敛 (科研迭代)？          → PiSSA ⭐⭐
├─ 量化 + 高质量初始化？          → LoftQ
├─ 多任务（1000 用户/任务）？     → VeRA + 共享 A/B
├─ 大 r 训练不稳？                → rsLoRA ($\alpha/\sqrt r$)
└─ 训练慢？                       → LoRA+ ($\eta_B \gg \eta_A$)
```

## 全专题学习目标自测

读完本专题后，你应该能回答：

1. **公式题**：写出 LoRA 公式 $h = W_0 x + \frac{\alpha}{r} BA x$ 的逐项含义（$B$、$A$、$\alpha$、$r$ 各代表什么）
2. **公式题**：证明 $\Delta W = BA$（LoRA）与 $\Delta W = P \Lambda Q^T$（AdaLoRA）在 $r$ 相同下表达力等价
3. **公式题**：推导 PiSSA 的 $B = U_{:r} \sqrt{\Sigma_{:r}}, A = \sqrt{\Sigma_{:r}} V_{:r}^T$ 满足 $BA = W_0^{\text{top}r}$
4. **公式题**：解释 NF4 在 $\mathcal{N}(0, 1)$ 输入上比 INT4 误差小（信息论角度）
5. **公式题**：写出 DoRA 的反向公式 $\frac{\partial \mathcal{L}}{\partial \boldsymbol{m}}$
6. **设计题**：为什么 LoKr 在 Stable Diffusion 上比 NLP 更有效？
7. **设计题**：DoRA 的 magnitude 用 column-wise 还是 row-wise？两者的取舍
8. **对比题**：在何种场景下 VeRA 比 LoRA 更优？反之呢？
9. **对比题**：QLoRA vs LoftQ 在 7B vs 65B 模型上的差异为什么不同
10. **实践题**：在 LLaMA-2-7B 上估算 QLoRA + DoRA (QDoRA) 的显存占用

## 运行验证（Runbook）

> 本专题的"可运行入口"即 [`runbook.yaml`](runbook.yaml) 登记的 **19 个 minimal/peft 直跑 demo**（12 种方法的手写实现 + peft 调包对照 + NF4 共享模块）。均无需传参，自带 smoke 规模（GPT-2 / TinyLlama）。已在 ERIC-3080Ti（RTX 3080 Ti Laptop 16GB，torch 2.11+cu128 / peft 0.19.1 / bitsandbytes 0.49.2）**V1 全部验证通过（19/19）**。
> 一键复验：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules lora-family --timeout 600
> ```

每种方法都可直接跑（`*_minimal.py` = 手写最小实现，`*_peft.py` = peft 调包对照）：

```powershell
# 01 LoRA（+ rsLoRA / LoRA+）
python learning/lora-family/src/lora_minimal.py
python learning/lora-family/src/lora_peft.py
python learning/lora-family/src/lora_extensions.py        # rsLoRA / LoRA+ 扩展
# 02 AdaLoRA
python learning/lora-family/src/adalora_minimal.py
python learning/lora-family/src/adalora_peft.py
# 03 PiSSA（+ OLoRA）
python learning/lora-family/src/pissa_minimal.py
python learning/lora-family/src/pissa_peft.py
# 04 VeRA
python learning/lora-family/src/vera_minimal.py
python learning/lora-family/src/vera_peft.py
# 05 LoHa + LoKr
python learning/lora-family/src/loha_minimal.py
python learning/lora-family/src/lokr_minimal.py
python learning/lora-family/src/loha_lokr_peft.py         # GPT-2 Conv1D 预期 ValueError（仅 nn.Linear 支持）
# NF4 共享量化模块（QLoRA/LoftQ 依赖）
python learning/lora-family/src/nf4_quant.py              # 纯数值自检，CPU 秒级
# 06 QLoRA
python learning/lora-family/src/qlora_minimal.py          # NF4 fake-quant，CPU/GPU 都跑
python learning/lora-family/src/qlora_peft.py             # bitsandbytes 真 4-bit + TinyLlama，GPU only
# 07 LoftQ
python learning/lora-family/src/loftq_minimal.py
python learning/lora-family/src/loftq_peft.py             # 需 scipy
# 08 DoRA
python learning/lora-family/src/dora_minimal.py
python learning/lora-family/src/dora_peft.py
```

**关键坑注记**：

- **QLoRA + bitsandbytes**：`qlora_peft.py` 用 `bitsandbytes` 真 4-bit NF4，**需 GPU**（4-bit 不支持 CPU）。本机 RTX 3080 Ti（Ampere sm_86）实测可跑（TinyLlama-1.1B 加载 + 前向通过）。无 GPU / 无 bitsandbytes 时它会**显式 SKIP**（提示改跑 `qlora_minimal.py` 的 fake-quant 版），真量化失败则 **fail-fast（非零退出）**——不再静默 print 成功（杜绝"假成功"反模式）。
- **LoftQ**：`loftq_peft.py` 依赖 `scipy`；初始化失败现已 **fail-fast** 而非静默 return。
- **LoHa / LoKr**：peft 的 `LoHaConfig`/`LoKrConfig` 不支持 GPT-2 的 `Conv1D`，`loha_lokr_peft.py` 故意捕获并打印**预期的** `ValueError`（这是 demo 的教学点，非错误）；在 LLaMA/BERT 等 `nn.Linear` 模型上可直接用。
- 所有 `*_minimal.py` 会下载 `gpt2`（首次 ~500MB），首跑稍慢属正常。

**测试（V2）**：10 个一致性测试校验 minimal 与 peft 数值对齐（本套件最重，~260s）：

```powershell
python scripts/eric_3080ti_env_audit.py --modules lora-family --tests --timeout 600
# 或单独：python learning/lora-family/src/tests/test_lora_consistency.py 等
```

> 环境自检（CPU + GPU + bitsandbytes 三段式）：`python learning/lora-family/environment/verify_env.py`

## 与 prompt-tuning-family 的衔接（跨专题对比）

LoRA 家族与上一专题（prompt-tuning-family）形成"两条主线"：

- **prompt-based**（输入侧）：Prompt Tuning、Prefix Tuning、P-Tuning v1/v2
- **weight-based**（权重侧，本专题）：LoRA、AdaLoRA、PiSSA、VeRA、LoHa、LoKr、QLoRA、LoftQ、DoRA

### 跨专题工程选型（meta-comparison）

| 问题 | 选 prompt-based | 选 weight-based |
|------|-----------------|-----------------|
| **65B 大模型 + 24GB GPU 微调** | ❌（无量化路径） | ⭐ QLoRA |
| **NLU 分类（小模型 RoBERTa）** | ⭐ P-Tuning v2 | LoRA 也可 |
| **NER 序列标注** | ⭐ P-Tuning v2（独家） | LoRA |
| **NLG 生成（GPT-2/3）** | Prefix Tuning | ⭐ LoRA / DoRA |
| **极致省参数（< 1K per layer）** | ⭐ Prompt Tuning（7.7K 全模型）| VeRA（30K，但参数效率更高）|
| **不占 context window** | ❌（必占 prompt 长度）| ⭐ 全部 |
| **可合并权重 0 推理时延** | ❌ | ⭐ LoRA/PiSSA/DoRA |
| **多任务（千个用户）** | Prompt Tuning（仅 input emb）| ⭐ VeRA（共享 A/B）|

### 选哪条主线？

- **Prompt-based 主线**适合：参数极致少、prompt 可解释性强、只在 input 侧加工
- **Weight-based 主线**适合：性能优先、可合并权重、量化兼容、工程友好

实践推荐：**先 LoRA**，慢的话上 **PiSSA**，大模型必上 **QLoRA**，质量优先用 **DoRA**。

## 下一步可学的专题

- **长上下文**：LongLoRA / PI / YaRN（4k → 128k context）
- **对齐**：RLHF / DPO / SimPO（回到书本主线 §3）
- **混合专家**：MoE / Mixtral / DeepSeek
- **推理优化**：vLLM / FlashAttention / Continuous Batching

## Git 里程碑

| Tag | Commit | 内容 |
|-----|--------|------|
| `lora-base` | `b464682` | LoRA + rsLoRA + LoRA+（含 mini training）|
| `lora-svd-trio` | `e1dde69` | AdaLoRA + PiSSA + VeRA（SVD 三剑客）|
| `lora-shape` | `ae74fcd` | LoHa + LoKr（Hadamard / Kronecker）|
| `lora-quant` | `23d45a7` | QLoRA + LoftQ（量化二人组）|
| `lora-family-complete` | _本次_ | DoRA + 完整 README |
