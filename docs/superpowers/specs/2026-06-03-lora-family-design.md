# LoRA 家族学习专题设计文档

> **作者**: Claude (协同用户)
> **日期**: 2026-06-03
> **状态**: 设计已批准，等待 implementation plan
> **承接**: `2026-06-02-prompt-tuning-family-design.md`（同系列第二个专题）

## 1. 背景与目标

用户正在阅读《大模型算法：强化学习、微调与对齐》（ISBN 9787121500725），刚学完 prompt-based 微调（专题 1：prompt-tuning-family），现在进入 LoRA 系列（专题 2）。

**学习目标**：

1. 完整理解 12 种 LoRA 家族方法的核心思想、公式、与 LoRA 的差异
2. 每种方法都有可运行的 minimal 手写实现 + peft 调包对照 + 一致性验证
3. 量化部分（QLoRA、LoftQ）通过 fake-quant 跨平台展示 NF4 算法逻辑，并在 GPU 上验证与 bitsandbytes 真量化的一致性
4. 形成一张全 12 方法的横向对比表，作为 LoRA 系列工程选型参考

**非目标**：

- 不复现论文 SOTA benchmark（GPT-2 base 上没意义）
- 不做 65B 大模型微调（硬件限制）
- 不深入 PEFT 库的实现细节（够用就好，重点是算法理解）

## 2. 方法范围与 Lecture 映射

12 个方法按"主线 + 附录"的混合粒度组织为 8 个 lecture：

| Lecture | 主方法 | 附录方法 | arXiv | 年份 |
|---------|--------|----------|-------|------|
| 01 LoRA | LoRA | rsLoRA, LoRA+ | 2106.09685 | 2021 |
| 02 AdaLoRA | AdaLoRA | — | 2303.10512 | 2023 |
| 03 PiSSA | PiSSA | OLoRA | 2404.02948 | 2024 |
| 04 VeRA | VeRA | — | 2310.11454 | 2024 |
| 05 LoHa+LoKr | LoHa, LoKr | — | 2108.06098 / FedPara (2108.06098 的衍生) | 2021/23 |
| 06 QLoRA | QLoRA | — | 2305.14314 | 2023 |
| 07 LoftQ | LoftQ | — | 2310.08659 | 2023 |
| 08 DoRA | DoRA | — | 2402.09353 | 2024 |

**顺序逻辑（"学习曲线"）**：

```
01 LoRA  ──▶  02 AdaLoRA ──▶ 03 PiSSA ──▶ 04 VeRA
基础           [SVD trio: 同一视角下的秩控制 / SVD 初始化 / 极致压缩]

      ──▶  05 LoHa+LoKr ──▶ 06 QLoRA ──▶ 07 LoftQ ──▶ 08 DoRA
            分解形式           [量化二人组]            权重分解 (压轴)
```

**附录方法整合规则**：

- **rsLoRA**（"$\alpha/r$ → $\alpha/\sqrt r$" 的 scaling 修正）：在 LoRA lecture 末尾加 4-5 张幻灯片
- **LoRA+**（"$\eta_B \gg \eta_A$" 的优化器修正）：在 LoRA lecture 末尾加 4-5 张幻灯片
- **OLoRA**（"QR 正交初始化"）：在 PiSSA lecture 末尾加 4-5 张幻灯片（同属"初始化优化"）

附录方法的代码作为主方法 `*_minimal.py` 的扩展（`lora_extensions.py`、`pissa_olora_extension.py`），不单独建立完整的 minimal/peft 文件。

## 3. 目录结构

```
learning/lora-family/
├── README.md                          # 总览 + 全 12 方法横向对比表 + 学习路径
├── environment/
│   ├── requirements.txt               # CUDA torch、peft>=0.13、bitsandbytes、...
│   └── verify_env.py                  # CPU + GPU + bitsandbytes 三段式自检
├── papers/
│   ├── 01-lora-2021.pdf
│   ├── 02-adalora-2023.pdf
│   ├── 03-pissa-2024.pdf
│   ├── 04-vera-2024.pdf
│   ├── 05a-loha-2021.pdf
│   ├── 05b-lokr-2023.pdf
│   ├── 06-qlora-2023.pdf
│   ├── 07-loftq-2023.pdf
│   ├── 08-dora-2024.pdf
│   ├── 附-rslora-2023.pdf
│   ├── 附-lora-plus-2024.pdf
│   └── 附-olora-2024.pdf
├── lectures/                          # 8 个 PPT-style 中文 lecture（每篇 28 张主 + 附录张数）
│   ├── 01-lora.md
│   ├── 02-adalora.md
│   ├── 03-pissa.md
│   ├── 04-vera.md
│   ├── 05-loha-lokr.md
│   ├── 06-qlora.md
│   ├── 07-loftq.md
│   └── 08-dora.md
├── src/
│   ├── common.py                      # 复用上次工具 + freeze_base + apply_lora_to 等通用工具
│   ├── nf4_quant.py                   # ⭐ NF4 fake-quant 共享模块（QLoRA/LoftQ 复用）
│   ├── lora_minimal.py                # 公式 (1): h = Wx + α/r * BAx
│   ├── lora_peft.py                   # peft.LoraConfig
│   ├── lora_extensions.py             # rsLoRA + LoRA+ 在 lora_minimal 之上加 8 行
│   ├── adalora_minimal.py             # 公式: ΔW = PΛQ^T + 重要性打分
│   ├── adalora_peft.py
│   ├── pissa_minimal.py               # 公式: A、B 取 W 的 SVD top-r
│   ├── pissa_peft.py
│   ├── pissa_olora_extension.py       # OLoRA: A、B 用 QR
│   ├── vera_minimal.py                # 公式: ΔW = Λ_d B Λ_b A, A/B 共享冻结随机
│   ├── vera_peft.py
│   ├── loha_minimal.py                # 公式: ΔW = (B_1 A_1) ⊙ (B_2 A_2)
│   ├── lokr_minimal.py                # 公式: ΔW = B ⊗ A (Kronecker)
│   ├── loha_lokr_peft.py
│   ├── qlora_minimal.py               # 公式: dequant(NF4(W)) + α/r BA, 用 nf4_quant.py
│   ├── qlora_peft.py                  # bitsandbytes 真量化（GPU only）
│   ├── loftq_minimal.py               # 公式: W ≈ NF4(W) + UΣV^T（量化误差用 LoRA 补偿）
│   ├── loftq_peft.py
│   ├── dora_minimal.py                # 公式: W = m · V/||V||, V = W_0 + BA
│   ├── dora_peft.py
│   └── tests/
│       ├── test_lora_consistency.py
│       ├── test_lora_extensions.py    # rsLoRA scaling 单元 + LoRA+ 学习率验证
│       ├── test_adalora_consistency.py
│       ├── test_pissa_consistency.py
│       ├── test_vera_consistency.py
│       ├── test_loha_lokr_consistency.py
│       ├── test_nf4_quant.py          # ⭐ NF4 fake-quant 算法验证（量化-反量化误差曲线）
│       ├── test_qlora_consistency.py
│       ├── test_loftq_consistency.py
│       └── test_dora_consistency.py
└── notebooks/                         # 8 个 ipynb（含选做 GPU cell）
    ├── 01-lora.ipynb
    ├── 02-adalora.ipynb
    ├── 03-pissa.ipynb
    ├── 04-vera.ipynb
    ├── 05-loha-lokr.ipynb
    ├── 06-qlora.ipynb                 # ⭐ 含 bitsandbytes + TinyLlama-1.1B GPU 选做 cell
    ├── 07-loftq.ipynb                 # ⭐ 含 GPU 选做 cell
    └── 08-dora.ipynb
```

## 4. Lecture 模板（沿用 prompt-tuning-family）

每个 lecture 28 张主幻灯片，结构稳定（方便横向对比）：

| 幻灯片号 | 内容 |
|----------|------|
| 1 | 封面与导读（论文标题、作者、arXiv、本节回答的 4 个问题、学习建议） |
| 2 | 符号速查表（与上次对齐：$L, H, d, d_h, n, r, \alpha, $ 等） |
| 3-5 | 上下文与历史（与前序方法的关系、本方法解决的痛点） |
| 6-10 | 核心思想 + 公式 (1)(2)(3) **逐项重述每个符号**（解决用户"易遗忘符号"问题） |
| 11-13 | 与前序方法对比（公式 diff + 概念 diff） |
| 14-16 | 架构示意图（Mermaid）+ 张量形状追踪 |
| 17-21 | 实验设置 + 关键实验结果（论文 Table/Figure 摘要 + 解读） |
| 22-23 | 优缺点 + 适用边界 |
| 24-25 | 横向对比表（全 12 方法的本期更新） |
| 26-27 | minimal 代码片段 + peft 调包对照 |
| 28 | 思考题（4-6 道，覆盖公式、代码、设计选择） |

**附录方法（rsLoRA / LoRA+ / OLoRA）的额外幻灯片**：附在 28 张主幻灯片之后，编号 29-33 左右。

## 5. 代码规范

### 5.1 文件命名与组织

- 每个方法：`{method}_minimal.py`（手写）+ `{method}_peft.py`（peft 调包）
- 附录方法：`{main}_{appendix}_extension.py`（如 `pissa_olora_extension.py`）
- 测试：`tests/test_{method}_consistency.py`
- Notebook：`{NN}-{method}.ipynb`

### 5.2 minimal 实现的硬性要求

- 单文件、自包含，每行 ≤ 100 字符
- 用 `nn.Module` 包装，构造函数中冻结 base 参数
- 公式逐项注释（公式号 + 数学表达 + 行内符号说明）
- 模型用 GPT-2 base（与上次一致，CPU 可跑，GPU 加速）
- 量化方法用 `nf4_quant.py` 共享模块

### 5.3 peft 实现的硬性要求

- 直接 `peft.get_peft_model(base, config)`
- 探测内部参数布局（用 `model.named_parameters()` 自检）
- 与 minimal 做 forward consistency 测试

### 5.4 代码深度策略

| 方法 | Forward 一致性 | mini training | GPU 选做 |
|------|---------------|---------------|----------|
| LoRA | ✅ | ❌ | — |
| **rsLoRA** | ✅ | ✅ ~50 step | — |
| **LoRA+** | ✅ | ✅ ~50 step | — |
| **AdaLoRA** | ✅ | ✅ ~50 step + 重要性可视化 | — |
| PiSSA | ✅ | ❌ | — |
| OLoRA | ✅ | ❌ | — |
| VeRA | ✅ | ❌ | — |
| LoHa | ✅ | ❌ | — |
| LoKr | ✅ | ❌ | — |
| **QLoRA** | ✅（fake-quant） | ❌ | ✅ bitsandbytes + TinyLlama |
| **LoftQ** | ✅（fake-quant） | ❌ | ✅ bitsandbytes 验证 |
| DoRA | ✅ | ❌ | — |

mini training：CPU 跑 ~50 step（GPU 自动加速），Adam 优化器，IMDb 子集或 ToySentimentDataset 上验证 loss 下降，画 matplotlib 曲线。

GPU 选做：QLoRA/LoftQ notebook 末尾加 cell，需要 GPU + bitsandbytes 才能跑。如果环境不就绪，notebook 输出 SKIP 提示。

## 6. NF4 Fake-Quant 共享模块

`src/nf4_quant.py` 是本专题的关键工程组件，QLoRA 和 LoftQ 都依赖它。

### 6.1 算法要点（与 QLoRA 论文 §3 对齐）

**NF4 (NormalFloat 4-bit)**：将服从 $\mathcal{N}(0, 1)$ 的权重映射到 16 个值的查找表（来自 normal 分布的 16 分位点），相比 INT4 在正态权重上有更小的量化误差。

**Double Quantization**：把 NF4 的 per-block scale（fp32）再用 8-bit 量化（每 256 个 scale 共享一个 fp32 outer scale），进一步省存。

**Fake-quant 实现策略**：

```python
def nf4_quantize_dequantize(W: Tensor, block_size: int = 64) -> Tensor:
    """前向：模拟 NF4 量化误差；反向：STE 让梯度直接穿过。"""
    # 1. blockwise normalize: scale = W.abs().max(dim=-1) per block
    # 2. quantize: 把 W/scale 映射到最近的 NF4 网格点
    # 3. dequantize: 反查找表 × scale
    # 4. STE: return W + (W_dequant - W).detach()
```

### 6.2 一致性测试设计（`test_nf4_quant.py`）

- **单元 1**：NF4 网格点正确性（与论文 §3.2 表对齐）
- **单元 2**：量化-反量化对 $\mathcal{N}(0, 1)$ 输入的相对误差 < 1%
- **单元 3**：与 bitsandbytes 真量化在 GPU 环境下的 logits 一致性（GPU 选做）

## 7. 一致性测试策略

继续沿用上次的"强一致 / 弱一致"二分：

| 类型 | 适用 | 容差 | 例子 |
|------|------|------|------|
| **强一致** | 算法确定（无 LSTM、无 dropout、无随机初始化或可对齐初始化）| `< 1e-4`（logits bit 精确）| LoRA、NF4 fake-quant、PiSSA 初始化、VeRA 共享投影 |
| **弱一致** | 有随机性或实现细节差异 | 相对误差 `< 1e-2` 或 loss 趋势一致 | AdaLoRA 重要性打分排序、QLoRA fake vs 真 NF4 |

**测试矩阵汇总**：

```
test_lora_consistency.py             强一致 ✅
test_lora_extensions.py              单元 ✅
test_adalora_consistency.py          弱一致（重要性打分有随机）
test_pissa_consistency.py            强一致 ✅（SVD 确定）
test_vera_consistency.py             强一致 ✅（共享投影固定 seed）
test_loha_lokr_consistency.py        强一致 ✅
test_nf4_quant.py                    单元 ✅
test_qlora_consistency.py            强一致（fake-quant vs fake-quant）+ 弱一致（GPU 选做）
test_loftq_consistency.py            弱一致
test_dora_consistency.py             强一致 ✅
```

## 8. 环境与依赖

### 8.1 当前环境（已检测）

- Python 3.x，torch `2.12.0+cpu` ⚠️（需升级）
- transformers ≥ 5.0（上次已装）
- peft（版本未知，可能 < 0.13）
- bitsandbytes ❌ 未装

### 8.2 升级目标

```text
# environment/requirements.txt
torch>=2.5            # CUDA 版本，sm_120 (Blackwell) 兼容
transformers>=5.0
peft>=0.13            # OLoRA、PiSSA、LoftQ init 支持
bitsandbytes>=0.43    # QLoRA peft 路径，GPU only
jupyterlab>=4.0
ipykernel>=6.0
matplotlib>=3.7
numpy>=1.24
accelerate>=0.30
datasets>=2.14        # 真训练用 IMDb / SST-2
scipy>=1.10           # SVD 工具（PiSSA、AdaLoRA）
```

### 8.3 升级命令（Plan Phase 1 执行）

```powershell
# 卸载 CPU 版 torch
pip uninstall -y torch torchvision torchaudio

# 装 CUDA 12.6 版 torch（Blackwell 兼容）
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

# 装其它
pip install --upgrade peft bitsandbytes accelerate datasets scipy matplotlib

# 验证
python learning/lora-family/environment/verify_env.py
```

### 8.4 verify_env.py 设计

三段式自检：

```python
# Part A: 基础（必须通过）
- torch import
- transformers import
- peft >= 0.13 import
- nf4_quant.py 模块导入

# Part B: GPU 检测（可选）
- torch.cuda.is_available()
- GPU 名称、显存
- 简单 GEMM 测试

# Part C: bitsandbytes（GPU 选做）
- bitsandbytes import
- 一次 NF4 quantize 调用（GPU 上）
- 与 fake-quant 的误差对比
```

## 9. Git 策略

每完成一个 Phase 就 commit，commit 消息中文，参考上次格式：

```
feat(lora-family): 建立专题骨架 + 升级 CUDA 环境
feat(nf4-quant): NF4 fake-quant 共享模块 + 单元测试
feat(lora): 完成 LoRA 章节（含 rsLoRA、LoRA+ 附录）
feat(adalora): 完成 AdaLoRA 章节（含 mini training）
feat(pissa): 完成 PiSSA 章节（含 OLoRA 附录）
feat(vera): 完成 VeRA 章节
feat(loha-lokr): 完成 LoHa+LoKr 章节
feat(qlora): 完成 QLoRA 章节（fake-quant 主战场）
feat(loftq): 完成 LoftQ 章节
feat(dora): 完成 DoRA 章节（专题压轴）
docs(lora-family): README 横向对比 + 全专题总结
```

**Tag 节点**：

- `lora-base`: Phase 3 完成后（LoRA + rsLoRA + LoRA+）
- `lora-svd-trio`: Phase 6 完成后（AdaLoRA + PiSSA + VeRA）
- `lora-shape`: Phase 7 完成后（LoHa+LoKr）
- `lora-quant`: Phase 9 完成后（QLoRA + LoftQ）
- `lora-family-complete`: Phase 11 完成后（专题闭环）

## 10. 思考题策略

每个 lecture 末尾 4-6 道，类型覆盖：

- **公式题**：推导某条公式、对比两个方法的公式差
- **代码题**：在 minimal 上加几行实现某种变体
- **设计题**：解释作者为什么选择 A 而不是 B
- **对比题**：与上一专题（prompt-tuning-family）或本专题前序方法对比
- **实践题**：在 GPU 选做 cell 里跑某个真训练，观察 loss

## 11. README 横向对比表设计

`learning/lora-family/README.md` 末尾的对比表跨越 12 个方法 + 全参微调基准：

| 方法 | 年份 | $\Delta W$ 形式 | 参数量公式 | GPT-2 base, $r=8$ 数值 | 主战场 | 典型学习率 | 训练复杂度 | 推理开销 |
|------|------|----------------|------------|----------------------|--------|------------|------------|----------|
| 全参 FT | — | $W \mathrel{+}= \Delta W$ | $d^2$ | 590K (per layer) | 通用 | 5e-5 | 高 | 0 |
| **LoRA** | 2021 | $BA$, $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$ | $2rd$ | 12K | 通用 | 1e-4 ~ 3e-4 | 低 | 0（合并） |
| **rsLoRA** | 2023 | 同 LoRA, scaling $\alpha/\sqrt r$ | $2rd$ | 12K | 通用 | 同 LoRA | 低 | 0 |
| **LoRA+** | 2024 | 同 LoRA | $2rd$ | 12K | 通用 | $\eta_B = 16 \eta_A$ | 低 | 0 |
| **AdaLoRA** | 2023 | $P \Lambda Q^T$（SVD 形式） | 约 $2.5 rd$（含打分） | 30K | NLU | 5e-4 | 中 | 0（剪枝后） |
| **PiSSA** | 2024 | $BA$（SVD top-r 初始化） | $2rd$ | 12K | 通用 | 1e-4 | 低 | 0 |
| **OLoRA** | 2024 | $BA$（QR 正交初始化） | $2rd$ | 12K | 通用 | 1e-4 | 低 | 0 |
| **VeRA** | 2024 | $\Lambda_d B \Lambda_b A$（A、B 冻结） | $r + d$（每层） | < 1K | 大批量任务 | 1e-2 | 低 | 0 |
| **LoHa** | 2021 | $(B_1 A_1) \odot (B_2 A_2)$ | $4rd$ | 24K | 高效率 | 1e-4 | 中 | 0 |
| **LoKr** | 2023 | $B \otimes A$ | $\sqrt{d_1 d_2} \cdot r$ | 几 K | 高效率 | 1e-4 | 中 | 0 |
| **QLoRA** | 2023 | $\text{dequant}(\text{NF4}(W)) + BA$ | $2rd$ + 量化表 | 12K | 大模型省显存 | 1e-4 | 中 | 反量化开销 |
| **LoftQ** | 2023 | $\text{NF4}(W) + UΣV^T_{r}$（量化感知初始化） | $2rd$ | 12K | 大模型省显存 | 1e-4 | 中 | 反量化开销 |
| **DoRA** | 2024 | $W = m \cdot \frac{V}{\|V\|}$，$V = W_0 + BA$ | $2rd + d$ | 12.8K | 通用 + 高质量 | 1e-4 | 中 | 0（合并） |

## 12. 自我评审（self-review）

### 12.1 Spec 覆盖度

- ✅ 范围（12 方法）已与用户确认
- ✅ 粒度（8 lecture，混合）已确认
- ✅ 顺序（学习曲线）已确认
- ✅ 代码深度（forward + 3 个 mini training）已确认
- ✅ 量化策略（fake-quant + GPU 选做）已确认
- ✅ 环境升级（CUDA torch + bitsandbytes）已确认
- ✅ 沿用上次 lecture / 测试 / git 模板

### 12.2 Placeholder 扫描

无 TBD / TODO / 待补。所有方法都列出了 minimal 公式、参数量数值、典型学习率。

### 12.3 内部一致性

- 第 2 章 8 个 lecture 与第 3 章目录结构、第 4 章模板、第 5 章代码深度表互相一致
- 第 6 章 NF4 fake-quant 与第 8 章环境（不强制 bitsandbytes）一致
- 第 11 章对比表与第 2 章方法清单一一对应

### 12.4 范围

单一专题、单一实现计划。无需进一步拆分。

## 13. 与 prompt-tuning-family 的衔接

LoRA 家族与上一专题（prompt-tuning-family）形成"两条主线"：

- **prompt-based**（输入侧）：Prompt Tuning, Prefix Tuning, P-Tuning v1/v2
- **weight-based**（权重侧）：LoRA, AdaLoRA, PiSSA, VeRA, LoHa, LoKr, QLoRA, LoftQ, DoRA（本专题）

读完两个专题后，用户应该能回答：

1. 在 65B 大模型 + 24GB 消费 GPU 上做 NLU 微调，选哪个？（→ QLoRA）
2. 在 GPT-2 small 上做 NER 序列标注，选哪个？（→ P-Tuning v2 或 LoRA）
3. 极致省参数（< 1K per layer），选哪个？（→ VeRA 或 Prompt Tuning）
4. 既要省显存又要高质量，选哪个？（→ DoRA + QLoRA 结合）

这些跨专题对比将作为本专题 README 末尾的"横向横向对比"（meta-comparison）总结。

---

**End of Spec.**
