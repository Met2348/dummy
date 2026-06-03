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

## 横向对比表（占位，Phase 11 填充完整版）

完整的 12 方法横向对比表（参数量 / 主战场 / 学习率 / 训练复杂度 / 部署）将在最后一个 Phase 完成后写入此处。

## 与 prompt-tuning-family 的衔接

LoRA 家族与上一专题（prompt-tuning-family）形成"两条主线"：

- **prompt-based**（输入侧）：Prompt Tuning、Prefix Tuning、P-Tuning v1/v2
- **weight-based**（权重侧，本专题）：LoRA、AdaLoRA、PiSSA、VeRA、LoHa、LoKr、QLoRA、LoftQ、DoRA

完整学习后应能回答：

1. 在 65B 大模型 + 24GB 消费 GPU 上做 NLU 微调，选哪个？
2. 在 GPT-2 small 上做 NER 序列标注，选哪个？
3. 极致省参数（< 1K per layer），选哪个？
4. 既要省显存又要高质量，选哪个？

这些跨专题对比将作为本专题闭环时的 meta-comparison。
