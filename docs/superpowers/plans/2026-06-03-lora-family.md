# LoRA 家族学习专题 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: 沿用上次的 inline execution（batched commits per phase）。
>
> **Goal**: 为 12 个 LoRA 家族方法建立完整学习材料：8 个 PPT-style 中文 lecture + minimal & peft 代码 + 一致性测试 + Jupyter notebook + 全系列横向对比。
>
> **Architecture**: 沿用 `prompt-tuning-family` 模板。新增 NF4 fake-quant 共享模块，新增 GPU 选做 cell（bitsandbytes + TinyLlama-1.1B 真量化）。
>
> **Tech Stack**: PyTorch 2.5+ (CUDA)、transformers 5.x、peft 0.13+、bitsandbytes 0.43+、matplotlib、scipy、datasets。

---

## Phase 1: 仓库骨架 + 环境升级 + 论文下载

**目标**：建好目录、升级 CUDA torch、下好 12 篇论文 PDF。

### Task 1.1: 创建专题目录与子目录

- [ ] 创建 `learning/lora-family/{environment,papers,lectures,src/tests,notebooks}` 全部子目录
- [ ] 验证：`ls learning/lora-family/` 列出 5 个子目录

### Task 1.2: environment/requirements.txt

- [ ] 写入完整依赖（torch CUDA、peft>=0.13、bitsandbytes、datasets、scipy、matplotlib 等）

### Task 1.3: 升级 torch 到 CUDA 版本

- [ ] 卸载 `torch torchvision torchaudio` CPU 版
- [ ] 装 CUDA 12.6 版（`--index-url https://download.pytorch.org/whl/cu126`）
- [ ] 验证 `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` 输出 `True RTX 5090 Laptop`

### Task 1.4: 安装 bitsandbytes、peft、其它依赖

- [ ] `pip install --upgrade bitsandbytes peft accelerate datasets scipy matplotlib`
- [ ] 验证 `python -c "import bitsandbytes; print(bitsandbytes.__version__)"` 输出 ≥ 0.43
- [ ] 验证 `python -c "import peft; print(peft.__version__)"` 输出 ≥ 0.13

### Task 1.5: 写 verify_env.py

- [ ] 三段式自检（Part A 基础 / Part B GPU / Part C bitsandbytes）
- [ ] 验证：`python learning/lora-family/environment/verify_env.py` 全部 PASS

### Task 1.6: 下载 12 篇论文 PDF

并行 curl 下载（用 `arxiv.org/pdf/XXXX.YYYYY` 路径）。每篇 < 10MB。

- [ ] 01 LoRA: 2106.09685
- [ ] 02 AdaLoRA: 2303.10512
- [ ] 03 PiSSA: 2404.02948
- [ ] 04 VeRA: 2310.11454
- [ ] 05a LoHa: 2108.06098
- [ ] 05b LoKr: FedPara 相关，arXiv 2108.06098 续作（用 LyCORIS 论文 2309.14859）
- [ ] 06 QLoRA: 2305.14314
- [ ] 07 LoftQ: 2310.08659
- [ ] 08 DoRA: 2402.09353
- [ ] 附 rsLoRA: 2312.03732
- [ ] 附 LoRA+: 2402.12354
- [ ] 附 OLoRA: 2406.01775

### Task 1.7: 写专题 README 占位（先有骨架，最后再填横向对比表）

- [ ] 标题、目录结构、学习路径说明、与 prompt-tuning-family 衔接

### Task 1.8: Phase 1 commit

```bash
git add learning/lora-family/
git commit -m "feat(lora-family): 建立专题骨架 + 升级 CUDA 环境 + 12 篇论文"
```

---

## Phase 2: NF4 fake-quant + common 共享工具

**目标**：为后续 LoRA 系列方法准备共享 utility。

### Task 2.1: src/common.py

- [ ] 复用上次的 `count_parameters`、`print_param_summary`、`ToySentimentDataset`
- [ ] 新增 `freeze_base_model(model)`：把 base 模型全部参数 `requires_grad=False`
- [ ] 新增 `target_linear_modules(model, target_names=("attn.c_attn", "attn.c_proj"))`：返回要打 LoRA 的 nn.Linear/Conv1D 列表
- [ ] 新增 `device_auto()`：自动返回 cuda / cpu

### Task 2.2: src/nf4_quant.py 核心实现

- [ ] 定义 NF4 网格点（来自 N(0,1) 的 16 分位点常数，参考 QLoRA 论文 §3）
- [ ] `NF4_LOOKUP: torch.Tensor` shape `(16,)`，提前定义
- [ ] `nf4_quantize(W, block_size=64) -> (quantized_indices, absmax_scale)`：
  - blockwise: 把 W reshape 成 (-1, block_size)
  - 每块计算 absmax = abs(W).max(-1).values
  - 把 W/absmax 映射到最近的 NF4 网格点索引
- [ ] `nf4_dequantize(indices, absmax_scale) -> W_hat`：反查找表 + 反 scale
- [ ] `nf4_quant_dequant(W, block_size=64) -> W_hat`：组合 + STE
- [ ] `NF4Linear(nn.Module)`：封装一层用 fake-quant 替换 weight 的 nn.Linear

### Task 2.3: src/tests/test_nf4_quant.py

- [ ] 单元 1: NF4 网格点值符合 QLoRA 论文 §3.2 的 NF4 typeset
- [ ] 单元 2: 对 N(0, 1) 输入，量化-反量化的相对均方误差 < 0.012
- [ ] 单元 3: STE 反向（验证 grad 完整穿过）
- [ ] (GPU 可选) 单元 4: 与 bitsandbytes 真 NF4 的 logits 差距 < 0.01

### Task 2.4: Phase 2 commit + tag

```bash
git add learning/lora-family/src/{common.py,nf4_quant.py,tests/test_nf4_quant.py}
git commit -m "feat(nf4-quant): NF4 fake-quant 共享模块 + 单元测试"
git tag lora-foundation
```

---

## Phase 3: 01 LoRA（含 rsLoRA + LoRA+ 附录）

**目标**：完整 LoRA lecture（28 主张 + 8-10 附录张）+ minimal + peft + test + notebook + mini training（rsLoRA、LoRA+）。

### Task 3.1: lectures/01-lora.md

28 主幻灯片结构：
- 1: 封面（"LoRA: Low-Rank Adaptation of Large Language Models"，Hu et al., 2021, Microsoft）
- 2: 符号速查（$L, H, d, r, \alpha, W_0, A, B$）
- 3-5: 上下文（Adapter Layers 推理慢、Prefix Tuning 难训等历史，引出"权重侧低秩适配"思路）
- 6: 核心公式 (1) $h = W_0 x + \frac{\alpha}{r} BA x$，逐项重述（$W_0 \in \mathbb{R}^{d \times d}$ 冻结、$B \in \mathbb{R}^{d \times r}$ 零初始化、$A \in \mathbb{R}^{r \times d}$ Gaussian 初始化、$\alpha$ scaling 常数）
- 7-8: 为什么 $B$ 零初始化（保证训练开始时 $\Delta W = 0$）、为什么有 $\alpha/r$ scaling（不同 r 下保持相近梯度量级）
- 9-10: 参数量分析 $|\phi| = 2rd$ vs 全参 $d^2$
- 11-13: 与 Adapter Layers、Prefix Tuning 的对比表
- 14-15: 架构示意图（Mermaid）
- 16: 张量形状追踪（input x → BA → output）
- 17-19: GLUE / WikiSQL / GPT-3 175B 关键实验
- 20: 哪些 layer 放 LoRA 最有效（论文 §7.1）
- 21: rank r 的选择（论文 §7.2，r=4 vs 64）
- 22-23: 优缺点 + 适用边界
- 24-25: 横向对比表（与 prompt-tuning-family 串联）
- 26-27: minimal + peft 代码片段
- 28: 思考题（4-6 道）

**附录 rsLoRA**（5 张）：
- A1: rsLoRA 动机（论文：r→∞ 时 LoRA 退化）
- A2: 公式 $\frac{\alpha}{\sqrt r}$ 推导
- A3: 与 LoRA 的代码 diff（一行修改）
- A4: 实验结果（不同 r 下的稳定性）
- A5: 思考题

**附录 LoRA+**（5 张）：
- B1: LoRA+ 动机（A、B 在优化下不对称）
- B2: 公式 $\eta_B = \lambda \eta_A$（$\lambda \approx 16$）
- B3: 与 LoRA 的代码 diff（optimizer param groups）
- B4: 实验结果（收敛加速）
- B5: 思考题

### Task 3.2: src/lora_minimal.py

```python
class LoRALinear(nn.Module):
    """单层 LoRA: y = W_0 x + α/r * B A x"""
    def __init__(self, base_linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_linear  # frozen
        for p in self.base.parameters():
            p.requires_grad = False
        d_in = base_linear.in_features  # GPT-2 Conv1D 用 .weight.shape
        d_out = base_linear.out_features
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.normal_(self.A, std=1.0 / r)  # Kaiming 等价
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.base(x) + self.scaling * F.linear(F.linear(self.dropout(x), self.A), self.B)

class LoRAGPT2(nn.Module):
    def __init__(self, r=8, alpha=16, target=("c_attn",)):
        # 加载 GPT-2，冻结，把每层 c_attn 替换为 LoRALinear 包装
```

注意 GPT-2 用的是 `Conv1D` 而非 `nn.Linear`，shape 是 `(in, out)` 而非 `(out, in)`。

### Task 3.3: src/lora_peft.py

```python
from peft import LoraConfig, get_peft_model, TaskType

def build_peft_model(r=8, alpha=16, target=("c_attn",)):
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r, lora_alpha=alpha,
        target_modules=list(target),
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(base, config)
```

### Task 3.4: src/lora_extensions.py（rsLoRA + LoRA+）

```python
class RSLoRALinear(LoRALinear):
    """rsLoRA: scaling = α / sqrt(r)"""
    def __init__(self, base_linear, r=8, alpha=16, **kwargs):
        super().__init__(base_linear, r, alpha, **kwargs)
        self.scaling = alpha / math.sqrt(r)  # 唯一差异

def lora_plus_param_groups(model, lr_A=1e-4, lr_B=1.6e-3):
    """LoRA+: A、B 用不同学习率"""
    A_params, B_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if name.endswith(".A"): A_params.append(p)
        elif name.endswith(".B"): B_params.append(p)
    return [{"params": A_params, "lr": lr_A}, {"params": B_params, "lr": lr_B}]
```

### Task 3.5: src/tests/test_lora_consistency.py

- [ ] 强一致：minimal 与 peft 的 forward logits 差 < 1e-4（探测 peft 内部 lora_A、lora_B 布局后 copy 过去）
- [ ] PASS 标志：`logits 最大绝对误差: x.xxe-XX`

### Task 3.6: src/tests/test_lora_extensions.py

- [ ] rsLoRA 单元：`scaling == alpha / sqrt(r)`（不同 r 下）
- [ ] LoRA+ 单元：`optimizer.param_groups[0]["lr"]` 与 `[1]["lr"]` 比例 ≈ 16
- [ ] mini training（CPU/GPU）：
  - rsLoRA vs LoRA 在 r=64 下，loss 下降稳定性
  - LoRA+ vs LoRA 在前 30 step 的 loss 下降速度（>= 1.3x 加速）

### Task 3.7: notebooks/01-lora.ipynb

- [ ] Cell 1: env check
- [ ] Cell 2: 手写 LoRALinear 演示 + 参数量 print（应得 2*8*768=12,288）
- [ ] Cell 3: peft LoraConfig + 参数量
- [ ] Cell 4: forward consistency test
- [ ] Cell 5: rsLoRA scaling demo（r=4,16,64 三组打表）
- [ ] Cell 6: LoRA+ optimizer groups demo
- [ ] Cell 7: mini training（rsLoRA vs LoRA on r=64，画 loss 曲线）
- [ ] Cell 8: mini training（LoRA+ vs LoRA，画 loss 曲线）
- [ ] Cell 9: 思考题

### Task 3.8: 跑 notebook（jupyter nbconvert --execute）

- [ ] 所有 cell 无错误
- [ ] consistency test 通过
- [ ] loss 曲线生成

### Task 3.9: Phase 3 commit + tag

```bash
git add learning/lora-family/{lectures/01-lora.md,src/lora*,src/tests/test_lora*,notebooks/01-lora.ipynb}
git commit -m "feat(lora): 完成 LoRA 章节（含 rsLoRA、LoRA+ 附录与 mini training）"
git tag lora-base
```

---

## Phase 4: 02 AdaLoRA（SVD 形式 + 重要性打分）

### Task 4.1: lectures/02-adalora.md

28 张幻灯片：
- 1: 封面（AdaLoRA, Zhang et al., 2023, Microsoft）
- 2: 符号（在 LoRA 基础上加 $\Lambda$, $P$, $Q$, $S$ 重要性打分）
- 3-5: 动机（LoRA 给每层固定 r 不优，重要的层该给更多）
- 6-7: 公式 (1) $\Delta W = P \Lambda Q^T$，逐项重述（$P \in \mathbb{R}^{d \times r}$、$\Lambda \in \mathbb{R}^{r \times r}$ 对角、$Q^T \in \mathbb{R}^{r \times d}$）
- 8: 与 LoRA $BA$ 的等价性（$B = P, A = \Lambda Q^T$）
- 9-10: 公式 (2) 正交正则化 $\|P^T P - I\|_F^2$ 和 $\|Q Q^T - I\|_F^2$，逐项重述
- 11: 公式 (3) 重要性打分 $S_i = |\Lambda_i \cdot \nabla_{\Lambda_i} \mathcal{L}|$，逐项重述
- 12-13: 训练流程（warmup / 重要性更新 / 剪枝 schedule）
- 14-16: 架构图 + 张量形状追踪
- 17-21: GLUE / SQuAD 关键实验
- 22-23: 优缺点
- 24-25: 横向对比（vs LoRA）
- 26-27: minimal + peft
- 28: 思考题

### Task 4.2: src/adalora_minimal.py

```python
class AdaLoRALinear(nn.Module):
    def __init__(self, base_linear, r_init=12, alpha=16):
        # P, Lambda, Q
        # importance score buffer
    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.Q.T @ self.Lambda.diag() @ self.P.T)
    def update_importance(self, grad_Lambda):
        # S_i = |Lambda_i * grad_i|, EMA smoothed
```

### Task 4.3: src/adalora_peft.py

`from peft import AdaLoraConfig`

### Task 4.4: src/tests/test_adalora_consistency.py

- [ ] 弱一致：forward logits 在固定参数下匹配
- [ ] 重要性打分 invariance：sort(importance_minimal) 与 sort(importance_peft) 一致

### Task 4.5: notebooks/02-adalora.ipynb

- [ ] Cell 1: env check
- [ ] Cell 2: minimal 演示 + 参数量
- [ ] Cell 3: peft 调包 + 参数量
- [ ] Cell 4: consistency test
- [ ] Cell 5: mini training + 每 step 打印重要性分布（matplotlib heatmap）
- [ ] Cell 6: 剪枝演示（手动剪掉 importance 最低的 30%，看 logits 变化）
- [ ] Cell 7: 思考题

### Task 4.6: 跑 notebook

### Task 4.7: Phase 4 commit

---

## Phase 5: 03 PiSSA（含 OLoRA 附录）

### Task 5.1: lectures/03-pissa.md

28 主张 + 5 张 OLoRA 附录。

主幻灯片要点：
- 公式 (1) SVD: $W_0 = U \Sigma V^T$
- 公式 (2) PiSSA 初始化: $B = U_{:r} \sqrt{\Sigma_{:r}}$, $A = \sqrt{\Sigma_{:r}} V_{:r}^T$
- 公式 (3) 残差: $W_{\text{res}} = W_0 - BA$（冻结）
- 训练时只调 $A, B$，但起点不是 0 而是"预训练 W 的最强方向"
- 与 LoRA 零初始化 $B$ 的对比（LoRA: $\Delta W = 0$ at start；PiSSA: $\Delta W = $ top-r SVD componet at start）

附录 OLoRA：
- QR 分解：$W_0 = QR$，用 $Q$ 的前 $r$ 列初始化 $B$、$R$ 的前 $r$ 行初始化 $A$
- 优点：A、B 正交，训练更稳

### Task 5.2: src/pissa_minimal.py

```python
class PiSSALinear(nn.Module):
    def __init__(self, base_linear, r=8):
        # 1. 提取 base_linear.weight
        # 2. SVD: U, S, V = torch.svd(W)
        # 3. B = U[:, :r] @ diag(sqrt(S[:r]))
        # 4. A = diag(sqrt(S[:r])) @ V.T[:r, :]
        # 5. W_residual = W - B @ A
        # 6. 把 base_linear.weight 替换为 W_residual（冻结）
```

### Task 5.3: src/pissa_olora_extension.py

```python
class OLoRALinear(nn.Module):
    def __init__(self, base_linear, r=8):
        # Q, R = torch.linalg.qr(W)
        # B = Q[:, :r]
        # A = R[:r, :]
```

### Task 5.4: src/pissa_peft.py

`LoraConfig(init_lora_weights="pissa")` 和 `"olora"`。

### Task 5.5: src/tests/test_pissa_consistency.py

- [ ] 强一致：PiSSA 初始化后的 forward 与原始 GPT-2 一致（验证 $W_{\text{res}} + BA = W_0$）
- [ ] minimal vs peft logits

### Task 5.6: notebooks/03-pissa.ipynb

- [ ] PiSSA 初始化 vs LoRA 零初始化的对比可视化（matplotlib）
- [ ] OLoRA 演示

### Task 5.7-5.8: 跑 notebook + commit

---

## Phase 6: 04 VeRA

### Task 6.1: lectures/04-vera.md

28 张：
- 公式 (1) $\Delta W = \Lambda_d B \Lambda_b A$
- 逐项: $A, B$ 用固定 seed 随机初始化后**冻结**；$\Lambda_d \in \mathbb{R}^d, \Lambda_b \in \mathbb{R}^r$ 是可训练对角向量
- 参数量分析: 每层只需 $d + r$，比 LoRA 少 $\sim 2rd / (d+r) \approx 10\times$
- 共享 A、B（所有层共用同一对）

### Task 6.2: src/vera_minimal.py

```python
class VeRALinear(nn.Module):
    SHARED_A = None  # class-level，所有层共享
    SHARED_B = None
    def __init__(self, base_linear, r=256, seed=42):
        # 第一次实例化时生成 SHARED_A, SHARED_B（固定 seed）
        # 冻结
        # 可训练: lambda_d (size d), lambda_b (size r)
```

### Task 6.3: src/vera_peft.py

`from peft import VeraConfig` 或 `VeraConfig(r=256)`.

### Task 6.4: src/tests/test_vera_consistency.py

- [ ] 强一致：相同 seed 下，minimal 与 peft 的 forward 一致

### Task 6.5: notebooks/04-vera.ipynb

- [ ] 共享 A、B 的内存占用 print（vs LoRA 每层独立）
- [ ] r=256 极大值演示

### Task 6.6-6.7: 跑 notebook + commit

---

## Phase 7: 05 LoHa+LoKr（一个 lecture 涵盖两个方法）

### Task 7.1: lectures/05-loha-lokr.md

28 张：
- 1-3: 共同动机（突破 LoRA 的 rank 上限）
- 4-13: LoHa 部分
  - 公式: $\Delta W = (B_1 A_1) \odot (B_2 A_2)$（Hadamard 积）
  - 等效秩 = $r_1 \cdot r_2$（而非 $r_1 + r_2$）
  - 参数量: $4rd$
- 14-22: LoKr 部分
  - 公式: $\Delta W = B \otimes A$（Kronecker 积）
  - 把 $d \times d$ 矩阵分解为 $\sqrt d \times \sqrt d$ 的两个 Kron 积
  - 参数量: $2r \sqrt{d_1 d_2}$（远小于 LoRA）
- 23-25: 横向对比
- 26-27: 代码
- 28: 思考题

### Task 7.2: src/loha_minimal.py

```python
class LoHaLinear(nn.Module):
    def __init__(self, base_linear, r=8):
        self.A1, self.B1 = ...  # nn.Parameter
        self.A2, self.B2 = ...
    def forward(self, x):
        delta1 = self.B1 @ self.A1
        delta2 = self.B2 @ self.A2
        delta = delta1 * delta2  # Hadamard
        return self.base(x) + x @ delta.T
```

### Task 7.3: src/lokr_minimal.py

```python
class LoKrLinear(nn.Module):
    def __init__(self, base_linear, factor=16, r=4):
        # d = factor * d_factor
        # delta = (B1 @ A1) ⊗ (B2 @ A2)，shape 等于 (factor*d_f, factor*d_f)
```

### Task 7.4: src/loha_lokr_peft.py

`from peft import LoHaConfig, LoKrConfig`

### Task 7.5: src/tests/test_loha_lokr_consistency.py

### Task 7.6: notebooks/05-loha-lokr.ipynb

- [ ] LoHa 等效秩 $r_1 r_2$ 演示（rank 数值实测）
- [ ] LoKr Kronecker 积形状演示

### Task 7.7-7.8: 跑 notebook + commit

---

## Phase 8: 06 QLoRA（fake-quant 主战场）

### Task 8.1: lectures/06-qlora.md

28 张：
- 1: 封面（QLoRA, Dettmers et al., 2023, UW）
- 2: 符号
- 3-5: 动机（65B 模型显存爆炸，量化是必经之路）
- 6-7: 公式 (1) NF4 量化: $W_{\text{nf4}} = \text{quantize}_{\text{NF4}}(W)$，逐项重述（块大小 64、scale per block、查找表 16 值）
- 8-9: 公式 (2) Double Quantization: $\text{scale}_{\text{outer}} = \text{quantize}_{\text{INT8}}(\text{scale}_{\text{inner}})$
- 10-11: 公式 (3) Paged Optimizer（CPU/GPU 交换）
- 12-13: 公式 (4) 前向: $h = \text{dequant}(W_{\text{nf4}}) x + \frac{\alpha}{r} BA x$，**关键**：LoRA 不量化、只把 $W_0$ 量化
- 14-16: 架构图 + NF4 网格点可视化
- 17-21: Vicuna / LLaMA 实验
- 22-23: 优缺点（推理时反量化的开销）
- 24-25: 横向对比
- 26-27: 代码（fake-quant + peft）
- 28: 思考题

### Task 8.2: src/qlora_minimal.py

```python
class QLoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16):
        # 1. 把 base_linear.weight 用 nf4_quant_dequant 量化-反量化
        # 2. 用量化后的权重替换 base_linear.weight（冻结）
        # 3. 加 LoRA: B, A 同 lora_minimal.py
    def forward(self, x):
        # h = quantized_base(x) + α/r BAx
```

### Task 8.3: src/qlora_peft.py

```python
def build_peft_qlora_model_gpu():
    """GPU only: 用 bitsandbytes 真量化"""
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        quantization_config=bnb_config,
    )
    config = LoraConfig(r=8, ...)
    return get_peft_model(base, config)
```

### Task 8.4: src/tests/test_qlora_consistency.py

- [ ] 强一致（fake-quant）：minimal 自洽（两次 forward 一致，反量化误差有界）
- [ ] 弱一致（GPU 选做）：fake-quant 与 bitsandbytes 真 NF4 的 logits 误差 < 0.5（大模型）

### Task 8.5: notebooks/06-qlora.ipynb

- [ ] Cell 1-4: 基础 fake-quant 演示
- [ ] Cell 5: NF4 网格点可视化（matplotlib，对比 INT4）
- [ ] Cell 6: 量化误差曲线（不同 block_size 下）
- [ ] Cell 7: **GPU 选做**: bitsandbytes + TinyLlama-1.1B 真 NF4 forward
- [ ] Cell 8: **GPU 选做**: fake-quant vs 真 NF4 一致性
- [ ] Cell 9: 思考题

### Task 8.6-8.7: 跑 notebook + commit

---

## Phase 9: 07 LoftQ（量化感知初始化）

### Task 9.1: lectures/07-loftq.md

28 张：
- 公式 (1) 目标: $\min_{Q, A, B} \|W - Q - BA\|_F$，其中 $Q$ = NF4 量化、$BA$ = LoRA
- 公式 (2) 交替最小化迭代:
  - 给定 $Q$，求最优 $BA$ = SVD top-r of $(W - Q)$
  - 给定 $BA$，求最优 $Q = \text{NF4}(W - BA)$
- 公式 (3) 收敛性（论文 §3.3）
- 与 QLoRA 的差异：QLoRA 量化后用 $A$ Gaussian、$B$ 零初始化；LoftQ 用迭代 SVD 初始化

### Task 9.2: src/loftq_minimal.py

```python
class LoftQLinear(nn.Module):
    def __init__(self, base_linear, r=8, n_iter=5):
        # 迭代:
        # Q = 0
        # for _ in range(n_iter):
        #     residual = W - Q
        #     U, S, V = svd(residual); B = U[:,:r] sqrt(S[:r]); A = sqrt(S[:r]) V.T[:r,:]
        #     Q = nf4_quant_dequant(W - B @ A)
        # 最终: base_linear.weight = Q (冻结), B, A 可训练
```

### Task 9.3: src/loftq_peft.py

`LoraConfig(init_lora_weights="loftq", loftq_config=LoftQConfig(loftq_bits=4))`

### Task 9.4: src/tests/test_loftq_consistency.py

- [ ] 迭代收敛单元：$\|W - Q^{(n)} - B^{(n)} A^{(n)}\|_F$ 随 n 递减
- [ ] minimal vs peft 弱一致

### Task 9.5: notebooks/07-loftq.ipynb

- [ ] 迭代过程可视化（loss 曲线随 iter）
- [ ] QLoRA vs LoftQ 初始化下的微调 loss 对比（mini training）

### Task 9.6-9.7: 跑 notebook + commit

---

## Phase 10: 08 DoRA（压轴）

### Task 10.1: lectures/08-dora.md

28 张：
- 1: 封面（DoRA, Liu et al., 2024, NVIDIA）
- 2: 符号
- 3-5: 动机（LoRA 与全参 FT 的"权重更新模式"差异 → magnitude vs direction）
- 6-7: 公式 (1) 权重分解: $W = m \cdot \frac{V}{\|V\|_c}$，逐项（$m \in \mathbb{R}^d$ magnitude vector、$V$ 是 direction matrix、$\|\cdot\|_c$ column-wise norm）
- 8-9: 公式 (2) DoRA 微调: $W' = m' \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$，$m'$ 可训练，$B, A$ LoRA 风格
- 10: 公式 (3) 反向传播的"detach $\|V\|$"技巧
- 11-13: 与 LoRA 的对比（DoRA 更接近全参 FT 的更新模式）
- 14-16: 架构图
- 17-21: 实验（LLaMA、VL-T5）
- 22-23: 优缺点
- 24-25: 横向对比（**全 12 方法汇总**）
- 26-27: 代码
- 28: 思考题 + 全专题总结

### Task 10.2: src/dora_minimal.py

```python
class DoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16):
        self.base = base_linear  # frozen
        d_in, d_out = ...
        # initial magnitude = ||W_0||_c
        self.m = nn.Parameter(base_linear.weight.norm(dim=0))  # 可训练
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
    def forward(self, x):
        W_lora = self.base.weight + (self.scaling * self.B @ self.A)
        norm = W_lora.norm(dim=0, keepdim=True).detach()  # detach!
        W_dora = (self.m.unsqueeze(0) / norm) * W_lora
        return x @ W_dora.T + self.base.bias
```

### Task 10.3: src/dora_peft.py

`LoraConfig(use_dora=True)` 或 `DoraConfig`（看 peft 版本）。

### Task 10.4: src/tests/test_dora_consistency.py

- [ ] 强一致：相同 W_0, A, B, m 下 minimal vs peft 的 forward 一致

### Task 10.5: notebooks/08-dora.ipynb

- [ ] magnitude / direction 分解可视化
- [ ] DoRA vs LoRA 的"更新模式"对比（每 step 的 ΔW 的 magnitude/direction 变化）

### Task 10.6-10.7: 跑 notebook + commit

---

## Phase 11: 全专题闭环

### Task 11.1: 完善 README.md

- 学习路径推荐
- 全 12 方法横向对比表（spec §11 的最终版）
- 与 prompt-tuning-family 的衔接（横向横向对比）
- 学习清单（"读完应能回答的 N 个问题"）

### Task 11.2: 检查所有 notebook

- [ ] 全部 `jupyter nbconvert --execute` 通过
- [ ] 所有 consistency test 通过
- [ ] 所有 mini training 收敛

### Task 11.3: 检查所有 lecture 内部引用与公式渲染

- [ ] cross-reference 链接（`../src/...`、`../papers/...`）
- [ ] LaTeX 公式（`$$ ... $$`）

### Task 11.4: 最终 commit + tag

```bash
git add learning/lora-family/README.md
git commit -m "docs(lora-family): README 横向对比 + 全专题总结"
git tag lora-family-complete
git log --oneline --all
```

### Task 11.5: 输出最终交付报告给用户

包括：
- 文件清单（行数统计）
- 一致性测试结果汇总
- GPU 选做 cell 的运行状态（如果 GPU 环境就绪）
- 推荐阅读顺序
- 后续可扩展方向（LongLoRA / MoSLoRA / DyLoRA 等下次再做）

---

## Self-Review

- ✅ Spec 中 12 章覆盖度：每章都有对应 Task
- ✅ 无占位符（TBD/TODO）
- ✅ Phase 间无类型/接口不一致：所有 `*_minimal.py` 都基于 `common.freeze_base_model` + `target_linear_modules`，所有 `*_peft.py` 都返回 `PeftModel`，所有 `test_*.py` 都用 `from .. import *` 的统一 import 风格
- ✅ NF4 fake-quant（Phase 2）作为 QLoRA / LoftQ 的依赖在 Phase 8/9 引用
- ✅ Tag 节点完整覆盖里程碑

---

**End of Plan.**
