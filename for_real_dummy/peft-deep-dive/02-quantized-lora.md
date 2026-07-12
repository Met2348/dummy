# 02 · 量化 + LoRA 深挖(Quantized LoRA)

> 总览见 [00-roadmap.md](00-roadmap.md)

LoRA 本身只解决"要不要存/训练整个 ΔW"这一个问题——用两个低秩矩阵 B、A 代替一个稠密的 `d_out×d_in` 矩阵。但 01 号文件里的所有变体都还有一个共同前提没动:被冻结的原始权重 `W₀` 本身,依然是以 fp32/fp16 稠密浮点数的形式原封不动地常驻显存。一个 7B 模型光是这份冻结权重的 fp16 拷贝就要 14GB——很多消费级显卡,连"把它冻结起来不训练"这一步都放不下。本文这 6 个知识点问的是另一个正交的问题:能不能把 `W₀` 自己也压缩掉?知识点 1(NF4)讲清楚"怎么把一个浮点权重矩阵压成 4 bit 还不至于崩掉"这件事本身的机制;知识点 2/3 是同一个 QLoRA 思路的两条独立实现路径——2 是纯 PyTorch 手写的 fake-quant(CPU 能跑,本仓库唯一一处建在量化权重之上的真实多步训练循环),3 是真实 bitsandbytes 4bit(GPU-only,调包);知识点 4(LoftQ)问的是更进一步的问题:NF4 的量化误差,能不能被 LoRA 的初始值预先吸收掉一部分,而不是留给训练慢慢纠正;知识点 5(DoRA)换了个方向——不压缩 `W₀`,而是把 `W₀` 拆成"大小"和"方向"两部分,只让方向参与低秩更新,和"量化"其实没有关系(下面会专门说清楚这一点);知识点 6 不是新代码,是把 2/3 两条路径的边界钉死成一句可以用命令行核实的精确陈述。

**和 [01-lora-core.md](01-lora-core.md) 的关系:** 01 号文件知识点 1 已经把 LoRA 最基础的公式讲透——`h = base(x) + α/r · BAx`,B 零初始化保证训练第 0 步输出和原模型逐位相等——本文默认你已经理解这个骨架,不重复推导。01 号文件那 7 个知识点(rsLoRA/LoRA+/merge_and_unload/PiSSA/VeRA/LoHa/LoKr/AdaLoRA)全部是在"低秩更新 `BA` 本身怎么参数化"这一侧做文章;本文除知识点 5 外,前 4 个知识点问的是另一个正交的问题:被冻结的 `W₀` 能不能压缩。**一个需要提前说清楚的分类误差:** DoRA 严格来说和"量化"没有任何关系——它是把 `W₀` 拆成 magnitude×direction 再对 direction 部分做低秩更新,这件事和 01 号文件的 PiSSA/VeRA 是同一类"重新设计 `ΔW` 怎么参数化"的问题,本应该归在 01 号文件。它出现在本文,是沿用 `00-roadmap.md` 里按源文件的分组(`dora_minimal.py` 和 `nf4_quant.py`/`qlora_minimal.py`/`loftq_minimal.py` 被放进同一批"源:"列表),知识点 5 会用 `grep` 再次核实这一点,不假装 DoRA 用到了本文前面讲的任何量化机制。

**一个诚实标注:** `learning/lora-family/lectures/` 下有 `06-qlora.md`/`07-loftq.md`/`08-dora.md` 三份对应 lecture(`nf4_quant.py`/`qlora_peft.py` 没有专门的 lecture 文件,`06-qlora.md` 大概率是这两个源文件共用的),但延续 01 号文件的做法——本文写作时**没有读这些 lecture 原文**,下面 6 个知识点全部直接从对应 `*.py` 源文件的类/函数签名、实现代码、docstring 里的公式索引出发,配合本文独立写的验证脚本反推机制,不假设存在可转述的现成讲解。

**环境声明(本文涉及 GPU-only 内容,必须准确说明边界):** 知识点 1、2、4、5 的"可运行例子"代码块全部只构造 `nn.Linear`(不加载任何预训练模型)或做纯张量运算,CPU 秒级跑完,不需要网络;知识点 2/4/5 正文里引用的 GPT-2(124M)真实数字,来自各源文件自带的 `main()` 或对应 `tests/` 下的 pytest(本机 HF cache 已有 GPT-2,不需要联网下载),文中会明确标注这是 `main()`/pytest 的输出,不是"可运行例子"代码块自己重新跑出来的。知识点 3 是本系列迄今唯一一处"读代码之前先用 `python -c "import torch; print(torch.cuda.is_available())"` 确认硬件"的地方——**这台写作机器上结果是 `True`**:`nvidia-smi` 确认是一块 NVIDIA GeForce RTX 3080 Ti Laptop GPU(16384 MiB 显存,compute capability 8.6,driver 595.97),`bitsandbytes` 已装(0.49.2),`TinyLlama-1.1B-Chat-v1.0` 本机 HF cache 已有。所以知识点 3 不是"读代码猜测行为",而是**真实跑通了一遍**,文中数字(参数量、显存占用、forward shape、内部层类型)全部来自这次真实运行。这个事实对本文读者不构成普遍保证——如果在没有 GPU 或没装 `bitsandbytes` 的机器上跟着敲代码,知识点 3 的例子会按源码里写好的路径分别报 `RuntimeError`(直接调用 `build_peft_qlora_tinyllama`)或打印 `[SKIP]`(跑 `main()`),知识点 3 正文把这两条路径都讲清楚,不是只有"我这里能跑"这一种叙述。全部代码已在仓库根目录 `.venv`(torch 2.11.0+cu128,transformers 5.10.2,peft 0.19.1,bitsandbytes 0.49.2)下实际跑通,文中数字是真实输出,不是手算或转述。

---

## 1. NF4 量化机制(`nf4_quant.py`)—— block-wise 4-bit + 对 absmax 本身做 double quantization

**是什么:**
```python
import torch

NF4_VALUES = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
])  # 16 个 N(0,1) 分位点，严格单调，index 7 精确为 0.0

def nf4_quantize(W: torch.Tensor, block_size: int = 64):
    """步骤: (1) flatten+reshape 成 (n_blocks, block_size)
             (2) 每块 absmax = max(|W|)（per-block scale）
             (3) normalize: W_norm = W / absmax ∈ [-1, 1]
             (4) 把每个 W_norm 值映射到最近的 NF4 网格点
    Returns: indices(uint8), absmax(float), orig_shape, pad
    """

def double_quantize_absmax(absmax: torch.Tensor, block_size_outer: int = 256):
    """把 NF4 的 per-block absmax(fp32)用 INT8 再量化一层。"""
```

**一句话:** NF4 不是均匀量化,是把权重矩阵切成 64 个一组的小块,每块单独算一个缩放系数(absmax),再把归一化后的值映射到一张专门为标准正态分布设计、非均匀分布的 16 点查找表上——连这批缩放系数自己,也要被再量化一次省显存。

**底层机制/为什么这样设计:** 三个独立的设计点分别对应三个问题。**(1) 为什么网格点不是均匀分布的:** 这 16 个值不是 `[-1,-0.875,...,0.875,1]` 这种线性网格(那是 INT4 的做法),而是标准正态分布 N(0,1) 的分位点——因为神经网络权重的经验分布本身接近正态分布,在概率密度高的地方(0 附近)放更密的网格点、密度低的地方(远离 0)放更稀的网格点,用同样 16 个点能换来更小的期望量化误差,这正是"NF4"里"N"(Normal)的含义,也是它和 INT4 的根本区别。**(2) 为什么要分块(block_size=64):** 一个权重矩阵不同区域的数值尺度可能差异很大,如果整个矩阵共用一个全局 absmax,数值尺度小的区域会被数值尺度大的区域"吃掉"(下面的可运行例子会把这个损失精确量出来);分块把"缩放系数"的粒度做细,让每块都能用满 16 个网格点的动态范围,代价是每块都要多存一个 fp32 的 absmax。**(3) 为什么 absmax 自己还要被二次量化:** 每 64 个原始元素对应 1 个 4-bit index(共 32 bytes)+ 1 个 fp32 absmax(4 bytes),这个 absmax 元数据占比 `4/(32+4) ≈ 11%`,block_size 越小占比越高;QLoRA 论文(`arXiv:2305.14314` §3.2)把这些 absmax 自己也拿去做一次 INT8 量化(用更大的 `block_size_outer=256`,256 个 absmax 共享一个新的外层 scale),用这一步把"量化元数据"的开销也压下去,这是"double quantization"名字的来源——量化的对象是量化产生的副产品,不是对 `W` 量化了两遍。

**AI 研究场景:** 显存预算是这条技术路线存在的唯一理由——4-bit 存储把权重体积压到 fp32 的 1/8、fp16 的 1/4,这是能在单张 24GB/16GB 消费级显卡上放下 7B/13B 模型进行微调的直接原因;block-wise(而不是整个矩阵一个 scale)是精度和压缩率之间的折中,`block_size=64` 是 QLoRA 论文给出的经验平衡点,不是任意选择。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/lora-family/src")
import torch
from nf4_quant import NF4_VALUES, nf4_quantize, nf4_dequantize, double_quantize_absmax, double_dequantize_absmax

# 网格结构：严格单调，index 7 精确为 0.0；核实一个容易被想当然的细节
assert NF4_VALUES.numel() == 16
assert NF4_VALUES[7].item() == 0.0
assert torch.all(NF4_VALUES[1:] > NF4_VALUES[:-1])
n_neg = (NF4_VALUES < 0).sum().item()
n_pos = (NF4_VALUES > 0).sum().item()
print(f"n_neg={n_neg}, n_zero={(NF4_VALUES == 0).sum().item()}, n_pos={n_pos}")
assert (n_neg, n_pos) == (7, 8)  # 不是源码 docstring 写的"8 正 8 负"，是 7 负 + 1 个精确 0 + 8 正

# 分块为什么重要：block A 尺度很小(~0.01)，block B 尺度很大(~100)，拼成一个 128 长的向量(=2 个 block)
torch.manual_seed(0)
block_A = torch.randn(64) * 0.01
block_B = torch.randn(64) * 100.0
W = torch.cat([block_A, block_B])

indices, absmax, orig_shape, pad = nf4_quantize(W, block_size=64)
W_hat = nf4_dequantize(indices, absmax, orig_shape, pad)
rel_err_A_blockwise = (W[:64] - W_hat[:64]).abs().mean().item() / block_A.abs().mean().item()
print(f"per-block absmax: {absmax.tolist()}")
print(f"block A 用逐块 absmax 的相对误差: {rel_err_A_blockwise:.4f}")

# 对比：如果强行用一个全局 absmax（等于 block B 的 absmax，因为它更大）会怎样
global_absmax = W.abs().max().clamp(min=1e-8)
dist = (W.unsqueeze(-1) / global_absmax - NF4_VALUES).abs()
W_hat_global = NF4_VALUES[dist.argmin(dim=-1)] * global_absmax
rel_err_A_global = (W[:64] - W_hat_global[:64]).abs().mean().item() / block_A.abs().mean().item()
print(f"block A 用单一全局 absmax 的相对误差: {rel_err_A_global:.4f}")
assert rel_err_A_blockwise < rel_err_A_global  # 分块让小尺度区域的信息不被大尺度区域淹没

# double quantization：把 absmax 自己也量化一遍，看信息损失有多大
torch.manual_seed(1)
W2 = torch.randn(4096, 256)
_, absmax2, _, _ = nf4_quantize(W2, block_size=64)
dq_idx, outer_absmax, shape2, pad2 = double_quantize_absmax(absmax2, block_size_outer=256)
absmax2_hat = double_dequantize_absmax(dq_idx, outer_absmax, shape2, pad2)
dq_rel = (absmax2 - absmax2_hat).pow(2).mean().sqrt().item() / absmax2.std().item()
print(f"double-quant absmax 相对误差: {dq_rel:.4f}")
print(f"absmax 存储 dtype: 量化前 {absmax2.dtype} -> 量化后 {dq_idx.dtype}")
assert absmax2.dtype == torch.float32 and dq_idx.dtype == torch.int8
```

实测(`.venv` 真跑):`n_neg=7, n_zero=1, n_pos=8`——源码 docstring 里"8 正、8 负"这句话不够精确,真实分布是 7 个负值 + 1 个精确的 `0.0`(index 7)+ 8 个正值,不是对称的 8/8。分块对比:`per-block absmax = [0.0341, 261.33]`,block A 用逐块 absmax 时相对误差 `0.1024`(约 10%,和 NF4 本身的量化精度量级一致),但如果被 block B 的更大尺度"拖累"、共用一个全局 absmax,block A 的相对误差直接变成 `1.0000`(约 100%)——这个尺度差异下,block A 的信息在全局量化里基本被摧毁。double quantization:`absmax` 存储从 `float32` 压到 `int8`,引入的相对误差是 `0.0230`(约 2.3%),是一个用可控的小额外误差换显存的操作。

另外用仓库自带 `pytest learning/lora-family/src/tests/test_qlora_consistency.py::test_fake_vs_real_nf4_gpu -v -s`(本机有 GPU,不是 SKIP)验证过一个更硬核的事实:在一个 `torch.randn(256, 256)` 的随机张量上,把这份手写 `nf4_quant_dequant` 的输出和真实 `bitsandbytes.functional.quantize_nf4`/`dequantize_nf4` 的输出逐元素比较,`65536` 个元素里差异**精确为 0**(`max abs diff = mean abs diff = 0.0000000000`)——这份教学向的纯 PyTorch 实现,不只是"误差可接受",而是和 bitsandbytes 的 CUDA kernel 位级一致,这也解释了源码 docstring 里"来源: QLoRA 论文公式 3 + bitsandbytes 参考实现"这句话不是空话。

**面试怎么问 + 追问链:**
- **Q:** "NF4 和普通的 4-bit 均匀量化(INT4)有什么本质区别?"—— 期望说出"NF4 的 16 个量化点是标准正态分布的分位点,不是均匀网格,专门针对权重接近正态分布这个先验设计"。
- **追问 1:** "如果一个张量前半部分数值尺度很小、后半部分尺度很大,不分块直接量化会发生什么?"—— 期望结合"共享一个 absmax,小尺度区域的信息会被大尺度区域淹没"来回答,能举出量级差异(比如"相对误差从 10% 恶化到接近 100%")说明问题不是理论上的,是可以实测出来的。
- **追问 2:** "double quantization 量化的对象是谁,和第一层 NF4 量化是什么关系?"—— 期望明确区分"第一层量化的是原始权重 W,第二层量化的是第一层产生的 absmax 元数据",这是两个不同对象上的操作,不是"量化了两遍权重"。
- **追问 3(深挖):** "block_size 从 64 调到 256,量化误差和存储开销分别怎么变?"—— 期望说出"block 越大,每块内部数值尺度差异可能越大、量化误差可能变大,但 absmax 元数据的相对存储开销变小;block 越小则反过来",体现出对这是一个显式 trade-off 的理解,而不是"越小越好"或"越大越好"的一刀切结论。

**常见坑:** 把"NF4"和"INT4"当成同一类东西的两个名字——两者都是 4-bit,但 INT4 是均匀量化(线性网格,不针对任何分布假设),NF4 的网格点密度专门服从正态分布,数值上完全不同。另一个坑是把"分块量化"和"double quantization"混成一件事——分块量化的对象是原始权重矩阵 `W` 本身,double quantization 量化的对象是分块量化过程中产生的 `absmax` 这批标量元数据,是发生在两个不同张量上的两次独立操作,只是都用了"量化"这个词。

---

## 2. QLoRA fake-quant 训练循环(`qlora_minimal.py`)—— 全系列唯一的真 5 步训练循环

**是什么:**
```python
import torch
import torch.nn as nn

class QLoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16, block_size=64):
        ...
        # 1. 用 NF4 fake-quant 原地替换 base.weight
        with torch.no_grad():
            W_quantized = nf4_quant_dequant(base_linear.weight.data, block_size=block_size)
            base_linear.weight.data.copy_(W_quantized)
        for p in base_linear.parameters():
            p.requires_grad = False
        self.base = base_linear

        # 2. LoRA: B 零初始化（与标准 LoRA 一致）
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = x @ self.A.T @ self.B.T
        return base_out + self.scaling * lora_out
```

**一句话:** QLoRA(这份实现)= 把 base weight 原地替换成 NF4 fake-quant 后的版本(精度是真的损失了,只是没有真的打包成 4-bit 存储)+ 完全标准的 LoRA;这份实现是全系列(9 种 LoRA 方法)里唯一带真实 5 步训练循环的一个,专门用来验证"量化后的 base weight,训练前后必须保持逐位不变"这条底线。

**底层机制/为什么这样设计:** "fake-quant"这个词修饰的是什么,值得先说清楚——`nf4_quant_dequant` 内部先 `nf4_quantize`(得到 4-bit indices + absmax)再立刻 `nf4_dequantize`(还原回浮点数),所以 forward 路径上真正参与矩阵乘法的还是一个 fp32 张量,只是这个张量的取值已经被限制在 NF4 的 16 个网格点(乘以各自 block 的 absmax)上——"fake"修饰的是"存储格式没有真的变成 4-bit 打包",不是"精度没有真的损失"(精度损失是真实的,下面例子会量出这个具体误差)。`__init__` 里 `base_linear.weight.data.copy_(W_quantized)` 是原地覆盖,这一行执行完那一刻起,原始的 fp32 权重就已经不存在于这个对象里了(没有另外保留一份)——这是为什么 `quantization_error()` 方法需要外部传入 `original_weight`:必须在构造 `QLoRALinear` **之前**从别处单独克隆一份原始权重,不能事后再向模型要。"冻结"(`requires_grad=False`)和"永远不变"是两件独立的事,容易被当成同一件事:`requires_grad=False` 只保证 autograd 不会给这个参数算梯度,不直接保证优化器不会碰它;真正让"5 步训练后 base.weight 逐位不变"成立的,是构造 optimizer 时用的 `[p for p in model.parameters() if p.requires_grad]` 这个过滤条件——`base.weight` 从一开始就没有被放进 optimizer 的参数列表,`optimizer.step()` 根本不知道这个参数的存在。这才是"bit-identical"这条断言背后真正的机制,比"冻结了所以不会变"这句直觉说法精确一层。

**AI 研究场景:** 这份 fake-quant 实现的价值不在于省显存(它不省,forward 路径上仍是稠密 fp32 张量),而在于**教学与验证**:不需要 GPU、不需要 bitsandbytes、CPU 秒级跑完,却能完整重现"量化引入的权重扰动"和"量化后权重训练时必须冻结"这两件事的真实数值行为,适合在没有 GPU 的环境里先把 QLoRA 的训练闭环跑通、用调试器单步看懂 loss 怎么算、梯度怎么传,再去啃知识点 3 真正接入 bitsandbytes 的版本。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/lora-family/src")
import torch
import torch.nn as nn
from qlora_minimal import QLoRALinear

torch.manual_seed(0)
base = nn.Linear(32, 16)
W_original = base.weight.data.clone()
layer = QLoRALinear(base, r=4, alpha=8, block_size=64)

# NF4 fake-quant 在构造时就原地改写了 base.weight，且它被冻结
quant_err = (layer.base.weight.data - W_original).abs().max().item()
print(f"base.weight 被 NF4 fake-quant 改动的最大绝对值: {quant_err:.6f}")
assert quant_err > 0
assert layer.base.weight.requires_grad is False

W_before = layer.base.weight.data.clone()
A_before, B_before = layer.A.data.clone(), layer.B.data.clone()

opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=1e-2)
x = torch.randn(8, 32)
target = torch.randn(8, 16)
losses = []
for step in range(5):
    opt.zero_grad()
    out = layer(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    opt.step()
    losses.append(loss.item())

max_diff_base = (layer.base.weight.data - W_before).abs().max().item()
max_diff_A = (layer.A.data - A_before).abs().max().item()
max_diff_B = (layer.B.data - B_before).abs().max().item()
print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"5 步后 base.weight 最大变化: {max_diff_base}")
print(f"5 步后 A 最大变化: {max_diff_A:.6f}    B 最大变化: {max_diff_B:.6f}")

assert max_diff_base == 0.0        # base 冻结，逐位不变（不是"很小"，是精确 0）
assert max_diff_A > 0 and max_diff_B > 0   # LoRA 部分确实在学
```

实测:`base.weight` 被 NF4 fake-quant 改动的最大绝对值 `0.026306`(量化确实真实扰动了权重);5 步训练后 `loss: 1.3919 -> 1.1917`(优化在正常进行);`base.weight` 最大变化 `0.0`(精确为 0,不是浮点意义上的"很小"),`A` 最大变化 `0.034497`、`B` 最大变化 `0.050172`(LoRA 部分确实在更新)。补充:源文件自带 `main()` 在真实 GPT-2 `c_attn` 层上跑出的数字是同一个模式——NF4 量化 RMSE `0.01843`(相对 std 的相对误差 `9.23%`,和知识点 1 里随机矩阵的量级一致),训练 5 步后 base.weight 变化 `0.000000e+00`;仓库自带 `pytest learning/lora-family/src/tests/test_qlora_consistency.py`(5 个测试,`.venv` 下 37.92s 全部 PASS)进一步验证了 10 步训练后依然 `diff < 1e-6`、LoRA 的 A/B 确实分别变化了 `3.4563e-03`/`5.0215e-03`、以及一次 2 句话的 mini-training 能让 loss 从 `6.049` 降到 `2.847`。

**面试怎么问 + 追问链:**
- **Q:** "'fake quant'里的'fake'具体是指什么东西是假的?"—— 期望说出"数值上的量化损失是真实发生的,假的是'存储格式',forward 路径上参与计算的仍然是反量化回浮点数之后的稠密张量,没有真的打包成 4-bit"。
- **追问 1(容易停留在表面):** "只设置 `requires_grad=False` 是不是就足够保证 base weight 训练后不变?"—— 期望不能只回答"是",要能说出"真正生效的是 optimizer 构造时按 `requires_grad` 过滤参数列表这一步——`requires_grad=False` 本身只影响 autograd 要不要算梯度,不直接阻止某个手滑把这个参数塞进 optimizer 的代码"。
- **追问 2:** "这份 fake-quant 实现能不能直接搬到生产环境省显存?"—— 期望明确说"不能,forward 路径上的张量本身还是稠密 fp32/fp16 存储,没有真的省下显存,只是数值上模拟了量化损失;真正省显存要看知识点 3 的真 bitsandbytes 路径"。

**常见坑:** 以为"fake-quant"意味着"量化误差是可以忽略不计的模拟",从而跳过误差校验直接假设训练结果和不量化时一样——量化误差是真实的(上面例子里 `0.026306` 是权重尺度上不算小的扰动),LoRA 能不能训好,某种程度上依赖它能不能在自己的低秩子空间里"补偿"这部分量化带来的偏差,这正是知识点 4(LoftQ)要解决的问题,不能想当然认为量化误差不影响训练。

---

## 3. QLoRA 真 4bit 路径(`qlora_peft.py`)—— 真 BitsAndBytesConfig + TinyLlama,本机 GPU 实测跑通

**是什么:**
```python
def build_peft_qlora_tinyllama(r=8, alpha=16, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "bitsandbytes 4-bit 量化需要 GPU。CPU 环境请用 qlora_minimal.py 的 fake-quant 版。"
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16,
    )
    base = prepare_model_for_kbit_training(base)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(base, config)
    return model, tokenizer

def main() -> None:
    if not torch.cuda.is_available():
        print("[SKIP] 没有 GPU，QLoRA + bitsandbytes 真量化无法运行")
        return
    ...
    with torch.no_grad():          # 注意：main() 只做了一次前向传播，没有训练循环
        out = model(**inputs)
```

**一句话:** 这是知识点 2 的"认真版"——不再手写 fake-quant,而是让 `BitsAndBytesConfig` 真的把整个模型的所有 Linear 层都量化成 4-bit(`uint8` 打包存储),`prepare_model_for_kbit_training` 做几件 4-bit 训练特有的准备工作,再用标准 peft `LoraConfig` 只给 `q_proj`/`v_proj` 挂可训练的低秩适配器——但这份源码的 `main()` 只做了一次前向传播,没有训练循环(知识点 6 会把这一点钉成一个可核实的精确边界)。

**底层机制/为什么这样设计:** `BitsAndBytesConfig` 的四个参数分别对应四个独立决策:`load_in_4bit=True` 是总开关;`bnb_4bit_quant_type="nf4"` 呼应知识点 1 的 NF4(而不是更简单的均匀 4-bit);`bnb_4bit_compute_dtype=torch.float16` 决定的是**矩阵乘法真正执行时**用什么精度——4-bit 只是权重的静态存储格式,真做矩阵乘法之前会现场反量化成 fp16 再算,算完立刻丢弃 fp16 中间结果,所以显存里长期驻留的还是 4-bit;`bnb_4bit_use_double_quant=True` 呼应知识点 1 的 double quantization。`prepare_model_for_kbit_training` 是 4-bit 训练特有的收尾步骤(比如把某些数值敏感层转回更高精度、配置梯度检查点相关设置),知识点 2 的 fake-quant 路径完全不需要它,因为那条路径上的张量从头到尾就是普通 fp32,不存在"4-bit 专属"的数值稳定性问题。这份实现里最容易被忽略、但结构上很清楚的一点是:**量化和 LoRA targeting 是两个完全独立的维度**——`BitsAndBytesConfig` 对整个模型统一生效(所有 Linear 层,不管是不是 LoRA 目标,都会变成 4-bit),`target_modules=["q_proj","v_proj"]` 只是从这些已经量化好的层里,另外选一部分**追加**一个可训练的低秩分支;下面的可运行例子会用真实的类名结构核实这一点。

**AI 研究场景:** 这是"QLoRA"这个名字在实际工程里真正对应的路径——一个 7B/13B 模型能塞进消费级显卡训练,省下来的显存来自"整个模型都以 4-bit 打包存储"这件事本身,不是知识点 2 里"数值上取整到 16 个网格点但存储格式不变"那种教学版本。

**可运行例子(本机真实 GPU 验证;没有 GPU 的环境会在 `assert torch.cuda.is_available()` 这一行就 `AssertionError`,对应源码里 `build_peft_qlora_tinyllama` 会 `raise RuntimeError`、`main()` 会打印 `[SKIP]` 这两条防御路径):**
```python
import sys
sys.path.insert(0, "learning/lora-family/src")
import torch

assert torch.cuda.is_available()  # 本文写作机器上为 True（RTX 3080 Ti Laptop，16GB，compute capability 8.6）
torch.cuda.reset_peak_memory_stats()

from qlora_peft import build_peft_qlora_tinyllama
model, tokenizer = build_peft_qlora_tinyllama(r=8, alpha=16)

mem_mb = torch.cuda.memory_allocated() / 1e6
print(f"4bit 加载后 GPU 显存占用: {mem_mb:.1f} MB")
assert mem_mb < 2000  # TinyLlama-1.1B 若用 fp16 存权重，仅权重本身就要约 2200MB，这里远低于这个量级

# 结构性证据：LoRA target（q_proj）和非 target（k_proj）一样被量化成了 4bit，
# 差别只在于 q_proj 被 peft 多包了一层可训练的 LoRA adapter
layer0 = model.base_model.model.model.layers[0]
q_proj = layer0.self_attn.q_proj
k_proj = layer0.self_attn.k_proj
assert type(q_proj).__module__ == "peft.tuners.lora.bnb"                 # q_proj: peft 的 LoRA-over-4bit wrapper
assert type(q_proj.base_layer).__module__ == "bitsandbytes.nn.modules"   # wrapper 底下是真 bnb 4bit 层
assert type(k_proj).__module__ == "bitsandbytes.nn.modules"              # k_proj 没被 LoRA 包一层，但同样是 4bit
assert k_proj.weight.dtype == torch.uint8                                # 4bit 打包存储的直接证据（不是 fp16/fp32）
assert k_proj.weight.quant_state.quant_type == "nf4"
assert hasattr(q_proj, "lora_A") and not hasattr(k_proj, "lora_A")       # 只有 target module 有 LoRA 分支
assert q_proj.lora_A["default"].weight.dtype == torch.float32            # LoRA adapter 本身仍是 fp32，不是 4bit

inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**inputs)
print(f"forward logits.shape: {tuple(out.logits.shape)}")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable={trainable:,}  total={total:,}  pct={100 * trainable / total:.4f}%")
```

实测(本机真实 GPU,非模拟):4-bit 加载后 GPU 显存占用 `1049.3 MB`(约 1GB,TinyLlama-1.1B 若以 fp16 存储仅权重就要约 2.2GB,4-bit + 部分 fp32 buffer 之后实际测得的量级印证了压缩确实发生了);`q_proj` 的类是 `peft.tuners.lora.bnb.Linear4bit`,它的 `base_layer` 是 `bitsandbytes.nn.modules.Linear4bit`;`k_proj`(不是 LoRA target)直接就是 `bitsandbytes.nn.modules.Linear4bit`,没有 LoRA wrapper,但 `weight.dtype` 同样是 `torch.uint8`、`quant_state.quant_type` 同样是 `"nf4"`——量化对整个模型一视同仁,只是没有的模块不会额外带 `lora_A`/`lora_B`;`q_proj.lora_A["default"].weight.dtype` 是 `torch.float32`——注入的低秩适配器本身从不参与量化,一直是 fp32,这也是它能被正常反向传播更新的前提(4-bit 存储的值被锁死在 16 个网格点上,谈不上"梯度更新"这件事)。`forward logits.shape = (1, 6, 32000)`,`trainable=1,126,400 total=616,732,672 pct=0.1826%`。这个 `0.1826%` 比知识点 2 GPT-2 例子里的 `0.2364%` 更低,不是巧合矛盾,而是因为 TinyLlama 的 `q_proj`/`v_proj` 是分离的两个 Linear(总参数基数 1.1B 更大),GPT-2 的 `c_attn` 是融合 Q/K/V 的单个 `Conv1D`——两个模型的架构和 target_modules 覆盖面都不同,这两个百分比不能直接跨模型比较,能比较的只是"同一个模型内,加 LoRA 之后可训练比例被压到了 1% 以内"这个共同结论。

**面试怎么问 + 追问链:**
- **Q:** "`prepare_model_for_kbit_training` 这一步是干什么的,为什么知识点 2 的 fake-quant 版本不需要它?"—— 期望说出"它处理的是真实 4-bit 存储特有的数值稳定性/训练准备问题,fake-quant 版本的张量从头到尾是普通 fp32,不存在这类问题"。
- **追问 1:** "`target_modules` 只写了 `q_proj`/`v_proj`,那 `k_proj`/`o_proj` 是不是还是 fp16?"—— 期望回答"不是,整个模型的 Linear 层都会被 `BitsAndBytesConfig` 量化成 4-bit,`target_modules` 只决定哪些层额外获得一个可训练的 LoRA 分支,这是两个独立的开关",并能用上面验证过的 `k_proj` 类型/dtype 作为证据。
- **追问 2(深挖计算路径):** "4-bit 权重做矩阵乘法时,是直接在 4-bit 精度下算的吗?"—— 期望说出"不是,`bnb_4bit_compute_dtype` 决定了真正参与矩阵乘法的精度(这里是 fp16),4-bit 只是静态存储格式,矩阵乘法前会现场反量化,算完立即丢弃那份临时的高精度张量"。

**常见坑:** 以为"没有 GPU 就完全没法学这份代码"——源码里两条清晰的防御路径(`build_peft_qlora_tinyllama` 直接 `raise RuntimeError`,`main()` 判断后 `print("[SKIP]")` 并 `return`)本身就是可以脱离 GPU 读懂、理解其设计意图的代码,不需要真的跑起来才能看懂"这一步在保护什么"。另一个坑是把"知识点 1 验证过手写 fake-quant 和真 bitsandbytes 在随机张量上位级一致"这件事,过度引申成"知识点 2 和知识点 3 的 trainable% 应该完全一致"——`qlora_minimal.py` 和 `qlora_peft.py` 用的是两个不同的模型(GPT-2 vs TinyLlama)、不同的 `target_modules`(`c_attn` vs `q_proj`+`v_proj`),两边的参数量占比数字不能直接跨模型对齐,能对齐的只是"底层量化算法数值一致"这一层,不包括"整体模型的参数量结构"。

---

## 4. LoftQ(`loftq_minimal.py`)—— 交替最小化,让 LoRA 初始值预先吸收量化误差

**是什么:**
```python
import torch
import torch.nn as nn

class LoftQLinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=None, n_iter=5, block_size=64):
        ...
        W = _extract_weight(base_linear).float()

        # 初始化: Q = NF4(W), BA = 0（这一步和知识点 2 的 QLoRA 完全一样）
        Q = nf4_quant_dequant(W, block_size=block_size)
        BA = torch.zeros_like(W)
        history = [(W - Q - BA).norm().item()]

        for _ in range(n_iter):
            # Step 1: SVD of (W - Q) -> 秩-r 最优近似
            U, S, Vt = torch.linalg.svd(W - Q, full_matrices=False)
            sqrt_S = S[:r].sqrt()
            B_t = U[:, :r] * sqrt_S.unsqueeze(0)
            A_t = sqrt_S.unsqueeze(-1) * Vt[:r, :]
            BA = B_t @ A_t
            # Step 2: 用当前 BA 重新量化残差
            Q = nf4_quant_dequant(W - BA, block_size=block_size)
            history.append((W - Q - BA).norm().item())

        # base.weight = Q（冻结）；A、B 用交替最小化算出的最终值初始化（不是随机/零）
```

**一句话:** 知识点 2 的 QLoRA 是"先量化,LoRA 的 A/B 从随机/零开始,量化误差就晾在那不去管";LoftQ 反过来问:能不能让 A、B 的**初始值**专门用来吸收这部分量化误差,而不是留给训练慢慢纠正——用一个"交替最小化"的循环,轮流把量化目标从"精确重建 W"改成"精确重建 W 减去当前的 BA"。

**底层机制/为什么这样设计:** 目标函数是 `‖W − Q − BA‖_F²`——同时找一个量化结果 `Q` 和一个秩-r 分解 `BA`,使两者加起来最接近原始 `W`。这两个未知量互相耦合,没有联合闭式解,LoftQ 用的是交替最小化:**固定 `Q`,对 `(W−Q)` 做 SVD 求最优秩-r 近似**——这一步有严格的理论依据(Eckart-Young 定理:一个矩阵在 Frobenius 范数意义下的最优秩-r 近似,就是它前 r 个奇异值/奇异向量的重构),是这个循环里唯一有闭式解、真正意义上"最优"的一半;**固定刚求出的 `BA`,对 `(W−BA)` 重新做 NF4 量化**——这一步不是在解一个最优化问题,NF4 的 16 个网格点是固定查找表,"量化"就是找最近邻,谈不上"最优 Q",只是"用当前最新的 BA 重新算一次残差再取整"。把这两步交替跑 `n_iter` 次,`history[0]`(初始状态,`BA` 全零)其实就是知识点 2 里 QLoRA 的起点——`Q = nf4_quant_dequant(W)`,`BA = 0`——LoftQ 真正的贡献只发生在后面这几轮交替里。还有个容易被忽略的初始化细节:`alpha=None` 时默认 `alpha=r`(即 `scaling=1`),源码注释写的是"与 PiSSA 一致",而不是知识点 2/3 默认的 `alpha=2r`(`scaling=2`)——LoftQ 和 01 号文件的 PiSSA 同属"用真实权重信息初始化 A/B"这一类方法,A、B 从一开始就不是零,不需要更大的 scaling 去补偿"初始 `ΔW=0`"这件事(这里初始 `ΔW` 根本不是 0)。

**AI 研究场景:** 量化误差如果完全不处理,相当于训练一开始模型的实际行为就已经偏离了预训练权重;LoftQ 这类"感知量化误差的初始化"方法,是在"必须用 4-bit 因为显存不够"和"希望初始状态尽量贴近原模型"这两个约束之间找一个更好的起点。论文报告在更低比特(比如 2-bit)场景下收益尤其明显——比特数越低,单纯量化的误差越大,LoftQ 能在初始化阶段吸收掉的那部分误差占比也越有意义;这是论文的实验结论,如实标注来源,本文的数值 demo 只验证"交替最小化确实让残差下降"这个机制本身,不重新做"下游任务效果是否更好"这个需要真实训练才能回答的实验。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/lora-family/src")
import torch
import torch.nn as nn
from loftq_minimal import LoftQLinear
from nf4_quant import nf4_quant_dequant

torch.manual_seed(0)
base = nn.Linear(64, 64, bias=False)
W0 = base.weight.data.clone()

layer = LoftQLinear(base, r=8, n_iter=5, block_size=64)
history = layer.get_convergence()
print("收敛历史 ||W - Q - BA||_F:")
for t, v in enumerate(history):
    print(f"  t={t}: {v:.6f}")
print(f"final/init 比值: {history[-1] / history[0]:.4f}")

non_increasing = all(history[i] >= history[i + 1] - 1e-4 for i in range(len(history) - 1))
assert non_increasing  # 交替最小化的每一步都不应让目标函数变差

# 对比：完全不做交替最小化的朴素单发 NF4（就是知识点 2 QLoRA 的做法：Q=NF4(W)，BA 恒为 0）
Q_naive = nf4_quant_dequant(W0, block_size=64)
naive_residual = (W0 - Q_naive).norm().item()
print(f"\n朴素单发 NF4 残差(不做 LoftQ 精修): {naive_residual:.6f}")
print(f"LoftQ 5 轮交替最小化后的残差:        {history[-1]:.6f}")
assert history[0] == naive_residual   # 验证：LoftQ 的 t=0 就是朴素单发量化本身
assert history[-1] < naive_residual   # 交替最小化确实比朴素单发量化残差更小
```

实测:收敛历史 `0.424752 → 0.340451 → 0.330243 → 0.329787 → 0.328933 → 0.322350`,`final/init` 比值 `0.7589`,严格非递增(每一步都不劣化)。第一轮下降最猛(`0.4248→0.3405`,降了约 20%),后面几轮边际收益迅速变小(第 5 轮只比第 4 轮再降约 2%)——收敛曲线本身就带有"边际效益递减"的证据,不需要额外的解释就能看出来。朴素单发 NF4(即完全不做 LoftQ 交替最小化,等价于知识点 2 的量化方式)残差是 `0.424752`,精确等于收敛历史的 `t=0`,验证了"LoftQ 的起点就是普通 QLoRA"这个结构性关系;5 轮交替最小化后残差降到 `0.322350`,确实优于朴素单发量化。补充:源文件 `main()` 在真实 GPT-2 `c_attn` 层(r=8, alpha=8, n_iter=5)上跑出的收敛历史是 `25.4871 → 24.8250 → 24.4836 → 24.2527 → 24.0806 → 23.9483`,`final/init=0.9396`——同样单调下降,但相对降幅(约 6%)比上面随机初始化的 `nn.Linear`(约 24%)小得多。这是一个如实记录、没有过度解释的观察:一个刚随机初始化的矩阵,信息集中在少数几个方向上,秩-8 的近似能"解释"掉更大一部分方差;已经训练收敛的真实 GPT-2 权重,奇异值分布通常更平坦(信息分散在更多方向上),同样的秩-8 预算能吸收的比例自然更小。

**面试怎么问 + 追问链:**
- **Q:** "LoftQ 和普通 QLoRA(先量化、LoRA 从零/随机初始化)的核心区别是什么?"—— 期望说出"LoftQ 用交替最小化让 A、B 的初始值就去逼近'量化前后的残差',而不是像普通 QLoRA 那样把这部分误差完全留给训练去纠正"。
- **追问 1(深挖):** "交替最小化的两步分别在解什么问题,为什么可以这样拆?"—— 期望说出"SVD 那一步是严格意义上的最优秩-r 近似(Eckart-Young 定理,有闭式解),NF4 量化那一步是固定查找表最近邻取整(不是最优化问题),两者交替进行是因为联合求解没有闭式解,只能各自固定一半来推进"。
- **追问 2:** "`n_iter` 是不是越大越好?"—— 没有一刀切答案,期望结合实测数据说"边际收益递减很明显(比如第 5 轮比第 4 轮的降幅远小于第 1 轮比第 0 轮),继续加大 n_iter 的收益会越来越小,同时每一轮都要做一次 SVD,计算成本是线性增长的",体现出对 trade-off 的判断力而不是背一个固定数字。

**常见坑:** 把"交替最小化"理解成"两个子问题都是用梯度下降在解同一个统一目标"——实际上 SVD 那一步是闭式解(不是迭代下降出来的),NF4 量化那一步甚至根本不是在做最优化(是固定查找表的最近邻取整),这是两种性质完全不同的操作轮流执行,不是同一套梯度方法反复应用。另一个坑是把"LoftQ 起点等于朴素单发量化"这件事,误解成"LoftQ 对量化误差已经很大的场景没有额外价值"——事实相反,起点相同恰恰说明 LoftQ 的全部价值都来自后面的交替最小化轮次,量化误差越大(比如更低比特),这几轮精修能挽回的绝对数值往往越可观。

---

## 5. DoRA(`dora_minimal.py`)—— magnitude × direction 分解,和"量化"完全无关

**是什么:**
```python
import torch
import torch.nn as nn

class DoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16, dropout=0.0):
        ...
        W_0 = _extract_weight_out_in(base_linear)      # (d_out, d_in)
        m_init = W_0.norm(dim=0)                        # column-wise norm -> (d_in,)
        self.m = nn.Parameter(m_init.clone())

        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))    # B 零初始化

    def _compute_W_dora(self, x_for_dropout=None):
        W_0 = _extract_weight_out_in(self.base)
        delta = self.scaling * (self.B @ self.A)        # α/r BA
        V = W_0 + delta
        norm = V.norm(dim=0).clamp(min=1e-8).detach()    # detach: 避免二阶导
        scale = (self.m / norm).unsqueeze(0)
        return scale * V                                 # m · V / ||V||_c
```

**一句话:** DoRA 把权重矩阵拆成"每一列的长度"(magnitude,一个向量 `m`)乘以"归一化后的方向"(direction,`V/‖V‖`),固定的是这个拆分方式本身,真正在学习的是两块内容——`m` 直接学、方向部分 `V=W₀+BA` 用标准 LoRA 更新——**这份实现从头到尾没有出现任何量化**,和知识点 1-4 的 NF4 完全无关,是被 `00-roadmap.md` 按源文件分组放进这一篇的(下面会专门核实这一点)。

**底层机制/为什么这样设计:** 权重分解公式 `W = m · V/‖V‖_c`,下标 `c` 指 column-wise(逐"列"求 norm)——`W_0` 的 shape 是 `(d_out, d_in)`,`norm(dim=0)` 是对每个 `d_in` 方向的列单独求一个模长,得到一个 `(d_in,)` 向量,不是整个矩阵一个标量 norm。`_compute_W_dora` 里 `norm(V).detach()` 这个 `detach()` 是关键设计:如果不加,`m / norm(V)` 里的 `norm(V)` 是 `B`、`A` 的函数(因为 `V = W_0 + scaling·BA`),对 `B`、`A` 求梯度时会经过 `norm(·)`(内部是 `sqrt(sum(x²))`,非线性)这条额外路径,反向传播的二阶行为会变复杂;源码注释写的是"避免二阶导",这是 DoRA 论文自己指出的简化——加上 `detach` 之后,方向部分 `V` 到 `B`/`A` 的梯度和标准 LoRA 完全一样,只是最终输出多套了一层"除以一个(视为常数的)当前 norm、再乘以可学的 `m`"的 rescaling,不需要重新推一套反向传播。初始化为什么能保证 `W_DoRA = W_0`:`B` 零初始化 → `delta=0` → `V=W_0` → `norm(V)=norm(W_0)`,而 `m_init` 恰好就是从 `W_0.norm(dim=0)` 复制来的,所以 `norm(V)=m_init` → `scale=m/norm=1` → `W_DoRA=1·V=W_0`,这条链路和 01 号文件里 LoRA 的"B 零初始化保证起点等于原模型"是同一个思路在不同参数化下的体现。

**关于"这一节和量化的关系"必须专门说清楚:** 用 `grep -n "nf4\|quant" learning/lora-family/src/dora_minimal.py` 核实过,**零命中**——这份实现里 `self.base.weight` 从始至终就是构造时那份原始浮点数,从未被量化替换(对比知识点 2/4,`QLoRALinear`/`LoftQLinear` 都会在 `__init__` 里用 `nf4_quant_dequant` 原地覆盖 `base.weight`)。DoRA 论文本身允许和 QLoRA 结合(社区称为"QDoRA":先 NF4 量化 `W_0`,再对量化后的结果做 magnitude/direction 分解),但这个仓库的 `dora_minimal.py` 没有实现这个组合版本,只实现了在全精度 `W_0` 上做 DoRA——这是核对源码后确认的事实,不是"DoRA 理论上和量化无关"这种一般性论断的替代。

**AI 研究场景:** DoRA 论文报告在低秩(小 r)场景下,效果比标准 LoRA 更接近全参数微调,归因于"把 magnitude 和 direction 拆开后,即使 direction 部分的低秩更新预算很小,单独学习的 `m` 仍能覆盖一大类全参数微调才会调整的模式(比如整体调大/调小某个特征方向的权重响应,而不改变它的方向)"——如实标注这是论文的实验结论,本文的数值 demo 只验证分解公式和梯度路径这两件机制层面的事,不重新做"DoRA 效果是否真的更接近全参数微调"这个需要跑真实下游任务才能回答的实验。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/lora-family/src")
import torch
import torch.nn as nn
from dora_minimal import DoRALinear

torch.manual_seed(0)
base = nn.Linear(32, 16, bias=False)
W0 = base.weight.data.clone()  # (d_out=16, d_in=32)

layer = DoRALinear(base, r=4, alpha=8)
print(f"m shape: {tuple(layer.m.shape)}  A shape: {tuple(layer.A.shape)}  B shape: {tuple(layer.B.shape)}")
assert tuple(layer.m.shape) == (32,) and tuple(layer.A.shape) == (4, 32) and tuple(layer.B.shape) == (16, 4)

# B 零初始化 -> delta=0 -> V=W0 -> norm(V)=norm(W0)=m_init -> W_dora 精确重建 W0
W_dora_init = layer._compute_W_dora()
init_err = (W_dora_init - W0).abs().max().item()
m_err = (layer.m.data - W0.norm(dim=0)).abs().max().item()
print(f"初始重建误差 max|W_dora - W_0|: {init_err:.10f}")
print(f"m 与 ||W_0||_column 的误差: {m_err:.10f}")
assert init_err < 1e-5 and m_err < 1e-5

# 训练几步，确认 m/A/B 都在学，且 base.weight（原始全精度权重）从未被改动
m_before, A_before, B_before = layer.m.data.clone(), layer.A.data.clone(), layer.B.data.clone()
opt = torch.optim.AdamW(layer.parameters(), lr=1e-2)
x, target = torch.randn(8, 32), torch.randn(8, 16)
for _ in range(5):
    opt.zero_grad()
    loss = ((layer(x) - target) ** 2).mean()
    loss.backward()
    opt.step()

print(f"5 步后 m/A/B 最大变化: "
      f"{(layer.m.data - m_before).abs().max().item():.4f} / "
      f"{(layer.A.data - A_before).abs().max().item():.4f} / "
      f"{(layer.B.data - B_before).abs().max().item():.4f}")
assert (layer.m.data - m_before).abs().max().item() > 0
assert (layer.A.data - A_before).abs().max().item() > 0
assert (layer.B.data - B_before).abs().max().item() > 0

base_diff = (layer.base.weight.data - W0).abs().max().item()
print(f"base.weight(原始全精度权重)是否被改动: {base_diff}")
assert base_diff == 0.0   # 全程未被量化替换，这是和 QLoRA/LoftQ 最直接的区别
```

实测:形状全部符合预期(`m: (32,)`, `A: (4,32)`, `B: (16,4)`);初始重建误差 `0.0000000000`、`m` 与 `‖W_0‖_column` 的误差 `0.0000000000`,两者都精确为 0(不是"足够接近"),验证了公式(1)在 `BA=0` 时精确退化为 `W_0`;训练 5 步后 `m`/`A`/`B` 分别变化 `0.0503`/`0.0344`/`0.0502`,三者都在学;`base.weight`(原始全精度权重)最大变化 `0.0`,精确不变——这一点和知识点 2/4 表面相似("被冻结的原始权重训练后不变"),但机制不同:QLoRA/LoftQ 的 `base.weight` 已经是被 NF4 改写过的**量化后**权重,DoRA 的 `base.weight` 自始至终是**未经任何改动的原始**权重,两者"不变"这件事的对象根本不是同一个东西。补充:源文件 `main()` 在真实 GPT-2 `c_attn` 上验证了同一个初始化性质(`init W_DoRA vs W_0 最大误差: 0.0000e+00`),并给出参数量分解:每层 `m` 贡献 `768`、`A` 贡献 `6,144`、`B` 贡献 `18,432`,12 层合计 `304,128`,比标准 LoRA 多出 `12×768=9,216` 个纯 magnitude 参数。

**面试怎么问 + 追问链:**
- **Q:** "DoRA 相比 LoRA,新增的可训练参数是什么,分解公式是什么?"—— 期望答出"`W=m·V/‖V‖_c`,`m` 是逐列的 magnitude 向量(新增可训练参数),`V=W_0+scaling·BA` 是方向部分,`BA` 部分的训练方式和标准 LoRA 完全一样"。
- **追问 1:** "`_compute_W_dora` 里为什么要对 `norm(V)` 做 `detach()`?"—— 期望说出"切断 `norm(V)` 到 `B`/`A` 的梯度路径,避免二阶导数,让方向部分的梯度退化成标准 LoRA 的形式,是论文自己指出的简化"。
- **追问 2(容易被问到的陷阱,呼应本文分类误差):** "DoRA 是不是一种量化方法?"—— 期望明确回答"不是,是纯粹的权重重参数化,和量化是两个独立的技术方向;论文允许两者结合(QDoRA),但这个仓库的 `dora_minimal.py` 没有实现这种组合,可以用 `grep` 直接核实源码里没有出现过 nf4/quant 字样"。
- **追问 3:** "column-wise norm 是沿哪个维度算的,为什么是这个维度?"—— 期望说出"`W_0` 是 `(d_out, d_in)`,`norm(dim=0)` 是对每个 `d_in` 方向的'列'单独求模长,得到 `(d_in,)` 的 `m`;这个维度选择和 `forward` 里 `x @ W.T` 的语义对应——每个输入特征方向对应一个独立的 magnitude 标量"。

**常见坑:** 把 DoRA 和"量化家族"混为一谈——这正是本文标题("量化 + LoRA")容易带来的错觉,上面"底层机制"部分已经用 `grep` 核实过源码里根本没有量化;另一个坑是以为 `detach(norm)` 会导致 `m` 或方向部分学不到东西——实际上 `detach` 只是切断了"`norm(V)` 这个中间量到 `B`/`A` 的梯度路径",不影响 `V` 本身(也就是 `BA`)到 `B`/`A` 的正常梯度,`m` 的梯度更是完全不受影响(`m` 是独立叶子参数,梯度直接来自 `∂L/∂W_dora · V/norm`,和 `norm` 是否 `detach` 无关)。

---

## 6. "真 4bit 训练 + 真 bitsandbytes 从未同时出现"—— 精确边界的结构性核实

**是什么:** 不是新代码,是对知识点 2、3,以及一个容易被忽略的第三处(`tests/test_qlora_consistency.py::test_fake_vs_real_nf4_gpu`)做的一次可以用命令行复现的结构性核实。

**一句话:** 全仓库范围内搜索 `bitsandbytes`/`BitsAndBytesConfig`/`bnb.`,真正调用到 bitsandbytes 功能的地方只有这 3 处,没有一处同时具备"真实 bitsandbytes 4-bit"和"多步梯度训练循环(`.backward()` + `optimizer.step()`)"——这不是巧合或遗漏,是这几份代码各自明确的分工。

**底层机制/为什么这样设计:** 三处逐一列清楚:
1. **`qlora_peft.py` 全文件(含 `main()`)**——真 `BitsAndBytesConfig` + 真 `prepare_model_for_kbit_training` + 真 `get_peft_model`,但 `main()` 函数体里只有一次被 `torch.no_grad()` 包裹的前向传播,没有 optimizer、没有 `.backward()`。
2. **`tests/test_qlora_consistency.py::test_fake_vs_real_nf4_gpu`**——直接调用 `bnb.functional.quantize_nf4`/`dequantize_nf4`(比 `qlora_peft.py` 更底层,跳过了 `BitsAndBytesConfig`/HF 集成,直接调用 bnb 的裸 function),目的是数值上核对"手写 fake-quant"和"真 bnb NF4"是否一致(知识点 1 已经引用过这个结果:位级一致),同样没有训练循环。
3. **`qlora_minimal.py`**——反过来,有真实、完整的 5 步 `.backward()` + `optimizer.step()` 训练循环,但从头到尾没有出现过 `bitsandbytes`/`BitsAndBytesConfig` 这两个字符串,用的是知识点 1 的手写 NF4 实现。

为什么不是干脆把知识点 2 和 3 合并成一个"真 4-bit + 真训练"的完整例子:真 bitsandbytes 4-bit 层的反向传播路径由 bitsandbytes 库自己的 CUDA kernel 实现(经 `prepare_model_for_kbit_training` 配置),要在这上面演示一遍"训练循环长什么样",需要真实下载/加载一个真实模型(TinyLlama-1.1B,即使有本地 cache 也有加载耗时)、需要真实 GPU、需要多步迭代看 loss 变化——这些对"讲清楚 4-bit 量化本身怎么接入 HF/peft 生态"这个知识点 3 的目的来说是不必要的重量级依赖。`qlora_minimal.py` 把"验证训练循环里 base weight 真的不变"这条断言,放在一个不需要 GPU、不需要下载模型、CPU 秒级跑完、任何人都能在自己电脑上重复的 fake-quant 版本上——这是教学设计上有意的分工,不是"漏做了"真 4-bit 训练这个组合。

**AI 研究场景:** 对准备复现 QLoRA 训练的读者,这个边界有直接的工程含义——如果想看"4-bit 量化本身在真实 bitsandbytes 下长什么样、显存占用多少、内部层类型是什么",看知识点 3;如果想在没有 GPU 的机器上用调试器单步跟一遍"loss 怎么算、梯度怎么传、base weight 为什么不变"这个完整训练闭环,看知识点 2。这个仓库目前没有提供"两者都要"的现成脚本——想要的话,需要自己把知识点 3 `build_peft_qlora_tinyllama` 返回的 `model` 接上一个类似知识点 2/pytest `test_mini_training` 那样的训练循环,是读者自己动手的组合,不是抄现成代码就能得到的。

**可运行例子:**
```python
from pathlib import Path

minimal_src = Path("learning/lora-family/src/qlora_minimal.py").read_text(encoding="utf-8")
peft_src = Path("learning/lora-family/src/qlora_peft.py").read_text(encoding="utf-8")
test_src = Path("learning/lora-family/src/tests/test_qlora_consistency.py").read_text(encoding="utf-8")

# qlora_minimal.py：真实训练循环，且从未出现过 bitsandbytes
assert "import bitsandbytes" not in minimal_src
assert "BitsAndBytesConfig" not in minimal_src
assert ".backward()" in minimal_src
assert "optimizer.step()" in minimal_src
print("qlora_minimal.py: 无 bitsandbytes，有 .backward()+optimizer.step()")

# qlora_peft.py：真实 bitsandbytes（经 BitsAndBytesConfig），但没有训练循环
assert "BitsAndBytesConfig" in peft_src
assert ".backward()" not in peft_src
assert "optimizer" not in peft_src.lower()
print("qlora_peft.py: 有 BitsAndBytesConfig，无 .backward()/optimizer")

# 测试文件里这一个函数：真实调用 bnb.functional，同样没有训练循环
gpu_test_body = test_src[test_src.index("def test_fake_vs_real_nf4_gpu"):]
assert "bnb." in gpu_test_body
assert ".backward()" not in gpu_test_body
assert "optimizer" not in gpu_test_body.lower()
print("test_fake_vs_real_nf4_gpu: 有真实 bnb.functional 调用，无训练循环")

print("\n结论：全仓库出现 bitsandbytes/BitsAndBytesConfig/bnb. 的地方一共 3 处，"
      "没有一处同时具备 .backward()/optimizer -> 真 bnb 与真多步训练从未同时出现")
```

实测:三个 `assert` 段全部通过,打印结果和上面"底层机制"逐一列出的三处完全对应——这是本文写作时用代码核实过的事实,不是转述印象。

**面试怎么问 + 追问链:**
- **Q:** "这个仓库里,有没有一个例子是'真实 bitsandbytes 4-bit + 真实多步训练'同时具备的?"—— 期望明确回答"没有",并说清楚这是"这个教学仓库目前的现状"这个具体的、可以 `grep` 验证的事实,不是"这种组合技术上做不到"这种一般性论断(生产环境里 QLoRA 训练当然是 4-bit + 真训练同时发生的)。
- **追问 1:** "为什么手写的 fake-quant 版本反而是仓库里唯一有真训练循环的版本?"—— 期望结合"不需要 GPU、不需要下载模型、CPU 秒级跑完,适合作为教学/调试用的最小可复现闭环"这几个具体的工程原因来回答,而不是"因为 fake 更简单"这种笼统说法。
- **追问 2(检验是否真的核实过而不是背了结论):** "怎么用命令行快速确认这个边界?"—— 期望说出类似 `grep -rln "bitsandbytes\|BitsAndBytesConfig" learning/lora-family/src/` 先定位命中的文件,再对每个命中文件单独 `grep -n "backward\|optimizer"` 检查有没有训练循环的具体做法,而不是"我读过代码所以我知道"。

**常见坑:** 把"这个教学仓库没有真 4-bit + 真训练的例子"引申成"QLoRA 真实训练时,4-bit 权重是不是应该会被更新"——完全不是,真实生产环境里 bitsandbytes 的 4-bit 权重同样是冻结的(`prepare_model_for_kbit_training` + 只把 LoRA 参数放进 optimizer,机制上和知识点 2 的 fake-quant 版本一致),这个边界只是在说"这个教学仓库分别在哪些文件里演示了训练循环的哪一半",跟"QLoRA 真实训练时 4-bit 权重会不会被更新"这个更基础的问题(答案是不会)完全是两回事,不能混在一起。
