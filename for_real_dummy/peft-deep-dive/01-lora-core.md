# 01 · LoRA 核心与初始化变体深挖(LoRA Core & Initialization Variants)

> 总览见 [00-roadmap.md](00-roadmap.md)

LoRA 本身——"冻结预训练权重 W₀,并联训练一对低秩矩阵 B、A,推理时可以合并回原权重"——只是这一整条思路里最基础的实现。本文这 7 个知识点,是同一个"低秩/稀疏化增量 ΔW"思路在不同维度上分别追问出来的延伸:rsLoRA 和 LoRA+ 追问的是"LoRA 自己的缩放公式和优化器设置,在大 r 下是不是有问题";`merge_and_unload` 追问的是"训练完之后,这层额外算出来的低秩分支,推理时到底还要不要一直算";PiSSA/OLoRA 追问的是"A、B 除了随机初始化,能不能直接用 W₀ 自己的信息初始化得更聪明";VeRA 追问的是"A、B 真的需要每一层都单独训一份吗";LoHa/LoKr 追问的是"同样的参数量,ΔW 的有效秩上限还能不能再往上顶";AdaLoRA 追问的是"r 这个超参能不能不手调,让模型自己在训练过程中动态决定每一层该分配多少秩"。7 个知识点,源码全部在 `learning/lora-family/src/`,除知识点 7(AdaLoRA)的可运行例子包含一次真实的 `.backward()` + `optimizer.step()` 之外,其余全部是纯前向传播或纯张量数值验证,CPU 秒级跑完。

**一个诚实标注:** `learning/lora-family/lectures/` 下其实有对应的 lecture 文件(`01-lora.md`/`02-adalora.md`/`03-pissa.md`/`04-vera.md`/`05-loha-lokr.md`),但本文写作时**没有读这些 lecture 原文**——下面 7 个知识点的讲解全部直接从对应 `*.py` 源文件的类/函数签名、实现代码、docstring 里的"公式索引"出发,配合本文自己独立写的验证脚本反推机制,不假设存在可转述的现成讲解,行文角度和 lecture 也不必然一致。知识点 3(`merge_weights()`)里提到的"docstring 写着会删除 A、B,但实现代码其实没有删,继续调用会双重计数"这个细节,是撰写本文时自己跑出来发现的——用 `grep -rn "merge" learning/lora-family/src/tests/` 确认过,repo 现有的测试文件里没有一处覆盖这个行为,如实标注这是本文自己的验证发现,不代表这是仓库作者已知的问题或既有共识。

**环境声明:** 本文对应源码为 `learning/lora-family/src/{lora_minimal,lora_extensions,pissa_minimal,vera_minimal,loha_minimal,lokr_minimal,loha_lokr_peft,adalora_minimal}.py`。除了知识点 4/5 里引用的 `pissa_minimal.py`/`vera_minimal.py` 自带 `main()` 的真实 GPT-2(124M,本机 HF cache 已有,不需要联网下载)数字、以及知识点 6 里 `loha_lokr_peft.py` 的真实 peft 报错之外,本文"可运行例子"代码块全部只构造小规模 `nn.Linear`(不加载任何预训练模型),CPU 秒级跑完;凡是引用真实 GPT-2 数字的地方,文中都会明确标注这是源文件自带 `main()` 的输出,不是可运行例子代码块本身重新触发的。知识点 7(AdaLoRA)的可运行例子是全文唯一一次真实反向传播(用于演示重要性分数 EMA 的更新),其余全部是纯前向传播 / 纯张量代数。全部代码已在仓库根目录 `.venv`(torch 2.11.0+cu128,transformers 5.10.2,peft 0.19.1)下实际跑通,文中数字是真实输出,不是手算或转述。

---

## 1. `LoRALinear`(`lora_minimal.py`)—— h=base(x)+α/r·BAx,B 零初始化保证起点等于原模型

**是什么:**
```python
class LoRALinear(nn.Module):
    """单层 LoRA: h = base(x) + α/r * B A x。"""

    def __init__(self, base_linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False        # 冻结 W_0

        d_in, d_out = get_in_out_dims(base_linear)
        self.scaling = alpha / r            # 公式 (1) 的 α/r
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))   # B 零初始化，A 用 Kaiming
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.A.T @ self.B.T
        return base_out + self.scaling * lora_out
```

**一句话:** LoRA 冻结原始权重 W₀,并联加一条"降维再升维"的低秩分支 `B@A`(先用 A 把输入投影到 r 维,再用 B 投影回 d_out 维),B 从全零开始训练,所以训练第 0 步的输出和原模型逐位相等。

**底层机制/为什么这样设计:** `scaling = α/r` 是整条分支唯一的额外缩放系数,`A` 用 Kaiming 初始化(有真实非零值),`B` 用全零初始化——两者之一为零就足以保证初始 `ΔW = B@A = 0`,这里选择让 B 归零而不是 A 归零,不是随意的对称选择,而是决定了训练**第 0 步谁先开始学**:把 `forward` 展成 `y = scaling · (x @ Aᵀ) @ Bᵀ`,对 B 求梯度 `∂y/∂B = scaling·(x@Aᵀ)`——只依赖 A,而 A 从一开始就是非零的 Kaiming 值,所以 B 在第 0 步就能拿到真实梯度;但对 A 求梯度要经过 B 这一层链式法则(`∂y/∂A` 正比于 B),B 全零意味着 **A 在第 0 步的梯度精确为零**——A 要等第一次优化器更新把 B 挪离零点之后,才能在第 1 步开始收到非零梯度。也就是说 B 零初始化换来的不只是"起点等于原模型"这一个静态性质,还隐含了一个动态的训练顺序:先让 B 学会怎么用 A 已经给出的这组随机投影方向,A 再跟着调整这组方向本身——下面的可运行例子会把这个梯度为零的瞬间精确量出来。GPT-2 的 `c_attn` 是 `Conv1D` 而不是 `nn.Linear`,两者 `weight` 形状是转置关系(`Conv1D.weight` 是 `(in,out)`,`nn.Linear.weight` 是 `(out,in)`),`LoRALinear` 靠 `common.py::get_in_out_dims`/`is_conv1d` 统一处理,`forward` 本身不需要关心这个差异(`self.base(x)` 两种模块都能正确调用),只有后面知识点 3 的 `merge_weights()` 才需要按类型分别转置。

**AI 研究场景:** 一个几十亿参数模型做全参数微调,不仅要存权重本身,还要存和权重同量级的梯度、以及 Adam 这类优化器每个参数 2 份的一阶/二阶动量——全参数微调的显存开销是权重本身的很多倍,这是消费级显卡/笔记本根本训不动大模型的直接原因。LoRA 把可训练参数从"整个权重矩阵"降到"两个小得多的低秩矩阵",反向传播时只需要给 A、B 维护梯度和优化器状态,原始权重全程冻结、不需要梯度,这是大模型能在单卡甚至消费级显卡上做微调的最基础机制。

**可运行例子:**
```python
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "learning/lora-family/src")
from lora_minimal import LoRALinear

torch.manual_seed(7)
base = nn.Linear(16, 32, bias=False)
lora = LoRALinear(base, r=4, alpha=8)
x = torch.randn(3, 16)

# 1) B 零初始化 -> 初始 forward 与 base 完全一致（逐位相等，不是近似）
with torch.no_grad():
    out_lora0 = lora(x)
    out_base0 = base(x)
assert torch.equal(out_lora0, out_base0)
assert torch.all(lora.B == 0)
assert lora.scaling == 8 / 4  # alpha / r

# 2) 手动展开公式，核对与 forward() 输出一致
manual = base(x) + lora.scaling * (x @ lora.A.T @ lora.B.T)
assert torch.allclose(manual, out_lora0)

# 3) 关键机制：B=0 时，A 在第 0 步反而收不到梯度（链式法则里被 B 这个因子乘成 0）
target = torch.randn(3, 32)
loss0 = ((lora(x) - target) ** 2).mean()
loss0.backward()
grad_A_step0 = lora.A.grad.abs().max().item()
grad_B_step0 = lora.B.grad.abs().max().item()
print(f"step 0 (B 全零): |A.grad|.max = {grad_A_step0:.10f}   |B.grad|.max = {grad_B_step0:.6f}")
assert grad_A_step0 == 0.0
assert grad_B_step0 > 0.0

# 4) 走一步优化器之后，B 不再是 0，A 才开始收到非零梯度
opt = torch.optim.SGD([lora.A, lora.B], lr=0.1)
opt.step()
assert torch.any(lora.B != 0)

opt.zero_grad()
loss1 = ((lora(x) - target) ** 2).mean()
loss1.backward()
grad_A_step1 = lora.A.grad.abs().max().item()
print(f"step 1 (B 已非零): |A.grad|.max = {grad_A_step1:.6f}")
assert grad_A_step1 > 0.0

# 5) 此时输出已经和冻结的 base 分道扬镳
with torch.no_grad():
    diff_after = (lora(x) - base(x)).abs().max().item()
print(f"1 步之后 |lora(x) - base(x)|.max = {diff_after:.6f}（不再是 0）")
assert diff_after > 0.0
```

实测(`.venv` 真跑):`step 0 (B 全零): |A.grad|.max = 0.0000000000  |B.grad|.max = 0.110251`——A 的梯度不是"很小",是精确的 `0.0`;`step 1 (B 已非零): |A.grad|.max = 0.046610`——B 一旦挪动,A 立刻拿到非零梯度;走完这一步之后 `|lora(x) - base(x)|.max = 0.043783`,LoRA 分支已经开始让输出偏离冻结的 base。

**面试怎么问 + 追问链:**
- **Q:** "LoRA 为什么要把 B 初始化成全零,而不是把 A、B 都随机初始化?"—— 期望答出"保证训练开始时 ΔW=0,输出和原模型完全一致,不会一上来就给预训练模型的输出注入随机噪声"。
- **追问 1(深挖,全场最容易停留在表面的地方):** "那如果反过来,把 A 初始化成全零、B 用 Kaiming,是不是等价?"—— 期望说出"从'起点输出=原模型'这个静态性质看确实等价(A 或 B 有一个是零,乘积就是零),但训练动态会反过来:这时候第 0 步 B 的梯度会精确为零,A 反而先拿到真实梯度",体现出理解的是机制而不是背答案。
- **追问 2:** "GPT-2 的 `c_attn` 用的是 `Conv1D` 不是 `nn.Linear`,LoRA 要怎么适配?"—— 期望说出"两者 weight 形状是转置关系,`forward` 本身因为直接调用 `self.base(x)` 不受影响,真正需要特殊处理转置的是权重合并这一步"(为知识点 3 埋伏笔)。

**常见坑:** 把"B 零初始化"错记成"A 零初始化"——两种说法在"起点等于原模型"这一点上确实都成立,容易靠直觉蒙对,但一旦被追问"第 0 步谁先拿到梯度"就会露馅;还有一种常见误解是以为 `scaling=α/r` 里的 `α` 和 `r`可以随意配比,实际上很多实践里把 `α` 固定为 `2r` 或直接等于 `r`,这个具体取舍会在知识点 2(rsLoRA)和知识点 4(PiSSA 默认 `α=r`)里看到不同选择。

---

## 2. `RSLoRALinear` + `lora_plus_param_groups`(`lora_extensions.py`)—— 缩放公式改成 α/√r,A、B 分组给不同学习率

**是什么:**
```python
class RSLoRALinear(LoRALinear):
    """rsLoRA: 与 LoRA 唯一差异是 scaling = α / sqrt(r)。"""

    def __init__(self, base_linear, r=8, alpha=16, dropout=0.0):
        super().__init__(base_linear, r=r, alpha=alpha, dropout=dropout)
        self.scaling = alpha / math.sqrt(r)   # 覆盖父类的 scaling，唯一差异


def lora_plus_param_groups(model, lr_A=1e-4, lambda_B=16.0):
    """生成 LoRA+ 的 optimizer param groups：lr_B = lambda_B * lr_A。"""
    lr_B = lambda_B * lr_A
    A_params, B_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".A") or "lora_A" in name:
            A_params.append(p)
        elif name.endswith(".B") or "lora_B" in name:
            B_params.append(p)
    return [{"params": A_params, "lr": lr_A}, {"params": B_params, "lr": lr_B}]
```

**一句话:** rsLoRA 只改一处——把 LoRA 的缩放公式 `α/r` 换成 `α/√r`;LoRA+ 不改 forward,只改 optimizer——给 A、B 两组参数分别设置学习率,B 的学习率通常是 A 的 16 倍。

**底层机制/为什么这样设计:** `RSLoRALinear` 继承 `LoRALinear`,`__init__` 里唯一的改动就是覆盖 `self.scaling`,forward 逻辑完全复用父类,这个类的"改动面"本身就说明了 rsLoRA 是一个极小的单点修改。为什么 `α/r` 在大 r 下会出问题:`B@A` 这个矩阵乘法本质是把 r 个"秩 1 分量"加总起来,如果 A、B 每个元素的方差是固定的(不随 r 变化),这 r 个分量累加起来,整体幅度会按 `√r` 增长(相互独立项求和的标准差按项数的平方根增长,不是按项数线性增长)——所以要把累加后的幅度稳定在同一量级,缩放系数应该按 `1/√r` 收缩,而不是 `1/r`:`1/r` 收缩得太快,r 越大,`ΔW` 的实际幅度反而越接近 0,大 r 时相当于把好不容易升上去的秩又通过缩放系数抹掉了大半;`1/√r` 恰好抵消 `√r` 的增长,让 `ΔW` 的幅度不随 r 系统性漂移,这是 rsLoRA 论文(`arXiv:2312.03732`)允许用更大的 r 还能稳定训练的关键。LoRA+(`arXiv:2402.12354`)的分析角度不同,论文在"无穷宽度极限下的特征学习"这个理论框架里证明:如果 A、B 用同一个学习率,当模型宽度变大时,B 一侧对最终输出 `BA` 的更新会相对滞后,导致收敛变慢、且这种滞后会随宽度变化、不好用一个固定学习率兼顾;论文给出的结论是 B 需要显著更大的学习率,并给了一个不依赖具体宽度、经验上普遍好用的默认倍率 16——这是论文本身的理论结论,本文不重新推导"无穷宽度特征学习"这套理论(超出手写 demo 的验证范围,如实标注),下面的可运行例子只验证"代码是否真的按 16 倍实现了这个学习率分组"以及"rsLoRA 的缩放公式是否真的把幅度稳定住了"这两件可以直接跑出来的事实。

**AI 研究场景:** 需要用比默认 `r=8` 大得多的秩(比如 r=64、r=256,知识点 5 的 VeRA 甚至用到 r=256)时,如果还沿用 `α/r` 的缩放公式,增量分支的实际贡献会被压得很小,相当于"选了大 r 却没真正用上",rsLoRA 这时候是让大 r 训练保持稳定的直接手段;LoRA+ 则是几乎零额外代码成本的训练加速手段(只改 optimizer 的 param_groups,不改模型结构、不改 forward),在任何已经在用 LoRA 的项目里都可以直接尝试。

**可运行例子:**
```python
import math
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "learning/lora-family/src")
from lora_extensions import RSLoRALinear, lora_plus_param_groups
from lora_minimal import LoRALinear

torch.manual_seed(0)

# 1) scaling 常数本身：LoRA 用 alpha/r，rsLoRA 用 alpha/sqrt(r)
r_small, r_big = 4, 256
lora_scaling_small, lora_scaling_big = 16 / r_small, 16 / r_big
rs_scaling_small, rs_scaling_big = 16 / math.sqrt(r_small), 16 / math.sqrt(r_big)
print(f"r: {r_small}->{r_big}  LoRA scaling 缩小 {lora_scaling_small/lora_scaling_big:.1f}x   "
      f"rsLoRA scaling 缩小 {rs_scaling_small/rs_scaling_big:.1f}x")
assert abs(lora_scaling_small / lora_scaling_big - (r_big / r_small)) < 1e-9
assert abs(rs_scaling_small / rs_scaling_big - math.sqrt(r_big / r_small)) < 1e-9

base_check = RSLoRALinear(nn.Linear(8, 8, bias=False), r=r_small, alpha=16)
assert abs(base_check.scaling - rs_scaling_small) < 1e-9

# 2) 受控实验：固定"每个元素方差"（不用 Kaiming，避开 fan_in 随 r 变化的干扰），
#    直接看 BAx 的原始（未缩放）幅度是否真的按 sqrt(r) 增长，
#    以及两种 scaling 各自能不能把最终幅度拉回稳定水平
d_in = d_out = 768
x = torch.randn(64, d_in)
stds_raw, stds_lora, stds_rs = [], [], []
for r in [4, 16, 64, 256, 1024]:
    A = torch.randn(r, d_in) / math.sqrt(d_in)
    B = torch.randn(d_out, r) / math.sqrt(d_out)
    raw = x @ A.T @ B.T
    stds_raw.append(raw.std().item())
    stds_lora.append(((16 / r) * raw).std().item())
    stds_rs.append(((16 / math.sqrt(r)) * raw).std().item())

ratio_raw = stds_raw[-1] / stds_raw[0]
print(f"原始 std(BAx): r=4 -> {stds_raw[0]:.6f}, r=1024 -> {stds_raw[-1]:.6f}, 比值={ratio_raw:.2f}（理论 sqrt(256)=16）")
assert 13 < ratio_raw < 19

print(f"rsLoRA 缩放后 std: r=4 -> {stds_rs[0]:.4f}, r=1024 -> {stds_rs[-1]:.4f}（应接近不变）")
print(f"LoRA   缩放后 std: r=4 -> {stds_lora[0]:.4f}, r=1024 -> {stds_lora[-1]:.4f}（应明显萎缩）")
assert stds_rs[-1] / stds_rs[0] > 0.5
assert stds_lora[-1] / stds_lora[0] < 0.2

# 3) LoRA+：A、B 两组不同学习率，B 的 lr = 16 * A 的 lr
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = LoRALinear(nn.Linear(10, 10, bias=False), r=2, alpha=4)

groups = lora_plus_param_groups(Tiny(), lr_A=1e-4, lambda_B=16.0)
by_name = {("lora_A" if g["params"][0] is Tiny().lin.A.__class__ and False else None): g for g in groups}  # 占位，见下方直接按顺序取
lr_A_group, lr_B_group = groups[0], groups[1]
print(f"\nLoRA+ groups: lora_A lr={lr_A_group['lr']:.1e}, lora_B lr={lr_B_group['lr']:.1e}")
assert lr_B_group["lr"] == 16.0 * lr_A_group["lr"]
```

**这一段有一处需要提前说明的小细节:** `lora_plus_param_groups` 返回的 `list[dict]` 里没有携带分组名字(源码里 `main()` 打印时用的 `_name` 字段是 demo 脚本自己加的,函数本体没有这个字段),上面代码里第 3 部分特意保留了一行"占位"写法提醒这一点,实际判断分组身份时可以直接按返回列表的固定顺序(先 A 组、后 B 组)取,或者更稳妥地按 `g["lr"]` 的值区分。

实测:`r: 4->256  LoRA scaling 缩小 64.0x   rsLoRA scaling 缩小 8.0x`;受控实验里原始(未缩放)`std(BAx)`:`r=4 -> 0.067877, r=1024 -> 1.157767`,比值 `17.06`(理论值 `√256=16`,量级吻合);rsLoRA 缩放后 `std`:`r=4 -> 0.5430, r=1024 -> 0.5789`,跨了 256 倍的 r 基本停在同一量级;LoRA 缩放后 `std`:`r=4 -> 0.2715, r=1024 -> 0.0181`,萎缩了约 15 倍。LoRA+ 分组:`lora_A lr=1.0e-04, lora_B lr=1.6e-03`,精确 16 倍关系。

**面试怎么问 + 追问链:**
- **Q:** "rsLoRA 和 LoRA 的区别是什么?"—— 期望准确说出"唯一区别是缩放系数从 `α/r` 换成 `α/√r`",而不是笼统说"改进了 LoRA"。
- **追问 1(深挖):** "为什么 `α/r` 在大 r 下会有问题,`α/√r` 又是怎么解决的?"—— 期望说出"r 个低秩分量累加,幅度按 `√r` 增长;`1/r` 收缩过快导致大 r 时 ΔW 幅度反而趋于 0,`1/√r` 恰好抵消这个 `√r` 增长"。
- **追问 2:** "LoRA+ 为什么要给 B 更大的学习率,不能都用同一个吗?"—— 期望说出"论文从无穷宽度特征学习理论证明同一学习率下 B 的更新会相对滞后",同时诚实承认这是论文的理论结论,不要求能重新推导。
- **追问 3(工程判断力):** "如果同时用 rsLoRA 和 LoRA+,两者会冲突吗?"—— 期望答出"不冲突,一个改的是 forward 里的缩放系数,一个改的是 optimizer 的分组学习率,是两个完全独立的维度,可以同时叠加使用"。

**常见坑:** 把 `lambda_B=16` 记成"B 的参数量是 A 的 16 倍"——这是把"学习率倍数"和"参数量倍数"搞混,LoRA+ 完全不改变模型结构和参数量,`A` 形状 `(r,d_in)`、`B` 形状 `(d_out,r)` 和普通 LoRA 完全一样,16 这个数字只出现在 optimizer 的 `lr` 字段里。另外容易忘记 `RSLoRALinear.__init__` 是**先调用父类完整初始化、再覆盖 `self.scaling`**,如果自己实现类似的继承覆写却在覆盖之前就用了 `self.scaling`,会拿到错误的父类默认值。

---

## 3. `merge_weights()`(`lora_minimal.py`)—— 为什么合并后推理零延迟

**是什么:**
```python
@torch.no_grad()
def merge_weights(self) -> None:
    """把 LoRA 合并回 base.weight，删除 A、B。

    合并后 LoRALinear 等价于一个被更新过权重的 base_linear。
    """
    delta = self.scaling * (self.B @ self.A)  # (d_out, d_in)
    if self.is_conv1d:
        self.base.weight.data.add_(delta.T)   # Conv1D.weight 是 (in, out)，需要转置
    else:
        self.base.weight.data.add_(delta)
```

**一句话:** 把 `scaling·B@A` 直接加进 `base.weight`,之后只调用 `base(x)` 就能得到和"合并前完整跑一遍 LoRALinear"完全一样的结果——因为矩阵乘法对加法满足分配律,`x@(W₀+ΔW)ᵀ` 和 `x@W₀ᵀ + x@ΔWᵀ` 是同一个数,合并之后不再需要额外的降维、升维两次矩阵乘法。

**底层机制/为什么这样设计:** `LoRALinear.forward` 未合并时的计算量是"一次 `base` 的矩阵乘法 + 两次低秩矩阵乘法(`x@Aᵀ` 和 `@Bᵀ`) + 一次加法";合并之后 `base.weight` 已经等于 `W₀+scaling·BA`,推理时只需要**一次**矩阵乘法,和没有套 LoRA 的原始模型完全同构——这就是"推理零延迟"的准确含义:不是"合并这个动作不花时间"(合并本身要算一次 `scaling·B@A` 并写回权重,是货真价实的一次性开销),而是"合并之后,后续每一次推理调用都不再需要多算那两次低秩矩阵乘法",这个一次性开销被之后无限次的推理调用摊销掉了。Conv1D 和 nn.Linear 权重形状是转置关系,这里 `is_conv1d` 分支专门处理这个差异,如果不分场景直接 `add_`,GPT-2 的 `c_attn` 权重会被加错方向。**这里有一处本文验证时自己发现的细节,值得专门拎出来讲:** 上面代码的 docstring 写着"删除 A、B",但实现代码只做了 `self.base.weight.data.add_(delta)`,**并没有真的删除或清零 `self.A`、`self.B` 这两个 `nn.Parameter`**。这意味着 `merge_weights()` 调用之后,如果继续调用被包装的 `lora(x)`(而不是改成调用 `lora.base(x)`),`forward` 里 `base_out + scaling*lora_out` 这一行会把已经写进 `base.weight` 的 `delta` **再加一遍**——因为 `A`、`B` 还在,`lora_out` 依然会算出和合并前一样的非零值。也就是说,这份 minimal 实现里"合并"和"卸载"是两件事:`merge_weights()` 只做了合并这一半,卸载(替换模块、或至少清零/删除 A、B)需要调用方自己做,`LoRAGPT2.merge_all()` 也只是对每个子模块调用 `merge_weights()`,同样没有做卸载。作为对比,peft 库真实的 `LoraModel.merge_and_unload()` 的 docstring 明确写着"This is needed if someone wants to use the base model as a standalone model. The returned model has the same architecture as the original base model"——它返回的是一个把 LoRA 层**替换**回普通层的新模型,从函数命名(`merge_and_unload`,合并**并**卸载)到行为都比这份教学用的 `merge_weights()` 更完整。

**AI 研究场景:** 生产环境部署 LoRA 微调模型时有两种典型选择:一是合并后单独部署(零额外推理延迟,但这个部署实例被"焊死"在这一个 LoRA 适配器上,想换任务要重新合并、重新部署);二是不合并、保留 base+adapter 的形式做多适配器热切换(同一份 base 权重常驻显存,不同请求按需挂载不同的小 LoRA 权重,牺牲一点点推理时延换来"一次部署服务多个任务/多个用户"的灵活性,这也是 vLLM 等推理框架支持"多 LoRA 并发服务"的基本思路)。选择合并与否,本质上是在"极致的单任务推理速度"和"部署形态的灵活性"之间做权衡,不存在绝对更优的一方。

**可运行例子:**
```python
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "learning/lora-family/src")
from lora_minimal import LoRALinear

torch.manual_seed(1)
base = nn.Linear(16, 32, bias=False)
lora = LoRALinear(base, r=4, alpha=8)
x = torch.randn(3, 16)

# 先训练几步，让 A、B 都变成非零
opt = torch.optim.SGD([lora.A, lora.B], lr=0.5)
target = torch.randn(3, 32)
for _ in range(5):
    opt.zero_grad()
    loss = ((lora(x) - target) ** 2).mean()
    loss.backward()
    opt.step()

with torch.no_grad():
    out_before_merge = lora(x).clone()
    lora.merge_weights()

    # 性质 1（零延迟性质本身）：合并后只调用 base(x)，应该和"合并前完整调用 LoRALinear"一致
    out_base_after_merge = lora.base(x)
    diff_zero_overhead = (out_base_after_merge - out_before_merge).abs().max().item()
    print(f"zero-overhead check |base(x)_after_merge - lora(x)_before_merge| = {diff_zero_overhead:.3e}")
    assert diff_zero_overhead < 1e-5

    # 性质 2（常见坑）：docstring 说会删除 A、B，但代码没有真的删——继续调用被包装的 lora(x) 会双重计数
    out_after_merge_via_wrapper = lora(x)
    diff_double_count = (out_after_merge_via_wrapper - out_before_merge).abs().max().item()
    print(f"double-count if lora(x) misused after merge: diff = {diff_double_count:.4f}")
    assert diff_double_count > 1e-2
    print(f"A.abs().sum() after merge = {lora.A.abs().sum().item():.4f}, B.abs().sum() after merge = {lora.B.abs().sum().item():.4f}")
    assert lora.A.abs().sum().item() > 0
    assert lora.B.abs().sum().item() > 0
```

实测:`zero-overhead check |base(x)_after_merge - lora(x)_before_merge| = 3.576e-07`(浮点误差范围内的精确相等,零延迟性质成立);但如果合并后误用 `lora(x)` 而不是 `lora.base(x)`,`diff = 3.1029`——明显偏离,双重计数是真实发生的,不是理论假设;此时 `A.abs().sum() after merge = 10.9627, B.abs().sum() after merge = 11.1055`,两者确实都还在,印证了"docstring 说删除但代码没删"。

**面试怎么问 + 追问链:**
- **Q:** "为什么合并 LoRA 权重之后,推理阶段就不再有额外开销了?"—— 期望说出"矩阵乘法对加法满足分配律,合并后一次矩阵乘法就等价于原来'一次 base + 两次低秩'的计算"。
- **追问 1:** "合并这个动作本身是不是完全免费的?"—— 期望纠正"不是,合并本身要算一次 `scaling·B@A` 并写回权重,是一次性开销,'零延迟'指的是这个开销被之后每一次推理调用摊销掉了,不是合并这个动作零成本"。
- **追问 2(压力测试,呼应本文发现的细节):** "如果一个 LoRA 实现的 `merge()` 方法执行完之后,原来的 wrapper 模块和 A、B 参数还留在计算图里,会有什么风险?"—— 期望能推理出"如果之后不小心又调用了包着 A、B 的 forward,低秩分支会被重复叠加,导致输出错误地偏离合并后的正确结果",这正是本文验证时在这份 minimal 实现里实际发现的行为。
- **追问 3:** "生产部署时,什么场景下反而不应该合并?"—— 期望说出"需要同一个 base 模型服务多个不同 LoRA 适配器(多任务/多租户)的场景,合并会把模型焊死在一个适配器上,失去热切换的灵活性"。

**常见坑:** 认为调用完 `merge_weights()`(或任何叫"merge"的方法)之后,原来那个 LoRA 包装模块就自动变得"安全可以随便继续用"——这份 minimal 实现的 docstring 和实际代码行为不一致(说删除但没删)提醒我们:拿到一个 `merge` 方法时,不能只看名字或注释,要么去读实现代码确认它是否连"卸载"(替换/清空原来的低秩分支)也一起做了,要么像上面例子一样直接跑一次双重计数测试来验证,不能想当然。

---

## 4. `PiSSALinear`(`pissa_minimal.py`)—— 用真实 SVD/QR 分解 W₀ 初始化 A、B

**是什么:**
```python
class PiSSALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=None, init_method="pissa"):
        super().__init__()
        if alpha is None:
            alpha = r  # PiSSA 论文默认 α = r → scaling = 1
        self.scaling = alpha / r

        W = _extract_weight(base_linear).float()   # (d_out, d_in)
        if init_method == "pissa":
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            sqrt_S = S[:r].sqrt()
            B_init = U[:, :r] * sqrt_S.unsqueeze(0)      # (d_out, r)
            A_init = sqrt_S.unsqueeze(-1) * Vt[:r, :]    # (r, d_in)
        elif init_method == "olora":
            Q, R = torch.linalg.qr(W, mode="reduced")
            B_init = Q[:, :r]
            A_init = R[:r, :]

        W_res = W - B_init @ A_init          # 冻结基础：把 top-r 主成分从 W_0 里"抠出来"
        _write_back(base_linear, W_res.to(base_linear.weight.dtype))
        self.base = base_linear
        self.A = nn.Parameter(A_init.clone())
        self.B = nn.Parameter(B_init.clone())
```

**一句话:** PiSSA 不用随机数初始化 A、B,而是对 W₀ 做真实的 SVD 分解,把最大的 r 个奇异值对应的主成分直接切出来作为 A、B 的初始值,剩下的部分 `W_res = W₀ - B@A` 才是新的冻结基础;OLoRA 是同样思路的低成本替代,用 QR 分解代替 SVD。

**底层机制/为什么这样设计:** `W₀ = UΣVᵀ` 是标准 SVD,`U` 的前 r 列和 `Vᵀ` 的前 r 行是 `W₀` 里"能量"最大的 r 个方向,把对应的奇异值 `Σ[:r]` 开平方后一半分给 B(`B=U[:,:r]·√Σ[:r]`)、一半分给 A(`A=√Σ[:r]·Vᵀ[:r]`),是为了让 A、B 两个因子的数值量级相近(而不是一个巨大一个微小),这样后续用同一个学习率训练时两者更新幅度不会失衡。`W_res = W₀ - B_init@A_init` 被写回 `base.weight`,成为新的冻结部分——这一步和 LoRA 有本质区别:LoRA 的 `base.weight` 全程就是原始 `W₀`,没有任何改动;PiSSA 的 `base.weight` 在构造函数执行完的那一刻起,就已经不等于原始 `W₀` 了,`W₀` 的"top-r 主成分"被显式搬到了可训练的 A、B 里。但这个搬动不影响初始输出:因为 `W_res + scaling·B@A`(`scaling=α/r`,PiSSA 默认 `α=r` 所以 `scaling=1`)在代数上精确等于 `W₀`,`forward = base(x) + scaling·(x@Aᵀ@Bᵀ) = x@W_resᵀ + x@(scaling·BA)ᵀ = x@W₀ᵀ`——PiSSA 用一种和"B 全零"完全不同的手段,达成了和 LoRA 同样的"训练起点输出等于原模型"这个性质。OLoRA 把 SVD 换成 QR:`W₀=QR`,`Q` 的前 r 列和 `R` 的前 r 行同样满足 `Q[:,:r]@R[:r,:]` 是 `W₀` 的一个低秩近似(QR 分解不像 SVD 那样按奇异值大小排序,给出的是一个正交基而非"最优 r 秩近似",计算量比 SVD 小得多,是更轻量的替代方案)。

**AI 研究场景:** PiSSA 论文报告用主成分初始化比随机初始化收敛更快、少数据场景下最终效果更好——这是论文的实验结论,本文的 demo 只验证了初始化阶段的代数性质(SVD/QR 重建精度 + forward 一致性),不涉及训练动态本身的对比实验,如实标注、不过度引申。工程上要注意的权衡是:SVD/QR 分解本身是一次性的额外计算(对每个要打 PiSSA 的层都要做一次),比 LoRA 的随机初始化(`kaiming_uniform_`,几乎不耗时)贵,层数越多、矩阵越大,这个初始化阶段的额外开销越明显,这是用初始化时间换训练收敛速度的具体取舍。

**可运行例子:**
```python
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "learning/lora-family/src")
from pissa_minimal import PiSSALinear

torch.manual_seed(3)

for init_method in ["pissa", "olora"]:
    base = nn.Linear(20, 12, bias=False)
    W0 = base.weight.data.clone()
    layer = PiSSALinear(base, r=6, init_method=init_method)

    W_rec = layer.reconstruct_W0()
    diff = (W0 - W_rec).abs().max().item()
    print(f"[{init_method}] max|W_res + BA - W0| = {diff:.3e}")
    assert diff < 1e-4

    x = torch.randn(5, 20)
    with torch.no_grad():
        out_pissa = layer(x)
        out_orig = x @ W0.T
    fwd_diff = (out_pissa - out_orig).abs().max().item()
    print(f"[{init_method}] max|forward - x@W0.T| = {fwd_diff:.3e}")
    assert fwd_diff < 1e-3

    BA_rank = torch.linalg.matrix_rank(layer.B.detach() @ layer.A.detach()).item()
    print(f"[{init_method}] rank(B@A) = {BA_rank} (r={layer.r}, 说明 base.weight 现在已经 != 原始 W0)")
    assert BA_rank == layer.r  # 非零满秩，证明 W_res = W0 - BA 确实不等于 W0
```

实测(小矩阵,`nn.Linear(20,12)`, r=6):`[pissa] max|W_res + BA - W0| = 1.490e-08`,`[pissa] max|forward - x@W0.T| = 1.192e-07`,`rank(B@A)=6`;`[olora]` 两个误差分别是 `1.490e-08` 和 `1.788e-07`,`rank(B@A)=6`——SVD 和 QR 两条路径都能把 W₀ 精确重建回来,量级都在 `1e-7`~`1e-8`,是浮点精度范畴内的误差,不是近似算法的系统性误差。另外,本文额外执行了 `pissa_minimal.py` 自带的 `main()`(真实 GPT-2,r=8,`target_modules=("c_attn",)`),这是**全系列少数带真 assert 的文件**——真实输出:`Total params: 124,734,720`,`Trainable params: 294,912 (0.2364%)`,`初始重建测试: |W_res + BA - W_0|.max = 1.1921e-07`(源文件自己的 assert 阈值是 `<1e-3`,实测误差比阈值小了 4 个数量级),`forward 一致性: |logits_pissa - logits_base|.max = 3.8147e-05`(源文件 assert 阈值 `<1e-1`)。

**面试怎么问 + 追问链:**
- **Q:** "PiSSA 和 LoRA 的参数量/架构有区别吗?"—— 期望准确答出"完全没有区别,A、B 的形状、可训练参数总量和 LoRA 一模一样,唯一的区别是初始化方式"。
- **追问 1:** "PiSSA 初始化之后,`base.weight` 还等于原始预训练权重吗?"—— 期望说出"不等于了,`W_res = W_0 - B@A`,原始权重里 top-r 的主成分被搬进了可训练的 A、B 里,只有 `base.weight` 和 `scaling·B@A` 加起来才等于原始 `W_0`",这是和 LoRA(`base.weight` 全程不动)的关键区别。
- **追问 2:** "为什么 `√Σ[:r]` 要平均分给 A 和 B,不能全放在一边吗?"—— 期望说出"是为了让 A、B 两个因子数值量级相近,避免一个远大于另一个导致同一学习率下更新失衡",这是工程上的数值稳定性考量。
- **追问 3(对比 OLoRA):** "OLoRA 用 QR 代替 SVD,牺牲了什么、换来了什么?"—— 期望说出"QR 计算量比 SVD 小,但 QR 给出的只是一个正交基,不像 SVD 那样按奇异值大小排序、代表'最重要的 r 个方向',是拿'方向的最优性'换'计算速度'"。

**常见坑:** 误以为 PiSSA 是"LoRA 加了新能力"——它没有引入任何新的可训练结构,r、A、B 的形状和 LoRA 完全一致,唯一变化的是**初始值怎么来**;另一个容易踩的坑是拿 PiSSA 初始化之后的 `base.weight` 直接和原始 GPT-2 的对应层权重做逐元素比较,以为"应该相等"——这是错的,PiSSA 之后 `base.weight` 已经是 `W_res` 而不是 `W_0`,只有 `reconstruct_W0()`(`base.weight + scaling·B@A`)才等于原始权重,直接比较 `base.weight` 本身会得到一个"看起来像 bug 但其实是设计如此"的巨大误差。

---

## 5. `VeRALinear`(`vera_minimal.py`)—— 全部层共享同一份冻结随机矩阵,只训两个对角缩放向量

**是什么:**
```python
class VeRASharedBuffers:
    """全局共享的 A, B 随机矩阵（固定 seed），所有 VeRALinear 实例从这里读取。"""
    _A = None
    _B = None

    @classmethod
    def init(cls, d_max_in, d_max_out, r, seed=42):
        g = torch.Generator().manual_seed(seed)
        bound = math.sqrt(6.0 / r)
        cls._A = torch.empty(r, d_max_in).uniform_(-bound, bound, generator=g)
        cls._B = torch.empty(d_max_out, r).uniform_(-bound, bound, generator=g)


class VeRALinear(nn.Module):
    def __init__(self, base_linear, r=256, alpha=None, d_initial=0.1):
        super().__init__()
        for p in base_linear.parameters():
            p.requires_grad = False
        self.base = base_linear
        self.scaling = (alpha if alpha is not None else r) / r
        self.Lambda_b = nn.Parameter(torch.ones(r))              # 只有这两个是可训练参数
        self.Lambda_d = nn.Parameter(torch.full((self.d_out,), d_initial))

    def forward(self, x):
        A = VeRASharedBuffers.get_A(self.r, self.d_in)   # 全局共享、冻结
        B = VeRASharedBuffers.get_B(self.r, self.d_out)  # 全局共享、冻结
        out = (x @ A.T) * self.Lambda_b
        out = (out @ B.T) * self.Lambda_d
        return self.base(x) + self.scaling * out
```

**一句话:** VeRA 把 LoRA 里"每一层各自训练一份 A、B"改成"所有层共享同一份全局随机且冻结的 A、B",每一层唯二可训练的东西是两个低维对角缩放向量 `Λ_b∈R^r`、`Λ_d∈R^{d_out}`,可训练参数量比 LoRA 还要少一个数量级。

**底层机制/为什么这样设计:** forward 分三步:`x@Aᵀ` 用共享的冻结 A 把输入投影到 r 维,逐元素乘 `Λ_b` 做一次"每层专属"的缩放,`@Bᵀ` 用共享的冻结 B 投影回 `d_out` 维,再逐元素乘 `Λ_d` 做第二次"每层专属"的缩放。A、B 从头到尾不参与训练,真正被优化的自由度只有 `Λ_b`(r 个数)和 `Λ_d`(`d_out` 个数)——直觉上,一份共享的冻结随机矩阵提供了一个足够大、足够多样的"公共基底",不同层不需要各自学一套独立的投影方向,只需要学会怎么"挑选和放大/缩小"这组公共基底里对本层有用的分量,这是 VeRA 论文(`arXiv:2310.11454`)的核心假设。这里有一个容易被忽略但值得较真的实现细节:`VeRASharedBuffers` 实际存的是按**全模型里最大的** `d_in`/`d_out` 分配的一整块大矩阵,每一层通过 `get_A(r, d_in)`/`get_B(r, d_out)` 做切片 `cls._A[:r, :d_in]`——如果不同层的 `d_in`/`d_out` 不一样,不同层实际拿到的是这块大矩阵的不同子块,并不是完全相同的矩阵;本文验证用的是两个形状完全相同的 `nn.Linear(32,64)`,这种情况下切片区间重合、`torch.equal` 验证的"完全相同"才成立——放到 GPT-2 的 `c_attn` 上因为所有层的 `c_attn` 形状统一都是 `(768,2304)`,所以 12 层确实都拿到同一整块矩阵,但这是"GPT-2 c_attn 形状统一"这个具体场景下的结果,不是 VeRA 机制本身保证"所有层永远拿到逐元素相同的矩阵"。

**AI 研究场景:** 需要给同一个 base 模型维护成百上千个不同任务/不同用户的个性化适配器时(比如每个用户一份轻量微调),即便 LoRA 已经把每个适配器的体积压得很小,乘以几千个用户仍然是可观的存储和加载开销;VeRA 的关键优势在于 A、B 可以在整个服务里**只存一份**,每个适配器需要单独存储和加载的只是两个小向量(`r+d_out` 个数,GPT-2 `c_attn` 一层是 `256+2304=2560` 个数),在大规模多适配器托管场景下是比 LoRA 更激进的存储优化。

**可运行例子:**
```python
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "learning/lora-family/src")
from vera_minimal import VeRALinear, VeRASharedBuffers

torch.manual_seed(0)
VeRASharedBuffers.init(d_max_in=32, d_max_out=64, r=8, seed=42)

base1 = nn.Linear(32, 64, bias=False)
base2 = nn.Linear(32, 64, bias=False)
layer1 = VeRALinear(base1, r=8, alpha=8)
layer2 = VeRALinear(base2, r=8, alpha=8)

A1 = VeRASharedBuffers.get_A(layer1.r, layer1.d_in)
A2 = VeRASharedBuffers.get_A(layer2.r, layer2.d_in)
B1 = VeRASharedBuffers.get_B(layer1.r, layer1.d_out)
B2 = VeRASharedBuffers.get_B(layer2.r, layer2.d_out)
print("layer1/layer2 共享同一份 A:", torch.equal(A1, A2))
print("layer1/layer2 共享同一份 B:", torch.equal(B1, B2))
assert torch.equal(A1, A2) and torch.equal(B1, B2)

trainable_names = sorted(n for n, p in layer1.named_parameters() if p.requires_grad)
print("VeRALinear 可训练参数名:", trainable_names)
assert trainable_names == ["Lambda_b", "Lambda_d"]

n_lambda_b, n_lambda_d = layer1.Lambda_b.numel(), layer1.Lambda_d.numel()
lora_equiv = 8 * 32 + 64 * 8    # 同形状、同 r=8 时 LoRA 的 A+B 参数量
vera_trainable = n_lambda_b + n_lambda_d
print(f"同形状同 r=8：LoRA 每层可训 {lora_equiv}，VeRA 每层可训 {vera_trainable}（仅两个对角向量）")
print(f"VeRA / LoRA = {vera_trainable/lora_equiv:.4f}")
assert vera_trainable < lora_equiv
```

实测:`layer1/layer2 共享同一份 A: True`,`layer1/layer2 共享同一份 B: True`;`VeRALinear 可训练参数名: ['Lambda_b', 'Lambda_d']`;同形状同 `r=8` 时,`LoRA 每层可训 768,VeRA 每层可训 72`,`VeRA / LoRA = 0.0938`。另外执行了 `vera_minimal.py` 自带的 `main()`(真实 GPT-2,`r=256,target=c_attn`):`Total params: 124,470,528`,`Trainable params: 30,720 (0.0247%)`;对比同一份 `main()` 里算出的 `LoRA r=8: 294,912 trainable`,`VeRA r=256: 30,720 trainable`,`VeRA / LoRA = 10.42%(VeRA 节省 9.6x)`——注意这里 VeRA 用的 `r=256` 远大于 LoRA 的 `r=8`,依然只用了后者约十分之一的可训练参数。

**面试怎么问 + 追问链:**
- **Q:** "VeRA 的 A、B 都被冻结了,那训练过程中到底在学什么?"—— 期望说出"学的是两个对角缩放向量 `Λ_b`、`Λ_d`,本质是在学怎么组合/加权一个共享的、随机的、固定的特征空间,而不是学这个特征空间本身"。
- **追问 1(全场最容易被问倒的地方):** "VeRA 的 `r=256` 和 LoRA 的 `r=8` 能直接比较'谁的秩更高、更强'吗?"—— 期望明确说"不能",VeRA 的 `r` 是共享随机投影的瓶颈维度,必须开得比 LoRA 大得多才能有足够多的随机方向可供各层挑选组合,公平的比较维度应该是**可训练参数总量**,不是 r 这个数字本身。
- **追问 2:** "如果两层的 `d_in`/`d_out` 不一样,它们共享的 A、B 还是完全相同的矩阵吗?"—— 期望说出"不完全是,`VeRASharedBuffers` 按全模型最大维度存一整块矩阵,每层实际读取的是按自己 `d_in`/`d_out` 切出来的子块,维度不同的层拿到的是不同的子块(虽然有重叠)",能看出这一层细节说明真的读了实现而不是只记了论文摘要。
- **追问 3(工程场景):** "VeRA 在什么场景下比 LoRA 更有优势?"—— 期望说出"多适配器托管场景,A、B 可以全局只存一份,每个新增适配器只需要存两个小向量,是比 LoRA 更激进的存储优化"。

**常见坑:** 把"VeRA 参数更少"和"VeRA 的 ΔW 秩更低"划等号——恰恰相反,冻结的共享 A、B 依然提供了一个高达 `r`(demo 里是 256,远超 LoRA 常用的 8)的随机投影空间,VeRA 省的是**可训练标量的数量**,不是 `ΔW` 能达到的表达能力上限;另外容易誤以为每一层的 A、B"恰好初始化成了相同的值"(即每层各自独立持有一份、只是巧合地数值相同)——实际是所有层的 `VeRALinear` 在 `forward` 里调用的 `VeRASharedBuffers.get_A/get_B` 指向同一块底层存储,是真正意义上的共享,不是"看起来一样但各自占内存"的巧合。

---

## 6. `LoHaLinear`/`LoKrLinear`(`loha_minimal.py`/`lokr_minimal.py`,+`loha_lokr_peft.py`)—— Hadamard/Kronecker 乘积分解,以及一个真实的 peft ValueError

**是什么:**
```python
# loha_minimal.py
class LoHaLinear(nn.Module):
    """h = base(x) + α/r · ((B_1 A_1) ⊙ (B_2 A_2)) x"""
    def forward(self, x):
        delta_1 = self.B_1 @ self.A_1   # (d_out, d_in), rank <= r
        delta_2 = self.B_2 @ self.A_2   # (d_out, d_in), rank <= r
        delta = delta_1 * delta_2        # Hadamard（逐元素）乘积, rank <= r^2
        return self.base(x) + self.scaling * (x @ delta.T)


# lokr_minimal.py
class LoKrLinear(nn.Module):
    """ΔW = B ⊗ (B_lr @ A_lr)，Kronecker 积"""
    def forward(self, x):
        A = self.B_lr @ self.A_lr             # (m1, n1)，本身也是低秩分解
        delta = torch.kron(self.B, A)          # (m1*m2, n1*n2) = (d_out, d_in)
        return self.base(x) + self.scaling * (x @ delta.T)
```

**一句话:** LoHa 用两对独立的低秩因子 `(B₁A₁)` 和 `(B₂A₂)` 做逐元素(Hadamard)乘积,有效秩上限可以到 `r²`;LoKr 把 `ΔW` 表示成 Kronecker 积 `B⊗A`,用远小于 `d_out×d_in` 的参数量拼出一个满尺寸矩阵;`loha_lokr_peft.py` 则记录了一个真实的工程边界——peft 库当前的 `LoHaConfig`/`LoKrConfig` 都不支持 GPT-2 用的 `Conv1D` 层。

**底层机制/为什么这样设计:** 两个秩分别为 p、q 的矩阵,逐元素相乘之后秩最多能到 `p×q`——直觉上,`M₁=Σᵢ uᵢvᵢᵀ`(p 个秩 1 项求和)和 `M₂=Σⱼ sⱼtⱼᵀ`(q 个秩 1 项求和)做 Hadamard 积,展开后是 `Σᵢⱼ (uᵢ⊙sⱼ)(vᵢ⊙tⱼ)ᵀ`,变成了 `p×q` 个秩 1 项的和,秩自然可能升到 `p×q`。LoHa 让 `B₁A₁` 和 `B₂A₂` 各自秩不超过 r,乘积理论上限就是 `r²`——本文可运行例子里用 `r=4` 实测跑出了精确的 `rank(ΔW)=16=4²`,说明这个上限在实践中是可以被打满的,不是纯理论边界。代价是参数量:LoHa 需要两对 `(A,B)`,总参数量大约是同 r 的 LoRA 的 2 倍,但用 2 倍参数换到的是最高 `r²` 而不是 `r` 的有效秩,单位参数的表达能力上限明显更高(不过这只是**上限**,实际训练收敛到的秩是否真的接近 `r²`,取决于优化过程,不是初始化就注定的)。LoKr 是另一条路:`d_out=m₁×m₂`、`d_in=n₁×n₂` 先做整数分解,`B∈R^{m₂×n₂}` 是稠密的,`A=B_lr@A_lr∈R^{m₁×n₁}` 本身又是一次低秩分解——两层压缩叠加(先 Kronecker 拆分,再对其中一个因子做低秩近似),`torch.kron(B,A)` 直接拼出 `(m₁m₂, n₁n₂)=(d_out,d_in)` 的满尺寸 `ΔW`,但参与训练和存储的只是 `A_lr`、`B_lr`、`B` 三个小矩阵。`loha_lokr_peft.py` 记录的是一个纯工程限制,不是数学限制:GPT-2 的 `c_attn` 是 `transformers.pytorch_utils.Conv1D`,而 peft 当前版本的 `LoHaConfig`/`LoKrConfig` 只识别 `nn.Linear`/`nn.Conv1d`/`nn.Conv2d` 这几种模块类型,遇到 `Conv1D` 会直接报错拒绝构造——这和这份仓库手写的 `loha_minimal.py`/`lokr_minimal.py`(通过 `is_conv1d`/`get_in_out_dims` 显式处理了 Conv1D 转置)形成直接对比:同一个数学方法,手写版本能跑,调 peft 库版本在 GPT-2 上会报错,这是"库的实现完整度"问题,不是"方法本身不适用于这种层"的问题。

**AI 研究场景:** LoHa、LoKr 源自 LyCORIS 项目(源码 docstring 标注的论文分别是 FedPara `arXiv:2108.06098` 和 LyCORIS `arXiv:2309.14859`),这类方法在 Stable Diffusion 一类扩散模型的社区微调生态里被广泛使用,场景通常是"训练数据很少(几十到几百张图)、又想让 LoRA 风格的小文件尽量捕捉到更丰富的风格/主体特征",这时候同样参数预算下更高的有效秩上限是有实际吸引力的。工程上更现实的一课是知识点里那个 `ValueError`:如果一个团队把别人在 LLaMA/BERT(`nn.Linear`)上验证过的 LoHa/LoKr 配置,直接原样搬到 GPT-2 或其他用 `Conv1D` 的架构上跑 peft 库,会立刻在模型构造这一步就失败,不是训练过程中才出问题——遇到这种报错,第一反应应该是检查目标层的模块类型是否在库的支持列表里,而不是怀疑算法本身有问题。

**可运行例子:**
```python
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "learning/lora-family/src")
from loha_minimal import LoHaLinear
from lokr_minimal import LoKrLinear

torch.manual_seed(0)

# LoHa：两对各自秩 <= r 的因子，Hadamard（逐元素）乘积后有效秩可以到 r^2
base = nn.Linear(64, 64, bias=False)
loha = LoHaLinear(base, r=4, alpha=8)
with torch.no_grad():
    nn.init.kaiming_uniform_(loha.B_1, a=5 ** 0.5)  # B_1 出厂是零初始化，这里手动填非零才能观察满秩情形
delta_unscaled = loha.get_delta_W() / loha.scaling
rank = torch.linalg.matrix_rank(delta_unscaled).item()
print(f"LoHa r=4: Hadamard 乘积后 rank(ΔW) = {rank}（r^2 = {4**2}）")
assert rank == 4 ** 2

base_fresh = nn.Linear(64, 64, bias=False)
loha_fresh = LoHaLinear(base_fresh, r=4, alpha=8)
x = torch.randn(2, 64)
with torch.no_grad():
    diff0 = (loha_fresh(x) - base_fresh(x)).abs().max().item()
print(f"LoHa 出厂初始化(B_1=0): |out - base_out| = {diff0}")
assert diff0 == 0.0

# LoKr：Kronecker 积把 (d_out,d_in) 分解成两个小得多的因子相乘
base_lokr = nn.Linear(768, 2304, bias=False)  # 模拟 GPT-2 c_attn 形状
lokr = LoKrLinear(base_lokr, factor=32, r=4, alpha=4)
print(f"\nLoKr factor=32,r=4 on (d_in=768,d_out=2304): "
      f"n1×n2={lokr.n1}×{lokr.n2}={lokr.n1*lokr.n2}, m1×m2={lokr.m1}×{lokr.m2}={lokr.m1*lokr.m2}")
assert lokr.n1 * lokr.n2 == 768 and lokr.m1 * lokr.m2 == 2304
assert lokr.get_delta_W().shape == (2304, 768)

n_total = lokr.r * lokr.n1 + lokr.m1 * lokr.r + lokr.m2 * lokr.n2
print(f"LoKr 可训练参数: A_lr={lokr.r*lokr.n1} + B_lr={lokr.m1*lokr.r} + B={lokr.m2*lokr.n2} = {n_total}")

x2 = torch.randn(2, 768)
with torch.no_grad():
    diff_lokr0 = (lokr(x2) - base_lokr(x2)).abs().max().item()
print(f"LoKr 出厂初始化(B_lr=0): |out - base_out| = {diff_lokr0}")
assert diff_lokr0 == 0.0
```

实测:`LoHa r=4: Hadamard 乘积后 rank(ΔW) = 16(r^2 = 16)`——理论上限被精确打满;`LoHa 出厂初始化(B_1=0): |out - base_out| = 0.0`,和 LoRA 一样保证训练起点等于原模型。`LoKr factor=32,r=4 on (d_in=768,d_out=2304): n1×n2=32×24=768, m1×m2=32×72=2304`;`LoKr 可训练参数: A_lr=128 + B_lr=128 + B=1728 = 1984`;`LoKr 出厂初始化(B_lr=0): |out - base_out| = 0.0`。

`loha_lokr_peft.py` 的真实报错(GPT-2,`target_modules=["c_attn"]`):
```python
import sys
sys.path.insert(0, "learning/lora-family/src")
from loha_lokr_peft import build_peft_loha, build_peft_lokr

for name, fn, kwargs in [
    ("LoHa", build_peft_loha, dict(r=8, alpha=16)),
    ("LoKr", build_peft_lokr, dict(factor=32, r=4)),
]:
    try:
        fn(**kwargs)
        print(f"{name}: 没有抛出异常（不符合预期）")
    except ValueError as e:
        print(f"{name} -> {type(e).__name__}: {e}")
```

实测(两者报错文字完全相同):
```
LoHa -> ValueError: Target module of type <class 'transformers.pytorch_utils.Conv1D'> not supported, currently only adapters for Conv2d, Conv1d, Linear are supported
LoKr -> ValueError: Target module of type <class 'transformers.pytorch_utils.Conv1D'> not supported, currently only adapters for Conv2d, Conv1d, Linear are supported
```

这印证了 roadmap 里"这是文档化的预期行为不是 bug"的定性——错误信息本身清楚地指出了原因(`Conv1D` 不在支持列表里),`loha_lokr_peft.py` 的 `main()` 用 `try/except ValueError` 主动捕获并打印这个错误,是把这次失败当成一个教学案例来演示,不是代码本身有缺陷。

**面试怎么问 + 追问链:**
- **Q:** "LoHa 为什么能用和 LoRA 差不多的参数量,达到更高的有效秩?"—— 期望说出"两个秩为 r 的矩阵做 Hadamard(逐元素)乘积,秩的上限可以到 r²,而不是简单的 r"。
- **追问 1(深挖数学):** "能不能直观解释一下为什么 Hadamard 积会让秩变成 p×q?"—— 期望能说出"把两个矩阵都展开成秩 1 项的和,Hadamard 积展开后变成两边所有秩 1 项两两配对相乘,数量正好是 p×q 个"这个层面的直觉,不要求完整的严格证明。
- **追问 2:** "LoKr 的 Kronecker 积和 LoHa 的 Hadamard 积,压缩参数的原理一样吗?"—— 期望说出"不一样:Hadamard 积是让'秩上限'变大(更强的表达能力),Kronecker 积是直接用两个小矩阵的乘积拼出一个大矩阵(更省参数,但不是奔着'秩上限'去的)",体现出对两者设计目标差异的清楚认识。
- **追问 3(工程判断力,呼应真实报错):** "遇到 `peft` 库对某个模块类型报 `ValueError not supported` 时,应该怎么排查?"—— 期望说出"先确认目标模型的对应层是不是标准 `nn.Linear`/`nn.Conv1d`/`nn.Conv2d`,如果像 GPT-2 这样用了非标准包装类型(`Conv1D`),要么换一个用标准层的模型架构,要么自己手写实现去处理这个类型转换",不能一遇到报错就怀疑算法本身不可行。

**常见坑:** 把"有效秩可达 r²"理解成"实际秩恒等于 r²"——这是一个**上限**,依赖 `B₁A₁`、`B₂A₂` 本身各自接近满秩(等于 r)且"足够通用"(不退化);如果两者恰好线性相关或存在特殊结构,Hadamard 积的实际秩可能明显低于 r²。另一个常见坑是看到 GPT-2 上 `LoHaConfig`/`LoKrConfig` 报错就直接下结论"LoHa/LoKr 不支持 GPT-2 这类模型"——更准确的说法是"peft 库当前这两个 Config 类的实现不支持 `Conv1D`",这是库的覆盖范围问题,`loha_minimal.py`/`lokr_minimal.py` 这两个手写实现已经证明了数学方法本身对 `Conv1D` 完全适用。

---

## 7. `AdaLoRALinear` + `cubic_schedule`(`adalora_minimal.py`)—— SVD 形式 ΔW + 基于重要性分数的自适应剪枝

**是什么:**
```python
class AdaLoRALinear(nn.Module):
    """h = base(x) + α/r * P diag(Λ) Q^T x"""
    def __init__(self, base_linear, r_init=12, alpha=16, ortho_lambda=0.1):
        super().__init__()
        self.P = nn.Parameter(torch.empty(d_out, r_init))       # 左"奇异向量"
        self.Lambda = nn.Parameter(torch.zeros(r_init))          # 对角"奇异值"，零初始化
        self.Q = nn.Parameter(torch.empty(d_in, r_init))         # 右"奇异向量"
        self.register_buffer("S_ema", torch.zeros(r_init))       # 重要性分数 EMA，不参与梯度
        self.register_buffer("mask", torch.ones(r_init))         # 剪枝 mask

    def ortho_loss(self):
        """||P^T P - I||^2 + ||Q^T Q - I||^2，鼓励（不强制）P、Q 接近正交。"""

    def update_importance(self, beta=0.85):
        """S_i = |λ_i * ∂L/∂λ_i| 的 EMA，须在 backward() 之后、step() 之前调用。"""
        S = (self.Lambda * self.Lambda.grad).abs().detach()
        self.S_ema.mul_(beta).add_((1 - beta) * S)

    def prune_to(self, r_target):
        """保留 S_ema 最大的 r_target 个 λ，其余 mask 置 0（不删除参数本身）。"""


def cubic_schedule(t, t_warmup, T, r_init, r_final):
    """立方衰减：warmup 前满秩，warmup~T 之间按 (1-progress)^3 衰减到 r_final。"""
```

**一句话:** AdaLoRA 把 LoRA 的 `ΔW=BA` 换成显式的 SVD 形式 `ΔW=PΛQᵀ`,训练过程中给每个"奇异值" `λᵢ` 算一个 EMA 平滑的重要性分数,再按一条立方衰减调度,周期性地把分数最低的一批 `λ` 的贡献剪掉(掩码为 0),相当于让模型自己决定每一层最终该分配多少有效秩,而不是所有层都手动固定同一个 r。

**底层机制/为什么这样设计:** `P`(`d_out×r_init`)、`Λ`(`r_init` 维对角)、`Q`(`d_in×r_init`)对应 SVD 的 `U`、`Σ`、`Vᵀ` 三个角色,但要注意这只是**软性模仿**:`ortho_loss()` 只是一个正则项(`‖PᵀP-I‖²+‖QᵀQ-I‖²`),训练时被加进总 loss 去鼓励 P、Q 趋于正交,而不是像 PiSSA 那样通过一次性 SVD 分解**硬性保证**正交——训练过程中 P、Q 完全可能不是精确正交的,这和 PiSSA 的"真 SVD"有本质区别,叫它"SVD 形式"指的是参数化方式模仿了 SVD 的结构(三个因子、对角中间层),不是每一步都满足 SVD 的数学性质。`Λ` 沿用了 LoRA "B 全零"的思路,零初始化保证训练起点 `ΔW=0`。重要性分数 `S_i=|λᵢ·∂L/∂λᵢ|` 是标准的一阶泰勒显著性打分(常见于剪枝文献里的"梯度×权重"打分法):这个值近似衡量"如果把 `λᵢ` 直接置零,loss 大概会变化多少",数值越大说明这个方向当前越重要;因为单步梯度噪声大(尤其 `Λ` 刚从零开始,早期梯度信号本身就不稳定),`update_importance` 用 `β=0.85` 的 EMA 而不是直接用当前步的瞬时值来排序。`prune_to(r_target)` 只是把 `mask` 里排名靠后的位置置 0,**不会删除或清零 `P`/`Λ`/`Q` 底层的参数本身**——本文验证过:被剪掉位置对应的 `Λ` 原始值训练后依然是非零浮点数,只是 `forward` 里算 `masked_lambda = self.Lambda * self.mask` 时这些方向的贡献被强制清零,张量的实际形状、显存占用全程不变,这份 minimal 实现里"剪枝"是纯粹的前向掩码,不是真的压缩了模型体积或省下了矩阵乘法的计算量(要拿到真实的推理加速,还需要额外一步把 mask 为 0 的行/列从张量里物理删除,这份实现没有做这一步)。`cubic_schedule` 决定"什么时候剪、剪到多少":`t≤t_warmup` 期间维持满秩(让模型先自由训练、攒够可靠的重要性分数),`t_warmup~T` 之间按 `(1-progress)³` 从 `r_init` 衰减到 `r_final`(三次方让衰减在中段更陡、两端更平缓),`t≥T` 之后固定在 `r_final`。

**AI 研究场景:** 普通 LoRA 需要手动给(通常是所有层统一)选一个 r,但一个模型不同层对下游任务的重要性天然是不均血的(比如浅层可能更多编码通用语法、深层更多编码任务相关语义),统一分配同一个 r 要么在不重要的层上浪费参数预算,要么在重要的层上限制过紧;AdaLoRA 的做法是先在所有目标层都给一个偏大的 `r_init` 预算,让训练过程自己通过重要性分数发现"哪些层的哪些方向其实没什么用",再把预算动态收回到真正需要的地方——本质是在一个固定的全局参数预算下做**跨层的秩重新分配**,替代人工按层调 r 这件事,在层数很多、又没有时间精力做逐层网格搜索的场景下有明确的工程价值。

**可运行例子:**
```python
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "learning/lora-family/src")
from adalora_minimal import AdaLoRALinear, cubic_schedule

torch.manual_seed(0)
base = nn.Linear(32, 16, bias=False)
layer = AdaLoRALinear(base, r_init=8, alpha=16)
x = torch.randn(4, 32)

with torch.no_grad():
    diff0 = (layer(x) - base(x)).abs().max().item()
print(f"AdaLoRA 出厂初始化(Lambda=0): |out - base_out| = {diff0}")
assert diff0 == 0.0
assert layer.active_rank == layer.r_init == 8

opt = torch.optim.SGD(layer.parameters(), lr=0.3)
target = torch.randn(4, 16)
for step in range(30):
    opt.zero_grad()
    loss = ((layer(x) - target) ** 2).mean() + layer.ortho_loss()
    loss.backward()
    layer.update_importance(beta=0.85)   # EMA 更新重要性分数，须在 backward 之后、step 之前调用
    opt.step()

print("训练 30 步后 S_ema:", [f"{v:.4e}" for v in layer.S_ema.tolist()])

r_target = 3
topk_idx = torch.topk(layer.S_ema, r_target).indices.sort().values
layer.prune_to(r_target)
kept_idx = layer.mask.nonzero().flatten().sort().values
print(f"S_ema 最高的 3 个下标: {topk_idx.tolist()}，prune_to(3) 后保留的下标: {kept_idx.tolist()}")
assert torch.equal(topk_idx, kept_idx)
assert layer.active_rank == 3

with torch.no_grad():
    contrib = (x @ layer.Q) * (layer.Lambda * layer.mask)
pruned_pos = (layer.mask == 0).nonzero().flatten()
print("被剪掉方向对 forward 的贡献 (应为 0):", contrib[:, pruned_pos].abs().max().item())
assert contrib[:, pruned_pos].abs().max().item() == 0.0
print("被剪掉方向的原始 Lambda 依然非零(只是被 mask 掉，没有被删除):",
      layer.Lambda[pruned_pos].detach().abs().tolist())

schedule_points = [0, 100, 200, 500, 800, 1000]
values = [cubic_schedule(t, 100, 1000, 12, 4) for t in schedule_points]
print("\ncubic_schedule(t_warmup=100, T=1000, r_init=12, r_final=4):", dict(zip(schedule_points, values)))
assert values == [12, 12, 10, 5, 4, 4]
```

实测:出厂 `|out - base_out| = 0.0`,`active_rank` 初始为 8;训练 30 步后 `S_ema = ['5.3004e-04', '1.0946e-02', '8.6028e-03', '3.4998e-02', '1.3267e-03', '1.3669e-02', '1.1424e-02', '2.6861e-03']`;`S_ema` 最高的 3 个下标是 `[3, 5, 6]`,`prune_to(3)` 之后 `mask` 里保留的下标同样是 `[3, 5, 6]`,两者精确一致;被剪掉方向对 `forward` 的贡献精确是 `0.0`,但这些位置的原始 `Λ` 值(比如 `[0.0647, 0.2471, 0.2286, 0.0932, 0.1425]`)依然是非零浮点数,印证"只掩码、不删除"。`cubic_schedule` 真实调度:`{0: 12, 100: 12, 200: 10, 500: 5, 800: 4, 1000: 4}`——`t_warmup=100` 之前维持满秩 12,`T=1000` 之后固定在 `r_final=4`,中间平滑过渡。

**面试怎么问 + 追问链:**
- **Q:** "AdaLoRA 怎么决定剪掉哪些方向?"—— 期望说出"给每个奇异值算一个基于梯度的重要性分数(`|λᵢ·∂L/∂λᵢ|`),做 EMA 平滑之后,周期性保留分数最高的一批,其余掩码为 0"。
- **追问 1:** "为什么重要性分数要做 EMA,不直接用当前这一步的梯度?"—— 期望说出"单步梯度噪声大,尤其 Λ 从零起步早期信号不稳定,EMA 用历史信息平滑掉噪声,让排序更可靠"。
- **追问 2(压力测试,呼应本文验证到的细节):** "`prune_to()` 剪掉的方向,是真的从模型里删除了吗,能带来推理加速吗?"—— 期望明确说"不是删除,是把 mask 置零,`P`/`Λ`/`Q` 的张量形状和显存占用完全不变,这份实现里的剪枝不产生真实的推理加速,要拿到真实加速需要额外做一步把 mask 为 0 的行列从张量里物理裁掉"。
- **追问 3:** "`ortho_loss()` 里的正交正则项是硬约束吗,不加会怎样?"—— 期望说出"是软的正则惩罚项,不是强制约束,训练过程中 P、Q 不一定精确正交;如果去掉这一项,`P`、`Q` 可能会退化(比如不同列之间线性相关),让'不同奇异值方向互相独立'这个 SVD 形式想表达的语义名不副实"。

**常见坑:** 以为 `prune_to()` 之后模型的参数量或显存占用会变小——不会,这份 minimal 实现的剪枝是纯前向掩码,`P`/`Λ`/`Q` 全程保持 `r_init` 大小的张量形状,`active_rank` 只是"当前 mask 里有效方向的计数",不是张量的真实维度;另一个常见坑是把 AdaLoRA 的 `P`/`Q` 当成和 PiSSA 里 SVD 分解出来的 `U`/`Vᵀ` 一样"精确正交"——AdaLoRA 的正交性只是训练目标里的一个软性正则项,不是像 PiSSA 那样通过一次性数学分解硬性保证的性质,这是两种方法里"SVD"含义不同的地方,容易被同一个关键词误导成同一件事。
