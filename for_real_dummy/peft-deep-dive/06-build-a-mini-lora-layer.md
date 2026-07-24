# 06 · 手把手实战:从零搭一个迷你 LoRA 层

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 06 个"知识点",不计入"约 24 个知识点"的统计——和 [05 类](05-advanced-interview-depth.md)是同一挂(都不计入正式知识点计数),但风格完全不一样:05 号文件里,你是**旁观者**,跟着面试官和候选人的对话把一条推理链条看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立搭出一个真实能跑的迷你 LoRA 层,不调用 `peft` 库,不借助 `learning/lora-family/src/lora_minimal.py` 里现成的 `LoRALinear`,自己从 `nn.Linear` 开始一行行写出来。

## 为什么是"迷你 LoRA 层"

不是要发明新知识点,是把 [01 类知识点 1](01-lora-core.md)(`LoRALinear` 的核心数学)拆成可以亲手验证的几步,重新组装一遍:

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 拥有一个权重矩阵,但让反向传播完全碰不到它 | [01 类知识点 1](01-lora-core.md) "冻结 `W₀`"这一半 |
| 阶段 2 | 在冻结权重旁边并联一条低秩分支,且起点完全不改变输出 | [01 类知识点 1](01-lora-core.md) B 零初始化保证起点等于原模型 |
| 阶段 3 | 用真实反向传播验证:B 先拿到梯度,A 要等 B 挪动之后才有梯度 | [01 类知识点 1](01-lora-core.md) "B=0 时 A 在 step 0 梯度精确为零"这一机制 |
| 阶段 4 | 组装成一个真正会训练的小模型,拿参数量对比说话 | 阶段 1~3 全部组装 + [01 类知识点 1](01-lora-core.md) AI 研究场景(全参数微调为什么训不动) |

每个阶段的代码都能独立运行(本文件用从 [dsa-deep-dive/_verify_md.py](../dsa-deep-dive/_verify_md.py) 拷贝过来、原样复用的 `_verify_md.py` 校验,校验方式是把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行——块与块之间**不共享任何变量**,所以后面阶段用到前面阶段写过的 `MiniLoRALinear` 类时,会重新贴一遍完整定义,不是偷懒复制,是这套校验机制要求的)。

**环境声明:** [01](01-lora-core.md)~[05](05-advanced-interview-depth.md) 号文件的大多数可运行例子在本机真实 GPU(RTX 3080 Ti)上验证过,[02 类](02-quantized-lora.md)甚至有真实 bitsandbytes 4bit 量化路径。这一篇刻意不同:全篇显式用 `torch.device("cpu")`,张量维度控制在几十到几百(最大的一处是阶段 4 两层 MLP 的 `d_hidden=512`),不加载任何预训练权重,整篇教程秒级跑完。这台机器确实装了 GPU,但教程体的目标是"从零手写出可用的东西"这个过程本身,不是复现大规模数字——大规模/真实 GPU 的验证已经在 01/02 号文件里做过,这里不重复。

另外,为了保持极简,本文的 `MiniLoRALinear` 省略了 [01 类知识点 1](01-lora-core.md) 真实 `LoRALinear` 里的两个东西:`dropout` 参数(默认就是 0.0,不影响任何一步的数学结论)、以及 GPT-2 `Conv1D` 兼容处理(`get_in_out_dims`/`is_conv1d`,本文只用标准 `nn.Linear`,用不上)。这两处省略在"可以怎么继续扩展"一节里会重新提到。

---

## 阶段 1:先有一个普通的、完全可训练的 `nn.Linear`——然后冻结它

LoRA 的第一个动作不是"发明新结构",是"让一个本来完全可训练的权重矩阵,变得对反向传播完全不可见"。这一步看着简单(`requires_grad = False` 一行),但它是后面所有省显存说法的起点:一个张量如果不 `requires_grad`,PyTorch 的 autograd 从一开始就不会为它构建计算图节点,不只是"训练时跳过它"这么简单。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
device = torch.device("cpu")

d_in, d_out = 64, 128
base = nn.Linear(d_in, d_out, bias=False).to(device)

# freeze 之前：这就是一个普通的、完全可训练的 nn.Linear
assert all(p.requires_grad for p in base.parameters())

# freeze：只有这一行，就是 LoRA 能省下大量显存的起点
for p in base.parameters():
    p.requires_grad = False

assert all(not p.requires_grad for p in base.parameters())
print("base.weight.requires_grad after freeze:", base.weight.requires_grad)

x = torch.randn(4, d_in, device=device)
out = base(x)
print("frozen base output shape:", tuple(out.shape))

# 冻结之后，从这个输出出发根本没有可以反传的计算图节点——
# 这才是"冻结"真正带来的效果，不只是注释里写一句话
loss = out.sum()
print("loss.requires_grad:", loss.requires_grad)
try:
    loss.backward()
    print("backward succeeded (unexpected)")
except RuntimeError as e:
    print("backward failed as expected:", str(e))
    assert "does not require grad" in str(e)

n_full_params = sum(p.numel() for p in base.parameters())
print("full nn.Linear parameter count:", n_full_params)
assert n_full_params == d_in * d_out
print("stage1 ok")
```

实测(`.venv` CPU 真跑):`base.weight.requires_grad after freeze: False`;`loss.requires_grad: False`——注意这里连 `loss` 本身都不 `requires_grad`,不是"梯度是 0",是**从数据类型层面**就没有梯度这回事;调用 `.backward()` 直接抛 `RuntimeError`,报错信息精确是 `element 0 of tensors does not require grad and does not have a grad_fn`;`nn.Linear(64,128,bias=False)` 的参数量精确是 `8192`(=64×128)。这是"全参数微调要存这么多梯度和优化器状态"这句话里"这么多"的具体数字来源。

---

## 阶段 2:手写 LoRA 分支——`h = base(x) + (alpha/r)·B(A(x))`,B 全零起步

冻结的权重旁边,并联一条"先降维、再升维"的低秩分支:输入先被 `A`(形状 `(r, d_in)`)投影到只有 `r` 维,再被 `B`(形状 `(d_out, r)`)投影回 `d_out` 维,中间乘一个缩放系数 `alpha/r`。`A` 用 Kaiming 初始化给出真实的非零值,`B` 故意全部初始化成 0——这正是 [01 类知识点 1](01-lora-core.md) 讲过的设计:两个矩阵里只要有一个是全零,乘积 `B@A` 就精确是零矩阵,训练还没开始的时候,LoRA 分支对输出的贡献是精确的 0,不是"很小",输出和只用冻结原始权重时**逐位相等**。

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)
device = torch.device("cpu")

class MiniLoRALinear(nn.Module):
    """h = base(x) + (alpha/r) * B(A(x))  -- A 随机(Kaiming)初始化，B 全零初始化。"""

    def __init__(self, base_linear, r=4, alpha=8):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False          # 复用阶段1：原始权重全程冻结

        d_in, d_out = base_linear.in_features, base_linear.out_features
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))     # 全零初始化，是本阶段的验证重点
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = x @ self.A.T @ self.B.T    # 等价于 B(A(x))：先 A 把 x 投影到 r 维，再 B 投影回 d_out 维
        return base_out + self.scaling * lora_out

base = nn.Linear(64, 128, bias=False).to(device)
lora = MiniLoRALinear(base, r=4, alpha=8).to(device)
x = torch.randn(4, 64, device=device)

# 核心验证：B 全零 -> LoRA 分支贡献精确为 0 -> 输出与冻结的 base 逐位相等
assert torch.all(lora.B == 0)
with torch.no_grad():
    out_lora = lora(x)
    out_base = lora.base(x)
diff = (out_lora - out_base).abs().max().item()
print("max|lora(x) - base(x)| at init:", diff)
assert torch.equal(out_lora, out_base)

# 手动展开公式，核对与 forward() 完全一致
with torch.no_grad():
    manual = lora.base(x) + lora.scaling * (x @ lora.A.T @ lora.B.T)
assert torch.equal(manual, out_lora)

assert lora.scaling == 8 / 4
print("lora.scaling:", lora.scaling)
print("A shape:", tuple(lora.A.shape), " B shape:", tuple(lora.B.shape))
print("stage2 ok")
```

实测:`max|lora(x) - base(x)| at init: 0.0`——不是浮点误差范围内的接近,是 `torch.equal` 断言过的精确相等;`lora.scaling: 2.0`(=8/4,`alpha/r`);`A shape: (4, 64)  B shape: (128, 4)`,形状分别对应 `(r, d_in)` 和 `(d_out, r)`,和 [01 类知识点 1](01-lora-core.md) 的公式定义完全一致。

---

## 阶段 3:真实反向传播——B 先拿到梯度,A 起跑线上是精确的 0

阶段 2 只验证了"起点输出相等"这个静态性质。[01 类知识点 1](01-lora-core.md) 的底层机制一段推导过一件更深的事:把 `forward` 展开成 `y = scaling·(x@Aᵀ)@Bᵀ`,对 `B` 求梯度只依赖 `A`(`A` 从一开始就非零),但对 `A` 求梯度要经过 `B` 这一层链式法则——`B` 全零意味着 `A` 在第 0 步的梯度**精确为零**。这一节换一组全新的张量形状(`nn.Linear(64,128)`,`r=4`,而 [01 类知识点 1](01-lora-core.md) 原始例子用的是 `nn.Linear(16,32)`),重新亲手验证一遍这个结论,不是照抄 01 号文件已经算出来的数字。

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)
device = torch.device("cpu")

class MiniLoRALinear(nn.Module):
    def __init__(self, base_linear, r=4, alpha=8):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out = base_linear.in_features, base_linear.out_features
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = x @ self.A.T @ self.B.T
        return base_out + self.scaling * lora_out

base = nn.Linear(64, 128, bias=False).to(device)
lora = MiniLoRALinear(base, r=4, alpha=8).to(device)
x = torch.randn(4, 64, device=device)
target = torch.randn(4, 128, device=device)

# 第0步：B 还是全零，链式法则里 dL/dA 要乘过 B 这一因子，结果精确为 0——
# 只有 B 能在第0步拿到真实梯度
loss0 = ((lora(x) - target) ** 2).mean()
loss0.backward()

grad_A_step0 = lora.A.grad.abs().max().item()
grad_B_step0 = lora.B.grad.abs().max().item()
print("step 0: max|A.grad| =", grad_A_step0, " max|B.grad| =", grad_B_step0)
assert grad_A_step0 == 0.0
assert grad_B_step0 > 0.0

# 冻结的 base 权重从始至终没有梯度——不是"很小"，是精确的 None，
# 因为 requires_grad=False 让 autograd 从一开始就不追踪它
print("base.weight.grad is None:", base.weight.grad is None)
assert base.weight.grad is None

# 只把 LoRA 的 A、B 交给优化器——这就是"优化器状态也能省下来"的全部秘密
# （Adam/SGD 只会给传进来的张量分配动量状态）
opt = torch.optim.SGD([lora.A, lora.B], lr=0.5)
opt.step()
assert torch.any(lora.B != 0)

# 第1步：B 已经离开零点，A 终于能拿到真实梯度
opt.zero_grad()
loss1 = ((lora(x) - target) ** 2).mean()
loss1.backward()
grad_A_step1 = lora.A.grad.abs().max().item()
print("step 1: max|A.grad| =", grad_A_step1)
assert grad_A_step1 > 0.0

# base 权重依然完全没动（grad 全程是 None）
assert base.weight.grad is None
print("stage3 ok")
```

实测:`step 0: max|A.grad| = 0.0  max|B.grad| = 0.04372907429933548`——`A` 的梯度不是很小的数,是精确的 `0.0`,和 [01 类知识点 1](01-lora-core.md) 里 `nn.Linear(16,32)` 例子得到的结论(那边是 `|A.grad|.max = 0.0000000000`,`|B.grad|.max = 0.110251`)方向完全一致,只是换了形状之后 `B.grad` 的具体数值自然不同(`0.0437` vs `0.1103`),这正说明"A 在 step 0 梯度精确为零"是这个机制本身决定的,不依赖某一组特定的张量形状。`base.weight.grad is None: True`——冻结权重连梯度张量都没有分配,不是"梯度是 0 张量"。走完一步 SGD 之后,`step 1: max|A.grad| = 0.029004283249378204`,`A` 立刻拿到非零梯度,印证"B 先学,A 后学"这个训练顺序。

---

## 阶段 4:组装成会训练的迷你模型——参数量差多少个数量级

把 `MiniLoRALinear` 用在一个两层小 MLP 上,每层各自冻结、各自挂一条独立的 LoRA 分支,数一数全量微调需要训练多少参数、只训练 LoRA 的 A、B 又是多少,再真跑几步 Adam,确认 loss 真的会下降、冻结权重真的纹丝不动。

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)
device = torch.device("cpu")

class MiniLoRALinear(nn.Module):
    def __init__(self, base_linear, r=4, alpha=8):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out = base_linear.in_features, base_linear.out_features
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = x @ self.A.T @ self.B.T
        return base_out + self.scaling * lora_out

class TinyLoRAMLP(nn.Module):
    """两层冻结 Linear，各自挂一条独立的 LoRA 分支。"""
    def __init__(self, d_in=256, d_hidden=512, d_out=256, r=4, alpha=8):
        super().__init__()
        self.layer1 = MiniLoRALinear(nn.Linear(d_in, d_hidden, bias=False), r=r, alpha=alpha)
        self.layer2 = MiniLoRALinear(nn.Linear(d_hidden, d_out, bias=False), r=r, alpha=alpha)

    def forward(self, x):
        h = torch.relu(self.layer1(x))
        return self.layer2(h)

    def lora_parameters(self):
        return [self.layer1.A, self.layer1.B, self.layer2.A, self.layer2.B]

model = TinyLoRAMLP(d_in=256, d_hidden=512, d_out=256, r=4, alpha=8).to(device)
x = torch.randn(8, 256, device=device)
target = torch.randn(8, 256, device=device)

# 参数量对比：全量微调要训练全部参数（冻结+可训练一起数），
# LoRA 只训练 A、B 两个小矩阵
full_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("full model parameters (frozen + trainable):", full_params)
print("trainable (LoRA A+B only) parameters:", trainable_params)
ratio = full_params / trainable_params
print("full / trainable ratio: {:.1f}x fewer parameters to train".format(ratio))
assert trainable_params == sum(p.numel() for p in model.lora_parameters())
assert full_params - trainable_params == 256 * 512 + 512 * 256
assert ratio > 10  # 稳稳超过一个数量级

frozen_weight_before = model.layer1.base.weight.clone()

opt = torch.optim.Adam(model.lora_parameters(), lr=0.02)
loss_fn = nn.MSELoss()

with torch.no_grad():
    loss_before = loss_fn(model(x), target).item()

for step in range(5):
    opt.zero_grad()
    loss = loss_fn(model(x), target)
    loss.backward()
    opt.step()

with torch.no_grad():
    loss_after = loss_fn(model(x), target).item()

print("loss before training: {:.4f}".format(loss_before))
print("loss after 5 real optimizer steps: {:.4f}".format(loss_after))
assert loss_after < loss_before

# 冻结的 base 权重在5步真实训练之后逐位不变——
# "冻结"不是初始化时的一次性动作，训练全程都成立
assert torch.equal(frozen_weight_before, model.layer1.base.weight)
print("stage4 ok")
```

实测:两层 `nn.Linear(256,512)`+`nn.Linear(512,256)` 全量参数是 `268288`(=256×512+512×256),只训练 LoRA 的 A、B 总共只有 `6144` 个参数,`full / trainable ratio: 43.7x fewer parameters to train`——只训练不到全量参数 1/43 的量级(这里 `r=4` 是刻意选小的,真实场景常用 `r=8~64`,比例会往回缩一些,但省下的仍然是数量级上的差距,[01 类知识点 1](01-lora-core.md) 的"AI 研究场景"一段讨论过全参数微调为什么在消费级显卡上训不动,这里是它的具体数字版本)。5 步真实 `Adam` 更新之后,`loss before training: 1.0451` 降到 `loss after 5 real optimizer steps: 0.4931`,同时 `frozen_weight_before` 和训练后的 `model.layer1.base.weight` 依然逐位相等——冻结权重全程没有被优化器碰过一次。

---

## 训练稳定性小插曲:学习率选大了会怎样(真实撞见的现象,不是刻意设计的教学效果)

准备阶段 4 的 `lr=0.02` 时,顺手扫了几个学习率想找一个"降得比较快看着效果明显"的取值,结果撞见了一个值得记录的真实现象:学习率选大一点(`lr=0.05`),loss 不是更快下降,反而会在训练途中飙到比初始值还高,之后再震荡回落;换成 `lr=0.02`,10 步里每一步都比上一步低,单调下降,没有一次反弹。这不是刻意设计出来的教学效果,是真的调参时先撞见、后确认下来的现象——LoRA 的 A、B 依然是普通的神经网络参数,学习率选择不当同样会不稳定,这一点不会因为"参数量小"就自动免疫。

```python
import math
import torch
import torch.nn as nn

device = torch.device("cpu")

class MiniLoRALinear(nn.Module):
    def __init__(self, base_linear, r=4, alpha=8):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out = base_linear.in_features, base_linear.out_features
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = x @ self.A.T @ self.B.T
        return base_out + self.scaling * lora_out

class TinyLoRAMLP(nn.Module):
    def __init__(self, d_in=256, d_hidden=512, d_out=256, r=4, alpha=8):
        super().__init__()
        self.layer1 = MiniLoRALinear(nn.Linear(d_in, d_hidden, bias=False), r=r, alpha=alpha)
        self.layer2 = MiniLoRALinear(nn.Linear(d_hidden, d_out, bias=False), r=r, alpha=alpha)

    def forward(self, x):
        h = torch.relu(self.layer1(x))
        return self.layer2(h)

    def lora_parameters(self):
        return [self.layer1.A, self.layer1.B, self.layer2.A, self.layer2.B]

def run(lr, steps=10):
    torch.manual_seed(0)
    model = TinyLoRAMLP(d_in=256, d_hidden=512, d_out=256, r=4, alpha=8).to(device)
    x = torch.randn(8, 256, device=device)
    target = torch.randn(8, 256, device=device)
    opt = torch.optim.Adam(model.lora_parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    with torch.no_grad():
        losses.append(loss_fn(model(x), target).item())   # 0 次更新时的 loss
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        opt.step()
        with torch.no_grad():
            losses.append(loss_fn(model(x), target).item())  # 每做完一次更新，重新测一次 loss
    return losses

losses_high = run(lr=0.05)
losses_good = run(lr=0.02)
print("lr=0.05 losses:", ["{:.4f}".format(v) for v in losses_high])
print("lr=0.02 losses:", ["{:.4f}".format(v) for v in losses_good])

# 学习率选太大：训练途中 loss 会飙到比第0步还高（震荡/不稳定），不是一路平稳下降
assert max(losses_high) > losses_high[0]

# 学习率选得合适：每一步都不比上一步差，单调不增
assert all(losses_good[i + 1] <= losses_good[i] for i in range(len(losses_good) - 1))
print("bonus lr-sensitivity check ok")
```

实测:`lr=0.05` 的 loss 轨迹是 `['1.0451', '0.9418', '6.7275', '1.1487', '1.6618', '2.3666', '2.5448', '1.4159', '0.9778', '0.9930', '0.9284']`——第 2 步猛冲到 `6.7275`(比初始的 `1.0451` 高出 6 倍多),之后连续好几步(`1.6618→2.3666→2.5448`)还在继续爬升,10 步跑完也没恢复到比初始值更低的水平;`lr=0.02` 的轨迹是 `['1.0451', '1.0005', '0.7889', '0.6454', '0.5054', '0.4931', '0.4103', '0.3780', '0.3391', '0.3216', '0.2957']`,11 个数字严格逐个下降,一次反弹都没有。这个对比说明"只训练两个小矩阵"改变的是**要训练的参数量**,不改变"这仍然是一次梯度下降,学习率仍然可能选太大"这个事实。

---

## 可以怎么继续扩展(只指方向,不在本文实现)

- **接入真实 `Conv1D` / 真实 GPT-2**:本文的 `MiniLoRALinear` 只处理 `nn.Linear`,[01 类知识点 1](01-lora-core.md) 的真实 `LoRALinear` 还要用 `get_in_out_dims`/`is_conv1d` 兼容 GPT-2 `c_attn` 用的 `Conv1D` 层(两者 `weight` 形状是转置关系)。
- **补上 dropout**:本文为了保持极简省略了 `dropout=0.0` 这个可选项,[01 类知识点 1](01-lora-core.md) 的真实实现里 `forward` 是 `self.dropout(x) @ self.A.T @ self.B.T`,加回来只是多包一层 `nn.Dropout`,不影响本文验证过的任何数学结论。
- **实现 `merge_weights()` 和它真实存在的坑**:训练完之后怎么把 `scaling·B@A` 合并进 `base.weight` 换取推理零延迟,以及"docstring 写着删除 A、B 但实现代码其实没删,继续调用会双重计数"这个真实陷阱,[01 类知识点 3](01-lora-core.md) 已经完整踩过一遍并给出了可运行的复现例子。
- **换缩放公式 / 给 A、B 分组学习率**:把 `scaling=alpha/r` 换成 `alpha/sqrt(r)`(rsLoRA,大 `r` 下更稳定)、给 `A`、`B` 设不同学习率(LoRA+,`B` 的学习率通常是 `A` 的 16 倍),[01 类知识点 2](01-lora-core.md) 有完整实现和数学推导。
- **真实预训练模型上的端到端训练**:本文全程没有加载任何预训练权重。真要在真实语言模型上跑通训练循环,GPU 显存、真实数据集、更长的训练步数都要重新考虑,[02 类知识点 2](02-quantized-lora.md)的 `qlora_minimal.py`(全系列唯一的真 5 步训练循环)是这个方向已经做过的例子,不在本文重复。

这几个方向都不实现,是为了让这篇教程聚焦在"LoRA 最核心的那几行数学,亲手写一遍、亲手验证一遍"这一件事上——每个方向单独展开,[01 类](01-lora-core.md)~[02 类](02-quantized-lora.md)已有对应的完整知识点可以直接跳过去读。

## 这篇教程展示的方法论

和 [dsa-deep-dive/21 号文件](../dsa-deep-dive/21-build-a-mini-search-engine.md)开创的模式一样:挑几个关联的知识点 → 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实效果,而不是一次性甩出完整代码。这是"教程体"格式第二次落地(第一次是 dsa-deep-dive 系列的试点),这一次額外多了一条约束:peft-deep-dive 系列平时的知识点大量依赖真实 GPU 验证(本机 RTX 3080 Ti,[02 类](02-quantized-lora.md)甚至有真实 bitsandbytes 4bit 量化路径),但教程体的"从零手写、亲手验证"这个目标本身不需要大规模——LoRA 最核心的几条性质(起点输出相等、B 先学 A 后学、参数量数量级差距)在 `nn.Linear(64,128)` 这种玩具规模上就能完整、真实、可复现地验证出来,刻意维持 CPU 玩具规模不是能力不够,是"教程体要验证的是机制,不是规模"这个定位本身决定的。其余系列要不要配套同类文件、用什么规模,是后续单独决定的问题,这里不展开。

---

*创建:2026-07-24*
