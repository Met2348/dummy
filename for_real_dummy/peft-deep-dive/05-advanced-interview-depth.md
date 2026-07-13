# 05 · 进阶深度追加:5 个真实二面级别的多级追问链案例

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是知识点,不计入统计——它和 [dsa-deep-dive/20-advanced-interview-depth.md](../dsa-deep-dive/20-advanced-interview-depth.md)、[alignment-algorithms-deep-dive/04-advanced-interview-depth.md](../alignment-algorithms-deep-dive/04-advanced-interview-depth.md) 是同一挂:方法论 + 案例,不是知识点列表。

## 为什么需要这篇追加内容

`01-04` 全部完成并自查通过之后,用户转达了一位有经验从业者的反馈:现有材料没有达到 **2026 年大厂技术二面** 的深度。`dsa-deep-dive/20-advanced-interview-depth.md` 和 `alignment-algorithms-deep-dive/04-advanced-interview-depth.md` 已经基于一次真实调研(检索中国大厂面经、西方大厂面经、面试官视角的元讨论)落地验证过一套格式,核心发现是:真实的追问不是"正确性 → 复杂度 → 能不能优化"这一条线性链,而是至少沿着 **5 条独立轴线** 展开,并且经常在同一道题里综合出现:

1. **规模递增轴**——单卡单模型验证过的结论,推到更大的模型/更多张卡上,第一个会先撞上什么。
2. **工程约束递增轴(并发/分布式)**——单次调用正确,不等于多个并发调用同时发生时还正确。
3. **方案批判迭代轴**——面试官连续指出具体的工程缺陷,逼你换方案,不是"不够好"这种空话。
4. **决策依据追问轴**——不纠错,只逼问"给定这个约束,你是怎么选这个、排除那些的"。
5. **真实性验证轴**——把"这个函数做了 X""这个结论已经验证过"这类表述,追问压向具体数字、具体验证过程,而且要能验证"权威表述本身"是不是全称成立,而不是"某一个例子恰好成立"。

`peft-deep-dive` 系列(`01-04`)覆盖 LoRA 家族(LoRA 本身、量化组合)和 Adapter 家族(核心构件、进阶与统一视角)约 24 个知识点,其中不少地方已经在正文里做过一次"读代码 + 独立验证"(比如 [01 类知识点 3](01-lora-core.md) 发现 `merge_weights()` 的 docstring 与实现不符、[02 类知识点 5](02-quantized-lora.md) 发现 DoRA 和量化其实没有关系、[03 类知识点 6](03-adapter-core.md) 发现 (IA)³ 手写/peft 两条路径参数总量相等是巧合)——这些发现本身就是很好的"真实性验证轴"种子,本篇把它们各自往深处再推一层,而不是简单复述。

本篇选了 **5 个案例**,组织原则是每个案例明确挂一条主轴线(部分案例会自然带出第二条轴线,在文末表格里如实标注,不强行只挂一条):

- **案例 1**(方案批判迭代轴):`LoRA → QLoRA → LoftQ → DoRA` 的显存/精度权衡链,建立在 [01 类知识点 1](01-lora-core.md) 和 [02 类知识点 1/2/4/5](02-quantized-lora.md) 之上。
- **案例 2**(决策依据追问轴):`Houlsby → Pfeiffer → Compacter → (IA)³` 参数压缩谱系,给定约束怎么选,建立在 [03 类知识点 2/4/6](03-adapter-core.md) 之上。
- **案例 3**(工程约束递增轴/并发):同一个 base 模型服务多个不同 LoRA 适配器时,`merge_weights()` 这条"零延迟"路径会出什么问题,建立在 [01 类知识点 3](01-lora-core.md) 的 AI 研究场景讨论之上。
- **案例 4**(真实性验证轴):`merge_weights()` 的"删除 A、B"承诺——这个仓库的 minimal 实现和真实 peft 库的 `merge_adapter()`/`merge_and_unload()` 分别是怎么(不)兑现这个承诺的,直接承接案例 3,把 [01 类知识点 3](01-lora-core.md) 已发现的问题对照生产库重新验证一遍。
- **案例 5**(真实性验证轴 + 规模递增轴):`"真 4bit 训练 + 真 bitsandbytes 从未同时出现"`([02 类知识点 6](02-quantized-lora.md))这条边界往 7B/多卡场景外推,第一步会先撞上什么、这台机器验证得到什么验证不到什么。

**范围声明:** 这是方法论范例,不是把 24 个知识点全部重写一遍。每个案例都要求读者能看到"同样的追问方式,怎么套到任何一个已经掌握的知识点上"——读完之后,应该能自己对着 01-04 里任何一个没在这里出现的知识点(比如 VeRA、AdaLoRA、AdapterFusion、MAM),现场把这 5 条轴线走一遍练习,而不是指望这篇文档穷举所有可能的追问。

---

## 案例 1:LoRA→QLoRA→LoftQ→DoRA——存不下→量化误差→"每一步都不会变差"当真吗→正交轴换维度(方案批判迭代轴)

建立在 [01 类知识点 1](01-lora-core.md)(`LoRALinear`,B 零初始化)和 [02 类知识点 1/2/4/5](02-quantized-lora.md)(NF4 量化、`qlora_minimal.py` 的真实训练循环、LoftQ 交替最小化、DoRA 的 magnitude/direction 分解且与量化无关)之上。这条链和 [alignment-algorithms-deep-dive 案例 2](../alignment-algorithms-deep-dive/04-advanced-interview-depth.md)(`DPO→IPO→KTO→ORPO/SimPO/CPO→DPOP`)结构相似:面试官不是在一个方案上反复深挖,而是针对每个方案指出一个具体的、可验证的缺陷逼你换方案;而且和那条链一样,最后一步会出现一个"这不是第 4 个候选方案,是一个正交轴"的转折——只是这一次,转折之前还多了一层：本案例在验证 LoftQ"代价"的过程中,现场发现了一个这个仓库 02 号文件撰写时没有发现的真实反例。

**追问链条完整还原:**

- **面试官给约束:** "一个几十亿参数的模型,消费级显卡做全参数微调存不下也训不动,你怎么办?"
- **候选人方案 1(LoRA):** "冻结预训练权重 W₀,只训练两个低秩矩阵 A、B。"([01 类知识点 1](01-lora-core.md) 已验证:B 零初始化保证起点等于原模型,可训练参数能压到不到 1%。)
- **面试官指出具体缺陷:** "可训练参数压到 1% 以内了,但你说的'存不下'问题真的解决了吗?被冻结的 W₀ 本身占多少显存?"——期望候选人意识到"冻结"和"不占显存"是两件事,W₀ 全程仍是稠密 fp16/fp32 存储,一个 7B 模型光是 fp16 权重就要 14GB,这是 LoRA 完全没有触碰的维度。
- **候选人方案 2(QLoRA):** "把 W₀ 本身也压缩——用 NF4 量化成 4-bit。"([02 类知识点 1](02-quantized-lora.md))
- **面试官指出具体缺陷:** "量化会引入真实的数值误差——你们自己验证过,如果一个 block 的数值尺度被另一个 block 拖累,相对误差能到接近 100%。这部分误差你打算怎么处理,晾在那,指望 A、B 训练的时候自己把它'纠正'回来?"
- **候选人方案 3(LoftQ):** "用交替最小化,让 A、B 的初始值直接去逼近'量化前后的残差',不完全依赖训练去纠正。"([02 类知识点 4](02-quantized-lora.md))
- **面试官指出新方案的代价(核心深挖):** "LoftQ 论文和这个仓库 02 号文件都验证过一句话——'交替最小化的每一步都不会让目标函数变差'(`assert non_increasing` 通过)。这句话对任意输入都成立吗,还是恰好这一个例子成立?"
- **候选人现场验证(不能只是相信一句"已经验证过"的结论):** 完全复用 02 号文件的配置(方阵 64×64,`n_iter=5`)换 40 个不同随机种子重新跑一遍,发现有 34 个种子(85%)在同样的迭代次数内就已经出现局部残差不降反升;02 号文件恰好用的种子 0 之所以"看起来单调",只是因为在第 5 步就停了——放宽到 `n_iter=15`,同一个种子从第 6 步开始同样出现残差回升。
- **深挖追问(逼问机制,不满足于"数据显示确实会变差"这个表面结论):** "为什么会变差?SVD 那一步不是有 Eckart-Young 定理托底,理论上不会变差吗?"——期望候选人现场推理/验证到:SVD 那一步(固定 Q,对 `W-Q` 求最优秩-r 近似)确实不会变差;但 NF4 重新量化那一步的 `absmax=max(abs(·))` 是"保证不溢出"的启发式,不是 L2 误差最优的缩放系数选择——当 BA 更新后,某个 block 里的新目标值可能出现一个新的离群值,把这个 block 的 absmax 推高,导致该 block 内其余"正常"量级的元素被量化得更粗糙;这意味着"沿用上一轮过时的 Q"未必比"按当前 target 重新算出的、在自己坐标系里已经最优的新 Q"更差——两个子问题各自局部最优,不保证拼起来的整体轨迹严格单调。
- **深挖追问(pivot 陷阱,考察候选人是否会把 DoRA 错误地当成 LoftQ 的延伸):** "所以 LoftQ 这条链下一步该怎么走,是继续加大 `n_iter`,或者换一个更精细的 scale 估计方式吗?"——期望候选人指出:这两条路径原则上还能继续打磨,但都是在"量化残差"这同一个维度上的局部优化,收益有限(而且刚验证过,`n_iter` 变大反而可能让残差更不稳定,不是单调收益递减这么简单);如果想要更大的改进,应该换一个完全不同的维度——不是继续和量化误差较劲,而是重新思考 `ΔW` 本身怎么参数化。
- **引出 DoRA,验证正交性:** 用 `grep` 现场确认 `dora_minimal.py` 里零命中 `nf4`/`quant` 字样([02 类知识点 5](02-quantized-lora.md) 已指出这一点),`DoRALinear.base.weight` 全程是原始全精度权重,从未被量化替换过——DoRA 不是"更强的 LoftQ",是完全独立的一条轴(把 `ΔW` 拆成 magnitude × direction,而不是继续处理量化残差)。
- **深挖追问(工程判断力,检验"正交"是否只是嘴上说说):** "既然是正交的两个维度,那能不能同时用 QLoRA 的显存节省和 DoRA 的 magnitude/direction 分解?这个仓库有现成的例子吗?"——期望候选人诚实回答"没有,这个仓库没有实现 QDoRA",但能现场推理组合方式:需要在**量化后的残差** `Q=NF4(W₀)` 上做 magnitude/direction 分解(`V=Q+scaling·BA`,`m` 从 `Q` 的列范数初始化),而不是在原始全精度 `W₀` 上做——如果能现场写出一个最小原型验证这个组合确实可行,是加分项。

**可运行例子(1/2):"每一步都不会变差"不是全称成立——大规模种子扫描 + 机制溯源,不是转述一句"论文说的"**

```python
import sys
sys.path.insert(0, "learning/lora-family/src")
import torch
import torch.nn as nn
from loftq_minimal import LoftQLinear
from nf4_quant import nf4_quantize, nf4_dequantize, nf4_quant_dequant

# Part 1: 完全复用 02 号文件的配置（方阵 64x64, n_iter=5），换 40 个新种子重新跑一遍
n_violate = 0
n_seeds = 40
for seed in range(n_seeds):
    torch.manual_seed(seed)
    base = nn.Linear(64, 64, bias=False)
    layer = LoftQLinear(base, r=8, n_iter=5, block_size=64)
    hist = layer.get_convergence()
    if any(hist[i + 1] > hist[i] + 1e-6 for i in range(len(hist) - 1)):
        n_violate += 1

assert n_violate > n_seeds * 0.7  # 绝大多数种子在同样的配置下就已经违反严格单调
print(f"OK part1: {n_violate}/{n_seeds} 个种子在 02 号文件的原始配置下出现局部残差不降反升"
      f"（'每一步都不会变差' 不是普遍成立的性质，只是 02 号文件恰好没撞上反例）。")

# 02 号文件用的种子 0：n_iter=5 时确实单调，但只是因为提前停止
torch.manual_seed(0)
base0 = nn.Linear(64, 64, bias=False)
hist0_short = LoftQLinear(base0, r=8, n_iter=5, block_size=64).get_convergence()
assert all(hist0_short[i] >= hist0_short[i + 1] - 1e-6 for i in range(len(hist0_short) - 1))

torch.manual_seed(0)
base0b = nn.Linear(64, 64, bias=False)
hist0_long = LoftQLinear(base0b, r=8, n_iter=15, block_size=64).get_convergence()
first_violation = next(i for i in range(len(hist0_long) - 1) if hist0_long[i + 1] > hist0_long[i] + 1e-6)
assert first_violation >= 5
print(f"OK part2: 种子0 n_iter=5 单调 {[round(v, 4) for v in hist0_short]}；"
      f"放宽到 n_iter=15 后从第 {first_violation} 步开始也回升 {[round(v, 4) for v in hist0_long]}。")

# Part 3: 机制溯源——重新量化时，"沿用上一轮过时的 Q" vs "按当前 target 重算的最优 Q"，
# 后者未必更好，因为 absmax=max(abs(.)) 不是 L2 误差最优的缩放选择
torch.manual_seed(0)
base_m = nn.Linear(64, 64, bias=False)
W = base_m.weight.data.clone().float()
r = 8
Q = nf4_quant_dequant(W, block_size=64)
stale_beats_fresh_count, total_blocks_checked = 0, 0
for it in range(6):
    U, S, Vt = torch.linalg.svd(W - Q, full_matrices=False)
    sqrt_S = S[:r].sqrt()
    BA = (U[:, :r] * sqrt_S.unsqueeze(0)) @ (sqrt_S.unsqueeze(-1) * Vt[:r, :])
    target = W - BA
    indices, absmax, orig_shape, pad = nf4_quantize(target, block_size=64)
    Q_new = nf4_dequantize(indices, absmax, orig_shape, pad, target.device, target.dtype)
    if it > 0:
        err_stale = (target - Q).pow(2).sum(dim=-1)      # 沿用上一轮的 Q，作用在这一轮的新 target 上
        err_fresh = (target - Q_new).pow(2).sum(dim=-1)  # 这一轮按当前 target 重新算出的、自己坐标系里最优的 Q
        stale_beats_fresh_count += (err_stale < err_fresh - 1e-8).sum().item()
        total_blocks_checked += err_stale.numel()
    Q = Q_new

assert stale_beats_fresh_count > total_blocks_checked * 0.3
print(f"OK part3: {total_blocks_checked} 个 block-iteration 样本里，有 {stale_beats_fresh_count} 次"
      f"（{stale_beats_fresh_count/total_blocks_checked*100:.1f}%）'沿用过时的 Q' 反而比'重新算出的最优 Q' 误差更小——"
      f"absmax=max(abs(.)) 只保证不溢出，不保证 L2 误差最优，这是交替最小化不能严格单调下降的根因。")
```

实测(`.venv` 真跑):`34/40` 个种子(85%)在 02 号文件的原始配置下就已经出现局部残差回升;种子 0 在 `n_iter=5` 时历史是 `[0.4248, 0.3405, 0.3302, 0.3298, 0.3289, 0.3224]`(严格单调),放宽到 `n_iter=15` 后历史变成 `[0.4248, 0.3405, 0.3302, 0.3298, 0.3289, 0.3224, 0.3213, 0.3239, 0.3283, 0.3326, 0.3316, 0.3348, 0.3348, 0.3363, 0.3467, 0.3501]`,从第 6 步(索引 6→7,`0.3213→0.3239`)开始回升,而且回升之后再也没有回到过第 6 步的水平。机制溯源:320 个 block-iteration 样本里,有 214 个(66.9%)"沿用过时的 Q"反而误差更小——这不是罕见的边缘情况,是这套 `max(abs(·))` block-wise 量化策略下相当普遍的现象。

**可运行例子(2/2):QDoRA 最小原型——现场验证"量化"和"magnitude/direction 分解"这两个轴真的可以正交组合,不是嘴上说说**

```python
import sys
sys.path.insert(0, "learning/lora-family/src")
import torch
import torch.nn as nn
from nf4_quant import nf4_quant_dequant

torch.manual_seed(55)

class QDoRALinear(nn.Module):
    """把 QLoRA 的量化轴和 DoRA 的 magnitude/direction 轴组合起来的最小原型——
    这个仓库没有实现，这里现场验证组合是否可行、初始化性质是否还成立。"""

    def __init__(self, base_linear, r=8, alpha=16, block_size=64):
        super().__init__()
        d_out, d_in = base_linear.weight.shape
        self.scaling = alpha / r

        W0 = base_linear.weight.data.clone().float()
        Q = nf4_quant_dequant(W0, block_size=block_size)   # 量化轴：base 换成量化后的 Q（冻结）
        base_linear.weight.data.copy_(Q.to(base_linear.weight.dtype))
        for p in base_linear.parameters():
            p.requires_grad = False
        self.base = base_linear

        # magnitude/direction 轴：m 的初始值从 Q（不是原始 W0）的列范数取
        self.m = nn.Parameter(Q.norm(dim=0).clone())
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def _compute_W(self):
        Q = self.base.weight.data.float()
        V = Q + self.scaling * (self.B @ self.A)
        norm = V.norm(dim=0).clamp(min=1e-8).detach()
        return (self.m / norm).unsqueeze(0) * V

    def forward(self, x):
        return x @ self._compute_W().T

d_out, d_in = 32, 20
base = nn.Linear(d_in, d_out, bias=False)
W0 = base.weight.data.clone()
layer = QDoRALinear(base, r=4, alpha=8)

# 起点应该精确等于量化后的 Q，而不是原始 W0（量化代价是真实存在的，没有被"吸收掉"）
Q_direct = nf4_quant_dequant(W0.float(), block_size=64)
W_init = layer._compute_W()
diff_vs_Q = (W_init - Q_direct).abs().max().item()
diff_vs_W0 = (W_init - W0).abs().max().item()
assert diff_vs_Q < 1e-5
assert diff_vs_W0 > 1e-4

# 训练几步：m/A/B 都要动，量化后的 base 权重全程不变
m0, A0, B0 = layer.m.data.clone(), layer.A.data.clone(), layer.B.data.clone()
base_w0 = layer.base.weight.data.clone()
opt = torch.optim.AdamW([layer.m, layer.A, layer.B], lr=1e-2)
x, target = torch.randn(6, d_in), torch.randn(6, d_out)
for _ in range(5):
    opt.zero_grad()
    loss = ((layer(x) - target) ** 2).mean()
    loss.backward()
    opt.step()

assert (layer.m.data - m0).abs().max().item() > 0
assert (layer.A.data - A0).abs().max().item() > 0
assert (layer.B.data - B0).abs().max().item() > 0
assert (layer.base.weight.data - base_w0).abs().max().item() == 0.0

print(f"OK: 起点精确等于量化后的 Q（diff={diff_vs_Q:.2e}），不等于原始 W0（diff={diff_vs_W0:.4f}，"
      f"量化代价真实存在，没有被 magnitude/direction 分解'吸收'掉）；训练 5 步后 m/A/B 全部更新，"
      f"量化后的 base 权重精确不变——两个轴（量化 W0、magnitude/direction 化 ΔW）确实可以正交组合。")
```

实测:`diff_vs_Q = 0.00000000`(起点精确等于量化后的 `Q`),`diff_vs_W0 = 0.03331219`(和原始 `W0` 有真实、不可忽略的差距,量化代价没有被"吸收"掉);训练 5 步后 `m`/`A`/`B` 全部移动,`base.weight`(量化后、冻结)最大变化精确为 `0.0`。

**常见坑:** 把"论文和仓库某个例子验证过 X 性质"直接当成"X 性质对任意输入普遍成立"——一个例子通过 `assert` 只能说明"这组特定输入没有触发反例",不是数学上的普遍保证,这条坑本身就是这个案例的核心教训,也是"真实性验证轴"最容易被忽视的一种形式:不是怀疑一个具体的实现有没有 bug,而是怀疑一句"已经验证过"的断言的**适用范围**。另一个坑是把 DoRA 当成 LoftQ 之后"更强的第 5 个量化改进方案",而不是一个完全独立、正交的参数化维度改动——这和 alignment-algorithms 案例 2 里"DPOP 不是第 6 个候选方案,而是回到 DPO 骨架上打的正交补丁"是同一种误区在不同系列里的重演,识别"新方案到底是同一条链的延伸,还是换了一个维度"是这条批判迭代轴上区分度最高的地方。

---

## 案例 2:Houlsby→Pfeiffer→Compacter→(IA)³——同一张压缩谱系,给定约束怎么选、"无解"是不是真无解(决策依据追问轴)

建立在 [03 类知识点 2](03-adapter-core.md)(Houlsby 插 2 次/Pfeiffer 插 1 次,参数量精确减半)、[03 类知识点 4](03-adapter-core.md)(Compacter 用 PHM/Kronecker 分解 + 跨层共享压缩到 83,264)、[03 类知识点 6](03-adapter-core.md)((IA)³ 用 3 个对角缩放向量压到 55,296,且因为不含新增非线性可以无损合并进权重)之上。这条追问轴不纠错,只逼问"给定一个具体约束,你会选哪个、为什么排除别的",而且要考察候选人拿到"无解"这种否定性结论时,能不能进一步区分"这是这几个具体配置凑不出解"还是"这个方法家族在任何超参数下都凑不出解"。

**追问链条完整还原:**

- **Q:** "如果我要求参数预算 < 100K,并且要求推理时能把 adapter **无损合并**进 base 权重(不能留任何额外的运行时开销),这 4 种方法里你选哪个?"——期望候选人先排除 Houlsby(608,640)/Pfeiffer(304,320)/Compacter(83,264 虽然满足参数预算,但内部 `down→GELU→up` 有非线性,`mean(GELU(·))≠GELU(mean(·))`,合并会破坏这个非线性,不能无损合并),只剩 (IA)³(55,296,纯逐元素缩放,可以直接把 `l_k`/`l_v`/`l_ff` 乘进对应权重的列,不引入任何新的非线性)。
- **追问 1(换一个约束维度):** "现在参数预算放宽到 < 400K,但我要求 adapter 内部必须保留非线性表达能力(比如你怀疑任务需要比线性缩放更复杂的变换),你选哪个?"——期望候选人排除 (IA)³(纯线性缩放,没有非线性)和 Houlsby(太贵),剩 Pfeiffer/Compacter 两个候选,并能说清楚这两者的进一步取舍依据(Compacter 参数更省但要多算 Kronecker 展开、多一层跨层共享带来的调试复杂度;Pfeiffer 实现更直接)。
- **追问 2(逼出"空集"这种否定性结论的能力):** "如果同时要求参数预算 < 60K,又要求保留非线性,这 4 种方法(在它们各自最常用的默认超参下)里,选得出来吗?"——期望候选人现场用代码而不是背题的方式验证:Compacter 默认配置(`r=16,n=4`)是 83,264,依然超过 60K;(IA)³ 满足参数预算但没有非线性;这个组合在默认配置下无解。
- **深挖追问(核心考点,检验"无解"是终点还是起点):** "这个'无解'是这 4 种方法在任何超参数下都凑不出来,还是只是你测的这几组默认配置凑不出来?"——期望候选人不满足于"无解"这个表面结论,进一步追问 Compacter 是这 4 个方法里唯一有"连续压缩旋钮"的(`n`——Kronecker 因子数,越大压缩越狠;`r`——瓶颈维度),现场调大 `n`(比如从 4 调到 8)或调小 `r`,重新测参数量,验证"默认配置下无解"和"这个方法家族在任意超参数下都无解"是两个不同强度的结论,不能混为一谈。
- **追问 3(收束,考察工程判断力而非死记硬背):** "既然调大 `n` 能把 Compacter 压到 60K 以下还保留非线性,那是不是 `n` 越大越好,应该默认就用一个很大的 `n`?"——期望候选人指出这不是免费的:`n` 增大,`n³` 那一项(独立于跨层共享的部分)会变大,而且 `in_features`/`out_features` 必须能被 `n` 整除(不是任意 `n` 都能构造出合法的 `PHMLinear`,这是一个真实的结构性约束,不是超参数搜索空间里可以随便取的自由变量),需要在"压缩率"和"是否存在合法的整除关系"之间做权衡,不是无脑调大。

**可运行例子(1/2):把决策过程写成程序化的约束求解器,包含"空集"这个否定性结论也是程序验证出来的**

```python
import sys
sys.path.insert(0, "learning/adapter-tuning-family/src")
import inspect
import torch
from houlsby_minimal import HoulsbyAdapter
from compacter_minimal import CompacterAdapter
from ia3_minimal import _IA3AttnWrapper, _IA3MlpWrapper

# 结构性核实"是否引入新的非线性"，不是凭印象猜——直接检查各自 forward 源码里有没有调用自己的激活函数
def has_own_nonlinearity(forward_fn):
    src = inspect.getsource(forward_fn)
    return "self.act(" in src

assert has_own_nonlinearity(HoulsbyAdapter.forward)      # Houlsby/Pfeiffer 共用这个类，down->act->up
assert has_own_nonlinearity(CompacterAdapter.forward)     # Compacter 同样是 down->act->up（PHM 版）
ia3_attn_src = inspect.getsource(_IA3AttnWrapper.forward)
ia3_mlp_src = inspect.getsource(_IA3MlpWrapper.forward)
assert "act(" not in ia3_attn_src                          # (IA)^3 的 K/V 缩放：纯逐元素乘法，没有新非线性
# mlp 侧确实调用了 act，但那是 GPT-2 MLP 自己原有的激活，l_ff 缩放发生在这次调用之前（作用在线性输出上）
assert ia3_mlp_src.index("l_ff") < ia3_mlp_src.index("self.base.act")

methods = {
    "houlsby":   {"params": 608_640, "own_nonlinear": True,  "merge_lossless": False},
    "pfeiffer":  {"params": 304_320, "own_nonlinear": True,  "merge_lossless": False},
    "compacter": {"params": 83_264,  "own_nonlinear": True,  "merge_lossless": False},
    "ia3":       {"params": 55_296,  "own_nonlinear": False, "merge_lossless": True},
}

def recommend(budget=None, need_nonlinear=None, need_lossless_merge=None):
    cands = set(methods.keys())
    if budget is not None:
        cands = {m for m in cands if methods[m]["params"] < budget}
    if need_nonlinear:
        cands = {m for m in cands if methods[m]["own_nonlinear"]}
    if need_lossless_merge:
        cands = {m for m in cands if methods[m]["merge_lossless"]}
    return cands

scenario1 = recommend(budget=100_000, need_lossless_merge=True)
scenario2 = recommend(budget=400_000, need_nonlinear=True)
scenario3 = recommend(budget=60_000, need_nonlinear=True)

assert scenario1 == {"ia3"}
assert scenario2 == {"pfeiffer", "compacter"}
assert scenario3 == set()   # 默认超参下无解，程序验证，不是拍脑袋

print("只有<100K参数+要求无损合并:", scenario1)
print("<400K参数+要求非线性:", scenario2)
print("<60K参数+要求非线性(默认超参):", scenario3, "-- 默认配置下无解")
```

实测(`.venv` 真跑):`Houlsby`/`Compacter` 的 `forward` 源码里确实各自调用了 `self.act(`,`(IA)³` 的 attention 侧 wrapper 源码里完全没有 `act(` 字样,FFN 侧 `l_ff` 的赋值位置确实在 `self.base.act` 调用之前;三个约束求解结果分别是 `{'ia3'}`、`{'pfeiffer', 'compacter'}`、`set()`(空集)。

**可运行例子(2/2):"无解"是默认超参凑不出来,还是这个方法家族在任意超参数下都凑不出来——现场打破这个空集**

```python
import sys
sys.path.insert(0, "learning/adapter-tuning-family/src")
import torch
from compacter_minimal import CompacterGPT2

# Compacter 是这 4 个方法里唯一有连续压缩旋钮的(r, n)——调大 n（更多跨层共享的 Kronecker 因子）
# 或调小 r，看能不能在保留 GELU 非线性的前提下，把参数量压到例子(1/2)里的 60K 预算以下
BUDGET = 60_000
configs_to_try = [(16, 4), (16, 8), (8, 4), (8, 8)]
results = {}
for r, n in configs_to_try:
    torch.manual_seed(3)
    m = CompacterGPT2(r=r, n=n)
    results[(r, n)] = sum(p.numel() for p in m.parameters() if p.requires_grad)

assert results[(16, 4)] >= BUDGET             # 这个仓库/03号文件默认用的配置，恰好是让例子(1/2)判定"无解"的那一组
under_budget = {cfg: v for cfg, v in results.items() if v < BUDGET}
assert len(under_budget) >= 2                  # 但换几组 (r, n) 就能重新打开这个约束

# 一个真实的结构性限制（不是可以随便调的超参数）：n 必须整除对应的输入/输出维度
try:
    CompacterGPT2(r=4, n=8)
    raised = False
except AssertionError:
    raised = True
assert raised

print("Compacter 在不同 (r,n) 下的可训练参数量:", results)
print(f"能在 <{BUDGET} 预算内同时保留 GELU 非线性的配置: {under_budget}")
print("r=4,n=8 无法构造（n 必须整除维度，这是真实的结构性约束，不是随便调的旋钮）")
print("\n结论：例子(1/2)里的'空集'只是默认超参（r=16全统一, n=4）下的结论，"
      "不是 Compacter 这个方法家族在任意超参下都无解——'当前配置无解' 和 '这个方法家族无解' "
      "是两个不同强度的结论，决策依据类追问经常就卡在这个区分上。")
```

实测:`Compacter` 在 `(16,4)/(16,8)/(8,4)/(8,8)` 四组配置下的可训练参数量分别是 `83,264`、`47,296`、`46,304`、`28,768`——默认的 `(16,4)` 确实超过 60K 预算(印证例子 1/2 的空集结论),但另外三组都能压到预算以下同时保留 GELU;`CompacterGPT2(r=4, n=8)` 真实抛出 `AssertionError: in/out must be divisible by n; got in=768, out=4, n=8`,证明不是任意 `(r,n)` 组合都能构造。

**常见坑:** 把决策依据追问答成"哪个方法更好"这种没有约束条件的价值判断——正确的回答方式永远是先复述约束是什么,再说明约束怎么筛掉了哪些选项。另一个更隐蔽的坑是拿到"无解"这个结论就直接停在这里,不去追问这个"无解"是对固定的几组配置成立,还是对整个方法家族在任意超参数下都成立——本案例已经示范过,前者往往可以通过调整一个连续旋钮重新打开,后者才是真正结构性的"无解"(比如 alignment-algorithms 案例 3 里"既要不需要 ref 又要不需要配对数据"那个空集,是几种方法各自的架构决定的,不存在任何超参数能同时满足两个条件,是更强的一种"无解")。

---

## 案例 3:LoRA 多租户并发服务——单适配器正确→合并方式撞车→并发合并丢更新(工程约束递增轴)

建立在 [01 类知识点 3](01-lora-core.md)(`merge_weights()`,合并后推理零延迟)之上——那个知识点的"AI 研究场景"部分已经提到生产环境有两种典型选择:合并后单独部署(零延迟但焊死在一个适配器上),或者不合并、保留 base+adapter 形式做多适配器热切换(vLLM 等推理框架支持的思路),但没有具体展开"如果真的要服务多个并发用户,这两条路径各自会先撞上什么"。这是这个系列里第一次沿"工程约束递增轴(并发)"展开的案例——`dsa-deep-dive` 的 LRU 缓存案例、`alignment-algorithms-deep-dive` 明确说这条轴线在偏好优化 loss 计算里没有天然对应的知识点,而 PEFT 的"多适配器服务"场景恰好是这条轴线在这个系列里的天然落点。

**追问链条完整还原:**

- **Q(基础,01 类已覆盖):** "LoRA 训练完之后,`merge_weights()` 把 `scaling·B@A` 写回 `base.weight`,这样换来推理零延迟,这一点已经验证过。"
- **追问 1(把单机推向并发):** "现在同一个 base GPT-2 模型要同时服务 100 个不同的用户,每个用户各自训练了自己的 LoRA 适配器(不同的 A、B),你打算怎么用 `merge_weights()` 这套机制服务他们?"——期望候选人推出:如果对同一个共享的 `base` 模块调用 `merge_weights()` 写入某个用户的 delta,这个 base 模块就被"焊死"在这个用户的适配器上了——不能同时正确服务另一个持有不同 delta 的用户,"零延迟"是用"失去多租户灵活性"换来的,这正是 01 类知识点 3 已经指出的权衡,但这里要把它具体推演到"多用户"这个场景。
- **追问 2(制造真实竞态,不是空谈"会有问题"):** "如果两个请求线程恰好同时对同一个共享 base 模块做类似合并的操作,一个想合并用户 A 的 delta,一个想合并用户 B 的 delta,会发生什么?"——期望候选人推理出这是共享可变状态上的经典竞态:一次"读当前权重、算出新权重、写回"的合并操作如果不是原子的,两个线程交错执行可能导致其中一个用户的 delta 被完全覆盖丢失(不是"两个 delta 都被錯误叠加"这种简单的过度计数,是更隐蔽的"更新丢失",数据损坏,不是变慢)。
- **追问 3(工程解法,考察是否知道生产系统怎么做):** "所以生产环境的多 LoRA 服务(比如 vLLM 的 multi-LoRA 支持)是怎么解决这个问题的?"——期望候选人推出:根本不去合并,`base` 权重全程只读、只有一份;每个请求带着自己的一小份 A、B,在同一次前向里对 batch 中不同请求分别应用各自的低秩增量(segmented/batched 低秩矩阵乘法,类似 Punica/S-LoRA 的做法)。这是从"合并进单一权重、靠锁串行化"转向"从来不合并、用批处理天然隔离"的范式转变——比 dsa-deep-dive LRU 缓存案例"别用一把全局锁,换成细粒度锁"更进一步:是压根不引入共享可变状态。
- **深挖追问(定量,不满足于"这样更省显存"这种定性描述):** "不合并、每个请求各自算一次 `base(x)+delta_i(x)`,相比 N 个用户各自合并出一份完整权重、各存一份,存储开销差多少?"——期望候选人能现场估算:N 份合并后的完整权重存储量是 `N×完整模型参数量`,不合并只需要 `完整模型参数量 + N×(A+B 的参数量)`——后者只随 N 线性增长的部分是"每个用户的小增量",不是"每个用户一份完整模型"。

**可运行例子(1/2):确定性复现"合并"操作在并发场景下的更新丢失——不是空谈会有竞态,是真实制造出来**

用 `threading.Event` 强制精确的线程交错时机,而不是依赖操作系统调度的运气:

```python
import threading
import torch

def naive_merge(shared_weight, delta, pause_event=None, resume_event=None):
    """模拟"把这个租户的 LoRA delta 合并进共享 base 权重"这个操作最朴素、最容易踩坑的写法：
    读当前值、算新值、写回，三步分开执行——如果没有精心设计的原子性保护，
    这正是一个手搓的多租户权重缓存最容易写成的样子。"""
    current = shared_weight.clone()             # 读
    if pause_event is not None:
        pause_event.set()
        resume_event.wait()                      # 强制卡在"读完、还没写回"这个精确点
    new_value = current + delta                  # 算（假设这段时间内 current 不会被别人改）
    shared_weight.copy_(new_value)                # 写回（是覆盖，不是原子加）

def run_scenario(use_lock):
    shared_weight = torch.zeros(4, 4)
    lock = threading.Lock() if use_lock else None
    delta_A, delta_B = torch.ones(4, 4) * 1.0, torch.ones(4, 4) * 2.0
    e1, e2 = threading.Event(), threading.Event()

    def worker_A():
        if lock:
            with lock:
                naive_merge(shared_weight, delta_A)
        else:
            naive_merge(shared_weight, delta_A, pause_event=e1, resume_event=e2)

    def worker_B():
        if lock:
            with lock:
                naive_merge(shared_weight, delta_B)
        else:
            naive_merge(shared_weight, delta_B)

    tA = threading.Thread(target=worker_A)
    tA.start()
    if not use_lock:
        e1.wait()          # 等 A 读完、暂停在"写回之前"这个精确点
    tB = threading.Thread(target=worker_B)
    tB.start()
    tB.join()               # 不加锁：让 B 在 A 暂停期间完整跑完，制造"A 用过期快照写回"的条件
    if not use_lock:
        e2.set()
    tA.join()
    return shared_weight.clone()

result_unsafe = run_scenario(use_lock=False)
result_safe = run_scenario(use_lock=True)

assert torch.equal(result_unsafe, torch.ones(4, 4) * 1.0)                     # B 的更新被 A 的过期写回完全覆盖丢失
assert torch.equal(result_safe, torch.ones(4, 4) * 1.0 + torch.ones(4, 4) * 2.0)  # 加锁后两个租户的 delta 都正确落地

print(f"OK: 不加锁时共享权重只反映租户A的delta {result_unsafe[0].tolist()}"
      f"（租户B的更新被完全覆盖丢失，不是'变慢'，是数据损坏）；"
      f"加锁后共享权重正确反映两个租户的delta之和 {result_safe[0].tolist()}。")
```

实测(`.venv` 真跑):不加锁时,`shared_weight` 最终只反映租户 A 的 delta(`[1.0, 1.0, 1.0, 1.0]`),租户 B 的更新被 A 用过期快照写回时完全覆盖丢失;加锁后 `shared_weight` 正确反映两个租户的 delta 之和(`[3.0, 3.0, 3.0, 3.0]`)。这个"更新丢失"用 `threading.Event` 精确卡时机 100% 确定性复现,不是"跑多次可能会出错"的概率性论证。

**可运行例子(2/2):不合并、批处理分别应用各自 delta 的正确性,以及"忘记按样本索引"这个真实的坑,再加存储量对比**

```python
import sys
sys.path.insert(0, "learning/lora-family/src")
import torch
import torch.nn as nn
from lora_minimal import LoRALinear

torch.manual_seed(3)
# 用接近 GPT-2 c_attn 的真实规模，而不是玩具尺寸——这样存储对比才有意义
d_in, d_out, r = 768, 2304, 8
base = nn.Linear(d_in, d_out, bias=False)
for p in base.parameters():
    p.requires_grad = False

N_TENANTS = 20
tenants = []
for i in range(N_TENANTS):
    torch.manual_seed(100 + i)
    t = LoRALinear(base, r=r, alpha=16)
    with torch.no_grad():
        t.B.normal_(std=0.05)   # 打破零初始化，让每个租户的 delta 各不相同且非零
    tenants.append(t)

torch.manual_seed(7)
x = torch.randn(N_TENANTS, d_in)   # 一个 batch，每一行属于不同租户

# 正确性基准：每个租户各自单独调用自己的 LoRALinear
per_tenant_out = torch.stack([tenants[i](x[i:i + 1]).squeeze(0) for i in range(N_TENANTS)])

# 不合并、批处理分别应用各自 delta（生产多 LoRA 服务的真实做法）
base_out_batch = base(x)
A_stack = torch.stack([t.A for t in tenants])   # (N, r, d_in)
B_stack = torch.stack([t.B for t in tenants])   # (N, d_out, r)
lora_out = torch.einsum("ni,nri->nr", x, A_stack)
lora_out = torch.einsum("nr,nor->no", lora_out, B_stack)
batched_out = base_out_batch + tenants[0].scaling * lora_out

diff = (per_tenant_out - batched_out).abs().max().item()
assert diff < 1e-4

# 真实的坑：忘记按样本索引，把租户0的 adapter 用到整个 batch 上——不会报错，只会悄悄serve错模型
wrong_out = base_out_batch + tenants[0].scaling * (x @ tenants[0].A.T @ tenants[0].B.T)
diff_tenant0 = (wrong_out[0] - per_tenant_out[0]).abs().max().item()   # 租户0自己的行，碰巧还是对的
diff_tenant1 = (wrong_out[1] - per_tenant_out[1]).abs().max().item()   # 其余租户，全部serve错
assert diff_tenant0 < 1e-4
assert diff_tenant1 > 1e-2

# 存储对比：N 份各自合并的完整模型 vs 一份共享 base + N 份小 delta
full_params = sum(p.numel() for p in base.parameters())
delta_params = tenants[0].A.numel() + tenants[0].B.numel()
merged_copies_storage = N_TENANTS * full_params
unmerged_storage = full_params + N_TENANTS * delta_params
assert merged_copies_storage > unmerged_storage * 3

print(f"OK: 批处理+按样本索引 vs 逐租户ground truth 最大diff={diff:.2e}（正确）；"
      f"忘记按样本索引：租户0 diff={diff_tenant0:.2e}(碰巧对)，租户1 diff={diff_tenant1:.2e}(悄悄serve错，不报错)；"
      f"N={N_TENANTS} 时，各自合并存储={merged_copies_storage:,} vs 共享base+小delta存储={unmerged_storage:,}，"
      f"前者是后者的 {merged_copies_storage/unmerged_storage:.1f} 倍。")
```

实测:批处理(按样本索引正确应用各自 delta)和逐租户 ground truth 的最大逐元素差异是 `1.192e-06`(浮点误差范围内一致);"忘记按样本索引、把租户 0 的 adapter 用到整个 batch"这个真实的坑,租户 0 自己那一行碰巧还是对的(`diff=8.345e-07`),但租户 1 那一行差异达到 `7.311e-01`——不报错,只是悄悄给租户 1 返回了租户 0 的适配器算出来的结果;存储对比,`N=20` 个租户时,各自合并存储 `35,389,440` 参数,共享 base + 小 delta 的存储只要 `2,260,992` 参数,前者是后者的 `15.7` 倍。

**常见坑:** 只想到"合并会让 base 焊死在一个适配器上"这个静态结论,想不到"如果真的对共享权重做并发合并会产生数据损坏"这个动态过程——两者是递进的,前者是设计选择的权衡,后者是没有设计好会真实发生的 bug。另一个坑是实现"批处理服务多租户"时,想当然地认为"batch 里每一行都过一遍同一个 adapter"就够了,不去确认每一行有没有按正确的样本索引取到**自己**的 A、B——这个错误不会报错、不会让张量形状不对,只会悄悄给部分用户返回错误模型的输出,是最难在测试里发现的一类 bug。

---

## 案例 4:`merge_weights()` 的"删除"承诺——minimal 双重计数→peft flag 防御→peft 结构级卸载(真实性验证轴)

直接承接案例 3,建立在 [01 类知识点 3](01-lora-core.md)(`merge_weights()` 的 docstring 写"删除 A、B",但实现代码没有真的删除,继续调用会双重计数)之上——01 号文件已经发现这个 minimal 实现的具体问题,但没有回答一个更重要的后续问题:**这个仓库教学用的简化实现有这个问题,不代表真正会被部署的 peft 库也有同样的问题**。这是"真实性验证轴"里最容易被跳过的一步——发现一个 toy 实现的 bug 之后,不去对照生产库重新验证,就默认"这是这类方法的通病",是过度泛化。

**追问链条完整还原:**

- **面试官:** "01 号文件发现这个仓库的 `LoRALinear.merge_weights()` docstring 说会删除 A、B,但实际代码没有删,继续调用 `lora(x)` 会把已经写进 `base.weight` 的 delta 再加一遍。这是不是意味着,任何 LoRA 库的'合并'操作都有这个双重计数的风险?"
- **候选人(容易偷懒的回答):** "这个仓库的实现有这个问题,应该是个通用的坑,用别的库估计也要小心。"
- **面试官追问(把"这个 toy 实现有 bug"和"这类方法本身有这个通病"拆开):** "你有没有拿真实的 peft 库跑一遍同样的场景,看它是不是也会双重计数?"
- **候选人如实说没有,现场去验证,依次得到下面几层发现:**
- **追问 1(现场用新的种子/维度复现 minimal 版本的问题,不是照抄 01 号文件的数字):** "先复现一遍——这次换一组新的种子和矩阵维度,minimal 版本的问题还在吗?"——期望复现:换 `nn.Linear(24,40)`,`r=6`,种子 909,训练 4 步后合并,`lora.base(x)`(正确路径)和合并前的 `lora(x)` 几乎一致(`diff≈3.6e-7`),但如果误用 `lora(x)`(合并后继续走原来的 wrapper),`diff` 会跳到接近 1(`≈0.988`)——不是种子 1 特有的巧合。
- **追问 2(真实 peft,第一层防御):** "peft 库的 `model.merge_adapter()`(合并但不卸载)会不会有同样的问题?"——期望候选人现场验证发现:`merge_adapter()` 之后,`lora_A`/`lora_B` **依然存在**(没有被删除,和这个仓库的 minimal 实现表面上一样"没删"),但连续两次调用 `model(...)` 得到的结果完全一致(`diff` 精确为 `0.0`)——真正生效的不是"删除",而是 `LoraLayer.forward()` 里对 `self.merged` 这个 boolean 标志的运行时检查:一旦 `merged=True`,`forward` 直接走 `self.base_layer(x)`,完全跳过 `lora_A`/`lora_B` 那个分支,即使这两组参数还占着显存。
- **深挖追问(逼问这层防御的代价):** "既然 `lora_A`/`lora_B` 还在,`merge_adapter()` 之后是不是白白浪费了这部分显存?"——期望候选人指出这是一个有意的设计权衡:保留这两组参数是为了能调用 `model.unmerge_adapter()` 把合并撤销、换回未合并状态(比如临时切换到另一个适配器),用"多占一点显存"换"合并状态可逆"的灵活性,不是遗漏。
- **追问 3(真实 peft,第二层防御):** "那 `merge_and_unload()` 呢,和 `merge_adapter()` 是同一件事吗?"——期望候选人现场验证发现:`merge_and_unload()` 更进一步,不仅合并权重,还把 `parent` 模块里的 LoRA wrapper **整个替换**成 `get_base_layer()` 返回的原始模块——合并之后 `type(layer)` 从 `peft.tuners.lora.layer.Linear` 变成了 `transformers.pytorch_utils.Conv1D`(GPT-2 的原始层类型),这个模块上连 `lora_A` 属性都不存在了,不是"运行时绕过",是结构上真的不在计算图里了,对应 `merge_and_unload` 这个名字里的"unload"部分。
- **深挖追问(收束,考察分类能力):** "把这三种'合并'行为按防御强度排个序。"——期望候选人总结:第一层(这个仓库的 minimal `merge_weights()`)没有任何防御,是纯粹的"我信任调用方不会在合并后误用旧的 wrapper";第二层(peft 的 `merge_adapter()`)是运行时防御,状态可逆,但没有释放内存;第三层(peft 的 `merge_and_unload()`)是结构级防御,彻底移除了双重计数的可能性,但状态不可逆(要切回未合并版本得重新构造一个 `PeftModel`)。

**可运行例子(1/2):用全新的种子/维度重新验证 minimal 实现的双重计数,不是复述 01 号文件的数字**

```python
import sys
sys.path.insert(0, "learning/lora-family/src")
import torch
import torch.nn as nn
from lora_minimal import LoRALinear

torch.manual_seed(909)  # 全新种子，和 01 号文件的种子 1 不同
base = nn.Linear(24, 40, bias=False)
lora = LoRALinear(base, r=6, alpha=12)
x = torch.randn(5, 24)

opt = torch.optim.SGD([lora.A, lora.B], lr=0.3)
target = torch.randn(5, 40)
for _ in range(4):
    opt.zero_grad()
    loss = ((lora(x) - target) ** 2).mean()
    loss.backward()
    opt.step()

with torch.no_grad():
    out_before_merge = lora(x).clone()
    lora.merge_weights()
    out_via_base_only = lora.base(x)          # 正确路径
    out_via_wrapper_again = lora(x)             # 误用：合并后又走了原来的 wrapper

diff_correct_path = (out_via_base_only - out_before_merge).abs().max().item()
diff_double_count = (out_via_wrapper_again - out_before_merge).abs().max().item()

assert diff_correct_path < 1e-4
assert diff_double_count > 1e-2
print(f"OK: 全新种子/维度下复现同一个问题——正确路径 diff={diff_correct_path:.3e}，"
      f"误用路径(双重计数) diff={diff_double_count:.3e}，不是 01 号文件那一组种子特有的巧合。")
```

实测:正确路径(合并后调用 `lora.base(x)`)和合并前 `lora(x)` 的差异是 `3.576e-07`(浮点精度内一致);误用路径(合并后继续调用 `lora(x)`)差异是 `9.884e-01`——换了全新的种子和矩阵维度,同一个问题依然复现,不是 01 号文件那组特定输入的巧合。

**可运行例子(2/2):真实 peft 的两层防御——`merge_adapter()` 的 flag 防御 vs `merge_and_unload()` 的结构级卸载**

```python
import sys
sys.path.insert(0, "learning/lora-family/src")
import torch
from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType

torch.manual_seed(303)
base = GPT2LMHeadModel.from_pretrained("gpt2")
config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, target_modules=["c_attn"], lora_dropout=0.0)
model = get_peft_model(base, config)
model.eval()
for name, p in model.named_parameters():
    if "lora_B" in name:
        torch.nn.init.normal_(p, std=0.03)   # 打破零初始化，让合并前后真的有数值差异可比

tok_ids = torch.tensor([[10, 20, 30, 40, 50]])
with torch.no_grad():
    out_before = model(tok_ids).logits.clone()

layer0 = model.base_model.model.transformer.h[0].attn.c_attn
assert layer0.merged is False

# 第一层防御：merge_adapter()——写回权重 + 设置 merged 标志，但不删除 lora_A/lora_B
model.merge_adapter()
assert layer0.merged is True
assert "default" in layer0.lora_A   # 依然存在，没有被删除

with torch.no_grad():
    out_after_merge = model(tok_ids).logits.clone()
    out_after_merge_again = model(tok_ids).logits.clone()   # 合并后再调用一次 forward

assert (out_before - out_after_merge).abs().max().item() < 1e-3
assert (out_after_merge - out_after_merge_again).abs().max().item() == 0.0   # 没有双重计数
model.unmerge_adapter()
assert layer0.merged is False   # 状态可逆——这正是保留 lora_A/lora_B 换来的灵活性

# 第二层防御：merge_and_unload()——不仅合并，还把整个 wrapper 模块换成原始 base 层
type_before = type(layer0).__name__
merged_model = model.merge_and_unload()
layer0_after = merged_model.transformer.h[0].attn.c_attn
type_after = type(layer0_after).__name__

assert type_before == "Linear"     # peft 的 LoRA wrapper 类
assert type_after == "Conv1D"      # GPT-2 原始层类型——wrapper 彻底不在了
assert not hasattr(layer0_after, "lora_A")

with torch.no_grad():
    out_unloaded = merged_model(tok_ids).logits.clone()
assert (out_before - out_unloaded).abs().max().item() < 1e-3

print(f"OK: merge_adapter() 后 lora_A 仍存在但 forward 不再双重计数（靠 merged 标志运行时防御，可逆）；"
      f"merge_and_unload() 后模块类型从 {type_before} 变成 {type_after}，lora_A 属性彻底消失"
      f"（结构级防御，不可逆）——这个仓库 minimal 实现的 merge_weights() 两层防御都没有。")
```

实测(`.venv` 真跑):`merge_adapter()` 前后 forward 差异 `4.578e-05`(合并本身引入的浮点误差,不是双重计数);合并后连续两次调用 forward,差异精确为 `0.0`——没有双重计数,即使 `lora_A`/`lora_B` 依然存在(`"default" in layer0.lora_A` 为 `True`);`unmerge_adapter()` 之后 `merged` 恢复 `False`,状态可逆。`merge_and_unload()` 之后,模块类型从 `Linear`(peft 的 LoRA wrapper)变成 `Conv1D`(GPT-2 原始层),`hasattr(layer0_after, "lora_A")` 为 `False`,forward 差异同样是 `4.578e-05`,和 `merge_adapter()` 一致。

**常见坑:** 发现一个 toy/教学实现有 bug 之后,不做进一步验证就把结论泛化成"这类方法/这个生产库大概率也有同样的问题"——这正是"真实性验证轴"要拆穿的典型陷阱,和简历上"做了性能优化"这类抽象表述一样,都需要用具体的、可复现的验证压实,而不是停留在"应该是这样吧"的合理推测。另一个坑是看到 `merge_adapter()` 之后 `lora_A` 依然存在,就断定"这也是一个 bug、没有真正合并"——没有理解到"参数还在显存里"和"参数是否还参与计算"是两件独立的事,`self.merged` 这个运行时标志专门负责后者,这是一种常见但容易被忽略的防御性设计模式(保留状态、用一个开关控制是否生效,而不是物理删除),在很多允许"撤销"操作的系统里都能看到同样的思路。

---

## 案例 5:QLoRA 4bit 训练边界——单卡单模型实测→外推到 7B→多卡该撞的墙(真实性验证轴 + 规模递增轴)

建立在 [02 类知识点 3](02-quantized-lora.md)(`qlora_peft.py`,本机真实 RTX 3080 Ti GPU 跑通,TinyLlama-1.1B 4bit 加载显存 1049.3MB)和 [02 类知识点 6](02-quantized-lora.md)("真 4bit 训练 + 真 bitsandbytes 从未同时出现"这条精确边界)之上。这是全篇的收尾案例,呼应 alignment-algorithms-deep-dive 案例 4(`dpo_minimal.py` 从 CPU/单卡/200 条数据的规模,外推到 7B/多卡会遇到什么)的结构,但这里有真实 GPU 可以做更进一步的量化外推,而不是纯粹的定性讨论。

**追问链条完整还原:**

- **Q:** "这个仓库有没有一个例子完整验证了'真实 4bit 量化 + 真实多步训练'在大模型上同时发生?"——期望:没有,[02 类知识点 6](02-quantized-lora.md) 已经用 `grep` 钉死这个边界(全仓库 3 处出现 `bitsandbytes` 的地方,没有一处同时具备训练循环)。
- **追问 1(把'单卡单模型能跑'推向更大规模):** "TinyLlama-1.1B 真实测过 4bit 加载显存 1049.3MB。如果换成 7B 模型,还能在这块 16GB 的 3080 Ti 上做 QLoRA 训练吗?"——期望候选人不要凭感觉猜,而是先问"能不能测出一个真实的、和具体模型无关的'每参数字节数'比例,再拿这个比例去外推",而不是直接把 TinyLlama 的总显存除以 TinyLlama 的参数量就当成放之四海而皆准的比例(那个比例混进了 embedding/lm_head 不被量化、LoRA 参数、分配器开销等因素,不是纯粹的"4bit 存储"本身)。
- **追问 2(方法论追问,逼问怎么把"比例"测干净):** "怎么测出一个不掺杂固定开销的、干净的'每参数字节数'?"——期望候选人想到:量两个不同大小的 4bit 层,用**差值**除以**参数量差值**,这样两次测量里共同的固定开销(CUDA context、Python 对象本身的内存)会被减掉,剩下的就是边际的、真正只跟参数量成正比的部分。
- **追问 3(诚实标注外推的局限,不能把外推当成实测):** "这个外推准吗?"——期望候选人现场做一次**交叉检验**:用测出来的干净比例反推 TinyLlama 应该占多少显存,和 02 号文件已经实测的 1049.3MB 比较,发现纯"参数量×每参数字节数"的估算明显偏低(说明真实模型里有相当一部分显存不是来自被量化的权重本身),进而对 7B 的外推结果做一次修正,而不是直接把"干净比例×7B"当成最终答案。
- **深挖追问(检验对多卡的理解是不是只停留在名词):** "如果算下来 7B 单卡确实放得下,那'多卡'能帮上什么忙,是不是纯粹为了更大的模型才需要?"——期望候选人说清楚:`device_map="auto"` 本身就能在多卡场景下自动把不同层分布到不同 GPU(pipeline 式切分),但这样切分并不会让训练变快(层与层之间依然是串行执行的依赖关系,只是显存被摊开了);真正要加速需要数据并行(每张卡一份完整模型,处理不同的数据,靠梯度同步对齐)或张量并行(把单个大矩阵乘法本身切开分布到多卡,需要卡间通信),这是三种解决完全不同问题的技术,不能因为都叫"多卡"就混为一谈。
- **再深挖(诚实边界,收束整个案例):** "这台机器能不能验证多卡场景下的任何一种说法?"——期望候选人主动用 `torch.cuda.device_count()` 现场检查,如实报告这台机器只有 1 张卡;即使调用了 `device_map="auto"`,也应该现场验证它在单卡机器上到底做了什么决策(是不是真的"什么都不用切,全放一张卡上"),而不是假装这台机器验证过多卡行为——"这个仓库/这台机器验证不到什么"和"验证到了什么"同样重要,是这整篇追加内容想反复强调的纪律。

**可运行例子(1/2):真实测量干净的"每参数字节数",外推到 7B,并用已有的真实测量做交叉检验(不是拍脑袋外推)**

```python
import torch
import bitsandbytes as bnb

assert torch.cuda.is_available()

def measure_linear4bit_memory(in_features, out_features, trials=3):
    best = None
    for _ in range(trials):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        layer = bnb.nn.Linear4bit(in_features, out_features, bias=False,
                                   compute_dtype=torch.float16, quant_type="nf4").to("cuda")
        torch.cuda.synchronize()
        delta = torch.cuda.memory_allocated() - mem_before
        best = delta if best is None else min(best, delta)
        del layer
        torch.cuda.empty_cache()
    return best

# 量两个不同大小的真实 4bit 层，用差值消掉两次测量共同的固定开销
size_small, size_large = (4096, 4096), (4096, 16384)
delta_small = measure_linear4bit_memory(*size_small)
delta_large = measure_linear4bit_memory(*size_large)
n_small, n_large = size_small[0] * size_small[1], size_large[0] * size_large[1]
marginal_bytes_per_param = (delta_large - delta_small) / (n_large - n_small)
assert 0.5 < marginal_bytes_per_param < 0.6   # 真实NF4打包是4bit(0.5B)+per-block absmax的少量开销

HYPOTHETICAL_7B = 7_000_000_000
naive_7b_gb = HYPOTHETICAL_7B * marginal_bytes_per_param / 1e9

# 交叉检验：用这个干净比例反推 TinyLlama 应该占多少显存，和 02 号文件已经实测的 1049.3MB 比较
TINYLLAMA_PARAMS = 1_100_000_000
TINYLLAMA_MEASURED_MB = 1049.3   # 引用 02-quantized-lora.md 知识点 3 的真实测量结果，这里不重新加载模型
tinyllama_naive_mb = TINYLLAMA_PARAMS * marginal_bytes_per_param / 1e6
correction_factor = TINYLLAMA_MEASURED_MB / tinyllama_naive_mb
assert correction_factor > 1.3   # 纯"参数量x每参数字节数"确实是真实占用的一个非平凡的低估

corrected_7b_gb = naive_7b_gb * correction_factor
total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
assert corrected_7b_gb < total_gpu_gb

print(f"OK: 测得干净比例={marginal_bytes_per_param:.4f}B/param；"
      f"naive 7B权重外推={naive_7b_gb:.2f}GB；"
      f"交叉检验：naive TinyLlama估算={tinyllama_naive_mb:.1f}MB vs 02号文件实测={TINYLLAMA_MEASURED_MB}MB，"
      f"修正系数={correction_factor:.2f}x（embedding/lm_head未必全部量化+LoRA参数+分配器开销）；"
      f"修正后7B估算={corrected_7b_gb:.2f}GB，本机GPU总显存={total_gpu_gb:.2f}GB，"
      f"权重本身大概率放得下，但这不包括激活值/KV cache/优化器状态，一个都没测。")
```

实测(`.venv` 真跑,本机真实 RTX 3080 Ti):测得干净比例 `0.5159` bytes/param;naive 7B 权重外推 `3.61GB`;交叉检验发现 naive 估算 TinyLlama 应占 `567.5MB`,而 02 号文件真实测量是 `1049.3MB`,修正系数 `1.85x`(说明有相当一部分显存来自没有被量化的层、LoRA 参数、显存分配器的额外开销,不是纯粹的"权重×每参数字节数"能覆盖的);修正后的 7B 估算 `6.68GB`,本机 GPU 总显存 `17.18GB`——权重本身大概率放得下,但这个数字完全不包括激活值、KV cache、优化器状态,一个都没有测。

**可运行例子(2/2):这台机器能验证到什么、验证不到什么——用真实 API 而不是猜测确认边界**

```python
import sys
sys.path.insert(0, "learning/lora-family/src")
import torch

assert torch.cuda.is_available()
device_count = torch.cuda.device_count()
assert device_count == 1   # 如实确认：这台机器只有一张卡

from qlora_peft import build_peft_qlora_tinyllama
model, tokenizer = build_peft_qlora_tinyllama(r=8, alpha=16)

# device_map="auto" 在有多张卡时会自动把不同层分布到不同 GPU；
# 在只有一张卡的机器上，它实际做的决策是什么？现场核实，不要假设
devices_used = set(str(p.device) for p in model.parameters())
assert len(devices_used) == 1
assert devices_used == {"cuda:0"}

print(f"OK: torch.cuda.device_count()={device_count}；"
      f"device_map='auto' 加载后，模型全部 {len(list(model.parameters()))} 组参数"
      f"实际占用的设备集合={devices_used}——在单卡机器上，'auto' 退化成'全部放在 cuda:0'，"
      f"这行代码在这台机器上完全没有机会执行到真正的多卡切分逻辑。任何关于'切分到多卡会怎样'"
      f"的讨论，都只是工程常识推理，不是这台机器能验证的实测结论。")
```

实测:`torch.cuda.device_count()=1`;`device_map="auto"` 加载 TinyLlama 之后,模型全部参数实际占用的设备集合是 `{'cuda:0'}`——在这台单卡机器上,"auto" 策略退化成"全部放在唯一的那张卡上",这份代码里真正负责多卡切分的那部分逻辑,在这台机器上完全没有被执行到过。

**常见坑:** 把"总显存 ÷ 总参数量"当成一个可以直接外推的"每参数字节数"——这个比例混进了没有被量化的层(embedding/lm_head 等)、LoRA adapter 参数、显存分配器的额外开销,不是纯粹的量化存储本身;真正干净的比例需要通过"量两个不同规模、做差值"的方法把固定开销消掉。另一个坑是被问到"7B/多卡会怎样"时,要么编一个听起来专业但没有依据的具体数字,要么因为没法验证就干脆不给出任何方向性判断——诚实的做法是把"可以外推的部分"(权重本身的静态存储,量级上有把握)和"外推不到的部分"(激活值、KV cache、优化器状态,以及任何需要真正多卡才能观察到的行为)明确分开说,并且愿意现场用一行 `torch.cuda.device_count()` 或者检查 `model.parameters()` 的 `.device` 去确认"这台机器到底能验证到哪一步",而不是含糊带过。

---

## 小结:5 个案例对应调研发现的哪些轴线

| 案例 | 规模递增轴 | 工程约束递增轴(并发) | 方案批判迭代轴 | 决策依据追问轴 | 真实性验证轴 |
|---|---|---|---|---|---|
| 1. LoRA→QLoRA→LoftQ→DoRA | | | ✅ 核心 | | ✅(交替最小化断言的适用范围) |
| 2. Houlsby→…→(IA)³ 压缩决策 | | | | ✅ 核心 | |
| 3. LoRA 多租户并发服务 | | ✅ 核心 | | | |
| 4. merge_weights() vs 真实 peft | | | | | ✅ 核心 |
| 5. QLoRA 4bit 边界 + 规模外推 | ✅ 核心 | | | | ✅(外推的交叉检验) |

这 5 个案例不是要把 24 个知识点全部重写——它们演示的是**方法论本身**:拿到任何一个已经掌握的知识点,都可以自己追问"面试官连续三次指出具体缺陷,我的下一个方案到底是同一条链的延伸,还是换了一个正交的维度""给定一个具体约束,排除掉的选项是这几个具体配置排除的,还是这个方法家族在任何超参数下都排除""这个操作在单次调用时正确,并发调用时还正确吗""这个仓库/这篇论文的某句结论,我是该无条件相信,还是应该自己用新的输入重新跑一遍""如果规模再往上跳一个数量级,我现在能给出的数字有多少是真的测出来的、有多少是外推、外推的误差可能有多大"。真正的二面深度,是能不能对着一个自己没准备过的知识点,现场把这几条轴线走一遍——而且,像案例 1 和案例 4 那样,愿意先怀疑一句"已经验证过"的断言,再动手验证一遍它到底对不对、对到什么范围为止。
