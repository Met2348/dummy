# 01 · RoPE 外推家族深挖(RoPE Scaling Family)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批是 `learning/long-context/` 整个系列的地基——后面 Ring Attention、Infini-Attention、NIAH/RULER 评测、capstone 训练,统统建立在"模型怎么知道 token 之间的相对距离"这件事上。RoPE(Rotary Position Embedding)是当前几乎所有主流开源 LLM(Llama/Qwen/Mistral/DeepSeek……)的默认位置编码方案,而"一个只在 4k/8k 长度上训练过的 RoPE 模型,不重新训练就能处理 32k/128k 输入"这件事,几乎是长上下文面试的第一道题——而且是一道"背得出名字 vs 真的懂原理"区分度极高的题:PI/NTK/YaRN 三个名字很多人都会背,能讲清楚"各自解决了什么、又留下了什么问题"的人不多。

**本文定位:** 和 `torch-deep-dive` 系列一样,这里只回顾对本文后续 5 个知识点**必需**的 RoPE 机制(旋转矩阵怎么把"位置"编码进 Q/K),不重复 [torch-deep-dive/01-tensor-memory-model.md](../torch-deep-dive/01-tensor-memory-model.md) 已经讲过的 tensor 内存/stride 机制——如果还没看过那一篇,建议先看完再回来看这里的"可运行例子",会更容易看懂 `x[..., 0::2]` 这种切片在内存里到底发生了什么。本系列和 `learning/long-context/` 的关系见 [00-roadmap.md](00-roadmap.md) 的差异化声明,这里不重复。

本文所有代码例子已在仓库 `.venv`(Windows 原生,Python 3.13,torch 2.11.0+cu128,transformers 5.10.2,全程纯 CPU)下实际跑通验证,凡是给出的数字(cos 值、inv_freq 比值、attn_scale……)都是真实跑出来的,不是从论文或文档转抄的近似值。因为 `learning/long-context/src/` 不是一个安装好的 Python 包,例子里统一用 `sys.path.insert(0, "learning/long-context/src")` 从仓库根目录导入——这和仓库自己的 `learning/long-context/src/tests/test_rope_extrapolation.py` 用的是同一种写法,不是本文自己发明的技巧。

**本篇统一结构(与 00-roadmap.md 的知识点结构模板完全一致):**
1. 签名/是什么
2. 一句话
3. **底层机制/为什么这样设计** —— 不停在"怎么用",讲到"为什么必须是这样"
4. AI 研究/工程场景
5. 可运行例子(带 `assert`,真的在仓库 `.venv` 里跑过)
6. **面试怎么问 + 追问链** —— 面试官大概率怎么问,追问会往哪个方向深挖
7. 常见坑

---

## 1. `inv_freq` / `build_cos_sin` / `apply_rope_interleaved` —— Vanilla RoPE 回顾

**是什么:**
```python
# learning/long-context/src/common.py
inv_freq(dim, base=10000.0, device=None, dtype=torch.float32)   # 每个"频率通道"转多快,shape (dim/2,)
build_cos_sin(t, inv_freq_, device=None, dtype=torch.float32)   # 每个位置 x 每个频率通道 的旋转角度,算出 cos/sin
apply_rope_interleaved(x, cos, sin)                              # 真正把 Q/K 向量按这个角度"转"一下
```

**一句话:** RoPE 不是"给每个位置分配一个独立的向量加到 Q/K 上"(那是绝对位置编码的思路),而是"把 Q/K 的每一对相邻维度看成一个 2D 平面上的点,按位置对应的角度把这个点**旋转**一下"——这个设计直接决定了后面 PI/NTK/YaRN 全部在改的东西只有两个自由度:**转多快**(inv_freq/base)和**转到哪**(position)。

**在讲"底层机制"之前:为什么"位置"信息需要额外注入,不能指望 attention 自己感知顺序**

上面这句话里"给 Q/K 注入位置信息"这件事,前提是 Q/K 本身**不会自动携带位置信息**——这不是随口一说,而是 self-attention 这个计算本身的性质决定的。如果还不知道 Q/K/V 到底是什么、attention 为什么要这样算,先看 [torch-deep-dive/04-layers-math-and-backward.md](../torch-deep-dive/04-layers-math-and-backward.md) 第 8 节"在讲拆分之前"那一段从零建立的内容——这里直接假设你已经知道 attention 在算什么,只讲"顺序"这一个维度。

self-attention 的计算——`Q·K` 算相似度、`softmax` 转成权重、再对 `V` 做加权平均——从头到尾只用到每个 token 的**内容**(具体的向量数值),没有任何一步用到"这个 token 排第几个"这个信息。后果是:把一句话的 token **顺序打乱**,只要每个 token 的内容本身不变,attention 给每个 token 算出来的输出也只是跟着**同步打乱**,每个 token 单独拿到的数值完全不变——这个性质叫**置换等变(permutation equivariant)**,可以现场验证:

```python
import torch, torch.nn.functional as F, math

torch.manual_seed(0)

def attn(x):                                   # 最朴素的 self-attention(不含位置编码,Q=K=V=x 这个特例)
    d = x.shape[-1]
    scores = (x @ x.transpose(-2, -1)) / math.sqrt(d)
    weights = F.softmax(scores, dim=-1)
    return weights @ x

x = torch.randn(4, 6)               # 4 个 token,每个 6 维,计算过程没有用到任何位置信息
out = attn(x)

x2 = x.clone()
x2[[0, 2]] = x2[[2, 0]]              # 只交换 token0 和 token2 的顺序,token1/token3 位置不动
out2 = attn(x2)

# 顺序没动过的 token1,它的 attention 输出应该一模一样(bit-exact,不是"差不多")——
# 因为 attn() 的计算里根本没有用到"谁排第几",只要"谁和谁在一起"这个集合不变,每个 token 算出来的内容就不变
assert torch.equal(out2[1], out[1])                          # 实测 diff 精确为 0.0

# 更一般的性质:整体打乱顺序,attention 的输出也只是跟着同步打乱(不是变成别的东西)
perm = torch.tensor([2, 0, 3, 1])
out_shuffled = attn(x[perm])
assert torch.allclose(out_shuffled, out[perm], atol=1e-6)    # 实测 max diff ≈ 2.4e-7(float32 噪声)
```

这正是问题所在:"猫追狗"和"狗追猫"用到的 token 集合相同、顺序相反,含义完全不同,但如果 attention 的计算本身对顺序不敏感,模型没法**只靠 attention 这一步**分辨这两句话的差异——必须有一种机制把"谁排第几"**额外**注入 Q/K,attention 的相似度计算才有机会感知到顺序。位置编码(不管是绝对位置编码,还是 RoPE 这种相对位置编码)存在的全部理由,就是补上这个缺口——下面的"底层机制"讲的正是 RoPE 具体怎么补。

**底层机制/为什么这样设计:**

先看 `inv_freq`:`1/base^(2i/dim)`,`i` 从 0 到 `dim/2-1`。这是一个随 `i` 指数衰减的数列——`i=0` 那个"频率通道"转得最快(每挪 1 个位置转 1 弧度),`i` 越大转得越慢。`build_cos_sin` 把"位置 `t`"和"每个通道的转速 `inv_freq`"相乘得到角度 `angle = pos * inv_freq`,再取 cos/sin。`apply_rope_interleaved` 则是标准的 2D 旋转矩阵作用在每一对维度 `(x[2k], x[2k+1])` 上:

```
[rot_2k  ]   [cos  -sin] [x_2k  ]
[rot_2k+1] = [sin   cos] [x_2k+1]
```

**用图示把"旋转"这件事本身画出来**(以其中一对维度 `(x_2k, x_2k+1)` 为例,把它看成 2D 平面上的一个点/向量 `v`):

旋转前:`v` 画在 `x_2k` 轴正方向,坐标 `(r, 0)`,方便量角度:
```
        x_2k+1
          │
          │
          │
          │
          └───────────●──────────  x_2k
        (0,0)      v = (r, 0)
```

旋转后:绕原点转了角度 `θ = pos·inv_freq[k]`,坐标变成 `(r·cosθ, r·sinθ)`:
```
        x_2k+1
          │      v' = (r·cosθ, r·sinθ)
          │     ╱
          │   ╱
          │ ╱
          │╱  θ
          └──────────────────  x_2k
        (0,0)
```

`|v| = |v'| = r`——长度完全不变,只有方向转了角度 `θ`(对应第 1 节可运行例子里 `assert x.norm(dim=-1)==x_rot.norm(dim=-1)` 验证的正是这件事)。

`pos` 越大,`θ = pos·inv_freq[k]` 转得越多;`inv_freq[k]` 越大(`k` 越小,越高频),同样 `pos` 下也转得越多——这两条自由度就是上面"一句话"提到的"转多快"和"转到哪"。

这个设计最核心的性质是:**Q 在位置 i、K 在位置 j 的点积,只取决于 i-j,不取决于 i、j 各自的绝对值**——因为两个旋转矩阵的乘积满足 `R(i)^T R(j) = R(j-i)`,旋转矩阵天然满足这种"角度可加性"。这就是为什么 RoPE 属于"相对位置编码":模型学到的不是"第 5 个 token 该怎么被 attend",而是"隔 3 个 token 的关系该怎么被 attend"。后面 PI/NTK/YaRN 所有的修改,本质上都是在动"`pos` 和 `inv_freq` 这两个自由度怎么映射成角度",不会破坏"点积只依赖相对位置"这条根本性质。

**这个恒等式不是凭空断言,推导只需要两条旋转矩阵的基本性质:**

**性质①(正交性):** 2D 旋转矩阵 `R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]]` 是正交矩阵(`R(θ) @ R(θ)^T = I`),而正交矩阵的转置就是它的逆。"旋转 θ 角度"这个操作的逆操作,直觉上就是"反过来转 -θ 角度转回原地"——代数上确实如此:`R(θ)^T = R(-θ)`。

**性质②(角度可加性):** 先转 θ1、再转 θ2,和一次性转 `θ1+θ2` 是同一个操作:`R(θ1) @ R(θ2) = R(θ1+θ2)`(旋转的复合就是角度相加,这是"旋转"这个几何操作最基本的直觉)。

**组合这两条性质:** `R(i)^T @ R(j) = R(-i) @ R(j) = R(j-i)`——第一步用性质①把转置换成"转 -i",第二步用性质②把两次旋转的复合合并成一次转 `j+(-i)=j-i`。这就是"Q 在位置 i、K 在位置 j 的点积只取决于 i-j"这句话背后完整的代数依据,不是照抄论文的断言:

```python
import torch, math

def R(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]])

theta1, theta2 = 0.7, 1.3

# 性质①:正交矩阵,转置 = 逆 = "转回去"(旋转 -θ)
R1 = R(theta1)
assert torch.allclose(R1 @ R1.T, torch.eye(2), atol=1e-6)      # R R^T = I
assert torch.allclose(R1.T, R(-theta1), atol=1e-6)              # 实测 R(θ)^T 和 R(-θ) 最大误差 = 0.0

# 性质②:旋转的复合 = 角度相加
R2 = R(theta2)
assert torch.allclose(R1 @ R2, R(theta1 + theta2), atol=1e-6)   # 实测最大误差 ≈ 6e-8(float32 噪声)

# 组合①②,验证 R(i)^T @ R(j) == R(j-i) 对具体的"位置" i=5, j=12 成立
i, j = 5, 12
Ri, Rj = R(float(i)), R(float(j))
lhs = Ri.T @ Rj
rhs = R(float(j - i))
assert torch.allclose(lhs, rhs, atol=1e-6)   # 实测两个 2x2 矩阵逐元素精确相等,最大误差 = 0.0
```
实测:`R(θ)^T` 和 `R(-θ)` 逐元素精确相等(最大误差 `0.0`);`R(θ1)@R(θ2)` 和 `R(θ1+θ2)` 最大误差 `≈5.96e-08`(float32 舍入噪声);组合出的 `R(i)^T @ R(j)` 和 `R(j-i)` 两个矩阵逐元素精确相等,最大误差同样是 `0.0`。(真实 RoPE 里 `θ` 不是位置本身,而是 `pos * inv_freq[k]`——换成这个真实角度重新跑一遍上面同样三步验证,结论不变,这里不重复展开。)

还有一个和"长上下文"这个主题直接相关、容易被忽略的细节:`build_cos_sin` 里 `pos = torch.arange(t, ..., dtype=torch.float32)` 强制用 float32 算位置和角度,即使模型主体在用 bf16/fp16 跑,也只在最后 `.to(dtype)` 才转换精度。这不是随手写的,而是因为**大整数在低精度浮点数里根本存不下**——见下面可运行例子,这是训练精度为 fp16/bf16 的长上下文模型里必须知道的一个坑。

**AI 研究/工程场景:** 这三个函数是本文档后面 5 个知识点的公共基座——PI 只改 `pos`(`build_cos_sin` 里的 `pos`),NTK 只改 `inv_freq`(通过改 `base`),YaRN 把两者结合起来按维度分段处理,M-RoPE 把 `apply_rope_interleaved` 分别用在切出来的 3 组维度上。吃透这三个函数,后面 5 节基本都是"改一两个参数"级别的增量。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/long-context/src")
import torch
from common import inv_freq, build_cos_sin, apply_rope_interleaved

f = inv_freq(dim=8, base=10000.0)
assert f.shape == (4,)
assert torch.allclose(f, torch.tensor([1.0, 0.1, 0.01, 0.001]), atol=1e-6)   # 实测:10000^(-2i/8) 恰好是整数次幂,数字很干净

# 位置 0 处旋转角恒为 0(任何频率转 0 步都还在原地)
cos0, sin0 = build_cos_sin(t=1, inv_freq_=f)
assert torch.equal(cos0, torch.ones(1, 4))
assert torch.equal(sin0, torch.zeros(1, 4))

# 旋转矩阵保范数:RoPE 只转向量方向,不改变长度
torch.manual_seed(0)
x = torch.randn(1, 5, 8)
cos, sin = build_cos_sin(t=5, inv_freq_=f)
x_rot = apply_rope_interleaved(x, cos, sin)
assert torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), atol=1e-5)

# 核心性质:只要相对位置不变,点积就不变(和绝对位置无关,只和 i-j 有关)
def rope_at(x, pos, inv_freq_):
    cos_p, sin_p = build_cos_sin(t=pos + 1, inv_freq_=inv_freq_)
    return apply_rope_interleaved(x, cos_p[pos:pos+1].unsqueeze(0), sin_p[pos:pos+1].unsqueeze(0))

q, k = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
dot_a = (rope_at(q, 2, f) * rope_at(k, 5, f)).sum()     # 相对距离 -3
dot_b = (rope_at(q, 12, f) * rope_at(k, 15, f)).sum()   # 同样相对距离 -3,只是整体平移了 10
assert torch.allclose(dot_a, dot_b, atol=1e-6)          # 实测 diff ≈ 6e-8(float32 噪声量级)

# 为什么 build_cos_sin 内部强制用 float32 算 position:bf16/fp16 存不下长上下文的大整数
pos_bf16 = torch.arange(100_000, 100_005, dtype=torch.bfloat16)
assert len(set(pos_bf16.tolist())) == 1   # 实测:100000~100004 这 5 个不同的位置,在 bf16 里全部塌缩成同一个数 99840.0
pos_fp16 = torch.arange(70_000, 70_004, dtype=torch.float16)
assert torch.isinf(pos_fp16).all()         # 实测:fp16 在 70000 这个量级直接溢出成 inf(fp16 最大可表示值是 65504)
```

**面试怎么问 + 追问链:**
- **Q:** "RoPE 和绝对位置编码(learned/sinusoidal)相比,核心区别是什么?为什么现在主流 LLM 基本都用 RoPE?" —— 期望答出"RoPE 编码的是相对位置,attention score 只依赖 i-j"。
- **追问 1:** "能不能证明一下,为什么 RoPE 的 attention score 只依赖相对位置?" —— 期望讲出旋转矩阵的角度可加性 `R(i)^T R(j) = R(j-i)`,最好能现场验证(上面的可运行例子)。
- **追问 2(区分度很高):** "如果 inference 时用到的位置超过了训练时见过的最大长度,RoPE 的 cos/sin 计算本身会出错吗(NaN/inf)?" —— 很多人会直接说"会出问题",精确答案是"数值上完全不会出错(cos/sin 对任意实数都有定义,可以现场 assert isfinite),真正的问题是这些角度对应的 (cos,sin) 组合模型训练时从没见过,是分布外泛化问题,不是数值错误"——这是过渡到 PI/NTK/YaRN 的关键铺垫问题。
- **追问 3(容易漏答):** "训练用 bf16 混合精度的长上下文模型,计算 RoPE 的 position/angle 时能不能也用 bf16?" —— 期望能说出"不能,大整数在 bf16/fp16 里会直接塌缩或溢出",最好能举出具体数字(bf16 在 10 万量级已经没有整数分辨率)。

**常见坑:** 把这里的 `apply_rope_interleaved`(相邻两维一组:`x[0::2]`/`x[1::2]`)当成唯一的 RoPE 写法,和 HuggingFace `transformers` 里 Llama/GPT-NeoX 风格的 `rotate_half`(前一半/后一半分组:`x[:d/2]`/`x[d/2:]`)搞混。实测把 `common.py` 这里的写法按"偶数下标、奇数下标"重新排列维度后,结果和 `transformers.models.llama.modeling_llama.rotate_half` 完全等价(数值 diff=0)——两者是同一个数学操作的两种内存布局,不是两种算法。但如果自己实现权重转换/量化工具时把两种布局的代码混用,会得到"形状对但数值全错"的旋转结果,这是一个真实存在、很难一眼看出来的 bug 来源。

---

## 2. `pi_cos_sin()` —— Position Interpolation:压缩位置而不是压缩频率

**是什么:**
```python
# learning/long-context/src/rope_pi.py
pi_cos_sin(t, dim, base=10000.0, scale_factor=4.0, device=None, dtype=torch.float32)
```
和 `build_cos_sin` 签名基本一致,只多了一个 `scale_factor`。

**一句话:** PI(Position Interpolation,Meta 2023)不改 `inv_freq`,只把传进去的位置先除以 `scale_factor`——用"位置除以 4"把一个本来会跑到 32k 的位置,硬压回模型训练时熟悉的 8k 以内的范围。

**底层机制/为什么这样设计:**

先说"最笨的想法":什么都不改,直接把训练时 max_position=8k 的模型喂进 32k 长度的输入,`build_cos_sin` 照常算出所有位置的 cos/sin——上一节已经验证过,这一步数值上不会出任何错误。问题出在**训练时模型的权重从来没被优化去处理这些角度组合**,尤其是低频维度:一个训练时最多只转过一小段角度的维度,现在要处理转了两圈多的角度,attention 的行为直接失控。这在真实训练好的模型上通常表现为困惑度(perplexity)爆炸性升高——不过这是训练好的模型才能观察到的现象,这几个孤立的 RoPE 函数没法演示"模型崩了",能验证的是更精确的问题定位:数值不出错,但落在分布外(具体验证见下一段)。

PI 的方案:**不让位置真的跑出训练区间**。做法极简单——`pos = arange(t) / scale_factor`。position=32(scale_factor=4)算出来的角度,和未缩放时 position=8 的角度完全一样,所以只要 `scale_factor` 取"目标长度 / 原训练长度",所有位置的角度就都被摁回了训练时见过的范围内。

代价是**分辨率被压缩了**。原来相邻两个 token(position 差 1)在每个维度上的角度差是 `inv_freq[i]`;PI 之后这个角度差变成 `inv_freq[i] / scale_factor`——所有维度统一按同一个比例被压缩,高频维度(本来靠"转得快"才能精细分辨临近 token 的相对位置)现在被压缩到和低频维度一样的力度,细节丢失最明显。这也是为什么 PI 论文强调需要**少量微调**才能让模型重新适应压缩后的分布,不是开箱即用——以及为什么下一节 NTK-aware 会说"PI 对所有维度一刀切地压缩不是好主意"。

**AI 研究/工程场景:** PI 是"用最简单的机制换取最大稳定性"的方案——不改 base、不引入分段逻辑,只有一个标量参数,微调成本低、行为容易预测,是长上下文扩展文献里事实上的 baseline,后面的方法基本都会拿 PI 当对比基准。

**可运行例子(核心代数恒等式,已现场验证):**
```python
import sys
sys.path.insert(0, "learning/long-context/src")
import torch
from common import inv_freq, build_cos_sin
from rope_pi import pi_cos_sin

# 核心恒等式:PI 缩放后 position=4(scale_factor=4)和未缩放的 position=1,angle 完全相同(4/4=1)
cos_pi, sin_pi = pi_cos_sin(t=8, dim=16, scale_factor=4.0)
f16 = inv_freq(16, 10000.0)
cos_base, sin_base = build_cos_sin(t=2, inv_freq_=f16)

assert cos_pi[4, 0].item() == cos_base[1, 0].item()   # 实测:两边都是 0.5403023362159729,bit-exact 相等,diff=0.0
assert sin_pi[4, 0].item() == sin_base[1, 0].item()    # 实测:两边都是 0.8414709568023682
assert cos_pi[4, 3].item() == cos_base[1, 3].item()    # 不止最高频那一维,任意频率通道都成立(实测 0.999500036239624)
assert cos_pi[0, 0].item() == 1.0                        # position 0 处两者当然也一致(都还没转)
```

**面试怎么问 + 追问链:**
- **Q:** "已经有一个 max_position=4096 训练好的模型,想让它支持 16k,最直接的想法是什么?PI 具体怎么做?" —— 期望答出"把位置压缩 4 倍,让所有位置都落在训练时见过的范围内"。
- **追问 1:** "PI 压缩位置会带来什么代价?" —— 期望答"分辨率下降,尤其高频维度损失局部细节,相邻 token 之间变得更难分辨"。
- **追问 2(区分度很高):** "PI 需要重新微调吗?不微调直接用会怎样?" —— 期望知道 PI 论文本身强调需要少量微调,不微调时模型看到的 cos/sin 分布和训练时不一样(虽然数值范围"压回训练区间"了,但同一位置对应的具体 cos/sin 组合变了),效果会明显下降。
- **追问 3(过渡追问):** "PI 对所有频率通道用同一个 `scale_factor`,这样做有什么隐藏问题?" —— 期望答出"没有区分高频/低频的不同需求,是一刀切方案",这是下一节 NTK-aware 的引入动机。

**常见坑:** 以为"压缩位置"和"压缩频率"是两回事——`pi_cos_sin` 的实现是改 `pos`(除以 `scale_factor`),但从最终 `angle = pos * inv_freq` 的结果看,这**等价于**把 `inv_freq` 整体除以 `scale_factor`。第 4 节 YaRN 的源码里 `pi_freq = inv_freq / factor` 就是直接用这种等价形式实现 PI 的效果,而不是改 `pos`。这两种写法数学上完全等价,但读代码时如果不知道这层等价关系,会误以为 YaRN 源码里没有实现"PI 那部分"。

---

## 3. `ntk_cos_sin()` —— NTK-aware RoPE:缩放 base 而不是位置

**是什么:**
```python
# learning/long-context/src/rope_ntk.py
ntk_cos_sin(t, dim, base=10000.0, scale_factor=4.0, device=None, dtype=torch.float32)
```

**一句话:** NTK-aware(2023.06,LocalLLaMA 社区提出)不动位置,只把 `base` 调大——`new_base = base * scale_factor^(dim/(dim-2))`——通过重新生成一整套 `inv_freq`,让高频维度几乎不受影响、低频维度被大幅压缩,一步到位实现"高频维度保留精度、低频维度让出空间"。

**底层机制/为什么这样设计:**

上一节的问题是"PI 对所有维度一刀切压缩"。NTK-aware 的洞察是:`inv_freq[i] = 1/base^(2i/dim)`,当 `i=0` 时指数是 0,`base^0` 恒等于 1——**不管 base 怎么调,最高频那个维度的 inv_freq 永远是 1,base 的变化在这里被指数 0 完全抵消**。而 `i` 越大,指数越大,base 的变化被放大得越厉害,所以同一次"调大 base"的操作,对高频维度几乎没有影响,对低频维度却有指数级的压缩效果。这是一个很巧妙的"一个标量参数,自动实现按维度区别对待"的技巧,不需要像 YaRN 那样手写分段函数。

代价是:NTK-aware 是一条连续的、隐式的权衡曲线,没法精确控制"从哪个维度开始压缩、压缩多猛"——曲线形状完全由 `base`、`dim`、`scale_factor` 三个数联立决定,不像后面 YaRN 那样可以显式设置分段边界。而且和 PI 一样,效果最好依然需要配合微调,不过社区实践里很多人直接零样本(不微调)用 NTK-aware 也能跑出不错的效果,比直接用 PI 更友好一些。

**AI 研究/工程场景:** NTK-aware 是社区(bloc97 等,r/LocalLLaMA)发现的技巧,严格来说和"神经正切核"(Neural Tangent Kernel)理论没有直接的数学推导关系,更多是借用了"高频信息应该被区别对待"这个直觉命名。它有一个重要变体叫 **Dynamic NTK**——推理时按当前实际序列长度实时计算 `scale_factor`(而不是固定死一个值),序列没超过原始训练长度时完全不缩放,超过多少就按比例缩放多少,是不少推理框架(比如 `rope_scaling: {"type": "dynamic"}`)的默认选项之一。

**可运行例子:**
```python
import torch
import sys
sys.path.insert(0, "learning/long-context/src")
from common import inv_freq as vanilla_inv_freq

dim, base, scale_factor = 16, 10000.0, 4.0
f_vanilla = vanilla_inv_freq(dim, base)

# 复刻 ntk_cos_sin 内部同款公式,直接检查 inv_freq 本身(cos/sin 只是它的下游)
new_base = base * (scale_factor ** (dim / max(dim - 2, 1)))
f_ntk = 1.0 / (new_base ** (torch.arange(0, dim, 2).float() / dim))

assert new_base == 48760.54616817902             # 实测:10000 * 4^(16/14)
assert f_vanilla[0].item() == f_ntk[0].item()      # 最高频维度(index 0):base^0=1,new_base 被完全抵消,两者精确相等(都是 1.0)
assert abs(f_ntk[-1].item() / f_vanilla[-1].item() - 1 / scale_factor) < 1e-6
# 实测最低频维度(index -1)压缩比例精确等于 0.25 == 1/scale_factor —— 这不是巧合:
# new_base 公式里的指数 dim/(dim-2) 就是专门设计出来的,在最后一个频率通道上
# (base/new_base)^((dim-2)/dim) 恰好等于 1/scale_factor,让 NTK 在"最低频那一维"上和 PI 的压缩力度对齐

ratios = f_ntk / f_vanilla
assert ratios[0].item() == 1.0
assert ratios[-1].item() == 0.25
assert (ratios[1:] < ratios[:-1]).all()   # 压缩比例随频率降低单调递减:频率越低,压得越狠
```

**面试怎么问 + 追问链:**
- **Q:** "NTK-aware RoPE 改的是 PI 里的哪个变量?和 PI 本质区别是什么?" —— 期望答"改 base(进而改变整套 inv_freq),不碰 position"。
- **追问 1:** "为什么调大 base 之后,高频维度几乎不变,低频维度却被大幅压缩?" —— 期望能现场推出 `inv_freq[i]=1/base^(2i/dim)`,`i=0` 时 `base^0=1` 恒成立,base 的影响被指数 0 抵消掉了。
- **追问 2(区分度很高):** "'NTK-aware'这个名字里的 NTK,和深度学习理论里的 Neural Tangent Kernel 是一回事吗?" —— 期望知道这更多是命名上借用的直觉("不同频率的信息应该区别对待"),不是从 NTK 理论严格推导出来的公式;能诚实说出"这是社区起的名字,不是我能现场证明的数学定理",比硬编一个不存在的推导更加分。
- **追问 3(开放):** "Dynamic NTK 和这里的静态 NTK-aware 有什么区别,你会在什么场景选哪个?" —— 期望知道 Dynamic NTK 按实际序列长度实时算 scale_factor,短序列时完全不缩放(不牺牲短文本效果),这是很多推理框架的默认行为。

**常见坑:** 以为 NTK-aware "完全不需要微调"是绝对结论——实测零样本效果通常好于 PI,但要达到最佳长上下文效果依然建议配合至少少量微调。另外容易和下一节 YaRN 的"NTK-by-parts"混淆——YaRN 内部虽然也借用了"改 base"的思路,但它真正生效的分段混合机制和这里的连续公式是两套不同的处理方式(见下一节)。

---

## 4. `yarn_cos_sin()` + `_yarn_ramp()` —— YaRN:NTK-by-parts 分段 ramp + attention temperature

**是什么:**
```python
# learning/long-context/src/rope_yarn.py
_yarn_ramp(low, high, dim)   # 分段权重:一条从 1 降到 0 的斜坡
yarn_cos_sin(t, dim, base=10000.0, factor=4.0, original_max_pos=2048, device=None, dtype=torch.float32)
# 返回 (cos, sin, attn_scale) —— 比前两种方法多返回一个 attn_scale
```

**一句话:** YaRN(Peng et al. 2023.09,当前 Llama-3.1/Qwen-2.5/DeepSeek-V3 都在用同类思路)做了两件独立的事:① 按"频率通道"分段决定每个维度该用 PI 那套压缩,还是干脆保持原样不动;② 额外校准一个和 RoPE 频率无关的 attention temperature,专门补偿长序列下 softmax 过于尖锐的问题。

**底层机制/为什么这样设计:**

先看分段。`_yarn_ramp(low=0.5, high=0.9, dim=half_d)` 生成一条从 1 平滑降到 0 的斜坡:归一化维度下标(`linspace(0,1,half_d)`)在 `low` 以下取 1,`high` 以上取 0,中间线性过渡。`yarn_cos_sin` 里真正生效的混合公式是:

```python
pi_freq = inv_freq / factor                          # PI 那一套(除以 factor)
inv_freq_yarn = mask * inv_freq + (1 - mask) * pi_freq
```

`mask=1`(低维度下标 = 高频)时用**原始未缩放的 `inv_freq`**;`mask=0`(高维度下标 = 低频)时切换成**除以 factor 的 `pi_freq`**。这和真实 `transformers` 生产代码(下一节详细对比)的命名完全对应:`inv_freq_extrapolation`(不缩放,字面意思就是"外推")用在高频维度,`inv_freq_interpolation`(即 PI,字面意思是"插值")用在低频维度——**是高频维度做外推、低频维度做插值,不是反过来**。原因是周期性:高频维度波长短,训练时哪怕只有 4096 个位置,也早就把这个维度的角度(mod 2π)转了个遍,继续外推到更长的位置,角度值虽然对应"新的绝对位置",落进的还是训练时见过的那个稠密角度分布里,所以外推对高频维度是安全的,不缩放才能保住局部关系需要的精细分辨率;低频维度波长长,训练时可能连四分之一圈都没转完(下面例子会验证一个具体数字:64 维、10000 base 下,最低频通道在 position=4096 时只转了约 8.69% 的一圈),继续外推会把角度推到训练时从没见过的全新区间,必须像 PI 一样压缩摁回去,代价是分辨率下降——但长程关系本来也不需要那么精细的分辨率,这笔"用分辨率换安全"的交易划算。

再看 attention temperature。这是一个和"改 inv_freq"完全独立的第二层机制:序列变长之后,`q·k` 的求和项数变多,softmax 输入的数值分布也会跟着变化,容易让 attention 过度集中在少数几个 token 上。`attn_scale` 就是给 attention score 再乘一个和 `factor` 相关的标量做补偿。这一项单独存在的意义是:哪怕分段 ramp 把频率处理得再精细,如果不校准这个整体温度,长序列下的 attention 分布依然可能不健康——两个机制解决的是两类不同的问题,不能只做一个。

(补充说明一个和分段边界有关的简化:本教学代码把 ramp 的 `low=0.5, high=0.9` 写死成两个常数,而真实 YaRN 论文和 `transformers` 生产实现是用 `beta_fast=32, beta_slow=1` 两个"目标旋转圈数"通过 `find_correction_dim` 公式反推出对应的维度下标区间,不是一个固定比例——这不是本文验证重点,下一节会讲一个更核心、已经验证过的差异。)

**AI 研究/工程场景:** 据 `learning/long-context/lectures/04-yarn.md` 的课程材料,Llama-3.1(`rope_scaling` type=`"llama3"`,概念上是 YaRN 风格的简化变体,不是严格意义上的原版 YaRN)、Qwen-2.5、DeepSeek-V3(factor=40,把 4096 一路推到 128k)都在用 YaRN 或其变体——这是目前长上下文开源模型事实上的标配方案,也是本仓库自己 capstone(`capstone_yarn_llama32.py`:Llama-3.2-1B + YaRN scale=4 + LoRA → 32k)选择的方案。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/long-context/src")
import torch
from rope_yarn import yarn_cos_sin, _yarn_ramp
from common import inv_freq as vanilla_inv_freq

dim, half_d = 16, 8
mask = _yarn_ramp(low=0.5, high=0.9, dim=half_d)
assert mask[0].item() == 1.0 and mask[-1].item() == 0.0   # 实测:两端分别精确是 1 和 0
assert (mask[1:] <= mask[:-1]).all()                        # 单调不增:频率越低,越偏向 PI

# 验证"高频用原样、低频用 PI"这个混合公式,和 yarn_cos_sin 真实输出精确对应
base, factor = 10000.0, 4.0
f = vanilla_inv_freq(dim, base)
pi_freq = f / factor
inv_freq_yarn_manual = mask * f + (1 - mask) * pi_freq

cos, sin, scale = yarn_cos_sin(t=2, dim=dim, base=base, factor=factor, original_max_pos=2048)
angle_pos1 = torch.atan2(sin[1], cos[1])                    # position=1 时角度就等于 inv_freq 本身
assert torch.allclose(angle_pos1, inv_freq_yarn_manual, atol=1e-6)   # 实测最大误差 ≈ 6e-8(float32 噪声)
```

**面试怎么问 + 追问链:**
- **Q:** "YaRN 相比纯 NTK-aware,多做了哪两件事?" —— 期望答"显式分段 ramp(而不是隐式连续公式)+ attention temperature 校准"。
- **追问 1:** "为什么要按维度分段处理,而不是像 NTK 那样用一个连续公式一次性解决?" —— 期望答"高频维度承载局部/短程关系不能丢分辨率,低频维度承载长程关系可以牺牲分辨率换安全,分段能显式控制这个取舍边界,连续公式的取舍曲线形状不受你直接控制"。
- **追问 2(区分度很高,容易反着答错):** "分段里,高频维度是做'外推'还是'插值'?低频呢?" —— 正确答案是高频维度做外推(维持原样不缩放)、低频维度做插值(PI 压缩),原因是高频波长短,训练时角度空间已经被转了很多遍,外推不会遇到陌生角度;不少人会凭直觉答反(以为"外推"听起来更适合"要扩展到更长范围"的低频维度)。
- **追问 3(开放):** "如果只做分段 ramp,不做 attention temperature 校准,会有什么后果?" —— 期望答"频率处理对了不代表 softmax 分布健康,这是两个独立生效的机制,长序列下依然可能因为缺乏温度校准导致 attention 过度集中"。

**常见坑:** 读 `rope_yarn.py` 源码时容易被 `inv_freq_ntk`/`new_base` 这两个变量名误导——它们确实被计算出来了(第 28-30 行,用的是第 3 节那套 NTK-aware 公式),但**从未在后续任何地方被引用**。实测把混合公式换成完全不依赖 `inv_freq_ntk` 的手写版本(只用 `mask`、原始 `inv_freq`、`pi_freq` 三者),算出来的结果和 `yarn_cos_sin` 真实输出逐位精确一致(上面例子的最后一个 assert)——这证明了 `inv_freq_ntk`/`new_base` 是死代码,真正生效的分段混合用的是"原样 vs PI"(extrapolation vs interpolation),不是"原样 vs NTK"。这是一个很好的提醒:**变量被算出来不代表它被用到,读代码要跟着数据流走,不能只看变量名**——也是下一节核心方法论的一次热身。

---

## 5. YaRN 教学代码 vs 真实 `transformers` 库的公式差异 ——【全篇重点】怎么核对教学简化和生产实现

**是什么:** 对比两处计算 attention temperature 的代码:
```python
# learning/long-context/src/rope_yarn.py 第 43 行
attn_scale = math.sqrt(1.0 / (0.1 * math.log(factor) + 1.0))

# .venv/Lib/site-packages/transformers/modeling_rope_utils.py 第 405-408 行
# (_compute_yarn_parameters 函数体内部的嵌套函数,不是模块顶层函数)
def get_mscale(scale, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
```

**一句话:** 本仓库教学代码在 `0.1·ln(factor)+1` 外面套了一层 `sqrt`,真实 `transformers==5.10.2` 的生产实现**没有这层 sqrt**——这不是"教学代码写错了"(它有自己的自洽性,下面会讲),而是一次现场示范:**看到教学代码,不能默认它和生产库实现完全一致,必须自己动手核对**。

**底层机制/为什么这样设计(这里的"设计"更多是"怎么验证",不是"公式为什么这样"):**

第一步,按最直接的思路尝试 `from transformers.modeling_rope_utils import get_mscale`——**这一步在仓库 `.venv` 里实测直接报错**:

```
ImportError: cannot import name 'get_mscale' from 'transformers.modeling_rope_utils'
```

原因是 `get_mscale` 不是模块顶层函数,而是定义在 `_compute_yarn_parameters(...)`(`modeling_rope_utils.py` 第 327 行开始)函数体内部的一个嵌套闭包,`get_mscale` 本体在其中第 405-408 行。这本身就是"教学代码不能默认和生产代码位置/结构一致"的第一层证据:连怎么 import 都得自己去 `.venv/Lib/site-packages/transformers/` 里 grep 源码才能确认,不能凭一个看起来合理的路径瞎猜。

于是改用真实的调用路径——构造一个真的 `LlamaConfig`,配上 `rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 2048}`,通过生产代码真正暴露的入口 `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["yarn"]` 走一遍完整流程,拿到的 `attention_factor` 实测精确等于 **1.138629436111989**——正是 `0.1*ln(4)+1`(不带 sqrt)算出来的数字。而 `rope_yarn.py` 的 `attn_scale`(factor=4)实测是 **0.9371493244339143**。两者不是"差一个 sqrt"这么简单的关系——`sqrt(1.138629436111989) = 1.0670658068329193`,取倒数才是 `0.9371493244339143`。

顺带验证了"两个教学文件互为倒数"这件事:`capstone_yarn_llama32.py::attn_temperature(4.0)` 实测是 **1.0670658068329193**,和 `rope_yarn.py` 的 `attn_scale` 相乘精确等于 1.0。用统一记号 `A = 0.1·ln(factor)+1`(生产库直接用的值),三者关系是:

| 来源 | 公式 | factor=4 实测值 |
|---|---|---|
| `transformers` 生产实现(`get_mscale`) | `A` | **1.1386294** |
| `capstone_yarn_llama32.py::attn_temperature` | `sqrt(A)` | 1.0670658 |
| `rope_yarn.py::attn_scale` | `sqrt(1/A)` | 0.9371493 |

生产库直接用 `A`,两份教学代码分别用 `sqrt(A)` 和 `sqrt(1/A)`——教学代码内部是自洽的(互为倒数,大概率是两个文件的作者分别选择了"当除数(temperature)"和"当乘数(scale)"两种不同但等价的表达习惯),但都不是生产库真正在用的数字。

再深一层:这个数字在真实模型里到底是怎么被用上的,值得继续追,拆成三步看:

**第一步,生产库把 `attention_factor` 乘进 cos/sin,Q 和 K 各乘一次。** `transformers` 的 `LlamaRotaryEmbedding`(`modeling_llama.py` 第 87、132-133 行)把 `attention_factor` 直接乘到 `cos`/`sin` 上:`cos = emb.cos() * self.attention_scaling`。而 `apply_rotary_pos_emb`(同文件第 146-168 行)对 Q、K **分别**用这组 cos/sin 做旋转:`q_embed = q*cos + rotate_half(q)*sin`,`k_embed` 同理。这意味着 `attention_factor` 会同时施加在 Q 和 K 上,两者点积 `q_embed·k_embed` 实际被放大的倍数是 `attention_factor²`,不是 `attention_factor` 本身。

**第二步,教学代码只在最终 score 上乘一次。** 本仓库 `learning/long-context/lectures/04-yarn.md` 描述的教学版用法是 `scores = (q@k^T)/sqrt(d) × attn_scale`——只在最终 score 上乘一次,不经过 cos/sin,自然也就没有"平方"这一层效果。

**第三步,现场量化这个差异到底有多大。** 用 `common.py::apply_rope_interleaved` 验证(见下面例子):生产库的实际效果相当于把原始点积放大约 **1.2965 倍**(`1.1386294²`),教学版则是把点积缩小到约 **0.9371 倍**——一个在放大分数,一个在缩小分数,连方向都不一样。

**"放大/缩小分数"具体会怎么影响 softmax,这里展开说清楚,不留一个模糊的形容词:** softmax 是对输入做指数放大再归一化(`softmax(x)_i = exp(x_i) / Σ_j exp(x_j)`)——输入项之间的**差距**越大,指数放大的效果越悬殊,最大的那一项在归一化后占比就越接近 1,其余项占比越接近 0,这就是"softmax **更尖锐**(sharper)"的具体含义:输出的概率分布更集中在少数几项上。反过来,把所有分数按同一个倍数缩小,项与项之间的差距被压缩得更小,归一化后各项概率会更接近均匀分布(几个候选项"旗鼓相当"),这就是"**更平滑**(smoother)"。用一组具体分数现场验证这个方向性:

```python
import torch

scores = torch.tensor([1.0, 2.0, 3.0])
base = torch.softmax(scores, dim=-1)
sharper = torch.softmax(scores * 1.2965, dim=-1)     # 用上面生产库实测的放大倍数
smoother = torch.softmax(scores * 0.9371, dim=-1)     # 用上面教学代码实测的缩小倍数

assert sharper.max().item() > base.max().item() > smoother.max().item()   # 放大后最大概率更突出
assert sharper.min().item() < base.min().item() < smoother.min().item()   # 放大后最小概率更接近0;缩小后更接近均匀
```
实测:`base=[0.0900, 0.2447, 0.6652]`,`sharper=[0.0555, 0.2028, 0.7417]`(最大项从 0.665 涨到 0.742,最小项从 0.090 跌到 0.055——分布更集中),`smoother=[0.0993, 0.2535, 0.6471]`(最大项跌到 0.647,最小项涨到 0.099——分布更接近均匀的 1/3≈0.333)。

回到 YaRN 这个具体场景:生产库这条路径让长序列下的 attention 分布**更尖锐**(更集中关注少数 token),教学代码则让分布**更平滑**(更均匀地看所有 token)——这已经超出"差一个 sqrt"的范畴,是应用方式本身的差异,值得作为"读代码不能只对一个数字,还要对它被用在哪里、怎么影响下游"的示范。

**AI 研究/工程场景:** 如果要手写一个"从 HuggingFace `config.json` 的 `rope_scaling` 字段还原/复现模型 RoPE 行为"的工具(比如模型转换、量化、或者自定义推理引擎适配),这种教学简化和生产实现的偏差是真实会咬人的地方。`_compute_yarn_parameters` 其实还支持一套更复杂的双参数版本(`mscale`/`mscale_all_dim` 分别作为分子分母各算一次 `get_mscale` 再相除,详见 `modeling_rope_utils.py` 第 358-365 行文档字符串),这是社区里一些模型常见的 config 写法(例如 DeepSeek 系列的 rope_scaling 配置据了解会用到这两个字段,这一点是通用背景知识,本文没有在 `.venv` 里加载 DeepSeek 的真实 config 核实,不作为验证结论)。本文验证的 factor=4 例子没传这两个参数,走的是简单分支;真要接手这类 config,必须照着生产源码走一遍,不能默认"论文公式"或者"教学代码"就是唯一实现。

**可运行例子:**
```python
import math
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# 第一步 —— 先证明直觉上的 import 路径本身是错的
try:
    from transformers.modeling_rope_utils import get_mscale
    assert False, "不应该 import 成功"
except ImportError:
    pass   # 实测确认:get_mscale 是嵌套函数,模块顶层拿不到

# 第二步 —— 走生产代码真实入口,拿到 factor=4 时的真实 attention_factor
cfg = LlamaConfig(hidden_size=64, num_attention_heads=4, max_position_embeddings=8192,
                   rope_theta=10000.0,
                   rope_scaling={"rope_type": "yarn", "factor": 4.0,
                                 "original_max_position_embeddings": 2048})
_, attention_factor = ROPE_INIT_FUNCTIONS["yarn"](cfg, device=None)
assert abs(attention_factor - 1.138629436111989) < 1e-9
assert abs(attention_factor - (0.1 * math.log(4.0) + 1.0)) < 1e-12   # 确认就是"无 sqrt"公式

# 第三步 —— 对比教学代码两个版本,验证互为倒数
teaching_scale = math.sqrt(1.0 / (0.1 * math.log(4.0) + 1.0))          # rope_yarn.py
capstone_temp = math.sqrt(0.1 * math.log(4.0) + 1.0)                    # capstone_yarn_llama32.py
assert abs(teaching_scale * capstone_temp - 1.0) < 1e-12
assert abs(teaching_scale - 0.9371493244339143) < 1e-9

# 第四步 —— 验证"应用位置"的差异:生产库把 attention_factor 乘进 cos/sin,Q、K 各乘一次 => 点积被放大 A^2
import sys
sys.path.insert(0, "learning/long-context/src")
import torch
from common import inv_freq, build_cos_sin, apply_rope_interleaved

f = inv_freq(8, 10000.0)
cos, sin = build_cos_sin(t=4, inv_freq_=f)
q, k = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
q_rot, k_rot = apply_rope_interleaved(q, cos, sin), apply_rope_interleaved(k, cos, sin)
q_rot_prod = apply_rope_interleaved(q, attention_factor * cos, attention_factor * sin)   # 生产库做法:cos/sin 各乘一次
k_rot_prod = apply_rope_interleaved(k, attention_factor * cos, attention_factor * sin)

score_base = (q_rot * k_rot).sum(-1)
score_prod = (q_rot_prod * k_rot_prod).sum(-1)
ratio = (score_prod / score_base).mean().item()
assert abs(ratio - attention_factor ** 2) < 1e-4   # 实测约 1.2965 倍,而不是 1.1386 倍
assert ratio > 1.0 and teaching_scale < 1.0          # 两个版本的"方向"是反的:一个放大分数,一个缩小分数
```

**面试怎么问 + 追问链:**
- **Q:** "你怎么知道一份教学/文档代码和它对应的生产库实现是不是完全一致?" —— 期望答"不能只看文档描述或者记忆,要自己 grep 已安装包的源码,现场跑一遍对比"。
- **追问 1:** "如果两边算出来的数字不一样,是不是意味着教学代码是错的?" —— 期望答"不一定——本例里 `rope_yarn.py` 和 `capstone_yarn_llama32.py` 两个数字互为倒数,内部是自洽的简化模型,只是和生产库的具体约定对不上,不代表教学逻辑本身错了"。
- **追问 2(区分度很高):** "这个 YaRN attn_scale 的例子,教学代码和生产代码具体差在哪,你是怎么验证的?" —— 期望候选人讲出完整验证链路:定位安装路径 → 尝试直接 import 发现失败 → grep 源码发现是嵌套函数 → 构造真实 config 走生产入口拿到真值 → 对比公式差了一个 sqrt,而且施加的位置也不同(教学版乘一次在 score 上,生产版通过 cos/sin 分别乘在 Q、K 上等效乘了两次)。只会说"少了个 sqrt"说明只做了公式层面的比对,没有深挖到应用位置这一层。
- **追问 3(开放,考察工程习惯):** "以后你自己写模型相关的教学/文档代码,怎么避免这种不一致误导别人?" —— 期望答出"标注清楚这是简化模型、标注对比过的库版本号、鼓励读者自己动手核对",而不是含糊地说"细节可能有出入"就带过。

**常见坑:** 把"公式对不上"简化理解成"少了一个 sqrt"就算完事——实测这个偏差不只是数值大小上的偏差,连**作用方向**都是反的(生产库放大 attention score 约 1.2965 倍让分布更尖锐,教学代码缩小到约 0.9371 倍让分布更平滑)。只对比最终返回的标量数字,不去看这个数字在下游到底是怎么被乘进 Q、K、还是 score 的,很容易得出"反正都差不多"这种错误的安全感。

---

## 6. `m_rope()` —— 3D-RoPE / M-RoPE:多模态场景下的三轴位置编码

**是什么:**
```python
# learning/long-context/src/rope_3d.py
m_rope(x, positions, base=10000.0)   # x: (..., dim);positions: (..., 3),每个 token 一组 (t, h, w) 坐标
```

**一句话:** 纯文本 token 只有"第几个"这一个位置维度,但图像/视频 token(Qwen2-VL 这类多模态模型)天然有时间/行/列三个坐标——M-RoPE 把 `head_dim` 切成 3 段,每段各自独立地对 t/h/w 中的一个坐标跑一次标准 RoPE,再拼回去。

**底层机制/为什么这样设计:**

把一段视频/图片直接拉平成 1D token 序列、只用一个递增下标做位置编码,会丢掉"哪些 token 在空间上挨着"这个信息(比如图片里横着相邻的两个 patch,拉平后在序列里可能隔得很远)。M-RoPE 的解法建立在第 1 节已经验证过的一个关键性质上:RoPE 的旋转是**逐维度对独立**的(`x[2k], x[2k+1]` 这一对只用自己的角度旋转,不依赖其它维度对),这意味着完全可以把 `head_dim` 切成几段,每段用不同的"位置"值分别调用同一个 RoPE 函数,互不干扰。`m_rope` 正是这么做的——`seg = dim // 6 * 2` 算出每个坐标轴分到的维度数(保证是偶数,因为 RoPE 按 pair 处理),依次对 t/h/w 三个坐标切出对应的维度段各自调用 `apply_rope_interleaved`,剩下 `dim - 3*seg` 除不尽的尾巴维度原样直传(不做任何旋转)。

**AI 研究/工程场景:** 这是 Qwen2-VL/Qwen2.5-VL 这类原生支持图文/视频混合输入的模型采用的方案——图像 patch 的 h/w 坐标反映它在图片网格里的行列位置,视频的 t 坐标反映帧序号,模型不需要额外设计"图像专用位置编码模块",直接复用 RoPE"按维度对独立旋转"这个性质分组编码即可。至于纯文本 token(混在图文序列里)怎么分配 t/h/w 三个坐标,`rope_3d.py` 这份教学代码没有实现这部分逻辑;据 Qwen2-VL 论文的公开描述,做法大致是让文本 token 的 t/h/w 取同一个递增值(退化成普通 1D RoPE)——这一点本文没有在仓库代码里验证到,面试时如果被问到这个细节,比较诚实的答法是说明这是从论文/公开资料了解到的间接认识,不是这份仓库代码验证过的结论。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/long-context/src")
import torch
from rope_3d import m_rope
from common import inv_freq, build_cos_sin, apply_rope_interleaved

torch.manual_seed(0)
dim = 24                                    # 24 能被 6 整除,seg = 24//6*2 = 8,3*seg=24,尾巴长度为 0
x = torch.randn(1, 3, dim)
positions = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]]).view(1, 3, 3)   # 只有 t 变化,h=w=0
y = m_rope(x, positions)
seg = dim // 6 * 2

# h、w 坐标全程为 0 => 旋转角恒为 0 => 对应维度段应该"原样直传"(旋转矩阵在 angle=0 时是单位矩阵)
assert torch.equal(x[..., seg:2*seg], y[..., seg:2*seg])      # h 段不变
assert torch.equal(x[..., 2*seg:3*seg], y[..., 2*seg:3*seg])  # w 段不变

# t 坐标从 0,1,2 变化 => t 段应该真的被旋转了,而且和单独调用标准 RoPE 结果完全一致
f_seg = inv_freq(seg, 10000.0)
cos_t, sin_t = build_cos_sin(t=3, inv_freq_=f_seg)
manual_t = apply_rope_interleaved(x[..., 0:seg], cos_t.unsqueeze(0), sin_t.unsqueeze(0))
assert torch.allclose(y[..., 0:seg], manual_t, atol=1e-6)
assert not torch.equal(x[..., 0:seg], y[..., 0:seg])           # 确实被转了,不是恒等

# dim 不能被 6 整除时,多出来的"尾巴"维度原样直传
dim2 = 20                                    # seg2 = 20//6*2 = 6,3*seg2=18,尾巴 2 维
seg2 = dim2 // 6 * 2
x2 = torch.randn(1, 2, dim2)
y2 = m_rope(x2, torch.tensor([[0, 0, 0], [3, 1, 2]]).view(1, 2, 3))
assert torch.equal(x2[..., 3*seg2:], y2[..., 3*seg2:])         # 尾巴 2 维完全不变
assert y2.shape == x2.shape

# 三个坐标轴互不干扰:只改 h,只有 h 段的输出会变
xa = torch.randn(1, 1, dim)
ya = m_rope(xa, torch.tensor([[5, 2, 7]]).view(1, 1, 3))
yb = m_rope(xa, torch.tensor([[5, 9, 7]]).view(1, 1, 3))        # 只把 h 从 2 改成 9
assert torch.equal(ya[..., 0:seg], yb[..., 0:seg])               # t 段不受影响
assert torch.equal(ya[..., 2*seg:3*seg], yb[..., 2*seg:3*seg])   # w 段不受影响
assert not torch.equal(ya[..., seg:2*seg], yb[..., seg:2*seg])   # h 段确实变了
```

**面试怎么问 + 追问链:**
- **Q:** "多模态模型(比如处理图片/视频)如果直接用标准 1D RoPE 会丢失什么信息?M-RoPE 怎么解决?" —— 期望答"丢失空间/时间的多维结构信息,M-RoPE 把维度切成几段分别编码不同坐标轴"。
- **追问 1:** "为什么可以直接把 head_dim 切成几段分别独立跑 RoPE,而不用担心相互干扰?" —— 期望能连回第 1 节:RoPE 本身就是逐维度对独立旋转的,切分维度天然不会产生耦合,不需要额外设计。
- **追问 2(区分度很高):** "纯文本 token 混在图文序列里时,t/h/w 三个坐标怎么赋值?" —— 这是本仓库代码没有实现的部分,期望候选人能诚实说明"这是我从论文了解到的做法(文本退化成 t=h=w 的 1D RoPE),不是我在这份代码里验证过的",而不是含糊带过装作很确定。
- **追问 3(开放):** "如果要处理的是长视频(而不只是单张图片),这套机制需要做什么调整?" —— 期望能提到 t 轴本身就是为视频时间维度设计的,图片场景下 t 通常是常数或者退化,视频场景下 t 才真正发挥作用,同时会带来"视频 token 数量远超图片,序列长度暴涨"这个新问题,是本系列后面长上下文 attention 架构(Ring Attention/Infini-Attention)要解决的场景。

**常见坑:** 假设 `dim` 一定能被 6 整除、`seg` 一定覆盖所有维度——实测当 `dim` 不能整除时(比如上面 `dim=20` 的例子),`m_rope` 会把多出来的"尾巴"维度原样传递、完全不做位置编码,这部分维度对任何 t/h/w 变化都没有感知能力。真实模型里 `head_dim` 通常会被有意设计成能整除,但如果自己在实验里随手改了 `head_dim` 或者三个轴的切分比例,忘记检查这个"尾巴"有多大,可能会悄悄丢掉一部分维度的位置编码能力而不自知。
