# 05 · 进阶深度追加:4 个真实二面级别的多级追问链案例

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是知识点,不计入"17 个知识点"的统计——它和 `dsa-deep-dive/20-advanced-interview-depth.md` 是同一挂:方法论 + 案例,不是知识点列表。

## 为什么需要这篇追加内容

`01-04` 全部完成并独立验证之后,用户转达了一位有经验从业者的反馈:现有材料没有达到 **2026 年大厂技术二面** 的深度。`dsa-deep-dive/20-advanced-interview-depth.md` 已经用一次真实调研验证过这个判断,并落地了一套格式:真实的追问不是"讲完原理就结束"这一条直线,而是至少沿着 5 条独立轴线展开——**规模递增轴**、**工程约束递增轴(并发/分布式)**、**方案批判迭代轴**、**决策依据追问轴**、**真实性验证轴**。这篇文档把同一套格式套用到 `long-context-deep-dive` 系列上,素材从 `01-04` 已经验证过的知识点里挑选——不是另起炉灶讲新技术,是在已经跑通的代码和数字基础上,往下再追问几层。

**组织原则:**每个案例开头都标注建立在哪个已有知识点之上,给出面试官视角完整还原的多级追问链(带参考答案),再给出至少一段本次现场重新验证过的可运行例子——**不是转述 `01-04` 已经写过的数字,而是在那些数字的基础上继续往前算一步**,证明"批判迭代""真实性验证""规模递增""决策依据"这几件事本身也是可以用代码验证的,不是只能靠嘴说。

**范围声明:**这 4 个案例不是要把 17 个知识点全部重新过一遍追问链——它们演示的是方法论本身。案例 1、2 建立在 [01-rope-scaling-family.md](01-rope-scaling-family.md);案例 3 建立在 [04-long-context-capstone.md](04-long-context-capstone.md);案例 4 建立在 [02-long-context-attention.md](02-long-context-attention.md)。读者应该能把同样的追问节奏自己套用到 03(评测方法论)或者 01/04 里没被选中的其它知识点上练习。

---

## 案例 1:RoPE 外推方案——vanilla 直接外推 → PI → NTK-aware → YaRN(方案批判迭代轴)

建立在 [01 类知识点 1-4](01-rope-scaling-family.md) 之上(vanilla RoPE 回顾、Position Interpolation、NTK-aware、YaRN)。这条链在 01 类里是按"技术介绍"的顺序平铺讲的,每一节内部有"底层机制"解释为什么要这样设计;这里把它重新组织成面试官视角的**方案批判迭代**——面试官不会主动帮你把 PI/NTK/YaRN 的名字和动机报出来,而是给一个具体约束,等你先提出一个方案,再针对这个具体方案指出一个可验证的具体缺陷,逼你换下一个方案。全部 4 轮的"缺陷"在下面都会现场算出具体数字,不是"效果不好"这种空话。

**追问链条完整还原(方案批判迭代,不是深挖同一方案):**

- **面试官给约束:** "有一个模型,RoPE 用 `base=10000`、`head_dim=64`,训练时最长见过 4096 个位置。现在业务方要求处理 16384(4 倍)长度的输入,不能重新训练,你会怎么做?"
- **候选人方案 1:** "RoPE 编码的是相对位置,attention score 只依赖 `i-j`,这是它天然优于绝对位置编码的地方——所以我直接把位置继续往上数到 16384,不用做任何改动,理论上应该能泛化。"
- **面试官指出具体缺陷(不是"效果会变差"这种空话,要求现场证明):** "你说'理论上能泛化',能不能证明一下?具体来说——16384 这个位置算出来的 cos/sin,和模型训练时见过的分布,差多远?给我一个数字,不要给我'应该没问题'。"——期望候选人现场把 64 维、10000 base 下每个频率通道在 4096 位置处"转了几圈"这件事算出来:最高频通道(index 0)在训练边界就已经转了 600 多圈,继续外推只是回到"转过很多遍、很熟悉"的角度区间;但最低频通道在训练边界处连 1/10 圈都没转完,4 倍外推之后依然连一整圈都没转完——**从第一天开始就在训练时从未见过的角度区间里单调递增,永远不会回到熟悉区间**,这是可以精确算出来的分布外证据,不是含糊的"可能有问题"。
- **候选人方案 2(换方案):** "那我把所有位置都先除以 4 再算 RoPE(Position Interpolation)——这样 16384 映射到的角度,和训练时位置 4096 处的角度完全一样,所有位置都被摁回了训练区间内。"
- **面试官指出新方案的代价:** "位置 16384 现在确实不会'转出训练区间'了。但你的方案改的是**所有**频率通道的位置,包括那些根本不需要改的——刚才你自己证明过,最高频通道在训练区间内已经转了几百圈,它的'安全区'比其它通道大得多。给我一个数字,压缩之后,相邻两个 token(距离只差 1)在这个通道上还能分辨到什么程度?"——期望候选人现场算出:未缩放时,最高频通道相邻 token 的角度差精确是 1 弧度;PI 压缩 4 倍之后精确变成 0.25 弧度——**局部、短距离关系最依赖的这个通道,分辨率被无差别地砍到 1/4**,而这个通道其实完全不需要保护,因为它从来就没有分布外问题。
- **候选人方案 3(换方案):** "那我别动位置,换成调大 `base`——这样 `inv_freq[0] = 1/base^0 = 1` 恒成立,最高频通道完全不受影响,而频率越低的通道,指数项越大,被压缩得越狠,不需要我手写'哪些该保护、哪些该压缩'的规则,一个公式自动搞定。"(NTK-aware)
- **面试官指出新方案的代价(区分度很高的深挖):** "你说'高频几乎不受影响',这句话对哪些通道成立?是所有'看起来是高频'的通道,还是只有 index=0 那一个?如果目标扩展倍数从 4 涨到 32(比如从 4k 扩到 128k),这个'几乎不受影响'的范围会怎么变?"——期望候选人现场把不同 `scale_factor` 下每个通道的压缩比例算出来:**只有 index=0 这一个通道被精确保护(比例恒为 1.0)**;`scale_factor=4` 时,压缩比例保持在 95% 以上的通道只有 2/32 个;`scale_factor=32` 时,只剩 index=0 自己(1/32)——所谓"高频自动被保护"其实是一条隐式的连续曲线,保护范围会随着你要扩展的倍数增大而持续缩水,不是一个你能显式控制、锁死的边界。
- **候选人方案 4(换方案,收敛到当前最佳实践):** "那我把'哪些通道该保护、哪些该压缩'显式地写成一条 ramp 函数——按维度下标设一个 `low`/`high` 分界,`low` 以下的通道 100% 保持原样(外推),`high` 以上的通道 100% 走 PI(插值),中间线性过渡;再额外加一个和频率无关的 attention temperature,专门补偿长序列下 softmax 分布过尖锐的问题。"(YaRN)
- **深挖追问(收尾,防止把 YaRN 当成没有代价的终点):** "这条 ramp 的 `low`/`high` 边界是怎么定出来的?你能保证这就是最优边界吗?YaRN 是不是就不需要再微调了?"——期望候选人诚实回答:边界本身是启发式选出来的(仓库教学代码里直接写死 `low=0.5, high=0.9`,真实论文和 `transformers` 生产实现是用 `beta_fast=32, beta_slow=1` 两个"目标旋转圈数"反推出对应边界,依然是启发式,不是理论最优解);YaRN 相比 PI 对训练的依赖更小,但依然不是"零样本永远最优"的银弹,社区实践里长上下文效果最好的版本通常还是配合至少少量微调。**每一代方法都在修上一代一个具体的、可指认的缺陷,不是在做无止境的空洞优化。**

**可运行例子(1/2):量化"vanilla 外推的分布外证据"和"PI 压缩掉的分辨率"——不是描述"应该有问题",是现场算出具体弧度和百分比**

```python
import sys, math
sys.path.insert(0, "learning/long-context/src")
import torch
from common import inv_freq as vanilla_inv_freq, build_cos_sin
from rope_pi import pi_cos_sin

dim, base, train_max_pos = 64, 10000.0, 4096
f = vanilla_inv_freq(dim, base)
half_d = dim // 2
assert half_d == 32

# ---- 候选人方案1(vanilla 直接外推)被要求证明"安全": 量化最低频通道离训练分布有多远 ----
angle_low_train = train_max_pos * f[-1].item()
frac_low_train = angle_low_train / (2 * math.pi)
assert 0.08 < frac_low_train < 0.09          # 实测约 8.69%:训练时这个通道连 1/10 圈都没转完

target_pos = train_max_pos * 4               # 目标外推到 4 倍长度(16384)
frac_low_target = (target_pos * f[-1].item()) / (2 * math.pi)
assert 0.30 < frac_low_target < 0.40          # 实测约 34.77%:依然连一整圈都没转完,自始至终是"没见过的角度"
assert frac_low_target > frac_low_train * 3.9  # 位置乘 4,角度、进而"新颖度"也近似乘 4(线性关系)

# 对照组: 最高频通道在训练范围内早就转了几百圈,外推只是"重新落回熟悉区间",不是新颖角度
n_rot_high_train = (train_max_pos * f[0].item()) / (2 * math.pi)
assert n_rot_high_train > 600                  # 实测约 651.9 圈,充分的周期性覆盖

# ---- 候选人方案2(PI)被指出代价: 用同一个 scale_factor 压缩了本不该压缩的高频通道 ----
scale_factor = 4.0
cos_pi, sin_pi = pi_cos_sin(t=10, dim=dim, base=base, scale_factor=scale_factor)
cos_v, sin_v = build_cos_sin(t=10, inv_freq_=f)

angle_pi = torch.atan2(sin_pi[:, 0], cos_pi[:, 0])      # 最高频通道(index 0)
angle_v = torch.atan2(sin_v[:, 0], cos_v[:, 0])
step_pi = (angle_pi[1] - angle_pi[0]).item()             # 相邻 token 的角度差(PI 压缩后)
step_v = (angle_v[1] - angle_v[0]).item()                 # 相邻 token 的角度差(未缩放)

assert math.isclose(step_v, 1.0, abs_tol=1e-4)             # 未缩放:相邻 token 差 1 弧度(该通道 inv_freq=1)
assert math.isclose(step_pi * scale_factor, step_v, rel_tol=1e-4)  # PI 把这个间距压缩了整整 scale_factor 倍
assert step_pi < step_v / 3.9                                # 分辨率确实被腰斩到 1/4 左右,不是"轻微下降"

print(f"vanilla 4x 外推: 最低频通道训练时只转了{frac_low_train*100:.2f}%圈,外推后仍只有{frac_low_target*100:.2f}%圈(从未回到熟悉区间);"
      f"最高频通道训练时已转{n_rot_high_train:.1f}圈(充分周期覆盖,外推不新颖)。"
      f"PI压缩后,最高频通道相邻token角度差从{step_v:.4f}rad被压到{step_pi:.4f}rad(缩小{step_v/step_pi:.2f}倍)"
      f"——这部分通道本不需要压缩,却被one-size-fits-all地牺牲了分辨率。")
```

实测输出:`vanilla 4x 外推: 最低频通道训练时只转了8.69%圈,外推后仍只有34.77%圈(从未回到熟悉区间);最高频通道训练时已转651.9圈(充分周期覆盖,外推不新颖)。PI压缩后,最高频通道相邻token角度差从1.0000rad被压到0.2500rad(缩小4.00倍)——这部分通道本不需要压缩,却被one-size-fits-all地牺牲了分辨率。`

**可运行例子(2/2):量化 NTK "隐式保护范围随目标倍数收缩" vs YaRN "显式边界与目标倍数解耦"**

```python
import sys
sys.path.insert(0, "learning/long-context/src")
import torch
from common import inv_freq as vanilla_inv_freq
from rope_yarn import _yarn_ramp

dim, base = 64, 10000.0
f_vanilla = vanilla_inv_freq(dim, base)

def ntk_ratios(scale_factor):
    new_base = base * (scale_factor ** (dim / max(dim - 2, 1)))
    f_ntk = 1.0 / (new_base ** (torch.arange(0, dim, 2).float() / dim))
    return f_ntk / f_vanilla

# ---- 候选人方案3(NTK-aware)被追问: "高频保护"是不是一个可控、显式的边界 ----
ratios_4 = ntk_ratios(4.0)
ratios_32 = ntk_ratios(32.0)

assert ratios_4[0].item() == 1.0 and ratios_32[0].item() == 1.0   # 唯一被精确保护的只有 index 0 这一个通道
n_safe_4 = (ratios_4 > 0.95).sum().item()
n_safe_32 = (ratios_32 > 0.95).sum().item()
assert n_safe_4 == 2                       # scale_factor=4 时,仅 2/32 个通道压缩比例大于 95%
assert n_safe_32 == 1                       # scale_factor=32 时,退化到只剩 index 0 自己(1/32)
assert n_safe_32 <= n_safe_4                 # "保护范围"随着目标扩展倍数增大而收缩,不是恒定不变的边界

# ---- 候选人方案4(YaRN)用显式 ramp 把边界钉死,与 scale_factor 完全解耦 ----
half_d = dim // 2
mask = _yarn_ramp(low=0.5, high=0.9, dim=half_d)
n_full_protect = (mask == 1.0).sum().item()
n_full_pi = (mask == 0.0).sum().item()
assert n_full_protect == 16 and n_full_pi == 4    # 16 个通道恒定 100% 保护、4 个通道恒定走 PI,与 factor 无关

# _yarn_ramp 的签名里根本没有 scale_factor 参数 —— 这不是"巧合没受影响",是设计上就没有这个自由度
import inspect
params = list(inspect.signature(_yarn_ramp).parameters)
assert params == ["low", "high", "dim"]              # 确认函数签名里确实不存在 factor/scale_factor

print(f"NTK: scale_factor=4 时有{n_safe_4}/32 个通道压缩<5%; scale_factor=32 时只剩{n_safe_32}/32 个"
      f"——'高频保护范围'是一条隐式曲线,会随目标扩展倍数收缩。"
      f"YaRN: 无论 factor 是多少,恒定 {n_full_protect}/32 个通道完全保护、{n_full_pi}/32 个通道完全走 PI"
      f"(_yarn_ramp 签名 {params} 里根本没有 factor,边界从设计上就和缩放倍数解耦)。")
```

实测输出:`NTK: scale_factor=4 时有2/32 个通道压缩<5%; scale_factor=32 时只剩1/32 个——'高频保护范围'是一条隐式曲线,会随目标扩展倍数收缩。YaRN: 无论 factor 是多少,恒定 16/32 个通道完全保护、4/32 个通道完全走 PI(_yarn_ramp 签名 ['low', 'high', 'dim'] 里根本没有 factor,边界从设计上就和缩放倍数解耦)。`

**常见坑:** 把"vanilla RoPE 外推会失效"简化成"位置太大会报错/出现 NaN"——01 类知识点 1 已经验证过数值上完全不会出错,真正的问题是分布外,必须用"转了几圈"这类具体指标才能量化,不能停留在"应该会崩"的直觉;讲 NTK-aware 时只会背"高频不变、低频压缩"这句结论,答不出"高频不变"这句话精确成立的范围其实只有 index=0 一个通道,而且这个"自动分野"的效果会随目标扩展倍数增大而变差——这正是 YaRN 要显式解决的问题,如果答不出这一层,说明只是记住了三个方法的名字和大致方向,没有真正理解"为什么 YaRN 是在 NTK 基础上的针对性修补,不是平地起高楼的另一套方案"。

---

## 案例 2:YaRN 教学代码 vs 生产库公式——一路追问到"是不是巧合"的真实性验证(真实性验证轴)

建立在 [01 类知识点 5](01-rope-scaling-family.md)(YaRN 教学代码 vs 真实 `transformers` 库的公式差异)之上。01 类已经完整走过一遍验证链路:直觉的 import 路径失败 → grep 源码发现是嵌套函数 → 构造真实 `LlamaConfig` 走生产入口 → 发现教学代码的 `attn_scale` 比生产库的 `attention_factor` 多套了一层 `sqrt`、而且作用方向相反。这个案例不重复那条链路本身,而是**在它已经成立的基础上继续追问**——面试官不会满足于"我发现了一个数字对不上",而是会追问"你怎么知道这不是巧合""这个差异重不重要""是不是只在你凑巧选的这一个参数下才成立"。真实性验证轴的核心就是这种不依不饶。

**追问链条完整还原:**

- **Q(基础,01 类已覆盖):** "这份仓库的 YaRN 教学代码,`attn_scale` 是怎么算出来的?"——期望候选人复述 `rope_yarn.py` 里的公式:`sqrt(1 / (0.1*ln(factor) + 1))`。
- **追问 1:** "你怎么知道这就是 YaRN 论文,或者说 `transformers` 生产库里的标准实现?"——期望答"不能只看文档描述或者凭印象,必须自己动手核对已安装库的真实源码"。
- **追问 2:** "假设你现在真的要去核对——生产库里管这个叫什么函数、定义在哪?你的第一步是什么?"——期望候选人讲出完整路径:先按直觉尝试直接 `import get_mscale`(01 类已验证这一步会失败,因为它是 `_compute_yarn_parameters` 内部的嵌套闭包,模块顶层拿不到),再改用真正暴露的公共入口。
- **追问 3(区分度很高):** "光是找到函数定义就够了吗?你怎么保证你调用的路径,和真实模型推理时内部真正走的调用路径是同一条,而不是你自己脑补出来的一条'看起来差不多'的路径?"——期望答"要走生产代码真正暴露的公共入口(`ROPE_INIT_FUNCTIONS["yarn"]`),构造一个真实的 `LlamaConfig`(带 `rope_scaling` 字段)走一遍完整流程,而不是把函数体复制出来本地重新跑一遍"——复制函数体本地跑,验证的是"我抄对了这段代码",不是"生产库真的是这么算的"。
- **追问 4(真实性验证轴的核心,逼问"是不是巧合"):** "你在 `factor=4` 这一个点上发现了差异。这会不会只是 `factor=4` 这个具体数字凑巧的巧合——比如某个四舍五入、或者某个特定数值下的重合?你怎么排除这种可能?"——期望候选人主动提出:不能只测一个点就下结论,必须在多个 `factor` 取值上重复验证,如果"生产库无 sqrt、教学代码互为倒数"这个关系在一整段区间都精确成立,才能说这是一个系统性的公式差异,不是单点巧合。
- **深挖追问(这个差异重不重要):** "就算不是巧合,这个差异在实际模型里造成的影响,是一个可以忽略不计的常数级小误差,还是会随着你要扩展的目标倍数变得更明显?"——期望候选人用同一组多点验证的数据回答:差距不是恒定的——`factor` 越大(意味着你在做越激进的上下文扩展,比如从 4k 一路扩到 128k),生产库的放大效应和教学代码的缩小效应之间的差距就越大,不是一个可以在任何场景下都无视的固定小数目。
- **深挖追问(收尾,工程判断力):** "发现这个之后,你会怎么处理这份教学代码,或者怎么跟依赖它的同事交代?"——期望答"标注清楚这是简化实现、注明对比过的库版本号、不能把这份代码算出来的数值直接当成生产库的真实行为写进汇报或者移植进生产代码",而不是含糊地说"细节可能有点出入"就带过去。

**可运行例子(1/2):复现"import 失败→找到真实入口"这条验证路径,并且把两侧结果都当场算出来比较,不写死任何一方的长小数**

```python
import math
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# 追问2 的现场复现: 直觉的 import 路径先验证是错的
try:
    from transformers.modeling_rope_utils import get_mscale
    assert False, "不应该 import 成功"
except ImportError:
    pass   # 确认: get_mscale 是嵌套在 _compute_yarn_parameters 内部的闭包,模块顶层拿不到

# 追问3 的现场复现: 走生产代码真正暴露的入口,而不是本地重新抄一遍公式
factor = 4.0
cfg = LlamaConfig(hidden_size=64, num_attention_heads=4,
                   max_position_embeddings=int(2048 * factor), rope_theta=10000.0,
                   rope_scaling={"rope_type": "yarn", "factor": factor,
                                 "original_max_position_embeddings": 2048})
_, attention_factor = ROPE_INIT_FUNCTIONS["yarn"](cfg, device=None)

# 两个"当场算出来的值"互相比较,不写死任何一方的长小数——这正是被反复强调的验证纪律
formula_no_sqrt = 0.1 * math.log(factor) + 1.0
teaching_scale = math.sqrt(1.0 / formula_no_sqrt)          # rope_yarn.py 的写法
capstone_temp = math.sqrt(formula_no_sqrt)                   # capstone_yarn_llama32.py 的写法

assert math.isclose(attention_factor, formula_no_sqrt, rel_tol=1e-9)     # 生产库 == "无 sqrt" 公式
assert not math.isclose(attention_factor, teaching_scale, rel_tol=1e-3)   # 生产库 != 教学代码(rope_yarn.py)
assert math.isclose(teaching_scale * capstone_temp, 1.0, rel_tol=1e-9)   # 两份教学代码互为倒数(自洽但都不是生产值)

print(f"factor={factor}: 生产库 attention_factor={attention_factor:.6f}(核实等于'无sqrt'公式);"
      f"rope_yarn.py 的 teaching_scale={teaching_scale:.6f}; capstone_yarn_llama32.py 的 capstone_temp={capstone_temp:.6f}"
      f"(两者乘积={teaching_scale*capstone_temp:.6f},确认互为倒数)。")
```

实测输出:`factor=4.0: 生产库 attention_factor=1.138629(核实等于'无sqrt'公式);rope_yarn.py 的 teaching_scale=0.937149; capstone_yarn_llama32.py 的 capstone_temp=1.067066(两者乘积=1.000000,确认互为倒数)。`

**可运行例子(2/2):追问 4——多个 `factor` 取值上重复验证"不是巧合",并且验证差距随 `factor` 扩大**

```python
import math
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

factors = [2.0, 4.0, 8.0, 16.0, 32.0]
amplify_ratios = []   # 生产库: Q、K 各乘一次 attention_factor,点积被放大 attention_factor^2 倍
damp_ratios = []      # 教学代码: 直接乘一次 attn_scale(<1),点积被缩小

for factor in factors:
    cfg = LlamaConfig(hidden_size=64, num_attention_heads=4,
                       max_position_embeddings=int(2048 * factor), rope_theta=10000.0,
                       rope_scaling={"rope_type": "yarn", "factor": factor,
                                     "original_max_position_embeddings": 2048})
    _, attention_factor = ROPE_INIT_FUNCTIONS["yarn"](cfg, device=None)
    formula_no_sqrt = 0.1 * math.log(factor) + 1.0
    teaching_scale = math.sqrt(1.0 / formula_no_sqrt)

    assert math.isclose(attention_factor, formula_no_sqrt, rel_tol=1e-9)  # 5 个 factor 全部精确匹配,不是 factor=4 的巧合
    amplify_ratios.append(attention_factor ** 2)
    damp_ratios.append(teaching_scale)

# "生产库始终放大、教学代码始终缩小"这个方向性差异,在整个区间无一例外
assert all(a > 1.0 for a in amplify_ratios)
assert all(d < 1.0 for d in damp_ratios)

# 差距是不是随 factor 增大而扩大?(不是恒定的小误差,而是越激进扩展、差距越大)
assert amplify_ratios[-1] > amplify_ratios[0] * 1.5     # factor=32 时的放大倍数明显超过 factor=2 时的 1.5 倍以上
assert damp_ratios[-1] < damp_ratios[0]                    # factor 越大,教学代码"缩小"得也越狠

for factor, amp, damp in zip(factors, amplify_ratios, damp_ratios):
    print(f"factor={factor:>5.1f}: 生产库把 Q·K 放大 {amp:.4f}x, 教学代码把 score 缩小 {damp:.4f}x")
```

实测输出:
```
factor=  2.0: 生产库把 Q·K 放大 1.1434x, 教学代码把 score 缩小 0.9670x
factor=  4.0: 生产库把 Q·K 放大 1.2965x, 教学代码把 score 缩小 0.9371x
factor=  8.0: 生产库把 Q·K 放大 1.4591x, 教学代码把 score 缩小 0.9099x
factor= 16.0: 生产库把 Q·K 放大 1.6314x, 教学代码把 score 缩小 0.8848x
factor= 32.0: 生产库把 Q·K 放大 1.8133x, 教学代码把 score 缩小 0.8618x
```
五个 `factor` 取值全部精确匹配"生产库无 sqrt"这条公式,`factor=4` 不是巧合;而且差距本身在扩大——`factor=2` 时生产库放大 1.14 倍、教学代码缩小到 0.97 倍(差距还比较小),`factor=32` 时变成放大 1.81 倍 vs 缩小到 0.86 倍(差距明显更大)。这意味着**越是要做激进的上下文扩展(更大的 factor),教学代码这个简化公式和生产库真实行为的差距就越不能忽略**,不是一个可以在任何场景下都无视的固定小误差。

**常见坑:** 只在一个参数点上发现"数字对不上"就直接下结论,没有意识到面试官(以及任何严肃的技术判断)会追问"这是不是巧合"——真实性验证轴的核心纪律就是"结论要能在参数扫描下站得住,不能只靠一个精心挑选(或者凑巧撞上)的例子";另一个常见坑是验证到"公式差一个 sqrt"就停下,不去追问"这个差异会不会随场景变化而变得更严重"——本案例已经现场验证过,这个差距不是常数,而是随 `factor` 增大而扩大,这一层追问经常是区分"验证到位"和"验证到位但没有深挖影响面"的分水岭。

---

## 案例 3:KV-cache 显存——上下文长度、并发数两条轴各自把单卡推到 OOM(规模递增轴)

建立在 [04 类知识点 4](04-long-context-capstone.md)(KV-cache 显存膨胀)之上。04 类已经把"1B 模型 + 128k 上下文,KV cache 是权重的 8.59 倍"这个具体算例跑通,还追加验证过"这个 8 倍的比例只在特定模型规模下成立,换成 8B/70B 模型比例完全不同"。这个案例换一条完全不同的递增轴——**固定模型规模不变,把"上下文长度"和"并发请求数"分别当成独立的自变量往上推,量化出单卡在哪个具体数字上开始装不下**,这是"这个模型能不能实际部署"这类系统设计问题的核心计算。

**追问链条完整还原:**

- **Q(基础,04 类已覆盖):** "给定层数、KV 头数、head_dim、精度,你能现场算一下某个具体上下文长度下 KV cache 占多少显存吗?"——期望候选人写出 `n_layers × n_kv_heads × head_dim × 2(K+V) × seq_len × bytes_per_elem` 这条公式并代入算出一个数字。
- **追问 1(规模往上跳一级):** "如果把这个模型部署在一张 80GB 的卡上(减去模型权重本身占用的显存),上下文长度从 4k 一路推到 1M,这张卡还能撑住吗?具体在哪个长度开始装不下?"——期望候选人现场把公式套进一个循环里逐档算出来,而不是凭感觉说"应该到几十万就不行了"。
- **追问 2(维度切换,不再是长度而是并发数):** "就算单个请求的上下文长度很保守、比如只有 8k,如果要同时服务多个用户呢?这张卡撑不撑得住,取决于哪个变量?"——期望候选人意识到 KV cache 公式里其实还隐含一个 `batch`(并发请求数)维度,把并发数当成第二条独立的自变量重新扫一遍,现场算出撑爆的具体并发数量级。
- **深挖追问(两条轴合并,区分度很高):** "现在两个条件一起上——32 个并发用户,每人 32k 上下文,你分别单独看这两个数字的时候是不是都觉得'还好'?合在一起呢?"——期望候选人现场推出:上下文长度和并发数在公式里是相乘关系,不是相加关系,两个"看起来都在预算内"的值乘在一起可能远超预算——这是很多人只对着"单个变量"做容量评估时会漏掉的陷阱。
- **深挖追问(工程对策,考察知道具体数字才能选对方案):** "扛不住之后,你有哪些工程选项?各自能把这条墙往后推多少,是随口说的,还是能算出来的?"——期望候选人至少举出 KV 量化这一具体可算选项,并且现场算出量化前后"最大并发数"这个指标具体从多少变到多少,而不是空泛地说"可以量化一下"。

**可运行例子(1/2):上下文长度、并发数两条轴各自独立把同一张卡推到 OOM**

```python
n_layers, n_kv_heads, head_dim, kv_factor, bytes_per_elem = 32, 8, 128, 2, 2  # bf16, 复用 04 类同款架构参数

def kv_cache_bytes(seq_len, batch=1, bpe=bytes_per_elem):
    return n_layers * n_kv_heads * head_dim * kv_factor * seq_len * bpe * batch

model_weight_GB = 8_000_000_000 * 2 / 1e9         # 8B 参数, bf16
GPU_BUDGET_GB = 80.0                                 # 单张 80GB 训练/推理卡(A100/H100 同规格量级)
avail_GB = GPU_BUDGET_GB - model_weight_GB           # 简化估算: 只算权重+KV cache,不计activation等其它开销
assert avail_GB == 64.0

# ---- 轴1: 固定 batch=1,把上下文长度一路推到 1M ----
ctx_results = {}
for ctx in [4096, 8192, 32768, 131072, 1_048_576]:
    ctx_results[ctx] = kv_cache_bytes(ctx) / 1e9

assert ctx_results[131072] < avail_GB                # 128k: 单请求还装得下(实测约17.18GB,约26.8%预算)
assert ctx_results[1_048_576] > avail_GB              # 1M: 单个请求自己就超过整卡剩余预算(实测约137GB)
assert ctx_results[1_048_576] > ctx_results[131072] * 7.9   # 长度乘 8(128k→1M),KV cache 也近似线性乘 8

# ---- 轴2: 固定 ctx=8192(单个请求很"便宜"),把并发数一路推上去 ----
batch_results = {}
for batch in [1, 8, 32, 64, 128, 256]:
    batch_results[batch] = kv_cache_bytes(8192, batch=batch) / 1e9

assert batch_results[32] < avail_GB                   # 32 并发还行(实测约34.36GB)
assert batch_results[64] > avail_GB                    # 64 并发就撑不住了(实测约68.72GB)

print(f"上下文长度轴(batch=1): 128k={ctx_results[131072]:.2f}GB(预算内) -> "
      f"1M={ctx_results[1_048_576]:.2f}GB(超预算{ctx_results[1_048_576]/avail_GB*100:.0f}%)")
print(f"并发数轴(ctx=8k): 32并发={batch_results[32]:.2f}GB(预算内) -> 64并发={batch_results[64]:.2f}GB(超预算)")
print("两条轴各自独立就能把同一张卡推到 OOM,不需要两个条件同时出现。")
```

实测输出:
```
上下文长度轴(batch=1): 128k=17.18GB(预算内) -> 1M=137.44GB(超预算215%)
并发数轴(ctx=8k): 32并发=34.36GB(预算内) -> 64并发=68.72GB(超预算)
两条轴各自独立就能把同一张卡推到 OOM,不需要两个条件同时出现。
```

**可运行例子(2/2):两条轴合并后的相乘效应 + KV 量化能把墙推到多远**

```python
n_layers, n_kv_heads, head_dim, kv_factor = 32, 8, 128, 2

def kv_cache_bytes(seq_len, batch=1, bpe=2):
    return n_layers * n_kv_heads * head_dim * kv_factor * seq_len * bpe * batch

model_weight_GB = 8_000_000_000 * 2 / 1e9
avail_GB = 80.0 - model_weight_GB

# ---- 深挖追问: 两条轴"看起来都还好"的取值,合在一起会怎样? ----
solo_ctx_GB = kv_cache_bytes(32768, batch=1) / 1e9       # 32k 上下文,单独看很安全
solo_batch_GB = kv_cache_bytes(2048, batch=32) / 1e9       # 32 并发,配合很短的上下文也很安全
combo_GB = kv_cache_bytes(32768, batch=32) / 1e9            # 32 并发 x 32k 上下文,两条轴一起上

assert solo_ctx_GB < avail_GB and solo_batch_GB < avail_GB    # 两条轴分开看都在预算内
assert combo_GB > avail_GB                                      # 但乘在一起立刻超预算
assert abs(combo_GB - solo_ctx_GB * 32) < 1e-6                   # 两条轴是相乘关系,不是相加关系(32倍恰好对应并发数)

# ---- 工程对策: KV 量化能把这条墙往后推多少? ----
ctx = 131072
bf16_GB = kv_cache_bytes(ctx, bpe=2) / 1e9
int4_GB = kv_cache_bytes(ctx, bpe=0.5) / 1e9
max_concurrent_bf16 = int(avail_GB // bf16_GB)
max_concurrent_int4 = int(avail_GB // int4_GB)

assert max_concurrent_int4 > max_concurrent_bf16 * 3          # int4 相比 bf16 是 4 字节->1 字节量级的省显存,并发上限明显跳升
print(f"32并发x32k: 单独看{solo_ctx_GB:.2f}GB和{solo_batch_GB:.2f}GB都在{avail_GB:.0f}GB预算内,"
      f"合起来却是{combo_GB:.2f}GB(超预算)——两条轴是相乘关系。")
print(f"128k上下文场景: bf16下最多能撑{max_concurrent_bf16}个并发请求,换成int4量化能撑到{max_concurrent_int4}个"
      f"(每字节数从2降到0.5,显存降为1/4,不是免费的——量化本身有精度损失代价)。")
```

实测输出:
```
32并发x32k: 单独看4.29GB和8.59GB都在64GB预算内,合起来却是137.44GB(超预算)——两条轴是相乘关系。
128k上下文场景: bf16下最多能撑3个并发请求,换成int4量化能撑到14个(每字节数从2降到0.5,显存降为1/4,不是免费的——量化本身有精度损失代价)。
```

**常见坑:** 只沿着一条轴(通常是"上下文长度")做容量评估,却忘了并发数是公式里另一个同样会线性放大显存的独立变量——生产环境里"单个请求的上下文不算特别长,但并发量很大"导致 OOM,和"单个请求上下文长得离谱"导致 OOM,是同样常见的两种真实故障模式,只准备了其中一种解释容易在追问"那如果是很多个短请求同时来呢"时卡住;另一个坑是想到"可以量化"就以为问题解决了,答不出量化之后具体能把并发上限抬到多少这个数字,也没有主动提及量化本身是有精度损失代价的,不是免费午餐。

---

## 案例 4:Ring / Striped / Infini-Attention 选型——面试官连续换约束,追问你的选择依据(决策依据追问轴)

建立在 [02 类知识点 1、3、4](02-long-context-attention.md)(Ring Attention、Infini-Attention、三种架构对比选型)之上。02 类的对比表已经把"精确 vs 近似""显存怎么被消化掉""FLOPs 有没有真的变少"这几个维度整理清楚了;这个案例不重复那张表,而是模拟面试官**不断切换约束条件**、追问"给这个约束你选哪个、为什么不选另外两个"——决策依据追问轴的特点是不纠错(候选人没有说错什么),纯粹逼问"你是怎么权衡的",而且追问里会要求把"够用"这种判断现算出一个具体数字,不能只凭直觉。

**追问链条完整还原:**

- **面试官给约束 1:** "预训练一个支持 128k 上下文的模型,你有 8 张 A100,要求梯度完全精确、不能有任何近似,Ring / Striped / Infini-Attention 你会选哪个?为什么不选另外两个?"——期望答"Ring 或 Striped(02 类已用 assert 验证过两者和标准 attention 数值等价,diff 在 1e-7 浮点误差量级),直接排除 Infini——它引入了有损压缩,不满足'梯度精确'这个硬约束"。
- **追问 1:** "Ring 和 Striped 之间,你怎么选?"——期望答"取决于是不是 causal(自回归)场景:causal 场景下朴素 Ring 会因为因果 mask 出现负载不均衡(持有序列末尾 Q 块的 rank 几乎不能跳过任何计算,持有开头的 rank 很早就没活干),Striped 通过重排 K 块的消费顺序缓解这个问题;non-causal/双向场景下两者本来就没有负载不均衡问题,用普通 Ring 足够"。
- **追问 2(区分度很高,逼问具体依据不是空话):** "你说'8 张卡应该够用',这个'够用'是算出来的,还是感觉出来的?如果目标长度从 128k 涨到 1M,还是 8 张卡,你的 Ring 方案还扛得住吗?"——期望候选人现场做一次容量规划,而不是直接说"应该没问题":给每张卡的峰值 attention score 矩阵设一个显存预算,现算出 128k 场景下最少需要多少张卡、1M 场景下又需要多少张卡,再对比 8 这个数字——**"够不够用"必须是一个可以现场算出来的数字,不是凭经验的估计**。
- **面试官切换约束 2:** "现在换个场景——只有单卡,推理阶段要支持几乎无限长的对话历史(几百万 token 量级的历史消息),不能加卡,你选哪个?"——期望答"Infini-Attention:它是三者里唯一能做到显存和 token 数基本无关的方案(02 类已验证 `M`/`Z` 这两个压缩记忆张量的 shape 不随输入序列长度变化),Ring/Striped 本质上仍然需要更多硬件去分摊显存,单卡场景下用不上"。现场可以把"如果不用 Infini、继续用标准 KV cache 存下这几百万 token 的历史"需要多少显存,和 Infini 压缩记忆的显存量级摆在一起对比,数量级差距会非常悬殊。
- **深挖追问(不能只念优点,要主动交代局限):** "选了 Infini-Attention 之后,有没有什么场景它其实做不好,你会不会提前跟业务方说清楚?"——期望答"02 类已经明确标注过:Infini 在 NIAH(单点检索)上表现不错,但在 RULER(更综合的评测,包含聚合、多跳追踪)上明显更弱——如果业务场景要求精确检索很久之前提到的一个具体事实(比如用户几十万 token 之前提到的一个电话号码),压缩记忆没法保证精确召回,这时候即使不能加卡,也应该优先考虑'标准 KV cache + 淘汰策略(H2O/SnapKV 之类)'这类折中方案,而不是无脑上 Infini",诚实交代局限比只念优点更能体现决策依据的可信度。

**可运行例子(1/2):Ring Attention 的 GPU 数量容量规划——小规模真实跑通验证切分关系,大规模用同一关系纯算术外推**

```python
import sys, math
sys.path.insert(0, "learning/long-context/src")
import torch
from ring_attention_naive import ring_attention_naive

# ---- 先用可行规模真的跑一遍,确认 chunk = t // n_rank 这个关系来自源码本身,不是臆测 ----
torch.manual_seed(0)
t_small, n_rank_small = 24, 4
q = torch.randn(1, 2, t_small, 8)
k = torch.randn(1, 2, t_small, 8)
v = torch.randn(1, 2, t_small, 8)
out = ring_attention_naive(q, k, v, n_rank=n_rank_small)
assert out.shape == q.shape
chunk_small = t_small // n_rank_small
assert chunk_small == 6         # 真实跑出来的切分关系:24 长度切 4 份,每份 6

# ---- 用同一个关系做纯算术外推(L=128k/1M 太大,不会真的分配张量去跑,只是代入公式)----
def min_gpus_for_budget(L, budget_bytes, bytes_per_elem=4):
    # 每张卡峰值 score 矩阵是 chunk x chunk(fp32),chunk = L / n_rank
    max_chunk = math.floor(math.sqrt(budget_bytes / bytes_per_elem))
    return math.ceil(L / max_chunk)

budget = 2.0 * 1e9   # 单卡给 attention score 矩阵留 2GB 预算(留白给权重/优化器状态等其它开销)
n_gpu_128k = min_gpus_for_budget(131072, budget)
n_gpu_1m = min_gpus_for_budget(1_048_576, budget)

assert n_gpu_128k == 6
assert n_gpu_1m == 47
assert n_gpu_1m > n_gpu_128k * 6      # 目标长度只涨了 8 倍,需要的卡数却涨了将近 8 倍,不是"多个几张就够"

ratio_len = 1_048_576 / 131072
ratio_gpu = n_gpu_1m / n_gpu_128k
print(f"budget=2GB/卡: 128k 至少需要 {n_gpu_128k} 张卡, 1M 至少需要 {n_gpu_1m} 张卡"
      f"(长度涨了{ratio_len:.0f}倍,卡数需求涨了约{ratio_gpu:.1f}倍)。"
      f"如果预训练规划阶段按 8 张卡配置(刚好够 128k 场景用,留了小量余地),"
      f"想不改架构直接把目标长度推到 1M,大概率要么把卡数加到 {n_gpu_1m} 张左右,"
      f"要么换 FlashAttention 式的分块 kernel 避免整块物化 score 矩阵——'8 张卡应该够用'这句话必须先算过才能说。")
```

实测输出:`budget=2GB/卡: 128k 至少需要 6 张卡, 1M 至少需要 47 张卡(长度涨了8倍,卡数需求涨了约7.8倍)。如果预训练规划阶段按 8 张卡配置(刚好够 128k 场景用,留了小量余地),想不改架构直接把目标长度推到 1M,大概率要么把卡数加到 47 张左右,要么换 FlashAttention 式的分块 kernel 避免整块物化 score 矩阵——'8 张卡应该够用'这句话必须先算过才能说。`(这里的"峰值 score 矩阵整块物化"是一个刻意简化的估算口径,真实生产系统内部通常会用 FlashAttention 式的分块 kernel 进一步降低单卡峰值显存,不会真的物化一整块 `chunk × chunk` 矩阵——这个简化和 04 类"块对角 mask 不应该真的物化 `O(n²)` 矩阵"是同一类简化,这里只是为了得到一个可以现场算的、方向正确的数量级估计,不是宣称这就是生产系统的精确实现细节。)

**可运行例子(2/2):Infini-Attention 记忆状态恒定 vs 标准 KV cache 线性增长——量级对比**

```python
import sys
sys.path.insert(0, "learning/long-context/src")
import torch
from infini_attention import InfiniBlock

torch.manual_seed(0)
m = InfiniBlock(d_model=32, n_head=4)

mem_bytes = []
for L in [8, 100, 2000]:
    x = torch.randn(1, L, 32)
    y, M, Z = m(x)
    mem_bytes.append(M.numel() * M.element_size() + Z.numel() * Z.element_size())

assert mem_bytes[0] == mem_bytes[1] == mem_bytes[2]     # 序列长度差了 250 倍,压缩记忆的字节数完全不变

# 和案例 3 同一条 KV-cache 公式对照(这里只是复用公式做对比,不重复案例 3 的完整推导)
n_layers, n_kv_heads, head_dim, kv_factor = 32, 8, 128, 2
def kv_cache_bytes(seq_len, bpe=2):
    return n_layers * n_kv_heads * head_dim * kv_factor * seq_len * bpe

for L in [8192, 131072, 1_048_576]:
    kv_gb = kv_cache_bytes(L) / 1e9
    infini_gb = mem_bytes[0] / 1e9
    assert kv_gb > infini_gb * 1000    # 差距是数量级级别的,不是"稍微省一点"
    print(f"seq_len={L:>8}: 标准 KV cache={kv_gb:9.3f}GB  vs  Infini 压缩记忆={infini_gb:.9f}GB(恒定)")

print(f"\n注意: InfiniBlock 这里 d_model=32/n_head=4 是教学玩具规模,{mem_bytes[0]}字节本身不是生产数值——"
      f"真实模型的 d_h 大得多,压缩记忆的绝对字节数会更大,但'和 L 无关、恒定不变'这个增长曲线的形状是一样的,"
      f"这才是这组对比真正要证明的结论,不是这个具体的字节数。")
```

实测输出:
```
seq_len=    8192: 标准 KV cache=    1.074GB  vs  Infini 压缩记忆=0.000001152GB(恒定)
seq_len=  131072: 标准 KV cache=   17.180GB  vs  Infini 压缩记忆=0.000001152GB(恒定)
seq_len= 1048576: 标准 KV cache=  137.439GB  vs  Infini 压缩记忆=0.000001152GB(恒定)
```

**常见坑:** 回答"选哪个方案"时不主动区分训练/推理场景、不主动区分硬件预算是否充足,给出一个不看约束条件的固定排位("Infini 最先进所以选它"或者"精确的肯定比近似的好")——02 类已经反复强调这不是排位赛,是针对约束的权衡;另一个坑是被问"8 张卡够不够"这类容量规划问题时,只会定性地说"应该够"或者"可能不太够",给不出一个可以现场推导的数字——这类追问专门筛选"背得出架构对比表"和"真能现场做容量估算"这两种不同层次的候选人。

---

## 小结:4 个案例对应调研发现的哪些轴线

| 案例 | 规模递增轴 | 工程约束递增轴(并发/分布式) | 方案批判迭代轴 | 决策依据追问轴 | 真实性验证轴 |
|---|---|---|---|---|---|
| 1. RoPE 外推方案批判迭代 | | | ✅ 核心 | | |
| 2. YaRN 教学代码 vs 生产库 | | | | | ✅ 核心 |
| 3. KV-cache 显存规模递增 | ✅ 核心 | ✅(并发数轴) | | | |
| 4. Ring/Striped/Infini 选型 | | ✅(GPU 数量规划) | | ✅ 核心 | |

这 4 个案例不是要把 `long-context-deep-dive` 17 个知识点全部重新过一遍追问链——它们演示的是**方法论本身**:拿到 01-04 里任何一个已经验证过的知识点,都可以自己追问"如果面试官不满足于我提出的第一个方案,连续换 3 次约束,我的方案应该怎么迭代""这个'教学代码实现了论文公式'的说法,我能不能自己动手核实到什么程度""这个数字在规模、并发同时往上顶的时候还成立吗""给我这个具体约束,我选择方案的依据是能现算出来的数字,还是一句经验之谈"。真正的二面深度,是能不能对着一个自己没有专门准备过的知识点,现场把这几条轴线走一遍,而且每一步都能用代码把"应该是这样"变成"我刚刚跑出来确实是这样"。
