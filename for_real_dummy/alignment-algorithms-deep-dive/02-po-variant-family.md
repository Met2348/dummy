# 02 · PO 变体家族深挖(DPO Variant Family)

> 总览见 [00-roadmap.md](00-roadmap.md)

DPO 把 RLHF 的三段式(reward model + PPO)压缩成一个可以直接在偏好数据上做最大似然的 loss。但"DPO 本身"只是这条思路里的第一个答案,不是唯一答案——本文这 6 段代码分别对 DPO 动了不同的一刀:IPO 嫌 DPO 的 loss 形状"贪心"(margin 再大也不嫌大),换成有明确目标值的平方损失;KTO 嫌 DPO 必须要求成对比较数据太苛刻,把配对要求整个拆掉;ORPO/SimPO/CPO 三个都盯上了"必须额外加载一个 reference 模型"这件事,用三种不同方式把它省掉;DPOP 则是找到了 DPO 一个具体的反直觉 bug(chosen 概率明明在降,loss 却说训练在变好),专门打了个补丁。7 个知识点,6 段代码,全部是秒级跑完的纯数值 demo。

**和 [01-dpo-foundations.md](01-dpo-foundations.md) 的关系:** 01 号文件讲 DPO 本身怎么从 RLHF 推导出来、loss 每一项对应什么。本文默认你已经理解 DPO loss 的骨架——`log_ratio = log π_actor(y|x) - log π_ref(y|x)`(actor 相对 ref 挪动了多少)、`margin = β·(log_ratio_chosen - log_ratio_rejected)`(相对偏好强度)、`loss = -log sigmoid(margin)` ——不再重复这部分推导,只讲 6 种变体各自改了这个骨架的哪一块。

**一个重要的诚实标注(继承自 00-roadmap.md 的差异化声明):** 下面 7 个知识点里,知识点 3(ORPO)和知识点 4(SimPO)有配套 lecture(`lectures/04-orpo.md`/`05-simpo.md`),写作时参考了它们对机制的讲解,但**所有可运行例子的数字都是本文用不同于 lecture/源文件自带 smoke test 的输入重新独立跑出来的**,不是照抄。知识点 1/2/5/6(IPO/KTO/CPO/DPOP)完全没有配套 lecture,全部直接从 `learning/dpo-family/src/*.py` 的函数签名、实现和 docstring 出发理解,不假设存在可转述的中文讲解。

**环境声明:** 本文涉及的全部 6 个源文件(`ipo_minimal.py`/`kto_minimal.py`/`orpo_minimal.py`/`simpo_minimal.py`/`cpo_minimal.py`/`dpop_minimal.py`)都是纯 torch 张量级数值 demo——不加载任何模型、不需要 GPU、不需要网络,秒级跑完。本文所有代码例子已在仓库根目录 `.venv`(torch 2.11.0+cu128)下实际跑通验证,文中数字是真实输出,不是手算或转述。

---

## 1. `ipo_loss()`(`ipo_minimal.py`)—— 把 DPO 的 sigmoid 换成平方损失,治"过度自信"

**是什么:**
```python
def ipo_loss(
    log_p_c_actor: torch.Tensor, log_p_c_ref: torch.Tensor,
    log_p_r_actor: torch.Tensor, log_p_r_ref: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """IPO: squared loss 防止 over-confidence."""
    log_ratio_c = log_p_c_actor - log_p_c_ref
    log_ratio_r = log_p_r_actor - log_p_r_ref
    h = log_ratio_c - log_ratio_r
    target = 1.0 / (2 * beta)
    return ((h - target) ** 2).mean()
```

**一句话:** DPO 的 loss 是"h(相对 margin)越大越好"的 sigmoid 形状,IPO 把它换成"h 距离某个目标值 `1/(2β)` 越近越好"的平方损失——从"多多益善"变成"刚刚好最好"。

**底层机制/为什么这样设计:** DPO loss `= -log sigmoid(β·h)`,对 h 求导 `dL/dh = -β·sigmoid(-β·h)`。这个导数对任意有限的 h **恒为负**——不管 h 已经多大,优化器永远收到"h 还要更大"的信号,只是随着 h 增大,`sigmoid(-β·h)→0`,导数的绝对值按指数衰减、趋近 0 但永远不等于 0。这正是 docstring 里"DPO 在 chosen/rejected margin 太大时仍然推优化方向,导致 over-fit"这句话的精确数学含义:DPO 没有"够了"这个概念。IPO 把 loss 换成 `(h - target)²`,`dL/dh = 2(h - target)`,在 `h = target` 处**精确为 0**(真正的极小值,不是"趋近于 0"),一旦 h 超过 target,导数变正,loss 反而重新增大——优化器会被主动推回来,而不是继续放任 margin 无限扩大。`target = 1/(2β)` 和 β 成反比,呼应 01 号文件里"β 控制离 ref 多远算合适"这个角色:β 越大(越不希望 actor 离 ref 太远),target 反而越小(margin 到一个更保守的值就算达标)。

**AI 研究场景:** reward over-optimization(奖励过度优化)是 DPO 类方法训练时间拉长后的经典失败模式——policy 在 chosen/rejected 这一个具体维度上无止境地加大优势,可能以牺牲没被这批数据覆盖到的能力为代价(类比过拟合)。在担心这种"训过头"的场景(数据集小、担心 reward hacking),IPO 这种"设定目标值、越界倒扣"的设计比 DPO 更保守稳妥。

**可运行例子:**
```python
import torch
import sys
sys.path.insert(0, "learning/dpo-family/src")
from ipo_minimal import ipo_loss, dpo_loss

beta = 0.1
target = 1.0 / (2 * beta)
assert target == 5.0

# 固定 ref logp = 0，只改 actor 一侧的 raw margin h = log_ratio_c - log_ratio_r
raw_margins = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0]
dpo_losses, ipo_losses = [], []
for h in raw_margins:
    c = torch.tensor([h]); r = torch.tensor([0.0]); zero = torch.tensor([0.0])
    dpo_losses.append(dpo_loss(c, zero, r, zero, beta).item())
    ipo_losses.append(ipo_loss(c, zero, r, zero, beta).item())

assert all(dpo_losses[i] > dpo_losses[i + 1] for i in range(len(dpo_losses) - 1))      # DPO 全程单调递减
idx5 = raw_margins.index(5.0)
assert ipo_losses[idx5] == 0.0                                                          # IPO 在 target 处精确为 0
assert all(ipo_losses[i] > ipo_losses[i + 1] for i in range(idx5))                      # target 之前，IPO 也在降
assert all(ipo_losses[i] < ipo_losses[i + 1] for i in range(idx5, len(ipo_losses) - 1))  # 过了 target，IPO 反而升高

# 用 autograd 直接验证 "DPO 永不停" vs "IPO 在 target 处梯度精确为 0"
h1 = torch.tensor([200.0], requires_grad=True)
dpo_loss(h1, torch.zeros(1), torch.zeros(1), torch.zeros(1), beta).backward()
assert h1.grad.item() < 0                          # margin 已经 200 了，DPO 梯度依然是负的（还在推）

h2 = torch.tensor([target], requires_grad=True)
ipo_loss(h2, torch.zeros(1), torch.zeros(1), torch.zeros(1), beta).backward()
assert abs(h2.grad.item()) < 1e-6                   # IPO 在 target 处梯度精确为 0（真正的极小值）
```

实测数字(`.venv` 真跑,节选;完整 11 个点见上面代码里的 `raw_margins`):

| raw margin h | DPO loss | IPO loss |
|---|---|---|
| 0.0 | 0.6931 | 25.0000 |
| 1.0 | 0.6444 | 16.0000 |
| 4.0 | 0.5130 | 1.0000 |
| 5.0(=target) | 0.4741 | **0.0000** |
| 6.0 | 0.4375 | 1.0000 |
| 10.0 | 0.3133 | 25.0000 |
| 50.0 | 0.0067 | 2025.0000 |

DPO 梯度(`dL/dh`)随 h 增大而衰减,但从未变号:h=1.0 时 `-0.0475`,h=10.0 时 `-0.0269`,h=50.0 时 `-0.00067`,h=200.0 时按 8 位小数打印已显示为 `-0.00000000`(解析值约 `-2×10⁻¹⁰`,仍是负数,只是超出了打印精度)——这就是"收益递减但不会真的停"的实测证据。IPO 在 h=target 处梯度精确为 `0.0000000000`。

**面试怎么问 + 追问链:**
- **Q:** "IPO 和 DPO 的 loss 本质区别是什么?"—— 期望答出"DPO 是 sigmoid 形状,鼓励 margin 越大越好;IPO 是二次函数,鼓励 margin 恰好等于某个目标值"。
- **追问 1:** "如果训练一直让 margin 变得非常大,DPO 和 IPO 分别会发生什么?"—— 期望结合梯度符号说清楚:DPO 梯度永远是负的(只是越来越小),IPO 过了目标点梯度会变正、主动往回拉,不是"不再更新"而是"反向修正"。
- **追问 2(深挖):** "IPO 的 `target = 1/(2β)` 这个值具体怎么来的,β 变大 target 会怎么变?"—— 期望看出两者成反比,并联系"β 本来就是控制'离 ref 多远算合适'的超参"这层背景(呼应 01 号文件),推出"β 越大(越保守)→ target 越小(更容易被满足)"这个一致的方向。

**常见坑:** 以为"margin 越大 IPO loss 就越小"——这是从 DPO 的直觉直接搬过来的,但 IPO 是 U 形,超过 target 之后 loss 会重新变大。如果拿"IPO loss 是否单调下降"去判断训练是否收敛,会在 margin 越过 target 后得出完全相反的错误结论。

---

## 2. `kto_loss()`(`kto_minimal.py`)—— 不要配对,只要"好/坏"二元标签

**是什么:**
```python
def kto_loss(
    log_p_actor: torch.Tensor,        # (B,) 每个样本 sum log π over response
    log_p_ref: torch.Tensor,           # (B,)
    is_desirable: torch.Tensor,        # (B,) 1=desired, 0=undesired
    beta: float = 0.1,
    lambda_d: float = 1.0,             # desired 权重
    lambda_u: float = 1.0,             # undesired 权重
) -> torch.Tensor:
```

**一句话:** KTO 把"同一个 prompt 必须准备一对 chosen/rejected"这个数据假设整个拆掉,只要求每条 `(response, 好/坏标签)` 独立标注,loss 是拿每条样本的 log-ratio 去和一个"batch 内参照水位"比较,而不是拿两条样本互相比较。

**底层机制/为什么这样设计:** 先纠正一个读代码之前很容易先入为主的误解——**KTO 不是"不需要 reference model"的方法**。它的函数签名里明明白白有一个 `log_p_ref` 参数,内部用 `log_ratio = log_p_actor - log_p_ref` 算出和 DPO 同源的"隐式 reward"。KTO 真正砍掉的是"pairing"(成对比较),不是 ref model,这是两件完全独立的事,只是在本系列后面 ORPO/SimPO/CPO 三个方法里恰好被一起砍掉了,容易被混为一谈。有了 `log_ratio` 之后,KTO 不是拿 chosen 的 log_ratio 减 rejected 的 log_ratio(DPO 的做法,需要同一个 prompt 下两条样本),而是让每条样本的 `log_ratio` 单独去和一个参照值 `z_ref` 比较:desired 样本希望 `log_ratio > z_ref`(用 `1 - sigmoid(β·(log_ratio - z_ref))` 作为 loss),undesired 样本希望 `log_ratio < z_ref`(用 `1 - sigmoid(β·(z_ref - log_ratio))`)。这份实现里 `z_ref` 是当前 batch 内 `log_ratio` 的均值(并 clamp 到非负)——是论文里"用一批（通常是打乱配对的）样本估计 `KL(π‖π_ref)`"这个思路的简化版,但核心机制一致:需要一个"整体参照水位"作为好坏的分界线,而不是每条样本各自为战。desired/undesired 两支分别用 `λ_d`/`λ_u` 加权,来自 Kahneman-Tversky 前景理论"人对损失比对等量收益更敏感"的直觉,这也是 KTO 名字的来源。

**AI 研究场景:** 真实产品里最容易大规模、低成本收集到的反馈通常是单边的——用户点了"踩"、审核标记"违规"、客服标注"这条回复不合格",这些都是"这一条 response 好还是不好"的独立标签,不会天然配对出"同一个 prompt 下另一条更好/更差的回复"。要凑出 DPO/IPO 需要的 pair,往往得额外生成第二条回复、再找人或模型比较,是明显更贵的数据构造步骤。KTO 允许直接用现成的单边反馈日志训练,是数据收集成本上的实质性降低,而不只是"少写一点代码"这种表面差异。

**可运行例子:**
```python
import inspect
import torch
import sys
sys.path.insert(0, "learning/dpo-family/src")
from kto_minimal import kto_loss

# 结构性证据：kto_loss 确实吃了 log_p_ref —— 说明它没有省掉 reference model
params = list(inspect.signature(kto_loss).parameters.keys())
assert params == ["log_p_actor", "log_p_ref", "is_desirable", "beta", "lambda_d", "lambda_u"]
assert "log_p_ref" in params

torch.manual_seed(42)
log_p_actor = torch.tensor([-1.0, -1.2, -0.8, -1.5, -0.9, -2.0, -2.5])
log_p_ref = torch.tensor([-1.3, -1.3, -1.3, -1.3, -1.3, -1.3, -1.3])
is_desirable = torch.tensor([1, 1, 1, 1, 1, 0, 0])          # 5 条好 + 2 条坏，注意：不是成对数据
L_mixed = kto_loss(log_p_actor, log_p_ref, is_desirable)
assert L_mixed.item() > 0

# KTO 甚至可以喂一个"全部都是好样本、一条坏样本都没有"的 batch —— DPO/IPO 的输入结构上做不到这件事
L_all_desirable = kto_loss(log_p_actor, log_p_ref, torch.ones(7))
assert L_all_desirable.item() > 0

print(f"mixed(5好+2坏, 不配对) loss = {L_mixed.item():.6f}")
print(f"all-desirable(7好+0坏) loss = {L_all_desirable.item():.6f}")
```

实测:`mixed(5好+2坏) loss = 0.489292`,`all-desirable(7好+0坏) loss = 0.502852`——两种截然不同的 batch 组成(甚至完全没有负样本)都能正常算出 loss,这在要求成对数据的方法里是不可能的输入形式。

**面试怎么问 + 追问链:**
- **Q:** "KTO 为什么不需要 chosen/rejected 配对数据?"—— 期望说出"用单条样本的 log_ratio 和一个参照水位比较,而不是两条样本互相比较"。
- **追问 1(全场最容易踩的坑):** "那 KTO 是不是就不需要 reference model 了?"—— 正确答案是"不是",需要明确指出 `kto_loss` 签名里仍然有 `log_p_ref`,KTO 省的是"配对",不是"ref 模型"。
- **追问 2:** "如果一个 batch 里全是 desirable 样本、一条 undesirable 都没有,DPO 能训练吗,KTO 呢?"—— 期望答"DPO/IPO 结构上要求每条数据自带一对 chosen/rejected,不存在'全是好样本'这种输入形式;KTO 每条样本独立算 loss,天然支持这种极端分布,虽然实践中样本极不均衡也会影响训练稳定性"。
- **追问 3(深挖 z_ref):** "z_ref 是什么,没有这个参照点会怎样?"—— 期望理解"log_ratio 需要一个可比较的基准,否则'多大算好'没有意义";这份实现用 batch 内均值近似,是论文用无关样本估计 KL 这个思路的简化版。

**常见坑:** 把"KTO 不需要配对数据"错误引申成"KTO 不需要 reference model"——这是网上不少总结性表格里常见的错误归类(把"免 ref"三兄弟 ORPO/SimPO/CPO 和"免配对"的 KTO 混成一类)。必须回到函数签名验证:KTO 依然要跑一次 ref model 的前向传播,只是每条样本只需要跑一次(不像 DPO 要对 chosen、rejected 各跑一次),这一点在知识点 7 的对比表里会进一步说清楚。

---

## 3. `orpo_loss()`(`orpo_minimal.py`,配套 `lectures/04-orpo.md`)—— odds ratio 替代 reference model

**是什么:**
```python
def log_odds(log_p: torch.Tensor) -> torch.Tensor:
    """log(p/(1-p)) = log p - log(1-p)."""
    log_p = log_p.clamp(max=-1e-6)
    return log_p - torch.log1p(-log_p.exp())

def orpo_loss(
    log_p_chosen: torch.Tensor,        # SFT NLL of chosen
    log_p_rejected: torch.Tensor,      # SFT NLL of rejected
    sft_loss: torch.Tensor,            # standard SFT loss
    lambda_or: float = 0.1,
) -> torch.Tensor:
    """ORPO = SFT loss + λ · log_odds_ratio penalty."""
```

**一句话:** ORPO 用"odds"(几率,`p/(1-p)`)代替 DPO 里"actor 相对 ref 挪了多少"这个量,让整个偏好项只依赖 actor 自己的输出概率,再叠加一个 SFT 项把 actor 锚住——于是**完全不需要加载 reference 模型**。

**底层机制/为什么这样设计:** DPO 的核心量是 `log_ratio = log π_actor(y|x) - log π_ref(y|x)`,衡量"actor 比 ref 更喜欢这个回答多少",这个定义天然需要一份 ref 快照。ORPO 换成 `log_odds(y|x) = log(p/(1-p)) = log p - log(1-p)`,其中 `p = π_actor(y|x)` 是 actor 自己对整条回答赋的概率(从 SFT 意义下的 NLL 换算而来),这个量完全不涉及"另一个模型",只是把 actor 自己的一个概率值通过 odds 变换映射成一个新分数。光有 `log_odds` 还不够——如果没有约束,一个没训好的 actor 也能靠 `log_odds` 打出乱七八糟的分数。ORPO 额外加一个标准 SFT loss(在 chosen 上做最大似然),把 actor 锚在"至少能正常生成"的区域;论文的论证方式是:在 SFT 主导、actor 分布合理的前提下,`log_odds` 之间的差值在效果上能起到和"相对 ref 的偏移量"类似的作用——**不是数学上严格等价,是工程上够用的替代品**(配套 lecture Slide 4 对这一点讲得很直白,本文认同这个定性但不重复它的公式推导)。数值稳定实现上,`log(1-p)` 直接算在 `p` 接近 1 时容易溢出,这里用 `torch.log1p(-log_p.exp())` 代替(`log1p(x)=log(1+x)` 对接近 0 的 `x` 数值更稳),是通用的浮点计算技巧,不是 ORPO 专属。

**AI 研究场景:** 训练 7B 级别模型时,同时常驻 actor + ref 两份权重是实打实的显存翻倍开销(配套 lecture Slide 6 给出的经验量级是 DPO 约 28GB vs ORPO 约 14GB,这是 lecture 里的架构性估算,不是本文数值 demo 能验证的部分,这里如实标注来源)。训练资源紧张、或者想在更小的卡上做偏好对齐时,ORPO 这类 ref-free 方法是直接可行的降本方案;代价是需要 SFT 阶段的数据质量本身就过关(细节见知识点 7)。

**可运行例子(独立于 lecture 重新验证,输入数字与 lecture/`orpo_minimal.py` 自带 smoke test 均不同):**
```python
import inspect
import torch
import sys
sys.path.insert(0, "learning/dpo-family/src")
from orpo_minimal import orpo_loss, log_odds

# 结构性证据：签名里没有任何 "ref" 参数
params = list(inspect.signature(orpo_loss).parameters.keys())
assert params == ["log_p_chosen", "log_p_rejected", "sft_loss", "lambda_or"]
assert not any("ref" in p for p in params)

log_p_c = torch.tensor([-0.4, -0.6, -0.3, -0.5])   # 4 条样本的 chosen SFT log prob（自定数字）
log_p_r = torch.tensor([-1.5, -1.8, -1.2, -2.0])   # rejected，明显更差
sft = torch.tensor(1.1)
L = orpo_loss(log_p_c, log_p_r, sft, lambda_or=0.1)
assert L.item() > sft.item()                        # OR 项恒为正，总 loss 一定比单纯 SFT loss 大

or_c = log_odds(log_p_c)
or_r = log_odds(log_p_r)
assert bool((or_c > or_r).all())                     # chosen 的 log_odds 全部大于 rejected
```

实测:`L_ORPO = 1.113003`(`sft` 单独为 `1.100000`,OR 项贡献了 `0.013003`);`log_odds(chosen) = [0.7096, 0.1959, 1.0502, 0.4328]`,`log_odds(rejected) = [-1.2475, -1.6193, -0.8416, -1.8546]`,四条样本上 chosen 的 log_odds 都明显高于 rejected。

**面试怎么问 + 追问链:**
- **Q:** "ORPO 怎么在完全不用 reference model 的情况下,还能像 DPO 一样区分 chosen/rejected?"—— 期望说出"用 odds ratio 替代相对 ref 的偏移量,靠 SFT 项锚住 actor"。
- **追问 1:** "odds ratio 和 DPO 的 log_ratio 是数学等价的吗?"—— 期望诚实回答"不是严格等价,是工程上的近似替代",不能不懂装懂说"完全一样"。
- **追问 2(数值细节):** "`log_odds` 里为什么用 `log1p(-p.exp())` 而不是直接 `log(1-p)`?"—— 期望说出"防止 p 接近 1 时 `1-p` 接近 0 导致的数值溢出/精度损失",这是浮点计算的通用技巧。
- **追问 3(开放):** "如果去掉 SFT 项,只留 `log_odds` 那一项,会发生什么?"—— 没有标准答案,考察对"SFT 项锚住 actor"这个机制的理解,合理方向是"actor 分布可能跑偏,log_odds 差值失去意义上的可解释性"。

**常见坑:** 把 `log_odds` 和 `log_ratio` 当成同一个数学对象来推导——两者定义完全不同(前者只用 actor 自己的概率,后者是两个模型的比值),数值上没有可比性。面试时如果被要求"手推一下这两个公式的关系",要能诚实说清楚它们是各自独立定义、经验上起类似作用的两个量,不是同一个东西的两种写法。

---

## 4. `simpo_loss()`(`simpo_minimal.py`,配套 `lectures/05-simpo.md`)—— 长度归一化 + 无 reference model

**是什么:**
```python
def length_normalized_logp(log_p_per_token: torch.Tensor,
                           response_mask: torch.Tensor) -> torch.Tensor:
    """(1/|y|) · sum log π."""

def simpo_loss(
    log_p_chosen: torch.Tensor,        # (B, T_c) per token
    mask_chosen: torch.Tensor,         # (B, T_c)
    log_p_rejected: torch.Tensor,      # (B, T_r) per token
    mask_rejected: torch.Tensor,       # (B, T_r)
    beta: float = 2.5,
    gamma: float = 1.0,
) -> torch.Tensor:
```

**一句话:** SimPO 同样去掉 reference model,但砍的角度和 ORPO 不同——它把"整条回答的 log prob 总和"换成"每个 token 的平均 log prob",再加一个目标 margin `γ`,直接用长度归一化后的分数做 DPO 式的 margin 对比。

**底层机制/为什么这样设计:** DPO/IPO 用的是整条回答的 log prob **总和**(`sum log π`),这个总和天然和长度挂钩——response 越长,只要每个 token 的 log prob 不是特别负,总和的量级就越容易被拉大。如果训练数据里 chosen 回答系统性地比 rejected 更长(常见情况:人类偏好更详细完整的回答),模型有可能学到"变长"这一条捷径。SimPO 把总和换成 `length_normalized_logp = sum(log_p · mask) / sum(mask)`——按真实 token 数(mask 求和,自动排除 padding)取平均,分数变成"平均每个 token 的 log prob",和长度解耦。`margin = β·(r_chosen - r_rejected) - γ`,`γ` 是目标 margin 下限——即使 chosen 已经比 rejected 好,只要差距没达到 `γ`,loss 依然不为 0,逼着优化器把差距拉得更开,而不是"刚好比 0 大一点就停"。和 ORPO 对比:两者都不需要 ref model,但砍掉 ref 的方式不同——ORPO 用 odds 替代"相对偏移"、还留了 SFT 项锚定;SimPO 直接用长度归一化后的 log prob 本身做 reward,连"相对偏移"这个概念都不要了,是这几种变体里实现最简单的一个(源文件只有 49 行,且不带任何 SFT 项)。

**AI 研究场景:** "训练完模型输出变啰嗦"是 DPO 类方法反复被观察到的副作用(配套 lecture Slide 4 给出的经验数字是 chosen 长度训练后↗30%、rejected↘20%,这是 lecture 引用的经验观察,本文数值 demo 单独验证了其中的"归一化消除长度差异"这一层机制,不是这组具体百分比)。如果产品对回答长度有约束(比如客服场景要求简洁),或者评测集对长度做惩罚,SimPO 的长度归一化能直接从损失函数层面消掉这类偏置,不需要额外加长度惩罚项或做后处理截断。

**可运行例子(独立验证,数字与 lecture/`simpo_minimal.py` 自带 smoke test 均不同):**
```python
import torch
import sys
sys.path.insert(0, "learning/dpo-family/src")
from simpo_minimal import simpo_loss, length_normalized_logp

# 构造两条"每 token 质量完全相同"(都是 -1.0/token)但真实长度不同的回答
# response A: 5 个真实 token；response B: 9 个真实 token，其余是 padding（mask=0）
log_p_a = -torch.ones(1, 10)
log_p_b = -torch.ones(1, 10)
mask_a = torch.tensor([[1] * 5 + [0] * 5], dtype=torch.float32)
mask_b = torch.tensor([[1] * 9 + [0] * 1], dtype=torch.float32)

reward_a = length_normalized_logp(log_p_a, mask_a)
reward_b = length_normalized_logp(log_p_b, mask_b)
assert torch.allclose(reward_a, reward_b)                # 长度归一化后：两条 reward 完全相同

raw_a = (log_p_a * mask_a).sum(dim=1)
raw_b = (log_p_b * mask_b).sum(dim=1)
assert raw_a.item() != raw_b.item()                        # 不归一化的话：raw sum 差异巨大

# 真实 loss 计算：chosen 明显更好的场景
log_p_c = torch.tensor([[-0.5, -0.6, -0.4, -0.5, 0, 0, 0, 0],
                        [-0.3, -0.4, -0.3, -0.2, -0.5, -0.6, 0, 0]])
mask_c = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.float32)
log_p_r = torch.full((2, 8), -1.0)
mask_r = torch.ones(2, 8)
L = simpo_loss(log_p_c, mask_c, log_p_r, mask_r, beta=2.5, gamma=1.0)
assert L.item() > 0
```

实测:长度归一化后 `reward_a = reward_b = -1.000000`(完全相同),但不归一化的 raw sum 是 `raw_a = -5.000000` vs `raw_b = -9.000000`(单纯因为长度不同就差了近一倍)。第二段真实 loss 计算里 `reward_chosen = [-0.5000, -0.3833]`,`reward_rejected = [-1.0000, -1.0000]`,`L_SimPO = 0.517244`。

**面试怎么问 + 追问链:**
- **Q:** "SimPO 的 length normalization 具体解决了什么问题?"—— 期望说出"log prob 总和天然偏向长回答,归一化成平均值后和长度解耦"。
- **追问 1:** "`γ` 这个超参是干什么用的,设成 0 会怎样?"—— 期望说出"`γ` 是目标 margin 下限,设成 0 就退化成'chosen 只要比 rejected 好一点点就满足',`γ>0` 强迫模型把差距拉得更明显"。
- **追问 2(和 ORPO 对比,深挖):** "SimPO 和 ORPO 都不需要 ref model,它们砍掉 ref 的方式是同一回事吗?"—— 期望明确说出"不是——ORPO 用 odds 替代相对偏移量、还留了 SFT 项锚定;SimPO 直接用归一化后的 log prob 本身做 reward,不留 SFT 项",能对比着讲清楚说明两个知识点是真正吃透了,而不是笼统归为"反正就是不用 ref"。

**常见坑:** 以为"length normalization"等于"模型对长度完全不敏感"。实际上它只是把"总量"换成"均值",如果两条回答每个 token 的质量确实有差异(不是上面构造的完全相同的极端例子),更长的回答依然可以靠维持较高的平均质量赢得更高 reward——归一化消除的是"单纯靠堆字数占便宜"这一种特定偏置,不是让模型对长度完全无感。

---

## 5. `cpo_loss()`(`cpo_minimal.py`)—— 对比 margin + SFT 项,用 actor 自己顶替 ref

**是什么:**
```python
def cpo_loss(
    log_p_chosen_actor: torch.Tensor,    # sum log π_chosen
    log_p_rejected_actor: torch.Tensor,  # sum log π_rejected
    sft_loss_chosen: torch.Tensor,       # standard NLL on chosen
    beta: float = 0.1,
    lambda_c: float = 0.5,
) -> torch.Tensor:
    """CPO = NLL(chosen) + λ · DPO-style margin (no ref)."""
```

**一句话:** CPO 是这几种变体里最朴素的一个——把 DPO margin 里"减去 ref"这一步直接删掉(相当于让 actor 自己顶替 ref 的角色),再加一个 SFT 项托底,两项加权求和。

**底层机制/为什么这样设计:** 对比 `dpo_loss` 的 margin 计算 `β·((log_p_c_actor - log_p_c_ref) - (log_p_r_actor - log_p_r_ref))`,`cpo_loss` 里的 margin 是 `β·(log_p_chosen_actor - log_p_rejected_actor)`——没有 `log_p_c_ref`/`log_p_r_ref` 这两项,直接用 actor 自己的 log prob 做对比。源码注释"用 actor 自身代替 ref"就是这个意思。单独看这个对比项(没有 SFT 项托底)其实有风险——存在一个平凡解:把 chosen 和 rejected 的绝对概率**一起**往下压,只要相对差距够大,loss 照样能降,不需要 chosen 的绝对质量真的提高。CPO 用 `sft_loss_chosen`(chosen 上的标准 NLL)项防住这个退化解,独立要求"chosen 的绝对概率不能太低"——这和 ORPO 用 SFT 项锚住 actor 是同一个防御思路,只是 CPO 的对比项本身比 ORPO 的 `log_odds` 更直接,连 odds 变换都没有,就是裸的 log prob 差。这也是这批文件里最短的一个(34 行):ORPO/SimPO 都在"删掉 ref"之外各自加了一层设计(odds 变换 / 长度归一化),CPO 没有,是"删掉 ref"这个思路里最不加修饰的版本。

**AI 研究场景:** 已经有一个还不错的 SFT checkpoint,只想用一批 pairwise 偏好数据做"轻量级"进一步微调、又不想为了这一步专门管理一份 ref 模型权重时,CPO 提供了几乎不增加额外工程复杂度的选项——一个 loss 函数、一次前向传播,没有 ORPO 的数值稳定 trick 也没有 SimPO 的 mask 归一化步骤,是这几种 ref-free 方法里实现和调试成本最低的。

**可运行例子(独立验证,数字与 `cpo_minimal.py` 自带 smoke test 不同):**
```python
import inspect
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "learning/dpo-family/src")
from cpo_minimal import cpo_loss

params = list(inspect.signature(cpo_loss).parameters.keys())
assert not any("ref" in p for p in params)          # 结构性证据：没有 ref 参数

log_p_c = torch.tensor([-0.9, -1.1, -0.7, -1.0])
log_p_r = torch.tensor([-2.0, -2.3, -1.8, -2.1])
sft = torch.tensor(1.4)
L = cpo_loss(log_p_c, log_p_r, sft, beta=0.1, lambda_c=0.5)
assert L.item() > sft.item() - 1e-4

# 手动拆开验证：CPO = SFT + lambda_c * (DPO 风格 margin，但没有 ref 项)
margin = 0.1 * (log_p_c - log_p_r)
L_contrast_only = -F.logsigmoid(margin).mean()
L_manual = sft + 0.5 * L_contrast_only
assert torch.allclose(L, L_manual, atol=1e-6)
```

实测:`L_CPO = 1.719240`(`sft` 单独为 `1.400000`),纯对比项 `L_contrast_only = 0.638481`,`0.5 * L_contrast_only = 0.319240`,`1.400000 + 0.319240 = 1.719240`,和 `cpo_loss` 直接算出的结果精确一致。

**面试怎么问 + 追问链:**
- **Q:** "CPO 的对比项和 DPO 的 margin 项有什么区别?"—— 期望说出"CPO 直接用 actor 自己的 log prob 差,没有减去任何 ref 的量"。
- **追问 1(压力测试):** "如果去掉 CPO 里的 SFT 项,只留对比项,训练可能出什么问题?"—— 期望说出"存在平凡解:把 chosen 和 rejected 的绝对概率一起往下压,只要相对差距够大 loss 就会变小,不需要 chosen 真的变好"——这正是"底层机制"里讲的核心风险。
- **追问 2:** "CPO、ORPO、SimPO 都是 ref-free,实现复杂度上有什么差别?"—— 期望能对比说出"CPO 最简单(裸 log prob 差 + SFT),ORPO 多了 odds 变换和数值稳定 trick,SimPO 多了长度归一化",体现三者虽然目标相同(省 ref)但走的是三条不同技术路线。

**常见坑:** 把 CPO 的对比项误认为和 DPO 的 margin 数学等价(只是把 ref 设成 0)——数值上确实可以这样理解,但训练动态不同:DPO 的 `log_ratio` 会随训练把 actor 推离一个**固定**的 ref 快照,CPO 没有这个"锚",纯靠 SFT 项拉住。如果 `lambda_c` 设置不当(对比项权重过高、SFT 权重相对过低),更容易出现上面讲的"一起压低概率"的退化解。

---

## 6. `dpop_loss()`(`dpop_minimal.py`,本节重点)—— chosen 概率下降的反例与 hinge 修复

**是什么:**
```python
def dpop_loss(
    log_p_c_actor: torch.Tensor, log_p_c_ref: torch.Tensor,
    log_p_r_actor: torch.Tensor, log_p_r_ref: torch.Tensor,
    beta: float = 0.1,
    lambda_p: float = 50.0,  # 高权重 hinge
) -> torch.Tensor:
    """DPOP = DPO + λ·max(0, log_p_c_ref - log_p_c_actor)."""
```

**一句话:** DPOP 在标准 DPO loss 上加一个 hinge 惩罚项,专门盯住一种 DPO 会失手的具体反例——chosen 的绝对概率其实在下降,但因为 rejected 掉得更多、相对 margin 依然为正,DPO loss 会误判成"训练在变好"。

**底层机制/为什么这样设计(完整反例,数字全部来自 `.venv` 真实运行):**

DPO loss 只看**相对**margin `h = log_ratio_chosen - log_ratio_rejected`,不直接约束 `log_ratio_chosen` 本身的符号。这意味着存在一种反直觉但完全合法的情况:chosen 的概率(相对 ref)在下降(`log_ratio_chosen < 0`),但只要 rejected 降得比 chosen 更多,`h` 依然是正的,DPO loss 依然会比"没有偏好信息"的基线(`log 2 ≈ 0.6931`)更小——也就是说 DPO 会把这种情况判定成"进步"。

用仓库 `dpop_minimal.py::demo_chosen_prob_drop()` 里的反例数字,逐步手工核对:

- actor 把 chosen 的概率从 ref 的 0.5 压到了 0.3:`log_ratio_chosen = log(0.3) - log(0.5) = -1.203973 - (-0.693147) = -0.510826`(绝对变差)
- actor 把 rejected 的概率从 ref 的 0.4 压到了 0.1:`log_ratio_rejected = log(0.1) - log(0.4) = -2.302585 - (-0.916291) = -1.386294`(变得更差,降幅比 chosen 大)
- 相对 margin `h = log_ratio_chosen - log_ratio_rejected = -0.510826 - (-1.386294) = +0.875469` —— 是正的,DPO"认为"chosen 相对 rejected 变好了
- `L_dpo = -log sigmoid(0.1 × 0.875469) = 0.650331`,比 `log(2) = 0.693147` **更小** —— DPO 判定这是净进步,尽管 chosen 的绝对概率从 0.5 掉到了 0.3(相对下降 40%)

DPOP 加的 hinge 项是 `λ_p · max(0, log_p_chosen_ref - log_p_chosen_actor)`——直接检查 `log_p_chosen_actor` 有没有比 `log_p_chosen_ref` 小(chosen 的绝对 log prob 是不是掉了),掉了就按下降幅度惩罚。在这个反例里,`hinge = relu(log(0.5) - log(0.3)) = relu(0.510826) = 0.510826`,默认权重 `λ_p = 50`(其余变体的超参通常在 0.1~2.5 量级,这个 50 是刻意设的高权重):`L_dpop = L_dpo + 50 × 0.510826 = 0.650331 + 25.541282 = 26.191614`,是 `L_dpo` 的 40 倍还多,彻底压过了 DPO 本身"这是进步"的误判。

仓库 `learning/dpo-family/src/tests/test_six_methods_consistency.py::test_dpop_punishes_chosen_drop` 用的是**同一组**数字(chosen: 0.5→0.3,rejected: 0.4→0.1),断言 `L_dpop.item() > L_dpo.item()`。本文写作时执行了 `pytest learning/dpo-family/src/tests/test_six_methods_consistency.py -v`,10 条测试**全部通过**(含这一条),不是只看代码猜测行为。

**AI 研究场景:** 这个反例不是理论假设,而是 DPO 训练里被观察到的真实失败模式(DPOP 论文的动机)——尤其当 chosen/rejected 两条回答本身比较接近、或数据集存在标注噪声时,优化器完全可能找到"两边概率都往下压,但压 rejected 更多"这条捷径来降低 loss,而不是真正提升 chosen 的生成质量。如果下游用"chosen 的生成概率/困惑度"作为质量代理指标,会观测到这个指标在 DPO 训练中途不升反降,DPOP 的 hinge 项就是专门堵上这个漏洞的。

**可运行例子:**
```python
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "learning/dpo-family/src")
from dpo_minimal import dpo_loss
from dpop_minimal import dpop_loss

# 反例：chosen 概率 0.5→0.3（绝对下降），rejected 概率 0.4→0.1（下降更多）
log_p_c_actor = torch.log(torch.tensor([0.3]))
log_p_c_ref = torch.log(torch.tensor([0.5]))
log_p_r_actor = torch.log(torch.tensor([0.1]))
log_p_r_ref = torch.log(torch.tensor([0.4]))

log_ratio_c = log_p_c_actor - log_p_c_ref
log_ratio_r = log_p_r_actor - log_p_r_ref
assert log_ratio_c.item() < 0                       # chosen 绝对变差了
raw_margin = (log_ratio_c - log_ratio_r).item()
assert raw_margin > 0                                 # 但相对 margin 是正的（chosen 比 rejected 掉得少）

L_dpo = dpo_loss(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref, beta=0.1)
L_dpop = dpop_loss(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref, beta=0.1, lambda_p=50.0)

log2 = torch.log(torch.tensor(2.0))
assert L_dpo.item() < log2.item()                    # DPO 认为这比"完全没有偏好信息"还好 —— 误判
assert L_dpop.item() > L_dpo.item()                   # DPOP 的 hinge 项把这个误判打回去

# 手动拆开验证 DPOP = DPO + lambda_p * hinge
hinge = F.relu(log_p_c_ref - log_p_c_actor).mean()
assert torch.allclose(L_dpop, L_dpo + 50.0 * hinge, atol=1e-5)
```

**面试怎么问 + 追问链:**
- **Q:** "DPO 训练时,chosen 的生成概率有没有可能不升反降?"—— 期望第一反应不是"不可能",而是"有可能,DPO 的 loss 只看相对 margin,不直接约束 chosen 的绝对概率方向"。
- **追问 1:** "能不能构造一个具体的反例?"—— 期望讲出"两条回答概率都下降,但 rejected 降得比 chosen 多,margin 依然是正的"这个结构,不需要精确背数字,但要讲清楚"相对"和"绝对"这两个概念的区别。
- **追问 2(深挖 hinge 设计):** "hinge 项为什么用 `max(0, ·)` 而不是直接加一个线性惩罚 `log_p_c_ref - log_p_c_actor`?"—— 期望说出"`max(0,·)` 只在 chosen 概率真的下降时(这一项为正)才惩罚;chosen 概率上升时这一项本身是负的,`max(0,·)` 把它截断成 0,不会因为'chosen 涨得多'反而倒扣分"——这是 hinge loss 的标准设计模式,不是 DPOP 专属发明。
- **追问 3(工程判断力):** "`λ_p=50` 明显比其他方法的超参大很多,为什么要设这么高?"—— 期望推出"DPO loss 本身的量级通常在 0~1 之间(`-log sigmoid` 的取值范围),hinge 项权重太小的话,遇到这种反例根本压不过 DPO 本身的误判,必须设得足够高才能在数值上真正纠正它",可以直接引用本知识点算出的真实数字(`0.65` vs `26.19`)佐证。

**常见坑:** 只记住"DPOP 修复了 chosen 概率下降的问题"这句话,但说不出具体是"相对 margin 和绝对概率方向可以不一致"这个根本原因——面试官追问"举个例子"时答不上来,说明理解停留在关键词层面。另外容易搞混比较对象:hinge 项惩罚的是 `log_p_c_ref - log_p_c_actor > 0`(actor 给 chosen 的概率比 **ref** 给 chosen 的概率低),不是"chosen 比 rejected 概率低"——后者是 DPO 本身已经在管的事,DPOP 关心的是 chosen 相对**自己 ref 版本**的绝对退化。

---

## 7. 8 种算法横向对比表 —— 参考模型 / 配对数据 / 计算开销

**是什么:** 不是新代码,是对知识点 1-6(加上 01 号文件的 DPO)按三个维度做的横向汇总表。三个维度全部可以从每个 `*_loss` 函数的签名和已验证过的行为直接读出来,不靠印象或转述。

**一句话:** "需不需要 ref model"和"需不需要配对数据"是两个完全独立的设计选择——8 种方法里,只有 ORPO/SimPO/CPO 同时不需要 ref;KTO 只砍了配对,没砍 ref;这一点是本篇最容易被记混的地方。

**底层机制/为什么这样设计:** 三列判断依据分别是:
- **需不需要参考模型** —— 直接看每个 `*_loss` 函数的参数列表里有没有形如 `*_ref` 的参数(结构性证据,来自知识点 1-6 逐个验证过的签名)。
- **需不需要配对(chosen+rejected)数据** —— 看参数列表里是不是同时出现"chosen 侧"和"rejected 侧"两组独立输入。
- **相对计算开销** —— 由前两列推出:需要 ref 就要多跑一次 ref model 的前向传播(不需要反传,但显存里要多驻留一份完整权重);需要配对就要对 chosen、rejected 各跑一次 actor 前向。

| 算法 | 源文件 | 需要 ref model? | 需要配对数据? | 相对计算开销 |
|---|---|---|---|---|
| DPO | `dpo_minimal.py` | 需要(`log_p_chosen_ref`+`log_p_rejected_ref`) | 需要 | actor 前向×2(chosen+rejected)+ ref 前向×2,显存常驻 2 份模型权重 |
| IPO | `ipo_minimal.py` | 需要(`log_p_c_ref`+`log_p_r_ref`) | 需要 | 同 DPO |
| KTO | `kto_minimal.py` | **需要**(`log_p_ref`,常被误认为不需要) | 不需要(单条 `(y, label)`) | actor 前向×1 + ref 前向×1(每条样本各一次,不需要 chosen/rejected 各一次) |
| ORPO | `orpo_minimal.py` | 不需要 | 需要 | 只有 actor 前向×2;SFT 项复用 chosen 那次前向的结果,不额外增加前向次数 |
| SimPO | `simpo_minimal.py` | 不需要 | 需要 | 只有 actor 前向×2;多一步 mask 归一化,计算量可忽略 |
| CPO | `cpo_minimal.py` | 不需要 | 需要 | 只有 actor 前向×2;SFT 项同样复用 chosen 那次前向的结果 |
| DPOP | `dpop_minimal.py` | 需要(`log_p_c_ref`+`log_p_r_ref`) | 需要 | 同 DPO,额外一个 hinge 项(一次减法 + relu,几乎不增加开销) |

关键结论,每一条都能追溯回具体知识点:

1. **"ref-free 三兄弟"是 ORPO/SimPO/CPO,不包括 KTO。** 这是最容易被记错的一点——KTO 常被和 ORPO/SimPO/CPO 归成一类"高效免 ref 方法",但知识点 2 已经用签名验证过 `kto_loss` 明确有 `log_p_ref` 参数。KTO 省的是配对,不是 ref 模型,这是两个独立的设计维度,只是恰好在这三个方法里被一起砍掉了。
2. **配对数据这一列,只有 KTO 是例外。** DPO/IPO/ORPO/SimPO/CPO/DPOP 六个的签名都同时要求 chosen 侧和 rejected 侧输入,只有 `kto_loss` 只吃一条样本 + 一个二元标签(知识点 2)。
3. **显存/计算开销和"要不要 ref"强相关,和"要不要配对"关系相对弱。** 配对只影响 actor 要跑几次前向(1 次 vs 2 次),但 actor 本来就要训练、权重必须常驻显存;是否需要 ref model,决定的是要不要**额外**再常驻一份完整模型权重——这是量级差异更大的开销(知识点 3 引用的 lecture 量级:7B 模型 DPO 约 28GB vs ORPO 约 14GB,差距主要来自这一份 ref 权重)。
4. **SFT 项不是额外的前向开销。** ORPO、CPO 都带了 SFT 项,但知识点 3/5 的源码显示 `sft_loss`/`sft_loss_chosen` 都是从 chosen 那次前向传播的输出直接算出来的 NLL,不需要专门再跑一次前向——"loss 里多了一项"不等于"多了一次前向传播",这个细节只读函数签名看不出来,必须看调用方式才能确认。

**AI 研究场景:** 挑选偏好优化算法时,这张表本身就是决策依据——显存/算力紧张(只塞得下一份模型权重)时,DPO/IPO/DPOP 直接被排除;数据只有单边标签、拿不到成对比较时,只有 KTO 能直接用;两个约束都不紧的情况下,才轮到"生成质量/收敛稳定性"这类需要跑实验才能确定的经验性因素(不是读代码能定论的,本文不做无依据的推荐)。

**可运行例子(把上面表格的前两列机械地跑一遍,而不是凭印象填表):**
```python
import inspect
import sys
sys.path.insert(0, "learning/dpo-family/src")
from dpo_minimal import dpo_loss
from ipo_minimal import ipo_loss
from kto_minimal import kto_loss
from orpo_minimal import orpo_loss
from simpo_minimal import simpo_loss
from cpo_minimal import cpo_loss
from dpop_minimal import dpop_loss

fns = {
    "DPO": dpo_loss, "IPO": ipo_loss, "KTO": kto_loss,
    "ORPO": orpo_loss, "SimPO": simpo_loss, "CPO": cpo_loss, "DPOP": dpop_loss,
}

# 第一列：需要 ref model？—— 签名里是否存在 *_ref 参数
needs_ref = {name: any("ref" in p.lower() for p in inspect.signature(fn).parameters)
             for name, fn in fns.items()}
assert needs_ref == {
    "DPO": True, "IPO": True, "KTO": True,
    "ORPO": False, "SimPO": False, "CPO": False, "DPOP": True,
}
ref_free = {name for name, v in needs_ref.items() if not v}
assert ref_free == {"ORPO", "SimPO", "CPO"}          # KTO 不在其中，这是本知识点的核心结论

# 第二列：需要配对数据？—— 逐个签名读出来的事实（打印在下面，供人工核对）
needs_paired = {
    "DPO": True,    # log_p_chosen_actor/ref + log_p_rejected_actor/ref
    "IPO": True,    # log_p_c_actor/ref + log_p_r_actor/ref
    "KTO": False,   # 只有 log_p_actor/ref + is_desirable，没有第二条"另一侧"输入
    "ORPO": True,   # log_p_chosen + log_p_rejected
    "SimPO": True,  # log_p_chosen(+mask) + log_p_rejected(+mask)
    "CPO": True,    # log_p_chosen_actor + log_p_rejected_actor
    "DPOP": True,   # log_p_c_actor/ref + log_p_r_actor/ref
}
for name, fn in fns.items():
    print(name, list(inspect.signature(fn).parameters.keys()))
assert sum(needs_paired.values()) == 6                 # 8 个方法里只有 KTO 一个例外
assert needs_paired["KTO"] is False
```

**面试怎么问 + 追问链:**
- **Q:** "这 7 种 DPO 变体,哪些不需要 reference model?"—— 期望准确说出"ORPO、SimPO、CPO 三个",而不是把 KTO 也算进去。
- **追问 1(核心陷阱):** "KTO 呢,它不是也不需要配对吗,是不是也不需要 ref?"—— 期望明确纠正"配对"和"ref"是两回事,KTO 只砍了前者。
- **追问 2(深挖开销):** "如果显存只够放一份模型,ORPO/SimPO/CPO 三个怎么选?"—— 没有标准答案,期望候选人能说出"这张表只能回答'能不能训练'这个及格线问题,'哪个训出来效果更好'需要看具体数据规模和任务,不能只凭这张表拍板",体现出对"结构性事实"和"经验性结论"的边界有清醒认知。

**常见坑:** 把这张表当成"效果排名"来用——表格里的三列全部是结构性事实(能从代码签名直接验证),不包含任何"哪个方法效果更好"的判断。哪个算法最终训出来效果好,取决于具体数据规模、数据质量、任务类型,这些是需要跑实验才能回答的经验性问题,不是读源码能得出的结论,本文不在没有实验支撑的前提下做这类推荐。
