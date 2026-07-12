# 01 · DPO 基础与推导深挖(DPO Foundations & Derivation)

> 总览见 [00-roadmap.md](00-roadmap.md)
> DPO(Direct Preference Optimization)是这整个"对齐算法深挖"系列的地基——后面 IPO/KTO/ORPO/SimPO/CPO/DPOP 这些变体,几乎都是在 DPO 这一行 loss 上"改一处公式"得到的(把 sigmoid 换成平方损失、把 reference model 去掉、把 log-prob 换成长度归一化……)。不先把 DPO 本身"reward model 是怎么被代换消掉的"这个核心技巧和它的代码实现搞清楚,后面每一个变体都只是在背"改了哪一行",而不是真的理解"为什么这么改"。

**本文和 `learning/dpo-family/` 的关系:** `paper/guide_01_direct_preference_optimization.md`(426 行)已经把这篇论文的完整数学推导讲得很深;本文知识点 1 会用给大二学生更友好的方式,把这个推导里最核心的一步(reward 怎么被代换、配分函数怎么被消掉)重新讲一遍——这是全系列**唯一一次"重讲推导"**,因为它是后面所有变体共享的地基。[00-roadmap.md](00-roadmap.md) 差异化声明里说的"不重复那份推导",针对的是本系列 02/03 两个文件——那两个文件不会再逐行走一遍论文推导,而是聚焦代码实现和横向对比。

本文所有"可运行例子"代码块已在仓库根目录 `.venv`(torch 2.11.0+cu128,CUDA 可用,`transformers` 5.10.2 / `datasets` 5.0.0)下逐块实际跑通验证,给出的每一个数字(包括 loss 的具体小数、`--help` 的完整参数列表、一次真实小规模训练的打印输出)都是现场跑出来的,不是转述文档或凭经验估算。知识点 4 涉及的 `dpo_minimal.py` 额外做了一次真实的小规模训练验证(2 条样本、CPU、真实加载两份 GPT-2),细节见该节。

**本篇统一结构(与 00-roadmap.md 的知识点结构模板完全一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(带 assert,真在 `.venv` 里跑过)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. 从 RLHF 闭式解到 Bradley-Terry 代换 —— DPO 怎么"干掉"reward model

**是什么:**
```text
第一步 · RLHF 的优化目标(PPO 求解的就是这个目标,PPO 本身怎么求解见 rl-foundations/04-ppo-core.md,这里不重复):
    max_π   E_x[ E_{y~π(·|x)} r(x,y) ]  -  β · KL( π(·|x) ‖ π_ref(·|x) )

第二步 · 这个目标(把 π 看成整个概率分布空间上的自由变量,不是某个具体网络)存在解析解:
    π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
    Z(x) 是配分函数(partition function),保证 π*(·|x) 对所有 y 求和/积分等于 1 —— 这一项通常"难算"

第三步 · 把上式反过来解,用 π* 和 π_ref 表示 reward:
    r(x,y) = β · log( π*(y|x) / π_ref(y|x) )  +  β · log Z(x)

第四步 · 代入 Bradley-Terry 偏好模型 P(y_w ≻ y_l | x) = sigmoid( r(x,y_w) - r(x,y_l) ):
    r(x,y_w) - r(x,y_l)
      = β·log(π*(y_w|x)/π_ref(y_w|x)) + β·log Z(x)
      - β·log(π*(y_l|x)/π_ref(y_l|x)) - β·log Z(x)
      = β·log(π*(y_w|x)/π_ref(y_w|x)) - β·log(π*(y_l|x)/π_ref(y_l|x))        ← β·log Z(x) 精确抵消
```

**一句话:** RLHF 目标的最优策略解 π* 本身就已经编码了 reward 的全部信息,把它反解出来会得到一个"用 π 和 π_ref 表示 reward"的公式;这个公式代入 Bradley-Terry 偏好模型、做一次减法之后,那个原本很难算的配分函数 Z(x) 会精确抵消——于是最终的偏好损失里完全不出现 Z(x),也完全不需要一个独立的 reward 网络。

**底层机制/为什么这样设计:**

这一步最容易被误解的地方,不是"闭式解怎么推出来的"(这是一个标准的 KL 正则化优化问题,论文附录有完整证明,这里不重新推导积分/拉格朗日那一步,直接使用结论),而是**为什么反解出来的 reward 里那个"难算"的 Z(x) 会消失**——很多人会把这一步记成"DPO 假设 Z(x)=1"或者"DPO 把 Z(x) 忽略不算",这两种说法都不对。精确的说法是:

- Z(x) 是**只依赖 prompt x、不依赖具体回答 y** 的一个数(对给定的 x,不管 y 取 y_w 还是 y_l,Z(x) 是同一个值)。
- Bradley-Terry 模型只关心**同一个 x 下**两个回答的 reward **之差**,而不关心 reward 的绝对值。
- 一个只依赖 x 的量,在同一个 x 下做减法,必然精确抵消——这是纯代数事实,不是近似,也不需要真的算出 Z(x) 是多少。

于是,"训练一个 reward model,再用 PPO 去优化这个 reward model 打的分"这件事,被替换成了"直接假设 reward 长成 `β·log(π_θ/π_ref)` 这个形式,把它代进 Bradley-Terry 的负对数似然里,对 π_θ 的参数做梯度下降"。**Reward 这个数学概念完全没有消失**,消失的只是"用一个独立神经网络头去表示 reward"这件事——reward 被重新参数化成了 policy 自己相对 reference 的对数比值,这也是论文标题"Your Language Model is Secretly a Reward Model"的字面意思。

对照 `learning/rlhf-classic/lectures/01-instructgpt.md` 里的三段管线(Stage 1 SFT → Stage 2 RM 训练 → Stage 3 PPO + KL),DPO 绕开的是 **Stage 2**(不再需要单独收集偏好数据去训练一个 reward model)和 **Stage 3 里"用 RL 算法在线求解"这部分**(不需要从当前 policy 采样、不需要 PPO 的 clip/GAE 这套机制)。Stage 3 要优化的 **目标本身**(`max E[r] - β·KL(π‖π_ref)`)DPO 完全保留,只是把"用 RL 迭代求解"换成了"直接在离线偏好对上做一次监督式的梯度下降"。Stage 1(SFT)在 DPO 里也还在,只是换了个身份:SFT 之后的模型通常就是这里的 `π_ref`,同时也是 actor 的初始化起点。PPO 的 clip/重要性采样比率这些"怎么用 RL 算法求解在线目标"的细节,属于 `learning/rl-foundations/04-ppo-core.md` 和 `learning/rlhf-classic/04-ppo-for-llm-deep.md` 的内容,这里不重复。

**AI 研究场景:** 标准 PPO-RLHF 训练要同时维护多个模型(policy/actor、value/critic、reward model、reference model),`learning/dpo-family/lectures/01-dpo.md` Slide 1 把这个现象总结成"4 模型协同 → 显存 4×";DPO 因为把 reward 重新参数化进了 policy 自身,只需要 actor + reference 两份模型权重,不需要 critic,也不需要单独的 reward model,训练循环长得几乎和普通 SFT 一样(forward → loss → backward → step),不需要在线采样/rollout 基础设施。这就是为什么很多资源有限的团队优先选 DPO 类方法而不是完整 PPO-RLHF 做对齐。

**可运行例子:**
```python
import torch

# 玩具设定:某个 prompt x 下只有 3 个可能回复 y1,y2,y3(真实场景 y 的空间是所有可能的
# token 序列,没法真的枚举;这里缩小到 3 个,只是为了能把 Z(x) 具体算出来做验证)
r_true = torch.tensor([2.0, 0.5, -1.0])   # "上帝视角"才知道的真实 reward r(x,y1/y2/y3)
pi_ref = torch.tensor([0.5, 0.3, 0.2])     # reference policy,概率求和必须为 1
assert torch.allclose(pi_ref.sum(), torch.tensor(1.0))
beta = 0.5

# 第一步:闭式最优解 pi_r(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x,y)/beta)
unnormalized = pi_ref * torch.exp(r_true / beta)
Z = unnormalized.sum()                      # partition function —— "难算"说的就是它
pi_r = unnormalized / Z
assert torch.allclose(pi_r.sum(), torch.tensor(1.0))

# 第二步:反解 r_hat(y) = beta*log(pi_r(y)/pi_ref(y)) + beta*log(Z),应该精确还原 r_true
r_hat = beta * torch.log(pi_r / pi_ref) + beta * torch.log(Z)
assert torch.allclose(r_hat, r_true, atol=1e-5)

# 第三步(核心):算"reward 差"时根本不需要知道 Z —— 有 Z 和没 Z 两种算法结果完全一致
diff_with_Z = (r_true[0] - r_true[1]).item()                                          # "诚实"算法:用上帝视角的真实reward
diff_without_Z = (beta * torch.log(pi_r[0] / pi_ref[0])
                   - beta * torch.log(pi_r[1] / pi_ref[1])).item()                     # DPO算法:压根不提Z
assert abs(diff_with_Z - diff_without_Z) < 1e-5
```

**面试怎么问 + 追问链:**
- **Q:** "DPO 号称'跳过了 reward model',它是不是完全不需要 reward 这个概念了?"—— 期望答"不是,reward 的理论框架完整保留,只是不再用一个独立神经网络头表示它,而是重新参数化成 `β·log(π_θ/π_ref)`"。
- **追问 1(本节最容易考倒人的问题):** "配分函数 Z(x) 到底是被设成 1、被近似掉,还是被忽略了?"—— 期望精确答出"都不是,是在计算同一个 x 下两个回答的 reward 差值时代数上精确抵消,因为 Z(x) 只依赖 x 不依赖 y";答成"约等于 1"或"可以忽略"说明只是记住了结论没理解机制。
- **追问 2(深挖,检验是否真理解"抵消"的前提):** "如果 chosen 来自 prompt x1、rejected 碰巧来自另一个不同的 prompt x2,这个抵消还成立吗?"—— 期望答"不成立,Z(x1) 和 Z(x2) 是两个不同的数,没法抵消;这也是为什么 DPO 训练样本必须是同一个 prompt 下的一对 chosen/rejected,不能跨 prompt 比较"。

**常见坑:** 把"绕开 reward model"理解成"DPO 的数学里不再涉及 reward"——不对,reward 的理论定义完全没变,变的只是它不再由一个独立网络参数化。把 DPO 说成"和 RLHF 是两件不相关的事"(guide 里提到的误区之一)也不准确,更精确的说法是:DPO 优化的是和 KL-约束 RLHF **完全同一个理论目标**,只是把"用 RL 迭代求解"换成了"直接对偏好对做监督式梯度下降"。

---

## 2. 最终 DPO loss 公式 ←→ `dpo_minimal.py::dpo_loss` 实现逐项对照

**是什么:**
```python
# learning/dpo-family/src/dpo_minimal.py 第 44-55 行(已核实,变量名为源码原文)
def dpo_loss(
    log_p_chosen_actor: torch.Tensor,
    log_p_chosen_ref: torch.Tensor,
    log_p_rejected_actor: torch.Tensor,
    log_p_rejected_ref: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO loss = -log sigmoid(β · (log_ratio_w - log_ratio_l))."""
    log_ratio_w = log_p_chosen_actor - log_p_chosen_ref
    log_ratio_l = log_p_rejected_actor - log_p_rejected_ref
    margin = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(margin).mean()
```

**一句话:** 公式 `L_DPO = -E[log sigmoid(β·((logπ_θ(y_w)-logπ_ref(y_w)) - (logπ_θ(y_l)-logπ_ref(y_l))))]` 里的每一项,在代码里都是同名含义的一个变量或一步运算——4 个输入张量对应公式里的 4 个 log-prob,`log_ratio_w`/`log_ratio_l` 对应两个隐式 reward,`margin` 对应 reward 差再乘 β,最后一行对应 Bradley-Terry 负对数似然再对 batch 求平均。

**底层机制/为什么这样设计:**

逐项对照表:

| 公式记号 | 代码变量/表达式 | 含义 |
|---|---|---|
| `log π_θ(y_w｜x)` | `log_p_chosen_actor` | 当前训练中的 policy 对 chosen 回答的**序列** log-prob(所有 token 的 log-prob 求和,不是单 token) |
| `log π_ref(y_w｜x)` | `log_p_chosen_ref` | 冻结的 reference policy 对 chosen 回答的序列 log-prob |
| `log π_θ(y_l｜x)` | `log_p_rejected_actor` | 当前 policy 对 rejected 回答的序列 log-prob |
| `log π_ref(y_l｜x)` | `log_p_rejected_ref` | reference policy 对 rejected 回答的序列 log-prob |
| `log(π_θ(y_w)/π_ref(y_w))` | `log_ratio_w = log_p_chosen_actor - log_p_chosen_ref` | chosen 的隐式 reward(除了一个 β 因子),就是知识点 1 推出的 `r_hat` |
| `log(π_θ(y_l)/π_ref(y_l))` | `log_ratio_l = log_p_rejected_actor - log_p_rejected_ref` | rejected 的隐式 reward |
| `β·(reward_w - reward_l)` | `margin = beta * (log_ratio_w - log_ratio_l)` | 两个隐式 reward 的差,乘 β 缩放(知识点 3 详细展开) |
| `-log sigmoid(·)` | `-F.logsigmoid(margin)` | Bradley-Terry 偏好的负对数似然 |
| `E_(x,y_w,y_l)[·]` | `.mean()` | 对 batch 里所有样本取平均 |

唯一一处"看起来是实现细节、实际是工程必要项"的地方是 `F.logsigmoid` 而不是手写 `-torch.log(torch.sigmoid(margin))`——两者数学上完全等价(`-log(sigmoid(x)) = softplus(-x) = log(1+exp(-x))`,这正是 `for_real_dummy/torch-deep-dive/05-loss-functions-and-numerical-stability.md` 讲过的 `BCEWithLogitsLoss` 用的同一套 log-sum-exp/softplus 稳定性技巧在这里的应用,这里不重复推导只指出对应关系),但数值上不等价:当 `margin` 是一个绝对值很大的负数时(训练早期模型明显"反着学"、chosen 的隐式 reward 远小于 rejected 时完全可能出现),朴素写法会先把 `sigmoid` 算到浮点下溢的精确 `0.0`,再对 `0.0` 取 `log` 得到 `-inf`;`F.logsigmoid` 走的是内部数值稳定的实现,同样的输入能给出有限、正确的值。

**AI 研究场景:** 一个 batch 里只要有一个样本的 margin 出现这种极端值,朴素写法算出的 `nan`/`inf` 经过 `.mean()` 会污染整个 batch 的 loss,直接冲垮训练(反向传播全部变成 `nan`)。`F.logsigmoid` 从源头上避免了这个问题——这是"看起来只是省一行代码,实际上是生产可用性必要条件"的典型例子,也是"手写 4 行 DPO loss"这种教学代码和能在真实、噪声更大的数据上稳定训练的代码之间的关键差距之一。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/dpo-family/src")
import torch
import torch.nn.functional as F
from dpo_minimal import dpo_loss

# --- 逐项对照:手写按公式重算一遍,和 dpo_loss() 的返回值必须完全一致 ---
log_p_chosen_actor = torch.tensor([-5.0])     # log π_θ(y_w|x)
log_p_chosen_ref = torch.tensor([-6.0])       # log π_ref(y_w|x)
log_p_rejected_actor = torch.tensor([-7.0])   # log π_θ(y_l|x)
log_p_rejected_ref = torch.tensor([-6.5])     # log π_ref(y_l|x)
beta = 0.1

loss = dpo_loss(log_p_chosen_actor, log_p_chosen_ref,
                 log_p_rejected_actor, log_p_rejected_ref, beta=beta)

log_ratio_w = log_p_chosen_actor - log_p_chosen_ref      # 隐式 reward(chosen)
log_ratio_l = log_p_rejected_actor - log_p_rejected_ref  # 隐式 reward(rejected)
margin = beta * (log_ratio_w - log_ratio_l)
manual_loss = -F.logsigmoid(margin).mean()

assert torch.allclose(loss, manual_loss)
assert abs(loss.item() - 0.6209570765495300) < 1e-6      # 现场实测的精确值,不是编的

# --- 为什么用 F.logsigmoid 而不是手写 -log(sigmoid(x)):数值稳定性 ---
very_negative_margin = torch.tensor([-90.0])   # 训练早期模型严重反着学时,margin 可能出现这种量级
stable = -F.logsigmoid(very_negative_margin)
naive = -torch.log(torch.sigmoid(very_negative_margin))
assert torch.isfinite(stable).all()            # logsigmoid: 有限、正确
assert torch.isinf(naive).all()                # naive写法: sigmoid先下溢到0.0,log(0)=-inf
assert abs(stable.item() - 90.0) < 1e-3         # 对很负的 margin,-log(sigmoid(x)) 约等于 -x
```

**面试怎么问 + 追问链:**
- **Q:** "`dpo_loss` 这个函数的 4 个输入参数,你能一眼说出哪个对应公式里的哪一项吗?"—— 期望能对着公式逐一点出 actor/ref、chosen/rejected 的对应关系,而不是含糊地说"就是几个概率"。
- **追问 1:** "这个实现为什么用 `F.logsigmoid(margin)` 而不是 `-torch.log(torch.sigmoid(margin))`?两者数学上不是完全等价吗?"—— 期望答"数学上等价,数值上不等价,`logsigmoid` 内部对下溢/上溢有专门的稳定实现"。
- **追问 2(现场验证类,深挖):** "margin 是一个很负的数,比如 -90,这两种写法会有什么具体不同的表现?"—— 期望候选人说出"朴素写法里 sigmoid 先下溢成精确的 0.0,再对 0.0 取 log 变成 -inf;`logsigmoid` 对这种情况有专门稳定路径,还能给出接近 90 的有限正确值",加分项是现场推出"`-log(sigmoid(x))` 对很负的 `x` 渐近等于 `-x`"这个近似行为。

**常见坑:** 混淆 chosen 和 rejected 的传参顺序——4 个参数都是同类型的张量,函数签名本身没有任何机制能防止你传反;传反之后代码不会报错,只会安静地训出一个"倒着学"的模型(margin 符号整体翻转,相当于在鼓励模型学 rejected、压低 chosen),只能靠观察 loss/margin 的训练趋势"看起来不对劲"才能发现,不是靠类型检查能拦住的错误。另一个坑是以为 `.mean()` 换成 `.sum()` 不影响什么——`.sum()` 会让 loss 数值随 batch size 线性变化,进而等价于放大了有效学习率,是不少"换了 batch size 效果差很多"问题的根源之一。

---

## 3. β 超参的作用 —— 控制"允许偏离参考模型多远"

**是什么:**
```python
def dpo_loss(..., beta: float = 0.1) -> torch.Tensor:
    ...
    margin = beta * (log_ratio_w - log_ratio_l)   # β 在这里,是送进 sigmoid 之前的缩放系数
    return -F.logsigmoid(margin).mean()
```

**一句话:** 这里的 `beta` 就是知识点 1 里 RLHF 目标 `max E[r] - β·KL(π‖π_ref)` 中同一个 β——它同时控制两件事:理论上,KL 项能容忍多大的策略偏移;实现上,loss 对同样大小的 margin 有多敏感;这两者其实是同一件事的两个描述角度。

**底层机制/为什么这样设计:**

纯从代码角度看,`beta` 是乘在 `margin` 外面、送进 `sigmoid` 之前的缩放系数。`sigmoid` 在输入接近 0 时近似线性、远离 0 时迅速饱和到 0 或 1。同一个未缩放的 reward 差 `(log_ratio_w - log_ratio_l)`:

- 乘上**小** `beta`,送进 sigmoid 的值仍然接近 0 —— loss 对这个 margin 不敏感,梯度小,模型更新慢、更贴近 `π_ref`(这正对应理论上"KL 约束很强,不允许偏离 reference 太远")。
- 乘上**大** `beta`,送进 sigmoid 的值迅速进入饱和区 —— loss 对 margin 极度敏感,一点点相对偏好优势就会被放大成接近 0 的 loss(和巨大的梯度),模型被推得离 `π_ref` 很远(对应理论上"KL 约束很弱")。

两个边界情况(已现场验证,见下面代码):

- **`beta = 0`**:不管 log-prob 是什么,`margin` 恒为 0,loss 恒等于 `log(2)`,而且梯度精确为 0。这精确对应理论上"KL 约束强度趋于无穷"的极限——模型完全不允许偏离 reference,自然学不到任何东西。
- **`beta` 很大**(实测 `beta=1000`,哪怕 margin 只有 0.01 这么小):loss 迅速逼近 0,几乎只看 margin 的**符号**、不看**大小**,退化成一个"硬边界"分类器。这时候 KL 约束形同虚设,`learning/dpo-family/lectures/01-dpo.md` Slide 9 的表格把这总结成"β > 1.0 时,DPO 几乎等价 BT-fitting,KL 不再起作用"。

**AI 研究场景:** 实践中 β 的典型搜索范围是 0.01–0.5,`0.1` 是 DPO 论文和 trl 的默认值。β 和数据质量会相互作用:偏好标注存在噪声时,大 β 会把噪声样本的错误偏好信号也一起放大,这是"β 调太大,margin 涨得飞快、loss 看起来降得很好,但模型实际输出质量变差"这个经典现象的根源之一——单看 loss/margin 数值不能替代实际检查生成样本质量。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/dpo-family/src")
import math
import torch
from dpo_minimal import dpo_loss

# 固定 margin(log_ratio_w - log_ratio_l = 1.0,chosen 略占优),只变 beta
args_pos = dict(
    log_p_chosen_actor=torch.tensor([1.0]),
    log_p_chosen_ref=torch.tensor([0.0]),
    log_p_rejected_actor=torch.tensor([0.0]),
    log_p_rejected_ref=torch.tensor([0.0]),
)
losses_pos = [dpo_loss(beta=b, **args_pos).item() for b in [0.01, 0.1, 0.5, 1.0, 5.0]]
assert losses_pos == sorted(losses_pos, reverse=True)     # beta 越大,loss 越小(奖励被放大)

# 同样的 margin,方向反过来(chosen 略占劣势),只变 beta
args_neg = dict(
    log_p_chosen_actor=torch.tensor([-1.0]),
    log_p_chosen_ref=torch.tensor([0.0]),
    log_p_rejected_actor=torch.tensor([0.0]),
    log_p_rejected_ref=torch.tensor([0.0]),
)
losses_neg = [dpo_loss(beta=b, **args_neg).item() for b in [0.01, 0.1, 0.5, 1.0, 5.0]]
assert losses_neg == sorted(losses_neg)                    # beta 越大,loss 越大(惩罚被放大)

# 边界一:beta=0 → margin 恒为 0,loss 恒为 log(2),且梯度精确消失
g_args = dict(
    log_p_chosen_actor=torch.tensor([5.0], requires_grad=True),
    log_p_chosen_ref=torch.tensor([0.0]),
    log_p_rejected_actor=torch.tensor([0.0], requires_grad=True),
    log_p_rejected_ref=torch.tensor([0.0]),
)
L0 = dpo_loss(beta=0.0, **g_args)
assert abs(L0.item() - math.log(2)) < 1e-6
L0.backward()
assert g_args["log_p_chosen_actor"].grad.item() == 0.0     # beta=0 时,不管log-prob怎么变,梯度都是0

# 边界二:beta 很大 → loss 只看 margin 的符号,不看大小(饱和成"硬边界"分类器)
tiny_margin_args = dict(
    log_p_chosen_actor=torch.tensor([0.01]),    # 极小的正 margin
    log_p_chosen_ref=torch.tensor([0.0]),
    log_p_rejected_actor=torch.tensor([0.0]),
    log_p_rejected_ref=torch.tensor([0.0]),
)
loss_beta1 = dpo_loss(beta=1.0, **tiny_margin_args).item()
loss_beta1000 = dpo_loss(beta=1000.0, **tiny_margin_args).item()
assert loss_beta1000 < 0.001 < loss_beta1     # beta=1000 时,哪怕 margin 只有 0.01,loss 也几乎为 0
```

**面试怎么问 + 追问链:**
- **Q:** "DPO 里的 β,和 RLHF/PPO 里 KL 惩罚系数的 β,是同一个东西吗?"—— 期望答"是同一个,DPO 的推导本身就是从 `max E[r] - β·KL(π‖π_ref)` 这个目标出发的,β 的理论含义完全没变,只是求解方式从 RL 变成了监督式分类"。
- **追问 1:** "如果把 β 设成 0,DPO 会训练出什么结果?"—— 期望答"loss 恒为 `log(2)`,margin 项被乘 0 抹掉,梯度消失,模型完全不会更新,等价于'绝对不允许偏离 reference model'"。
- **追问 2(深挖,考察是否会盲信 loss 数值):** "β 太大会有什么风险,实践中你会观察到什么现象?"—— 期望答"loss/margin 看起来涨得非常快、很快逼近极值,给人一种'训练效果很好'的错觉,但可能只是在过拟合偏好标签的顺序、偏离 reference 太远导致语言质量或多样性下降,需要结合实际生成样本质量判断,不能只看 loss 数值"。

**常见坑:** 认为 β 越大越好(loss 数值降得快看起来"学习效果好",实际上可能只是过拟合偏好标签、偏离参考模型太远导致语言质量下降,这是 `paper/guide_01_direct_preference_optimization.md` 明确点出的误区之一)。另一个坑是把 β 和学习率 `lr` 混着调、分不清是谁导致的"margin 异常飞涨"——两者都会造成类似的数值剧烈变化,需要分别做消融(固定 lr 只调 β,再固定 β 只调 lr)才能确定是哪个参数导致的。

---

## 4. `dpo_minimal.py` —— 全系列(8 个 PO 算法 + RainbowPO 共 9 个文件)唯一真训练脚本

**是什么:**
```text
# learning/dpo-family/src/dpo_minimal.py 开头的模块 docstring(原文):
"""手写 DPO loss — 4 行核心实现.

DPO loss = -log sigmoid( β · (log_ratio_chosen - log_ratio_rejected) )
其中 log_ratio = log π_θ(y|x) - log π_ref(y|x)

运行：
    python learning/dpo-family/src/dpo_minimal.py
"""
```
真实的命令行参数(现场跑 `--help` 实测输出,不是凭记忆写的):
```text
$ python learning/dpo-family/src/dpo_minimal.py --help
usage: dpo_minimal.py [-h] [--model MODEL] [--n-train N_TRAIN]
                      [--epochs EPOCHS] [--lr LR] [--beta BETA]
                      [--max-length MAX_LENGTH] [--log-interval LOG_INTERVAL]
                      [--cpu]

options:
  -h, --help            show this help message and exit
  --model MODEL
  --n-train N_TRAIN
  --epochs EPOCHS
  --lr LR
  --beta BETA
  --max-length MAX_LENGTH
  --log-interval LOG_INTERVAL
  --cpu
```
默认值(读源码 `main()` 确认):`--model gpt2`、`--n-train 200`、`--epochs 1`、`--lr 5e-7`、`--beta 0.1`、`--max-length 256`、`--log-interval 20`。

**一句话:** 这是 `learning/dpo-family/src/` 9 个 `*.py` 文件里唯一一个会真实加载两份 GPT-2 模型权重(actor 一份、frozen reference 一份)、尝试真实下载 `Anthropic/hh-rlhf` 偏好数据集(失败则降级到内置的 dummy pair,但仍然是真训练)、跑真实 forward + backward + `optimizer.step()` 的脚本;其余 8 个都是不加载任何模型、纯张量级的数值 demo。

**底层机制/为什么这样设计:**

读 `train(args)` 函数(第 58-126 行)确认的关键设计点:

- **两份模型、一份冻结**:`actor = AutoModelForCausalLM.from_pretrained(args.model)`,`ref` 同样加载一次;`ref.eval()` + `for p in ref.parameters(): p.requires_grad = False` 双重冻结;`opt = torch.optim.AdamW(actor.parameters(), lr=args.lr)` 只把 `actor` 的参数交给优化器——这直接对应知识点 1 里"`π_ref` 必须是固定锚点"的理论要求。ref 的前向还额外包在 `with torch.no_grad():` 里(第 102-108 行),不只是不更新参数,连计算图都不构建,省显存。
- **数据集加载失败会优雅降级,但降级后仍是真训练**:`try: ds = load_dataset("Anthropic/hh-rlhf", ...) except Exception: ds = [{"chosen": "answer A", "rejected": "answer B"} for _ in range(args.n_train)]`(第 74-80 行)。不管走哪条分支,后面的 tokenize → forward → `dpo_loss` → `backward` → `opt.step()` 是完全相同的真实流程——降级只换了"训练数据的来源",不是伪装成功的假训练。本机实测 `Anthropic/hh-rlhf` 已在本地 HF 缓存中,字段确认是 `["chosen", "rejected"]`,和源码 `ex["chosen"]`/`ex["rejected"]` 的取值方式完全对得上。
- **一个诚实的简化,值得点出**:`resp_mask_c = torch.ones_like(chosen.input_ids[:, 1:], dtype=torch.float32)`(第 93 行)——这个 mask 全部是 1,也就是说 `get_log_probs_for_labels` 把 **整段文本**(`ex["chosen"]` 本身就是 `"\n\nHuman: ...\n\nAssistant: ..."` 这样的完整对话,prompt 和回答没有分开存)的 log-prob 全部加总,并不是只统计 Assistant 回复那一段。这和 `lectures/01-dpo.md` Slide 20 强调的"log_probs masking 必须正确(仅 response 段计数)"这条工程要求并不完全一致——是这个"minimal"教学脚本为了让 4 行核心 loss 逻辑保持简单做出的取舍,不是一个需要惊慌的 bug,但如果照着这份代码去改造成生产可用的 DPO 训练脚本,这是第一处需要修的地方。
- **没有真正的 batching**:`for ex in ds:` 是逐条样本的 Python 循环,每次只 tokenize 一条 chosen/一条 rejected,不是把多条样本 pad 到一起做张量批处理——工程上等价于 batch size = 1,GPU 并行度用不满,这是"minimal 教学脚本"和"生产训练脚本"的另一处差距。
- **训练循环里有梯度裁剪**:`nn.utils.clip_grad_norm_(actor.parameters(), 1.0)`(第 114 行),在 `loss.backward()` 之后、`opt.step()` 之前,防止某个异常 batch 的梯度过大冲垮训练。
- **一处值得指出的文档漂移**:`lectures/01-dpo.md` Slide 25 的"实战入口"里写了 `python learning/dpo-family/src/dpo_trl.py`,但现场核实 `learning/dpo-family/src/` 目录下**并没有 `dpo_trl.py` 这个文件**——只有 `dpo_minimal.py`。这是 lecture 文档相对代码现状的一处过时引用,不是本文编造的,读者如果照着 Slide 25 敲命令会遇到 `FileNotFoundError`。

**AI 研究场景:** 这是"从 4 行 loss 公式到真实训练"最短的验证路径——不需要自己写训练循环、tokenize 逻辑,直接改几个 `--n-train`/`--epochs`/`--max-length` 参数就能在本地做一次极小规模的 smoke test,确认自己对 loss 实现的理解(知识点 2、3)真的能在一个会更新参数的真实模型上跑起来,而不只是停留在手写张量的层面。

**可运行例子(可选进阶验证,不要求一定要跑一次完整训练):**

首先确认 `--help` 真实可用(上面"是什么"里的输出就是这条命令现场跑出来的)。如果愿意花几分钟,可以再跑一次真正的、样本数极少的训练:

```bash
python learning/dpo-family/src/dpo_minimal.py --n-train 2 --epochs 1 --log-interval 1 --max-length 32 --cpu
```

本机实测输出(CPU,2 条真实 `Anthropic/hh-rlhf` 样本,真实加载两份 `gpt2`):
```text
Step     1 | loss 0.6931 | margin +0.000
Step     2 | loss 0.6931 | margin +0.000
== Epoch 1 done ==
```

这个输出本身就是知识点 5 的一次"实战版"印证:`actor` 和 `ref` 是从**同一个** `gpt2` checkpoint 分别独立加载的两份权重,第一步 `opt.step()` 更新 `actor` 之前,两者对同一输入给出完全相同的 log-prob,`margin` 精确为 0,`loss` 因此精确等于 `log(2)≈0.6931`——和知识点 5 的 pytest 断言是同一个数学事实,只是这里是从一次真实模型前向里观察到的,不是手写张量算出来的。第二行数值没变,不代表没有训练:`--lr` 默认 `5e-7` 是特意调得很小的(`lectures/01-dpo.md` Slide 10 备注"比 PPO 小 10×"),两条样本、一步更新后模型变化幅度在打印的小数位数下还看不出来,属于预期行为。

**面试怎么问 + 追问链:**
- **Q:** "这个脚本 `loss.backward()` 的时候,`ref` 模型会不会被更新?"—— 期望答"不会,`ref.eval()` + 所有参数 `requires_grad=False` 双重冻结,而且 `ref` 的前向包在 `torch.no_grad()` 里,连计算图都不会构建"。
- **追问 1:** "如果不小心把 `ref.parameters()` 也传进了 optimizer,会发生什么?"—— 期望答"`ref` 会跟着被更新,不再是固定锚点,KL 约束'不能离参考模型太远'这个理论意义就丢了,`ref` 和 `actor` 可能一起漂移甚至坍缩到同一个退化解"。
- **追问 2(需要真的读过源码才能答上来,深挖):** "这个脚本的 `response_mask` 是怎么构造的,真的只统计了 response 部分的 log-prob 吗?"—— 期望候选人能看出 `resp_mask_c = torch.ones_like(...)` 是全 1,把 `"Human: ...\nAssistant: ..."` 整段文本都算进了 log-prob 求和,并不是"标准正确"实现里应该做的"只 mask 出 Assistant 回复部分"——这是一个必须读源码才能发现的细节,考察的是候选人会不会盲信函数名字(`get_log_probs_for_labels` 听起来很正规)而不去看它实际怎么被调用。

**常见坑:** 在没有 GPU 或不确定 CUDA 是否可用的机器上跑,忘记加 `--cpu`(脚本本身的设备选择逻辑是 `"cuda" if torch.cuda.is_available() and not args.cpu else "cpu"`,不会自动优雅降级到"没 CUDA 就用 CPU 之外"的情况,但如果 CUDA 环境本身有问题,不加 `--cpu` 可能报错而不是静默退化)。另一个坑是直接跑默认参数 `--n-train 200 --epochs 1`(不加 `--n-train`/`--max-length` 覆盖),在 CPU 上会明显变慢,不是"秒级 demo"——验证脚本能跑通,几条样本、小 `--max-length` 就足够,没必要跑完整默认配置。最后,数据集加载失败切换到 dummy pair 时脚本只会打印一行 `⚠️ HH 加载失败，用 dummy` 提示,不会报错退出,容易被忽略掉"其实这次训练用的不是真实偏好数据"这个事实。

---

## 5. pytest 断言:零 margin 时 loss 精确等于 `log(2)`

**是什么:**
```python
# learning/dpo-family/src/tests/test_dpo_loss_equivalence.py 第 17-26 行(原文)
def test_dpo_loss_zero_margin():
    """log_ratio_w == log_ratio_l → loss = -log 0.5 = log 2."""
    L = dpo_loss(
        log_p_chosen_actor=torch.tensor([0.0]),
        log_p_chosen_ref=torch.tensor([0.0]),
        log_p_rejected_actor=torch.tensor([0.0]),
        log_p_rejected_ref=torch.tensor([0.0]),
        beta=0.1,
    ).item()
    assert abs(L - 0.6931) < 1e-3, L
```

**一句话:** 当 chosen 和 rejected 的隐式 reward 完全相等(`margin=0`,模型对两者"毫无偏好")时,`sigmoid(0)=0.5`,`-log(0.5)=log(2)≈0.6931`——这是验证任何"`-log sigmoid(·)`"形式的 loss 实现是否正确最基础的一条边界检查,原理和检查一个多分类模型"训练刚开始、还没学到任何东西时 loss 应该约等于 `log(类别数)`"是同一类"无信息时的期望 loss"思路。

**底层机制/为什么这样设计:**

这条断言只依赖两个事实:(1)`margin = beta * (log_ratio_w - log_ratio_l)`,当 `log_ratio_w == log_ratio_l` 时 `margin` 精确为 0,和 `beta` 取值无关(乘的是 0);(2)`sigmoid(0) = 0.5` 是一个精确值,不是近似。所以 `-log(sigmoid(0)) = -log(0.5) = log(2)` 是一个可以精确算到浮点精度极限的数字,不是"大概应该是这个量级"的估计——这也是为什么它适合作为单元测试里的第一道边界检查:数值是确定的,任何实现只要在这个特殊输入下偏离 `log(2)` 超过浮点误差,就一定是实现错了(比如把 `-` 号写丢、`beta` 乘错了位置、用了 `sum()` 而不是这个单样本场景下等价的写法等)。

**AI 研究场景:** 知识点 4 的真实训练输出(`Step 1 | loss 0.6931 | margin +0.000`)就是这条单元测试在真实模型上的"活证据"——`actor` 和 `ref` 初始化自同一个 checkpoint 时,第一步的 loss 理应精确落在 `log(2)` 附近。这也是实践中一个有用的训练前 sanity check:如果一次 DPO 训练**第一步**的 loss 明显不是 `log(2)`(比如离谱地大或离谱地小),大概率说明 `ref` 没有真的冻结、`ref` 加载的不是和 `actor` 一致的初始权重,或者 actor/ref 两条前向的 mask/tokenize 逻辑不一致导致 log-prob 没法直接比较。

**可运行例子:**
```python
import sys
import math
sys.path.insert(0, "learning/dpo-family/src")
import torch
from dpo_minimal import dpo_loss

# 复现 src/tests/test_dpo_loss_equivalence.py::test_dpo_loss_zero_margin
# 零 margin:actor 和 ref 对 chosen/rejected 给出完全相同的 log-prob
# (典型场景:训练刚开始,actor 就是 ref 的一份拷贝,两者对同一输入算出的log-prob理应相等)
L = dpo_loss(
    log_p_chosen_actor=torch.tensor([0.0]),
    log_p_chosen_ref=torch.tensor([0.0]),
    log_p_rejected_actor=torch.tensor([0.0]),
    log_p_rejected_ref=torch.tensor([0.0]),
    beta=0.1,
).item()
assert abs(L - math.log(2)) < 1e-6                 # 精确等于 ln(2) ≈ 0.6931,不是约等于
assert abs(L - 0.6931) < 1e-3                       # 原测试文件里用的容差写法,同样成立

# 额外验证:零 margin 时,不管 beta 取多少,loss 都精确是 log(2)
# 因为 margin = beta * (log_ratio_w - log_ratio_l),log_ratio_w == log_ratio_l 时
# beta 乘的是 0,不管 beta 多大,结果都还是 0
for beta in [0.01, 0.1, 1.0, 10.0]:
    L_b = dpo_loss(
        log_p_chosen_actor=torch.tensor([3.0]),
        log_p_chosen_ref=torch.tensor([3.0]),
        log_p_rejected_actor=torch.tensor([-7.0]),
        log_p_rejected_ref=torch.tensor([-7.0]),
        beta=beta,
    ).item()
    assert abs(L_b - math.log(2)) < 1e-6
```

`pytest learning/dpo-family/src/tests/test_dpo_loss_equivalence.py -v` 现场实测 4 条全部通过(`test_dpo_loss_zero_margin`/`test_dpo_loss_positive_margin`/`test_dpo_loss_beta_effect`/`test_dpo_loss_negative_margin`),零 margin 只是这个文件里最基础的一条,同文件里另外 3 条分别检查了"正 margin 时 loss 明显小于 log(2)""β 变大会放大 margin 效应""负 margin 时 loss 明显大于 log(2)"这 3 种更全面的行为,合起来才是一份相对完整的回归测试。

**面试怎么问 + 追问链:**
- **Q:** "为什么 DPO loss 在 margin=0 时应该精确等于 0.6931(即 `ln2`)?"—— 期望能推出 `sigmoid(0)=0.5` → `-log(0.5)=log(2)`,而不是只背这个数字。
- **追问 1:** "这个 `log(2)` 的值和 β 的取值有关系吗?"—— 期望答"没有关系,因为 `margin = beta * 0 = 0` 对任意 `beta` 都成立,`sigmoid(0)` 恒为 0.5"(上面代码已现场验证 β 取 0.01/0.1/1.0/10.0 结果完全一致)。
- **追问 2(连接工程实践):** "在一次真实的 DPO 训练里,你会期望第一步的 loss 大概是多少?如果明显不是这个数,说明什么?"—— 期望连回知识点 4 的真实训练输出,答出"如果 actor 和 ref 是同一个 checkpoint 初始化的,第一步 loss 理应接近 `log(2)`;如果不是,要去检查 ref 是不是真的冻结了、两边 tokenize/mask 是否一致"。

**常见坑:** 用 `==` 做浮点数精确比较而不是带容差的比较——原测试文件用的是 `abs(L - 0.6931) < 1e-3`,不是 `L == 0.6931`,这是浮点数比较的基本纪律(`for_real_dummy/torch-deep-dive` 系列里反复强调过同一个原则)。另一个坑是把"零 margin 这一条测试通过"当成"DPO loss 实现完全正确"的证明——它只验证了一个特殊边界点,同文件里正 margin、负 margin、β 单调性那 3 条测试合在一起才覆盖了更完整的行为空间,单独一条边界测试通过不代表实现在一般情况下也是对的。

---
