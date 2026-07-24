# 05 · 手把手实战:从零搭一个迷你 DPO 训练循环

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 05 个"知识点",不计入"约 15 个知识点"的统计——是仓库"教程体"内容形态在本系列的第一次落地,格式参照 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 首次验证过的模式:01-03 号文件里,你在读"公式和代码逐项对照"或者"变体之间具体差在哪一行公式";04 号文件里,你在旁观一条多级追问链是怎么被面试官一步步逼问出来的。这一篇不一样——你是**动手的人**:从一个空文件开始,一步步敲代码,每写一段就跑一次、亲眼看到真实数字变化,最后独立组装出一个真实能跑起来的迷你 DPO 训练循环。

## 这篇教程要补的是哪一块空白

01 号文件知识点 2 已经把 `dpo_loss` 这个函数和公式逐项对照过;知识点 4 已经把 `dpo_minimal.py` 怎么真实训练两份 GPT-2 讲得很细。但这两个知识点讲的是同一件事的两端:**loss 公式本身**(输入是 4 个已经算好的 log-prob)、和**一整条从真实文本到 log-prob 的完整链路**(tokenize → GPT-2 forward → log_softmax → gather → mask → sum,`get_log_probs_for_labels` 一个函数干了 5 件事)。中间缺的一环是:**log-prob 到底是"怎么从一个可训练的参数里长出来的"**——这一步在 01-04 号文件里始终是给定的,要么直接是手写的张量(01 号文件知识点 2/3 的例子),要么来自一次真实但复杂的 GPT-2 forward(知识点 4)。这篇教程把这条链路压缩成能想象出来的最小版本:一个 5x8 的矩阵(40 个可训练标量)模拟"policy 对 5 个 toy prompt、每个 prompt 8 个候选回复"的 logits,`log_softmax` 一下就是 log-prob——足够小,几秒内跑完;又足够真实,确实是"参数 → logits → log_softmax → log-prob → dpo_loss → backward → 参数更新"这条完整链路,不是伪装的。

**和 `learning/dpo-family/src/capstone_dpo_comparison.py` 的关系(必须先说清楚,避免重复):** 03 号文件知识点 2 已经讲过这个 capstone 脚本——它也有一个"多轮 mock 训练循环"(`mock_step`),但简化方式和这篇教程不同:capstone 直接把 `log_p_c_a`/`log_p_r_a`(chosen/rejected 的 log-prob 本身)当成可训练的自由参数,每步直接对 log-prob 求梯度、更新 log-prob。这样做的好处是省事,但也跳过了一个真实模型里必然存在的结构:log-prob 不是自由变量,它是 `log_softmax(logits)` 的输出,同一个 prompt 下所有候选回复的 log-prob 必须相互牵制(概率和为 1)。这篇教程刻意多走一步,从 logits 矩阵出发而不是直接摆弄 log-prob——阶段 2 会看到这一步差异带来一个真实的、意料之外的发现。

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 从一个 policy logits 矩阵算出 chosen/rejected 的 log-prob,喂进 01 号文件的 `dpo_loss` 公式,得到一个真实的 loss 数字 | [01-dpo-foundations.md 知识点 2](01-dpo-foundations.md) dpo_loss 公式与代码逐项对照、[01-dpo-foundations.md 知识点 5](01-dpo-foundations.md) margin=0 时 loss 精确等于 log(2) |
| 阶段 2 | 一次真实的 `backward()` 拿到梯度,手写一次参数更新,亲眼看到 loss 下降 | [01-dpo-foundations.md 知识点 1](01-dpo-foundations.md) Z(x) 精确抵消的代数结构(这里会遇到它的一个变体)、[00-roadmap.md](00-roadmap.md) "全系列手写张量级实现,zero trl import"的架构选择 |
| 阶段 3 | 把"一步"包装成循环,跑多轮,现场观察 margin 逐轮真实拉开 | [01-dpo-foundations.md 知识点 3](01-dpo-foundations.md) β 如何缩放 margin 的敏感度、[01-dpo-foundations.md 知识点 4](01-dpo-foundations.md) 真实训练里 margin 随步数变化的现象 |
| 阶段 4 | 把训练循环包装成一个可复用的小训练器,验证每一条偏好对(不只是平均值)都真实被拉开,且 ref 全程没被更新过一次 | [01-dpo-foundations.md 知识点 4](01-dpo-foundations.md) `dpo_minimal.py` "两份模型、一份冻结、只有 actor 进优化器"的架构 |

每个阶段的代码都能独立运行。本文件用同目录下的 [_verify_md.py](_verify_md.py) 校验(内容复制自 [dsa-deep-dive/_verify_md.py](../dsa-deep-dive/_verify_md.py),脚本本身是通用工具、和系列内容无关,原样拷贝过来)——校验方式是把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行,块与块之间**不共享任何变量**,所以每个阶段用到前面阶段定义的 `dpo_loss`/`log_probs_for_pairs` 时会重新贴一遍,不是偷懒复制,是这套校验机制要求的。全部代码只用 `torch`(和 00-roadmap.md 环境声明里"全系列零 trl 依赖、全部手写张量级实现"是同一个约定),没有任何 `.cuda()`/`.to(device)` 调用,所有张量默认在 CPU 上,和 00-roadmap.md 环境声明里其余 8 个纯数值 demo 文件同类,不需要真实 GPU。

---

## 阶段 1:从一个 policy logits 矩阵,到一个真实的 loss 数字

先搭 toy 数据集和最小的"policy"。不用真实文本:5 个 prompt 各用一个整数 id 代表(0-4),每个 prompt 假设有 8 个可能的候选回复(也用整数 id 代表,0-7)——可以把它想成"这个 policy 面对某个 prompt,可能的完整回复被压缩成只有 8 种"这么极端简化的一个玩具版本。5 条 toy 偏好对,每条是 `(prompt_id, chosen_id, rejected_id)`——这个三元组的写法本身就把 01 号文件反复强调的"chosen 和 rejected 必须来自同一个 prompt"这条规则刻进了数据结构里,没法写出跨 prompt 的错误偏好对。

```python
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# ---- toy "偏好数据集":5 个 prompt，每个 prompt 有 8 个候选回复，但每条偏好对只挑其中 2 个比较 ----
# 不用真实文本：每个 prompt 用一个整数 id 代表，每个候选回复也用一个整数 id 代表。
# pairs 里的每一项是 (prompt_id, chosen_id, rejected_id) —— 和 01 号文件反复强调的
# "chosen/rejected 必须来自同一个 prompt" 完全对应，这里从数据结构上就不允许写出跨 prompt 的对。
P, R = 5, 8
pairs = [
    (0, 2, 6),
    (1, 5, 1),
    (2, 0, 3),
    (3, 4, 7),
    (4, 1, 5),
]


def dpo_loss(log_p_chosen_actor, log_p_chosen_ref, log_p_rejected_actor, log_p_rejected_ref, beta=0.1):
    """和 01-dpo-foundations.md 知识点 2 逐项对照过的 dpo_minimal.py::dpo_loss 完全一致的实现——
    变量名、公式、F.logsigmoid 数值稳定写法都没有改动。这里重新贴一遍不是重新发明一套公式，
    是因为 _verify_md.py 要求每个代码块独立可运行，不能跨代码块 import 前面定义的函数。"""
    log_ratio_w = log_p_chosen_actor - log_p_chosen_ref
    log_ratio_l = log_p_rejected_actor - log_p_rejected_ref
    margin = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(margin).mean()


def log_probs_for_pairs(logits, pairs):
    """logits: (P, R) 的 policy logits 矩阵，模拟一个真实策略网络最后一层输出的分数。
    对每一行做 log_softmax，就得到这个 prompt 下 R 个候选回复各自的 log-prob——
    这一步是本文和 01-04 号文件的关键差别：01-04 号文件的 log-prob 都是从真实 GPT-2
    forward 或直接手写的张量算出来的，这里额外补上了"logits 矩阵 -> log_softmax -> log-prob"
    这一步，让"policy 网络最后一层长什么样"这件事也变得可见。"""
    log_p = F.log_softmax(logits, dim=-1)
    chosen_lp = torch.stack([log_p[p, c] for p, c, l in pairs])
    rejected_lp = torch.stack([log_p[p, l] for p, c, l in pairs])
    return chosen_lp, rejected_lp


# actor 的 policy logits：这是本教程唯一的可训练参数——一个 5x8=40 个标量的矩阵
theta = torch.randn(P, R, requires_grad=True)
# reference policy：和 01 号文件知识点 1 的身份表完全对应——SFT 模型复制一份，冻结，
# 初始化和 actor 相同（这里直接 detach + clone 出 theta 当前的值）
theta_ref = theta.detach().clone()
theta_ref.requires_grad_(False)

beta = 0.5

chosen_lp_actor, rejected_lp_actor = log_probs_for_pairs(theta, pairs)
with torch.no_grad():
    chosen_lp_ref, rejected_lp_ref = log_probs_for_pairs(theta_ref, pairs)

loss_init = dpo_loss(chosen_lp_actor, chosen_lp_ref, rejected_lp_actor, rejected_lp_ref, beta=beta)
print(f"loss at init (actor == ref): {loss_init.item():.10f}")
print(f"log(2)                    : {math.log(2):.10f}")

# 复现 01 号文件知识点 5 的 sanity check：actor 和 ref 完全一样时，margin 恒为 0，
# loss 必须精确等于 log(2)，和 beta 取值无关——这里 beta=0.5，和 01 号文件用的 beta=0.1 不同，
# 也应该同样成立，用来确认这条 sanity check 不是知识点 5 里 beta=0.1 场景下的巧合。
assert abs(loss_init.item() - math.log(2)) < 1e-5

# 额外验证：手工构造一个"明显更好"和一个"明显更差"的 actor，loss 应该分别明显小于/大于 log(2)
theta_good = theta_ref.clone()
theta_bad = theta_ref.clone()
for p, c, l in pairs:
    theta_good[p, c] += 5.0   # 明显偏向 chosen
    theta_good[p, l] -= 5.0
    theta_bad[p, c] -= 5.0    # 明显偏向 rejected（反着学）
    theta_bad[p, l] += 5.0

chosen_lp_good, rejected_lp_good = log_probs_for_pairs(theta_good, pairs)
chosen_lp_bad, rejected_lp_bad = log_probs_for_pairs(theta_bad, pairs)
loss_good = dpo_loss(chosen_lp_good, chosen_lp_ref, rejected_lp_good, rejected_lp_ref, beta=beta)
loss_bad = dpo_loss(chosen_lp_bad, chosen_lp_ref, rejected_lp_bad, rejected_lp_ref, beta=beta)
print(f"loss with deliberately GOOD actor: {loss_good.item():.6f}")
print(f"loss with deliberately BAD  actor: {loss_bad.item():.6f}")

assert loss_good.item() < math.log(2) < loss_bad.item()
print("stage1 ok: forward loss computed from a logits matrix behaves as expected")
```

`dpo_loss` 函数是从 01 号文件知识点 2 原样搬过来的——变量名(`log_ratio_w`/`log_ratio_l`/`margin`)、公式、`F.logsigmoid` 而不是手写 `-log(sigmoid(...))` 的数值稳定写法,一个字都没改。新增的只有 `log_probs_for_pairs`:给一个 `(P, R)` 的 logits 矩阵按行做 `log_softmax`,再按 `pairs` 里的下标取出 chosen/rejected 各自的 log-prob——这一步就是 `dpo_minimal.py::get_log_probs_for_labels` 在这个玩具设定下的最小版本(那边多出来的复杂度全部来自"文本变长、需要 mask、需要对多个 token 求和"这些真实文本才有的麻烦,这里每个候选回复被压缩成单个"整体单元",不需要再对 token 维度求和)。

第一次跑的时候,`theta` 是 `theta_ref` 的精确拷贝——`log_ratio_w = log_ratio_l = 0`,`margin` 精确为 0,loss 应该精确等于 `log(2)`。这正是 01 号文件知识点 5 那条 pytest 断言(margin=0 时 loss=log(2))在"参数 → logits → log-prob"这条更长链路下的复现,现场实测:`loss at init = 0.6931471825`,`log(2) = 0.6931471806`,差在 1e-9 量级,是浮点噪声,不是系统性偏差——这条断言用 `beta=0.5`(和 01 号文件知识点 5 用的 `beta=0.1` 不同),确认这条 sanity check 不依赖某个特定的 beta 取值。额外做了两个手工构造的极端案例交叉验证方向感:把每条 pair 的 chosen logit 手动加 5、rejected 减 5("明显更好"的 policy),loss 应声跌到 0.006715;反过来("明显更差"),loss 涨到 5.006715——好 policy 的 loss 明显小于 `log(2)`,差 policy 明显大于,方向完全符合直觉。

---

## 阶段 2:一次真实的梯度下降——`backward()` + 手写参数更新

阶段 1 只算了一次 loss,还没有更新任何参数。这一步要做的是最小的一个完整训练单元:一次 `backward()` 拿到真实梯度,手写一行参数更新——这一行代码之后,`theta` 才真的"学习"了一点点。

```python
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)

P, R = 5, 8
pairs = [
    (0, 2, 6),
    (1, 5, 1),
    (2, 0, 3),
    (3, 4, 7),
    (4, 1, 5),
]


def dpo_loss(log_p_chosen_actor, log_p_chosen_ref, log_p_rejected_actor, log_p_rejected_ref, beta=0.1):
    log_ratio_w = log_p_chosen_actor - log_p_chosen_ref
    log_ratio_l = log_p_rejected_actor - log_p_rejected_ref
    margin = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(margin).mean()


def log_probs_for_pairs(logits, pairs):
    log_p = F.log_softmax(logits, dim=-1)
    chosen_lp = torch.stack([log_p[p, c] for p, c, l in pairs])
    rejected_lp = torch.stack([log_p[p, l] for p, c, l in pairs])
    return chosen_lp, rejected_lp


theta = torch.randn(P, R, requires_grad=True)
theta_ref = theta.detach().clone()
theta_ref.requires_grad_(False)
beta = 0.5

chosen_lp_actor, rejected_lp_actor = log_probs_for_pairs(theta, pairs)
with torch.no_grad():
    chosen_lp_ref, rejected_lp_ref = log_probs_for_pairs(theta_ref, pairs)
loss = dpo_loss(chosen_lp_actor, chosen_lp_ref, rejected_lp_actor, rejected_lp_ref, beta=beta)

# ---- 手写一次梯度计算：torch.autograd，和 00-roadmap.md 环境声明里
#      "全系列 9 个 src 文件零 trl 依赖、全部手写张量级实现"用的是同一套工具 ----
loss.backward()

# ref 不应该收到梯度——和 dpo_minimal.py 里 opt = torch.optim.AdamW(actor.parameters(), ...)
# 只把 actor 交给优化器是同一件事：这里没有优化器，但 theta_ref.requires_grad_(False)
# 起的是完全一样的作用
assert theta_ref.grad is None
print("theta_ref.grad is None:", theta_ref.grad is None)

p0, c0, l0 = pairs[0]   # (0, 2, 6)
row0_grad = theta.grad[p0].tolist()
print(f"full gradient row for prompt {p0} (8 candidates): {row0_grad}")

# 反直觉但可以现场验证的事实：R=8 个候选回复共享同一个 log_softmax 归一化，
# 直觉上"提高 chosen 的概率"应该连带牵动同一行里所有其它候选的概率（softmax 是零和的）。
# 但 dpo_loss 只依赖 (log_p_chosen - log_p_rejected) 这一个差值，而 log_softmax 的归一化项
# 只跟 prompt 有关、跟具体候选无关——和 01 号文件知识点 1 里 "Z(x) 只依赖 x，做差时精确抵消"
# 是同一个代数结构，这里精确抵消的是 log_softmax 的归一化常数。真实验证：这一行 8 个位置里，
# 除了 chosen(索引2)/rejected(索引6)，其余 6 个位置的梯度精确为 0，不是"很小"。
for k in range(R):
    if k == c0 or k == l0:
        continue
    assert theta.grad[p0, k].item() == 0.0
# chosen 和 rejected 两个索引的梯度精确互为相反数（同样是上面那条代数事实的推论）
grad_chosen = theta.grad[p0, c0].item()
grad_rejected = theta.grad[p0, l0].item()
assert grad_chosen + grad_rejected == 0.0
print(f"grad[chosen]={grad_chosen:+.8f}  grad[rejected]={grad_rejected:+.8f}  sum={grad_chosen + grad_rejected}")

# ---- 手写一次参数更新（SGD 风格，和 capstone_dpo_comparison.py::mock_step 同款写法）----
lr = 1.0
with torch.no_grad():
    theta -= lr * theta.grad
theta.grad = None   # 手动清零，为下一次 backward 做准备

chosen_lp_actor2, rejected_lp_actor2 = log_probs_for_pairs(theta, pairs)
with torch.no_grad():
    chosen_lp_ref2, rejected_lp_ref2 = log_probs_for_pairs(theta_ref, pairs)
loss_after = dpo_loss(chosen_lp_actor2, chosen_lp_ref2, rejected_lp_actor2, rejected_lp_ref2, beta=beta)

print(f"loss before this one step: {loss.item():.6f}")
print(f"loss after  this one step: {loss_after.item():.6f}")
assert loss_after.item() < loss.item()
print("stage2 ok: one real backward + one manual parameter update makes loss go down")
```

第一个断言(`theta_ref.grad is None`)和真实 `dpo_minimal.py` 里"只把 `actor.parameters()` 交给优化器"是同一件事的另一种实现方式——这里没有优化器,`theta_ref.requires_grad_(False)` 起的是完全一样的隔离作用。

后面几个断言验证的是一个动手之前没有把握、动手之后才敢确认的事实。`theta` 这一行(prompt 0)一共有 8 个候选回复共享同一个 `log_softmax` 归一化——直觉上,"提高 chosen(索引 2)的概率"应该会牵动同一行里其它 7 个候选的概率(`softmax` 输出的和恒为 1,一个上去别的就要让出份额)。真跑出来的完整梯度行是 `[0.0, 0.0, -0.05000000074505806, 0.0, 0.0, 0.0, 0.05000000074505806, 0.0]`——8 个位置里,只有 chosen(索引 2)和 rejected(索引 6)非零,其余 6 个**精确的 0.0**,不是"很小",是浮点精度下的精确 0。

原因和 01 号文件知识点 1 里 `Z(x)` 精确抵消的代数结构是同一类:`dpo_loss` 只依赖 `log_p_chosen - log_p_rejected` 这一个差值,而 `log_softmax` 的归一化项(分母的 `log(Σexp(...))`)对同一行里所有候选都是同一个数——做差的时候精确抵消,不管这一行还有几个候选、它们的值是多少。抵消之后剩下的只是 `theta_c - theta_l` 本身,和这一行其它 6 个位置完全没有关系,梯度自然精确为 0。这是全文第一个"以为会牵一发动全身,实测发现精确无关"的反直觉发现,阶段 4 后面还会看到它的一个延伸后果。`grad[chosen]` 和 `grad[rejected]` 精确互为相反数也不是巧合——两者都来自同一个标量 `sigmoid(margin)-1` 分别乘上 `+1` 和 `-1`(chosen 端和 rejected 端在 `log_ratio_w - log_ratio_l` 里的系数正好相反),浮点上 `x + (-x)` 恒等于精确的 `0.0`,这也是为什么 `assert grad_chosen + grad_rejected == 0.0` 敢用不带容差的精确比较。更新之后 loss 从 0.693147 降到 0.668460,一次真实的下降,不是断言出来的。

**一个容易忽略的坑,值得在这里点破:** 这里手写的 `theta.grad = None` 不是可有可无的一步。PyTorch 的梯度默认是**累加**的——如果不清零,下一次 `backward()` 会把新梯度加到这一次的梯度上面,越滚越大,`torch.optim` 系列的 `optimizer.zero_grad()` 替我们做的就是这一步,手写训练循环时很容易忘掉这一行;忘了不会立刻报错,只会让训练在几步之后开始出现无法解释的数值跳变。

---

## 阶段 3:把"一步"循环起来——现场看着 margin 被训练拉开

阶段 2 只做了一步。把同一件事包进一个 `for` 循环,连续跑 25 轮,每一轮都把真实的 loss 和 margin 打印出来——不是只看开头和结尾,是每一轮都亲眼看到。

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)

P, R = 5, 8
pairs = [
    (0, 2, 6),
    (1, 5, 1),
    (2, 0, 3),
    (3, 4, 7),
    (4, 1, 5),
]


def dpo_loss(log_p_chosen_actor, log_p_chosen_ref, log_p_rejected_actor, log_p_rejected_ref, beta=0.1):
    log_ratio_w = log_p_chosen_actor - log_p_chosen_ref
    log_ratio_l = log_p_rejected_actor - log_p_rejected_ref
    margin = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(margin).mean()


def log_probs_for_pairs(logits, pairs):
    log_p = F.log_softmax(logits, dim=-1)
    chosen_lp = torch.stack([log_p[p, c] for p, c, l in pairs])
    rejected_lp = torch.stack([log_p[p, l] for p, c, l in pairs])
    return chosen_lp, rejected_lp


theta = torch.randn(P, R, requires_grad=True)
theta_ref = theta.detach().clone()
theta_ref.requires_grad_(False)

beta = 0.5
lr = 1.0
n_steps = 25

# 把每一轮的 (loss, margin) 真实存下来，最后的 assert 建立在这份真实记录上，
# 不是只看最后一行打印、也不是手算出来的期望值
history = []
for step in range(1, n_steps + 1):
    chosen_lp_actor, rejected_lp_actor = log_probs_for_pairs(theta, pairs)
    with torch.no_grad():
        chosen_lp_ref, rejected_lp_ref = log_probs_for_pairs(theta_ref, pairs)
    loss = dpo_loss(chosen_lp_actor, chosen_lp_ref, rejected_lp_actor, rejected_lp_ref, beta=beta)
    # margin 用的是 dpo_loss 内部同名变量一样的定义：beta * (log_ratio_w - log_ratio_l)，
    # 和 dpo_minimal.py 真实训练打印的 "margin" 是同一个量
    margin = (beta * ((chosen_lp_actor - chosen_lp_ref) - (rejected_lp_actor - rejected_lp_ref))).mean()

    theta.grad = None
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad

    history.append((step, loss.item(), margin.item()))
    print(f"step {step:3d} | loss {loss.item():.4f} | margin {margin.item():+.4f}")

# chosen/rejected 的 margin 应该随训练逐渐拉开——这里不是只断言"最后比最开始大"，
# 是把全部 25 轮真实记录下来的 margin 逐轮比较，要求严格单调递增
margins = [m for _step, _loss, m in history]
assert margins[0] == 0.0   # 第一轮 actor 和 ref 完全相同，margin 必须精确是 0
assert all(margins[i] < margins[i + 1] for i in range(len(margins) - 1))
assert margins[-1] > 0.9   # 25 轮之后 margin 应该已经明显拉开（真实测得 ~0.92）

losses = [l for _step, l, _m in history]
assert losses[-1] < losses[0] - 0.3   # loss 也应该明显下降（真实测得从 0.6931 降到 0.3355 左右）
print("stage3 ok: margin grows every single round, loss drops every single round")
```

本机实测输出(25 行,一轮不落):

```text
step   1 | loss 0.6931 | margin +0.0000
step   2 | loss 0.6685 | margin +0.0500
step   3 | loss 0.6450 | margin +0.0988
step   4 | loss 0.6227 | margin +0.1463
step   5 | loss 0.6015 | margin +0.1926
step   6 | loss 0.5813 | margin +0.2378
step   7 | loss 0.5621 | margin +0.2819
step   8 | loss 0.5438 | margin +0.3249
step   9 | loss 0.5264 | margin +0.3669
step  10 | loss 0.5099 | margin +0.4078
step  11 | loss 0.4941 | margin +0.4477
step  12 | loss 0.4791 | margin +0.4867
step  13 | loss 0.4648 | margin +0.5248
step  14 | loss 0.4511 | margin +0.5620
step  15 | loss 0.4381 | margin +0.5983
step  16 | loss 0.4257 | margin +0.6337
step  17 | loss 0.4138 | margin +0.6684
step  18 | loss 0.4024 | margin +0.7023
step  19 | loss 0.3916 | margin +0.7354
step  20 | loss 0.3812 | margin +0.7678
step  21 | loss 0.3712 | margin +0.7995
step  22 | loss 0.3617 | margin +0.8305
step  23 | loss 0.3526 | margin +0.8609
step  24 | loss 0.3439 | margin +0.8906
step  25 | loss 0.3355 | margin +0.9197
stage3 ok: margin grows every single round, loss drops every single round
```

从第 1 轮 `margin +0.0000` 一路到第 25 轮 `margin +0.9197`,loss 从 0.6931 降到 0.3355,现场实测——`assert` 建立在完整存下来的 `history` 列表上,要求**每一轮都比上一轮更大**(严格单调,不是"最后比开头大"这种弱断言),这份实测数据确实全程单调,一次没有回退。margin 这里用的定义和 `dpo_loss` 内部同名变量、以及 01 号文件知识点 4 真实训练打印的"margin"是同一个量:`beta * (log_ratio_w - log_ratio_l)`——这条曲线拉开的正是"chosen 相对 ref 涨了多少、rejected 相对 ref 跌了多少"这个隐式 reward 差,和真实 DPO 训练里应该观察到的现象在方向上完全一致,只是这里是在一个 40 参数的玩具矩阵上,几秒内就能看完整个过程。

---

## 阶段 4:组装成一个可复用的 `MiniDPOTrainer`

把阶段 1-3 的逻辑收进一个小类:装下 `theta`/`theta_ref`,`step()` 方法做一次"前向 → backward → 参数更新",外面套一个 `train_mini_dpo` 跑更长的循环(200 轮,每 25 轮打印一次——这个打印频率不是随手选的,是在照抄真实 `dpo_minimal.py` 的 `--log-interval` 参数:真实训练也不会每一步都打印,只在关键节奏上抽样打印)。这一步除了验证"平均 margin 拉开",还要验证**每一条**偏好对(不是只看平均值)都真实被拉开,以及 `theta_ref` 全程一次都没被动过。

```python
import torch
import torch.nn.functional as F


def dpo_loss(log_p_chosen_actor, log_p_chosen_ref, log_p_rejected_actor, log_p_rejected_ref, beta=0.1):
    log_ratio_w = log_p_chosen_actor - log_p_chosen_ref
    log_ratio_l = log_p_rejected_actor - log_p_rejected_ref
    margin = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(margin).mean()


def log_probs_for_pairs(logits, pairs):
    log_p = F.log_softmax(logits, dim=-1)
    chosen_lp = torch.stack([log_p[p, c] for p, c, l in pairs])
    rejected_lp = torch.stack([log_p[p, l] for p, c, l in pairs])
    return chosen_lp, rejected_lp


class MiniDPOTrainer:
    """把阶段 1-3 拼成一个可复用的小训练器：装一个 policy logits 矩阵(actor)+
    一份冻结的 reference 矩阵，反复做 前向 -> backward -> 手写参数更新。
    对应真实 dpo_minimal.py 里 actor/ref 两份模型 + AdamW 只优化 actor 的架构，
    只是把"两份 GPT-2 权重"换成了"两个 P x R 矩阵"。"""

    def __init__(self, P, R, pairs, beta=0.5, lr=1.0, seed=0):
        torch.manual_seed(seed)
        self.pairs = pairs
        self.beta = beta
        self.lr = lr
        self.theta = torch.randn(P, R, requires_grad=True)
        self.theta_ref = self.theta.detach().clone()
        self.theta_ref.requires_grad_(False)
        self._ref_snapshot = self.theta_ref.clone()   # 单独留一份快照，训练结束后核对 ref 真的没变过

    def step(self):
        chosen_lp_actor, rejected_lp_actor = log_probs_for_pairs(self.theta, self.pairs)
        with torch.no_grad():
            chosen_lp_ref, rejected_lp_ref = log_probs_for_pairs(self.theta_ref, self.pairs)
        loss = dpo_loss(chosen_lp_actor, chosen_lp_ref, rejected_lp_actor, rejected_lp_ref, beta=self.beta)
        per_pair_margin = (self.beta * ((chosen_lp_actor - chosen_lp_ref)
                                         - (rejected_lp_actor - rejected_lp_ref))).detach()

        self.theta.grad = None
        loss.backward()
        with torch.no_grad():
            self.theta -= self.lr * self.theta.grad

        return loss.item(), per_pair_margin

    def ref_is_untouched(self):
        return torch.equal(self.theta_ref, self._ref_snapshot) and self.theta_ref.grad is None


def train_mini_dpo(P, R, pairs, n_steps, beta=0.5, lr=1.0, log_every=25, seed=0):
    trainer = MiniDPOTrainer(P, R, pairs, beta=beta, lr=lr, seed=seed)
    per_pair_history = [[] for _ in pairs]
    loss_history = []
    for step in range(1, n_steps + 1):
        loss_val, per_pair_margin = trainer.step()
        loss_history.append(loss_val)
        for i in range(len(pairs)):
            per_pair_history[i].append(per_pair_margin[i].item())
        if step == 1 or step % log_every == 0:
            print(f"step {step:3d} | loss {loss_val:.4f} | mean margin {per_pair_margin.mean().item():+.4f}")
    return trainer, loss_history, per_pair_history


P, R = 5, 8
pairs = [
    (0, 2, 6),
    (1, 5, 1),
    (2, 0, 3),
    (3, 4, 7),
    (4, 1, 5),
]

trainer, loss_history, per_pair_history = train_mini_dpo(P, R, pairs, n_steps=200, beta=0.5, lr=1.0, log_every=25)

print()
print("ref matrix never touched across 200 steps:", trainer.ref_is_untouched())
assert trainer.ref_is_untouched()

print()
print(f"{'pair':6s} {'prompt':7s} {'chosen':7s} {'rejected':9s} {'start':9s} {'end':9s}")
for i, (p, c, l) in enumerate(pairs):
    start = per_pair_history[i][0]
    end = per_pair_history[i][-1]
    print(f"{i:<6d} {p:<7d} {c:<7d} {l:<9d} {start:<9.4f} {end:<9.4f}")

# 每一条偏好对（不只是平均值）margin 都应该真实拉开，且全程单调不减
for i in range(len(pairs)):
    h = per_pair_history[i]
    assert h[0] == 0.0
    assert h[-1] > 2.5   # 真实测得 5 条 pair 最终都在 ~2.90 附近
    assert all(h[t] <= h[t + 1] + 1e-9 for t in range(len(h) - 1))

assert loss_history[-1] < 0.1   # 真实测得 200 步后 loss 降到 ~0.054
print()
print("stage4 ok: assembled trainer improves every pair, ref frozen the entire time")
```

本机实测输出:

```text
step   1 | loss 0.6931 | mean margin +0.0000
step  25 | loss 0.3355 | mean margin +0.9197
step  50 | loss 0.2025 | mean margin +1.4943
step  75 | loss 0.1414 | mean margin +1.8844
step 100 | loss 0.1076 | mean margin +2.1753
step 125 | loss 0.0864 | mean margin +2.4056
step 150 | loss 0.0720 | mean margin +2.5955
step 175 | loss 0.0616 | mean margin +2.7568
step 200 | loss 0.0537 | mean margin +2.8968

ref matrix never touched across 200 steps: True

pair   prompt  chosen  rejected  start     end      
0      0       2       6         0.0000    2.8968   
1      1       5       1         0.0000    2.8968   
2      2       0       3         0.0000    2.8968   
3      3       4       7         0.0000    2.8968   
4      4       1       5         0.0000    2.8968   

stage4 ok: assembled trainer improves every pair, ref frozen the entire time
```

200 步之后,5 条 toy 偏好对的 margin **全部**从 0.0000 涨到 2.8968——不只是平均值涨了,是每一条都涨了完全相同的量。第一次看到这张表,以为是代码写错了(5 条 pair 起点随机初始化都不一样,凭什么终点会完全一样);实际推了一下阶段 2 发现的那个"归一化精确抵消"的后果,才想明白这不是 bug,是这个玩具设定的结构性产物:

因为每条 pair 都占用一个**独立的行**(5 条 pair、5 个 prompt,互不重叠),阶段 2 已经证明"某一行里没被选中比较的候选,梯度精确为 0"——推广一步就是,每一行的更新轨迹只取决于这一行自己的 `theta_c - theta_l`,和这一行其它候选的具体数值、和其它行发生了什么,完全没有关系。而所有行的 `theta_c - theta_l` 相对 ref 的差在第 0 步都精确是 0(因为 `actor` 初始化就是 `ref` 的精确拷贝),后续每一步的更新量又只由**当前这一步的 margin 大小**决定(同一个 `beta`、同一个 `lr`,对所有行都一样)——五条独立的轨迹,起点相同、递推公式相同,**必然**重合到小数点后第四位都不差。这是这个玩具设置"每条偏好对占一整行、互不重叠"这个简化选择带来的产物,不是 DPO 本身的性质:真实 LM 训练时,所有偏好对共享同一套网络参数,不同样本的梯度会在同一组参数上叠加、互相干扰,不会有这种"整整齐齐"的巧合。这个局限也是下面"可以怎么继续扩展"第一条要解决的问题。

---

## 附加实验(不计入四个阶段):两条互相矛盾的偏好对,会怎样

上面这个"整整齐齐"的发现,立刻带出一个可以现场验证的问题:如果两条偏好对**共享同一行**,会不会真的相互干扰?构造一个最极端的版本——在 prompt 0 这一行上,追加第 6 条 pair,内容和第 1 条完全相反(第 1 条说"回复 2 比回复 6 好",第 6 条说"回复 6 比回复 2 好"),这是一份存心自相矛盾的 toy 偏好数据集。

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)

P, R = 5, 8
# pair 5 和 pair 0 共享同一个 prompt(0)，但 chosen/rejected 完全颠倒：
# pair 0 说 "2 比 6 好"，pair 5 说 "6 比 2 好" —— 一份存心自相矛盾的偏好数据
pairs = [(0, 2, 6), (1, 5, 1), (2, 0, 3), (3, 4, 7), (4, 1, 5), (0, 6, 2)]


def dpo_loss(log_p_chosen_actor, log_p_chosen_ref, log_p_rejected_actor, log_p_rejected_ref, beta=0.5):
    log_ratio_w = log_p_chosen_actor - log_p_chosen_ref
    log_ratio_l = log_p_rejected_actor - log_p_rejected_ref
    margin = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(margin).mean()


def log_probs_for_pairs(logits, pairs):
    log_p = F.log_softmax(logits, dim=-1)
    chosen_lp = torch.stack([log_p[p, c] for p, c, l in pairs])
    rejected_lp = torch.stack([log_p[p, l] for p, c, l in pairs])
    return chosen_lp, rejected_lp


theta = torch.randn(P, R, requires_grad=True)
theta_ref = theta.detach().clone()
theta_ref.requires_grad_(False)
beta, lr = 0.5, 1.0

conflict_pair_margins = []
for step in range(1, 51):
    chosen_lp_actor, rejected_lp_actor = log_probs_for_pairs(theta, pairs)
    with torch.no_grad():
        chosen_lp_ref, rejected_lp_ref = log_probs_for_pairs(theta_ref, pairs)
    loss = dpo_loss(chosen_lp_actor, chosen_lp_ref, rejected_lp_actor, rejected_lp_ref, beta=beta)
    per_pair_margin = (beta * ((chosen_lp_actor - chosen_lp_ref) - (rejected_lp_actor - rejected_lp_ref))).detach()
    theta.grad = None
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    conflict_pair_margins.append((per_pair_margin[0].item(), per_pair_margin[5].item()))

print("pair0 (2>6) and pair5 (6>2) margins at steps 1,10,25,50:")
for s in [0, 9, 24, 49]:
    print(f"  step {s + 1:3d}: pair0={conflict_pair_margins[s][0]:+.6f}  pair5={conflict_pair_margins[s][1]:+.6f}")

# 两条正面冲突的偏好对，margin 应该被精确锁死在 0，连续 50 步一次都不动
assert all(abs(m0) < 1e-6 and abs(m5) < 1e-6 for m0, m5 in conflict_pair_margins)
print("confirmed: two pairs with opposite preference on the exact same row stay deadlocked at 0 forever")
```

本机实测输出:

```text
pair0 (2>6) and pair5 (6>2) margins at steps 1,10,25,50:
  step   1: pair0=+0.000000  pair5=+0.000000
  step  10: pair0=+0.000000  pair5=+0.000000
  step  25: pair0=+0.000000  pair5=+0.000000
  step  50: pair0=+0.000000  pair5=+0.000000
confirmed: two pairs with opposite preference on the exact same row stay deadlocked at 0 forever
```

两条互相冲突的偏好对,margin 全程精确锁在 0.000000,连续 50 步一次都没有移动过——不是训练卡住了(其它 4 条独立的 pair 同一时间正常在涨),是这两条 pair 在同一个 `theta[0,2]`/`theta[0,6]` 上给出方向完全相反、大小精确相等的梯度,每一步的更新量精确抵消。这是一个小而真实的例子,说明**当不同偏好对开始共享参数时,DPO 训练不再是几条独立曲线简单相加**——真实 LM 的参数共享程度比这里高得多(几十亿参数被所有训练样本共同拉扯),矛盾的偏好标注不会像这里这样精确锁死在 0(不同样本很少会精确共享**同一组**参数、精确抵消的概率极低),但"标注质量差、存在矛盾偏好会拖慢甚至抵消训练信号"这个定性结论,在这个玩具例子里已经能看到一个精确、可复现的雏形。

---

## 可以怎么继续扩展(只指方向,不在本文实现)

- **把自由参数矩阵换成一个真实的小型网络**:比如一个 `nn.Embedding(P, d)` 接一个 `nn.Linear(d, R)`,让 logits 由"prompt embedding 过一层线性层"算出来,而不是直接摆一个 `(P,R)` 矩阵当参数——这样不同 prompt 之间会通过共享的 `nn.Linear` 权重产生真实的相互影响,不再是阶段 4 发现的"整整齐齐互不干扰"这种特例,是更接近真实 LM(所有 prompt 共享同一套 transformer 权重)的下一步。
- **mini-batch 采样**:现在每一轮都用全部 5 条 pair 做 full-batch 更新,换成每轮随机采样一个更小的子集,观察 loss/margin 曲线从光滑变得有噪声——这是"玩具 full-batch demo"和"真实 SGD 训练"之间的一个经典差距。
- **换成 `torch.optim.AdamW`**:这里为了让"参数怎么被更新"完全透明,手写了最朴素的 SGD;`dpo_minimal.py` 真实用的是 `torch.optim.AdamW`,把阶段 4 的手写更新换成 `AdamW`,同样的 25/200 步,对比收敛速度和曲线形状的差异。
- **直接对接 `dpo_minimal.py`**:本文的 `theta` 矩阵在真实场景下对应的正是 `dpo_minimal.py::get_log_probs_for_labels` 算出来的那个标量——只是这里跳过了"文本 → tokenizer → GPT-2 forward → log_softmax → gather → mask → sum"这一整条链路,直接从一个可训练矩阵出发。想看这条完整链路真实跑起来,01 号文件知识点 4 已经给出了具体命令(`python learning/dpo-family/src/dpo_minimal.py --n-train 2 --epochs 1 --log-interval 1 --max-length 32 --cpu`)和真实输出。
- **换一个 loss**:把 `dpo_loss` 这一行换成 02 号文件讲过的 `ipo_loss`/`kto_loss` 等其它变体,在同一个 toy 循环上跑一遍,对比 margin 曲线的形状差异(比如 02 号文件知识点 1 提到 IPO 用平方损失,行为上不会像 DPO 这样无限制地把 margin 一直推向无穷大)。

## 这篇教程展示的方法论

这篇教程没有发明任何新公式——`dpo_loss` 一个字符都没有改动,是从 01 号文件知识点 2 原样搬过来的。新增的知识只有一件事:log-prob 在真实场景里"怎么长出来"的最小可能版本(logits 矩阵 → log_softmax → log-prob),以及围绕这个最小版本能做的最基本的一整套训练动作(forward、backward、参数更新、循环、组装)。这和 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 确立的"教程体"方法论是同一套:不重新发明知识点,是把已经验证过的知识点串成一个可以亲手运行的完整流程。

三个中途真实发现、不是提前设计好的现象,都被如实记录下来,而不是回避:阶段 2 里"未被选中比较的候选梯度精确为 0"(和 01 号文件 Z(x) 抵消是同一类代数结构);阶段 4 里"独立行的偏好对轨迹精确重合"(一开始怀疑是 bug,现场推导后确认是这个玩具设定的结构性产物,不是 DPO 的普遍性质);附加实验里"直接冲突的偏好对精确锁死在 0"(前一个发现的自然推论,现场构造反例验证)。这三个发现有一条共同的方法论:**先让代码把真实数字跑出来,看到和预期不一致的地方,再去推导"为什么",而不是反过来先假定一个结论再挑数字凑**——和 00-roadmap.md 里"手写而不直接用 trl"这个架构选择背后的精神是一致的:愿意多写几行代码,换来每一步都看得见、能现场解释的透明度。

---

*创建:2026-07-24*
