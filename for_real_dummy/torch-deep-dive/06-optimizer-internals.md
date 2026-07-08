# 06 · 优化器内部机制深挖(Optimizer Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批讲"优化器到底在算什么"——不是"`optimizer.step()` 就完事了",而是每一行更新公式背后的数学推导,以及 PyTorch **实际实现**和教科书写法之间那些容易被忽略、但面试会专门挖出来问的细节差异。SGD/Momentum/Adam/AdamW 是训练一切模型的地基,也是大厂技术面试二三四面公认的高频深挖区——尤其"Adam 为什么要 bias correction"和"AdamW 和 Adam+L2 到底有什么区别"这两题,能不能讲到底,是判断一个人是"背过公式"还是"真的理解"最快的试金石。

**本文和前几批的关系:** 优化器更新参数,本质上是[01-tensor-memory-model.md](01-tensor-memory-model.md)第 6 节讲的 **in-place 操作**——本篇第 7 节会现场读源码印证这一点;优化器消费的 `.grad`,是 autograd 反向传播（下一批 [02-autograd-internals.md](02-autograd-internals.md)）算出来、并通过梯度累加机制留在每个叶子 tensor 上的东西——`optimizer.zero_grad()` 具体怎么清空、`set_to_none` 是什么,留给 [07-training-loop-internals.md](07-training-loop-internals.md) 详细展开,本篇不重复。

**验证方法论(呼应 01 篇的教训):** 01 篇作者在写作过程中踩过两个坑——①数值从别处复制、没有针对新场景重新验证;②float32 用 `==` 比较因精度问题失败,必须用 `torch.allclose`。这两条本篇同样适用,而且优化器这个主题还有第三条特有的坑:**教科书公式和 PyTorch 实际实现可能存在细节差异**（比如 eps 加在哪一步、momentum buffer 第一步怎么初始化、AdamW 是不是一个独立实现)。本文每一个公式都做了两件事:①读 PyTorch 源码原文确认实际实现;②手写一个 mini 版优化器,和 `torch.optim.*` 跑同一个小问题做逐步 `torch.allclose` 比对,不是凭记忆或教程转述。所有代码已在仓库 `.venv`(torch 2.11.0+cu128,CUDA 可用)下实际跑通验证。

**本篇统一结构(同 00-roadmap.md):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**(公式推导为核心,必须验证)
4. AI 研究/工程场景
5. 可运行例子(手写实现 vs PyTorch 内置对比验证)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. SGD 基础更新公式:`param = param - lr * grad`

**是什么:**
```python
torch.optim.SGD(model.parameters(), lr=0.1)
```
最朴素的参数更新规则:沿梯度反方向走一步,步长由学习率 `lr` 控制。

**一句话:** 损失函数在当前参数点的梯度指向"loss 上升最快"的方向,所以要让 loss 下降,就往梯度的**反方向**挪一小步——这是几乎所有优化器的共同起点,后面 Momentum/Adam 都是在这个基础上加"记忆"和"自适应缩放"。

**底层机制/为什么这样设计:** 梯度 `∇f(θ)` 是 `f` 在 `θ` 点的一阶泰勒展开系数,意味着在 `θ` 附近, `f(θ - ε·∇f(θ)) ≈ f(θ) - ε·||∇f(θ)||²`——只要 `ε`(也就是 `lr`)足够小,右边第二项恒为负,loss 一定下降。这就是"梯度下降"这四个字的全部数学内容:`lr` 太大会导致这个一阶近似失效(甚至发散,第 2 节会现场验证这一点),太小则收敛慢,这也是后面第 9 节学习率调度器存在的根本原因。

**AI 研究场景:** 现代大模型训练几乎不用不带动量/自适应的纯 SGD(收敛太慢),但纯 SGD(+ 大 batch + 精心调的学习率调度)在部分视觉任务、以及作为"最简单、最可预测、最省显存(不需要额外存 momentum/二阶矩状态)"的基线,仍然有实际使用场景——理解它是理解一切改进版本的起点。

**可运行例子:**
```python
import torch

torch.manual_seed(0)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = 2 * x   # 拟合目标: y = 2x,真实参数 w*=2.0

def loss_fn(w):
    return ((w * x - y) ** 2).mean()

lr, steps = 0.1, 20
w_manual = torch.tensor([0.0], requires_grad=True)
w_torch  = torch.tensor([0.0], requires_grad=True)
opt = torch.optim.SGD([w_torch], lr=lr)

for step in range(steps):
    # 手写版:自己算梯度、自己写更新公式
    loss_m = loss_fn(w_manual)
    w_manual.grad = None
    loss_m.backward()
    with torch.no_grad():
        w_manual -= lr * w_manual.grad          # param = param - lr * grad,就这一行

    # PyTorch 内置版
    loss_t = loss_fn(w_torch)
    opt.zero_grad()
    loss_t.backward()
    opt.step()

    assert torch.allclose(w_manual, w_torch, atol=1e-6), (step, w_manual, w_torch)

# 实测:20步后 w_manual = w_torch = 1.999927(target=2.0),逐步轨迹前6个值:
# [0.0, 0.8, 1.28, 1.568, 1.7408, 1.8445] —— 单调靠近 2.0,符合"沿负梯度方向走"的直觉
print(w_manual.item(), w_torch.item())
```

**面试怎么问 + 追问链:**
- **Q:** "SGD 的更新公式是什么,为什么减去梯度就能让 loss 下降?"—— 期望能推一阶泰勒展开,而不是只背公式。
- **追问 1:** "如果 `lr` 设得特别大会怎样?"—— 期望答"一阶近似失效,可能越过最优点甚至发散",最好能举出二次函数场景下 `lr` 超过某个阈值(`2/曲率`)就会发散的具体分析(第 2 节的 ravine 例子会现场验证这个直觉)。
- **追问 2:** "这里的 `grad` 是对哪个 batch 算的?SGD 里的'S'(Stochastic)体现在哪?"—— 期望答出"随机性来自每步用的是 mini-batch 梯度(是全量梯度的无偏估计,但有噪声),不是 `torch.optim.SGD` 这个类本身有随机行为——这个类只实现了'怎么用梯度更新参数','用哪个 batch 算梯度'是 DataLoader 层面的事"。

**常见坑:** 把 `lr` 理解成"绝对步长"而忽视它和梯度尺度是耦合的——同一个 `lr`,在梯度普遍很大的层(比如某些没做归一化的层)和梯度很小的层上,实际移动距离天差地别,这正是第 4 节 Adam 要解决的问题的动机之一。

---

## 2. Momentum 动量项推导

**是什么:**
```python
torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
```

**一句话:** 给参数更新加一个"速度"变量,每一步不是直接用当前梯度,而是用"过去速度的衰减累积 + 当前梯度"的混合量去更新参数——类比物理里的惯性:一个在斜坡上滚动的球,不会因为坡度瞬间变化就立刻改变运动方向,它的运动状态是历史受力的累积结果。

**底层机制/为什么这样设计(现场读源码验证,不凭教科书印象):**

教科书最常见的写法是 `v = momentum*v + grad; param -= lr*v`。读 `torch/optim/sgd.py` 源码(`_single_tensor_sgd` 函数)可以看到 PyTorch 的真实实现是:

```python
if momentum != 0:
    buf = momentum_buffer_list[i]
    if buf is None:
        buf = grad.detach().clone()              # 第一步:buffer 直接就是梯度本身
        momentum_buffer_list[i] = buf
    else:
        buf.mul_(momentum).add_(grad, alpha=1 - dampening)   # 之后:buf = momentum*buf + (1-dampening)*grad
    grad = buf                                     # (nesterov=False 时)
param.add_(grad, alpha=-lr)                         # param -= lr * grad(这里的 grad 已经是 buf)
```

两个和"教科书印象"不完全一致、但源码写得明明白白的细节:
1. **第一步的初始化不是 `momentum*0+grad`(概念上等价,但源码显式把第一步单独处理成"buffer = 梯度本身"),而且 `dampening`(默认 0,不影响本节但影响细节)只从第二步开始生效**——PyTorch 官方文档原话:"the initial value of momentum buffer is set to the gradient value at the first step... dampening will be applied starting at the second step"。
2. **`lr` 在动量递推之外**:`v_{t+1} = momentum*v_t + grad_{t+1}`,`param -= lr*v_{t+1}`——这是 PyTorch 文档明确写出来、并且专门强调"和 Sutskever 等其他框架不同"的一点。其他框架的写法是 `v_{t+1} = momentum*v_t + lr*grad_{t+1}; param -= v_{t+1}`(`lr` 在递推**内部**)。

这两种写法在 `lr` 不变时数学上完全等价(可以归纳证明 `v_other = lr * v_pytorch`),但**只要 `lr` 会变(几乎所有训练都会配 scheduler),两者就会给出不同的轨迹**——因为 PyTorch 的写法每一步都用**当前**lr 去缩放整个动量 buffer,相当于连历史动量的贡献也按新 lr "重新定价";而其他写法里,历史动量项永远保留着它们被计算出来那一刻的 lr。这是本篇验证过程中发现的、多数教程不会提到的一个真实差异,已用代码验证(见下)。

**动量为什么能加速收敛/减少震荡:** 在损失面弯曲程度不同的方向上(比如一个"细长山谷":一个方向曲率大、一个方向曲率小),沿曲率大的方向,单步梯度下降容易走一步就"跨过"谷底,下一步又要往回走,来回震荡;沿曲率小的方向,梯度一直很小、每步移动量也很小,收敛很慢。动量把历史梯度做指数加权累积:如果某个方向上梯度**方向持续一致**(曲率小的那个方向),历史贡献会不断叠加、越滚越快;如果某个方向上梯度**符号来回变化**(曲率大、震荡的那个方向),历史贡献会相互抵消,不会无限放大。

**AI 研究场景:** 几乎所有实际训练配置都会打开 momentum(SGD+momentum 是 CV 领域经典配置,ResNet 论文用的就是 `momentum=0.9`);momentum 也是 Adam 的"一阶矩"(第 4 节)在结构上的直接前身。

**可运行例子(手写 vs PyTorch,严格对齐源码细节):**
```python
import torch

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]); y = 2 * x   # 同第1节的 y=2x 玩具问题
def loss_fn(w):
    return ((w * x - y) ** 2).mean()

lr, momentum, steps = 0.1, 0.9, 20
w_manual = torch.tensor([0.0], requires_grad=True)
w_torch  = torch.tensor([0.0], requires_grad=True)
opt = torch.optim.SGD([w_torch], lr=lr, momentum=momentum)

buf = None
for step in range(steps):
    loss_m = loss_fn(w_manual)          # 沿用第1节的 y=2x 玩具问题
    w_manual.grad = None
    loss_m.backward()
    with torch.no_grad():
        g = w_manual.grad.clone()
        buf = g.clone() if buf is None else momentum * buf + g   # 严格按源码:首步=grad,之后才递推(dampening=0)
        w_manual -= lr * buf

    loss_t = loss_fn(w_torch)
    opt.zero_grad(); loss_t.backward(); opt.step()
    assert torch.allclose(w_manual, w_torch, atol=1e-6), (step, w_manual.item(), w_torch.item())
# 实测:20步后手写版与 torch.optim.SGD(momentum=0.9) 完全一致,final=1.534188
```

**"lr 变化时两种约定分道扬镳"的验证:**
```python
import torch

momentum, steps = 0.9, 20
def lr_schedule(step):                       # 模拟一次 scheduler 降 lr
    return 0.1 if step < 10 else 0.01

# PyTorch 约定
w_pt = torch.tensor([0.0], requires_grad=True)
opt = torch.optim.SGD([w_pt], lr=0.1, momentum=momentum)
for step in range(steps):
    for g in opt.param_groups: g["lr"] = lr_schedule(step)
    loss = loss_fn(w_pt); opt.zero_grad(); loss.backward(); opt.step()

# "其他框架"约定(v 递推内部乘 lr)
w_other, v = torch.tensor([0.0], requires_grad=True), None
for step in range(steps):
    lr = lr_schedule(step)
    loss = loss_fn(w_other); w_other.grad = None; loss.backward()
    with torch.no_grad():
        g = w_other.grad.clone()
        v = lr * g if v is None else momentum * v + lr * g
        w_other -= v

# 实测:lr 不变时两者几乎相同(浮点误差级别);lr 在第10步减半后,两者明显分道扬镳:
# pytorch 约定 final = 2.2217   other 约定 final = 2.7959  —— 差距不是舍入误差
print(w_pt.item(), w_other.item())
assert abs(w_pt.item() - w_other.item()) > 0.1
```

**震荡/加速的直观验证(细长山谷 `f(a,b)=0.5*(25a²+b²)`,`lr=0.02`,40步):**
```python
import torch

def ravine(p):
    return 0.5 * (25 * p[0] ** 2 + p[1] ** 2)   # a方向曲率25(陡), b方向曲率1(缓)

p_plain = torch.tensor([1.0, 1.0], requires_grad=True)
p_mom   = torch.tensor([1.0, 1.0], requires_grad=True)
opt_plain = torch.optim.SGD([p_plain], lr=0.02)
opt_mom   = torch.optim.SGD([p_mom], lr=0.02, momentum=0.9)
for _ in range(40):
    for p, opt in [(p_plain, opt_plain), (p_mom, opt_mom)]:
        l = ravine(p); opt.zero_grad(); l.backward(); opt.step()

# 实测(前12步): 缓方向 b —— plain: 0.98→0.8007   momentum: 0.98→0.2199(快得多,验证"加速")
#               陡方向 a —— plain: 1.0→0.0005(单调收敛)  momentum: 1.0→-0.3307(反复越过0,验证"震荡")
# 40步后总 loss:plain=0.099324  momentum=0.003000 —— 动量虽在陡方向震荡,靠缓方向的巨大提速仍整体获胜
assert ravine(p_mom) < ravine(p_plain)
```

**面试怎么问 + 追问链:**
- **Q:** "SGD 加 momentum 之后更新公式是什么?"—— 期望写出 `v=momentum*v+grad; param-=lr*v`,而不是随口说"就是加个惯性"。
- **追问 1(区分度高):** "PyTorch 里 momentum buffer 第一步是怎么初始化的?"—— 期望知道"直接等于第一步的梯度",而不是想当然地说"初始化成 0,第一步 `v=momentum*0+grad`"——两者数值结果一样,但源码明确把这当成一个独立分支处理,`dampening` 从第二步才生效,这个细节说明面试者是否真的读过源码或做过底层验证。
- **追问 2(杀伤力很强):** "如果训练中途 lr 变了(比如触发了 scheduler),momentum buffer 会不会受影响?"—— 期望能说出"PyTorch 的实现里,lr 在 v 的递推之外,所以每一步都用当前 lr 重新缩放整个 buffer;这和某些教材/其他框架'v 递推内部就乘了 lr'的写法不同,lr 不变时等价,lr 一变就会产生不同的轨迹"——这是本篇验证中发现的一个真实差异,能答上来说明对实现细节有第一手认识,而不是转述文档。
- **追问 3:** "为什么说 momentum 能'加速收敛、减少震荡'?能不能举个具体场景?"—— 期望能讲清楚"曲率不同的方向,梯度历史一致 vs 来回变化,累积效果不同",最好能像上面的 ravine 例子一样具体。

**常见坑:** 认为 momentum 无条件让训练更稳——上面的 ravine 例子已经验证:如果 `lr` 对于某个方向的曲率来说偏大,momentum 反而会**放大**该方向的震荡(因为动量相当于放大了有效步长),这也是实践中"开了 momentum 之后 loss 变得更不稳定,得把 lr 调小一点"这条经验的数学根源。

---

## 3. Nesterov Momentum 与普通 Momentum 的区别

**是什么:**
```python
torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
```

**一句话:** 普通 momentum 是"在**当前**位置算梯度,再往动量方向走";Nesterov 的核心思想是"先往动量方向**预走一步**,在那个预估位置算梯度",相当于"边走边看前方路况调整",而不是"看完当前路况再决定往哪走、走多远"。

**底层机制/为什么这样设计(现场验证实现差异,不停留在直觉描述):**

读源码(`_single_tensor_sgd`)可以看到,当 `nesterov=True` 时,唯一的区别在于用哪个量去更新参数:
```python
buf = momentum * buf + grad          # 普通 momentum 的递推,和第2节完全一样
if nesterov:
    grad = grad + momentum * buf     # 多这一步:把"当前梯度"和"更新后的buf"再组合一次
else:
    grad = buf
param -= lr * grad
```
**这里必须澄清一个常被讲错的细节:** PyTorch 并**没有**真的把参数临时挪到"预估位置" `param - momentum*buf` 再单独调用一次 `backward()` 算梯度(那样需要多一次完整的前向+反向,开销翻倍)。它用的是 Sutskever et al.(2013,`On the importance of initialization and momentum in deep learning`)给出的**代数等价重参数化**:只用"当前位置"算一次梯度,通过 `grad + momentum*buf` 这个线性组合,就能达到和"真的跑到预估位置算梯度"完全相同的效果。

这不是"约等于",是可以严格验证的精确等价——本文额外做了一次数值验证:分别实现"字面上的两步预看"版本(在 `θ+momentum*v` 处求梯度)和 PyTorch 的"当前点代数组合"版本,发现前者每一步的**预估点** `Θ_t = θ_t + momentum*v_t` 序列,和后者的**参数值** `θ_t` 序列,逐步完全相等(12 步内最大误差 `4.4e-16`,是 float 精度的噪声级别,不是近似):

```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]); y = 2 * x
def grad_fn(w):  # 解析梯度,不依赖 autograd,这样才能在任意点(包括预估点)求值
    return float((2 * (w * x - y) * x).mean())

lr, mu, steps = 0.1, 0.9, 12

# 版本A:字面"预看" —— v_{t+1}=mu*v_t-lr*grad(theta_t+mu*v_t); theta_{t+1}=theta_t+v_{t+1}
theta, v, lookahead_traj = 0.0, 0.0, []
for t in range(steps):
    Theta_t = theta + mu * v            # 预估点(该"看"的地方)
    lookahead_traj.append(Theta_t)
    g = grad_fn(Theta_t)                # 在预估点上真的算一次梯度
    v = mu * v - lr * g
    theta = theta + v

# 版本B:PyTorch 的代数组合 —— 只在当前点算梯度
theta_pt, b, pt_traj = 0.0, None, []
for t in range(steps):
    pt_traj.append(theta_pt)
    g = grad_fn(theta_pt)               # 只在"当前位置"算梯度,没有预看
    b = g if b is None else mu * b + g
    theta_pt -= lr * (g + mu * b)

max_diff = max(abs(a - b_) for a, b_ in zip(lookahead_traj, pt_traj))
assert max_diff < 1e-9   # 实测 4.44e-16,机器精度级别 —— 两者是精确等价,不是近似
```

**AI 研究场景:** `nesterov=True` 在 CV 分类任务里是常见的"免费提升"配置项(几乎不增加计算量,经验上通常比普通 momentum 收敛更快、更稳);理解它"只需一次梯度求值"这一点,也解释了为什么 Nesterov 动量在深度学习框架里能被普遍默认支持,而不是被当成一个"两倍算力"的奢侈选项。

**可运行例子(手写 vs PyTorch,以及两种轨迹确实不同):**
```python
import torch

lr, momentum, steps = 0.1, 0.9, 15
w_manual = torch.tensor([0.0], requires_grad=True)
w_torch  = torch.tensor([0.0], requires_grad=True)
opt = torch.optim.SGD([w_torch], lr=lr, momentum=momentum, nesterov=True)

buf = None
for step in range(steps):
    loss_m = loss_fn(w_manual)
    w_manual.grad = None
    loss_m.backward()
    with torch.no_grad():
        g = w_manual.grad.clone()
        buf = g.clone() if buf is None else momentum * buf + g
        g_used = g + momentum * buf              # nesterov 组合
        w_manual -= lr * g_used

    loss_t = loss_fn(w_torch)
    opt.zero_grad(); loss_t.backward(); opt.step()
    assert torch.allclose(w_manual, w_torch, atol=1e-6), (step, w_manual.item(), w_torch.item())
# 实测:15步后手写 nesterov 与 torch.optim.SGD(nesterov=True) 完全一致,final=2.003176

# 从同一起点对比 plain momentum 与 nesterov 的轨迹
w_p = torch.tensor([0.0], requires_grad=True); opt_p = torch.optim.SGD([w_p], lr=lr, momentum=momentum)
w_n = torch.tensor([0.0], requires_grad=True); opt_n = torch.optim.SGD([w_n], lr=lr, momentum=momentum, nesterov=True)
traj_p, traj_n = [0.0], [0.0]
for _ in range(steps):
    for p, opt, traj in [(w_p, opt_p, traj_p), (w_n, opt_n, traj_n)]:
        l = loss_fn(p); opt.zero_grad(); l.backward(); opt.step(); traj.append(round(p.item(), 4))
# 实测第1步就分道扬镳: plain[1]=0.8   nesterov[1]=1.52 —— nesterov 提前"看到"了动量要带它去哪,反应更快
assert traj_p != traj_n
```

**面试怎么问 + 追问链:**
- **Q:** "Nesterov Momentum 和普通 Momentum 有什么区别?"—— 期望答"在动量方向上先走一步,在那个位置算梯度",而不是"就是 momentum 的一个变种,效果更好一点"这种不知其所以然的回答。
- **追问 1(高区分度):** "PyTorch 真的会把参数移到预估位置、重新做一次前向+反向来算这个'预看'的梯度吗?"—— 大部分人会想当然地回答"是",正确答案是"不是",PyTorch 用代数重组(`grad + momentum*buf`)在**当前点**一次求值就等价地实现了预看效果,不需要额外的前向/反向传播——这是本节验证的核心。
- **追问 2(深挖):** "你怎么证明这种代数写法和字面上的'两步预看'是等价的,而不只是效果差不多?"—— 期望能提到重参数化/换元的思路,或者至少认可"这是能通过数值验证的精确等价,不是经验性的近似"——上面的验证代码就是标准答案。
- **追问 3(开放):** "既然预看版本和当前点版本数值轨迹能对齐,这种'不增加额外前向反向就能获得预看效果'的技巧,还能用在哪些场景?"—— 可以联系到 Lookahead 优化器、部分元学习/双层优化里的近似技巧,考察知识迁移能力。

**常见坑:** 把 Nesterov 理解成"更复杂、更慢"的版本——实际上它的额外开销只有一次标量乘加(`grad + momentum*buf`),和普通 momentum 几乎没有性能差异,不应该因为"预看"这个描述就默认它更耗资源。另外,`nesterov=True` 要求 `momentum>0` 且 `dampening=0`(源码里有显式检查),否则初始化时直接报错。

---

## 4. Adam 一阶二阶矩估计

**是什么:**
```python
torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
```

**一句话:** Adam 同时维护两个指数移动平均(EMA)统计量——一阶矩 `m`(梯度的"降噪平均方向",作用类似 momentum)和二阶矩 `v`(梯度平方的移动平均,衡量这个参数的梯度历史上"波动有多大"),用 `m` 决定往哪走,用 `v` 给每个参数**独立**地自适应缩放有效学习率。

**底层机制/为什么这样设计(源码逐行核对,不是转述论文公式):**

读 `torch/optim/adam.py`(`_single_tensor_adam`),核心更新逻辑是:
```python
m = beta1 * m + (1 - beta1) * grad          # 一阶矩:梯度的指数移动平均
v = beta2 * v + (1 - beta2) * grad * grad   # 二阶矩:梯度平方的指数移动平均
m_hat = m / (1 - beta1 ** t)                # bias correction,第5节详细推导
v_hat = v / (1 - beta2 ** t)
param -= lr * m_hat / (v_hat.sqrt() + eps)  # 注意 eps 加在 sqrt 之后,不是 sqrt(v_hat+eps)!
```
最后一行有一个容易凭记忆写错的细节,已用源码确认:**`eps` 是加在 `sqrt(v_hat)` 算出来之后**,不是先把 `eps` 加进 `v_hat` 里再开方(`sqrt(v_hat+eps)`)——这两种写法数值上不完全相同(虽然通常差异极小),PyTorch 严格遵循 Adam 原论文(Kingma & Ba, 2014)的写法 `θ ← θ - α·m̂/(√v̂+ε)`。

**`m` 和 `v` 分别在干什么:**
- **`m`(一阶矩)** 是梯度的降噪平均——单步的 mini-batch 梯度带噪声(方向可能左右摆动),EMA 平均掉高频噪声,留下"最近一段时间梯度大体指向哪"这个更可靠的方向信号,和第 2 节 momentum 的 `v` 是同一个数学对象(只是 Adam 里多了 `(1-beta1)` 这个归一化系数,让 `m` 的量纲和 `grad` 一致,而不是像 momentum 那样量纲会随 `1/(1-momentum)` 放大)。
- **`v`(二阶矩)** 是梯度平方的历史统计——如果某个参数的梯度历史上一直很大(比如上游没做好归一化的层),`v` 就大,`sqrt(v)` 作为分母就会**压低**这个参数的有效学习率;反之梯度历史上一直很小的参数,`v` 小,分母小,相当于**放大**它的有效学习率。这就是"自适应"三个字的全部含义:每个参数拥有自己独立的、由自己梯度历史决定的有效步长,而不是像 SGD 那样所有参数共享同一个 `lr`。

**AI 研究场景:** 现代深度学习(尤其 Transformer/LLM 训练)几乎默认用 Adam 家族优化器,一个直接原因是模型里不同层/不同参数的梯度尺度天然差异巨大(embedding 层、LayerNorm 参数、深层权重的梯度量级可以相差几个数量级),SGD 用同一个 `lr` 很难兼顾所有参数,Adam 的自适应缩放天然缓解了这个问题。

**可运行例子(手写 vs PyTorch,并用一个"两个参数梯度尺度差100倍"的问题直接展示自适应缩放效果):**
```python
import torch

# 构造两个梯度尺度差 100 倍的参数:x1 尺度 [-1,1],x2 尺度 [-100,100]
x1 = torch.linspace(-1, 1, 5); x2 = torch.linspace(-100, 100, 5)
y2 = 3 * x1 + 0.01 * x2
def loss_fn2(w):
    return ((w[0] * x1 + w[1] * x2 - y2) ** 2).mean()
# 实测 w=[0,0] 处原始梯度 = [-4.0, -400.0],w[1] 的梯度天生是 w[0] 的 100 倍

lr, beta1, beta2, eps, steps = 0.1, 0.9, 0.999, 1e-8, 30
w_manual = torch.zeros(2, requires_grad=True)
w_torch  = torch.zeros(2, requires_grad=True)
opt = torch.optim.Adam([w_torch], lr=lr, betas=(beta1, beta2), eps=eps)

m, v = torch.zeros(2), torch.zeros(2)
for t in range(1, steps + 1):
    loss_m = loss_fn2(w_manual)
    w_manual.grad = None
    loss_m.backward()
    with torch.no_grad():
        g = w_manual.grad.clone()
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w_manual -= lr * m_hat / (v_hat.sqrt() + eps)

    loss_t = loss_fn2(w_torch)
    opt.zero_grad(); loss_t.backward(); opt.step()
    assert torch.allclose(w_manual, w_torch, atol=1e-6), (t, w_manual.tolist(), w_torch.tolist())

# 实测:30步后两个参数几乎同步收敛到 [0.0340, 0.0340] —— 尽管原始梯度尺度差100倍,
# 最终 m_hat/(sqrt(v_hat)+eps) 这个"有效更新量"对两个参数的量级被拉到了同一个数量级,
# 这就是二阶矩自适应缩放在真实起作用的证据,不是文字描述。
print(w_manual.tolist())
```

**面试怎么问 + 追问链:**
- **Q:** "Adam 里的一阶矩和二阶矩分别是做什么用的?"—— 期望完整答出"一阶矩=降噪后的梯度方向估计,二阶矩=给每个参数自适应缩放有效学习率",而不是只会背两条更新公式。
- **追问 1:** "二阶矩为什么要用梯度的**平方**,而不是梯度本身的绝对值或者其他统计量?"—— 期望能提到"平方让统计量恒正、且是方差/二阶矩的自然估计,`sqrt(v)` 之后量纲和梯度一致,可以直接做分母";更进一步可以提到这是对"梯度的均方根"(RMS)的移动平均估计,呼应 RMSProp(Adam 可以理解成 Momentum + RMSProp 的结合)。
- **追问 2(工程向):** "两个参数原始梯度尺度差 100 倍,用 Adam 训练后,它们的更新幅度会怎样?"—— 期望能答出"会被自适应缩放拉到接近的量级",最好能像上面例子一样说出具体的验证思路。

**常见坑:** 把 Adam 的"自适应"理解成"自动调 `lr`"——严格说 `lr` 这个超参数本身并没有被优化器自动调整,被"自适应"的是**每个参数各自的有效步长比例**(通过除以 `sqrt(v_hat)`),`lr` 仍然是你手动设置、且仍然需要调的超参数,只是所有参数不再共享同一个"裸" `lr` 直接生效。

---

## 5. bias correction 为什么需要(面试高频深挖题)

**是什么:** `m` 和 `v` 分别除以 `(1-beta1^t)` 和 `(1-beta2^t)`(`t` 是当前 step 数,从 1 开始计数)。

**一句话:** `m` 和 `v` 初始化为 0,训练刚开始的几步相当于"0 和真实梯度信号做加权平均",估计值会**系统性地偏向 0**,除以 `(1-beta^t)` 是对这个系统性偏差的精确解析修正,不是经验性的补丁。

**底层机制/为什么这样设计(几何级数推导 + 现场验证"不做会怎样"):**

把 `m` 的递推展开成显式求和形式。由 `m_t = beta1*m_{t-1} + (1-beta1)*g_t`,`m_0=0`,展开得:

```
m_t = (1-beta1) * Σ_{i=1}^{t} beta1^{t-i} * g_i
```

这是历史梯度的指数加权和,权重加起来等于多少?对权重求和:`(1-beta1) * Σ_{i=1}^{t} beta1^{t-i} = (1-beta1) * (1-beta1^t)/(1-beta1) = 1 - beta1^t`——**权重之和不是 1,而是 `1-beta1^t`**(`t` 越小、`beta1` 越接近 1,这个值离 1 越远)。

如果假设梯度近似平稳(`E[g_i] ≈ g`,Adam 论文本身推导时用的简化假设),两边取期望:

```
E[m_t] = g * (1 - beta1^t)          <-  不等于 g,而是被压低了 (1-beta1^t) 这个系数!
```

这就是"系统性偏向 0"的精确数学含义:`m_t` 不是 `g` 的无偏估计,而是 `g` 乘上一个恒小于 1、且随 `t` 增大才趋近于 1 的衰减系数。除以 `(1-beta1^t)` 之后 `E[m̂_t] = E[m_t]/(1-beta1^t) = g`,精确抵消了这个系统性偏差——这不是"大概修正一下",是解析可推导、可以严格证明的精确修正(`v` 的 bias correction 推导完全同构,把 `g` 换成 `g²` 即可)。

**"不做 bias correction 会怎样"——现场验证,而且结论比大多数教程讲的更微妙:**

朴素直觉可能会认为"不做修正,`m`、`v` 都偏小,更新幅度应该也偏小"。但 `m` 被压低的系数是 `(1-beta1^t)`,`v` (在开方前)被压低的系数是 `(1-beta2^t)`——这是**两个不同的衰减速度**(`beta2=0.999` 比 `beta1=0.9` 更接近 1,所以 `v` 早期被压低得更狠),而更新量正比于 `m/sqrt(v)`,分子分母被压低的比例不同,**没有修正的更新幅度到底是偏大还是偏小,并不能想当然下结论,必须实际算**:

```python
lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
xs = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]); ys = 2 * xs
def loss1d(w): return ((w * xs - ys) ** 2).mean()

w_bc, m_bc, v_bc = torch.zeros(1, requires_grad=True), torch.zeros(1), torch.zeros(1)
w_nobc, m_nobc, v_nobc = torch.zeros(1, requires_grad=True), torch.zeros(1), torch.zeros(1)

for t in range(1, 9):
    loss_bc = loss1d(w_bc); w_bc.grad = None; loss_bc.backward()
    with torch.no_grad():
        g = w_bc.grad.clone()
        m_bc = beta1 * m_bc + (1 - beta1) * g
        v_bc = beta2 * v_bc + (1 - beta2) * g * g
        upd_bc = lr * (m_bc / (1 - beta1**t)) / ((v_bc / (1 - beta2**t)).sqrt() + eps)
        w_bc -= upd_bc

    loss_nobc = loss1d(w_nobc); w_nobc.grad = None; loss_nobc.backward()
    with torch.no_grad():
        g = w_nobc.grad.clone()
        m_nobc = beta1 * m_nobc + (1 - beta1) * g
        v_nobc = beta2 * v_nobc + (1 - beta2) * g * g
        upd_nobc = lr * m_nobc / (v_nobc.sqrt() + eps)   # 不除以 (1-beta^t)
        w_nobc -= upd_nobc
```

**实测结果(t=1..8,`|upd_nobc|/|upd_bc|` 之比):**

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 比值(不修正/修正) | 3.16 | 4.22 | 4.82 | 5.04 | 4.84 | 4.18 | 3.08 | 1.77 |

**结论和多数教程说的"更新太小"正好相反:在这个问题里,不做 bias correction 的更新幅度在前 8 步始终比修正后的版本大(最多接近 5 倍),而不是更小。** 第 1 步可以精确推导出解析比值:此时 `m_1=(1-beta1)*g`、`v_1=(1-beta2)*g²`,`m̂_1=g`、`v̂_1=g²`(bias correction 在第一步是**精确**抵消,不是近似),所以

```
|upd_nobc| / |upd_bc| = (1-beta1) / sqrt(1-beta2) = 0.1 / sqrt(0.001) ≈ 3.162
```

代码算出来的第一步比值(3.1623)和这个解析公式完全吻合。更深一层的原因是:`beta2=0.999` 比 `beta1=0.9` 更接近 1,导致 `v`(分母)被压低的**相对幅度**在开方之后反而比 `m`(分子)更小,所以早期未修正的"有效学习率" `m/sqrt(v)` 是被**放大**而不是缩小的——这正是 bias correction 存在的实际意义:如果不修正,训练最初几步会用一个不可预测、时大时小的"隐藏学习率"在更新参数,而不是一个稳定值,bias correction 把这一切校正成"任何时刻的有效更新量都只由真实梯度决定,和 `t` 无关"。

**AI 研究场景:** 这也是为什么很多训练框架里 Adam 的前几十到几百步(尤其结合 warmup,见第 9 节)表现会比中后期"更敏感"——`beta1^t`、`beta2^t` 还没充分衰减到 0 时,即使有 bias correction,`m̂`、`v̂` 依然是基于很少几个样本估计出来的,方差本身就大,这是 bias correction 修正"系统性偏差"之外、依然residual 存在的"估计方差大"问题,warmup 通过压低早期 `lr` 来缓解的正是这一层。

**面试怎么问 + 追问链(这是全篇要求"讲透"的两个题之一):**
- **Q:** "Adam 为什么需要 bias correction?"—— 期望不满足于"因为初始化是0",而是能说出"`m`/`v` 是历史梯度的加权平均,但权重之和不为1,导致系统性偏向0,需要除以权重和来精确修正"。
- **追问 1(核心,必须能推导):** "能不能推一下这个偏差具体是多少?"—— 期望能现场写出 `m_t=(1-beta1)Σβ1^{t-i}g_i` 这个几何级数展开,算出权重和 `1-beta1^t`,得到 `E[m_t]=g(1-beta1^t)`——这是本节最核心的推导,答不出说明只是记住了结论没理解来源。
- **追问 2(杀伤力很强、区分度极高):** "如果不做 bias correction,训练早期的更新幅度会变大还是变小?"—— **这是一道陷阱题**,大多数人会不假思索地回答"变小"(因为想到"m、v都偏向0"),正确答案是"不一定,取决于 beta1 和 beta2 的相对大小,在标准配置(beta1=0.9, beta2=0.999)下实际是更大"——能现场推出 `(1-beta1)/sqrt(1-beta2)` 这个比值公式、或者至少给出验证过的具体数字,是候选人是否真正做过实验/推导、而非死记硬背的分水岭。
- **追问 3:** "这个偏差会一直存在吗?训练久了呢?"—— 期望答"`t`增大,`beta1^t`和`beta2^t`都趋于0,bias correction 因子趋于1,修正的必要性自然消失——这本质是个只影响训练早期的现象"。
- **追问 4(开放,连接调参经验):** "Adam 里 `beta1` 和 `beta2` 通常设成 0.9 和 0.999,如果把 `beta2` 也设成 0.9,bias correction 的影响会有什么变化?"—— 期望能代入 `(1-beta1)/sqrt(1-beta2)` 的形式定性分析(两者更接近时比值趋近于 `sqrt(1-beta1)`,不再是好几倍的悬殊差异,但依然不等于1)——考察能不能把已推出的公式灵活应用到新参数组合,而不是只会讲固定配置下的结论。

**常见坑:** 把 bias correction 的作用简单等同于"让早期更新变小、训练更稳"——上面的验证已经说明,它实际在做的是"消除一个和 `t` 相关、方向不确定的系统性缩放因子",至于修正前更新是偏大还是偏小,取决于具体的 `beta1`/`beta2` 组合,不能一概而论。

---

## 6. AdamW 与 Adam+L2 正则化的本质区别(面试高频深挖题)

**是什么:**
```python
torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)    # L2 正则化,混入梯度
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)   # 解耦 weight decay,不混入梯度
```

**一句话:** Adam 的 `weight_decay` 参数是把 `weight_decay*param` 加进梯度里,让这个正则化项也被卷入一阶二阶矩的自适应缩放机制;AdamW 把 weight decay 从梯度里**解耦**出来,在参数更新那一步单独做一次和梯度历史无关的、纯粹的乘法收缩——这是两种效果不同的正则化实现,不是同一个东西换了个名字。

**底层机制/为什么这样设计(源码确认 + 隔离实验精确量化"扭曲"程度):**

读 `torch/optim/adam.py` 源码可以确认一个很多人不知道的事实:在这版 PyTorch 里,**`AdamW` 类本身没有独立实现,字面上就是 `Adam.__init__(..., decoupled_weight_decay=True)`**(`adamw.py` 里 `AdamW(Adam)` 直接继承并传这个 flag)。两种模式的核心分支:

```python
if weight_decay != 0:
    if decoupled_weight_decay:              # AdamW 路径
        param.mul_(1 - lr * weight_decay)   # 直接对参数做乘法收缩,和梯度/历史动量完全无关
    else:                                    # Adam+L2 路径
        grad = grad.add(param, alpha=weight_decay)   # 把 weight_decay*param 混进梯度里
# 之后两条路径都用(可能已经改过的)grad 去更新 m、v,再做常规 Adam 更新
```

问题出在 Adam+L2 这条路径:混入梯度后的 `weight_decay*param` 这一项,会和"真正的"损失梯度一起被送进 `m`、`v` 的 EMA 里,**被那套本来是为了自适应缩放梯度设计的机制,一并拿去自适应缩放了**。具体后果:一个历史梯度一直很大(`v` 很大)的参数,它的正则化力度会被 `sqrt(v)` 压得很低;一个历史梯度一直很小(`v` 很小)的参数,正则化力度几乎不受影响——这完全违背了 weight decay"均匀地把所有权重往 0 拉"的本意,AdamW 论文(Loshchilov & Hutter,*Decoupled Weight Decay Regularization*,ICLR 2019)正是为了修正这个问题提出的。

**隔离实验:把"真实梯度"人为设为 0,只看 weight decay 单独造成多大移动,精确量化"扭曲"了多少倍:**
```python
lr_, beta1_, beta2_, eps_, wd_ = 0.1, 0.9, 0.999, 1e-8, 0.01
t_ = 100                              # t 足够大,bias correction 因子约等于1,排除干扰
w_a0, w_b0 = 1.0, 1.0                 # 两个参数当前值相同
v_a0, v_b0 = 1000.0, 0.001            # w_a: 历史梯度一直很大;w_b: 历史梯度一直很小

def adam_l2_decay_only(w, v):
    g = wd_ * w                        # fresh grad = 0,只剩 L2 项
    m = (1 - beta1_) * g
    v_new = beta2_ * v + (1 - beta2_) * g * g
    upd = lr_ * (m / (1 - beta1_**t_)) / ((v_new / (1 - beta2_**t_)) ** 0.5 + eps_)
    return w - upd

w_a_l2 = adam_l2_decay_only(w_a0, v_a0)
w_b_l2 = adam_l2_decay_only(w_b0, v_b0)
w_a_adamw = w_a0 * (1 - lr_ * wd_)
w_b_adamw = w_b0 * (1 - lr_ * wd_)
```

**实测结果:**
- **Adam+L2:** `w_a`(历史梯度大)只收缩了 `9.76e-7`;`w_b`(历史梯度小)收缩了 `9.76e-4`——**`w_b` 受到的有效衰减力度是 `w_a` 的约 1000 倍**,纯粹是自适应缩放机制"扭曲"出来的结果,和这两个参数"该不该被正则化"毫无关系。
- **AdamW:** `w_a` 和 `w_b` **各收缩了完全相同的 `1.0e-3`**(比值精确等于 `1.0000`)——衰减力度和梯度历史完全无关,是纯粹、均匀的收缩。

再配合完整轨迹验证(手写实现 vs 官方类,以及两种方法最终解确实不同):
```python
# 完整轨迹:手写 Adam+L2 == torch.optim.Adam(weight_decay=wd);手写 AdamW == torch.optim.AdamW(weight_decay=wd)
# (完整代码结构同第4节,区别只在 weight_decay 项混入梯度 vs 解耦到参数,均用 allclose 验证到 atol=1e-5)
# 实测(wd=0.5,20步,第4节的双尺度玩具问题):
#   Adam+L2 终值 = [0.0321, 0.0421]
#   AdamW   终值 = [0.0332, 0.0332]
#   两者差值   = [0.0011, 0.0089]  —— 明确不同,不是重命名关系
```

**AI 研究场景:** Transformer/LLM 训练几乎清一色用 AdamW(BERT、GPT 系列、LLaMA 等的官方配置都是如此),原因正是这里验证的机制——这些模型不同层(embedding、attention、FFN、LayerNorm)的梯度尺度差异极大,如果用 Adam+L2,weight decay 事实上只会均匀作用在"梯度历史较小"的那部分参数上,对大梯度参数几乎不起正则化效果;AdamW 保证不管梯度尺度如何,每个参数受到的相对衰减力度都一致,这才是"weight decay"这个名字应有的行为。

**面试怎么问 + 追问链(这是全篇要求"讲透"的两个题之二):**
- **Q:** "AdamW 和 Adam 加 L2 正则化有什么区别?"—— 期望不满足于"AdamW 效果更好"这种模糊回答,而是能准确说出"L2 混入梯度、被自适应缩放机制影响;AdamW 解耦,直接对参数做均匀收缩"。
- **追问 1(核心,必须能讲清楚机制):** "具体是怎么'被自适应缩放机制影响'的,能量化吗?"—— 期望能讲出"weight_decay*param 被当成梯度的一部分送进 v 的 EMA,历史梯度大的参数,它的这部分'正则化梯度'也会被 sqrt(v) 压低",最好能像上面的隔离实验一样给出具体倍数(哪怕只是数量级层面)。
- **追问 2:** "PyTorch 里 AdamW 是重新写的一个优化器类吗?"—— 期望知道(或者能通过读源码得出)"不是,是 Adam 内部一个 `decoupled_weight_decay` 分支,`AdamW` 类只是把这个 flag 设成 True 的封装"——这个问题专门考察有没有动手读过源码,而不是只看过文档/博客描述。
- **追问 3(工程向,容易被问倒):** "如果一个训练脚本原来用 `Adam(weight_decay=0.01)`,现在想换成 `AdamW`,直接把类名换掉、`weight_decay` 数值不变,合理吗?"—— 期望能答出"不一定合理,两种机制下同一个数值代表的'实际衰减力度'不同(尤其对梯度尺度差异大的参数),换成 AdamW 后通常需要重新搜索 weight_decay 的合适取值,不能想当然认为数值可以直接复用"。
- **追问 4(论文/历史向):** "这个问题是哪篇论文提出来的,论文是怎么论证的?"—— 期望知道 Loshchilov & Hutter 的 *Decoupled Weight Decay Regularization*(ICLR 2019),核心论点正是"在 Adam 里,L2 正则化和 weight decay 不再等价(尽管在纯 SGD 里两者数学等价)",这也是这篇论文标题里"Decoupled"的含义来源。

**常见坑:** 认为"AdamW 就是 Adam 换了个名字,内部实现一样,只是文档写法不同"——本节的隔离实验已经精确证伪:在梯度尺度差异大的参数上,两者对 weight_decay 数值的"实际使用方式"可以相差几个数量级,不是无关紧要的实现细节。另一个常见误解是把"L2 正则化"和"weight decay"当成任何优化器下都完全等价的两个概念——它们只在**朴素 SGD**(没有自适应缩放、没有动量对梯度做非线性变换)下数学等价,这也是 AdamW 论文强调的前提条件。

---

## 7. `optimizer.step()` 内部到底做了什么

**是什么:** 遍历 `self.param_groups`,对每一组超参数下的每个参数,过滤出"当前有梯度"的那些,按该优化器的公式做 **in-place** 更新。

**一句话:** `step()` 本身不算梯度(梯度是 `backward()` 算出来、挂在 `.grad` 上的),它只负责"拿着已经算好的 `.grad`,按公式改 `.data`"这一件事,而且是原地改,不新分配和参数一样大的内存。

**底层机制/为什么这样设计(现场读源码,不转述):**

```python
import inspect
print(inspect.getsource(torch.optim.SGD.step))
```
实测输出(节选核心逻辑):
```python
def step(self, closure=None):
    ...
    for group in self.param_groups:                    # 第一层循环:遍历每一组超参数
        params, grads, momentum_buffer_list = [], [], []
        has_sparse_grad = self._init_group(group, params, grads, momentum_buffer_list)
        sgd(params, grads, momentum_buffer_list,        # 第二层:调用真正做数学运算的函数
            weight_decay=group["weight_decay"], momentum=group["momentum"],
            lr=group["lr"], dampening=group["dampening"], nesterov=group["nesterov"], ...)
        ...
```
再读 `_init_group`(决定哪些参数会被这一步实际更新):
```python
def _init_group(self, group, params, grads, momentum_buffer_list):
    for p in group["params"]:
        if p.grad is not None:            # <- 真正的过滤条件:不是"requires_grad=True"字面判断,而是".grad 不为 None"
            params.append(p); grads.append(p.grad)
            ...
```
这里有个值得注意的精确表述:源码的过滤条件是 `p.grad is not None`,不是直接检查 `requires_grad`。两者通常等价(只有 `requires_grad=True` 的叶子 tensor 才可能在 `backward()` 后拥有非 `None` 的 `.grad`),但精确来说,`step()` 真正跳过的是"这次没有梯度的参数"——可能是因为 `requires_grad=False`(压根不参与 autograd),也可能是 `requires_grad=True` 但这次前向根本没用到这个参数(loss 计算图没连到它,`.grad` 保持 `None`)。

**AI 研究场景:** 理解"只有 `.grad is not None` 的参数才会被更新"这条规则,是**冻结部分参数微调**(比如只训练新加的分类头,backbone 设 `requires_grad=False`)能生效的根本原因——被冻结的参数即使留在同一个 `optimizer` 里,只要它们的 `.grad` 一直是 `None`,`step()` 就会自动跳过它们,不需要手动从 `param_groups` 里摘除。

**可运行例子(验证"in-place"+"跳过无梯度参数"这两条机制):**
```python
import torch

w1 = torch.tensor([1.0], requires_grad=True)   # 会被用到,会有梯度
w3 = torch.tensor([3.0], requires_grad=True)   # requires_grad=True,但不参与 loss,.grad 会是 None

loss = (w1 * 3).sum()
opt = torch.optim.SGD([w1, w3], lr=0.1)
opt.zero_grad(); loss.backward()
assert w1.grad is not None and w3.grad is None

ptr_w1, ptr_w3, val_w3 = w1.data_ptr(), w3.data_ptr(), w3.item()
opt.step()

assert w1.data_ptr() == ptr_w1              # in-place:同一块内存,没有重新分配
assert torch.allclose(w1, torch.tensor([0.7]))   # 1.0 - 0.1*3.0 = 0.7,确实更新了
assert w3.data_ptr() == ptr_w3 and w3.item() == val_w3   # .grad is None -> 完全跳过,数值和地址都没变
```

**面试怎么问 + 追问链:**
- **Q:** "`optimizer.step()` 内部做了什么?"—— 期望说出"遍历 param_groups,对有梯度的参数按公式原地更新",而不是"更新参数"这种同义反复。
- **追问 1:** "'原地更新'具体体现在哪?为什么要原地?"—— 期望连回 [01-tensor-memory-model.md](01-tensor-memory-model.md) 第 6 节:省掉每一步都重新分配一份和参数等大的内存,这在大模型场景下是不可忽视的开销,能现场验证 `data_ptr()` 不变是加分项。
- **追问 2(容易漏答):** "如果一个参数 `requires_grad=True`,但这次前向没用到它,`step()` 会更新它吗?"—— 期望答"不会",并且能说出准确原因是"`.grad` 仍是 `None`,被 `_init_group` 的过滤条件跳过",而不是笼统地说"因为它不需要梯度"。
- **追问 3(开放,连接实践):** "如果我想冻结模型的一部分参数不参与训练,除了 `requires_grad=False`,直接不把这些参数传给 optimizer 构造函数可以吗?两种做法有什么区别?"—— 期望能对比:两种做法效果类似(都不会被更新),但 `requires_grad=False` 还会让 autograd 在构建计算图时直接跳过这部分子图(省反向计算量),而"传不传给 optimizer"只影响 `step()` 这一层,如果没同时设 `requires_grad=False`,反向传播仍然会经过这部分参数并计算梯度(只是没人用),白白浪费算力。

**常见坑:** 混淆"`step()` 跳过没梯度的参数"和"`step()` 会自动帮你设置 `requires_grad=False`"——`step()` 只是被动地根据当前 `.grad` 状态决定要不要更新,它不会反过来修改任何参数的 `requires_grad` 属性。另外,源码里其实还有 `_multi_tensor_sgd`(foreach,批量向量化)和 `_fused_sgd`(fused kernel)两套并行实现,默认优先走 foreach 路径而不是这里读到的 `_single_tensor_sgd`——三者数学上完全等价,只是性能优化层面的选择,读源码时不要因为看到多套实现而困惑。

---

## 8. `param_groups` 机制

**是什么:** 一个 `Optimizer` 内部维护一个 `list[dict]`(`self.param_groups`),每个 dict 是一"组"参数 + 这一组专属的超参数(`lr`/`weight_decay`/`momentum` 等),没显式指定的超参数会从构造函数的"全局默认值"里回填。

**一句话:** 同一个优化器实例可以同时管理多组参数,每组用不同的学习率/权重衰减等超参数训练——最常见的场景就是"预训练 backbone 用小学习率微调,新加的分类头用大学习率从头训练"。

**底层机制/为什么这样设计(源码确认默认值回填规则):**

读 `Optimizer.add_param_group` 源码,关键逻辑是:
```python
for name, default in self.defaults.items():
    if default is required and name not in param_group:
        raise ValueError(...)          # 没提供、且这个超参数是"必填"的,直接报错
    else:
        param_group.setdefault(name, default)   # 否则:这一组没写的超参数,用全局默认值补上
```
这解释了为什么可以这样写:某一组显式给 `lr`,另一组不给,不给的那组会自动使用构造函数顶层传的 `lr`;`weight_decay` 等其他超参数同理。

**AI 研究场景(backbone 小 lr 微调 + 新分类头大 lr 从头训练):**
```python
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)   # 假装是预训练好的
        self.head = nn.Linear(4, 2)        # 假装是新加的、随机初始化

    def forward(self, x):
        return self.head(torch.relu(self.backbone(x)))

model = TinyModel()
backbone_lr, head_lr = 1e-4, 1e-2
opt = torch.optim.SGD([
    {"params": model.backbone.parameters(), "lr": backbone_lr},
    {"params": model.head.parameters(), "lr": head_lr},
])
assert opt.param_groups[0]["lr"] == backbone_lr
assert opt.param_groups[1]["lr"] == head_lr
```

**可运行例子(验证每组真的按自己的 lr 更新,而不是共享一个全局 lr):**
```python
x = torch.randn(3, 4); target = torch.randn(3, 2)
loss = ((model(x) - target) ** 2).mean()
opt.zero_grad(); loss.backward()

w_backbone_before = model.backbone.weight.detach().clone()
w_head_before = model.head.weight.detach().clone()
g_backbone = model.backbone.weight.grad.clone()
g_head = model.head.weight.grad.clone()
opt.step()

# 注意:backbone_lr=1e-4 让 delta 量级降到 ~1e-5/1e-6,float32 下默认 atol=1e-8 反而会因为
# "参数相减" vs "lr*grad直接算"两条计算路径的舍入误差而误判 —— 又一次印证 01 篇的教训:
# float32 比较必须用足够宽松、且符合数值量级的 atol,不能死认一个很小的默认值。
backbone_delta = w_backbone_before - model.backbone.weight.detach()
head_delta = w_head_before - model.head.weight.detach()
assert torch.allclose(backbone_delta, backbone_lr * g_backbone, atol=1e-7)
assert torch.allclose(head_delta, head_lr * g_head, atol=1e-7)
# 实测:backbone 权重变化量 == 1e-4 * 梯度;head 权重变化量 == 1e-2 * 梯度 —— 两组各用各的 lr,互不影响

# 未显式指定的超参数会从全局默认值回填
opt2 = torch.optim.SGD([
    {"params": model.backbone.parameters(), "lr": backbone_lr, "weight_decay": 0.0},
    {"params": model.head.parameters()},              # lr、weight_decay 都没写
], lr=head_lr, weight_decay=1e-3)
assert opt2.param_groups[1]["lr"] == head_lr           # 回填自顶层默认值
assert opt2.param_groups[1]["weight_decay"] == 1e-3     # 同上

# 训练中途动态解冻更多参数(常见微调套路):
extra = nn.Linear(2, 2)
opt.add_param_group({"params": extra.parameters(), "lr": 5e-3})
assert len(opt.param_groups) == 3
```

**面试怎么问 + 追问链:**
- **Q:** "如果想让模型不同部分用不同学习率训练,PyTorch 里怎么做?"—— 期望直接说出 `param_groups`,并给出 backbone/head 这类具体例子。
- **追问 1:** "如果某一组没写 `weight_decay`,它会用什么值?"—— 期望知道"回填 optimizer 构造函数顶层传入的默认值",而不是"报错"或"用某个写死的默认值"这类不准确的猜测。
- **追问 2(工程向):** "训练到一半想解冻更多参数、加入新的学习率配置,需要重新创建一个 optimizer 吗?"—— 期望知道 `add_param_group()` 可以动态追加,不需要重建整个 optimizer(重建的代价是会丢失已有参数组的 momentum/Adam 状态等历史统计量)。

**常见坑:** 以为每个 `nn.Module` 的参数只能整体属于一个 `param_group`——实际上 `param_groups` 的划分粒度完全由你在构造 optimizer 时怎么分组决定,可以精细到"某一层的 `weight` 一组、`bias` 另一组"(比如常见的"bias 和 LayerNorm 参数不做 weight decay"配置,就是靠这种精细分组实现的,不是优化器自动识别参数类型)。

---

## 9. 学习率调度器原理:StepLR / CosineAnnealingLR / OneCycleLR + warmup

**是什么:** `lr_scheduler` 系列类,在训练过程中按预定规则修改 `optimizer.param_groups[i]["lr"]`,不参与梯度计算,只是"定时改一个数字"。

**一句话:** 优化器只知道"当前这一步该往哪走、走多远(由当前 `lr` 决定)",调度器负责"随着训练推进,把 `lr` 这个旋钮按规划好的曲线拧到哪"——两者是完全解耦的两层机制。

**底层机制/为什么这样设计(逐个读源码公式,逐个验证):**

**StepLR**:每隔 `step_size` 个 epoch,把 `lr` 乘以 `gamma`(阶梯式衰减)。
```python
w = nn.Parameter(torch.zeros(1))
opt = torch.optim.SGD([w], lr=0.1)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
lrs = [opt.param_groups[0]["lr"]]
for epoch in range(10):
    opt.step(); sched.step()
    lrs.append(opt.param_groups[0]["lr"])
# 实测: [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0.0125, 0.0125]
manual = [0.1 * 0.5 ** (e // 3) for e in range(11)]
assert lrs == manual   # 精确匹配 base_lr * gamma**(epoch//step_size),阶梯形状
```

**CosineAnnealingLR**:源码文档给出的闭式解是 `eta_min + 0.5*(base_lr-eta_min)*(1+cos(pi*t/T_max))`——`t=0` 时 `cos(0)=1`,值等于 `base_lr`;`t=T_max` 时 `cos(pi)=-1`,值等于 `eta_min`;中间平滑地沿余弦曲线下降(不是直线,下降速度在两端更慢、中段更快)。
```python
import math
T_max = 10
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=T_max, eta_min=0.0)
# ...(同样跑10个epoch,记录 lrs2)
manual2 = [0.5 * 0.1 * (1 + math.cos(math.pi * e / T_max)) for e in range(11)]
assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(lrs2, manual2))
# 实测: [0.1, 0.09755, 0.09045, 0.07939, 0.06545, 0.05, 0.03455, 0.02061, 0.00955, 0.00245, 0.0]
# 单调递减,验证通过;形状是"两端平、中间陡"的余弦曲线,不是StepLR的阶梯,也不是线性衰减
```

**OneCycleLR**:训练前 `pct_start` 比例的步数里,`lr` 从 `max_lr/div_factor` **线性/余弦上升**到 `max_lr`;剩下的步数里再退火下降到 `max_lr/div_factor/final_div_factor`(一个比初始值还低得多的值)。同时(默认 `cycle_momentum=True`)momentum 会**反向**周期变化:`lr` 最高点恰好是 momentum 最低点。
```python
sched3 = torch.optim.lr_scheduler.OneCycleLR(
    opt3, max_lr=1.0, total_steps=20, pct_start=0.3, div_factor=25.0, final_div_factor=1e4)
# 实测 lr 序列(20步): 0.04 -> 一路升到 1.0(第5步,约等于 pct_start*20=6 附近)-> 一路降到 0.000004
# 实测 momentum 序列: 0.95 -> 降到 0.85(同样在第5步触底)-> 升回 0.95  —— 和 lr 曲线完全反向同步
assert math.isclose(sched3_lrs[0], 1.0/25.0, rel_tol=1e-4)          # 验证初始 lr = max_lr/div_factor
peak_step = sched3_lrs.index(max(sched3_lrs))
assert sched3_lrs[-1] < sched3_lrs[peak_step]                        # 验证确实"先升后降"
```

**warmup 的作用:** 手写一个"前几步线性升温、之后余弦衰减"的调度,并且验证它和 PyTorch 官方 `SequentialLR(LinearLR, CosineAnnealingLR)` 组合出来的曲线完全一致:
```python
warmup_steps, total, base_lr = 5, 20, 0.1
def lr_at(step):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))

# 官方组合:LinearLR(热身) + CosineAnnealingLR(退火),用 SequentialLR 拼接
warmup_sched = torch.optim.lr_scheduler.LinearLR(opt4, start_factor=1/warmup_steps, end_factor=1.0, total_iters=warmup_steps-1)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt4, T_max=total-warmup_steps)
seq_sched = torch.optim.lr_scheduler.SequentialLR(opt4, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
# 实测:手写版和 SequentialLR 组合版,20步的lr序列逐点相等(都是 [0.02, 0.04, 0.06, 0.08, 0.1, 0.1, 0.0989, ...])
```
**为什么要 warmup:** 训练刚开始时,模型参数是随机初始化的,梯度方向和量级都很不可靠;如果用 Adam,还叠加了第 5 节验证过的问题——`m`、`v` 这两个统计量本身也是刚开始累积,即使做了 bias correction 消除系统性偏差,**估计的方差依然很大**(只用了 1~2 个样本估计出来的均值/方差,置信度天然低)。用一个很小的 `lr` 开局,相当于在"统计量还不可靠"的这段时间里主动降低每一步的影响力,等 `m`/`v` 累积了足够多的历史、`beta^t` 充分衰减之后,再把 `lr` 拉到正常水平——这和 bias correction 要解决的问题是同一层"训练早期统计量不稳定"的问题,只是 bias correction 是解析修正,warmup 是更保守的工程手段(两者经常同时使用,并不互斥)。

**AI 研究场景:** Transformer 训练几乎标配 warmup(原始 Transformer 论文的 `noam` 调度就是"线性warmup + 平方根衰减");OneCycleLR(及其推广"超收敛"思路)在有限算力的快速训练场景很受欢迎,因为它允许用远高于常规配置的峰值学习率而不发散。

**面试怎么问 + 追问链:**
- **Q:** "StepLR、CosineAnnealingLR、OneCycleLR 分别是什么形状,分别适合什么场景?"—— 期望能画出/描述三条曲线的直观形状(阶梯 vs 平滑余弦 vs 先升后降),而不是只记名字。
- **追问 1:** "为什么训练要用 warmup,不能一开始就用目标 lr 吗?"—— 期望连接到"早期梯度/统计量不可靠"这层原因,最好能提到和 bias correction 是同一类问题的不同应对方式。
- **追问 2(工程向):** "OneCycleLR 里 momentum 为什么要和 lr 反向变化?"—— 期望能说出直觉:lr 高的阶段,单步移动本身已经很大,如果 momentum 还很高会进一步放大有效步长、容易震荡(呼应第 2 节"lr 偏大时 momentum 放大震荡"的验证结论);lr 低的阶段,momentum 高一点有助于沿着已经建立的方向继续推进。

**常见坑:** 把 scheduler 的 `step()` 调用时机搞混——大部分 epoch 级调度器(`StepLR`/`CosineAnnealingLR`)应该在每个 epoch 结束后调用一次 `scheduler.step()`,而 `OneCycleLR` 明确要求**每个 batch(每次 `optimizer.step()`)之后**都调用一次(文档原话:"changes the learning rate after every batch"),搞混调用粒度会导致 lr 曲线的实际形状和预期完全不一致(比如把 OneCycleLR 放在 epoch 循环里调用,总步数会被严重低估)。

---

## 10. `clip_grad_norm_` 的实现原理

**是什么:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**一句话:** 把所有参数的梯度拼在一起,当成一个超长向量算它的**全局** L2 范数,如果这个范数超过阈值,就按同一个比例把**所有**梯度一起缩小到阈值以内——方向完全不变,只压缩大小,防止某一步梯度爆炸导致参数更新失控。

**底层机制/为什么这样设计(源码确认两个容易被忽略的细节):**

读 `torch/nn/utils/clip_grad.py` 源码,核心公式(函数文档里也直接写出来了):
```python
grad = grad * min(max_norm / (total_norm + 1e-6), 1)
```
两个手写实现时容易漏掉、但源码里明确存在的细节:
1. 分母上有一个 **`1e-6` 的 eps**,防止 `total_norm` 恰好为 0 时除零。
2. `clip_coef` 会被 `clamp(max=1.0)`——**只会缩小,不会放大**:如果 `total_norm` 本来就小于 `max_norm`,`clip_coef` 会大于 1,但被 clamp 到 1,梯度原样不动(这是个 no-op,不是"扩大梯度让它正好等于 max_norm"这种反直觉行为)。

**"全局"这个词的精确含义:** 不是对每个参数张量各自算范数、各自裁剪(那样等价于给每个参数单独设了一个 `max_norm`,层数越多总的"允许幅度"上限相当于被放大),而是把所有参数的梯度**展平拼接成一个向量**之后统一求一次范数——这意味着如果模型里只有一个参数梯度巨大、其余都正常,巨大的那个也会拖累整体范数超标,导致所有参数的梯度都被按同一比例压缩,即使大部分参数的梯度本身很正常。

**AI 研究场景:** RNN/Transformer 训练中"梯度爆炸"是经典问题(尤其序列较长、或者训练不稳定的早期阶段),`clip_grad_norm_` 是几乎所有大模型训练脚本的标配防御措施,通常配合一个较大的 `max_norm`(比如 1.0)使用,只在梯度真正异常时触发,不干扰正常训练。

**可运行例子(手写实现 vs torch 内置,多形状参数验证"全局"而非"逐张量"):**
```python
import torch

def manual_clip_grad_norm_(parameters, max_norm, eps=1e-6):
    params = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in params))  # 全局范数
    clip_coef = torch.clamp(max_norm / (total_norm + eps), max=1.0)             # 只缩小,不放大
    for p in params:
        p.grad.detach().mul_(clip_coef)
    return total_norm

# 三个不同形状的参数,故意给很大的梯度让裁剪真正触发
torch.manual_seed(42)   # 固定种子,保证下面的具体数值可复现(呼应01篇"别凭记忆里的数字"的教训)
p1 = torch.nn.Parameter(torch.randn(3, 4));    p1.grad = torch.randn(3, 4) * 10
p2 = torch.nn.Parameter(torch.randn(5));        p2.grad = torch.randn(5) * 50
p3 = torch.nn.Parameter(torch.randn(2, 2, 2));  p3.grad = torch.randn(2, 2, 2) * 1

grads_backup = [p1.grad.clone(), p2.grad.clone(), p3.grad.clone()]
max_norm = 5.0

torch_norm = torch.nn.utils.clip_grad_norm_([p1, p2, p3], max_norm=max_norm)

p1b, p2b, p3b = [torch.nn.Parameter(torch.zeros_like(p)) for p in (p1, p2, p3)]
p1b.grad, p2b.grad, p3b.grad = [g.clone() for g in grads_backup]
manual_norm = manual_clip_grad_norm_([p1b, p2b, p3b], max_norm=max_norm)

# torch 内部是"对每个张量先分别求范数,再对这些范数求一次范数",和"直接拼成一个大向量求范数"
# 数学上等价,但浮点求和顺序不同,会有 float32 级别的极小误差(实测约 1e-5 相对误差),
# allclose 的 atol 需要放宽到 1e-4 左右,再一次印证"float32 别用 =="这条 01 篇的教训
assert torch.allclose(torch_norm, manual_norm, atol=1e-4)
for g_torch, g_manual in zip([p1.grad, p2.grad, p3.grad], [p1b.grad, p2b.grad, p3b.grad]):
    assert torch.allclose(g_torch, g_manual, atol=1e-6)

# 裁剪后全局范数应精确等于 max_norm(因为原始范数确实超标了)
post_norm = torch.cat([p1.grad.flatten(), p2.grad.flatten(), p3.grad.flatten()]).norm(2)
assert abs(post_norm.item() - max_norm) < 1e-3

# 方向不变:每个参数的梯度都被同一个标量 clip_coef 缩放,不是各自独立缩放
# 用上面已经算出的 torch_norm(真实原始范数),而不是凭印象写一个硬编码数字——
# 种子42下实测 raw_total_norm≈72.093,clip_coef=5.0/(72.093+1e-6)≈0.06936
clip_coef = (max_norm / (torch_norm + 1e-6)).item()
for g_before, g_after in zip(grads_backup, [p1.grad, p2.grad, p3.grad]):
    ratio = g_after / g_before
    assert torch.allclose(ratio, torch.full_like(ratio, ratio.flatten()[0].item()), atol=1e-4)
    assert torch.allclose(ratio, torch.full_like(ratio, clip_coef), atol=1e-4)

# 未超标时是 no-op
p_small = torch.nn.Parameter(torch.zeros(3)); p_small.grad = torch.tensor([0.01, 0.01, 0.01])
g_before_small = p_small.grad.clone()
torch.nn.utils.clip_grad_norm_([p_small], max_norm=100.0)
assert torch.allclose(p_small.grad, g_before_small)
```

**面试怎么问 + 追问链:**
- **Q:** "`clip_grad_norm_` 是怎么实现的?"—— 期望说出"把所有参数梯度拼接成一个向量算全局 L2 范数,超过阈值就按同一比例整体缩小",而不是"把每个梯度限制在某个范围内"(那是 `clip_grad_value_`,是完全不同的机制,按元素裁剪而不是按整体范数缩放,常被面试者搞混)。
- **追问 1(区分度高):** "'全局'范数和'每个参数张量各自'求范数裁剪,有什么区别?"—— 期望能说出"全局范数下,一个参数梯度异常会拖累所有参数一起被压缩;逐张量裁剪则每个参数有独立的上限,总体允许的'能量'会随参数量增多而变相放大",最好能举出模型层数越多、逐张量裁剪的总体效果越宽松这个推论。
- **追问 2:** "如果梯度本来就没超过 `max_norm`,会发生什么?"—— 期望答"no-op,不会被放大凑到刚好等于 max_norm",能提到源码里 `clamp(max=1.0)` 这个细节是加分项。
- **追问 3(常见混淆):** "`clip_grad_norm_` 和 `clip_grad_value_` 有什么区别?"—— 期望清楚区分"前者基于全局范数整体等比例缩放(保方向),后者是逐元素截断到 `[-clip_value, clip_value]`(不保方向,极值维度会被强行削平,其余维度不受影响)"。

**常见坑:** 忘记调用时机——`clip_grad_norm_` 必须在 `loss.backward()` 之后、`optimizer.step()` 之前调用(此时 `.grad` 才被填充,且还没被用去更新参数);另一个常见错误是以为它会返回"裁剪后"的范数,实际上返回值是**裁剪前**的原始 `total_norm`(可以用来监控训练中梯度范数的变化趋势,是排查梯度爆炸的常用指标)。

---

## 小结:这一批 10 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | SGD 基础更新公式 | `param -= lr*grad`,一阶泰勒展开保证只要 lr 够小 loss 必然下降 |
| 2 | Momentum 推导 | PyTorch 实际公式 `v=momentum*v+grad; param-=lr*v`(lr在递推外),与"其他框架"写法仅在 lr 不变时等价;惯性累积一致方向、抵消震荡方向 |
| 3 | Nesterov Momentum | 不做字面预看(不多一次前向反向),用代数重组 `grad+momentum*buf` 精确等价于预看梯度(已数值验证到机器精度) |
| 4 | Adam 一阶二阶矩 | `m`=降噪梯度方向,`v`=梯度平方历史,`sqrt(v)`做分母实现每参数独立自适应缩放 |
| 5 | bias correction | `m_t=(1-beta1^t)*E[g]`,除以`(1-beta1^t)`精确修正;不修正时更新幅度不是简单变小,而是可能明显变大(已验证) |
| 6 | AdamW vs Adam+L2 | L2混入梯度被自适应缩放"扭曲"(隔离实验测得达约1000倍力度差异);AdamW解耦、均匀收缩,PyTorch里AdamW字面就是Adam的一个flag |
| 7 | `optimizer.step()`内部 | 遍历param_groups,对`.grad is not None`的参数原地更新,呼应01篇in-place机制 |
| 8 | `param_groups`机制 | 一个optimizer管理多组参数各自的超参数,未指定项回填全局默认值,可动态`add_param_group` |
| 9 | 学习率调度器+warmup | StepLR阶梯、CosineAnnealingLR平滑余弦、OneCycleLR先升后降;warmup缓解早期统计量不稳定,和bias correction同源问题不同解法 |
| 10 | `clip_grad_norm_` | 全局L2范数(拼接所有梯度),超阈值按统一比例整体缩小(保方向),只缩小不放大 |

下一批:[07-training-loop-internals.md](07-training-loop-internals.md) —— 训练循环深层机制(`zero_grad(set_to_none=True)`、梯度累加、`autocast`/`GradScaler`、gradient checkpointing)。

---

*更新:2026-07-07*
