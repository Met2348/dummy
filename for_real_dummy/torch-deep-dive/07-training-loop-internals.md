# 07 · 训练循环深层机制(Training Loop Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 前几批分别讲透了 autograd(02)、nn.Module(03)、层的数学(04)、损失函数(05)、优化器(06)——这些都是训练循环里**单个环节**的内部机制。本篇讲的是把这些环节**组装成一个真实、能跑在生产环境里的训练 step** 时,还需要额外理解的一层机制:`zero_grad` 到底清空了什么、梯度累加为什么不总是"等价于大 batch"、混合精度训练(`autocast`+`GradScaler`)怎么在省显存和不炸精度之间找平衡、gradient checkpointing 怎么用算力换显存、`torch.compile` 怎么让同一段代码跑得更快。这几块单独拿出来看都不难,但**组装顺序错了会产生不报错、但悄悄训练失败的 bug**——第 7 节会现场复现一个这样的真实陷阱。

**本篇和 06 篇学习率 warmup 内容的关系:** 06 篇第 9 节已经把"为什么需要 warmup"、`StepLR`/`CosineAnnealingLR`/`OneCycleLR`/warmup 的完整数学和源码讲透了,本篇不重复推导——第 7 节组装完整训练 step 时会明确指出 `scheduler.step()` 在循环里的具体位置,直接引用 06 篇的结论,不再重新证明一遍。

**验证方法论(与前几批一致):** 本文所有代码已在仓库 `.venv`(torch 2.11.0+cu128,GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU)下实际跑通验证,给出的具体数字都是本机实测,不是转述文档;凡是随机性/硬件相关、可能因环境而异的数字,会标注"方向稳定,具体数值会因环境而异"。

**本篇统一结构(同前几批):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(现场验证,不转述)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. `zero_grad(set_to_none=True)` —— 不是"清零",是"扔掉"

**是什么:**
```python
optimizer.zero_grad(set_to_none=True)   # 当前 PyTorch 版本的默认值就是 True(已验证签名)
```

**一句话:** 老版本的默认行为是把每个参数的 `.grad` 原地清零(`grad.zero_()`,tensor 对象还在,只是数值变 0);`set_to_none=True` 则是直接把 `.grad` 设成 `None`——不是"变成一个值为0的梯度",而是"这个参数暂时没有梯度这回事",这个区别在优化器眼里是两种完全不同的状态。

**底层机制/为什么这样设计:**

现场读官方 docstring(已验证,一字不差):

> "Instead of setting to zero, set the grads to None. This will in general have lower memory footprint, and can modestly improve performance. However, it changes certain behaviors. ... **`torch.optim` optimizers have a different behavior if the gradient is 0 or None (in one case it does the step with a gradient of 0 and in the other it skips the step altogether).**"

最后这句话是本节的核心,也是最容易被轻描淡写略过的一句——"梯度是0"和"梯度是None"对 `optimizer.step()` 而言不是同一件事(呼应 [06-optimizer-internals.md](06-optimizer-internals.md) 第 7 节:`step()` 内部的过滤条件是 `p.grad is not None`,不是"梯度是不是0")。这两种状态的差异在 SGD(无 momentum)上看不出来——反正梯度是0,更新量也是0,两种写法结果一样。但在 **Adam** 这类带指数移动平均历史状态的优化器上,差异是真实存在、而且方向违反直觉的:

- `grad=None` → `step()` 直接跳过这个参数,`m`/`v` 状态维持原样,参数纹丝不动。
- `grad=zeros`(一个真实存在、只是数值为0的 tensor)→ `step()` **仍然执行**完整的 Adam 更新公式:`m = beta1*m + (1-beta1)*0 = beta1*m`,`v = beta2*v + (1-beta2)*0 = beta2*v`——`m`、`v` 只是被"衰减"了,不是被清零,如果它们之前有非零历史,`lr*m_hat/(sqrt(v_hat)+eps)` 依然是非零的,**参数会真的移动,即使这一步真实梯度是0**。

**可运行例子:**

```python
import torch

torch.manual_seed(0)
w1 = torch.tensor([1.0, 1.0], requires_grad=True)
w2 = torch.tensor([1.0, 1.0], requires_grad=True)
opt1 = torch.optim.Adam([w1], lr=0.1)
opt2 = torch.optim.Adam([w2], lr=0.1)

# 先跑5步真实梯度,让 m/v 都积累出非零历史(两边同步跑,状态完全一致)
for step in range(5):
    loss1 = (w1 * 2).sum(); opt1.zero_grad(set_to_none=True); loss1.backward(); opt1.step()
    loss2 = (w2 * 2).sum(); opt2.zero_grad(set_to_none=True); loss2.backward(); opt2.step()
assert torch.allclose(w1, w2)
# 实测:5步后 w1==w2==[0.5, 0.5]

w1_before, w2_before = w1.detach().clone(), w2.detach().clone()

opt1.zero_grad(set_to_none=True)   # grad=None,不再 backward
opt1.step()
assert torch.equal(w1, w1_before)   # 完全没动:step() 因为 grad is None 直接跳过这个参数

opt2.zero_grad(set_to_none=True)
w2.grad = torch.zeros_like(w2)      # 手动造一个"真实存在、值为0"的梯度(不是None)
opt2.step()
assert not torch.equal(w2, w2_before)   # 真的动了!
# 实测: w2 从 [0.5, 0.5] 移动到 [0.4138, 0.4138] —— 仅仅因为历史动量还在衰减,
# 即使这一步"真实梯度"确实是0
```

**AI 研究场景:** 这个区别在"部分参数不是每一步都参与计算"的场景里是真实的坑,而不是纸面推理——比如 Mixture-of-Experts 里没被路由选中的专家、条件计算(某个分支这一步没走到)、多任务学习里某个 task head 这一步没有对应数据。如果训练框架的某处不小心把"没有梯度"的参数用 `torch.zeros_like` 填了一个假梯度(而不是让它保持 `None`),Adam/AdamW 会让这些参数在"应该纹丝不动"的时候悄悄继续漂移,而且不会有任何报错或警告——这是本节验证的这个机制在真实系统里的直接后果。

**面试怎么问 + 追问链:**
- **Q:** "`zero_grad(set_to_none=True)` 和默认(旧版本)行为比,区别是什么?"—— 期望不满足于"省内存",能提到"改变了`.grad`是None还是0这个语义区别"。
- **追问 1(核心,官方文档原话就是这道题的答案):** "这个区别对训练结果有实质影响吗,还是纯粹的内部实现细节?"—— 期望答"有实质影响,尤其是 Adam 这类有状态的优化器:`grad=None` 会让 `step()` 跳过该参数,`grad=0` 仍会执行更新(只是这一步的梯度贡献是0),如果 `m`/`v` 有非零历史,后者会让参数继续移动"——这是官方 docstring 明确写出来、但很少有人真正深挖的一句话。
- **追问 2(深挖):** "什么场景下,一个参数会出现'这一步有梯度是0的tensor,而不是None'的情况?"—— 期望能想到"某个自定义训练逻辑手动构造了梯度"、或者更常见的"某些框架的旧代码路径在不该给梯度的地方误用了 `torch.zeros_like` 兜底,而不是保持 None"。
- **追问 3(工程向):** "如果我要判断某个参数这一步'到底有没有真的参与计算图',该怎么判断更可靠?"—— 期望答"检查 `param.grad is None`,而不是检查 `param.grad.abs().sum() == 0`——后者没法区分'真的没参与计算'和'参与了但刚好梯度算出来是0'这两种完全不同的情况"。

**常见坑:** 把"这一步没有梯度"错误地实现成"手动赋值一个全0梯度",而不是保持 `None` 或干脆不管——两者在无状态优化器(纯 SGD)上表现一致,一旦换成 Adam/AdamW 这类有状态优化器,这个"看似等价"的写法会让参数产生不该有的漂移,而且因为不报错、数值变化很小,极难在日常调试里被发现,通常要等到"为什么这个应该被冻结的参数,几千步之后还是变了"这种诡异现象出现才会被追查到。

---

## 2. 梯度累加(gradient accumulation)—— 对梯度是精确等价,对 BatchNorm 不是

**是什么:**
```python
optimizer.zero_grad(set_to_none=True)
for i, micro_batch in enumerate(micro_batches):
    loss = criterion(model(micro_batch), target) / accumulation_steps
    loss.backward()               # 不调用 zero_grad,梯度在多个 micro_batch 间累加
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

**一句话:** 显存装不下一次跑完整个大 batch 时,把大 batch 切成 N 个小的 micro-batch,依次跑 forward+backward 但不清空梯度(梯度在 `.grad` 上自然累加,这正是 [02-autograd-internals.md](02-autograd-internals.md) 讲过的默认累加机制),攒够 N 个 micro-batch 才真正 `step()` 一次——用"更多次小的反向传播"模拟"一次大 batch 的反向传播",省显存但不省计算量和时间。

**底层机制/为什么这样设计:**

**梯度这一侧,是精确等价的:** loss 对参数的梯度是线性可加的——`mean_loss_over_full_batch` 的梯度,严格等于"每个 micro-batch 的 mean loss 除以 accumulation_steps 之后梯度求和"(这正是 [05-loss-functions-and-numerical-stability.md](05-loss-functions-and-numerical-stability.md) 第 2 节讲过的 `reduction='mean'` 除以 batch size 的效应在多个 micro-batch 之间的自然延伸)。只要每个 micro-batch 的 loss 都除以 `accumulation_steps` 再 backward,最终累加出来的梯度,和真的一次性跑完整个大 batch 算出来的梯度,在数学上完全相同。

**但 BatchNorm 这一侧,不是等价的:** `BatchNorm` 的均值/方差是"当前这一次 forward 看到的 batch"上统计出来的——如果把一个 256 的大 batch 切成 8 个 32 的 micro-batch,每个 micro-batch 各自独立算一次均值/方差,并且各自触发一次 `running_mean`/`running_var` 的 EMA 更新(8 次独立的、以 32 为样本量的小幅更新),这和"一次性对 256 个样本算一次统计量、做一次 EMA 更新"是两个不同的数学过程,统计意义上并不等价(体现在:8 次基于 32 样本的估计,噪声天然比 1 次基于 256 样本的估计更大;而且 EMA 是非线性的滑动平均,多次小步更新和一次大步更新最终收敛到的值也不同)。这一点会在下面用真实数字验证。

**可运行例子:**

```python
import torch, torch.nn as nn

torch.manual_seed(0)
model_accum = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
model_full = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
model_full.load_state_dict(model_accum.state_dict())   # 起点完全一致
model_accum.train(); model_full.train()

torch.manual_seed(1)
full_batch = torch.randn(256, 4)
micro_batches = full_batch.split(32)   # 切成8份,和full_batch是同一批数据,不是重新采样
assert len(micro_batches) == 8

opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.01)
opt_accum.zero_grad(set_to_none=True)
for mb in micro_batches:
    loss = (model_accum(mb) ** 2).mean() / 8      # 除以累加步数,呼应05篇的reduction教训
    loss.backward()
opt_accum.step()

opt_full = torch.optim.SGD(model_full.parameters(), lr=0.01)
opt_full.zero_grad(set_to_none=True)
loss_full = (model_full(full_batch) ** 2).mean()
loss_full.backward()
opt_full.step()

# 梯度(通过Linear层weight.grad间接验证):精确一致
assert torch.allclose(model_accum[0].weight.grad, model_full[0].weight.grad, atol=1e-4)

# BatchNorm的running_mean:不一致!
bn_accum, bn_full = model_accum[1], model_full[1]
assert not torch.allclose(bn_accum.running_mean, bn_full.running_mean, atol=1e-4)
# 实测: accum running_mean=[0.117, 0.196, -0.187, -0.125]
#       full  running_mean=[0.021, 0.033, -0.033, -0.023] —— 明显不同,不是舍入误差级别的差异
```

**AI 研究场景:** 用梯度累加模拟大 batch 训练(常见于显存有限、但想复现论文里"大 batch"配置的场景)时,如果模型里有 `BatchNorm`,要清楚意识到"有效梯度确实等价于大 batch",但"`BatchNorm` 统计量不等价于大 batch"——这也是为什么很多用梯度累加训练大模型的场景会优先选择 `LayerNorm`(逐样本归一化,不依赖 batch 维度,天然不受这个问题影响,见 [04-layers-math-and-backward.md](04-layers-math-and-backward.md) 第 5 节)而不是 `BatchNorm`;真的必须用 `BatchNorm` 又要用梯度累加时,[09-distributed-training-basics.md](09-distributed-training-basics.md) 第 5 节的 `SyncBatchNorm` 思路(合并多份统计量)在原理上是同一类问题的解法,但 `SyncBatchNorm` 解决的是"跨进程"的统计量合并,不直接解决"同进程内多个 micro-batch"这个场景,不能直接套用。

**面试怎么问 + 追问链:**
- **Q:** "梯度累加和真的用大 batch 训练,是完全等价的吗?"—— **这是一道陷阱题**,大多数人会不假思索回答"是",完整回答应该是"对梯度更新是精确等价的,但如果模型里有 BatchNorm,统计量不等价"。
- **追问 1(核心,必须能讲清楚为什么):** "为什么梯度能精确等价,BatchNorm 却不行?"—— 期望答出"梯度是线性可加的运算(求和/求平均的导数还是线性运算),BatchNorm 统计量的计算(尤其是 EMA 更新)不是线性可加的,多次基于小样本的独立估计和一次基于全量样本的估计,统计意义不同"。
- **追问 2(工程向):** "如果非要用梯度累加又要用 BatchNorm,有什么缓解办法?"—— 期望能提到几个方向:换用 `LayerNorm`/`GroupNorm`(不依赖 batch 统计)、只在真正的大 batch step 上更新一次 BN 统计量(需要额外改造 forward 逻辑,不是开箱即用)、或者接受这个近似(如果 micro-batch 本身已经不算太小,统计噪声可能可以接受)。
- **追问 3(容易漏答):** "梯度累加省的是显存还是时间?"—— 期望答"只省显存(峰值激活值显存,因为每次只有一个 micro-batch 的中间激活值同时存在),不省时间甚至可能因为更多次 kernel launch 略慢——总的前向+反向计算量和不切分时完全一样,只是拆成了多次做"。

**常见坑:**
- 忘记把每个 micro-batch 的 loss 除以 `accumulation_steps`——这是 05 篇 `reduction='mean'` 教训的直接复现:不除的话,累加出来的梯度是"不切分时梯度的 `accumulation_steps` 倍",相当于学习率被隐式放大了好几倍。
- 想当然认为梯度累加"处处等价于大 batch",忽略模型里 `BatchNorm` 层的存在——上面的实测已经证明这个假设不成立。
- 每个 micro-batch 之间不小心调用了 `zero_grad()`——这会导致每次 `step()` 时只用到最后一个 micro-batch 的梯度,前面几个白算了,这是梯度累加最常见的实现 bug。

---

## 3. `autocast` 自动混合精度 —— 不是整个模型变 fp16,是"分算子决定"

**是什么:**
```python
with torch.autocast(device_type='cuda', dtype=torch.float16):   # 或 torch.bfloat16
    output = model(input)
    loss = criterion(output, target)
```

**一句话:** `autocast` 不会把模型参数或输入统一转成 fp16——它在**算子调用**这一层拦截,维护一份"哪些算子应该被自动转 fp16 计算、哪些必须强制留在 fp32"的名单,同一个 `with` 块里,矩阵乘法这类算子的输出是 fp16,而 `softmax`/`sum`/`exp`/损失函数这类数值敏感的算子,不管输入是什么精度,输出都会被强制转回 fp32。

**底层机制/为什么这样设计:**

**为什么要区别对待,不能整个网络无脑转 fp16:** fp16 用 5 位指数、10 位尾数,数值范围(约 6e-5 到 6.5e4)和精度都远小于 fp32——矩阵乘法/卷积这类算子在 Tensor Core(现代 NVIDIA GPU 上专门加速矩阵乘加运算的硬件单元,对 fp16/bf16 这类低精度输入吞吐更高,[08-memory-and-performance.md](08-memory-and-performance.md) 第 3 节会展开它偏爱什么样的内存排布)上用 fp16 计算能获得数倍加速且精度损失通常可接受(因为大量乘加操作的误差会在统计上部分抵消);但 `softmax`/`exp`/`sum`/`layer_norm`/损失函数这类算子,要么涉及大量数值的连续求和(误差会累积,不会抵消)、要么涉及指数运算(输入稍大就容易溢出,呼应 [05-loss-functions-and-numerical-stability.md](05-loss-functions-and-numerical-stability.md) 第 1 节的 log-sum-exp 数值稳定性问题),用 fp16 算这些操作,溢出/精度损失的风险明显更高。`autocast` 的解决方案是维护这份算子分类表,把"用fp16计算收益大、风险小"的算子转 fp16,把"用fp16风险大"的算子强制转回/保持 fp32,两头都要。

**现场验证这份"名单"确实分开处理不同算子(不是转述文档):**

```python
import torch, torch.nn as nn

device = 'cuda'
a = torch.randn(4, 4, device=device)

with torch.autocast(device_type='cuda', dtype=torch.float16):
    c_matmul = a @ a
    c_sum = c_matmul.sum()
    c_softmax = torch.softmax(a, dim=-1)
    c_exp = torch.exp(a)
    c_relu = torch.relu(a)
    loss = nn.functional.mse_loss(c_matmul, a)
    c_layernorm = nn.functional.layer_norm(a, (4,))

    assert c_matmul.dtype == torch.float16    # matmul: 转fp16(tensor core友好)
    assert c_sum.dtype == torch.float32        # sum: 强制fp32(累积误差敏感)
    assert c_softmax.dtype == torch.float32    # softmax: 强制fp32(指数运算,溢出敏感)
    assert c_exp.dtype == torch.float32        # exp: 强制fp32
    assert c_relu.dtype == torch.float32        # relu: 不在转换名单里,原样保留输入dtype(a是fp32)
    assert loss.dtype == torch.float32          # 损失函数: 强制fp32
    assert c_layernorm.dtype == torch.float32   # layer_norm: 强制fp32

    # relu 保留"输入dtype"而不是"强制转某个类型" —— 喂一个fp16输入进去,输出也是fp16
    relu_on_fp16 = torch.relu(c_matmul)
    assert relu_on_fp16.dtype == torch.float16

# 关键细节:模型参数本身的dtype完全没被autocast动过,只有"计算的中间结果"被转了精度
lin = nn.Linear(4, 4).to(device)
with torch.autocast(device_type='cuda', dtype=torch.float16):
    out = lin(a)
assert out.dtype == torch.float16          # 这次调用的计算结果是fp16
assert lin.weight.dtype == torch.float32    # 但权重参数本身,自始至终都是fp32!
```

**AI 研究场景:** 现代几乎所有训练脚本(尤其大模型)都会用 `autocast`(或它更常见的搭档 `bfloat16`——数值范围和 fp32 一致、只是精度更低,不容易溢出,大模型训练更常用 `bfloat16` 而不是 `float16` 正是因为这个原因),混合精度训练能在几乎不影响收敛效果的前提下显著提速、省显存,是训练大模型的标配技术,不是可选的锦上添花项。

**面试怎么问 + 追问链:**
- **Q:** "`autocast` 是把模型转成 fp16 训练吗?"—— **陷阱题**,期望答"不是,模型参数(权重)本身自始至终是 fp32,`autocast` 只在特定算子的计算过程中使用 fp16,不同算子处理方式不同"。
- **追问 1(核心):** "为什么不是所有算子都转 fp16,而要区分对待?"—— 期望能讲出"矩阵乘法这类适合 Tensor Core、误差能统计抵消的算子转 fp16 收益大风险小;softmax/exp/sum/loss 这类数值敏感、容易溢出或误差累积的算子必须强制留在 fp32",最好能举出上面验证过的具体算子例子。
- **追问 2(容易问倒):** "既然 relu 不在'转 fp16'的名单里,那 relu 处理一个 fp16 输入,输出是什么精度?"—— 期望答"fp16,relu 不强制转换,只是‘原样传递输入的精度’,不是‘强制转 fp32’——这和 sum/softmax 这类'强制fp32'的算子行为不一样,容易被搞混"。
- **追问 3(工程向):** "fp16 和 bfloat16 在 autocast 里怎么选?"—— 期望知道"bfloat16 指数位和 fp32 一样多(数值范围一致,不容易溢出,但尾数更短、精度更低),fp16 数值范围小但精度相对高;大模型训练更常用 bfloat16 是因为担心某些中间值超出 fp16 的表示范围,bfloat16 从设计上就不容易有这个问题"。

**常见坑:** 认为 `autocast` 会自动帮你把模型 checkpoint 存成 fp16、或者以为退出 `with autocast()` 块之后,块内产生的 fp16 tensor 会自动变回 fp32——都不对:`autocast` 只在其上下文范围内、对特定算子的计算精度生效,不涉及参数存储格式,也不会在退出上下文后做任何"转换回来"的操作,tensor 该是什么 dtype 就还是什么 dtype。

---

## 4. `GradScaler` —— 用一个会自动调整的缩放因子,防止 fp16 梯度下溢成 0

**是什么:**
```python
scaler = torch.amp.GradScaler(device='cuda')
scaler.scale(loss).backward()   # loss先放大scale倍再backward,让梯度也跟着放大
scaler.step(optimizer)          # 内部先把梯度除回真实大小,检测inf/nan,没问题才真的调用optimizer.step()
scaler.update()                 # 根据这一步有没有溢出,决定下一步的scale要不要调整
```

**一句话:** fp16 能表示的最小正规数远大于 fp32(约 6e-5 vs 约 1.2e-38),训练后期很多梯度值本来就很小,直接用 fp16 算反向传播容易让这些小梯度"下溢"成精确的 0(不是变小,是彻底消失,不再对参数更新有任何贡献);`GradScaler` 的解法是先把 loss 乘一个远大于 1 的缩放因子 `scale` 再做反向传播(链式法则下,梯度也会被同比例放大,给这些小梯度腾出足够的表示空间),更新参数前再除回来,并且这个 `scale` 值会根据"最近有没有出现数值溢出"动态调整,不是一个写死的常数。

**底层机制/为什么这样设计:**

**为什么先放大再缩小是安全的:** 反向传播的链式法则本质是一连串乘法,`d(scale*L)/dx = scale * dL/dx` 对每一层都成立(标量放大因子可以直接提到最外面),所以"整个 loss 乘以 scale 再反向传播"得到的每一层梯度,精确等于"真实梯度乘以 scale"——数学上是精确操作,不是近似,只要最后除回 `scale` 就能拿到真实梯度。

**动态调整 scale 的机制(现场实测验证,不是纸面描述):**

1. **正常一步:** 梯度没有 inf/nan,`scaler.step()` 真的调用 `optimizer.step()`,参数会更新。
2. **连续 `growth_interval` 步都正常:** `scale` 会按 `growth_factor` 自动调大(试探性地找一个"尽量大但还不溢出"的缩放倍数)。
3. **一旦某一步检测到 inf/nan:** `scaler.step()` **直接跳过这一步的 `optimizer.step()`**(相当于白算了这一步,参数不更新),同时把 `scale` 立刻减半,更保守地对待接下来的训练。

```python
import torch, torch.nn as nn

device = 'cuda'
torch.manual_seed(0)

# ---- 场景1:小scale,干净跑一步,应该正常更新参数 ----
model = nn.Linear(4, 4).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
scaler = torch.amp.GradScaler(device='cuda', init_scale=2.0, growth_interval=2000)
w_before = model.weight.detach().clone()
x = torch.randn(4, 4, device=device)
with torch.autocast(device_type='cuda', dtype=torch.float16):
    loss = model(x).mean()
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
assert not torch.equal(w_before, model.weight)   # 真的更新了
assert scaler.get_scale() == 2.0                   # 还没到growth_interval,scale不变

# ---- 场景2:连续多步干净训练,scale会按growth_interval自动调大 ----
model2 = nn.Linear(4, 4).to(device)
opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
scaler2 = torch.amp.GradScaler(device='cuda', init_scale=2.0, growth_interval=5, growth_factor=2.0)
scales_seen = [scaler2.get_scale()]
for i in range(12):
    x2 = torch.randn(4, 4, device=device)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss2 = model2(x2).mean()
    scaler2.scale(loss2).backward()
    scaler2.step(opt2)
    scaler2.update()
    opt2.zero_grad(set_to_none=True)
    scales_seen.append(scaler2.get_scale())
# 实测: [2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0]
# 每凑够5个"干净"的step,scale翻倍一次,精确对应 growth_interval=5
assert scales_seen == [2.0]*5 + [4.0]*5 + [8.0]*3

# ---- 场景3:故意用超大init_scale制造溢出,验证"跳过step + scale减半" ----
model3 = nn.Linear(4, 4).to(device)
opt3 = torch.optim.SGD(model3.parameters(), lr=0.1)
scaler3 = torch.amp.GradScaler(device='cuda', init_scale=2.**30, growth_interval=2000)
w3_before = model3.weight.detach().clone()
x3 = torch.randn(4, 4, device=device) * 1000   # 放大输入,配合超大scale制造真实溢出
with torch.autocast(device_type='cuda', dtype=torch.float16):
    loss3 = model3(x3).mean()
scaler3.scale(loss3).backward()
has_inf = any(torch.isinf(p.grad).any() or torch.isnan(p.grad).any()
              for p in model3.parameters() if p.grad is not None)
assert has_inf   # 确认这一步真的溢出了
scaler3.step(opt3)
scaler3.update()
assert torch.equal(w3_before, model3.weight)          # 参数完全没动:这一步被跳过了
assert scaler3.get_scale() == 2.**30 / 2               # scale立刻减半
```

**AI 研究场景:** `GradScaler` 和 `autocast` 几乎总是配对使用(`torch.amp` 模块把两者打包成一套标准的混合精度训练配方),现代训练框架(HuggingFace Trainer、PyTorch Lightning 等)都默认内置了这套逻辑,自己写训练循环时如果手动引入混合精度,必须理解"`scale` 会自动变化、某些 step 会被跳过"这两点,否则容易在 debug 训练曲线时把"这一步被跳过了所以loss没变"误判成其他 bug。

**面试怎么问 + 追问链:**
- **Q:** "为什么混合精度训练需要 `GradScaler`,只用 `autocast` 不够吗?"—— 期望答出"fp16 表示范围小,小梯度容易下溢成0,GradScaler 通过放大loss再反向传播来避免这个问题"。
- **追问 1(容易漏答):** "`scaler.step(optimizer)` 是不是每次都会真的调用 `optimizer.step()`?"—— **陷阱题**,期望答"不一定,如果这一步梯度里出现 inf/nan(说明当前的 scale 太大导致溢出了),这一步会被跳过,不执行真正的参数更新",这是本节验证的核心机制。
- **追问 2(深挖):** "`scale` 这个值是怎么变化的,一直不变还是会调整?"—— 期望答"会动态调整:连续多步没出问题会尝试调大(用更大的缩放争取覆盖更小的梯度),一旦检测到溢出立刻减半、更保守",最好能提到这是在'尽量放大来保护小梯度'和'别大到溢出'之间的动态权衡。
- **追问 3(工程向):** "如果观察到训练日志里 loss 曲线偶尔'原地不动一步',可能是什么原因?"—— 期望能联系到"可能是那一步触发了 GradScaler 的 skip,不是训练卡住了或者代码有bug",这是实际调试混合精度训练时的常见现象。

**常见坑:** 只用了 `autocast` 做前向,却忘记配套用 `GradScaler` 缩放 loss 和梯度——不会报错,但在训练后期梯度普遍偏小的阶段,会有相当比例的梯度悄悄下溢成0而不自知,导致这些参数事实上停止学习,且没有任何报错信号提示这一点;另一个常见坑是把 `scaler.update()` 忘掉或者调用顺序搞乱(必须是 `scale(loss).backward()` → `step(optimizer)` → `update()` 这个顺序),打乱顺序会让 scale 的动态调整逻辑读到错误的状态。

---

## 5. gradient checkpointing —— 用重新计算换显存

**是什么:**
```python
from torch.utils.checkpoint import checkpoint
output = checkpoint(some_module_or_function, input, use_reentrant=False)
```

**一句话:** 普通前向传播会把每一层的中间激活值都留在显存里,等反向传播用完才释放(呼应 [08-memory-and-performance.md](08-memory-and-performance.md) 第 5 节讲过的"`backward()` 用完才释放中间值"这个机制);gradient checkpointing 反其道而行:前向传播时**故意不保存**某一段区域的中间激活值,只记住"这段区域的输入是什么、要跑哪段计算",等反向传播真的需要这段区域的梯度时,**现场重新跑一遍这段区域的前向**,临时算出需要的激活值再继续反向——用多花一次前向计算的时间,换回不用同时持有这段区域全部中间结果的显存。

**底层机制/为什么这样设计:**

**现场验证"真的重新算了一遍,不是省略了这一步":**

```python
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint

device = 'cuda'
call_count = {'n': 0}

class CountingBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
    def forward(self, x):
        call_count['n'] += 1
        return torch.relu(self.lin(x))

torch.manual_seed(0)
block_ckpt = CountingBlock(256).to(device)
x = torch.randn(64, 256, device=device, requires_grad=True)

call_count['n'] = 0
out = checkpoint(block_ckpt, x, use_reentrant=False)
assert call_count['n'] == 1        # 前向阶段:forward只被真正调用了1次
out.sum().backward()
assert call_count['n'] == 2        # backward之后:变成2次 —— 反向传播时确实又跑了一遍forward

# 对比不用checkpoint的情况
call_count['n'] = 0
block_plain = CountingBlock(256).to(device)
block_plain.load_state_dict(block_ckpt.state_dict())
x2 = x.detach().clone().requires_grad_(True)
out2 = block_plain(x2)
assert call_count['n'] == 1
out2.sum().backward()
assert call_count['n'] == 1        # 不用checkpoint:backward不会重新调用forward,始终是1次

# 梯度数值完全一致 —— checkpointing是精确的,不是近似算法
assert torch.allclose(x.grad, x2.grad, atol=1e-5)
```

**显存收益(真实测量,用大 batch 让激活值显存占主导,否则参数本身的显存会掩盖掉这个效应):**

```python
def make_deep_model(width=512, n_layers=16):
    layers = []
    for _ in range(n_layers):
        layers += [nn.Linear(width, width), nn.ReLU()]
    return nn.Sequential(*layers).to(device)

BATCH = 8192
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
m1 = make_deep_model()
x1 = torch.randn(BATCH, 512, device=device, requires_grad=True)
m1(x1).sum().backward()
peak_plain = torch.cuda.max_memory_allocated()

torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

def run_checkpointed(model, x, n_chunks=4):
    def segment(x, layers):
        for l in layers: x = l(x)
        return x
    layers = list(model)
    chunk_size = len(layers) // n_chunks
    for i in range(n_chunks):
        seg = layers[i*chunk_size:(i+1)*chunk_size]
        x = checkpoint(lambda x, seg=seg: segment(x, seg), x, use_reentrant=False)
    return x

m2 = make_deep_model()
x2b = torch.randn(BATCH, 512, device=device, requires_grad=True)
run_checkpointed(m2, x2b).sum().backward()
peak_ckpt = torch.cuda.max_memory_allocated()

print(f"peak plain: {peak_plain/1024**2:.1f}MB, peak checkpointed: {peak_ckpt/1024**2:.1f}MB, "
      f"ratio: {peak_plain/peak_ckpt:.2f}x")
# 实测: peak plain=370.3MB, peak checkpointed=227.3MB, ratio=1.63x
assert peak_ckpt < peak_plain
```

**AI 研究场景:** 训练超深/超大模型(Transformer 的层数动辄几十上百层)时,前向激活值往往是显存占用的大头(呼应 08 篇第 7 节的显存 profiling 结论);gradient checkpointing 是在"batch size 拉不大"和"模型装不下"之间常用的折中手段——典型做法是每隔几层设一个 checkpoint 边界(就像上面例子里把 16 层切成 4 段),用大约"多跑 1/n_chunks 的计算量"换取显著的显存节省,让本来装不下的 batch size/模型深度变得可行。

**面试怎么问 + 追问链:**
- **Q:** "gradient checkpointing 是怎么省显存的?代价是什么?"—— 期望答"前向不保存中间激活值,反向时重新计算,用时间换空间",代价是"多了一次前向计算的时间开销"。
- **追问 1(容易漏答):** "重新计算出来的梯度,和不做 checkpoint 相比,数值上完全一样吗,还是某种近似?"—— 期望答"完全一样,是精确操作不是近似——因为重新计算用的是完全相同的输入和参数,前向计算是确定性的(排除某些非确定性算子/dropout这类特殊情况),重跑一遍得到的中间值和第一次跑的时候完全相同"。
- **追问 2(工程向):** "checkpoint 的粒度(每隔几层设一个边界)怎么选?"—— 期望能类比 [06-optimizer-internals.md](06-optimizer-internals.md) 第 8 节 `bucket_cap_mb` 的权衡思路:边界切得太细,重新计算的开销占比升高(每个小段都要重新触发一次前向,launch开销变多);切得太粗,省下的峰值显存有限(大段内部的所有激活值还是会同时存在)。
- **追问 3:** "如果一个模块内部用了 `dropout`,checkpoint 它安全吗?"—— 开放题,期望能想到"需要注意随机性:重新计算那次的随机数如果和第一次不一致,重算出来的激活值就会和第一次不同,导致梯度错误——PyTorch 的 checkpoint 机制内部有处理 RNG 状态保存/恢复的逻辑来避免这个问题,但这是使用重计算类技术时必须意识到的一个潜在陷阱"。

**常见坑:** 以为 checkpoint 是"免费"的显存优化,忽略它真实增加了训练时间(通常增加 20%~30% 量级,具体取决于切分粒度)——不是所有场景都值得用,只有"显存是当前的硬约束、时间还有富余"时才是好的权衡;另外容易忽略 `use_reentrant` 这个参数——旧的 reentrant 实现(`use_reentrant=True`,曾经的默认值)在处理某些场景(比如 checkpoint 内部又调用了 autograd 相关的高级功能)上有已知的局限和坑,官方现在推荐显式传 `use_reentrant=False`。

---

## 6. `torch.compile` 简介 —— 同一段代码,第一次慢很多、之后快很多

**是什么:**
```python
compiled_model = torch.compile(model)   # 或者直接包一个函数: torch.compile(some_fn)
```

**一句话:** `torch.compile` 不是逐行解释执行 Python 代码,而是在**第一次真正调用**时,把这次执行过程"捕获"成一张计算图(TorchDynamo 负责),再把这张图编译成针对当前硬件优化过、通常是多个算子融合在一起的高效 kernel(TorchInductor 负责生成,GPU 上通常是 Triton kernel)——这次编译有实打实的时间开销,但**之后只要输入的形状/dtype 等特征不变**,就可以直接复用编译好的结果,不用每次都重新走一遍 Python 解释器和算子调度的开销。

**底层机制/为什么这样设计:**

Eager 模式(平时默认的执行方式)每次调用 `model(x)`,都要重新走一遍 Python 解释器逐行执行 `forward` 方法、每一个算子调用都单独触发一次 kernel launch——这些"胶水"开销在小算子密集的模型上会显著累积。`torch.compile` 用 TorchDynamo 在字节码层面拦截并捕获整个(或者一大段)执行过程,变成一张静态的计算图;TorchInductor 拿到这张图后,会做算子融合(把多个能合并的小算子编译成一个大 kernel,减少 kernel launch 次数和中间结果的显存读写)、生成针对当前 GPU 优化过的代码。**这次编译是有代价的**,而且编译产物是和"这一组具体的输入形状/dtype"绑定的——换一组不同形状的输入,原来编译好的 kernel 没法直接复用,会触发**重新编译**。

**现场验证"编译一次、同形状复用、换形状重新编译"这三个阶段:**

```python
import torch, time

device = 'cuda'
def f(x):
    return torch.relu(x @ x.T).sum()

compiled_f = torch.compile(f)
x = torch.randn(512, 512, device=device)

torch.cuda.synchronize(); t0 = time.perf_counter()
out1 = compiled_f(x)
torch.cuda.synchronize()
t_first = time.perf_counter() - t0          # 第一次调用:包含真正的编译时间

torch.cuda.synchronize(); t0 = time.perf_counter()
out2 = compiled_f(x)
torch.cuda.synchronize()
t_second = time.perf_counter() - t0          # 第二次调用,同样是512x512:直接复用编译产物

x_diff_shape = torch.randn(256, 256, device=device)
torch.cuda.synchronize(); t0 = time.perf_counter()
out3 = compiled_f(x_diff_shape)
torch.cuda.synchronize()
t_third = time.perf_counter() - t0            # 换形状:触发重新编译

print(f"first(compile+run)={t_first:.3f}s  second(cached)={t_second:.4f}s  "
      f"third(new shape, recompile)={t_third:.3f}s")
# 实测: first=1.539s  second=0.0005s  third=0.639s
# 二/一 的比值是 3186x —— 编译产物复用带来的差距非常悬殊(方向稳定,具体倍数因硬件/模型而异)
assert t_second < t_first / 100
assert t_third > t_second * 100    # 换形状后明显比"缓存命中"慢,虽然比首次编译快一些
```

**AI 研究场景:** `torch.compile` 是 PyTorch 2.x 系列的核心性能特性,一行代码包一下模型往往就能拿到有意义的提速(具体收益因模型结构、是否有大量小算子/控制流而差异很大),但"输入形状变了要重新编译"这个特性,意味着**训练循环里如果每个 batch 的形状不固定**(比如 NLP 任务里不同 batch 的序列长度不一样、又没有做 padding 对齐),会导致几乎每个 step 都触发重新编译,不但吃不到编译带来的提速,反而因为频繁编译的开销让训练比不用 `torch.compile` 还慢——这是本节验证的现象在真实训练场景里最直接的坑,也是"变长序列任务用 `torch.compile` 之前通常需要先把输入 pad/bucket 到少数几个固定长度"这条工程经验的根本原因。

**面试怎么问 + 追问链:**
- **Q:** "`torch.compile` 大概是怎么工作的,为什么能提速?"—— 期望能提到"字节码层面捕获计算图 + 算子融合编译成优化kernel",不需要精确到 Dynamo/Inductor 的具体名字,但要理解"减少了Python调度开销和kernel launch次数"这个核心收益来源。
- **追问 1(核心,容易被面试官用来筛选'只会用不懂原理'的人):** "如果我的训练数据每个 batch 的形状都不一样(比如变长序列),`torch.compile` 还会有效吗?"—— 期望答"不一定,甚至可能更慢——因为输入形状变化会触发重新编译,如果每个batch形状都不同,相当于每步都要重新编译一次,编译开销可能超过它带来的执行加速",这是本节验证的核心陷阱。
- **追问 2(工程向):** "怎么缓解'形状总在变导致反复编译'这个问题?"—— 期望能提到"把不同长度的输入 pad/bucket 到少数几个固定档位的长度,用空间换'编译复用次数'",或者提到 PyTorch 也在推进的动态形状(dynamic shapes)支持来缓解这个问题,不强求候选人对动态形状机制的内部细节很熟悉。
- **追问 3:** "第一次调用为什么慢,后面为什么快这么多?"—— 期望能讲出"第一次要做完整的图捕获+编译,后面只要形状匹配就直接执行编译好的kernel,不用重新走Python解释器和逐算子调度",最好能提到这也是为什么训练脚本通常会建议"用一两个真实 batch 做 warmup 调用"再开始正式计时/训练。

**常见坑:** 在形状经常变化的场景(变长序列、动态 batch size)盲目套用 `torch.compile` 却没有意识到重复编译的开销,反而拖慢训练;另外 `torch.compile` 并不保证对所有 Python 代码都能完整捕获成一张图——遇到不支持的构造(某些复杂控制流、部分第三方库调用)会产生"graph break",退回到 eager 模式执行那一小段,不会报错但也吃不到编译收益,想确认是否发生了 graph break、发生在哪里,需要用官方提供的调试工具(比如 `torch._dynamo.explain`)排查,不能仅凭"代码跑起来没报错"就认为整个模型都被成功编译了。

---

## 7. 组装一个完整训练 step —— 顺序错了,不报错但悄悄训练失败

把前 6 节的机制,加上 06 篇的优化器/学习率调度、05 篇的损失函数,拼成一个生产环境里真实会写的训练 step:

```python
scaler = torch.amp.GradScaler(device='cuda')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)   # 06篇第9节

for epoch in range(total_epochs):
    for i, (x, y) in enumerate(dataloader):
        with torch.autocast(device_type='cuda', dtype=torch.float16):    # 第3节
            loss = criterion(model(x), y) / accumulation_steps            # 05篇 + 第2节

        scaler.scale(loss).backward()                                     # 第4节:放大后再backward

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)                                    # 关键:裁剪前必须先unscale!
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 06篇第10节
            scaler.step(optimizer)                                        # 内部会再检测一次inf/nan
            scaler.update()
            optimizer.zero_grad(set_to_none=True)                        # 第1节

    scheduler.step()   # epoch级调度器(OneCycleLR则要放进内层、每个batch后调,06篇第9节已强调)
```

把上面这段代码的控制流画成一张图更容易看清"哪些步骤每个 batch 都做、哪些步骤只在攒够 `accumulation_steps` 时才做":

```text
dataloader 每吐出一个 batch,下面这条路径都会走一次:

  autocast 前向(第3节) ──► loss = criterion(...) / accumulation_steps
        │
        ▼
  scaler.scale(loss).backward()(第4节,.grad 里此刻累加的是被放大 scale 倍的梯度)
        │
        ▼
  (i+1) % accumulation_steps == 0 ?
     │否 ────────────────────────────► 直接回去读下一个 batch(不做下面任何一步)
     │是
     ▼
  ① scaler.unscale_(optimizer)            —— 梯度先除回真实尺度
        │
        ▼
  ② clip_grad_norm_(max_norm=1.0)(06篇第10节) —— 必须排在①之后!
        │                                     顺序颠倒 = 裁的是放大过 scale 倍的梯度,
        │                                     等效阈值被错误压缩到 max_norm/scale
        ▼
  ③ scaler.step(optimizer)                —— 内部再查一次 inf/nan,没问题才真正更新参数
        │
        ▼
  ④ scaler.update()                       —— 按这一步有没有溢出,调整下一步的 scale
        │
        ▼
  ⑤ optimizer.zero_grad(set_to_none=True)(第1节)

每个 epoch 结束后,在最外层循环单独执行一次:
  scheduler.step()(06篇第9节;OneCycleLR 例外,要挪进上面这条路径内部、每个 batch 都调一次)
```

**这一节唯一的"新知识点"、也是最容易被忽略的一处顺序陷阱:`clip_grad_norm_` 必须在 `scaler.unscale_(optimizer)` **之后**调用,不能直接对 `scaler.scale(loss).backward()` 产生的梯度做裁剪。**

**底层机制/为什么这样设计:**

`scaler.scale(loss).backward()` 算出来的 `.grad`,是被 `scale`(可能是几百到几千倍)放大过的,不是真实梯度。如果这时候直接用 `clip_grad_norm_(max_norm=1.0)` 去裁剪——`max_norm=1.0` 这个阈值是按"真实梯度的合理范数"设的,但传进去的梯度是被放大过的,`clip_grad_norm_` 会把这个**远超1.0的放大梯度**强行裁剪到全局范数恰好等于 1.0(呼应 [06-optimizer-internals.md](06-optimizer-internals.md) 第 10 节 `clip_grad_norm_` "只缩小、把范数压到阈值"的机制)——问题是,这个"压到1.0"是在**被放大过的尺度**上做的,换算回真实梯度尺度,相当于把真实梯度裁剪到了 `max_norm / scale` 这么小的范数,比你原本想要的阈值离谱地小了好几个数量级。

**精确量化这个陷阱(现场实测):**

```python
import torch, torch.nn as nn

device = 'cuda'
torch.manual_seed(0)
model = nn.Linear(4, 4).to(device)
x = torch.randn(4, 4, device=device)
scale_factor = 1024.0

def global_norm(params):
    return torch.sqrt(sum(p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None))

# 路径A(错误顺序):忘记unscale,直接用"真实阈值"max_norm=1.0裁剪仍被放大过的梯度
modelA = nn.Linear(4, 4).to(device); modelA.load_state_dict(model.state_dict())
scalerA = torch.amp.GradScaler(device='cuda', init_scale=scale_factor)
with torch.autocast(device_type='cuda', dtype=torch.float16):
    lossA = modelA(x).mean()
scalerA.scale(lossA).backward()
torch.nn.utils.clip_grad_norm_(modelA.parameters(), max_norm=1.0)   # 错误:裁的是放大过的梯度
norm_after_clip_A = global_norm(modelA.parameters()).item()
effective_real_norm_A = norm_after_clip_A / scale_factor             # 换算回真实尺度

# 路径B(正确顺序):先unscale_再裁剪
modelB = nn.Linear(4, 4).to(device); modelB.load_state_dict(model.state_dict())
optB = torch.optim.SGD(modelB.parameters(), lr=0.1)
scalerB = torch.amp.GradScaler(device='cuda', init_scale=scale_factor)
with torch.autocast(device_type='cuda', dtype=torch.float16):
    lossB = modelB(x).mean()
scalerB.scale(lossB).backward()
scalerB.unscale_(optB)
torch.nn.utils.clip_grad_norm_(modelB.parameters(), max_norm=1.0)
norm_after_clip_B = global_norm(modelB.parameters()).item()

print(f"错误顺序:裁剪后(放大域)范数={norm_after_clip_A:.4f}, 换算回真实尺度={effective_real_norm_A:.6f}")
print(f"正确顺序:裁剪后(真实域)范数={norm_after_clip_B:.4f}")
# 实测: 错误顺序换算回真实尺度后 = 0.0009765625 (=1/1024,精确等于 1/scale_factor)
#       正确顺序 = 0.8447 (裁剪前范数本来就小于1.0,clip是no-op,维持真实梯度不变)
assert abs(norm_after_clip_A - 1.0) < 0.05          # 错误路径:被压到放大域的1.0
assert effective_real_norm_A < norm_after_clip_B / 100    # 换算回真实尺度,比正确结果小了两个数量级以上
```

也就是说:错误顺序不会报任何错、loss 曲线甚至可能看起来"很稳定"(因为梯度被过度裁剪到几乎消失,参数更新幅度极小,训练看起来"平稳"实际是"几乎不再学习")——这是一类典型的"静默训练失败"(silent training failure),比直接报错的 bug 更难排查,因为所有代码都能正常运行、没有任何异常堆栈可以定位。

**AI 研究场景:** 这个顺序陷阱在"从纯 fp32 训练脚本迁移到混合精度训练"时最容易发生——原来的脚本里 `clip_grad_norm_` 紧跟在 `loss.backward()` 后面,直接照抄这个位置、只是把 `loss.backward()` 换成 `scaler.scale(loss).backward()`,而没有在中间插入 `scaler.unscale_(optimizer)`,是一个真实、常见、PyTorch 官方混合精度训练教程里专门用整整一段警告强调过的坑。

**面试怎么问 + 追问链:**
- **Q:** "在混合精度训练里做梯度裁剪,需要注意什么?"—— 期望直接说出"必须先 `scaler.unscale_(optimizer)` 再裁剪,不能直接对 scale 过的梯度做裁剪"。
- **追问 1(核心,要求量化):** "如果顺序错了,具体会发生什么,能定量说一下吗?"—— 期望能说出"梯度会在'放大域'被裁剪到 max_norm,换算回真实尺度后,相当于用了一个比预期小 `scale` 倍的裁剪阈值",最好能提到这是静默发生、不会报错的。
- **追问 2(深挖):** "为什么 `scaler.step(optimizer)` 内部还会再做一次 unscale/检测,不会因为我已经手动 `unscale_` 过而出问题吗?"—— 期望知道 `GradScaler` 内部会记录"这个 optimizer 这一轮是否已经 unscale 过",`step()` 发现已经 unscale 过就不会重复操作,这是一个专门设计好、可以安全配合手动 `unscale_()`+裁剪 使用的机制,不是需要用户自己去重复处理的边界情况。
- **追问 3(呼应06篇,考察跨章节整合能力):** "上面这套代码里,`scheduler.step()` 应该放在训练循环的哪个位置?如果换成 `OneCycleLR` 呢?"—— 期望能直接调用 06 篇第 9 节的结论:大多数 epoch 级调度器(`CosineAnnealingLR`)在每个 epoch 结束调用一次;`OneCycleLR` 要求每个 batch(每次真正 `optimizer.step()` 之后)调用一次,搞混调用粒度会让实际的 lr 曲线和预期完全不一致。

**常见坑:**
- 上面详细验证的"忘记 unscale 就裁剪"顺序错误,是本节最重要的坑。
- 梯度累加(第2节)和混合精度(第3/4节)一起用时,只在真正 `optimizer.step()` 的那个 iteration 才做 `unscale_`/裁剪/`scheduler.step()`,中间纯粹累加梯度的 iteration 不要重复调用这些——上面的代码框架用 `if (i+1) % accumulation_steps == 0:` 显式区分了这两种 iteration,漏掉这个判断会让裁剪阈值和调度节奏都乱套。
- 把 `scaler.update()` 放在 `if` 判断外面、每个 micro-batch iteration 都调用——`update()` 应该只在真正尝试过 `step()` 之后调用一次,不然会用还没真正参与过 `step()` 判断的状态去更新 `scale`,导致动态调整的逻辑失真。

---

## 小结:这一批 7 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | `zero_grad(set_to_none=True)` | 默认值就是True;`grad=None`让`step()`跳过参数,`grad=0`仍会用衰减后的动量更新参数——两者不等价(已用Adam实测验证) |
| 2 | 梯度累加 | 对梯度精确等价于大batch(线性可加);对BatchNorm统计量不等价(多次小EMA更新≠一次大批量统计,已实测数值不同) |
| 3 | `autocast` | 不转换模型参数,只按算子名单分别处理:matmul等转fp16,sum/softmax/exp/loss等强制fp32,relu等保留输入精度 |
| 4 | `GradScaler` | 放大loss防止fp16梯度下溢,`scale`会按连续无溢出步数自动调大、一遇溢出立刻减半且跳过该步`step()`(已实测三种场景) |
| 5 | gradient checkpointing | 前向不存中间激活值,反向时重新计算(已验证forward确实被多调用一次);梯度精确不是近似;大batch下实测省1.63x显存 |
| 6 | `torch.compile` | 首次调用需完整编译,同形状复用编译产物极快(实测3186x),换形状触发重新编译——变长输入场景要小心 |
| 7 | 完整训练step组装 | 裁剪梯度必须在`scaler.unscale_()`之后,否则会在错误的尺度上裁剪,导致静默的、不报错的训练失效(已定量验证) |

下一批:[10-serialization-and-deployment.md](10-serialization-and-deployment.md) —— 序列化与部署基础(`state_dict()` 保存加载、`map_location`、ONNX/TorchScript 导出简介)。

---

*更新:2026-07-07*
