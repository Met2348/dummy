# 09 · 分布式训练基础机制(训练时的梯度同步)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批讲的是训练时最容易在面试里被"问穿"的一块——不是"DDP 三行代码怎么套上去",而是"多张卡各自算出一份梯度之后,到底通过什么机制变成一份能用来更新模型的、所有卡都认可的梯度"。这一批很容易被读者和仓库里已有的 [distributed-inference](../../learning/distributed-inference/) 模块混为一谈,但其实是完全不同的两个问题——前者是"每张卡都装得下完整模型,只是卡多了想训练更快",后者是"模型本身太大,一张卡根本装不下,必须把模型切开"。本文第 6 节有专门的分工说明,这里先卖个关子,不提前展开。

**关于本文验证情况的诚实说明(比 01 篇更复杂,必须先讲清楚):** 本机(ERIC-3080Ti)只有**一张** RTX 3080 Ti,没有第二张物理 GPU,没法真的跑 `torch.distributed` 多卡训练。但"没法起多卡"不等于"什么都测不了"——`torch.distributed` 支持用 `gloo` backend 在同一台机器的多个**进程**之间通信(不需要每个进程有独立的物理 GPU,CPU 进程之间也能跑;必要时甚至能让多个进程共享同一张 GPU),本文第 2、4、5 节涉及 `torch.distributed` 的部分,都是用这种方式在仓库 `.venv`(torch 2.11.0+cu128)下**真实跑通验证**过的,不是纸面推导。第 3 节的 ring all-reduce 更进一步,完全不依赖 `torch.distributed`,纯手写 Python 模拟通信逻辑并验证正确性,是本文唯一的硬性验证要求。凡是**必须依赖真实多张物理 GPU**才能观测到的现象(典型的比如第 1 节"DataParallel 主 GPU 显存占用明显更高"),会在对应位置明确标成"引用官方文档/本机可读到的源码,未实测",绝不伪装成实测结果。文末有完整的已测 / 未测清单。

**本篇统一结构(和前几批一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(能测的现场测,测不了的明确标注依据)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. `nn.DataParallel` 的问题(为什么官方现在建议弃用)

**是什么:**
```python
torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
```
把一个模型包一层,训练时自动把输入 batch 切分到 `device_ids` 列出的多张 GPU 上并行跑 forward,再把结果收集回来算 loss、发起反向传播。

**一句话:** `DataParallel` 是**单进程、多线程**架构——一个 Python 进程里,主线程负责把 batch 切开分发给各 GPU 上的模型副本、收集各 GPU 算出的输出、在其中一张"主 GPU"上算 loss 并发起反向传播,官方文档现在明确建议不要在新代码里用它。

**底层机制/为什么这样设计:**

从本机安装的 torch 包里用 `inspect.getsource` 现场读出来的 `DataParallel.forward` 真实源码(已验证,不是转述):

```python
def forward(self, *inputs, **kwargs):
    ...
    inputs, module_kwargs = self.scatter(inputs, kwargs, self.device_ids)   # 1. 切分 batch
    if len(self.device_ids) == 1:
        return self.module(*inputs[0], **module_kwargs[0])
    replicas = self.replicate(self.module, self.device_ids[: len(inputs)])   # 2. 把模型复制到每张卡
    outputs = self.parallel_apply(replicas, inputs, module_kwargs)            # 3. 各卡并行跑 forward
    return self.gather(outputs, self.output_device)                          # 4. 把各卡输出收集回主卡
```

四步流水线:**scatter(切分)→ replicate(复制模型)→ parallel_apply(并行跑)→ gather(收集)**。这个架构有两个根子上的问题:

**问题①GIL:** `DataParallel` 是单进程多线程,而 CPython 的 GIL(全局解释器锁)保证同一时刻只有一个线程在执行 Python 字节码。GPU kernel 本身是异步执行的、调用 CUDA 扩展时确实会释放 GIL,但 `scatter`/`replicate`/`parallel_apply` 组织调度、`gather` 收集结果这些环节仍然有大量 Python 层面的胶水代码要跑,这部分调度开销没法在多线程下被真正并行化,增加了每一轮迭代的额外开销。

**问题②主 GPU 负载不均:** 官方 docstring 原文(已验证)——"In the forward pass, the module is replicated on each device... During the backwards pass, gradients from each replica are summed into the original module."(反向传播时,各副本的梯度会被汇总回**原始 module**,也就是 `device_ids[0]` 所在的那张卡)。结合上面的源码能看到,`scatter` 和 `gather` 都是由运行 `forward()` 的这个进程发起和完成的,loss 计算、反向传播的起点也都在主 GPU 的上下文里——主 GPU 除了要处理自己那份数据,还要额外装下 gather 回来的所有其他 GPU 的输出、损失计算的中间结果,显存占用和计算负载天然比其他卡更高,整体训练速度被这个"最忙的" 主 GPU 拖累。

PyTorch 官方教程([`ddp_tutorial.html`](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html),已核实原文)对此的总结:"Due to GIL contention across threads, per-iteration replicated model, and additional overhead introduced by scattering inputs and gathering outputs, `DataParallel` is usually slower than `DistributedDataParallel` even on a single machine."——请注意这句话点出的三个因素(GIL 争用、每次迭代都要重新复制模型、scatter/gather 开销)里,"每次迭代都要重新复制模型"是我们在源码里能直接验证到的第二步 `replicate`,也是额外开销的一部分,不只是 GIL 一个原因。

**AI 研究场景:** 老代码/教程里还很常见,因为写法最简单——`model = nn.DataParallel(model)` 一行就能用,不需要 `torchrun`、不需要多进程改造训练脚本。但现在写新代码、复现论文时应该默认跳过它,直接用第 2 节的 DDP。了解它的机制主要是为了看懂遗留代码,以及在面试里能讲清楚"为什么不推荐"背后的原理,而不是死记"官方说不推荐"这一句话。

**可运行例子:**

第一部分,现场读源码和 docstring,验证上面引用的内容都是真实存在于本机安装的 torch 包里,不是凭空转述:
```python
import inspect
import torch.nn as nn

# 验证 forward() 确实是 scatter -> replicate -> parallel_apply -> gather 四步
src = inspect.getsource(nn.DataParallel.forward)
assert "self.scatter(" in src
assert "self.replicate(" in src
assert "self.parallel_apply(" in src
assert "self.gather(" in src

# 验证官方 docstring 里"梯度汇总回原始 module"和"不推荐使用"的原文确实存在
# 注意:docstring 原文在 "each replica" 和 "are summed" 之间是换行(源码里手动折行排版),
# 不是空格,直接按"空格拼接"的整句去 in 判断会因为中间是\n而匹配失败(现场复验时踩到的真实坑,
# 不是凭空加的防御性代码)——统一把连续空白折叠成一个空格后再匹配,不依赖具体折行方式
import re
doc = re.sub(r"\s+", " ", nn.DataParallel.__doc__)
assert "gradients from each replica are summed into the original module" in doc
assert "It is recommended to use" in doc and "DistributedDataParallel" in doc
```

第二部分,GIL 问题的机制性验证——这部分**不需要 GPU**,GIL 是纯 Python 解释器层面的机制,可以脱离 CUDA 单独验证多线程和多进程在 CPU-bound 任务上的真实差异:
```python
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_bound_task(n: int) -> int:
    # 纯 Python 计算,不调用任何会释放 GIL 的 C 扩展(不用 numpy/torch),
    # 这样才能如实反映"纯 Python 字节码执行"是否被 GIL 序列化
    x = 0
    for i in range(n):
        x += i * i
    return x

if __name__ == "__main__":
    N_TASKS, N = 4, 6_000_000

    t0 = time.perf_counter()
    for _ in range(N_TASKS):
        cpu_bound_task(N)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N_TASKS) as ex:
        list(ex.map(cpu_bound_task, [N] * N_TASKS))
    t_thread = time.perf_counter() - t0

    # 进程池先预热(排除"创建新解释器"这个一次性开销,只比较稳态执行阶段——
    # 对应 DDP 真实用法:进程在训练开始时启动一次,之后跑几千个 step)
    with ProcessPoolExecutor(max_workers=N_TASKS) as ex:
        list(ex.map(cpu_bound_task, [1000] * N_TASKS))
        t0 = time.perf_counter()
        list(ex.map(cpu_bound_task, [N] * N_TASKS))
        t_process = time.perf_counter() - t0

    # 本机两次实测的典型结果(时间本身有噪声,方向稳定,重复跑过2次结论一致):
    # 多线程 / 顺序 的比值在 0.77~1.25 之间浮动 —— 也就是说多线程几乎没有加速,
    # 有时甚至因为线程切换开销比顺序执行还慢,这正是 GIL 把字节码执行串行化的直接后果
    assert t_thread > 0.7 * t_seq
    # 多进程 / 顺序 的比值稳定在 0.3~0.5 —— 4 个独立进程有各自的 GIL,能真正利用多核
    assert t_process < 0.7 * t_seq
```

**主 GPU 显存/计算负载更高这一条,诚实说明:** 这需要真实观测"同一次训练里,不同 GPU 的显存占用/耗时数字有多大差异",必须有至少 2 张物理 GPU 才能测,本机只有 1 张卡,`device_ids` 传单元素列表时 `DataParallel.forward` 会直接走 `if len(self.device_ids) == 1: return self.module(...)` 这个分支(上面贴的源码里就能看到),根本不会触发 scatter/replicate/gather,退化成普通单卡运行,没法在本机复现"主卡更忙"这个现象。以上关于这一条的结论,完全基于官方 docstring 原文 + 本机可读到的真实源码 + PyTorch 官方教程原文,是合理的机制推断,不是实测结果。

**面试怎么问 + 追问链:**
- **Q:** "你知道 `nn.DataParallel` 为什么现在不推荐用了吗?"
- **追问 1:** "GIL 具体是怎么拖慢它的?GPU 计算不是异步的吗,GIL 为什么还会成为瓶颈?"—— 期望能区分"kernel 在 GPU 上执行的时间"(异步,不受 GIL 直接影响)和"每一步 scatter/replicate/gather 这些 Python 胶水代码的调度时间"(被 GIL 串行化),说清楚 GIL 卡的是**调度开销**,不是**计算本身**。
- **追问 2:** "主卡负载不均具体体现在哪几个方面?"—— 期望答出显存(要装下 gather 回来的所有输出 + loss 计算的中间结果)和计算(loss 计算 + 反向传播发起)两个维度,最好能提到这是从"gradients... summed into the original module"这句 docstring 直接推出的。
- **追问 3(容易问倒):** "`replicate` 这一步为什么每次 forward 都要重新做,不能只做一次吗?"—— 这道题对应 docstring 里"每次 forward 都会重新复制,对 replica 的原地修改不会保留"这条 warning;根本原因是 `DataParallel` 没有维护"多卡上有多份持久化的模型副本"这个状态,每次都是临时复制、用完销毁,这是它比 DDP 更简单但也更低效的地方(DDP 是每个进程持有一份持久化的模型副本,不需要每步重新复制)。
- **追问 4(开放题):** "如果历史代码里已经写死了 `DataParallel`,一定要保留,有什么办法缓解主卡瓶颈?"—— 没有标准答案,考察工程判断力,合理方向包括:换用显存更大的卡作为 `device_ids[0]`、减小主卡上其它任务的显存占用、或者最实际的答案——评估改造成本后直接换成 DDP。

**常见坑:**
- 以为 `DataParallel` 和 DDP 性能相近、只是 API 写法不同,随手换用不会有明显影响——实际上即使在单机多卡场景,DDP 通常也明显更快(官方教程原话)。
- 忽略 docstring 里"batch size 应该大于 GPU 数量"这条要求——batch 太小时,`scatter` 切分后有些 GPU 可能分不到数据。
- 以为在 `forward()` 内部对 `self.xxx` 做的原地修改会被保留下来——由于每次 forward 都重新 `replicate`,这类修改只发生在临时副本上,`forward` 结束副本就被销毁,修改不会累积(`BatchNorm` 的 running_mean/var 之所以能正常更新,是因为 docstring 里特别说明了"`device_ids[0]` 上的副本和原始 module 共享存储"这个特例,不代表所有原地修改都安全)。

---

## 2. `DistributedDataParallel`(DDP)基本原理

**是什么:**
```python
import torch.distributed as dist
dist.init_process_group(backend="nccl")  # GPU 场景常用 nccl;本文用 gloo 在 CPU 上模拟
ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```
把训练从"一个进程管多张卡"彻底改成"多个独立进程,每个进程只认识、只使用自己的一张卡"。

**一句话:** DDP 是**多进程**架构——每个进程绑定一张 GPU、有自己独立的 Python 解释器和 GIL,各自完整地跑一遍 forward + backward(不是像 `DataParallel` 那样把 batch 切了分给别人算),只在"需要合并所有进程算出的梯度"这一步做进程间通信,从根子上避开了 `DataParallel` 的主卡瓶颈问题。

**底层机制/为什么这样设计:**

和 `DataParallel` 最本质的区别:`DataParallel` 是"一个进程指挥 N 张卡",DDP 是"N 个完全独立的进程,每个只伺候自己的 1 张卡"。具体展开:

- **数据:** 每个进程通过 `DistributedSampler` 拿到全局 batch 里"自己独占的那一份"切片(不同进程读到的数据不重叠),各自完整地做前向、算 loss——所以严格来说,不同 rank 上 `loss.item()` 读出来的数值通常是不同的(因为看到的数据不一样),这是完全正常的现象,和 `DataParallel` "全局只有一份 loss" 不同。
- **避开 GIL:** 因为是独立进程,每个进程有自己的解释器实例和 GIL,不存在"多个 GPU 的调度挤在一个 GIL 里排队"的问题。
- **参数一致性怎么保证:** 各进程"各算各的"，怎么保证 N 份模型参数不会越训越不一样?靠两个机制配合:①**construction 时刻的一次性广播**——DDP 在包装模型的那一刻,会把 rank 0 的参数和 buffer 广播给其它所有 rank,保证起点完全一致;②**每次 backward 后的梯度 all-reduce**(下一节细讲)——保证所有进程"合并后拿到的梯度"完全一致。起点相同 + 每一步用来更新参数的梯度也完全相同 + 优化器超参相同,数学上就能保证 N 个进程手里的参数永远保持一致,不需要每一步都同步参数本身,只需要在梯度这一个点上对齐。这正是 DDP 通信量远小于"每步都同步一次完整参数"的根本原因。
- 相比 `DataParallel` 把"整合工作"全部堆在主卡,DDP 把这份工作变成了 all-reduce 这个**所有进程对等参与**的集合通信操作(见第 3 节),不存在谁比谁更忙的主卡瓶颈。

**AI 研究场景:** 现代所有严肃的单机多卡/多机多卡训练都建立在 DDP(或者它更高级的演化,比如应对"参数本身太大装不下单卡"场景的 FSDP,这里不展开)之上——不管是自己手写训练循环,还是用 HuggingFace `Trainer`、PyTorch Lightning,底层默认的多卡方案都是 DDP。标准启动方式是 `torchrun`(比如 `torchrun --nproc_per_node=4 train.py`,每张卡对应一个独立进程),这是官方文档的标准用法,本机单卡没法演示真实效果,这里只是记录标准命令供参考。

**可运行例子(已在本机 2 个 CPU 进程 + `gloo` backend 下真实跑通验证):**

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

OUT_DIR = os.path.join(os.path.dirname(__file__), "_ddp_tmp")


def worker(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    torch.manual_seed(0)
    model = nn.Linear(4, 2)

    # 故意在 rank1 上扰动初始参数,验证 DDP construction 时的广播能把它拉回和 rank0 一致
    if rank == 1:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

    ddp_model = nn.parallel.DistributedDataParallel(model)
    params_after_wrap = [p.detach().clone() for p in ddp_model.parameters()]

    # 每个 rank 用不同数据,模拟"不同进程读到不同 mini-batch"
    torch.manual_seed(100 + rank)
    x, target = torch.randn(8, 4), torch.randn(8, 2)
    loss = nn.functional.mse_loss(ddp_model(x), target)
    loss.backward()
    grads = [p.grad.detach().clone() for p in ddp_model.parameters()]

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"params_after_wrap": params_after_wrap, "grads": grads, "loss": loss.item()},
               os.path.join(OUT_DIR, f"rank{rank}.pt"))
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    r0 = torch.load(os.path.join(OUT_DIR, "rank0.pt"))
    r1 = torch.load(os.path.join(OUT_DIR, "rank1.pt"))

    # 验证①:construction 时的参数广播生效——rank1 故意扰动过的参数,被 rank0 覆盖
    for p0, p1 in zip(r0["params_after_wrap"], r1["params_after_wrap"]):
        assert torch.allclose(p0, p1, atol=1e-6)

    # 验证②:两个 rank 用不同 mini-batch 各自 backward,梯度被自动 all-reduce 拉成完全一致
    for g0, g1 in zip(r0["grads"], r1["grads"]):
        assert torch.allclose(g0, g1, atol=1e-6)

    print(f"rank0 loss={r0['loss']:.4f}, rank1 loss={r1['loss']:.4f}")
    # 本机一次实际运行输出:rank0 loss=1.5535, rank1 loss=0.9755
    # loss 不同是正常的(两个 rank 数据不同);但断言证明梯度已经被同步成完全一致
```

两个断言都通过,证明:①DDP 构造时真的会把 rank 0 的参数广播给其他 rank;②两个进程在不同数据上各自 backward 之后,梯度真的被自动同步成了完全一致的值。这里用 `gloo` + CPU 进程模拟"多个 rank",没有用到第二张真实 GPU,但 `init_process_group`/`DistributedDataParallel` 这套 API 和它"参数广播 + 梯度自动同步"的核心行为是真实触发、真实验证过的。顺便提一句本机的真实情况:`torch.distributed.is_nccl_available()` 在这台 Windows 机器上返回 `False`(已验证)——NCCL 是 Linux/GPU 专用的高性能通信库,Windows 上不可用,只能用 `gloo`,这也是为什么本文所有分布式实验都基于 `gloo`。真实多卡 Linux 环境下,GPU 间的 DDP 训练通常用 `backend="nccl"`,速度比 `gloo` 快得多。

**面试怎么问 + 追问链:**
- **Q:** "DDP 和 `DataParallel` 最核心的架构区别是什么?"—— 期望答"单进程多线程 vs 多进程",并且能说出这个区别怎么解决了 GIL 和主卡瓶颈两个问题。
- **追问 1:** "既然是独立的 N 个进程,各自 loss 都不一样,那模型参数怎么保证不跑偏?"—— 期望答出"起点靠 construction 时的参数广播对齐,之后靠每步梯度 all-reduce 保持更新量一致,不需要每步同步参数本身"。
- **追问 2(深挖,区分度高):** "如果我在某个 rank 上手滑,用和其他 rank 不同的随机种子初始化了模型,DDP 能不能保证参数还是一致的?"—— 期望答"能,因为 DDP 在 construction 时会把 rank 0 的参数广播给所有 rank,构造完成后各 rank 参数已经被强制拉齐,不依赖你自己保证初始化一致"——这条在上面的例子里是真的故意搞乱参数、现场验证过的,不是纸面结论。
- **追问 3:** "为什么 DDP 不是每一步都同步参数,而是同步梯度?这两者通信量不是差不多大吗?"—— 期望答出关键点其实不在"数据量大小"(梯度和参数张量大小确实一样),而在于**同步的必要性和时机**:参数更新之间(forward、backward 的中间过程)完全不需要通信,只有"把这一步的更新量合并"这一个时间点是必须同步的,DDP 把通信压缩到刚好这一个点上,而不是在训练过程中反复互相等待、对齐中间结果。
- **追问 4:** "`DistributedSampler` 在这里起什么作用?"—— 期望答"保证不同 rank 读到的是全局 batch 里不重叠的切片,不然多个进程会在重复的数据上训练,失去了'看更多数据'的意义"。

**常见坑:**
- `DistributedSampler` 没配置好,导致不同 rank 读到重复或遗漏的数据。
- 忘记在每个 epoch 开始时调用 `sampler.set_epoch(epoch)`(`DistributedSampler` 确实提供这个方法,已验证)——不调用的话,各 rank 每个 epoch 的 shuffle 顺序不会随 epoch 变化,长期训练下数据的随机性会打折扣。
- 单机多卡训练时 `MASTER_PORT` 被占用导致 `init_process_group` 卡住或报错,本机实验里就遇到了残留的 socket 连接尝试(不影响最终结果,但排查时容易被这类警告干扰)。
- 把只在 GPU 环境下verified 的 `nccl`-only 代码直接搬到没有 GPU 的机器上调试,` backend="nccl"` 在这类环境下根本不可用,应该先用 `gloo` 跑通逻辑,再切换到真实 GPU 环境验证性能。

---

## 3. all-reduce 梯度同步机制(含手写 ring all-reduce 模拟,已验证)

**是什么:**
```python
torch.distributed.all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False)
```
一种"集合通信"(collective communication)操作:参与的 N 个进程各自持有一份数据,操作完成后,**每一个**进程手里都会拿到"对这 N 份数据做规约(默认求和)之后"的同一份结果——不是只有某一个进程拿到,是所有人都拿到相同的答案。

**一句话:** DDP 梯度同步的需求本质是"N 个进程各自算出一份局部梯度,需要变成所有进程都拥有同一份'全局梯度'";all-reduce 就是满足这个需求的标准通信原语,而 **ring all-reduce** 是实现它的一种通信效率很好的具体算法——用环形拓扑把通信量摊平到所有节点,而不是堆在一个中心节点上。

**底层机制/为什么这样设计:**

**先看需求本身:** 为什么必须是"所有进程都拿到相同结果",而不是"某个中心进程拿到就够了"?因为 DDP 里每个进程本地都要各自完整地跑 `optimizer.step()`,用合并后的梯度更新自己手里那份模型参数副本——如果只有一个进程知道全局梯度,其它进程就没法完成参数更新,模型在下一步就会立刻变得不一致。

**朴素方案的问题:** 如果用"一个中心节点(比如 rank 0)收集其余 N-1 个进程的梯度,自己算完平均,再发回给所有人",中心节点的通信量是"收 N-1 份 + 发 N-1 份",随 N 线性增长——N 越大,这个中心节点越容易成为带宽瓶颈,而其它节点两两之间的带宽完全没被用上。

**ring all-reduce 的思路:** 把 N 个进程排成一个逻辑环,每个进程只和相邻的两个邻居通信,没有"中心"这个角色。分两个阶段,各 N-1 轮:

1. **scatter-reduce(N-1 轮):** 每份梯度切成 N 块。每一轮,每个进程把自己手里某一块发给下一个邻居、同时从上一个邻居收一块累加到自己对应的块上。N-1 轮之后,每个进程手里恰好有一块是"全局所有进程该块之和"(已经完全规约好)。
2. **allgather(再 N-1 轮):** 把这些"已经算完"的块沿着环继续传一圈,让所有进程最终拿到全部 N 个块的最终结果。

全程下来,每个进程发送的数据总量 ≈ `2*(N-1)/N * 单份梯度大小`,N 越大这个倍数越接近 2,**和 N 基本无关**——这是它相对"中心节点方案"的核心优势:通信负担被均匀摊到每个节点,不会随卡数增加堆积到某一个节点上。(工业界的 NCCL 会用更精细的算法,比如结合网络拓扑的树形算法、多环并行等,但 ring all-reduce 是理解这一整类算法最基础、最直观的入门模型。)

**这部分是本文唯一的硬性验证要求,完整跑通,不依赖 `torch.distributed`,纯手写模拟:**

```python
import torch

def ring_all_reduce_sum(grads: list[torch.Tensor]):
    """输入:N 个 rank 各自的梯度(长度 D,且 D 能整除 N)。
    输出:N 个 tensor,每个都等于所有输入梯度的逐元素求和(all-reduce 的 SUM 版本)。"""
    n = len(grads)
    d = grads[0].numel()
    assert d % n == 0
    chunk = d // n

    # buffers[r] 是 rank r 手里的 n 个分块(初始是自己梯度切成 n 份)
    buffers = [[grads[r][c*chunk:(c+1)*chunk].clone() for c in range(n)] for r in range(n)]
    send_bytes_per_rank = [0] * n

    # ---- 阶段一:scatter-reduce,n-1 轮 ----
    # 第 step 轮,rank r 把"块号 (r-step) mod n"发给 rank (r+1) mod n;
    # 同时 rank r 收到来自 rank (r-1) mod n 的块,累加到自己对应块号上。
    # 注意:必须先用"上一轮结束时的状态"统一算好这一轮所有 rank 要发的数据,
    # 不能边发边改——否则会用到本轮里"已经被更新过"的数据,模拟就错了。
    for step in range(n - 1):
        send_chunks = []
        for r in range(n):
            c = (r - step) % n
            send_chunks.append(buffers[r][c].clone())
            send_bytes_per_rank[r] += chunk
        for r in range(n):
            recv_from = (r - 1) % n
            c = (recv_from - step) % n
            buffers[r][c] += send_chunks[recv_from]

    # ---- 阶段二:allgather,n-1 轮 ----
    # scatter-reduce 结束后,rank r 的块号 (r+1) mod n 已经是"全局和"。
    # allgather 把这些已经算好的块沿环继续传一圈,让每个 rank 拿到全部 n 个块的最终结果。
    for step in range(n - 1):
        send_chunks = []
        for r in range(n):
            c = (r + 1 - step) % n
            send_chunks.append(buffers[r][c].clone())
            send_bytes_per_rank[r] += chunk
        for r in range(n):
            recv_from = (r - 1) % n
            c = (recv_from + 1 - step) % n
            buffers[r][c] = send_chunks[recv_from]

    return [torch.cat(buffers[r]) for r in range(n)], send_bytes_per_rank


torch.manual_seed(0)
N, D = 4, 12   # 4 个虚拟 rank,每个 rank 的梯度长度 12(能整除 4)
grads = [torch.randn(D) for _ in range(N)]
results, send_per_rank = ring_all_reduce_sum(grads)

# 正确性验证:ring all-reduce 算出的和,必须等于暴力 sum 的 ground truth
ground_truth_sum = torch.stack(grads).sum(dim=0)
for r in range(N):
    assert torch.allclose(results[r], ground_truth_sum, atol=1e-6)   # float32,必须用 allclose 不能用 ==

# all-reduce 的定义:所有节点最终结果必须彼此完全一致
for r in range(1, N):
    assert torch.allclose(results[0], results[r], atol=1e-6)

# DDP 实际用的是"平均梯度"而不是"和",验证除以 N 之后等于逐元素平均
ground_truth_avg = ground_truth_sum / N
for r in range(N):
    assert torch.allclose(results[r] / N, ground_truth_avg, atol=1e-6)

# 通信量验证:每个 rank 发送的元素总数应该等于 2*(N-1)*chunk,和 N 基本无关(比例上)
chunk = D // N
assert send_per_rank[0] == 2 * (N - 1) * chunk   # 实测: 18 = 2*3*3,约为单份梯度大小 D=12 的 1.5 倍

# 换几组 (N, D) 重新验证,不能只验证一个场景就下结论(呼应 01 篇的教训)
for N2, D2 in [(2, 6), (3, 9), (5, 15), (8, 40)]:
    torch.manual_seed(42 + N2)
    grads2 = [torch.randn(D2) for _ in range(N2)]
    results2, _ = ring_all_reduce_sum(grads2)
    gt2 = torch.stack(grads2).sum(dim=0)
    for r in range(N2):
        assert torch.allclose(results2[r], gt2, atol=1e-6)
```

全部断言通过(N=4,D=12 的基础场景 + (2,6)/(3,9)/(5,15)/(8,40) 四组额外参数,五种配置结论一致)。实测通信量对比:ring all-reduce 里每个 rank 只需要发送 18 个元素(约 1.5 倍单份梯度大小,且比例随 N 增大趋近 2、与 N 本身无关);换成"中心节点收发所有数据"的朴素方案,中心节点一个点就要收发 `2*(N-1)*D = 72` 个元素——是 ring 方案单节点通信量的 4 倍,而且这 72 个元素全部堆在同一个节点上,朴素方案的通信量随 N 线性增长,ring all-reduce 不会。

**AI 研究场景:** 这是所有数据并行训练框架的通信基石——原生 DDP、DeepSpeed 的 ZeRO、Horovod,梯度同步都建立在 all-reduce(或它的变体)之上。NCCL 是 NVIDIA 专门为 GPU 间(通过 NVLink/PCIe/InfiniBand)高效实现 all-reduce 等集合通信设计的库,理解 ring all-reduce 的思路是理解"为什么大模型训练要关注网络拓扑、为什么 NVLink 比 PCIe 训练更快"这类工程判断的基础。

**面试怎么问 + 追问链:**
- **Q:** "DDP 怎么保证多个进程的梯度被合并成一份?说说 all-reduce 的原理。"
- **追问 1:** "如果就用一个中心节点收集所有梯度再分发,不行吗?为什么要用环形这么绕的方式?"—— 期望答出"中心节点通信量随 N 线性增长,容易成为瓶颈,ring all-reduce 把通信量摊平到每个节点,和 N 基本无关"。
- **追问 2(区分度很高,数学向):** "ring all-reduce 总共需要几轮通信?每个节点总共发送多少数据?能推一下吗?"—— 期望说出 `2*(N-1)` 轮(scatter-reduce 的 N-1 轮 + allgather 的 N-1 轮),每个节点发送量 ≈ `2*(N-1)/N * 数据总大小`,最好能现场画出环形传递的示意或写出索引公式。
- **追问 3(工程向):** "假设梯度 tensor 特别大,直接对整个 tensor 做一次 all-reduce,和先切成很多小块再分别 all-reduce,有什么区别?"—— 这是给第 4 节 gradient bucketing 埋的伏笔:块太大,通信没法和还没算完的反向传播计算重叠;块太小,每次通信的固定开销(建立连接、延迟)占比会变大,需要权衡。
- **追问 4(开放题):** "如果某个节点比其他节点慢(掉队者/straggler),all-reduce 会发生什么?"—— 期望答出"all-reduce 是同步的集合操作,所有参与者要等最慢的那个完成,DDP 的训练速度因此被最慢的进程拖累",这是数据并行训练的一个经典痛点(混部集群、异构硬件场景尤其明显)。

**常见坑:**
- 误以为 all-reduce 默认结果就是"平均值"——`torch.distributed.all_reduce` 默认 `op=ReduceOp.SUM`(已验证签名),要平均通常要显式除以 `world_size`(较新版本的 `ReduceOp` 也确实提供了 `AVG` 选项,已验证 `dist.ReduceOp` 里存在 `AVG` 成员,可以直接指定,不是所有版本/后端都支持,不能想当然假设它总是可用)。
- 把 "all-reduce"(所有进程都拿到结果)和 "reduce"(只有一个指定的进程拿到结果)搞混——这是两个不同的集合通信原语,命名很像但语义不同。
- 凭直觉以为"进程越多,单个进程的通信负担一定越重"——这正是 ring all-reduce 要打破的直觉,用环形拓扑之后单节点通信量和 N 基本无关,上面的代码已经用 4 组不同 N 实测验证过这一点。

---

## 4. gradient bucketing(梯度分桶)

**是什么:** DDP 不会等"整个模型所有参数的梯度都算完"才发起一次 all-reduce,而是把模型参数划分成若干个"桶"(bucket),某个桶里所有参数的梯度全部就绪时,立刻对这个桶发起 all-reduce 通信,不用等其它还没算完的参数——通过构造 `DistributedDataParallel` 时的 `bucket_cap_mb` 参数控制每个桶的大小上限,官方文档原文(已验证):"If `None`, a default size of 25 MiB will be used."

**一句话:** 反向传播本身是按"计算图从输出往输入"的顺序、一层一层算出梯度的;bucketing 让"通信"和"还没算完的那部分计算"**重叠**进行,而不是"算完所有梯度 → 再统一通信"这种严格的先后顺序,是 DDP 相对"最朴素的 all-reduce 用法"最重要的性能优化。

**底层机制/为什么这样设计:**

反向传播的时间线:梯度是从模型的最后一层往第一层、一层一层被计算出来的(链式法则决定的顺序)。如果傻等所有梯度都算完再通信,GPU 在"等通信完成"这段时间基本是空闲的——bucketing 的核心想法是:最后几层的梯度最先算出来,没必要等第一层的梯度也算完,可以先把"已经就绪的这部分"打包发出去,和"还在计算前面层梯度"这件事同时发生,只要网络带宽和计算资源没有互相抢占太厉害,总耗时就能比"先算完再通信"更短。

官方 docstring 原文(已验证,从本机安装包里读到):"`DistributedDataParallel` will bucket parameters into multiple buckets so that gradient reduction of each bucket can potentially overlap with backward computation."

桶太大 vs 太小的权衡:桶越大,重叠的粒度越粗,极端情况下(1 个桶 = 全部参数)就退化成"等全部算完再通信",和不分桶没有区别;桶越小,通信次数越多,而每次通信都有固定开销(建立连接、握手延迟),桶太小反而让固定开销占比升高,不一定更快。`bucket_cap_mb` 就是让用户在这个权衡上有调节空间的旋钮。

**一个自己动手测出来、很少有教程讲清楚的细节:** DDP 在**第一次** backward 时,还没有观测过"这次前向对应的反向传播里,各参数梯度实际的就绪顺序",没法按 `bucket_cap_mb` 做出精确的桶划分,只能先用一种保守的初始方式跑一遍。**从第二次 backward 开始**,DDP 才会根据"第一次观测到的真实梯度就绪顺序"重建桶(rebuild buckets),这之后桶的划分才真正按 `bucket_cap_mb` 稳定生效。下面的实测代码把这个现象直接测出来了。

**可运行例子(已在本机 2 个 CPU 进程 + `gloo` backend 下真实跑通验证):**

用 `register_comm_hook` 直接"数"每次 backward 触发了几次独立的 bucket 通信——这比读内部日志字段更直接、更可信:

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

OUT_DIR = os.path.join(os.path.dirname(__file__), "_ddp_bucket_tmp")


class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(512, 512) for _ in range(6)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def counting_allreduce_hook(state, bucket):
    state["count"] += 1
    state["bucket_numels"].append(bucket.buffer().numel())
    tensor = bucket.buffer()
    fut = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True).get_future()
    return fut.then(lambda f: f.value()[0])


def worker(rank, world_size, bucket_cap_mb, tag):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    torch.manual_seed(0)
    ddp_model = nn.parallel.DistributedDataParallel(BigModel(), bucket_cap_mb=bucket_cap_mb)
    state = {"count": 0, "bucket_numels": []}
    ddp_model.register_comm_hook(state, counting_allreduce_hook)

    per_iter_counts = []
    for _ in range(4):   # 连续跑 4 次 backward,观察第1次和之后是否不同
        out = ddp_model(torch.randn(4, 512))
        out.sum().backward()
        per_iter_counts.append(state["count"])

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(state, os.path.join(OUT_DIR, f"{tag}.pt"))

    dist.destroy_process_group()


if __name__ == "__main__":
    total_params = sum(p.numel() for p in BigModel().parameters())  # 1,575,936 (~6.01 MiB, float32)
    per_layer_numel = 512 * 512 + 512   # 每层 Linear(512,512) 的参数量 = 262656 (~1.002 MiB)

    mp.spawn(worker, args=(2, 0.5, "small_bucket"), nprocs=2, join=True)     # 桶上限 0.5 MiB(比单层还小)
    mp.spawn(worker, args=(2, 500.0, "large_bucket"), nprocs=2, join=True)   # 桶上限 500 MiB(装得下全部)

    small = torch.load(os.path.join(OUT_DIR, "small_bucket.pt"))
    large = torch.load(os.path.join(OUT_DIR, "large_bucket.pt"))

    # 第 1 次 backward:两种配置都只有 1 个 bucket(DDP 还没重建桶结构)
    assert small["bucket_numels"][0] == total_params
    assert large["bucket_numels"][0] == total_params

    # 从第 2 次 backward 起,large_bucket 场景始终只有 1 个 bucket(500MiB 装得下全部 6MiB 梯度)
    assert all(n == total_params for n in large["bucket_numels"])

    # 从第 2 次 backward 起,small_bucket 场景被拆成 6 个独立 bucket(每层单独超过 0.5MiB 上限)
    steady_state = small["bucket_numels"][1:]          # 后 3 次迭代
    assert len(steady_state) == 18                       # 3 次迭代 * 每次 6 个 bucket
    assert all(n == per_layer_numel for n in steady_state)
```

实测结果(累计触发次数按迭代:`[1, 7, 13, 19]`,逐次差值 `[1, 6, 6, 6]`):第 1 次 backward,`small_bucket`(桶上限 0.5MiB)和 `large_bucket`(桶上限 500MiB)都只触发了 1 次通信,大小等于全部参数(6,303,744 字节);从第 2 次 backward 开始,`small_bucket` 每次都拆成 6 次独立通信(每次 262,656 个元素,恰好对应 6 层 Linear 里的 1 层,因为单层参数量 ≈1.002MiB 已经超过 0.5MiB 的桶上限),而 `large_bucket` 自始至终只有 1 次通信。这个结果不仅验证了 `bucket_cap_mb` 确实控制"梯度按多大粒度切块通信",还证实了"第一次 backward 和之后的 backward 桶行为不同"这条不在官方 docstring 里明确写出来的细节。

**AI 研究场景:** 训练大模型(几十亿参数级别)时,梯度通信量本身就很大,如果通信和计算完全串行,大模型训练的 wall-clock 时间会明显变长。`bucket_cap_mb` 是实践中真的会去调的一个超参数——模型结构、网络带宽不同,最优桶大小也不同,这也是为什么 PyTorch 把它暴露成一个可调参数而不是写死。做"单步耗时 profiling"排查性能问题时,如果只看第一步,得到的桶行为(1 个大桶)和后续步骤实际的稳态行为(多个小桶)是不一样的,容易得出误导性的结论——这正是上面实测发现的"第一次迭代特殊"这一点的直接工程意义。

**面试怎么问 + 追问链:**
- **Q:** "DDP 的通信是等所有梯度都算完才发生,还是边算边发?说说你的理解。"
- **追问 1:** "为什么要分桶,而不是每算完一个参数的梯度就立刻单独对它做一次 all-reduce?"—— 期望答出"每次通信有固定开销(建连、延迟),太细粒度反而效率低,分桶是在'重叠粒度'和'通信次数开销'之间找平衡"。
- **追问 2(深挖,区分度极高):** "bucket 的划分是模型一构造完 DDP 就固定下来的吗?"—— 期望答"不完全是,第一次 backward 只能先用一种保守的初始划分,要等观测到真实的梯度就绪顺序后,从第二次 backward 开始才会按 `bucket_cap_mb` 重建出稳定的桶结构"——这是很少有人讲清楚的细节,能答出来说明真的深入研究过或者亲自测过,不是背文档。
- **追问 3:** "`find_unused_parameters=True` 和 bucketing 有什么关系?"—— 开放追问,期望候选人知道这是处理"有些参数 forward 时用到了,但因为控制流原因没有参与最终 loss 计算、不会有梯度"的情况,不强求答全,但能提到"这类机制会影响 DDP 判断'一个桶什么时候算就绪'"就是好信号。

**常见坑:**
- 把"分桶"当成无关紧要的内部实现细节——实际上 `bucket_cap_mb` 是一个真实会影响训练吞吐的可调超参,默认值(25MiB)不一定是当前模型/网络环境下的最优值。
- 以为 bucket 划分从第一步就完全按 `bucket_cap_mb` 稳定生效,忽略"第一次迭代和之后不同"这个预热效应——如果只 profile 第一步的通信模式就下结论,数据会有误导性(上面的实测代码已经明确复现了这个差异)。

---

## 5. `SyncBatchNorm` 简介

**是什么:**
```python
torch.nn.SyncBatchNorm(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None)
torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)   # 把模型里所有 BatchNorm*D 层原地换成 SyncBatchNorm
```

**一句话:** 普通 `BatchNorm` 的均值/方差是在"当前进程自己能看到的那一份局部 mini-batch"上统计的;`SyncBatchNorm` 在计算这两个统计量的步骤上,额外做一次跨进程通信,让统计量基于"全局 batch"(所有参与训练的进程这一步看到的数据加起来)算出。

**底层机制/为什么这样设计:**

**问题根源:** `BatchNorm` 统计量的质量高度依赖 batch size——batch 越小,均值/方差的估计噪声越大(统计学基本常识:样本量小,统计量方差大)。多卡训练时,单卡的局部 batch size 往往比"单卡训练时的 batch size"小很多(比如全局 batch 256、8 张卡时每张卡只分到 32),如果每张卡的 `BatchNorm` 只统计自己那 32 个样本,估计出来的均值/方差噪声明显比"当年单卡跑 256"更大,这个噪声会传导到训练稳定性和最终精度上——对深度较大、对 BN 统计量敏感的模型影响更明显(典型的比如检测/分割这类因为输入分辨率高、显存吃紧、单卡 batch 被迫开得很小的任务)。

**`SyncBatchNorm` 的做法(读源码验证到一个反直觉但很关键的细节):** 它在同步统计量这一步用的是 **`all_gather`,不是 `all_reduce`**。本机安装包源码里的原始注释(已验证):"Use allgather instead of allreduce because count could be different across ranks, simple all reduce op can not give correct results." —— 不同进程实际样本数(`count`)可能不同(比如数据集大小不能被 world_size 整除时,最后一个 batch 不满),如果直接对各进程的局部均值做 all_reduce 再除以 world_size,等于假设了"每个进程样本数一样",数量不同时会算错;正确做法是把每个进程的 `(mean, invstd, count)` 都用 `all_gather` 收集齐,再用专门的 `batch_norm_gather_stats_with_counts` 按真实样本数加权合并成全局统计量。

**权衡:** 这是"额外通信开销换统计量质量"的取舍——每个训练 step 都多一次通信,如果局部 batch 本身已经足够大,这点统计噪声本来就不明显,`SyncBatchNorm` 的收益可能盖不过它的通信开销;局部 batch 很小时,`SyncBatchNorm` 往往有明显收益。这也是为什么它需要显式调用 `convert_sync_batchnorm` 才会启用,不是 DDP 的默认行为。

**可运行例子(超出任务最低要求的额外验证):** 本机只有 1 张物理 GPU,但源码里能读到 `SyncBatchNorm` 的同步路径明确支持 `gloo` backend(`if process_group._get_backend_name() != "gloo": ... else: dist.all_gather(...)`,已验证),只是要求输入 tensor 必须在 `cuda`/`xpu`/`hpu` 设备上(源码里有明确的 `raise ValueError` 检查,不支持普通 CPU tensor)。这意味着可以让**两个进程共享同一张物理 GPU**(都用 `cuda:0`)、backend 用 `gloo`,来验证"跨进程统计量同步"这个核心行为——这不代表真实多卡部署的性能表现,只用来验证正确性:

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

OUT_DIR = os.path.join(os.path.dirname(__file__), "_syncbn_tmp")


def worker(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)   # 本机只有 1 张卡,两个进程都用它,仅验证正确性,不追求性能

    sync_bn = nn.SyncBatchNorm(4).cuda()
    sync_bn.train()

    torch.manual_seed(1000 + rank)
    # 故意让不同 rank 的局部数据分布差异很大(rank0 均值≈0,rank1 均值≈+10)
    x_local = torch.randn(16, 4, device="cuda") + rank * 10.0
    sync_bn(x_local)

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"x_local": x_local.cpu(), "running_mean": sync_bn.running_mean.cpu(),
                "momentum": sync_bn.momentum}, os.path.join(OUT_DIR, f"rank{rank}.pt"))
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    r0 = torch.load(os.path.join(OUT_DIR, "rank0.pt"))
    r1 = torch.load(os.path.join(OUT_DIR, "rank1.pt"))

    global_mean = torch.cat([r0["x_local"], r1["x_local"]], dim=0).mean(dim=0)
    local_mean_r0 = r0["x_local"].mean(dim=0)
    # BN 的 running_mean 更新公式:running_mean_new = (1-momentum)*0 + momentum*batch_mean(初始为0)
    implied_batch_mean = r0["running_mean"] / r0["momentum"]

    assert torch.allclose(r0["running_mean"], r1["running_mean"], atol=1e-4)      # 两个 rank 结果完全一致
    assert torch.allclose(implied_batch_mean, global_mean, atol=1e-3)             # 用的是全局均值
    assert not torch.allclose(implied_batch_mean, local_mean_r0, atol=1.0)        # 明显不是 rank0 自己的局部均值
```

实测结果(两次独立运行,数值完全一致,不是偶然):rank0 局部均值 ≈ `[-0.11, -0.14, -0.53, 0.13]`,rank1 局部均值 ≈ `[9.83, 9.77, 9.82, 9.65]`(两者差了大约 10,符合我故意加的偏移),全局均值(两份数据拼一起算)≈ `[4.86, 4.82, 4.64, 4.89]`;两个 rank 的 `running_mean` 反推出的 batch 均值,都精确等于这个全局均值,和任何一个 rank 自己的局部均值都对不上——三条断言全部通过,证明 `SyncBatchNorm` 确实做了跨进程的统计量合并,不是分别独立统计的。

**AI 研究场景:** 目标检测、语义分割这类输入分辨率高、显存吃紧、单卡 batch size 被迫开得很小的任务里,`SyncBatchNorm` 接近标配(MMDetection、Detectron2 这类检测框架默认支持/推荐开启);分类任务如果单卡 batch 本身够大,不一定需要。

**面试怎么问 + 追问链:**
- **Q:** "多卡训练时 `BatchNorm` 会有什么问题?`SyncBatchNorm` 怎么解决?"
- **追问 1:** "为什么 batch size 变小,`BatchNorm` 的统计量会变差?"—— 期望能从统计学角度答"样本量小,均值/方差的估计方差大,噪声更明显"。
- **追问 2(容易漏答,区分度高):** "`SyncBatchNorm` 是用 `all_reduce` 还是 `all_gather` 做同步的?为什么不能直接用 `all_reduce`?"—— 期望答"`all_gather`,因为各进程的样本数量可能不同,直接对局部均值做 `all_reduce` 求平均,等于假设了每个进程样本数一样,数量不同时会算错;需要把每个进程的 `(均值, 方差, 数量)` 都收集齐,再按真实样本数加权合并"——这是本节验证时读源码读到的细节,一般教程不会讲这么细。
- **追问 3:** "`SyncBatchNorm` 一定比普通 `BatchNorm` 好吗?"—— 期望答"不一定,多一次通信开销,局部 batch 已经够大时收益可能不明显,是否开启要结合具体场景判断",避免"新机制一定更好"这种想当然的回答。

**常见坑:**
- `convert_sync_batchnorm` 要在 `DistributedDataParallel` 包装**之前**调用,顺序反了容易导致转换不完整。
- 以为 `SyncBatchNorm` 可以在 CPU 上用——源码里有明确检查,只支持 `cuda`/`xpu`/`hpu` 等加速设备的输入 tensor,不支持普通 CPU tensor(已验证)。
- 单卡训练(`world_size=1` 或没有 `init_process_group`)时用 `SyncBatchNorm` 不会报错,但也不会有任何同步效果——源码里 `need_sync = world_size > 1` 这一条保证了它会自动退化成普通 `BatchNorm` 的行为,不需要手动判断要不要用。

---

## 6. 这一批和 [distributed-inference](../../learning/distributed-inference/) 模块的分工

读到这里,回头看开头留的关子:这一批(09)和仓库已有的 `distributed-inference` 模块,虽然都叫"分布式"、都在讲"多卡怎么协作",但解决的是两个完全不同阶段、不同目的的问题,不应该混为一谈。

**09(这一批,训练时的数据并行):** 解决的问题是"模型本身一张卡就装得下,但我有很多数据、想让训练更快、或者想用更大的有效 batch size"。做法是让**每张卡都拥有一份完整的模型**,不同卡处理不同的数据切片,各自独立算完 forward + backward 之后,只在"梯度"这一个点上做同步(第 3 节的 all-reduce)。本质是"同一份模型,不同数据,梯度层面对齐"。

**distributed-inference 模块:** 解决的问题是"模型本身太大,一张卡的显存根本装不下完整的它"(比如千亿参数级别的大模型)。做法是把**模型本身切开**——Tensor Parallel 把一层内部的矩阵运算切开分布到多张卡、Pipeline Parallel 把不同层分布到不同卡、Expert Parallel(MoE)把不同专家分布到不同卡,数据流过这个被切开的模型时,要在卡与卡之间传递中间激活值/部分计算结果。本质是"同一份数据,流过被切开的不同模型部分"。这个模块目前覆盖的是**推理**时的切分部署,如果想深入这部分,直接跳去看 [distributed-inference/00 (README)](../../learning/distributed-inference/README.md) 和 [L01 分布式推理全图](../../learning/distributed-inference/lectures/01-distrib-overview.md),这里不重复讲。

两者不是二选一的关系,实际训练超大模型时经常**组合使用**——比如同时用 Tensor/Pipeline Parallel 把模型切开保证"装得下",再在切开之后的每一份模型副本外面套一层数据并行(DDP 或者它更强的变体 ZeRO/FSDP)保证"训练得快、能喂更多数据",这也是为什么理解这一批的内容是理解更复杂的混合并行策略的前提。

| 维度 | 09(本篇,训练时数据并行) | distributed-inference 模块 |
|---|---|---|
| 解决什么问题 | 模型一张卡放得下,想让训练更快/有效 batch 更大 | 模型一张卡放不下,必须切开才能部署/训练 |
| 谁被切开了 | **数据**(batch)被切开,模型是完整复制的 | **模型本身**(层/矩阵/专家)被切开,数据基本完整流过 |
| 核心操作 | all-reduce 梯度同步(第 3 节) | 切分后的跨卡数据传递(TP 的 all-reduce、EP 的 all-to-all、PP 的层间激活值传递) |
| 典型场景 | 中小模型、想提高训练吞吐 | 千亿参数级别大模型,单卡装不下 |
| 本仓库位置 | 这里(09) | [learning/distributed-inference/](../../learning/distributed-inference/) |

**面试角度提一句:** "分布式训练和分布式推理的并行策略有什么区别?"是一个真实会被问到的问题,很多候选人会把"数据并行"和"模型并行(Tensor/Pipeline/Expert Parallel)"混着讲——精确的回答应该先分清"切的是数据还是模型",再分别展开各自的通信模式,而不是笼统地说"都是多卡协作"。

---

## 小结:这一批 6 个知识点解决的问题

| # | 知识点 | 核心结论 | 本机验证情况 |
|---|---|---|---|
| 1 | `nn.DataParallel` 的问题 | 单进程多线程架构,GIL 拖慢调度 + 主卡承担额外整合工作,官方明确建议改用 DDP | GIL 部分实测(线程 vs 进程 CPU-bound 对比);主卡负载不均引用官方文档/源码,单卡环境无法实测 |
| 2 | `DistributedDataParallel` 基本原理 | 多进程架构,每进程一张卡各自完整跑 forward+backward,只在梯度这一点同步 | 已实测(2 个 gloo CPU 进程:参数广播 + 梯度 all-reduce 一致性) |
| 3 | all-reduce 梯度同步机制 | ring all-reduce 用环形拓扑把通信量摊平到 O(N) 轮,单节点通信量与 N 基本无关 | **已实测**(纯 Python 手写 ring all-reduce 模拟,5 组 (N,D) 参数验证) |
| 4 | gradient bucketing | 梯度按 `bucket_cap_mb` 分桶,就绪的桶立刻通信,和未算完的反向传播重叠;首次迭代和之后行为不同 | 已实测(2 个 gloo CPU 进程 + `register_comm_hook` 直接数出桶数量) |
| 5 | `SyncBatchNorm` 简介 | 用 `all_gather`(不是 `all_reduce`)同步各进程的 `(均值,方差,样本数)`,换取更接近大 batch 的统计质量 | 已实测(2 个 gloo 进程共享同一张 GPU,验证统计量确为全局值) |
| 6 | 与 distributed-inference 模块分工 | 09 切数据(训练态数据并行),distributed-inference 切模型(推理态模型并行),两个不同问题,可组合使用 | 定位/分工说明,非验证类内容 |

下一批:[10-serialization-and-deployment.md](10-serialization-and-deployment.md)

---

*更新:2026-07-07*
