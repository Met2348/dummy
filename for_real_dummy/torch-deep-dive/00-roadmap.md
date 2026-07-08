# torch 逐机制精讲 —— 路线图与进度表

> 目标:至少 100 个 PyTorch 知识点,由浅入深,分批次完成(约 100+)。
> **深度要求(比 numpy-deep-dive 系列更高):** 不满足于"这个函数怎么调",每个知识点都要讲到**底层机制**(为什么这样设计、内部实际发生了什么)、给出**面试会怎么问 + 追问链**——目标是读完能扛住大厂技术面试二三面的深挖提问,不是背 API。
> 定位:这依然是"精读一遍建立认知"的参考资料,不是要求背下来——参考 [numpy-deep-dive/](../numpy-deep-dive/00-roadmap.md) 系列已经验证过的心态。torch 的 tensor 基础操作(创建/形状/索引)和 numpy 是同一套心智模型,这里不重复,只讲 **torch 独有、且是面试重灾区**的部分:autograd、nn.Module、优化器、训练机制的底层原理。
> 前提:建议先看完 [02-pytorch-basics.md](../02-pytorch-basics.md)(Tensor/autograd/nn.Module 的入门介绍)。

---

## 每个知识点的固定讲解结构(比 numpy 系列多两块)

1. **签名/是什么** —— API 或概念定义,人话翻译
2. **一句话** —— 是什么
3. **底层机制 / 为什么这样设计** —— 不停在"怎么用",讲到"为什么必须是这样"、内部实际发生了什么
4. **AI 研究/工程场景** —— 具体在论文/研究代码里怎么用
5. **可运行例子** —— 带 `assert` 验证,能内省的地方现场打印内部状态(`grad_fn`/`_version`/`data_ptr()`等),不是转述文档
6. **面试怎么问 + 追问链** —— 面试官大概率怎么问,追问会往哪个方向深挖,体现"面试二三四面深度"这个要求
7. **常见坑**

---

## 进度表(由浅入深)

| # | 分类 | 文件 | 知识点数(约) | 状态 |
|---|------|------|-----------|------|
| 01 | Tensor 内存模型与基础操作深挖 | [01-tensor-memory-model.md](01-tensor-memory-model.md) | 12 | ✅ 已完成(已验证,含2处自查修正) |
| 02 | Autograd 核心机制(全系列重中之重) | [02-autograd-internals.md](02-autograd-internals.md) | 14 | ✅ 已完成(已验证) |
| 03 | nn.Module 系统内核 | [03-nn-module-internals.md](03-nn-module-internals.md) | 10 | ✅ 已完成(已验证) |
| 04 | 常用层前向反向数学推导 | [04-layers-math-and-backward.md](04-layers-math-and-backward.md) | 11 | ✅ 已完成(已验证) |
| 05 | 损失函数与数值稳定性 | [05-loss-functions-and-numerical-stability.md](05-loss-functions-and-numerical-stability.md) | 7 | ✅ 已完成(已验证) |
| 06 | 优化器内部机制 | [06-optimizer-internals.md](06-optimizer-internals.md) | 10 | ✅ 已完成(已验证,含1处自查修正) |
| 07 | 训练循环深层机制 | [07-training-loop-internals.md](07-training-loop-internals.md) | 7 | ✅ 已完成(已验证) |
| 08 | 内存与性能 | [08-memory-and-performance.md](08-memory-and-performance.md) | 8 | ✅ 已完成(已验证,含1处实验设计修正+1处补充发现) |
| 09 | 分布式训练基础机制 | [09-distributed-training-basics.md](09-distributed-training-basics.md) | 6 | ✅ 已完成(已验证,含1处自查修正) |
| 10 | 序列化与部署基础 | [10-serialization-and-deployment.md](10-serialization-and-deployment.md) | 6 | ✅ 已完成(已验证) |
| 11 | 调试与常见报错精解 | [11-debugging-and-common-errors.md](11-debugging-and-common-errors.md) | 9 | ✅ 已完成(已验证,系列收尾) |

**合计:100 个知识点,11 篇全部完成并独立验证。**

---

## 每一批具体覆盖哪些知识点(明细)

### 01 Tensor 内存模型与基础操作深挖(已完成)
`storage()`/`untyped_storage()` `stride()` `is_contiguous()`/`contiguous()` `view()` vs `reshape()` `.T`/`transpose()`/`permute()` in-place操作与内存复用 in-place与autograd版本计数器机制 `.detach()` vs `.clone()` vs `.data` `.expand()` vs `.repeat()` `.to(device)`的no-op判定 `pin_memory()`/`non_blocking` dtype转换

### 02 Autograd 核心机制
计算图的动态构建原理(define-by-run) `grad_fn` `next_functions`链式结构 `backward()`的实际执行过程(反向拓扑遍历) leaf tensor vs non-leaf tensor `retain_grad()` `retain_graph=True` `create_graph=True`与高阶导数 自定义`autograd.Function`(forward/backward/ctx) 梯度累加机制(为什么需要zero_grad) `torch.no_grad()` vs `requires_grad_(False)` vs `.detach()`三者对比 requires_grad传播规则 `register_hook` `.backward()` vs `torch.autograd.grad()`

### 03 nn.Module 系统内核
`__setattr__`自动注册机制 `nn.Parameter` `parameters()`/`named_parameters()`递归遍历 `state_dict()`与parameters()的区别 `register_buffer`与buffer概念 `train()`/`eval()`模式切换的真实影响 `register_forward_hook`系列 `nn.ModuleList`/`ModuleDict` vs 普通list/dict `children()` vs `modules()` 参数共享(weight tying)

### 04 常用层前向反向数学推导
`nn.Linear`前向反向手推 `F.relu` vs `nn.ReLU` `Conv2d`的im2col原理+反向 `BatchNorm`训练/推理模式+反向公式推导 `LayerNorm` vs `BatchNorm`本质区别 `Dropout`的inverted scale技巧 `Embedding`稀疏梯度 `MultiheadAttention`内部拆分逻辑 残差连接对梯度传播的影响 ReLU死亡神经元 GELU等现代激活函数梯度特性

### 05 损失函数与数值稳定性
`CrossEntropyLoss`=log_softmax+nll_loss(数值稳定性) `reduction`模式 `MSELoss`/`L1Loss`/`SmoothL1Loss` label smoothing 类别不均衡加权 `NLLLoss`与CrossEntropyLoss的关系 `BCEWithLogitsLoss` vs `BCELoss`+Sigmoid

### 06 优化器内部机制
SGD基础更新公式 Momentum动量推导 Nesterov Momentum Adam一阶二阶矩估计 bias correction为什么需要 AdamW与Adam+L2正则化的本质区别 `optimizer.step()`内部机制 `param_groups`机制 学习率调度器原理+warmup `clip_grad_norm_`实现原理

### 07 训练循环深层机制
`zero_grad(set_to_none=True)` 梯度累加实现原理 `autocast`原理 `GradScaler`原理(loss scaling) gradient checkpointing `torch.compile`简介 学习率warmup的作用

### 08 内存与性能
CUDA缓存分配器基本原理 `.item()`/`.cpu()`同步开销 `memory_format` channels_last `torch.jit.script` vs `trace` 显存泄漏常见成因 `empty_cache()`的作用与误解 显存profiling基础 `num_workers`/`pin_memory`对DataLoader吞吐的影响

### 09 分布式训练基础机制
`DataParallel`的问题(为什么被弃用) `DistributedDataParallel`基本原理 all-reduce梯度同步机制 gradient bucketing `SyncBatchNorm`简介 与本仓库 [distributed-inference](../../learning/distributed-inference/) 模块的分工说明(那边讲推理时的模型并行,这里讲训练时的梯度同步)

### 10 序列化与部署基础
`state_dict()`保存/加载机制 `strict=False`使用场景 `map_location`跨设备加载 只存state_dict vs 存整个模型对象 ONNX导出简介 TorchScript序列化

### 11 调试与常见报错精解
CUDA out of memory 排查思路 "modified by an inplace operation"精解(呼应01/02篇) "element 0 of tensors does not require grad" 报错解读 size/shape不匹配debug方法论 `detect_anomaly()`使用 nan/inf定位方法 设备不一致报错 requires_grad误用 reshape/view报错debug(呼应01篇)

---

*更新:2026-07-07*
