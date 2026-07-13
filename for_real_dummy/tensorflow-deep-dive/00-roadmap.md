# TensorFlow 逐机制精讲 —— 路线图与进度表

> 目标:100 个 TensorFlow 知识点,由浅入深,分批次完成,深度对标 [torch-deep-dive/](../torch-deep-dive/00-roadmap.md)(面试二三四面深度,不是"这个函数怎么调")。
> 定位:这是仓库第五条深挖系列,和 torch-deep-dive 是"同一套心智模型,两个框架"的关系——两边都是 DL 框架的底层机制精讲,可以对照着看,遇到概念相通的地方会互相交叉引用,不重复推导。

---

## 0. 环境声明(先读,决定了本系列所有例子的真实行为)

**运行环境:** WSL2(Ubuntu 24.04 LTS)+ 独立 venv(`~/tf-venv`,不与仓库另一套尚未实际安装的 torch+vllm+verl WSL 环境混用)。Windows 原生环境从 TensorFlow 2.11 起官方不再支持 GPU,这是本系列必须跑在 WSL2 里的唯一原因,不是过度谨慎。

**锁定版本(本系列所有例子的实测环境,和 torch-deep-dive 锁定"2.11.0+cu128"是同一个做法):**
- TensorFlow **2.21.0**
- GPU:NVIDIA GeForce RTX 3080 Ti Laptop GPU(WSL2 GPU 直通,驱动 595.97),`tf.config.list_physical_devices('GPU')` 已验证非空
- **Keras 版本策略,这是本系列一个主动决定,不是默认行为**:自 TF 2.16 起 `pip install tensorflow` 默认让 `tf.keras` 指向 **Keras 3**(多后端架构,和历史上内嵌 TF 里的 Keras 2 语义/内部结构不同)。但面试题库和多数教程默认的心智模型仍是经典 tf.keras——本系列因此显式安装 `tf_keras` 包并设置环境变量 `TF_USE_LEGACY_KERAS=1`,让 `tf.keras` 解析回经典实现(已验证:`type(model).__module__` 落在 `tf_keras.*` 而不是 `keras.src.*`)。**如果你在自己电脑上复现例子,必须先 `pip install tf_keras` 并设置这个环境变量,否则部分例子的内部机制描述(尤其 04 类)可能和你观察到的实际行为对不上。** 04/12 类各有一条知识点专门讲这个 Keras 2/3 分裂现状,面试被问到该怎么答。
- 环境变量(已写入 `~/tf-venv/bin/activate`,`source` 激活即生效,不用每次手动设置):
  ```
  export LD_LIBRARY_PATH=$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib | paste -sd: -)
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  export TF_CPP_MIN_LOG_LEVEL=2
  export TF_USE_LEGACY_KERAS=1
  ```
  第一条是本系列踩到的第一个真实坑:`pip install tensorflow[and-cuda]` 装的 NVIDIA CUDA 库分散在十几个 `nvidia/*/lib` 子目录里,TF 2.21 无法自动发现,不设置这行 `import tensorflow` 能成功但 GPU 列表是空的,报错信息也不会明说缺哪个库——这本身就是 13 类"调试与常见报错精解"里的一条真实素材。

**关于"AI 研究/工程场景"这一段的诚实声明(全系列统一适用,不逐条重复):** 仓库 `learning/` 目录下没有任何 TensorFlow/Keras 代码(和 numpy/torch 系列不同,那两条能大量引用博士学长自己写的真实代码)。本系列的"AI 研究/工程场景"段落因此**不是仓库引用,是根据真实训练/部署中会遇到的具体问题重构的场景化例子**——不是编造的玩具场景,但也不能像 torch-deep-dive 那样标注仓库文件路径+行号。少数能和仓库已有内容做"跨框架分工对照"的地方(比如分布式训练一节可能对照 `learning/distributed-inference/`)会照常标注,那是真实关联,不在此声明豁免范围内。

---

## 每个知识点的固定讲解结构(与 torch-deep-dive 完全一致)

1. **签名/是什么** —— API 或概念定义,人话翻译
2. **一句话** —— 是什么
3. **底层机制 / 为什么这样设计** —— 不停在"怎么用",讲到"为什么必须是这样"、内部实际发生了什么
4. **AI 研究/工程场景** —— 场景化例子(见上方声明,非仓库引用)
5. **可运行例子** —— 带 `assert` 验证,能内省的地方现场打印内部状态,真实在 WSL2 `~/tf-venv` 里跑过
6. **面试怎么问 + 追问链** —— 面试官大概率怎么问,追问会往哪个方向深挖
7. **常见坑**

---

## 进度表(由浅入深)

| # | 分类 | 文件 | 知识点数 | 状态 |
|---|------|------|---------|------|
| 01 | Tensor 基础与 tf.Variable | [01-tensor-and-variable.md](01-tensor-and-variable.md) | 8 | ✅ 已完成(已验证,8/8代码块) |
| 02 | GradientTape 自动微分机制(★全系列重中之重) | [02-gradienttape-internals.md](02-gradienttape-internals.md) | 10 | ✅ 已完成(已验证,20/20代码块) |
| 03 | tf.function 与 AutoGraph 计算图机制(★全系列重中之重,TF独有) | [03-tf-function-and-autograph.md](03-tf-function-and-autograph.md) | 10 | ✅ 已完成(已验证,12/12代码块) |
| 04 | Keras 模型构建三套 API 内核 | [04-keras-api-internals.md](04-keras-api-internals.md) | 7 | ✅ 已完成(已验证,21/21代码块) |
| 05 | 常用层前向反向数学推导 | [05-layers-math-and-backward.md](05-layers-math-and-backward.md) | 7 | ✅ 已完成(已验证,7/7代码块) |
| 06 | 损失函数与数值稳定性 | [06-loss-functions-and-numerical-stability.md](06-loss-functions-and-numerical-stability.md) | 5 | ✅ 已完成(已验证,5/5代码块) |
| 07 | 优化器内部机制 | [07-optimizer-internals.md](07-optimizer-internals.md) | 7 | ✅ 已完成(已验证,26/26代码块) |
| 08 | fit() 内核与自定义训练循环 | [08-training-loop-internals.md](08-training-loop-internals.md) | 9 | ✅ 已完成(已验证,14/14代码块,含4处自查修正) |
| 09 | tf.data 输入管道机制(TF独有) | [09-tf-data-pipeline.md](09-tf-data-pipeline.md) | 8 | ✅ 已完成(已验证,19/19代码块) |
| 10 | 内存与性能 | [10-memory-and-performance.md](10-memory-and-performance.md) | 6 | ✅ 已完成(已验证,14/14代码块) |
| 11 | 分布式训练基础机制 | [11-distributed-training-basics.md](11-distributed-training-basics.md) | 6 | ✅ 已完成(已验证,9/9代码块) |
| 12 | 序列化与部署基础 | [12-serialization-and-deployment.md](12-serialization-and-deployment.md) | 8 | ✅ 已完成(已验证,13/13代码块) |
| 13 | 调试与常见报错精解 | [13-debugging-and-common-errors.md](13-debugging-and-common-errors.md) | 9 | ✅ 已完成(已验证,27/27代码块,含AutoGraph机制修正,系列收尾) |
| 14 | 进阶深度追加:5 个多级追问链案例 | [14-advanced-interview-depth.md](14-advanced-interview-depth.md) | 5案例(不计入100) | ✅ 已完成(已验证,10/10代码块独立通过;基于真实WebSearch调研的5条追问轴线撰写——retracing性能陷阱诊断链(现场证明`reduce_retracing=True`对Python标量退化完全无效、对Tensor shape变化真实有效,并额外隔离出"optimizer首次创建slot变量导致多trace一次"这一独立机制,25步真实训练step量化出5.41倍速度差异)、GPU显存贪婪分配决策依据(双OS进程真实共享一张物理GPU实测,发现"默认贪婪邻居已占84.7%显存"后另一进程申请8-13GB仍能成功这一调研阶段未预料到的真实现象,继而现场定界出"组合占用被收在物理16384MiB容量内、且请求量推高到15000MiB确实会真实OOM"的具体边界)、训练规模递增分布式决策(真实计算量化eager下MirroredStrategy.run比tf.function包裹慢5.28倍;并现场测出2个虚拟副本吞吐反而是单设备的0.48倍,量化验证"虚拟设备不能评估真实多卡收益"这条边界)、部署格式选型方案批判迭代(SavedModel/.h5/.keras/TFLite五种交付形态体积实测对比,以及"isinstance在SavedModel重载后失效"导致fine-tuning脚本静默冻结0层而非1层的真实bug复现)、Keras 2/3环境声明真实性验证边界(现场证明"tf.keras解析正确"不代表裸`import keras`也是Keras 2,且同一根因在Layer混用时报错清晰、在Callback混用时报错高度隐晦,诊断难度不在一个量级)) |

**合计:100 个知识点,13 篇 + 1 篇进阶深度追加(5 个案例,不计入 100),全部完成并独立验证。**

02(GradientTape)+ 03(tf.function/AutoGraph)合计 20 项,占全系列五分之一,这个权重是有意为之:torch-deep-dive 的"重中之重"是 Autograd 一个类目,但 TF2 相比 PyTorch 最大的心智负担不是自动微分本身(两边概念上很像),而是"eager 代码什么时候会被悄悄 trace 成图、trace 之后哪些 Python 语义会变"——这恰好是 02+03 两个类目共同覆盖的内容。

---

## 每一批具体覆盖哪些知识点(明细)

### 01 Tensor 基础与 tf.Variable
`tf.constant` vs `tf.Variable`(不可变性,ResourceVariable底层实现) eager执行模型与`.numpy()`互操作 `tf.TensorShape`静态shape vs 动态shape(None维度,呼应03类) dtype转换与自动提升规则 `tf.device`设备放置与GPU/CPU数据搬运 ragged tensor简介 sparse tensor简介 tensor与numpy的内存共享/拷贝语义

### 02 GradientTape 自动微分机制
tape工作原理:记录哪些运算、怎么记录 `watch()`与自动watch规则(trainable Variable自动被watch) `persistent=True`与tape生命周期 高阶导数(nested tape) `stop_gradient` `custom_gradient`装饰器 `jacobian`与`batch_jacobian` `tape.gradient()` vs `tf.gradients()`(TF1遗留API) 多输出多输入梯度计算(`unconnected_gradients`参数) 与PyTorch autograd设计差异对比(tape-based vs define-by-run)

### 03 tf.function 与 AutoGraph 计算图机制
tracing追踪机制 retracing触发条件(shape/dtype变化、Python对象参数) AutoGraph如何把Python控制流(if/for/while)转成图操作 concrete function与多态函数 `tf.function`内`print`/副作用陷阱(`tf.print` vs `print`) `input_signature`固定签名减少retracing `tf.TensorArray`(为什么trace出的图里不能用Python list) `tf.control_dependencies`手动控制依赖 vs TF2自动控制依赖排序 `tf.function`内变量创建的限制(只能第一次调用时创建) XLA `jit_compile`机制本体(tracing之后为什么还要再编译一次)

### 04 Keras 模型构建三套 API 内核
`tf.Module`:Keras Layer/Model的真正基类 Sequential API Functional API与KerasTensor符号追踪(为什么能自动`summary()`) Model子类化API `call()`方法与`__call__`的关系、`training`参数自动传播陷阱 `build()`延迟构建机制 `get_config()`/`from_config()`序列化契约,以及Keras 2/3现状面试怎么答

### 05 常用层前向反向数学推导
Dense层 Conv2D(交叉引用torch-deep-dive的im2col推导) BatchNormalization(`momentum`语义与PyTorch相反的经典坑) LayerNormalization Dropout Embedding MultiHeadAttention内部拆分(`einsum`用法交叉引用numpy-deep-dive)

### 06 损失函数与数值稳定性
`SparseCategoricalCrossentropy` vs `CategoricalCrossentropy` `from_logits`参数的数值稳定性意义 `reduction`模式 自定义loss正确写法(继承`Loss`类 vs 函数式) `BinaryCrossentropy`相关坑

### 07 优化器内部机制
`tf.keras.optimizers`系列总览与legacy/新API切换 `apply_gradients`机制 `LearningRateSchedule`作为可调用对象(不是命令式`.step()`) weight decay在Adam/AdamW中的实现差异 优化器状态(slots)机制 gradient clipping 与torch优化器机制对比

### 08 fit() 内核与自定义训练循环
`model.fit()`内部实际发生了什么 `Model.train_step()`自定义覆写("全自动"和"手写eager循环"之间的第三条路) GradientTape手写训练循环 Keras Callback自定义写法 内置callbacks机制(`ModelCheckpoint`/`EarlyStopping`如何被`fit()`调用) metrics状态累积机制(`update_state`/`result`/`reset_state`) 混合精度与`LossScaleOptimizer`(loss scaling机制) `compile()`参数与`fit()`的关系 `validation_data`/`validation_split`内部机制

### 09 tf.data 输入管道机制
`Dataset.from_tensor_slices` `map()`与并行化(`num_parallel_calls`) `batch`/`shuffle`/`prefetch`正确顺序 `cache()` `AUTOTUNE` `interleave` 与PyTorch DataLoader的设计差异对比 与`learning/distributed-inference/`的跨框架关联说明(如适用)

### 10 内存与性能
`tf.config.experimental.set_memory_growth`(呼应00环境声明里踩过的GPU显存坑) `mixed_float16`实践(与08类loss scaling交叉引用) XLA性能实践(与03类机制本体交叉引用) `tf.profiler`简介 `tf.function`图优化(grappler简介) GPU显存OOM排查

### 11 分布式训练基础机制
`tf.distribute.MirroredStrategy` `strategy.scope()`拦截变量创建的机制 `TPUStrategy`简介 `MultiWorkerMirroredStrategy`简介 与PyTorch DDP的all-reduce机制对比 与`learning/distributed-inference/`的分工说明式交叉引用(呼应torch-deep-dive09的同款处理)

### 12 序列化与部署基础
SavedModel格式 HDF5(`.h5`)格式 新版`.keras`格式 `tf.train.Checkpoint` vs 整模型保存 `tf.saved_model.save`的signature机制(concrete function导出、`serving_default`) TFLite转换简介 ONNX互操作 Keras 3多后端与legacy tf.keras的选择(面试怎么答,呼应00环境声明)

### 13 调试与常见报错精解
`tf.function`追踪失败排查 shape不匹配debug(动态None维度导致的延迟报错) eager模式关闭排查(`tf.executing_eagerly()`) OOM排查 NaN/Inf定位(`tf.debugging`模块) retracing性能陷阱的识别与修复 设备不一致报错 Keras 2/3版本冲突报错排查 梯度为None排查(常见于变量未被watch或运算脱离tape)

---

*更新:2026-07-13(13篇知识点 + 14号进阶深度追加共5案例,全部完成并独立验证)*
