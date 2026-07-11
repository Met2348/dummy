# 10 · 内存与性能深挖(Memory and Performance)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批不是"新概念",而是把前面几篇建立的机制认知接到"性能"这个维度上——一个 TF 训练脚本跑得慢、显存爆了、`nvidia-smi` 和代码里查到的数字对不上,原因几乎总能落在本篇这 6 个知识点里的某一个。这也是"面试二三四面"最喜欢挖的一类问题:面试官通常不问"这个 API 怎么用",而是甩给你一个"训练脚本莫名其妙 OOM 了/变慢了",看你有没有系统性的排查思路——这一点和 [torch-deep-dive/08-memory-and-performance.md](../torch-deep-dive/08-memory-and-performance.md) 的定位完全一致,两边讲的是"同一类工程问题,两套不同框架的机制",遇到概念相通的地方本篇会顺手点一句,不重复推导。

**本文和其他类目的关系:** 03 类目(`tf.function` 与 AutoGraph 计算图机制)讲 `jit_compile=True` 这个开关本身在做什么、tracing 和编译是两件不同的事(见 [03-tf-function-and-autograph.md](03-tf-function-and-autograph.md) 第10条);本篇第3节直接借用这个机制结论,只关心"开了之后实测快了多少、什么时候不值得开"。08 类目(`fit()` 内核与自定义训练循环)会讲 `LossScaleOptimizer` 和 loss scaling 的具体机制——为什么 fp16 梯度会下溢、scale 系数怎么动态调整(见 [08-training-loop-internals.md](08-training-loop-internals.md));本篇第2节不重复这部分,只讲 `mixed_float16` 实测的性能数据和"哪些地方该留 fp32"。

**关于"AI 研究/工程场景"段落的说明(全系列统一声明,详见 00-roadmap.md):** 仓库里没有可引用的 TensorFlow/Keras 代码,以下场景是根据真实训练/部署中会遇到的具体问题重构的,不是仓库文件引用。

本文所有代码例子已在 WSL2(Ubuntu 24.04)+ `~/tf-venv`(TensorFlow **2.21.0**,GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU,**16384 MiB** 显存,compute capability 8.6,驱动 595.97)下实际跑通验证。**本篇对"性能/内存有没有差异"这类问题格外较真**:凡是给出的数字,都是当场用 `nvidia-smi`、`tf.config.experimental.get_memory_info()`、`time.perf_counter()` 实测得到的,不是转述文档或博客里的"经验数字";多次重跑方向不稳定、或者差异不明显的地方,会如实标注,不会为了"证明观点"挑一次好看的结果贴上去。

**本篇统一结构(与前面各篇一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(现场测量,不转述)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. `tf.config.experimental.set_memory_growth` —— GPU 显存增长模式

**是什么:**
```
tf.config.experimental.set_memory_growth(device, enable)   # device 是 PhysicalDevice 对象;必须在这块设备被真正"初始化"之前调用
tf.config.experimental.get_memory_growth(device)             # 读取"记录下来的配置值"——注意这不完全等于"运行时实际生效的行为",本节会验证这个坑
tf.config.set_logical_device_configuration(
    device, [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
)                                                              # 另一条路:不是"按需增长",而是把物理GPU硬切一块固定大小的"逻辑设备"
tf.config.experimental.set_virtual_device_configuration       # 同一个函数的旧(experimental)名字,本节会验证两者是同一个对象
```

**一句话:** TF 默认在第一次真正用到某块 GPU 的那一刻,会一次性向 CUDA driver 申请这块卡上几乎全部可用显存(哪怕这一步只是算一个三元素的向量),这是为了后续所有分配都能直接命中已经拿到手的这一大块显存池,不用再频繁找 driver 打交道;`set_memory_growth(device, True)` 把这个策略换成"按需增长"——第一次用多少就只拿多少,不够了再向 driver 申请更多。

**底层机制/为什么这样设计:** TF 用的也是一个类似 PyTorch caching allocator 的自管理显存池(BFC allocator,Best-Fit with Coalescing),目的和 [torch-deep-dive/08 第1节](../torch-deep-dive/08-memory-and-performance.md)完全一样——避免每次 tensor 分配/释放都触发一次昂贵的 `cudaMalloc`/`cudaFree`。但两边"池子初始怎么长大"的默认策略不同:PyTorch 默认第一次分配只按需要的大小申请,池子随后续请求逐步扩大;TF 的 BFC allocator 默认在设备**第一次被使用时**,就查询 driver 报告的空闲显存,按一个略小于 1 的系数直接申请下"几乎整块卡"——对单进程独占一张卡、从头训到尾的场景,这样做能保证后续训练过程中的显存请求永远命中已有的池子,不会训练跑到一半才发现显存不够,是一种"用空间换确定性"的设计取舍。代价在于:如果这块 GPU 要被多个独立进程共享(本系列在同一台机器上并发跑多个验证脚本、Jupyter 和另一个推理服务共用一张卡都是这种场景),第一个启动的 TF 进程会把几乎全部显存据为己有,哪怕它自己只用得上几十 MB,后启动的进程会直接因为申请不到显存而 OOM——这正是本系列环境搭建阶段真实撞到的坑(见 00-roadmap.md 环境声明),`~/tf-venv/bin/activate` 里因此写死了 `TF_FORCE_GPU_ALLOW_GROWTH=true`。

**AI 研究/工程场景:** 同一台开发机上,一个 Jupyter kernel 常驻着一个模型用来调试,同时另开一个终端跑单元测试或验证脚本(本系列自己的验证流程就是这种场景的真实例子)——如果两边都用 TF 默认的贪婪分配,后启动的那个几乎必然 OOM,哪怕两边真正用到的显存加起来远小于整卡容量;CI 流水线上多个 GPU 测试 job 共享同一张卡并发跑,同样会撞到这个问题。工程上的标准应对不是"改小 batch size",而是从一开始就让每个进程只按需要的量申请。

**可运行例子:**

先看这个仓库自己的 venv 配置(`TF_FORCE_GPU_ALLOW_GROWTH=true`)下,一次极小的运算实际拿了多少显存:
```python
import subprocess
import tensorflow as tf

def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])

gpus = tf.config.list_physical_devices('GPU')
assert len(gpus) == 1

before = gpu_used_mib()
with tf.device('/GPU:0'):
    x = tf.constant([1.0, 2.0, 3.0])
    _ = (x * 2).numpy()
after = gpu_used_mib()
delta = after - before
print(f"TF_FORCE_GPU_ALLOW_GROWTH=true(本仓库venv默认配置): 显存增量 = {delta} MiB")
# 实测: 342 MiB —— 只按需要的很小一块申请
assert delta < 2048   # 远小于2GB,证明growth模式下不会一次性拿走大块显存
```

再还原 TF 在"没有这条环境变量"时的出厂默认行为,对比同一卡(总显存 16384 MiB)的差异:
```python
import os, subprocess
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"   # 覆盖本仓库venv的默认设置,还原TF出厂行为

def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])

before = gpu_used_mib()
import tensorflow as tf
with tf.device('/GPU:0'):
    x = tf.constant([1.0, 2.0, 3.0])
    _ = (x * 2).numpy()
after = gpu_used_mib()
delta = after - before
print(f"TF_FORCE_GPU_ALLOW_GROWTH=false(出厂默认): 同样一次tensor乘法后,显存增量 = {delta} MiB")
# 实测: 13873 MiB —— 这块16384MiB显存的卡,一次三元素向量乘法就吃掉了约84.7%
assert delta > 10 * 1024   # 超过10GB,证明是"贪婪一次性抢占"而不是按需分配
```

再验证一个容易被忽略的坑:如果环境变量和代码里的显式调用"打架",谁赢?
```python
import os, subprocess
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"   # 环境变量说: 不要增长模式
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("调用 set_memory_growth 前, get_memory_growth() 返回:", tf.config.experimental.get_memory_growth(gpus[0]))
# 实测: None —— 环境变量虽然已经在生效,但只要没显式调用过 set_memory_growth(),这个查询接口就还没有"记录"

tf.config.experimental.set_memory_growth(gpus[0], True)   # 代码里说: 要增长模式
print("显式调用 set_memory_growth(gpus[0], True) 后:", tf.config.experimental.get_memory_growth(gpus[0]))
# 实测: True —— 但这只是"记录下你设置过的值",不代表运行时一定按这个值执行,往下看

def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])

before = gpu_used_mib()
with tf.device('/GPU:0'):
    x = tf.constant([1.0, 2.0, 3.0])
    _ = (x * 2).numpy()
after = gpu_used_mib()
delta = after - before
print(f"环境变量=false + 代码里 set_memory_growth(True) -> 实际显存增量: {delta} MiB")
# 实测: 13873 MiB —— 和"环境变量=false、代码里什么都不调用"完全一样!
# 代码里的 set_memory_growth(True) 被环境变量的 false 悄悄盖过了,get_memory_growth() 报的 True 是"假的"
assert delta > 10 * 1024   # 证明代码里的 set_memory_growth(True) 完全没有生效
```

用 `TF_CPP_MIN_LOG_LEVEL=0`(本仓库默认是 `2`,会把这条日志过滤掉)单独跑一遍上面这段代码,能在 stderr 里现场看到 TF 自己承认了这次覆盖(原文,来自 `gpu_bfc_allocator.cc:39`):
```
W0000 00:00:1783528697.734494    2778 gpu_bfc_allocator.cc:39] Overriding orig_value setting
because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was true.
```
这条日志把"orig_value"(代码里设置的 `True`)明确标成了被覆盖的对象——环境变量的优先级高于 Python 层的 `set_memory_growth()` 调用,这个结论反过来也成立(环境变量=true 时,代码里显式设 `False` 同样会被悄悄改回增长模式,`get_memory_growth()` 依然只回显你最后一次调用传的参数,不代表真实生效值)。

最后确认调用时机的硬限制,以及"硬切一块固定大小"这条替代路径:
```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
with tf.device('/GPU:0'):
    x = tf.constant([1.0])
    _ = (x * 2).numpy()   # GPU 已经被真正初始化

try:
    tf.config.experimental.set_memory_growth(gpus[0], False)
    raise AssertionError("预期这里应该抛出 RuntimeError")
except RuntimeError as e:
    print("RuntimeError:", repr(str(e)))
    # 实测原文: 'Physical devices cannot be modified after being initialized'
    assert "cannot be modified after being initialized" in str(e)

# 另一条路: 不做"按需增长",而是从物理GPU上硬切一块固定大小的逻辑设备出来
# (必须在新的进程里做,因为上面已经初始化过物理设备了——这里只验证API本身能正常调用)
gpus2 = tf.config.list_physical_devices('GPU')
assert tf.config.experimental.set_virtual_device_configuration is tf.config.set_logical_device_configuration
# 两者是同一个函数对象,set_virtual_device_configuration 只是新旧命名过渡期留下的 experimental 别名
```

**面试怎么问 + 追问链:**
- **Q:** "为什么 TF 训练脚本一启动,`nvidia-smi` 立刻显示占了十几 GB 显存,但模型明明很小?"—— 期望答出 TF 默认在设备首次使用时一次性预分配几乎全部显存这个机制,而不是猜"是不是内存泄漏"。
- **追问 1:** "怎么让它不这么做?"—— 期望说出 `set_memory_growth()` 或 `TF_FORCE_GPU_ALLOW_GROWTH` 环境变量,加分项是能讲清楚两者都存在时环境变量优先。
- **追问 2(深挖,区分度高):** "我在代码里写了 `set_memory_growth(gpu, True)`,为什么线上跑起来还是把显存占满了?"—— 期望能想到"检查有没有环境变量在悄悄覆盖代码里的设置",而不是反复检查调用时机对不对——这正是本节验证出来的真实优先级坑。
- **追问 3:** "`set_memory_growth` 和 `set_logical_device_configuration` 切固定大小,这两种显存管理思路有什么本质区别,分别适合什么场景?"—— 期望答出"增长模式上限仍是整卡,只是不预先占满;固定切分是给这块 GPU 立一个硬顶,适合需要在同一张物理卡上稳定跑多个独立进程、且想保证互相不越界抢占的场景(比如给多个模型服务各分一块)"。

**常见坑:** 只看到代码里写了 `set_memory_growth(gpu, True)` 就认为"这个进程一定是按需增长的"——如果运行环境(尤其是继承别人写好的 Docker 镜像、CI 配置、`activate` 脚本)里已经设置了 `TF_FORCE_GPU_ALLOW_GROWTH`,环境变量会不声不响地覆盖代码里的设置,`get_memory_growth()` 查询到的值也不能反映真实生效状态,只能反映"最后一次调用参数是什么"。另外 `set_memory_growth`/`set_logical_device_configuration` 都必须在这块 GPU 被任何真实运算触碰之前调用(哪怕只是 `tf.constant(...)` 这样的小操作),包括在同一个物理设备上不能对已经切分好的逻辑设备再重新配置——这类"物理设备已经初始化"的 `RuntimeError` 在 notebook 里反复执行同一个 cell 时格外容易碰到,因为 GPU 只在 kernel 生命周期内初始化一次,第二次重跑配置代码必然报错。

---

## 2. `mixed_float16` 实践 —— 性能数据与最佳实践

**是什么:**
```
tf.keras.mixed_precision.set_global_policy('mixed_float16')   # 全局切换到混合精度策略
policy = tf.keras.mixed_precision.global_policy()
policy.compute_dtype     # 前向/反向计算使用的dtype: 'float16'
policy.variable_dtype     # 变量(权重)存储使用的dtype: 'float32'
```

**一句话:** `mixed_float16` 让绝大部分层的前向/反向计算以 float16 进行(在有 Tensor Core 的 GPU 上吞吐更高、显存占用更小),但权重本身仍以 float32 存储("master weights"),部分对数值范围敏感的中间量(比如 BatchNormalization 的滑动统计量)也会继续留在 float32——计算精度降了,但"记账"精度没有全面降,这是它既能吃到速度红利、又不至于训练崩掉的关键设计;fp16 数值范围窄导致的梯度下溢问题由 `LossScaleOptimizer` 负责,机制本身见 08 类目,本节不重复。

**底层机制/为什么这样设计:** Ampere 及之后的 GPU(本机 3080 Ti Laptop,compute capability 8.6)上,Tensor Core 对 fp16/bf16 矩阵乘加运算的吞吐远高于 fp32——这是 `mixed_float16` 速度收益的硬件来源。但 fp16 只有 10 位尾数、5 位指数,可表示范围和精度都远小于 fp32:如果权重更新量、或者某些需要跨很多步骤累积的统计量也用 fp16 存储,长期累积误差或者数值下溢/溢出的风险很高。Keras 的 `Policy` 因此把"计算用什么精度"和"存储用什么精度"拆成两个独立维度:`compute_dtype='float16'` 只决定矩阵乘、卷积这类前向算子实际执行时的 dtype;`variable_dtype='float32'` 保证每个 `tf.Variable`(权重)本身始终是 float32——每次前向时权重被临时 cast 成 float16 参与计算,反向算出的梯度再以 float32 精度更新回主权重副本,这样计算的高吞吐和权重更新的数值稳定性可以同时兼顾。BatchNormalization 是一个专门做了例外处理的例子(下面例子会现场验证):它的输出计算 dtype 仍然是 float16,但内部的 `moving_mean`/`moving_variance`(训练过程里跨很多个 batch 累积的滑动统计量,不是可训练权重)被单独固定成 float32,不受全局 `compute_dtype` 支配——因为这类统计量需要长期跨步骤累积,fp16 精度下累积误差会明显更大。

**AI 研究/工程场景:** 大模型训练/微调几乎默认开启混合精度(fp16 或 bf16)以压榨 GPU 吞吐、节省显存,但不是所有层都能无脑转 fp16——softmax、loss 计算这类对数值范围敏感的操作即使在整体 fp16 策略下也经常需要手动确认或强制以 float32 计算,否则容易在训练中后期出现 loss 变成 NaN 但复现起来很随机的诡异 bug;实践中,决定"值不值得开混合精度",应该像本节一样先用具体的层配置和 profiling 数据验证收益,而不是照抄一个通用配置就假设一定有效——收益幅度和模型结构、batch size、算子构成关系很大。

**可运行例子:**

先确认 policy 的两个 dtype 维度,以及几种典型层在这个策略下的真实 dtype 分工:
```python
import tensorflow as tf

assert tf.keras.mixed_precision.global_policy().name == 'float32'   # 默认策略

tf.keras.mixed_precision.set_global_policy('mixed_float16')
policy = tf.keras.mixed_precision.global_policy()
assert policy.compute_dtype == 'float16'
assert policy.variable_dtype == 'float32'

dense = tf.keras.layers.Dense(8)
x = tf.random.normal((4, 16))
y = dense(x)
print("Dense: 输出dtype =", y.dtype, " kernel(权重)dtype =", dense.kernel.dtype)
assert y.dtype == tf.float16       # 计算输出是 fp16
assert dense.kernel.dtype == tf.float32   # 权重仍是 fp32 master weights

bn = tf.keras.layers.BatchNormalization()
xb = tf.random.normal((4, 8, 8, 3))
yb = bn(xb, training=True)
print("BatchNorm: 输出dtype =", yb.dtype, " gamma dtype =", bn.gamma.dtype,
      " moving_mean dtype =", bn.moving_mean.dtype)
assert yb.dtype == tf.float16
assert bn.gamma.dtype == tf.float32          # 可训练参数,遵循 variable_dtype
assert bn.moving_mean.dtype == tf.float32     # 滑动统计量,即使 compute_dtype=float16 也强制留 fp32

tf.keras.mixed_precision.set_global_policy('float32')  # 用完复位,避免影响同进程里的下一段代码
```

再实测一次完整训练 step(forward + backward + apply_gradients)的真实速度差异(Dense 堆叠,4096 维,batch=256,充分 warmup 后用 `tf.function` 跑,30 次取均值)。**注意:为了只测量 `mixed_float16` 本身的算子速度,这里刻意不接 `LossScaleOptimizer`(loss scaling 机制见 08 类目,这里接上会把两件事的开销混在一起测)——生产训练代码不能这样省略,必须按 08 类讲的方式包一层,否则 fp16 梯度下溢是真实风险,这里只是出于测量目的的取舍:**
```python
import tensorflow as tf
import time

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(1000),
    ])

def bench(policy_name, batch=256, in_dim=4096, iters=30, warmup=10):
    tf.keras.mixed_precision.set_global_policy(policy_name)
    model = build_model()
    opt = tf.keras.optimizers.SGD(0.01)
    x = tf.random.normal((batch, in_dim))
    y = tf.random.uniform((batch,), maxval=1000, dtype=tf.int32)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for _ in range(warmup):
        step()
    _ = step().numpy()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = step()
    _ = r.numpy()
    dt = (time.perf_counter() - t0) / iters
    tf.keras.mixed_precision.set_global_policy('float32')
    return dt

t_fp32 = bench('float32')
t_mp = bench('mixed_float16')
print(f"float32: {t_fp32*1000:.3f} ms/step, mixed_float16: {t_mp*1000:.3f} ms/step, speedup={t_fp32/t_mp:.2f}x")
# 实测: float32=10.227ms/step, mixed_float16=8.581ms/step, speedup=1.19x
assert t_mp < t_fp32   # 方向稳定: mixed_float16 更快,但幅度是中等的~1.2x,不是夸张的"数倍"
```

再单独隔离出"纯矩阵乘法"这个 Tensor Core 最能发挥的场景,看收益的上限大概在什么量级(4096×4096,30次取均值):
```python
import tensorflow as tf
import time

def bench_matmul(dtype, n=4096, iters=30, warmup=10):
    a = tf.random.normal((n, n), dtype=dtype)
    b = tf.random.normal((n, n), dtype=dtype)

    @tf.function
    def mm(a, b):
        return a @ b

    for _ in range(warmup):
        r = mm(a, b)
    _ = r.numpy()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = mm(a, b)
    _ = r.numpy()
    return (time.perf_counter() - t0) / iters

t32 = bench_matmul(tf.float32)
t16 = bench_matmul(tf.float16)
print(f"纯4096x4096矩阵乘法: float32={t32*1000:.3f}ms, float16={t16*1000:.3f}ms, speedup={t32/t16:.2f}x")
# 实测: float32=10.017ms, float16=7.695ms, speedup=1.30x
# 比完整训练step的1.19x略高——纯矩阵乘法是Tensor Core最能发挥的场景,
# 完整训练step里还混着算loss、算梯度、apply_gradients等非纯矩阵乘的部分,拉低了整体收益比例
assert t16 < t32
```

**面试怎么问 + 追问链:**
- **Q:** "`mixed_float16` 为什么能加速训练,是不是所有层都变成 fp16 了?"—— 期望讲出 compute_dtype 和 variable_dtype 拆分的设计,权重始终有一份 fp32 master copy。
- **追问 1:** "为什么权重不能也直接存 fp16,省下来的显存不是更多吗?"—— 期望答出"小梯度更新量在 fp16 精度下经常会被舍入成 0,长期训练权重实质上不再更新",这是权重必须保留 fp32 master copy 的核心原因。
- **追问 2(深挖):** "BatchNorm 层在 `mixed_float16` 策略下是完全按 fp16 计算的吗?"—— 期望答"输出计算是 fp16,但内部的滑动统计量(moving_mean/moving_variance)固定用 fp32 累积",能进一步追问为什么(长期累积对精度更敏感),说明真的读过源码或验证过,而不是死记"BN 要留 fp32"这一句话。
- **追问 3(工程开放题):** "如果开了 `mixed_float16` 后训练早期一切正常,几千 step 后 loss 突然变成 NaN,你会怎么排查?"—— 期望联系到 loss scaling/梯度下溢或上溢(机制见08类目),以及 `tf.debugging.enable_check_numerics()` 这类工具定位是哪个算子先出的 NaN。

**常见坑:** 把"开了 `mixed_float16`"和"一定能提速数倍"划等号——本节实测的完整训练 step 提速只有约 1.2 倍,提速幅度强依赖于模型里矩阵乘法/卷积这类能吃到 Tensor Core 红利的算子占比,如果模型主要瓶颈在数据加载、非矩阵运算(比如大量小算子的胶水代码)、或者矩阵本身太小根本喂不满 Tensor Core,mixed_float16 可能几乎没有收益甚至因为多了 cast 开销略微变慢;另外容易忽略的是,`set_global_policy` 是进程级的全局状态,写测试或者在同一个脚本里对比不同配置时,如果忘记在测完一组之后切回 `'float32'`,后面的代码会在不知情的情况下继续用 fp16 策略跑。

---

## 3. XLA 性能实践 —— `jit_compile=True` 实测数据

**是什么:**
```
@tf.function(jit_compile=True)
def f(x):
    ...
```
`jit_compile=True` 本身的编译机制(tracing 之后为什么还要再经过一次 XLA 编译、和普通 `tf.function` 图执行的区别)属于 03 类目范围(见 [03-tf-function-and-autograph.md](03-tf-function-and-autograph.md) 第10条),本节不重复,只关心它实测出来的性能数据,以及"什么时候不值得开"。

**一句话:** XLA 把 tracing 得到的图再编译一次,针对目标硬件生成融合、专用的机器码——对由很多小算子串联、能被融合成少数几个大 kernel 的计算图(典型:一长串逐元素运算),实测收益非常可观;但对本来就由 cuBLAS/cuDNN 这类已经高度优化的单个大算子主导的计算(典型:一次大矩阵乘法),XLA 编译带来的收益很小甚至可以忽略,而且编译本身有实打实的一次性开销,不是"开了就一定划算"的免费午餐。

**底层机制/为什么这样设计:** 一长串逐元素运算(`relu` → 乘常数 → `tanh` → ……)如果不做任何图级别优化,每一步都是一次独立的 CUDA kernel launch,每次 launch 都要把上一步的结果从显存读出来、算完再写回显存——这些中间结果本身没有复用价值,纯粹是"版本文件搬来搬去"的开销。XLA 编译时能看到完整的算子序列,把这一串能连续执行、没有旁路依赖的逐元素运算融合成一个 kernel,中间结果直接留在寄存器/片上缓存里流转,不需要每一步都读写显存,launch 次数也从 N 次变成 1 次——这是"融合"带来收益的根本原因,和 [torch-deep-dive/08 第7节](../torch-deep-dive/08-memory-and-performance.md)里"channels_last 让访存模式匹配硬件"是同一个大类的优化思路(减少不必要的显存读写),只是作用的层面不同(这里是算子间融合,而不是单个算子内部的数据排布)。而像 `a @ b` 这样的大矩阵乘法,cuBLAS 本身已经是经过高度调优、充分利用 Tensor Core 的实现,XLA 编译出来的 kernel 很难比它更快,融合也没有意义(前后没有别的逐元素运算可融合),这时候 XLA 唯一确定会带来的是编译本身的开销,收益基本为零。

**AI 研究/工程场景:** 自定义的、由较多小算子拼接而成的激活函数或正则化项(比如某些论文里定义的新激活函数、手写的近似计算),或者强化学习里那种有大量逐元素运算和条件判断、but 没有大矩阵乘主导的计算,是 XLA 收益最明显的场景;反过来,一个几乎全部由标准 `Dense`/`Conv2D`/`MultiHeadAttention` 堆出来的模型,主要瓶颈本来就在 cuBLAS/cuDNN 的大算子上,`jit_compile=True` 值不值得开,应该像本节一样先实测,而不是默认"开了总没坏处"——尤其是输入 shape 会变化的场景,每种新 shape 都要重新触发一次真实的 XLA 编译,如果 shape 变化很频繁,编译开销可能完全抵消甚至超过运行时收益。

**可运行例子:**

先看 XLA 融合收益最明显的场景——一长串逐元素运算(50 次 `relu`→缩放→`tanh`,2048×2048,30 次取均值):
```python
import tensorflow as tf
import time

def elementwise_chain(x):
    for _ in range(50):
        x = tf.nn.relu(x)
        x = x * 1.0001 + 0.0001
        x = tf.math.tanh(x)
    return x

plain_fn = tf.function(elementwise_chain)
xla_fn = tf.function(elementwise_chain, jit_compile=True)
x = tf.random.normal((2048, 2048))

def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        r = fn(x)
    _ = r.numpy()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn(x)
    _ = r.numpy()
    return (time.perf_counter() - t0) / iters

t_plain = bench(plain_fn)
t_xla = bench(xla_fn)
print(f"[逐元素链式运算] 普通tf.function: {t_plain*1000:.3f}ms/iter, jit_compile=True: {t_xla*1000:.3f}ms/iter, "
      f"speedup={t_plain/t_xla:.2f}x")
# 实测(独立两轮): 18.536ms vs 0.744ms (24.90x) ; 18.951ms vs 0.700ms (27.06x)
# 方向非常稳定,幅度在25~27倍这个量级——50次kernel launch被融合成极少数几次,launch开销和
# 中间结果显存读写被大量省掉,这正是XLA融合对"一长串小算子"最有效的教科书场景
assert t_xla < t_plain / 5   # 保守断言至少5倍,不卡具体倍数(避免机器状态波动导致断言过脆)
```

再看编译本身的一次性开销有多大(同一个 XLA 函数,第一次调用 vs 第二次调用):
```python
import tensorflow as tf
import time

def elementwise_chain(x):
    for _ in range(50):
        x = tf.nn.relu(x)
        x = x * 1.0001 + 0.0001
        x = tf.math.tanh(x)
    return x

xla_fn = tf.function(elementwise_chain, jit_compile=True)
x = tf.random.normal((2048, 2048))

t0 = time.perf_counter()
r = xla_fn(x)
_ = r.numpy()
t_first = time.perf_counter() - t0

t0 = time.perf_counter()
r = xla_fn(x)
_ = r.numpy()
t_second = time.perf_counter() - t0

print(f"[编译开销] 第一次调用: {t_first*1000:.3f}ms, 第二次调用: {t_second*1000:.3f}ms, 比值={t_first/t_second:.1f}x")
# 实测(独立两轮): 1419.9ms vs 8.0ms (177.3x) ; 848.8ms vs 12.0ms (70.9x)
# 编译开销本身在这台机器上大约0.8~1.4秒,量级远超单次运行耗时——这是"输入shape频繁变化的场景
# 不适合jit_compile"的直接证据:如果每个新shape都要重新付出接近1秒的编译成本,很容易得不偿失
assert t_first > t_second * 10   # 第一次调用(含编译)比第二次慢一个数量级以上
```

最后看 XLA "不怎么帮忙"的场景——一次大矩阵乘法(4096×4096,cuBLAS 本来就高度优化过):
```python
import tensorflow as tf
import time

def matmul_only(a, b):
    return a @ b

plain_mm = tf.function(matmul_only)
xla_mm = tf.function(matmul_only, jit_compile=True)
a = tf.random.normal((4096, 4096))
b = tf.random.normal((4096, 4096))

def bench2(fn, iters=30, warmup=10):
    for _ in range(warmup):
        r = fn(a, b)
    _ = r.numpy()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn(a, b)
    _ = r.numpy()
    return (time.perf_counter() - t0) / iters

t_plain_mm = bench2(plain_mm)
t_xla_mm = bench2(xla_mm)
print(f"[纯大矩阵乘法] 普通: {t_plain_mm*1000:.3f}ms/iter, jit_compile=True: {t_xla_mm*1000:.3f}ms/iter, "
      f"ratio={t_plain_mm/t_xla_mm:.2f}x")
# 实测(独立两轮): 9.405ms vs 9.191ms (1.02x) ; 9.610ms vs 8.631ms (1.11x)
# 差异在噪声量级附近,如实报告: 这个场景下jit_compile带来的收益微乎其微,
# 不像逐元素链式运算那样有数量级的提升——cuBLAS本身已经把这块硬件的算力压得很满了
print("如实说明: 大矩阵乘法场景下XLA收益不明显(1.0~1.1x量级),不夸大为'总是有用'")
```

**面试怎么问 + 追问链:**
- **Q:** "`jit_compile=True` 是不是应该默认给所有 `tf.function` 都开上?"—— 期望答"不是",并能讲出收益取决于计算图的结构(能不能融合)。
- **追问 1:** "什么样的计算最能吃到 XLA 的收益?"—— 期望答出"由很多小算子(尤其逐元素运算)串联、能被融合成少数几个 kernel 的计算",最好能举出本节的量级(实测约25~27倍)。
- **追问 2(深挖,反直觉):** "对一个几乎全是大矩阵乘法的模型开 XLA,会不会明显变快?"—— 期望答"通常不会",因为 cuBLAS 本身已经是高度优化的实现,XLA 编译很难再显著超越它——本节实测这种场景下差异在 1.0~1.1x 量级,基本可以视为噪声。
- **追问 3(工程场景):** "如果模型的输入 shape 会在训练中动态变化(比如变长序列不做 padding),开 `jit_compile=True` 需要注意什么?"—— 期望联系到"每个新 shape 都要重新触发一次真实编译",本节实测单次编译开销接近 1 秒,shape 变化频繁的话编译开销可能完全抵消收益,这也是 03 类目"retracing"话题在 XLA 场景下的延伸。

**常见坑:** 只用一次"随手写的例子"验证 XLA "有没有用",然后把结论泛化到所有场景——本节两个截然不同的测试(逐元素链 vs 大矩阵乘法)证明收益幅度和计算图结构强相关,不存在一个通用的"XLA 让 TF 快 N 倍"的数字。另外容易忽略编译开销:如果一个训练循环里 shape 频繁变化(动态 batch、变长序列),`jit_compile=True` 可能导致频繁重新编译,反而比不开还慢,这时候应该结合 03 类目讲的 `input_signature`/padding 策略先把 retracing/recompile 控制住,再考虑要不要开 XLA。

---

## 4. `tf.profiler` 简介 —— 抓一段训练的性能画像

**是什么:**
```
tf.profiler.experimental.start(logdir)              # 开始记录一段区间内的性能事件
tf.profiler.experimental.stop()                        # 停止记录,把结果写盘
with tf.profiler.experimental.Trace('train_step', step_num=i):
    ...                                                  # 给某一段代码打一个可在时间线上识别的标签
```
更贴近日常使用的入口其实是 Keras 回调:`tf.keras.callbacks.TensorBoard(log_dir=..., profile_batch='10,20')`——`fit()` 内部会在指定的 batch 区间自动调用上面这套底层 API,不需要手写 start/stop。

**一句话:** `tf.profiler` 在指定的一段代码执行期间,记录下这段时间里实际发生的算子调度、kernel 执行、Python/C++ 调用栈等事件,写成 TensorBoard 能读取的 profile 文件——用来回答"这一步训练的时间到底花在哪"这种单靠 `time.perf_counter()` 只能测出总耗时、测不出内部构成的问题。

**底层机制/为什么这样设计:** 只测"一个训练 step 总共花了多少毫秒",遇到"变慢了"这种问题时定位不到根因——可能是某个特定算子慢、可能是数据没喂上导致 GPU 在等、也可能是 CPU 端 Python 逻辑本身占用了不该占用的时间。`tf.profiler.experimental` 在底层同时接入两类信号源:一类是 host 端的事件(Python/C++ 函数调用、`tf.function` 的 trace 触发点),另一类是 device 端的真实 kernel 执行事件(通过 NVIDIA CUPTI 接口拿到每个 CUDA kernel 何时开始、何时结束、用了多少显存带宽)——两条时间线对齐后写进一个 `.xplane.pb` 文件,TensorBoard 的 Profiler 插件读取这个文件后能画出"host 在干什么、GPU 在干什么、两者有没有互相等待"的时间线视图,这正是排查"GPU 利用率低但训练还在跑"(呼应 [torch-deep-dive/08 第8节](../torch-deep-dive/08-memory-and-performance.md)提到的同一类症状)这种问题时最直接的诊断工具。

**AI 研究/工程场景:** 训练脚本"看起来在正常跑,但 GPU 利用率长期只有 20~30%"是最经典的需要上 profiler 的场景——不打开性能画像,很难判断这 70% 的空闲时间到底是 `tf.data` 输入管道跟不上(见 09 类目)、是某个自定义层写得低效、还是每一步都被一次不必要的 `.numpy()`/`print` 同步调用打断了流水线。相比"改一下代码看看是不是变快了"这种试错式排查,先抓一段真实的性能画像再决定往哪个方向优化,是更系统的工程方法。

**可运行例子:**

```python
import os, shutil, glob
import tensorflow as tf

logdir = "/tmp/tf_profiler_test"
shutil.rmtree(logdir, ignore_errors=True)
os.makedirs(logdir, exist_ok=True)

@tf.function
def train_step(x, w):
    return tf.nn.relu(x @ w)

x = tf.random.normal((256, 256))
w = tf.random.normal((256, 256))
for _ in range(3):
    train_step(x, w)   # warmup: 排除首次trace/编译开销,不让它污染profile结果

tf.profiler.experimental.start(logdir)
for i in range(10):
    with tf.profiler.experimental.Trace('train_step', step_num=i):
        train_step(x, w)
tf.profiler.experimental.stop()

found = sorted(glob.glob(logdir + "/**/*", recursive=True))
print("logdir下生成的文件/目录:")
for f in found:
    print(" ", f)

# 实测: 会生成 <logdir>/plugins/profile/<时间戳>/<hostname>.xplane.pb
assert any(f.endswith(".xplane.pb") for f in found)
assert any("/plugins/profile/" in f.replace("\\", "/") for f in found)
```

**面试怎么问 + 追问链:**
- **Q:** "训练脚本 GPU 利用率上不去,你会怎么排查?"—— 期望提到用 profiler 抓一段真实的时间线,而不是靠猜或者盲目调参数。
- **追问 1:** "光看 `time.perf_counter()` 测出来的总耗时,为什么不够?"—— 期望答出"测不出耗时具体花在 host 端还是 device 端、卡在哪个算子",profiler 能把这条时间线拆开看。
- **追问 2(深挖):** "为什么要先跑几步 warmup 再开始 profiling,不能一开始就抓?"—— 期望联系到 03 类目的 tracing/(如果开了XLA)编译开销:如果不排除首次调用,profile 结果里会混进"这一次性的编译时间",不能代表稳定训练阶段的真实性能画像,本节例子里特意warmup三步再start。
- **追问 3(工程场景):** "`profile_batch='10,20'` 这种写法是什么意思,为什么不直接 profile 整个训练过程?"—— 期望答出"只 profile 一小段区间"是有意为之:profile 文件会记录海量细粒度事件,长时间全程记录会导致文件巨大、写盘本身也拖慢训练,通常只需要挑一段"已经进入稳定状态"的区间抓一次就够诊断问题。

**常见坑:** 忘记做 warmup 直接开始 profiling,导致抓到的画像里第一步的 tracing/(若开了 XLA)编译耗时占了大头,误判"这一步怎么这么慢",其实只是首次调用的固定开销,不代表训练稳定跑起来之后的真实速度——这和 03 类目里"retracing 排查"是同一类思维方式,判断问题前先把"一次性开销"和"稳定态开销"分清楚。另外要注意生成的 profile 文件本身只是数据,真要在 TensorBoard 里可视化查看时间线视图,还需要额外安装 `tensorboard-plugin-profile` 这个独立打包的插件包,只装 `tensorboard` 本体是打不开 Profile 面板的。

---

## 5. `tf.function` 图优化 —— Grappler 简介

**是什么:**
```
tf.config.optimizer.set_experimental_options({
    "constant_folding": True,    # 常量折叠: 编译期能算出的子表达式直接算掉,不留到运行时
    "remapping": True,             # 算子融合/重映射: 把常见的算子组合替换成一个更高效的融合算子
    "arithmetic_optimization": True,
    ...
})
```

**一句话:** Grappler 是 TF 内建的、基于规则的计算图重写系统,在图被真正执行之前自动跑一遍,做常量折叠、公共子表达式消除、算子融合等经典编译优化——它和 XLA 是两个不同层次的优化:Grappler 在图这一层做"结构性"重写(改变图里有哪些节点、节点怎么连),XLA 则是在图确定之后再把它编译成硬件相关的机器码,两者不冲突,默认都是开着的,可以叠加生效。

**底层机制/为什么这样设计:** 一段 `tf.function` 追踪出来的原始图,常常包含明显可以简化的结构——比如两个都是编译期已知的常量相乘,没有理由留到运行时才真正执行那次乘法;又比如 `Conv2D` 后面紧跟 `BiasAdd` 再紧跟 `Relu`,与其发起三次独立的 kernel launch,不如识别出这个常见模式、融合成一个专门的 `_FusedConv2D` kernel 一次做完。Grappler 用一系列独立的"优化器 pass"(constant folding、arithmetic optimizer、remapper、layout optimizer、memory optimizer……)依次遍历、重写这张图,每个 pass 只关心一类特定的重写规则,互相之间可以独立开关。**这里有一个容易踩的认知坑,本节例子会现场验证:** 用 `concrete_function.graph.as_graph_def()` 拿到的,是 Python 层 tracing 阶段产出的**原始**图,Grappler 优化发生在这之后、真正执行之前的运行时阶段,直接检查 `concrete_function.graph` 看不到任何 Grappler 重写的痕迹——想现场验证 Grappler 到底做了什么,需要用 `tensorflow.python.grappler.tf_optimizer.OptimizeGraph` 这个更底层的接口,手动把 Grappler 的优化 pass 跑一遍。

**AI 研究/工程场景:** Grappler 是默认开启、自动生效的优化,日常写模型代码基本不需要手动干预;但排查"这段计算图为什么执行得比预期慢"或者"这两次运行的图结构好像不一样"时,知道 Grappler 会在背后重写图,能避免"直接打印 concrete function 的 graph_def、却看不出任何优化痕迹"这类误判——这正是本节要重点验证的坑。少数场景下会需要手动关闭某个 pass 排查问题(比如怀疑某个自定义算子和某个融合规则冲突导致数值结果不对),`tf.config.optimizer.set_experimental_options` 就是用来做这种细粒度开关的接口。

**可运行例子:**

先验证"直接看 concrete function 的 graph_def 看不到 Grappler 优化"这个认知坑,再用正确的方式(`tf_optimizer.OptimizeGraph`)现场验证常量折叠确实发生了:
```python
import tensorflow as tf
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.framework import meta_graph
from tensorflow.core.protobuf import rewriter_config_pb2, config_pb2

@tf.function
def f(x):
    a = tf.constant(2.0)
    b = tf.constant(3.0)
    c = a * b          # a、b都是编译期已知的常量,理论上可以被提前算掉
    return x + c

cf = f.get_concrete_function(tf.TensorSpec([], tf.float32))
raw_graph_def = cf.graph.as_graph_def()
raw_ops = [n.op for n in raw_graph_def.node]
print("直接看 concrete_function.graph_def 的op列表(tracing阶段的原始图):", raw_ops)
# 实测: ['Placeholder', 'Const', 'Const', 'Mul', 'AddV2', 'Identity'] —— Mul还在,
# 说明这里拿到的确实是Grappler优化"之前"的原始图,不是运行时真正执行的那张图
assert "Mul" in raw_ops

# 用grappler的底层接口,手动把常量折叠这个pass跑一遍,看真正执行时的图长什么样
mgd = meta_graph.create_meta_graph_def(graph_def=raw_graph_def)
fetch_name = cf.graph.outputs[0].op.name
mgd.collection_def["train_op"].node_list.value.append(fetch_name)   # 告诉grappler哪个节点是最终输出,不能被裁掉

def optimize(rewriter_config):
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.CopyFrom(rewriter_config)
    out = tf_optimizer.OptimizeGraph(config, mgd)
    return [n.op for n in out.node]

on_cfg = rewriter_config_pb2.RewriterConfig()
on_cfg.constant_folding = rewriter_config_pb2.RewriterConfig.ON
ops_on = optimize(on_cfg)
print("grappler常量折叠=ON:", ops_on)
# 实测: ['Const', 'Placeholder', 'AddV2', 'Identity'] —— Mul消失了!
# a*b在图优化阶段就被直接算成一个常量6.0,运行时根本不会再发起一次乘法kernel
assert "Mul" not in ops_on

off_cfg = rewriter_config_pb2.RewriterConfig()
off_cfg.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
ops_off = optimize(off_cfg)
print("grappler常量折叠=OFF:", ops_off)
# 实测: ['Const', 'Const', 'Placeholder', 'Mul', 'AddV2', 'Identity'] —— 和原始图一致,Mul保留
assert "Mul" in ops_off
```

**面试怎么问 + 追问链:**
- **Q:** "Grappler 是做什么的,和 XLA 是同一回事吗?"—— 期望讲出"图结构层面的规则重写"和"面向硬件的编译"是两个不同层次,默认都开着、可以叠加。
- **追问 1:** "举一个 Grappler 具体做了什么优化的例子。"—— 期望能说出常量折叠(本节现场验证过)或者算子融合(比如 `Conv2D+BiasAdd+Relu` 融合成 `_FusedConv2D`)这类具体例子,而不是只会说"图优化"这种空泛的词。
- **追问 2(深挖,区分度高):** "我直接打印 `concrete_function.graph.as_graph_def()`,为什么完全看不出 Grappler 做了任何优化?"—— 期望答出"这是 tracing 阶段的原始图,Grappler 在真正执行前的运行时阶段才生效",能进一步说出需要用 `tf_optimizer.OptimizeGraph` 之类的接口才能看到优化后的结果,说明真的动手验证过,而不是单纯背结论。
- **追问 3(工程场景):** "什么情况下你会想手动关掉某个 Grappler 优化 pass?"—— 开放题,合理答案包括"怀疑某个自定义算子和某个融合/重写规则有冲突,产出数值不对,想通过关闭某个pass做排查性隔离"。

**常见坑:** 用 `concrete_function.graph.as_graph_def()` 或者 `tf.function` 的 `pretty_printed_concrete_signatures()` 去验证"某个优化到底生效没有"——这些接口拿到的都是 tracing 阶段的原始图或签名信息,不反映 Grappler 在执行前做的重写,本节已经现场验证过这个认知坑;另外,`tf.config.optimizer.set_experimental_options()` 是进程级全局配置,在同一个进程里反复切换不同配置做对比实验时,记得每次测完要把选项改回默认值(或者显式设成你下一段代码需要的值),否则容易把"上一次实验遗留的配置"误认为是"当前配置的默认行为"。

---

## 6. GPU 显存 OOM 排查 —— 常见成因和诊断方法

**是什么:**
```
tf.config.experimental.get_memory_info(device)     # {'current': 当前真实占用字节数, 'peak': 峰值字节数}
tf.config.experimental.reset_memory_stats(device)    # 把峰值统计清零,重新开始统计一段代码的峰值
tf.errors.ResourceExhaustedError                       # OOM时TF抛出的异常类型
```

**一句话:** GPU 显存 OOM 的报错信息本身通常是准确的(TF 会明确说清楚是在给哪个 shape 的哪个 op 分配显存时失败),真正需要排查功夫的是"为什么会走到这一步"——常见原因不外乎"这一步确实需要的显存超过了物理上限"(batch/模型太大)、"另一个进程按第1节讲的贪婪模式占掉了大半张卡"(不是真的显存不够,是分配策略问题)、"`tf.function` 因为参数不稳定反复 retrace,每次 retrace 都留下新的图结构不被及时释放"(本节会现场验证这第三种、容易被忽视的成因)。

**底层机制/为什么这样设计:** `get_memory_info()` 返回的是"逻辑上真正被 tensor 引用"的字节数(和第1节讨论的 BFC 分配池"实际占用"是同一层含义,只是这里是从 TF 自己的统计接口读,不是从 `nvidia-smi` 外部观察),`current` 是这一刻的快照,`peak` 是从上次 reset 以来出现过的历史最大值——判断"这个训练配置到底需要多大显存的卡才够用"应该看 `peak`,而不是训练稳定跑起来之后某一刻的 `current`,因为真正的显存高峰经常出现在一个容易被忽略的瞬间(比如前向的中间激活值还没释放完、反向的梯度已经在分配的交界点),这个道理和 [torch-deep-dive/08 第7节](../torch-deep-dive/08-memory-and-performance.md)完全一致。TF eager 模式下,一个 tensor 不再被任何 Python 变量引用时,它占用的显存会立刻被自动释放回 BFC 池(基于引用计数,不需要像 PyTorch 那样等 `backward()` 主动清空计算图——TF eager 的 `GradientTape` 默认只在 `with` 语句块内维护记录,`tape.gradient()` 调用完之后,没有 `persistent=True` 的 tape 会自动释放这些记录),所以"某个 Python list 里存了一堆裸 tensor 忘记及时清理"这类典型的 PyTorch 式泄漏场景,在 TF eager 下通常不会持续累积。TF 训练里真正常见、容易被漏查的隐蔽显存增长源,是 `tf.function` 的 **retracing**:每一次新的 trace 都会生成一套新的 `ConcreteFunction`/图结构(如果开了 XLA,还包括一份新的编译产物),这些结构本身占用真实显存,如果输入参数(尤其是被当成 Python 值而不是 tensor 传入的标量、或者 shape 一直在变的输入)导致每次调用都触发新的 trace,这些图结构会随着训练进行不断堆积——这是"看起来在正常训练,显存却在缓慢爬升"的一个和 PyTorch 完全不同的、TF 特有的成因,03 类目会讲 retracing 本身的触发条件,这里只关心它和显存增长的关系。

**AI 研究/工程场景:** 提交到共享集群的训练任务,一次因为显存预估错误的 OOM 可能浪费数小时排队等到的机器资源——上线前用 `get_memory_info` 的 `peak` 字段先摸清楚真实峰值需求,比反复提交、被 OOM 打回来再调小 batch size 试探效率高得多;长跑任务"训练到某个 epoch 突然 OOM",排查思路应该是定期打点 `get_memory_info()['current']`,看是不是随着 step 数近似线性增长——如果是,再进一步区分"是不是某个数据相关的 Python 值被当成参数传给了 `tf.function`,导致每步都在 retrace"(本节现场复现这个场景),而不是想当然地怀疑是显存泄漏。

**可运行例子:**

先用第1节的"硬切固定大小"机制,把显存人为限制到一个很小的值,构造一个能快速、可复现触发的 OOM,同时验证 `get_memory_info` 怎么随着连续分配逐步逼近上限:
```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=512)]   # 硬切512MB,让OOM能快速、稳定复现
)

def mb(x):
    return x / 1024**2

with tf.device('/GPU:0'):
    chunks = []
    triggered = False
    try:
        for i in range(20):
            chunks.append(tf.zeros((10_000_000,), dtype=tf.float32))   # 每块约38.1MB
            info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"step {i}: current={mb(info['current']):.1f}MB peak={mb(info['peak']):.1f}MB")
    except tf.errors.ResourceExhaustedError as e:
        triggered = True
        msg = str(e)
        print("\n--- 真实触发的 ResourceExhaustedError ---")
        print(msg[:260])
        # 实测原文(节选): "OOM when allocating tensor with shape[10000000] and type float
        #   on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc [Op:Fill]"
        assert "OOM when allocating tensor" in msg
        assert "GPU_0_bfc" in msg

    assert triggered   # 512MB的硬上限,连续分配38MB的块,必然会撞到
    final_info = tf.config.experimental.get_memory_info('GPU:0')
    print("最后一次成功分配后的状态:", {k: round(mb(v), 1) for k, v in final_info.items()}, "MB")
```

再验证第三种容易被忽视的成因——`tf.function` 因为参数不稳定反复 retrace,即使张量本身没有真正意义上的"泄漏",也会带来实打实的显存增长:
```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def mb(x):
    return x / 1024**2

@tf.function
def f(x, scale):
    return x * scale   # scale 是一个 Python float,不是 tensor —— 每次传不同的值都会被当成新的trace key

with tf.device('/GPU:0'):
    x = tf.ones((1_000_000,), dtype=tf.float32)   # ~3.8MB

    base = tf.config.experimental.get_memory_info('GPU:0')
    for i in range(30):
        _ = f(x, float(i) + 0.001)   # 每次都是不同的python float -> 每次都retrace
    after_retrace = tf.config.experimental.get_memory_info('GPU:0')
    n_traces = f.pretty_printed_concrete_signatures().count("Input Parameters")
    delta_retrace = mb(after_retrace['current'] - base['current'])
    print(f"30次调用、每次传不同python float: 显存增量={delta_retrace:.1f}MB, 实际trace次数={n_traces}")
    # 实测: 增量4.2MB, trace了30次(每次调用都retrace)

@tf.function
def g(x, scale):
    return x * scale

with tf.device('/GPU:0'):
    scale_t = tf.constant(2.0)
    base2 = tf.config.experimental.get_memory_info('GPU:0')
    for i in range(30):
        _ = g(x, scale_t)   # 每次传同一个tensor -> 只会trace一次,后面29次都复用
    after2 = tf.config.experimental.get_memory_info('GPU:0')
    n_traces2 = g.pretty_printed_concrete_signatures().count("Input Parameters")
    delta_stable = mb(after2['current'] - base2['current'])
    print(f"30次调用、每次传同一个tensor: 显存增量={delta_stable:.1f}MB, 实际trace次数={n_traces2}")
    # 实测: 增量0.0MB, 只trace了1次

assert n_traces == 30 and n_traces2 == 1
assert delta_retrace > delta_stable   # 同样是"跑30次"这段代码,retrace与否带来的显存增长完全不同
```

调用 `f(x, float(i)+0.001)` 这种写法在真实训练代码里会打印一条肉眼可见的警告(实测原文,来自 TF 的 retracing 检测逻辑,地址部分每次运行都不同):
```
WARNING:tensorflow:5 out of the last 5 calls to <function f at 0x...> triggered tf.function retracing.
Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function
repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead
of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has
reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to
https://www.tensorflow.org/guide/function#controlling_retracing ...
```
这条警告本身已经把三类最常见的 retracing 成因列全了,是排查这类"显存缓慢爬升"问题时第一个该确认的线索。

**面试怎么问 + 追问链:**
- **Q:** "训练脚本跑到一半 OOM 了,你的排查顺序是什么?"—— 期望有系统性的顺序,比如先看报错信息本身在哪个 op/shape 上失败(直接给出容量线索)、再查是不是有其他进程按默认贪婪模式占着显存(第1节)、再查 `get_memory_info` 的 `current` 是不是随 step 数持续爬升。
- **追问 1:** "如果 `current` 确实在持续爬升,下一步怎么定位是哪里增长的?"—— 期望提到分阶段打点(forward前后、backward前后)和检查是否存在 retracing(比如 `pretty_printed_concrete_signatures()` 或者留意 retracing 警告),而不是笼统地说"用 `del` 试试"。
- **追问 2(深挖,区分度高):** "TF eager 模式下会不会出现 PyTorch 那种'忘记 `.item()`,计算图一直不释放'的泄漏?"—— 期望答"机制不同,TF eager 下 tensor 引用计数归零就释放,不需要等 backward 主动清空;但 `tf.function` 的 retracing 是 TF 特有的、容易被忽视的显存增长源",能对比出两个框架"泄漏"的根因完全不同,而不是简单套用 PyTorch 的排查经验。
- **追问 3(工程开放题):** "怎么在训练脚本里自动发现这类问题,而不是等 OOM 了才排查?"—— 没有标准答案,合理方向包括"定期打点 `get_memory_info`,曲线近似线性增长就报警"以及"关注 retracing 警告不要当成无害噪音直接忽略"。

**常见坑:** 看到 `ResourceExhaustedError` 就第一反应"调小 batch size"——这只解决了"这一步需要的显存确实超过物理上限"这一种成因,如果真实原因是另一个进程按第1节讲的默认贪婪模式占掉了大半张卡,调小 batch size 只是缓解症状,没有解决"多进程共享 GPU 显存分配策略"这个根本问题;反过来,如果原因是 retracing 导致显存缓慢爬升,调小 batch size 甚至可能完全没用(因为增长量和 batch size 无关,和"调用了多少次不同参数的 trace"有关)。另外要注意 `reset_memory_stats()` 重置的是统计意义上的"峰值追踪起点",不会真的释放任何显存,如果目的是想在两段代码之间腾出显存对比测量,该用的是确保没有 tensor 存活引用(必要时配合 `del` 和 Python 的垃圾回收),而不是指望 `reset_memory_stats()` 顺带做了这件事。

---

## 小结:这一批 6 个知识点解决的问题

| # | 知识点 | 核心结论(均为本机实测) |
|---|------|-----------------------|
| 1 | `set_memory_growth` / 显存增长模式 | 默认贪婪模式下一次极小运算就吃掉约 **13873MiB**(占16384MiB整卡的84.7%);开启增长模式后同样操作只占 **342MiB**;环境变量优先级高于代码里的显式调用,`get_memory_growth()` 不能反映真实生效值 |
| 2 | `mixed_float16` 实践 | compute_dtype=float16、variable_dtype=float32 双轨制;BatchNorm 的 moving_mean/moving_variance 即使在 compute_dtype=float16 下仍强制 float32;完整训练 step 实测提速 **1.19x**,纯矩阵乘法上限约 **1.30x**,不是"数倍"级别 |
| 3 | XLA 性能实践 | 逐元素链式运算融合收益巨大,实测 **~25~27x**;编译本身一次性开销约 0.8~1.4 秒;大矩阵乘法场景收益微乎其微,实测仅 **1.0~1.1x** |
| 4 | `tf.profiler` 简介 | `experimental.start/stop` + `Trace` 现场生成 `.xplane.pb` 文件,用于 TensorBoard 可视化时间线,排查 host/device 时间分布问题 |
| 5 | `tf.function` 图优化(Grappler) | `concrete_function.graph.as_graph_def()` 看到的是 tracing 原始图,看不到 Grappler 优化;用 `tf_optimizer.OptimizeGraph` 现场验证常量折叠确实让 `Mul` 节点消失 |
| 6 | GPU 显存 OOM 排查 | 用硬切显存上限(512MB)可快速复现真实 `ResourceExhaustedError`;`tf.function` retracing 是 TF 特有的隐蔽显存增长源,30次不同参数调用触发30次retrace、增长4.2MB,而稳定参数只trace 1次、增长0.0MB |

下一批:[11-distributed-training-basics.md](11-distributed-training-basics.md) —— 分布式训练基础机制(`MirroredStrategy`、`strategy.scope()` 变量创建拦截机制)。

---

*更新:2026-07-09*
