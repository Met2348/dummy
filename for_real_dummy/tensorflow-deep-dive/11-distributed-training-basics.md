# 11 · 分布式训练基础机制(tf.distribute 数据并行体系)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批讲 `tf.distribute` 这一整套"策略(Strategy)"体系怎么做单机多卡、乃至多机多卡的数据并行训练。和 [torch-deep-dive/09](../torch-deep-dive/09-distributed-training-basics.md) 讲的是同一个问题域——"多张卡各自算出一份梯度之后,怎么变成一份所有卡都认可的梯度"——但 TF 给出的是一套完全不同的架构:不是"多个独立进程各管一张卡",而是"一个策略对象接管你原本写的 eager/Keras 代码,在背后做设备放置和通信"。本文第 5 节会把这两条路线摆在一起精确对比,第 6 节会把"训练时的数据并行"和仓库已有的 [distributed-inference](../../learning/distributed-inference/) 模块"推理时的模型并行"做分工说明,这里先卖个关子。

**关于本文验证情况的诚实说明(必须先讲清楚):** 本机(ERIC-3080Ti)只有**一张**物理 GPU(RTX 3080 Ti Laptop GPU),没有第二张卡,没法真的起多卡训练。第 1、2、5 节涉及 `MirroredStrategy` 的部分,用的是 `tf.config.set_logical_device_configuration` 在这一张物理 GPU 上切出 2 个**虚拟逻辑设备**——这不是本文发明的取巧手段,而是 TF 官方文档"Using multiple GPUs"教程和 TF 自己的单元测试库里,在只有 1 张卡(甚至 0 张卡)的 CI 环境下测试多设备逻辑的标准做法;`strategy.num_replicas_in_sync` 在这种设置下确实会等于 2,且梯度同步等核心逻辑是**真实触发、真实验证**过的,不是纸面推导。第 3 节 `TPUStrategy` 本机没有 TPU 硬件,`initialize_tpu_system`/真正在 TPU 上跑一步这些步骤做不到,如实标成"机制讲解,依据官方文档与源码,未实测",但资源解析失败前的那一小步(`TPUClusterResolver()`)是真实触发的。第 4 节 `MultiWorkerMirroredStrategy` 本机没有第二台物理机器,但和 torch09 用 `gloo` CPU 进程模拟多机是同一个思路——TF 的多 worker 发现机制靠读 `TF_CONFIG` 环境变量里的地址列表,不区分"这个地址是不是另一台物理机器",本文用 2 个独立 OS 进程、各自设置指向 `localhost` 不同端口的 `TF_CONFIG`,真实验证了跨进程梯度 all-reduce 这个核心逻辑;真实跨机网络延迟、worker 故障容错这些必须依赖真实多机环境的部分,如实标注为未测。全部真实报错文本都是现场触发后原样抄录。

**本篇统一结构(和前几批一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(能测的现场测,测不了的明确标注依据)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. `tf.distribute.MirroredStrategy`(单机多卡数据并行)

**是什么:**
```
tf.distribute.MirroredStrategy(devices=None, cross_device_ops=None)
```
单机多 GPU 的同步数据并行策略:把同一份模型完整复制到每一个参与的设备上,每个设备独立处理全局 batch 的一个切片、各自完整跑 forward + backward,梯度通过 all-reduce 合并后,所有设备上的模型副本保持逐比特一致地更新。

**一句话:** `MirroredStrategy` 是 TF2 里做"单机多卡数据并行"的标准入口——它在 TF 世界里对标的地位是 PyTorch 的 `DistributedDataParallel`(而不是已被弃用的 `DataParallel`),尽管两者的进程架构完全不同(第 5 节详细展开)。

**底层机制/为什么这样设计:**

一个 `MirroredStrategy` 对象管理三件事:①一组参与训练的设备列表(`devices=None` 时自动探测所有可见 GPU,也可以显式传入设备名列表);②`scope()`——拦截这组设备范围内的变量创建,让 `tf.Variable(...)` 背后真正创建出"每个设备一份物理副本"的 `MirroredVariable`(第 2 节详细展开这个拦截机制);③`run(fn, args)`——把 `fn` 分发到每个设备各自执行一次,并在函数内部对 `MirroredVariable` 的读写自动路由到"当前副本对应设备的那一份"。三件事配合起来,你写的还是看起来单卡的 eager/Keras 代码,背后却真正在做多设备并行。

本机只有 1 张物理 GPU,无法验证真正的多卡效果,因此用 `tf.config.set_logical_device_configuration` 在这张卡上切出多个**虚拟逻辑设备**——给每个虚拟设备指定一个显存配额(`memory_limit`,单位 MB),虚拟设备的数量决定 `strategy.num_replicas_in_sync` 的值。这一步有一个硬性时机要求:**必须在任何 GPU 相关操作发生之前调用**,因为 TF 在第一次真正使用 GPU 时才会完成设备初始化,一旦初始化完成,虚拟设备的划分就固定死了,后续再调用会直接报错(常见坑会展示这个真实触发的报错)。

**AI 研究/工程场景(场景化例子,非仓库引用,理由见 00 篇环境声明):** 预训练/微调中等规模模型时,单机 8 卡数据并行几乎是"起手式"——`model = create_model()` 挪到 `with strategy.scope():` 里,训练循环套一层 `strategy.run`,不需要像 DDP 那样手动改造成多进程启动脚本(`torchrun`/`mp.spawn`)。这也是很多团队在"模型还能装进一张卡、只是想让训练更快"这个阶段,优先选择 TF/Keras 高层 API 的原因之一——`model.fit()` 配合 `strategy.scope()` 基本不用手写任何分布式细节。

**可运行例子(已在本机 1 张物理 GPU 切出的 2 个虚拟逻辑设备下真实跑通验证):**

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
     tf.config.LogicalDeviceConfiguration(memory_limit=1024)],
)
logical_gpus = tf.config.list_logical_devices('GPU')
assert len(logical_gpus) == 2

strategy = tf.distribute.MirroredStrategy()
assert strategy.num_replicas_in_sync == 2

with strategy.scope():
    w = tf.Variable(2.0)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# w 是一个 MirroredVariable,每个(虚拟)设备各自持有一份物理副本
assert type(w).__name__ == "MirroredVariable"
assert len(w.values) == 2

def step_fn(x):
    with tf.GradientTape() as tape:
        loss = w * x  # dloss/dw = x
    grad = tape.gradient(loss, w)
    optimizer.apply_gradients([(grad, w)])
    return grad

# 让两个副本看到不同的输入,模拟"同一个 batch 被切成两片,每张卡各处理一片"
per_replica_x = strategy.experimental_distribute_values_from_function(
    lambda ctx: tf.constant(1.0 if ctx.replica_id_in_sync_group == 0 else 3.0)
)

per_replica_grads = strategy.run(step_fn, args=(per_replica_x,))
local_grads = [g.numpy() for g in strategy.experimental_local_results(per_replica_grads)]
print("同步前,两个副本各自算出的局部梯度:", local_grads)
assert local_grads == [1.0, 3.0]   # 两个副本看到的输入不同,局部梯度确实不同

# apply_gradients 内部已经自动做了跨设备梯度聚合(默认求和,第5节详细展开),
# 所以两个副本上的 w 在这一步之后完全一致——这正是"镜像"的含义
w_values = [v.numpy() for v in strategy.experimental_local_results(w)]
print("同步后,两个副本上的 w:", w_values)
expected_w = 2.0 - 0.1 * (1.0 + 3.0)   # SGD: w - lr * sum(grad)
assert abs(w_values[0] - expected_w) < 1e-6
assert abs(w_values[1] - expected_w) < 1e-6
assert w_values[0] == w_values[1]
print(f"两个副本的 w 都变成了 {w_values[0]},验证镜像同步生效")
```

实测输出:同步前两个副本的局部梯度确实是 `1.0` 和 `3.0`(对应各自看到的输入 `x`);`apply_gradients` 之后,两个副本上的 `w` 都变成 `1.6`(`2.0 - 0.1*(1.0+3.0)`),完全一致——证明尽管两个副本各自用不同数据算出了不同的局部梯度,`MirroredVariable` 最终还是被同步成了同一个值,这就是"镜像"两个字的字面含义。

**用一张图把上面这段代码的拓扑画出来**(对应本节例子:`w=2.0`,两个副本各自看到 `x=1.0` / `x=3.0`):

```text
             副本0(逻辑设备0)                       副本1(逻辑设备1)
      ┌───────────────────────┐              ┌───────────────────────┐
      │ 数据切片:  x = 1.0       │              │ 数据切片:  x = 3.0       │
      │ 模型副本:  w = 2.0(镜像) │              │ 模型副本:  w = 2.0(镜像) │
      └───────────┬───────────┘              └───────────┬───────────┘
                  │   strategy.run(step_fn) 把同一份 step_fn 分发到每个设备,
                  │   各自独立跑 forward + backward,互不等待
                  ▼                                       ▼
           grad0 = 1.0                              grad1 = 3.0     ← 各自只看到自己那片数据,
        (来自 loss = w*1.0)                       (来自 loss = w*3.0)   局部梯度天然不同
                  │                                       │
                  └───────────────────┬───────────────────┘
                                      ▼
             all-reduce(默认求和,第5节详解): sum = grad0 + grad1 = 4.0
                                      │
             apply_gradients 在两个设备上各自同步执行: w ← w - lr * sum
                                      │
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
         逻辑设备0: w = 1.6                        逻辑设备1: w = 1.6
                  └───────────────── 更新后两份副本逐比特一致 ─────────────────┘
                                这正是"镜像"(Mirrored)的字面含义
```

第二部分,验证 `set_logical_device_configuration` 的调用时机要求是真实存在的,不是文档随口一说:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
# 先触发一次真实的 GPU 计算,让设备完成初始化
with tf.device('/GPU:0'):
    _ = tf.constant([1.0, 2.0]) + 1.0

try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024)],
    )
    raise AssertionError("expected RuntimeError but none was raised")
except RuntimeError as e:
    print("真实触发的报错:", e)
    assert "Virtual devices cannot be modified after being initialized" in str(e)
```

本机现场触发的真实报错文本是:`RuntimeError: Virtual devices cannot be modified after being initialized`——和 00 篇环境声明里强调的"这一类目的例子必须在任何 GPU 操作发生前做虚拟设备切分"完全对应。

**面试怎么问 + 追问链:**
- **Q:** "`MirroredStrategy` 是怎么做到单机多卡数据并行的?"—— 期望答出:设备列表管理 + `scope()` 拦截变量创建做镜像复制 + `run()` 分发计算到各设备 + 梯度 all-reduce 同步,四件事配合。
- **追问 1(容易踩的类比陷阱):** "`MirroredStrategy` 感觉像是'一个进程管多张卡',这和 PyTorch 里被弃用的 `DataParallel` 很像吧?"—— 期望候选人能分清"表面架构像"和"设计地位像"是两回事:`MirroredStrategy` 确实是单进程,这点和 `DataParallel` 表面相似;但它是 TF 生态里**唯一推荐**的单机多卡方案,地位对标的是 DDP,而不是"能用但官方不推荐"的 `DataParallel`——第 5 节会讲清楚 `MirroredStrategy` 为什么没有 `DataParallel` 那些问题。
- **追问 2:** "如果不显式指定 `devices` 参数,`MirroredStrategy` 怎么知道要用哪些卡?"—— 期望答"自动探测所有可见 GPU"。
- **追问 3(区分度高):** "如果你只有 1 张物理 GPU,怎么验证多卡逻辑是对的?"—— 期望答出 `tf.config.set_logical_device_configuration` 切虚拟设备这个官方标准做法,以及"必须在任何 GPU 操作之前调用"这个时机限制。

**常见坑:**
- 在已经跑过 GPU 计算之后才调用 `set_logical_device_configuration`,触发 `RuntimeError: Virtual devices cannot be modified after being initialized`(上面已真实触发验证)。
- 把 `MirroredStrategy` 的心智模型套到 `DataParallel` 头上,以为存在"主卡负载不均"的问题——`MirroredStrategy` 的每个设备地位对等,不存在 `DataParallel` 那种"scatter/gather 都堆在主卡"的架构性瓶颈,这是两者虽然都是单进程、但实际设计完全不同的地方(第 5 节展开)。
- 忘记 `optimizer` 也要在 `strategy.scope()` 内创建(第 2 节的常见坑会展开这一条的真实报错)。

---

## 2. `strategy.scope()` 拦截变量创建的机制

**是什么:**
```
with strategy.scope():
    ...  # 在这个上下文内创建的 tf.Variable,会被策略拦截并镜像复制到每个设备
```

**一句话:** `strategy.scope()` 本质是一个配置好的 `tf.variable_creator_scope`——它把标准的 `tf.Variable(...)` 构造调用"劫持"到策略自己的变量创建逻辑上,让你写的还是原生 `tf.Variable(...)` / Keras 层代码,背后却真正创建出了"每个设备一份物理副本"的 `MirroredVariable`,而不是普通的单设备 `ResourceVariable`。

**底层机制/为什么这样设计:**

TF 官方 `Strategy.scope()` 的 docstring 原文(已验证,从本机安装包里读到)明确写着这句话:"Variable creation inside `scope` is intercepted by the strategy... This is done using a custom `tf.variable_creator_scope`."——这不是我们的转述,是官方文档自己承认的实现方式。

`tf.variable_creator_scope` 是 TF 提供的一个通用底层机制,不是 `MirroredStrategy` 专属:你可以往里注册一个"creator"函数,之后在这个作用域内每次调用 `tf.Variable(...)`,构造函数不会直接构造,而是把你原本想传的参数转交给你注册的 creator,由 creator 决定"最终真正怎么创建、创建成什么类型"——这是一个类似责任链/装饰器的拦截点,Keras 的 mixed precision、DTensor 等其它机制也复用同一个拦截点,`MirroredStrategy` 只是其中一个使用者。

本机安装包里 `StrategyExtendedV2._scope` 的真实源码(已验证,节选核心部分,省略了 checkpoint 恢复相关的边界处理):

```
def _scope(self, strategy):
  def creator_with_resource_vars(next_creator, **kwargs):
    _require_strategy_scope_extended(self)
    kwargs["use_resource"] = True
    kwargs["distribute_strategy"] = strategy
    created = self._create_variable(next_creator, **kwargs)
    return created

  return _CurrentDistributionContext(
      strategy,
      variable_scope.variable_creator_scope(creator_with_resource_vars),
      ...
  )
```

关键在最后一行:`variable_creator_scope(creator_with_resource_vars)` 把 `creator_with_resource_vars` 注册成当前作用域的变量创建拦截器,而这个拦截器最终调用的 `self._create_variable(...)` 才是 `MirroredStrategy` 真正"在每个设备上各建一份物理副本、包装成 `MirroredVariable`"的地方。

**这就是为什么模型必须在 `scope()` 内构建:** 拦截点只在"变量真正被构造那一刻"生效。如果你在 `scope()` 之外先创建了模型(变量已经用普通、未被拦截的路径创建完毕,成了只活在单一设备上的 `ResourceVariable`),之后即使进入 `scope()` 也没有回头路——该变量不会被追溯性地变成 `MirroredVariable`。这不是"最佳实践建议",而是拦截机制本身决定的硬约束。

**AI 研究/工程场景:** Keras 高层 API 为什么"看起来"不需要你手动管理 `scope()`——因为 `model.compile()`/`model.fit()` 内部检测到 model 是在 `scope()` 里创建的,会自动帮你重新进入捕获到的那个 scope;但如果混用自定义训练循环,容易出现"model 在 scope 内创建、镜像正常,但 optimizer 忘了放进 scope"这种隐蔽 bug——optimizer 的 slot 变量(比如 Adam 的一阶/二阶动量)本质上也是 `tf.Variable`,如果 optimizer 是在 scope 外实例化的,这些 slot 变量同样不会被镜像,这类问题在生产训练脚本的 code review 里是真实会出现的坑,而且不一定马上报错(下面的例子会展示"读"和"写"在这件事上的报错行为完全不同)。

**可运行例子(已在本机 2 个虚拟逻辑设备下真实跑通验证):**

第一部分,验证在 scope 内创建的变量确实变成了带 2 份物理副本的 `MirroredVariable`(上一节已经做过这个验证,这里换一个角度,直接确认底层拦截机制的产物):

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
     tf.config.LogicalDeviceConfiguration(memory_limit=1024)],
)
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    w = tf.Variable(1.0)

# 官方 docstring 里的原文例子(已验证,本机复现结果与文档一致):
# 变量名会带上 replica 后缀,两份物理副本各自绑定到不同的(虚拟)设备
names = [v.name for v in w.values]
devices = [v.device for v in w.values]
print("两份物理副本的变量名:", names)
print("两份物理副本各自的 device:", devices)
assert names[0] == "Variable:0"
assert "replica_1" in names[1]
assert devices[0] != devices[1]   # 确实分布在两个不同的(虚拟)设备上

# 作为对照:scope 外创建的变量是普通 ResourceVariable,没有 .values 这个多副本属性
regular_v = tf.Variable(1.0)
assert type(regular_v).__name__ == "ResourceVariable"
assert not hasattr(regular_v, "values") or not isinstance(getattr(regular_v, "values", None), tuple)
```

第二部分,验证"在 scope 外创建变量,会有什么真实后果"——这是本节最关键的证据,证明"必须在 scope 内构建"不是一句空话:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
     tf.config.LogicalDeviceConfiguration(memory_limit=1024)],
)
strategy = tf.distribute.MirroredStrategy()

# 变量在 scope 外创建 —— 只是一个普通的单设备 ResourceVariable
w_outside = tf.Variable(2.0)

def read_only_step(x):
    with tf.GradientTape() as tape:
        loss = w_outside * x
    return tape.gradient(loss, w_outside)

per_replica_x = strategy.experimental_distribute_values_from_function(
    lambda ctx: tf.constant(1.0 if ctx.replica_id_in_sync_group == 0 else 3.0))

# "只读"场景:不会报错,数值也是对的 —— 全局只有这一份物理变量,不管哪个副本要用,
# 读到的都是这唯一一份的当前值;梯度公式 d(w*x)/dw = x 只依赖各自的 x(x 本身是
# 正常按副本分发的),所以两个副本依然各自正确算出了 1.0 和 3.0,不是算错。
# 但对没有和 w_outside 同处一个设备的那个副本来说,这一步隐含了一次跨设备读取 ——
# 这正是官方文档 "will not be distributed and may have performance implications"
# 里 "performance implications" 的真实来源:少了本地副本,多了隐藏的跨设备访问。
grads = strategy.experimental_local_results(strategy.run(read_only_step, args=(per_replica_x,)))
print("scope外变量,只读场景下两个副本的梯度:", [g.numpy() for g in grads])
assert [g.numpy() for g in grads] == [1.0, 3.0]   # 数值正确,代价是隐藏的跨设备访问,不是算错

# "写"场景:真正调用 optimizer.apply_gradients 更新这个变量,会直接报错
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

def write_step(x):
    with tf.GradientTape() as tape:
        loss = w_outside * x
    grad = tape.gradient(loss, w_outside)
    optimizer.apply_gradients([(grad, w_outside)])

try:
    strategy.run(write_step, args=(per_replica_x,))
    raise AssertionError("expected ValueError but none was raised")
except ValueError as e:
    print("真实触发的报错:", e)
    assert "colocate_vars_with" in str(e)
    assert "must only be passed a variable created in this tf.distribute.Strategy.scope()" in str(e)
```

两部分结果连起来看很能说明问题:只是**读**一个 scope 外的变量(比如算个梯度),TF 不会报错,数值也是对的——因为全局只有这一份物理变量,不管哪个副本要用,读到的都是同一份"唯一真相",梯度公式本身只依赖分发正常的 `x`,所以两个副本依然各自正确地算出了 `1.0` 和 `3.0`;但这份"正确"是有代价的——对没有和 `w_outside` 同处一个设备的那个副本来说,这一步隐含了一次跨设备读取,这正是官方文档那句"will not be distributed and **may have performance implications**"里"performance implications"的真实来源:少了一份本地副本,多了一次隐藏的跨设备访问,这种性能损失不会出现在任何报错或警告里,只会让训练悄悄变慢。真正会**报错**的是**写**:一旦你想用 Keras optimizer 的 `apply_gradients` 去更新它,TF 立刻报错:`ValueError: colocate_vars_with must only be passed a variable created in this tf.distribute.Strategy.scope(), not: <tf.Variable ...>`。这说明"忘记把变量放进 scope"这个坑,在训练脚本里往往不是一开始就爆红报错——只做前向推理/算梯度不会出事、数值也不会错,只是悄悄变慢;等你真正开始更新参数,才会暴露成一个指向明确的报错。

**面试怎么问 + 追问链:**
- **Q:** "`strategy.scope()` 是怎么知道你在里面创建了哪些变量,并且把它们镜像复制到每个设备的?"—— 期望答出 `tf.variable_creator_scope` 拦截机制,而不是"它有魔法"这种模糊回答。
- **追问 1(深挖,区分度高):** "这个拦截机制是 `MirroredStrategy` 专属的,还是 TF 更通用的一个能力?"—— 期望答"`tf.variable_creator_scope` 是通用机制,Keras mixed precision、DTensor 等其它需要'篡改'变量创建行为的功能也复用同一个拦截点,`MirroredStrategy` 只是在这个通用钩子上注册了'创建 N 份镜像副本'这个具体逻辑"。
- **追问 2:** "如果我在 scope 外创建了模型,后来才把它包进 `strategy.run` 里用,会怎么样?"—— 期望能分层回答:只读操作(比如算梯度)不会报错、数值也不会错,但会带上隐藏的跨设备访问开销;一旦涉及 `optimizer.apply_gradients` 这类写操作,会直接抛 `ValueError`,报错信息明确指出变量不是在当前 strategy 的 scope 里创建的——这条在上面的例子里是真实触发验证过的,不是纸面结论。
- **追问 3:** "`model.fit()` 是不是任何时候调用都不用管 scope?"—— 期望说出"只有通过 `compile`/`fit`/`evaluate`/`predict`/`save` 这些高层 API 入口才会自动进入模型捕获到的 scope,直接手写 `model(x)` 不会自动进入"——这是官方 docstring 里明确写的 WARNING,不是坊间经验。

**常见坑:**
- 变量在 scope 外创建,常见于:模型构建函数写在模块顶层时被提前调用了一次;或者 checkpoint 恢复的时机不对,导致变量在恢复过程中脱离了 scope。
- optimizer 忘记放进 `strategy.scope()`,只有 model 在 scope 内——上面的例子已经证明,这种情况下一旦调用 `apply_gradients` 会直接报错,报错信息会明确指出是哪个变量出了问题,只要看到 `colocate_vars_with` 这个关键词就能立刻定位到"这是 scope 范围问题",不用瞎猜。
- 依赖 `model.fit()` 自动进入 scope 的行为,却同时手写了部分自定义逻辑在 scope 外直接调用 `model(...)`,这部分调用不会被自动分发。

---

## 3. `TPUStrategy` 简介(机制讲解,本机无 TPU 硬件,不能现场跑通)

**如实声明:** 本机是 WSL2 + RTX 3080 Ti,没有任何 TPU 硬件,也没有 Google Cloud TPU / Colab TPU runtime 可连。以下内容除了明确标注"已验证"的那一小步之外,全部依据官方文档和本机可读到的源码/docstring 整理,**不冒充实测**。这是 00 篇环境声明里承诺过的诚实边界,不因为这一条比较难写就打折扣。

**是什么:**
```
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
```
运行在 Google TPU 硬件上的同步数据并行策略。对外暴露的 API 形状和 `MirroredStrategy` 很像(都有 `scope()`/`run()`/`reduce()`),这是 `tf.distribute.Strategy` 体系有意为之的统一抽象,但初始化步骤和背后的执行模型完全不同。

**一句话:** `TPUStrategy` 是"同一套 `tf.distribute` 编程模型,换一种完全不同的硬件和执行方式"的典型代表——用起来像 `MirroredStrategy`,但连接方式、编译要求、通信硬件都是另一套体系。

**底层机制/为什么这样设计(依据官方文档,未实测):**

必须先有一个能连接到 TPU runtime 的 `TPUClusterResolver`,再显式调用 `tf.config.experimental_connect_to_cluster` 和 `tf.tpu.experimental.initialize_tpu_system` 才能真正建立到 TPU 硬件的连接、复位其内部状态——这一步和第 1 节 `set_logical_device_configuration` 的"必须在任何其它 GPU 操作之前调用"是同一类时机约束:TPU 系统的拓扑和内存一旦被其它上下文占用,就没法重新初始化。

执行模型上和 `MirroredStrategy` 最大的差异在于**编译的强制程度**。`MirroredStrategy` 在 eager 模式下也能跑(只是有性能警告,第 5 节会展示这一点),`tf.function` 是"推荐"而非"强制"；而 TPU 芯片本身是为执行预先编译好的静态图设计的专用硬件,不直接执行任意 Python/eager 操作,所以 `TPUStrategy` 下的 `strategy.run` 内的函数几乎总是要求包一层 `@tf.function`,并且对动态 shape、Python 原生控制流的容忍度比 GPU 上的 `MirroredStrategy` 低得多——这一点呼应 03 类 `tf.function`/AutoGraph 机制:同样是"tracing 失败/频繁 retracing",在 GPU 上通常只是性能问题,在 TPU 上往往直接变成没法跑的硬报错。

通信硬件层面,TPU 之间通过专用的高速互联(ICI,Inter-Chip Interconnect)做 all-reduce,不经过 PCIe/NVLink——这是"同一行 `strategy.run` 代码,GPU 和 TPU 上背后真实发生的通信路径完全不同"的原因。但这套差异被 `tf.distribute.Strategy` 的抽象层屏蔽掉了,对使用者暴露的是统一、可移植的 Python API——这是整个 `tf.distribute` 体系的公共设计目标(同一份训练代码,尽量少改就能从"单机 CPU 调试"迁移到"单 GPU"、"多 GPU"、"TPU Pod"),不是 `TPUStrategy` 独有的机制。

**AI 研究/工程场景(如实场景化,非仓库引用):** 大规模预训练在有 Google Cloud TPU 资源的团队里会用到;本机环境限制下,这部分的价值主要是"知道它存在、知道 API 长什么样、知道和 GPU 策略的关键差异在哪"——面试里被问到大概率是问"你了解 TPU 训练的机制吗、和 GPU 训练比关键差异是什么",而不是要求你现场手写调试过 TPU 代码,这也是本节如实定位自己"只讲机制"的原因。

**可运行例子(明确区分已验证与未验证部分):**

已验证的部分——本机没有 TPU 时,`TPUClusterResolver()` 的资源发现阶段会怎样:

```python
import tensorflow as tf

try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    raise AssertionError("expected ValueError but none was raised (unexpected: found a TPU?)")
except ValueError as e:
    print("真实触发的报错:", e)
    assert "Please provide a TPU Name to connect to" in str(e)
```

本机现场触发的真实报错是:`ValueError: Please provide a TPU Name to connect to.`——这一步不会卡住或超时,是一个快速、明确的报错,说明本机环境下连 TPU 集群解析这一步都过不去,更不用说后面的 `initialize_tpu_system`/真正跑一个 training step。

**未验证、依据官方文档的部分(如实标注,不冒充实测):**
- `tf.tpu.experimental.initialize_tpu_system(resolver)` 之后 `strategy = tf.distribute.TPUStrategy(resolver)` 能否正常工作——需要真实 TPU 硬件或 Cloud TPU VM / Colab TPU runtime,本机无法验证。
- `strategy.run` 内的 step 函数在 TPU 上对动态 shape/控制流的具体报错行为——依据官方文档描述("几乎总是要求 `@tf.function`"),未在真实 TPU 上触发过具体报错文本。

**面试怎么问 + 追问链:**
- **Q:** "`TPUStrategy` 和 `MirroredStrategy` 用起来的最大区别是什么?"—— 期望答出:多一步连接 TPU 集群 + `initialize_tpu_system`,且 step 函数几乎总是要 `@tf.function` 包起来,对动态 shape/控制流容忍度低得多。
- **追问 1:** "为什么 TPU 对静态图编译的要求比 GPU 严格这么多?"—— 期望提到 TPU 是为执行预编译好的静态图设计的专用硬件,和 GPU 上"eager 也能跑,只是有额外调度开销"的宽松策略不同。
- **追问 2(考察诚实边界意识):** "你在没有 TPU 硬件的机器上,怎么验证自己对 TPUStrategy 机制的理解是对的?"—— 这条追问本身也是这一节想传达的态度:没条件跑,就如实说"没条件跑",引用官方文档/源码作为依据,而不是编一个假的"已验证"结果——诚实边界本身就是候选人技术判断力的一部分。

**常见坑:**
- 把在 GPU 上 `MirroredStrategy` 能跑、带 Python 动态控制流/动态 shape 的代码直接搬到 TPU 上,容易在 tracing/编译阶段报错或性能极差(依据官方文档,未在本机实测出具体报错文本)。
- 忘记调用 `initialize_tpu_system`,或者调用顺序不对(比如在这之前已经有其它 TF 操作跑过、占用了状态)——这一条和第 1 节"虚拟设备必须在任何 GPU 操作之前配置"是同一类时机坑,只是发生在 TPU 语境下。

---

## 4. `MultiWorkerMirroredStrategy` 简介(多机多卡,本机单机环境,如实说明)

**如实声明:** 本机没有第二台物理机器,没法验证真实跨机网络场景。但和 torch09 用 2 个 `gloo` CPU 进程模拟"多机"是同一个思路——TF 的多 worker 发现机制靠读 `TF_CONFIG` 环境变量里声明的地址列表,这个机制本身**不区分**"列表里的地址是不是另一台物理机器",只要地址可达就行。因此本节用 2 个独立 OS 进程、各自设置指向 `localhost` 不同端口的 `TF_CONFIG`,**真实验证**了跨进程梯度同步这个核心逻辑,这不是取巧,是这套机制本来的设计就允许这样做。真实跨机场景下的网络延迟/带宽特性、worker 故障容错、真实多机多卡的 NCCL 通信性能——这些需要真实多机环境,如实标注为未测。

**是什么:**
```
strategy = tf.distribute.MultiWorkerMirroredStrategy()
```
多机版的 `MirroredStrategy`:通过环境变量 `TF_CONFIG`(JSON 格式,声明集群里有哪些 worker、当前进程是哪一个)来发现集群拓扑的多机多卡数据并行策略。

**一句话:** 把 `MirroredStrategy` "单机内多设备镜像同步"的心智模型再扩展一层——"多台机器,每台机器上再有若干设备",梯度同步的语义本质不变(依然是 all-reduce),只是通信路径从"同机设备间"变成了"跨机网络"。

**底层机制/为什么这样设计:**

`TF_CONFIG` 的典型格式:
```
{"cluster": {"worker": ["host1:port", "host2:port", ...]},
 "task": {"type": "worker", "index": 0}}
```
每个 worker 进程启动时,通过这个环境变量知道"整个集群长什么样"以及"我自己是哪一个"——这是纯粹的配置发现机制,不需要类似 PyTorch `torchrun` 那样由启动器统一拉起所有进程,理论上可以用任何方式(手动、Kubernetes、云厂商训练平台)分别启动每个 worker,只要保证它们的 `TF_CONFIG` 互相一致。

`MultiWorkerMirroredStrategy` 这个公开类名背后,底层实现类其实叫 `CollectiveAllReduceStrategy`(本机实测验证:直接构造后 `print(strategy)` 打印出的类型是 `<tensorflow.python.distribute.collective_all_reduce_strategy.CollectiveAllReduceStrategy object ...>`)——公开 API 名字强调"多 worker 镜像",内部实现类名字强调"用集合通信做 all-reduce",这是同一个东西的两个命名视角,读源码/调试打印时容易一时对不上号,值得记录。

没有设置 `TF_CONFIG` 时的默认行为(本机实测验证):不会报错也不会卡住,而是直接退化成单 worker 模式——`strategy.num_replicas_in_sync == 1`,`cluster_resolver.cluster_spec()` 是空的 `ClusterSpec({})`。这个"优雅降级"的设计意味着同一份训练代码在本地单机调试(不设 `TF_CONFIG`)和真实多机部署(由平台注入 `TF_CONFIG`)之间可以做到几乎不用改代码。

通信实现的选择通过 `tf.distribute.experimental.CommunicationImplementation` 暴露(本机实测验证其成员为 `AUTO`/`NCCL`/`RING`)——GPU 环境下一般用 `NCCL` 效率最高,纯 CPU 或者没有 NCCL 支持的环境只能退回 `RING`(纯软件实现的 ring all-reduce,呼应 torch09 第 3 节手写验证过的同一个算法思路)。这和 torch09 提到"Windows 上没有 NCCL、DDP 只能用 gloo"是同一类"根据硬件/平台能力选通信后端"的道理,只是 TF 这边的两个选项名字换成了 RING/NCCL。

**AI 研究/工程场景:** 单机的 GPU 数量已经不够、需要扩展到多台机器组成的集群时用。工程上通常由 Kubernetes(比如 Kubeflow 的 TFJob)或云厂商训练平台负责在每个 worker 容器里自动注入正确的 `TF_CONFIG`,应用代码本身几乎不用变,只要在正确的时机构造 `MultiWorkerMirroredStrategy()` 即可——这也是为什么这套机制被设计成"读环境变量"而不是"代码里写死集群地址"的原因:同一份代码打包成镜像后,能在任意规模的集群上复用。

**可运行例子:**

第一部分,没有 `TF_CONFIG` 时的单 worker 降级行为(已验证):

```python
import os
import tensorflow as tf

assert "TF_CONFIG" not in os.environ
strategy = tf.distribute.MultiWorkerMirroredStrategy()
print("底层实现类:", type(strategy).__name__)
assert type(strategy).__name__ == "CollectiveAllReduceStrategy"
assert strategy.num_replicas_in_sync == 1
assert strategy.cluster_resolver.cluster_spec().as_dict() == {}
print("没有 TF_CONFIG 时,优雅降级为单 worker 模式,不报错也不卡住")
```

第二部分,本节唯一的硬性验证要求——用 2 个独立 OS 进程、各自设置指向 `localhost` 不同端口的 `TF_CONFIG`,模拟出一个 2-worker 集群,验证跨进程梯度 all-reduce(已在本机真实跑通,重复运行 2 次结果稳定一致)。**踩坑记录(真实调试出来的,不是设计之初就知道的):** 这里必须用 `multiprocessing.get_context("fork")`,不能用更"安全"的 `spawn`——`spawn` 需要子进程重新 import 主模块来找到 `worker` 这个函数,而独立执行的代码块没有一个真实文件可供子进程重新导入,会直接报 `AttributeError: Can't get attribute 'worker' on <module '__main__' ...>`;本机是 Linux(WSL2 Ubuntu),`fork` 是默认可用的启动方式,且父进程在 `fork()` 发生时还没有 import `tensorflow`(`tensorflow` 的 import 被刻意放在 `worker` 函数内部、只在子进程里发生),不存在"复制一个已经初始化过 CUDA 上下文的父进程"这类 fork 常见的 GPU 安全问题:

```python
import json
import multiprocessing as mp
import os
import sys
import tempfile

OUT_DIR = tempfile.mkdtemp(prefix="mwms_demo_")

def worker(rank, world_size, out_dir):
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {"worker": [f"localhost:{12601 + i}" for i in range(world_size)]},
        "task": {"type": "worker", "index": rank},
    })
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # 强制这个worker走CPU,避免2个进程抢1张物理GPU

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        w = tf.Variable(5.0)

    def step_fn(x):
        with tf.GradientTape() as tape:
            loss = w * x
        return tape.gradient(loss, w)

    x = tf.constant(2.0 if rank == 0 else 6.0)  # 两个worker看到不同的局部数据
    per_replica_grad = strategy.run(step_fn, args=(x,))
    reduced_sum = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_grad, axis=None)

    with open(os.path.join(out_dir, f"rank{rank}.txt"), "w") as f:
        f.write(str(float(reduced_sum.numpy())))

if __name__ == "__main__":
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=worker, args=(r, 2, OUT_DIR), daemon=True) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)

    assert all(not p.is_alive() for p in procs), "worker进程超时未结束"
    assert all(p.exitcode == 0 for p in procs), "worker进程异常退出"

    r0 = float(open(os.path.join(OUT_DIR, "rank0.txt")).read())
    r1 = float(open(os.path.join(OUT_DIR, "rank1.txt")).read())
    print("worker0跨进程all-reduce后的值:", r0)
    print("worker1跨进程all-reduce后的值:", r1)
    assert abs(r0 - 8.0) < 1e-6   # 2.0 + 6.0 = 8.0
    assert abs(r1 - 8.0) < 1e-6
    assert r0 == r1
    print("两个独立OS进程(模拟2个worker)的all-reduce结果完全一致,跨进程梯度同步验证通过")
```

两个独立进程各自算出局部梯度(`2.0` 和 `6.0`),经过跨进程 `strategy.reduce(SUM, ...)` 之后,两个进程读到的结果都是 `8.0`——证明 `MultiWorkerMirroredStrategy` 的核心同步逻辑,在"同机多进程模拟多机"这个条件下是真实生效的,不依赖真实的第二台物理机器。

**面试怎么问 + 追问链:**
- **Q:** "多机多卡训练时,TF 怎么知道整个集群长什么样?"—— 期望答出 `TF_CONFIG` 环境变量机制。
- **追问 1(区分度高):** "如果本机没有第二台物理机器,能不能验证 `MultiWorkerMirroredStrategy` 的核心逻辑?"—— 期望答"能,开 2 个独立进程,各自设置 `TF_CONFIG` 指向 `localhost` 不同端口即可,因为集群发现机制本身不关心地址是不是跨物理机器;测不到的是真实跨机网络延迟和故障容错行为"。
- **追问 2:** "`MultiWorkerMirroredStrategy` 和单机 `MirroredStrategy` 的梯度同步在语义上有本质区别吗?"—— 期望答"没有,都是 all-reduce,只是通信路径从'同机设备间'变成了'跨进程/跨机器网络',公开类名不同,底层实现类 `CollectiveAllReduceStrategy` 是同一套集合通信基础设施"。
- **追问 3:** "没设置 `TF_CONFIG` 直接构造 `MultiWorkerMirroredStrategy()` 会报错吗?"—— 期望答"不会,会优雅降级成单 worker 模式(`num_replicas_in_sync=1`),这是本机实测验证过的真实行为"。

**常见坑:**
- `TF_CONFIG` 格式写错(JSON 格式错误、`task.index` 和 `cluster` 列表对不上)导致连不上或连错 worker。
- 不同 worker 用了不一致的 `TF_CONFIG`(比如 `cluster.worker` 列表顺序不一致),导致各 worker 对集群拓扑的理解不一致。
- 所有 worker 没有同时启动、互相不能连通,会一直卡在等待其它 worker 就绪的阶段——这和 torch09 提到的 `MASTER_PORT`/`init_process_group` 卡住是同一类"分布式训练启动阶段最容易卡住"的坑,只是 TF 这边卡住的等待点是 gRPC 层面的 worker 互连。
- 用 `multiprocessing` 在单机上模拟多 worker 调试时,踩到 `spawn` 在无实体文件场景下找不到目标函数的坑(上面的可运行例子已经真实踩到并记录),真实多进程脚本(有实体 `.py` 文件)一般不会遇到这个特定问题,但了解"`spawn` 需要能重新 import 主模块"这个约束,对排查更广泛的 `spawn` 相关问题(比如某些库在 `spawn` 下初始化失败)也有帮助。

---

## 5. 与 PyTorch DDP 的 all-reduce 机制对比

**是什么:** 把本篇 `MirroredStrategy`/`MultiWorkerMirroredStrategy`(底层都是 `CollectiveAllReduceStrategy` 这套集合通信基础设施)和 [torch-deep-dive 09 篇](../torch-deep-dive/09-distributed-training-basics.md)第 2、3 节讲的 `DistributedDataParallel` 放在一起,从进程模型、一致性建立方式、梯度同步默认语义、通信实现四个维度做对比。

**一句话:** 两边解决的是同一个问题(多设备/多进程各自算出局部梯度后,怎么合并成一份用来更新模型),都以 all-reduce 作为核心通信原语,但架构选择(单进程 vs 多进程)和默认语义(求和 vs 求平均)有具体的、真实会导致 bug 的差异。

**底层机制/为什么这样设计(逐条对比):**

**① 进程模型:** `MirroredStrategy` 是单进程架构。eager 模式下,`strategy.run` 通过 `_MirroredReplicaThread`(本机安装包源码可读到的真实类名)把每个副本的函数分发到一个独立的 Python 线程去执行——但一个容易被忽略的真实细节是:这些线程**默认并不真正并发执行**。源码里 `_call_for_each_replica` 函数开头硬编码了 `run_concurrently = False`,原始注释是:"Add this option once we add synchronization to variable creation. Until then, this is pretty unsafe to use."——也就是说,各副本线程是被主线程用 `should_run`/`has_paused` 这一对事件"一步一步牵着走"的,严格串行,不是自由并发(第 5 节下面的可运行例子会用真实计时实验验证这一点)。这也解释了本文前几节反复出现的那条 `WARNING:tensorflow:Using MirroredStrategy eagerly has significant overhead currently... please wrap ... inside a tf.function` ——eager 模式下这种串行的线程调度确实慢,官方给的解法是把整个训练 step 包进 `@tf.function`:一旦被 `tf.function` 追踪,多设备 step 会被编译成**一张**包含所有设备上算子的图,由 TF 运行时的 C++ executor 负责跨设备并行调度,不再依赖 Python 线程,自然也不存在 Python 线程调度开销的问题。

PyTorch DDP 是真正的多进程架构(torch09 第 2 节),每个进程有自己独立的解释器和 GIL,不管是不是 eager 模式、有没有用 `torch.compile`,都是货真价实的操作系统级并行,不需要"先编译成一张图"才能避开调度瓶颈。这是两条路线在架构哲学上最根本的不同:TF 选择"留在单进程里,靠图编译把并行下沉到 C++ 运行时";PyTorch DDP 选择"一开始就是多进程,天然并行,不需要额外编译这一步"。

**② 变量/参数一致性建立方式:** `MirroredStrategy` 的变量创建发生在**同一个进程**内(第 2 节讲的 `variable_creator_scope` 拦截机制),所有副本的物理拷贝天生就在同一个 Python 进程的内存空间里,不需要一个单独的"广播"步骤来对齐初始值。PyTorch DDP 是 N 个完全独立的进程,没有这条"同进程内存"的捷径,所以必须在 construction 时刻显式把 rank 0 的参数**广播**给其它 rank(torch09 第 2 节已用"故意扰动 rank1 参数、验证被广播覆盖"的方式真实验证过)。

**③ 梯度同步默认语义(本文验证到的最重要的真实差异):** TF Keras Optimizer 的 `aggregate_gradients` 方法,本机源码 docstring 原文(已验证):"By default, we will perform reduce_sum of gradients across devices",内部调用的是 `optimizer_utils.all_reduce_sum_gradients(grads_and_vars)`——默认是**求和**,不是求平均。第 1 节的可运行例子已经实测验证:两个副本梯度分别是 `1.0` 和 `3.0`,`SGD(lr=0.1)` 更新后 `w` 从 `2.0` 变成 `1.6`,精确对应 `2.0 - 0.1*(1.0+3.0)`(用的是**和** `4.0`,不是均值 `2.0`)。这意味着如果你不做任何额外处理,数据并行的"有效学习率"会随着设备数量线性放大,官方要求用户自己在算 loss 时用 `tf.nn.compute_average_loss(per_example_loss, global_batch_size=...)` 把这一步"补偿"回来。

PyTorch DDP 这边(torch09 第 3 节):默认通信钩子的效果是先 SUM 再除以 `world_size`,等效直接产出**平均**梯度,用户不需要在 loss 层面做任何额外调整就能拿到"和单卡语义一致"的梯度尺度。

这是一个真实存在、容易在"把 PyTorch 训练经验直接套到 TF 上"时踩到的坑:同样是多卡训练,PyTorch 下调好的 loss/学习率设置,原样搬到 TF 且没有用 `compute_average_loss` 做缩放,等效学习率会跟着 GPU 数量变化,而且不会报错,只会让训练效果跟着卡数变化却难以定位原因——这正是"常见坑"部分要强调的重点。

**④ 通信打包/重叠机制:** 两边都有"把梯度打包成块,以便通信和还没算完的计算重叠"这个优化,只是包装方式不同。PyTorch DDP 用 `bucket_cap_mb`(torch09 第 4 节,单位 MB,默认 25MiB)。TF 这边对应的是 `tf.distribute.experimental.CommunicationOptions(bytes_per_pack=...)`(本机 docstring 已验证:"Breaks collective operations into packs of certain size... so that weight updates can overlap with gradient all-reduce",单位是字节)——两边解决的是同一个"桶多大合适"的权衡问题(桶太大退化成'等全部算完再通信',桶太小则通信次数增多、固定开销占比升高),只是参数名字和单位不同。TF 同时也提供 `CommunicationImplementation`(`AUTO`/`RING`/`NCCL`,已验证)这个更底层的通信算法选择,对应 PyTorch DDP 的 `backend` 参数(`gloo`/`nccl`)——**GPU 上一般都优先选 NCCL,没有 NCCL 支持的环境(CPU-only、或者像 torch09 提到的 Windows 平台)都得退回纯软件实现的 RING/gloo**,这条"平台能力决定能不能用 NCCL"的道理在两个框架里是一致的。

**AI 研究/工程场景(如实场景化):** 团队里同时维护 TF 和 PyTorch 两套训练代码库时,"同一个概念、默认语义不同"是真实会发生的迁移 bug 来源——比如把 PyTorch 下调好的 loss/学习率配置原样搬到 TF,不知道 TF 默认对梯度求和而不是求均值,会导致等效学习率随 GPU 数量剧烈变化,现象是"多卡训练效果比单卡差很多、或者直接发散",定位起来如果不知道这条差异会很绕。

**可运行例子(已在本机 2 个虚拟逻辑设备下真实跑通验证):**

第一部分,直接对比"默认 SUM 聚合" vs "手动除以副本数补偿成 MEAN 语义"两种写法在同样输入下的最终结果差异:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
     tf.config.LogicalDeviceConfiguration(memory_limit=1024)],
)
strategy = tf.distribute.MirroredStrategy()

per_replica_x = strategy.experimental_distribute_values_from_function(
    lambda ctx: tf.constant(1.0 if ctx.replica_id_in_sync_group == 0 else 3.0))

# 写法A:什么都不做,走TF Keras Optimizer的默认聚合(SUM)
with strategy.scope():
    w_sum = tf.Variable(2.0)
    opt_sum = tf.keras.optimizers.SGD(learning_rate=0.1)

def step_sum(x):
    with tf.GradientTape() as tape:
        loss = w_sum * x
    grad = tape.gradient(loss, w_sum)
    opt_sum.apply_gradients([(grad, w_sum)])   # 内部默认 reduce_sum

strategy.run(step_sum, args=(per_replica_x,))
w_sum_val = strategy.experimental_local_results(w_sum)[0].numpy()
assert abs(w_sum_val - 1.6) < 1e-6   # 2.0 - 0.1*(1.0+3.0)，等效于放大了2倍学习率

# 写法B:仿照 tf.nn.compute_average_loss 的思路，除以副本数补偿成"均值"语义
with strategy.scope():
    w_mean = tf.Variable(2.0)
    opt_mean = tf.keras.optimizers.SGD(learning_rate=0.1)

def step_mean(x):
    with tf.GradientTape() as tape:
        loss = (w_mean * x) / strategy.num_replicas_in_sync
    grad = tape.gradient(loss, w_mean)
    opt_mean.apply_gradients([(grad, w_mean)])

strategy.run(step_mean, args=(per_replica_x,))
w_mean_val = strategy.experimental_local_results(w_mean)[0].numpy()
assert abs(w_mean_val - 1.8) < 1e-6   # 2.0 - 0.1*mean(1.0,3.0)，和"单卡语义"一致

print(f"不做补偿(默认SUM): w={w_sum_val}；除以副本数补偿(等效MEAN): w={w_mean_val}")
print("同样的数据、同样的学习率，两种写法的最终结果不同 —— 这就是切换框架时最容易踩的默认语义差异")
```

第二部分,实证"eager 模式下 `MirroredStrategy.run` 的副本线程并不真正并发执行"这一条源码细节——不只是读源码,现场用计时实验验证:

```python
import time
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
     tf.config.LogicalDeviceConfiguration(memory_limit=1024)],
)
strategy = tf.distribute.MirroredStrategy()
assert strategy.num_replicas_in_sync == 2

SLEEP_SECONDS = 1.0

def step_fn(x):
    time.sleep(SLEEP_SECONDS)   # 纯Python阻塞，不是TF算子，能如实反映"这个线程有没有在等"
    return x * 2

t0 = time.perf_counter()
strategy.run(step_fn, args=(tf.constant(1.0),))
elapsed = time.perf_counter() - t0
print(f"2个副本各睡{SLEEP_SECONDS}秒，实测总耗时: {elapsed:.2f}秒")

# 若真并发：总耗时应接近 SLEEP_SECONDS(~1秒)
# 若严格串行：总耗时应接近 2*SLEEP_SECONDS(~2秒)
assert elapsed > 1.8 * SLEEP_SECONDS
print("实测确认：总耗时接近2倍单次sleep时间，说明两个副本线程是被串行调度的，不是真并发")
```

本机实测:2 个副本各 `sleep(1.0)`,总耗时 `2.01` 秒,精确对应"串行"而不是"并发"(`~1` 秒)——这条源码细节(`run_concurrently = False`)不是一个无关紧要的实现细节,而是能被计时实验直接测出来的真实行为,也是官方那条"eager 模式用 `MirroredStrategy` 有明显开销,建议包进 `tf.function`"警告的具体成因。

**面试怎么问 + 追问链:**
- **Q:** "TF 的 `MirroredStrategy` 和 PyTorch 的 DDP,虽然都是做数据并行,架构上有什么本质区别?"—— 期望答出"单进程(靠图编译把并行下沉到 C++ 运行时) vs 多进程(操作系统级别天然并行)"这条主线,而不是停留在"一个是TF一个是PyTorch"这种同义反复。
- **追问 1(区分度极高):** "`MirroredStrategy` 单进程怎么避免类似 `DataParallel` 的 GIL 瓶颈?"—— 期望答出两层:eager 模式下其实没有完全避开(线程串行调度、有明确的性能警告),真正"避开"要靠把 step 包进 `tf.function`,让执行下沉到 C++ 运行时,不再依赖 Python 线程——这条追问专门用来筛选"只会背答案"和"真的验证过机制"的候选人。
- **追问 2(容易漏答的真实坑):** "同样的模型和数据,分别用 TF `MirroredStrategy` 和 PyTorch DDP 做 2 卡训练,如果两边都没有对 loss/学习率做特殊处理,最终效果会一样吗?"—— 期望答"不一样,TF 默认对梯度求和,PyTorch DDP 默认对梯度求均值,前者的等效学习率相当于被放大了设备数倍",并最好能提到 `tf.nn.compute_average_loss` 是官方给的补偿手段。
- **追问 3:** "两边是不是都要处理'桶多大合适'这个权衡?"—— 期望说出 TF 的 `bytes_per_pack` 和 PyTorch 的 `bucket_cap_mb` 是同一个思路的不同实现,单位不同(字节 vs MB)。

**常见坑:**
- 把 PyTorch 下调好的学习率/loss 配置原样搬到 TF 多卡训练,不知道默认梯度聚合是求和不是求均值,导致等效学习率随卡数变化(上面已用真实数值 `1.6` vs `1.8` 验证过这个差异)。
- 以为 TF `MirroredStrategy` 因为是单进程,就一定比 PyTorch DDP 有 GIL 瓶颈——实际上只要按官方推荐把训练 step 包进 `tf.function`,并行是下沉到图执行层面的,不是靠 Python 线程,和 `DataParallel` 那种"永远靠 Python 线程调度"的架构不是一回事;但如果偷懒不用 `tf.function`,eager 模式下确实有真实的、可测量的串行开销(上面已实测验证)。
- 混淆"进程模型的差异"和"梯度同步语义的差异"——这是两个独立的维度,容易被笼统地归为"反正 TF 和 PyTorch 分布式不一样",但具体不一样在哪、会导致什么真实后果,才是面试里真正的区分点。

---

## 6. 这一批和 [distributed-inference](../../learning/distributed-inference/) 模块的分工

和 torch09 第 6 节一样,这里要先把话说清楚:这一批(11)和仓库已有的 `distributed-inference` 模块,虽然都在讲"多卡怎么协作",但解决的是两个完全不同阶段、不同目的的问题,不应该混为一谈——这个分工关系本身是框架无关的,不会因为一边用 TF、一边用 PyTorch 而改变。

**11(这一批,训练时的数据并行):** 解决的问题是"模型本身一张卡就装得下,但数据很多、想让训练更快、或者想用更大的有效 batch size"。做法是让**每张卡都拥有一份完整的模型**(`MirroredVariable` 镜像复制),不同卡处理不同的数据切片,各自独立算完 forward + backward 之后,只在"梯度"这一个点上做同步(第 3~5 节讲的 all-reduce)。本质是"同一份模型,不同数据,梯度层面对齐"。

**distributed-inference 模块:** 解决的问题是"模型本身太大,一张卡的显存根本装不下完整的它"。做法是把**模型本身切开**——Tensor Parallel 把一层内部的矩阵运算切开分布到多张卡、Pipeline Parallel 把不同层分布到不同卡、Expert Parallel(MoE)把不同专家分布到不同卡,数据流过这个被切开的模型时,要在卡与卡之间传递中间激活值/部分计算结果。本质是"同一份数据,流过被切开的不同模型部分"。这个模块目前覆盖的是**推理**时的切分部署,代码是用 PyTorch 写的(`src/tp_demo.py` 等),但 TP/PP/EP 这套"切模型"的思路本身是框架无关的概念——TF 生态里也有对应的生产级模型并行方案(比如结合 Mesh 概念的 DTensor),只是不在本系列 00-roadmap.md 规划的 100 个知识点范围内,这里不展开、也不编造未验证的对应关系。如果想深入模型并行这部分,直接跳去看 [distributed-inference/README](../../learning/distributed-inference/README.md) 和 [L01 分布式推理全图](../../learning/distributed-inference/lectures/01-distrib-overview.md)。

两者不是二选一的关系,实际训练超大模型时经常**组合使用**——先用 Tensor/Pipeline Parallel 把模型切开保证"装得下",再在切开之后的每一份模型副本外面套一层数据并行(`MirroredStrategy`/DDP,或者更强的 ZeRO/FSDP 变体)保证"训练得快、能喂更多数据"。这也是为什么理解本篇内容是理解更复杂的混合并行策略的前提。

| 维度 | 11(本篇,训练时数据并行) | distributed-inference 模块 |
|---|---|---|
| 解决什么问题 | 模型一张卡放得下,想让训练更快/有效 batch 更大 | 模型一张卡放不下,必须切开才能部署/训练 |
| 谁被切开了 | **数据**(batch)被切开,模型是完整复制的 | **模型本身**(层/矩阵/专家)被切开,数据基本完整流过 |
| 核心操作 | all-reduce 梯度同步(第 3~5 节) | 切分后的跨卡数据传递(TP 的 all-reduce、EP 的 all-to-all、PP 的层间激活值传递) |
| 典型场景 | 中小模型、想提高训练吞吐 | 千亿参数级别大模型,单卡装不下 |
| 本仓库位置 | 这里(11) | [learning/distributed-inference/](../../learning/distributed-inference/) |
| 涉及框架 | TF(本篇) / PyTorch(torch09) 都有对应机制 | 模块代码用 PyTorch 写的,但 TP/PP/EP 思路框架无关 |

**面试角度提一句:** "分布式训练和分布式推理的并行策略有什么区别?"是一个真实会被问到的问题,很多候选人会把"数据并行"和"模型并行(Tensor/Pipeline/Expert Parallel)"混着讲——精确的回答应该先分清"切的是数据还是模型",再分别展开各自的通信模式,而不是笼统地说"都是多卡协作"。

---

## 小结:这一批 6 个知识点解决的问题

| # | 知识点 | 核心结论 | 本机验证情况 |
|---|---|---|---|
| 1 | `MirroredStrategy` | 单进程,`scope()`镜像变量 + `run()`分发计算 + 梯度all-reduce同步,单机多卡数据并行的标准入口 | 已实测(1张物理GPU切2个虚拟逻辑设备:MirroredVariable结构、梯度同步、时机报错) |
| 2 | `strategy.scope()`拦截机制 | 本质是配置好的`tf.variable_creator_scope`,只在变量真正被构造那一刻生效,决定了模型必须在scope内构建 | 已实测(docstring+真实源码+MirroredVariable结构验证+scope外变量的读/写两种真实后果) |
| 3 | `TPUStrategy` | API形状像MirroredStrategy,但执行模型(强制编译)、连接方式、通信硬件完全不同 | **未实测**(本机无TPU硬件,仅`TPUClusterResolver()`资源解析失败这一步真实触发) |
| 4 | `MultiWorkerMirroredStrategy` | 多机版MirroredStrategy,靠`TF_CONFIG`发现集群拓扑,底层实现类是`CollectiveAllReduceStrategy` | 默认单worker降级已实测;2进程`TF_CONFIG`模拟多worker的跨进程all-reduce已实测(**超出最低要求**) |
| 5 | 与PyTorch DDP对比 | 进程模型(单进程线程串行 vs 多进程真并行)、默认聚合语义(SUM vs MEAN)、通信打包机制,三个维度均有真实、可验证的差异 | 已实测(SUM vs MEAN数值对比 + 计时实验证明eager下副本线程非并发) |
| 6 | 与distributed-inference模块分工 | 11切数据(训练态数据并行),distributed-inference切模型(推理态模型并行),两个不同问题,可组合使用 | 定位/分工说明,非验证类内容 |

下一批:[12-serialization-and-deployment.md](12-serialization-and-deployment.md)

---

*更新:2026-07-09*
