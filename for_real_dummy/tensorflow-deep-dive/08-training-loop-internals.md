# 08 · fit() 内核与自定义训练循环深度机制(Training Loop Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)
> **这是全系列和 PyTorch 心智模型差距最大的一批。** [torch-deep-dive/07-training-loop-internals.md](../torch-deep-dive/07-training-loop-internals.md) 开篇就点明过:PyTorch 没有 `fit()` 这种全自动训练循环封装,永远是手写 eager 循环。反过来说,这正是本篇要讲的东西——TensorFlow/Keras 在"训练循环要不要自动化"这件事上同时给了三条路,而且都是一等公民,不是谁将就谁:
>
> 1. **`model.fit()` 全自动**——只管 `compile()` + `fit()`,数据怎么取批次、梯度怎么算、进度条怎么打、checkpoint 什么时候存,全部交给框架内置的一套 `train_step` 逻辑。
> 2. **覆写 `Model.train_step()`**——保留 `fit()` 的全部基础设施(callbacks、进度条、`tf.distribute` 分布式支持、`History` 记录、`steps_per_execution` 性能优化……),只替换"每一步具体算什么"这一小块逻辑。**这条路径在 PyTorch 生态里没有官方对应物**——第三方的 PyTorch Lightning 用 `LightningModule.training_step()` + `Trainer` 某种程度上是在复刻同一个设计,但那是一整个独立的第三方框架,不是 PyTorch 内置能力;Keras 把这条路直接做进了核心库。
> 3. **`GradientTape` 手写循环**——彻底抛开 `compile()`/`fit()`,自己写 `for epoch: for batch: ...`,这条路径和 PyTorch 的心智模型几乎一模一样。
>
> 这三条路不是三选一的互斥替代品,而是同一套底层机制(`GradientTape` + 优化器 + 损失/metrics)包了几层厚度不同的"自动化外壳"——路径 1 外壳最厚,路径 3 完全不裹。本篇第 1-3 节会依次拆开这三层,并反复回过头讲清楚它们内部到底共享了哪些代码、边界画在哪里、什么时候该选哪一条。第 4-9 节再逐个讲透支撑路径 1/2 的具体基础设施:callback 机制、metrics 状态协议、混合精度、`compile()`/`fit()`的两阶段设计、验证集机制。

**关于"AI 研究/工程场景"这一段的诚实声明:** 见 [00-roadmap.md](00-roadmap.md) 环境声明——仓库里没有真实的 TensorFlow/Keras 代码可引用,本篇场景段落是根据真实训练/部署中会遇到的具体问题重构的场景化例子,不是仓库引用。

**验证方法论:** 本文所有代码已在 WSL2(Ubuntu 24.04)`~/tf-venv` 环境下实际跑通验证(TensorFlow 2.21.0,GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU,`TF_USE_LEGACY_KERAS=1` 锁定经典 `tf.keras`,详见 00 篇环境声明)。所有实测数字、报错文本、内部属性名都是现场跑出来后原样记录,不是转述文档。

**本篇统一结构(与 00 篇模板一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(现场验证,不转述)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. `model.fit()` 内部实际发生了什么——拆开这层最厚的自动化外壳

**是什么:**
```
history = model.fit(x, y, batch_size=32, epochs=10, callbacks=[...], validation_data=(...))
# 返回一个 History 对象,history.history 是 {'loss': [...], 'accuracy': [...], ...} 这样的逐epoch记录
```
*(以上是签名示意,`model`/`x`/`y` 代表已构造好的模型和数据,不是可执行代码——完整可运行版本见本节"可运行例子")*

**一句话:** `fit()` 不是一个不可拆解的黑箱函数,而是一段可以在源码里逐行读到的**编排逻辑**(orchestration):把你的数据包装成迭代器,把 `train_step`(第 2 节详讲)包进一个 `tf.function`,然后跑一个双层 `for epoch: for step:` 循环,循环的每个缝隙里插入约定好的 callback 钩子调用——`ModelCheckpoint`/`EarlyStopping`(第 5 节)、进度条、`History` 记录,都是靠这些钩子才能在恰当的时机介入,不是和 `fit()` 耦合在一起的特殊逻辑。

**底层机制/为什么这样设计:**

直接读 `tf_keras/src/engine/training.py` 里 `Model.fit()` 的主循环(已验证,结构原样摘录,省略部分异常处理/分布式细节):

```
callbacks.on_train_begin()
for epoch, iterator in data_handler.enumerate_epochs():
    self.reset_metrics()                          # 第6节:每个epoch开始,所有metrics清零
    callbacks.on_epoch_begin(epoch)
    for step in data_handler.steps():
        callbacks.on_train_batch_begin(step)
        logs = self.train_function(iterator)       # 真正的一步训练,内部调用 train_step(第2节)
        callbacks.on_train_batch_end(step, logs)
    if validation_data and self._should_eval(epoch, validation_freq):   # 第9节
        val_logs = self.evaluate(x=val_x, y=val_y, ..., callbacks=callbacks)
        epoch_logs.update({"val_" + k: v for k, v in val_logs.items()})
    callbacks.on_epoch_end(epoch, epoch_logs)
callbacks.on_train_end(logs=training_logs)
return self.history
```
*(以上是 `tf_keras` 源码结构摘录,`self`/`callbacks`/`data_handler` 等是源码内部变量,不是可独立执行的代码——本节"可运行例子"从外部现场验证这段逻辑的实际行为)*

这段编排逻辑背后有三个关键机制,分别对应 fit() 一次调用背后的"数据 → 计算 → 反馈"三个环节:

**1)数据取批次——`DataHandler`。** 不管你传进去的 `x`/`y` 是 numpy 数组、`tf.data.Dataset` 还是 Python generator,`fit()` 第一步都会用 `data_adapter.get_data_handler(...)` 把它统一包装成内部的 `DataHandler`,自动推断 `steps_per_epoch`(比如 100 条数据、`batch_size=32` 会推出 4 步,最后一步只有 4 条,这是第 9 类"内存与性能"和 09 类"tf.data"要展开的内容,这里不重复)、按 `epochs` 数生成对应次数的迭代器。这一步之后,不管原始输入是什么形式,后面的循环代码看到的都是统一的迭代器接口。

**2)计算——`train_function`,`train_step` 被 `tf.function` 包了一层。** 第一次调用 `fit()` 时,`self.train_function = self.make_train_function()` 会把 `train_step`(第 2 节详讲的那个方法)包进一个 `tf.function(..., reduce_retracing=True)`。这意味着:虽然你在 Python 里写的是逐行 eager 代码,`fit()` 循环里真正反复调用的,是一张**已经被追踪、编译好的计算图**——这是 03 类 `tf.function`/AutoGraph 机制在训练循环这个具体场景里的落地,`fit()` 的性能优势(比手写 `GradientTape` 循环快,尤其是小 batch、多 step 的场景)根源就在这里。

**3)反馈——`CallbackList` 在循环缝隙里插钩子。** `callbacks` 参数(不管你传不传)会被统一包装成 `CallbackList`,并且**自动追加两个你没显式要求过的 callback**:`History`(负责 `model.fit()` 最终返回值 `history.history` 的记录)和 `ProgbarLogger`(负责你在终端看到的那个进度条,`verbose=0` 时会跳过)。`fit()` 主循环在 `on_train_begin`/`on_epoch_begin`/`on_train_batch_begin`/`on_train_batch_end`/`on_epoch_end`/`on_train_end` 这几个精确的缝隙调用 `callbacks.xxx()`——第 4/5 节讲的自定义 callback 和内置 `ModelCheckpoint`/`EarlyStopping`,靠的都是这几个钩子,没有任何"内置 callback 专属"的特殊通道。

**现场验证"train_step 被包进 tf.function"这件事本身,顺带验证一个容易踩的坑——Python 副作用不是每步都执行:**

```python
import tensorflow as tf
import numpy as np

py_call_count = {"n": 0}

class CountingModel(tf.keras.Model):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.real_step_counter = tf.Variable(0, dtype=tf.int64, trainable=False)

    def train_step(self, data):
        py_call_count["n"] += 1                       # 纯Python副作用:只在trace时执行
        self.real_step_counter.assign_add(1)           # 图内op:每次真实执行都会跑
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

x = np.random.rand(12, 4).astype("float32")
y = np.random.randint(0, 2, size=(12,)).astype("float32")
inputs = tf.keras.Input(shape=(4,))
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(inputs)
model = CountingModel(inputs, outputs)
model.compile(optimizer="sgd", loss="binary_crossentropy")

model.fit(x, y, batch_size=4, epochs=4, verbose=0)   # 3 step/epoch * 4 epoch = 12 次真实执行

print("Python计数器(trace次数):", py_call_count["n"])
print("tf.Variable计数器(真实执行次数):", int(model.real_step_counter.numpy()))
print("官方API确认的tracing次数:", model.train_tf_function.experimental_get_tracing_count())

assert int(model.real_step_counter.numpy()) == 12        # 真实执行了12次,一次没少
assert py_call_count["n"] == 2                             # 但Python副作用只触发了2次!
assert model.train_tf_function.experimental_get_tracing_count() == 2   # 官方API交叉验证,不是巧合
```

`reduce_retracing=True`(`make_train_function` 显式传的参数)是"2 次"这个数字背后的真正原因——它不是"trace 一次就够了,剩下是某种缓存巧合",而是**故意**先按精确输入签名 trace 一次,再额外多花一次 trace 生成一份"泛化过的"签名(把具体的 batch 维度这类容易变化的维度放宽),用来吸收后续调用里那些形状相近但不完全相同的输入,换取"不用每次形状一变就重新编译"的稳定性。用一批数量不能被 `batch_size` 整除的数据(最后一个 batch 形状会变小)可以直接验证这个"泛化 trace"确实在生效:

```python
import tensorflow as tf
import numpy as np

py_call_count = {"n": 0}

class CountingModel(tf.keras.Model):
    def train_step(self, data):
        py_call_count["n"] += 1
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

# 13条数据,batch_size=4 -> 批大小序列是 4,4,4,1(最后一批"余数"更小,形状和前面不一样)
x = np.random.rand(13, 4).astype("float32")
y = np.random.randint(0, 2, size=(13,)).astype("float32")
inputs = tf.keras.Input(shape=(4,))
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(inputs)
model = CountingModel(inputs, outputs)
model.compile(optimizer="sgd", loss="binary_crossentropy")
model.fit(x, y, batch_size=4, epochs=1, verbose=0)

print("trace次数(含形状会变化的余批):", py_call_count["n"])
print("官方API:", model.train_tf_function.experimental_get_tracing_count())
# 如果 reduce_retracing 不起作用,形状不同的第4个余批理应触发第3次retrace;
# 实测仍然是2次——第2次trace生成的"泛化签名"直接吃下了这个不同形状的余批,没有触发第3次
assert py_call_count["n"] == 2
assert model.train_tf_function.experimental_get_tracing_count() == 2
```

这个发现对写自定义 `train_step`(第 2 节)有直接的工程含义:**不要在 `train_step` 里用纯 Python 状态(`print`、`list.append`、普通字典计数器)去观察"这一步发生了什么"**——这类代码只在 trace 时执行,不代表真实的每步行为,`fit()` 真正跑起来之后,能反映每步真实情况的只有图内 op:`tf.Variable`、`tf.print`,或者返回值字典里会被 callback 看到的 `logs`。

**AI 研究/工程场景:** 这套编排逻辑的最大价值在于"和其他基础设施无缝集成"——`step_function` 内部实际是通过 `self.distribute_strategy.run(run_step, args=(data,))` 调用 `train_step` 的(11 类分布式训练要展开的内容,这里先点出结论),也就是说只要走的是 `fit()`(或第 2 节的 `train_step` 覆写,同样复用这条编排逻辑),换成 `tf.distribute.MirroredStrategy` 多卡训练往往只需要在 `strategy.scope()` 里构建模型,`fit()` 调用本身一行不用改。反过来,如果选择第 3 节的 `GradientTape` 手写循环,分布式这部分工作要自己接管(`strategy.run()`、跨副本梯度聚合都要手写)——这是三条路线取舍上一个具体、非抽象的差异点,也是"要不要用 fit()"这个决策在真实项目里最常被提到的理由之一。

**面试怎么问 + 追问链:**
- **Q:** "`model.fit()` 内部到底做了什么?不看文档,你能讲一下一次 `fit()` 调用背后大致的执行顺序吗?" —— 期望答出三个环节:`DataHandler` 统一数据接口并推断 `steps_per_epoch`;`train_step` 被包进 `tf.function`(`make_train_function`)反复执行;`CallbackList` 在 `on_train_begin/on_epoch_begin/on_train_batch_begin/on_train_batch_end/on_epoch_end/on_train_end` 这几个精确缝隙插入回调,`History`/进度条都是自动追加的 callback,不是特殊逻辑。
- **追问 1(核心):** "为什么 `fit()` 要把 `train_step` 包进 `tf.function`,而不是直接 eager 跑?" —— 期望提到"减少 Python 逐行解释和逐算子调度的开销,拿到接近静态图的执行效率",这是 03 类 `tf.function`/AutoGraph 机制在训练循环里的具体落地,不需要重新展开 AutoGraph 的完整机制。
- **深挖追问(区分度很高):** "如果我在自定义 `train_step` 里用一个普通 Python 列表 `.append()` 记录每一步的 loss,拿来画训练曲线,这样做对不对?" —— **陷阱题**,期望候选人能想到"不对,`train_step` 会被 trace 进 `tf.function`,`.append()` 这类 Python 副作用只在 trace 那几次真正执行,不代表真实的每步都在跑;要么用 `tf.Variable`/`tf.TensorArray` 之类的图内状态,要么依赖 `train_step` 返回值字典被 callback 在每个真实 step 读到的 `logs`"。
- **追问 2(工程向):** "`steps_per_execution` 这个 `compile()` 参数是干什么的?" —— 期望能答出"让一次 `train_function` 调用内部用 `tf.range` 循环跑多个 step,减少 Python 侧和 `on_train_batch_*` 回调在每一步之间来回切换的开销,用更粗的回调粒度换执行效率",不要求记住具体源码分支,只要理解这是"把多步计算打包进同一次图执行"的优化手段。

**常见坑:**
- 在自定义 `train_step` 里用纯 Python 状态做计数/日志,期望"每步都被调用",实际只在 trace 那几次(本节实测是 2 次)生效——这是本节验证过的最容易踩、还不报错的坑。
- 误以为 `fit()` 比手写 `GradientTape` 循环慢、为了"追求性能"改手写循环却忘记加 `@tf.function`——`fit()` 内部本来就是编译过的图,手写循环如果不显式包 `tf.function`,反而会更慢,这是后面第 3 节会再次强调的点。
- 以为 `verbose=0` 会关掉 `fit()` 内部除进度条外的所有开销——实际上 `History`、内部的 `_train_counter`、`reset_metrics()` 这些机制照常运行,`verbose` 只控制 `ProgbarLogger` 这一个 callback 是否被加进 `CallbackList`。

---

## 2. `Model.train_step()` 自定义覆写——"全自动"和"手写"之间的第三条路(★本类目核心)

**是什么:**
```
class CustomModel(tf.keras.Model):
    def train_step(self, data):
        # 覆写这一个方法,决定"一步具体怎么算"
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])   # compile()/fit()用法完全不变
model.fit(x, y, epochs=10, callbacks=[...])                    # callbacks/进度条/分布式支持全部照常生效
```
*(以上是签名示意,`inputs`/`outputs`/`x`/`y` 代表已构造好的张量和数据,不是可执行代码——完整可运行版本见本节"可运行例子")*

**一句话:** 只替换 `fit()` 编排逻辑里"计算"这一个环节(第 1 节讲的三个环节之一),"数据取批次"和"回调触发"两个环节原封不动地继续由 `fit()` 接管——这是"既要 `fit()` 的基础设施、又要自定义每一步具体算什么"这个真实工程诉求存在的**专门设计**,不是文档里顺带提一句的边角特性。

**底层机制/为什么这样设计:**

这个设计要解决的矛盾,在第 1 节梳理过的 `fit()` 三环节里看得非常清楚:`compile()` 能配置的东西——一个 `optimizer`、一个(或一组)`loss`、一组 `metrics`——本质上是在假设"每一步的计算形状都是同一种模式:forward 一次、算一个 loss、backward 一次、更新一次"。但真实场景里,不少训练逻辑天生不长这个形状,典型例子是 GAN:一个 batch 里要交替做两次独立的 forward+backward,用两个不同的 `optimizer` 分别更新判别器和生成器的参数——这种逻辑没法通过给 `compile()` 传参数表达出来,因为它不是"换一个 loss 函数"就能解决的问题,而是训练**步骤结构本身**变了。

如果为了这类需求放弃 `fit()`,退回第 3 节的手写 `GradientTape` 循环,又要把 `fit()` 免费提供的一大堆基础设施重新写一遍:`callbacks` 在正确时机触发、进度条、`History` 记录、`tf.distribute` 分布式适配、`steps_per_execution` 性能优化(第 1 节)……这些和"这一步具体怎么算"完全正交,重写一遍纯粹是浪费而且容易出 bug。

Keras 的解法是把 `train_step` 设计成一个**可替换的缝**:`make_train_function`(第 1 节读过的源码)包裹的从来都只是 `self.train_step`——一个通过 Python 正常的方法解析(MRO)得到的引用,`make_train_function` 自己完全不关心这个方法内部具体做了什么。子类覆写 `train_step` 之后,`fit()` 循环、`tf.function` 包装、`tf.distribute.Strategy.run()` 调度这些第 1 节讲过的机制**原封不动地继续包裹这个新方法**,因为它们的作用对象本来就是"随便哪个可调用的 `self.train_step`",不是写死的默认实现。基类的默认实现本身(已验证,原样摘录)其实很短:

```
def train_step(self, data):
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compute_loss(x, y, y_pred, sample_weight)   # 内部调用 self.compiled_loss(第8节)
    self._validate_target_and_loss(y, loss)
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)   # gradient+apply_gradients的合并写法
    return self.compute_metrics(x, y, y_pred, sample_weight)     # 内部调用 self.compiled_metrics.update_state
```
*(以上是 `tf_keras` 源码摘录,`data_adapter` 等是源码内部依赖,不是独立可执行代码)*

值得注意 `self.optimizer.minimize(loss, self.trainable_variables, tape=tape)` 这一行——它是 `tape.gradient(...)` + `optimizer.apply_gradients(...)` 两步的合并写法(新版 `tf.keras.optimizers.Optimizer` 提供的便捷方法,接收外部创建的 `tape`),覆写 `train_step` 时两种写法都合法,后面例子为了展示"中间要插入自定义逻辑"用的是拆开的两步写法,便于在 `tape.gradient()` 之后、`apply_gradients()` 之前插自己的处理。

**这条路径在 PyTorch 生态里没有官方对应物**——原生 `torch.nn.Module` 没有"内置 fit() + 可覆写的单步逻辑"这种设计,永远是第 3 节风格的手写循环(呼应 [torch-deep-dive/07-training-loop-internals.md](../torch-deep-dive/07-training-loop-internals.md) 的定位)。第三方的 **PyTorch Lightning** 用 `LightningModule.training_step()` + `Trainer` 复刻了几乎一模一样的设计理念(保留 `Trainer` 的 callbacks/进度条/分布式,只覆写单步逻辑)——这从侧面说明"训练循环基础设施"和"每一步具体算什么"解耦是一个独立于框架的通用工程问题,Keras 把答案直接做进了核心库,PyTorch 生态是靠社区库另起一套框架来补上,这个对比本身就是一个很好的面试谈资。

**可运行例子(验证覆写后 fit() 的三块基础设施原样生效,以及一个基类做不到的自定义指标):**

```python
import tensorflow as tf
import numpy as np

x = np.random.rand(12, 4).astype("float32")
y = np.random.randint(0, 2, size=(12,)).astype("float32")

class CustomModel(tf.keras.Model):
    """覆写train_step,加一个compile()参数化接口完全没有对应位置的自定义指标:
    每步梯度的全局范数(global gradient norm)。"""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.real_step_counter = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.grad_norm_tracker = tf.keras.metrics.Mean(name="grad_norm")

    def train_step(self, data):
        self.real_step_counter.assign_add(1)
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.grad_norm_tracker.update_state(tf.linalg.global_norm(grads))   # 自定义逻辑插在这里
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results["grad_norm"] = self.grad_norm_tracker.result()
        return results

def make_model():
    inputs = tf.keras.Input(shape=(4,))
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(inputs)
    return CustomModel(inputs, outputs)

model = make_model()
model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
hist = model.fit(x, y, batch_size=4, epochs=2, verbose=0)

assert int(model.real_step_counter.numpy()) == 6         # 3 step/epoch * 2 epoch,真实执行次数精确
assert "grad_norm" in hist.history                        # 自定义指标自动出现在History里,不用手动接线
print("history keys:", list(hist.history.keys()))
print("grad_norm per epoch:", hist.history["grad_norm"])

# 惊喜之一(已验证,不是想当然):grad_norm_tracker 没有做任何"注册"动作,
# 仅仅是 __init__ 里的一次属性赋值(self.grad_norm_tracker = ...),
# 就已经被 model.metrics 自动收录——Keras 对 Metric 类型属性做了和 tf.Variable
# 属性同样的自动追踪(Layer.__setattr__),不需要额外手写 `metrics` 这个 property
assert "grad_norm" in [m.name for m in model.metrics]

# 覆写train_step之后,ModelCheckpoint/自定义callback(第4/5节)照常在正确时机被调用,完全不用改
import os
ckpt_path = "/tmp/custom_train_step_ckpt.weights.h5"
class OrderLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.events = []
    def on_epoch_begin(self, epoch, logs=None): self.events.append(f"epoch_begin({epoch})")
    def on_epoch_end(self, epoch, logs=None): self.events.append(f"epoch_end({epoch})")

model2 = make_model()
model2.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_weights_only=True, save_freq="epoch")
logger = OrderLogger()
model2.fit(x, y, batch_size=4, epochs=2, verbose=0, callbacks=[ckpt_cb, logger])
assert os.path.exists(ckpt_path)
assert logger.events == ["epoch_begin(0)", "epoch_end(0)", "epoch_begin(1)", "epoch_end(1)"]
print("ModelCheckpoint + 自定义callback在覆写train_step后依然按fit()的编排时机正常触发")
```

**AI 研究/工程场景:** 官方"Customizing what happens in fit"指南把 GAN 训练作为覆写 `train_step` 的典型场景——判别器和生成器分别用独立的 `optimizer`,一个 batch 里各自 forward/backward 一次,这种"一步内部有多套梯度更新"的结构,只有深入到"一步具体怎么算"这一层才能表达;知识蒸馏(knowledge distillation)是另一个常见场景——一步之内要同时跑 teacher 和 student 两个模型的 forward,用两者输出算蒸馏损失,`compile()` 的单一 `loss` 参数模型完全不适配这种"需要访问额外模型"的逻辑;上面例子里"记录每步梯度范数"这类训练诊断信息(排查梯度爆炸/消失时很有用)也是同一类需求——都是"确实需要精确控制一步内部发生什么,但仍然想要 `fit()` 的 callback/分布式/进度条这套基础设施"的真实场景,不是为了炫技而覆写。

**面试怎么问 + 追问链:**
- **Q:** "如果我想要 `fit()` 的 callback、进度条这些基础设施,又需要自定义训练每一步的具体计算逻辑(比如 GAN 需要交替更新两个 optimizer),你会怎么做?" —— 期望候选人不掉进"要么全自动 fit()、要么整个重写循环"的二选一陷阱,直接说出"覆写 `Model.train_step()`"。
- **追问 1(核心,区分度高):** "为什么不直接在 `compile()` 时传一个自定义 loss 函数就够了,为什么还需要覆写整个 `train_step`?" —— 期望能讲清楚"`compile()` 能参数化的是'一个 loss + 一组 metrics + 一个 optimizer'这种单一、同步的训练步骤形状;像 GAN 那样一步内部有多套 forward/backward、多个 optimizer 的逻辑,不是换一个 loss 函数能表达的,必须深入到'一步具体怎么算'这一层"。
- **追问 2(工程向,考察对 PyTorch 生态的理解深度):** "PyTorch 有没有类似的设计?" —— 期望提到"核心 PyTorch 没有,因为 PyTorch 本身没有 `fit()` 这层封装;第三方的 PyTorch Lightning 用 `LightningModule.training_step()` + `Trainer` 复刻了几乎一样的理念——这说明'训练循环基础设施'和'每步具体算什么'解耦是通用工程问题,不是 Keras 独有的怪癖,只是 Keras 把它做进了核心库、PyTorch 生态靠社区库来补"。
- **追问 3(源码理解,埋下第 8 节伏笔):** "如果不覆写 `train_step`,`loss` 和 `metrics` 具体是通过什么对象被实际计算和累积的?" —— 期望提到 `self.compiled_loss`/`self.compiled_metrics`(内部是 `LossesContainer`/`MetricsContainer`),这两个对象在 `compile()` 时创建、在 `train_step` 里被调用,是第 8 节"`compile()`和`fit()`两阶段设计"要展开的内容。
- **深挖追问:** "覆写 `train_step` 时,自定义的 `tf.keras.metrics.Metric` 对象要怎么接入,才能让 `fit()` 每个 epoch 自动帮你重置它?" —— 期望答"只要把它赋值成 `self.xxx` 这样的属性,Keras 的属性追踪机制会自动把它收进 `model.metrics`,`fit()` 每个 epoch 开始时统一调用的 `reset_metrics()` 就会自动重置它,不需要额外手写一个 `metrics` property 去手动汇总"——这是本节验证过、比想象中更省心的一点。

**常见坑:**
- **忘记在覆写的 `train_step` 里调用 `self.compiled_metrics.update_state(...)`**——这是本节实测验证过的真实坑,而且比"metric 显示成 0"更隐蔽:`compiled_metrics` 内部的 `MetricsContainer` 是**懒构建**的,只有真正被 `update_state()` 调用过一次才会"生成"对应的 metric 对象并计入 `model.metrics`;一次都没调用过,`compile()` 里传的 `metrics=["accuracy"]` 不会以任何形式出现在 `History` 里——`hist.history` 里压根没有 `"accuracy"` 这个 key,不是"存在但等于0",训练本身(loss 正常下降、参数正常更新)看起来完全正常,很容易让人误以为是"这个 metric 就是刚好一直是0"而不是"这个 metric从来没被算过"。
- 混淆 `self.optimizer.minimize(loss, vars, tape=tape)`(便捷合并写法)和手动 `tape.gradient()` + `apply_gradients()` 两步写法——两者等价,但如果要在梯度算出来之后、真正更新参数之前插入自定义逻辑(裁剪、加噪声、记录范数),必须用拆开的两步写法,`minimize()` 这个便捷方法没有给中间过程留口子。
- 以为覆写 `train_step` 之后还需要手动处理 `tf.distribute`/`tf.function` 包装或者手动触发 callback——完全不需要,这些都是 `make_train_function` 在更外层负责的,覆写者只需要关心"这一步怎么算",这正是第 1 节反复强调的"缝"的位置。

---

## 3. `GradientTape` 手写训练循环——和 PyTorch 心智模型最接近的写法

**是什么:**
```
for epoch in range(epochs):
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc_metric.update_state(y_batch, y_pred)
    # epoch结束:手动读取、手动重置metric,没有History,没有callbacks
    print(f"epoch {epoch}: acc={float(train_acc_metric.result())}")
    train_acc_metric.reset_state()
```
*(以上是签名示意,`epochs`/`dataset`/`model`/`loss_fn`/`optimizer`/`train_acc_metric` 代表已构造好的对象,不是可执行代码——完整可运行版本见本节"可运行例子")*

**一句话:** 彻底不用 `compile()`/`fit()`,`model`、`optimizer`、`loss_fn`、`metric` 都是独立的普通 Python 对象,`for epoch: for batch:` 自己写、梯度自己算自己应用、metric 自己更新自己读自己重置——这条路径和 [torch-deep-dive/07-training-loop-internals.md](../torch-deep-dive/07-training-loop-internals.md) 里手写 `zero_grad`/`backward`/`step` 的 PyTorch 循环,在心智模型上几乎是同一件事的两种语法。

**底层机制/为什么这样设计:**

这条路径不是"阉割版的 fit()",而是第 1/2 节讲的那套编排逻辑**唯一的地基**——`DataHandler`、`CallbackList`、`train_function` 的 `tf.function` 包装,全部是 `fit()` 在这套地基上叠的外壳;地基本身(`GradientTape` + 优化器 `apply_gradients` + metrics 的 `update_state`/`result`/`reset_state`)从第 0 层就能独立工作,不需要 `compile()` 参与。这一点可以直接验证:一个从未调用过 `compile()` 的模型,`model.optimizer` 属性是 `None`(这个属性要等 `compile()` 才会被赋值,第 8 节详细展开),但这完全不妨碍你自己拿一个独立的 `tf.keras.optimizers.SGD` 实例来更新它的变量——`compile()`/`optimizer` 属性和"能不能训练"这两件事,在这条路径上是彻底解耦的。

代价也同样直接:`fit()` 免费提供的一切,这里都要手写。`callbacks` 不会被自动调用(`ModelCheckpoint`/`EarlyStopping` 这类第 5 节讲的内置 callback 完全用不上,除非你自己在恰当位置手动触发它们的钩子方法);没有 `History` 对象,想要训练曲线要自己在每个 epoch 末尾攒列表;metrics 的 `reset_state()` 不会像 `fit()` 里那样每个 epoch 自动调用一次(第 6 节详讲的协议),忘记手动调用,跨 epoch 的统计口径就会串。性能上还有一个容易被忽略的点:`fit()` 内部的 `train_function` 是被 `tf.function` 包过的编译图(第 1 节验证过),而这里最直接、最像 PyTorch 的写法是纯 eager——如果不自己再手动加一层 `@tf.function`,速度上会有真实、明显的差距。

**可运行例子(验证"compile()对这条路径不是必需品"+ 完整跑通,并且对照验证 fit() 反过来必须依赖 compile()):**

```python
import tensorflow as tf
import numpy as np

x = np.random.rand(12, 4).astype("float32")
y = np.random.randint(0, 2, size=(12, 1)).astype("float32")
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)

model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="sigmoid")])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
loss_fn = tf.keras.losses.BinaryCrossentropy()
train_acc = tf.keras.metrics.BinaryAccuracy()

# 从未调用过compile() —— model.optimizer属性是None,但完全不影响下面手写训练
assert model.optimizer is None

for epoch in range(2):
    train_acc.reset_state()                          # 手动重置,fit()里这一步是自动的(第1/6节)
    for step, (xb, yb) in enumerate(dataset):
        with tf.GradientTape() as tape:
            pred = model(xb, training=True)
            loss = loss_fn(yb, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc.update_state(yb, pred)
    print(f"epoch {epoch}: loss={float(loss):.4f} acc={float(train_acc.result()):.4f}")
# 实测: epoch 0: loss=0.7832 acc=0.4167   epoch 1: loss=0.7951 acc=0.5833
# 没有History对象、没有callbacks —— 这两行print就是你唯一能拿到的训练记录,全靠手动管理

# 反过来验证:fit()这条路径离不开compile() —— 直接fit()会报错
model_uncompiled = tf.keras.Sequential([tf.keras.layers.Dense(1)])
try:
    model_uncompiled.fit(x, y, epochs=1, verbose=0)
    raise SystemExit("expected RuntimeError but fit() succeeded")
except RuntimeError as e:
    print("RuntimeError:", str(e))
    assert "must compile" in str(e).lower()
# 实测报错文本: "You must compile your model before training/testing.
#              Use `model.compile(optimizer, loss)`."
```

**性能对照(现场实测,量化"手写eager循环不加`@tf.function`会慢多少"):**

```python
import tensorflow as tf
import numpy as np
import time

N_STEPS, BATCH, DIM = 200, 64, 256
x = np.random.rand(N_STEPS * BATCH, DIM).astype("float32")
y = np.random.randint(0, 2, size=(N_STEPS * BATCH, 1)).astype("float32")
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH)

def make_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

loss_fn = tf.keras.losses.BinaryCrossentropy()

# 1) 纯eager手写循环,不加tf.function
model_eager = make_model()
opt_eager = tf.keras.optimizers.SGD(0.01)
xb0, yb0 = next(iter(dataset))
with tf.GradientTape() as tape:                      # warmup,建立变量,不计入计时
    p0 = model_eager(xb0, training=True); l0 = loss_fn(yb0, p0)
g0 = tape.gradient(l0, model_eager.trainable_variables)
opt_eager.apply_gradients(zip(g0, model_eager.trainable_variables))

t0 = time.perf_counter()
for xb, yb in dataset:
    with tf.GradientTape() as tape:
        pred = model_eager(xb, training=True)
        loss = loss_fn(yb, pred)
    grads = tape.gradient(loss, model_eager.trainable_variables)
    opt_eager.apply_gradients(zip(grads, model_eager.trainable_variables))
t_eager = time.perf_counter() - t0

# 2) 同样的手写循环,只是把单步逻辑包进 @tf.function(呼应03类机制,这里先用结论)
model_tf = make_model()
opt_tf = tf.keras.optimizers.SGD(0.01)

@tf.function
def train_step_fn(xb, yb):
    with tf.GradientTape() as tape:
        pred = model_tf(xb, training=True)
        loss = loss_fn(yb, pred)
    grads = tape.gradient(loss, model_tf.trainable_variables)
    opt_tf.apply_gradients(zip(grads, model_tf.trainable_variables))
    return loss

_ = train_step_fn(xb0, yb0)   # warmup,触发trace,不计入计时

t0 = time.perf_counter()
for xb, yb in dataset:
    train_step_fn(xb, yb)
t_tf_function = time.perf_counter() - t0

print(f"纯eager手写循环: {t_eager:.3f}s / {N_STEPS}步")
print(f"@tf.function包装后: {t_tf_function:.3f}s / {N_STEPS}步")
print(f"倍率: {t_eager/t_tf_function:.2f}x")
# 实测: 纯eager=2.599s  tf.function包装=0.715s  倍率=3.64x(方向稳定,具体倍数因环境而异)
assert t_eager > t_tf_function * 2   # 差距是真实的数量级差异,不是测量噪声
```

**AI 研究/工程场景:** 强化学习里 agent 每个 step 要不要多展开一次 rollout、要不要额外做一次 TD 更新,往往由运行时的 reward/环境反馈决定,这类"每步内部逻辑本身可能不同"的场景,比 `train_step` 覆写(前提是"这一步的计算图结构固定")更适合直接手写;元学习(meta-learning)的内外双层循环——内层几步梯度更新、外层再对内层结果求一次梯度(MAML 这类算法)——涉及嵌套 `GradientTape` 和跨越多个"step"的梯度依赖,`train_step` 只覆盖单个 step 的抽象,直接手写循环反而更直接;调试一个全新模型结构时,直接在循环中间打断点看中间张量数值,也是这条路径相比另外两条路径最大的即时反馈优势(这也是 PyTorch 整个生态的默认体验)。

**面试怎么问 + 追问链:**
- **Q:** "什么情况下你会选择完全抛开 `fit()`,自己写 `GradientTape` 循环,而不是用 `train_step` 覆写?" —— 期望能说出"当训练逻辑本身不是'每个 batch 跑一次固定结构的 step'能表达的时候"——比如需要运行时决定循环次数、需要嵌套 tape 算高阶梯度跨越多个 step、需要在训练中间灵活插入非训练逻辑(在线评估、动态调整数据管线)。
- **追问 1(核心):** "这条路径下,`compile()` 还需要调用吗?" —— **陷阱题**,期望候选人能明确说"不需要,`compile()` 是 `fit()`/`train_step` 覆写这两条路径的机制,和这里的 `optimizer.apply_gradients()`、`GradientTape.gradient()` 完全无关,`model.optimizer` 这个属性此时甚至是 `None`"。
- **追问 2(工程向,容易被面试官用来筛选'知其然不知其所以然'的候选人):** "有人抱怨'TensorFlow 比 PyTorch 慢',结果一看是手写的 `GradientTape` 循环没加 `@tf.function`,你怎么解释这个现象、怎么建议他修?" —— 期望结合本节实测数字(纯 eager 比 `@tf.function` 包装慢 3 倍以上)讲清楚"差距来自有没有用编译图执行,不是框架本身的速度差异;`fit()` 默认是编译过的,手写循环要拿到同等速度必须自己在单步函数上加 `@tf.function`"。
- **追问 3:** "这种写法下,`EarlyStopping`/`ModelCheckpoint` 这些内置 callback 还能用吗?" —— 期望答"不能直接用,它们是围绕 `CallbackList` 在 `fit()` 编排逻辑里的钩子设计的(第 4/5 节);要在手写循环里得到类似效果,必须自己在 epoch 末尾写等价的判断逻辑(比较 loss 有没有改善、手动调用 `model.save_weights()`),或者干脆换回 `train_step` 覆写以保留这层基础设施"。

**常见坑:**
- 忘记每个 epoch 手动调用 metric 的 `reset_state()`——`fit()` 路径下这一步是自动的(第 1 节源码 `self.reset_metrics()`),手写循环里如果漏掉,`result()` 返回的会是跨多个 epoch 的累积统计,不是当前 epoch 的真实表现,而且不会有任何报错提示这个语义已经串了(第 6 节会更完整地展开这个协议本身)。
- 手写循环里不加 `@tf.function` 却拿它跟 `fit()`/`train_step` 覆写比速度,得出"TensorFlow 手写就是慢"的错误结论——本节实测的 3 倍以上差距,根源是"有没有用编译图",不是两条路径本身谁更快。
- 想在 `GradientTape` 手写循环里用 `model.trainable_variables`,却在模型第一次真正被调用(第一次 forward)之前就去访问它——`Sequential`/子类化模型的权重是**延迟构建**的(04 类会展开 `build()` 机制),第一次拿到非空 `trainable_variables` 列表必须在至少一次 `model(x)` 调用之后,循环第一步里"先 forward 后拿 `trainable_variables`"这个顺序不能颠倒。

---

## 4. Keras Callback 自定义写法——`fit()` 编排逻辑对外暴露的正式接口

**是什么:**
```
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # self.model 由 CallbackList 在训练开始前自动注入,不是构造函数参数
        print(f"epoch {epoch} 结束, logs={logs}")
```
*(以上是签名示意,省略了 `import tensorflow as tf`——完整可运行版本见本节"可运行例子")*

**一句话:** 第 1 节讲的 `CallbackList` 在 `on_train_begin/on_epoch_begin/on_train_batch_begin/on_train_batch_end/on_epoch_end/on_train_end`(以及 `evaluate()`/`predict()` 各自对应的 `on_test_*`/`on_predict_*`)这些精确缝隙调用钩子方法,自定义 Callback 就是继承 `tf.keras.callbacks.Callback` 基类、挑你关心的钩子方法覆写——基类对所有钩子都提供了空实现,不覆写的钩子等于什么都不做。

**底层机制/为什么这样设计:**

`self.model` 这个属性是理解自定义 callback 机制的关键,而且它的注入方式和很多人的第一直觉不一样:**不是构造函数参数**。`fit()` 在 `callbacks = callbacks_module.CallbackList(callbacks, ..., model=self, ...)` 这一步(第 1 节读过的源码)把当前模型实例传给 `CallbackList`,`CallbackList` 再统一调用每个 callback 的 `set_model()` 方法把 `self.model` 设置好。这意味着:你的自定义 callback 类可以脱离任何具体模型先定义好、先实例化,`self.model` 要等真正塞进某次 `fit()` 调用之后才存在——已验证:构造后立即访问 `cb.model` 是 `None`,`fit()` 跑完之后才变成真正的模型实例引用,而且是同一个 Python 对象(`is` 恒等),不是拷贝。

第二个容易被忽略的机制:**`logs` 是所有同一钩子的 callback 共享的同一个字典对象,不是各自的独立副本**。`CallbackList.on_epoch_end` 内部就是一个 `for callback in self.callbacks: callback.on_epoch_end(epoch, logs)` 循环,`logs` 只 `_process_logs` 处理一次,之后原样传给列表里的每一个 callback——如果排在前面的 callback 在自己的 `on_epoch_end` 里往 `logs` 写了新 key 或者改了已有 key 的值,排在后面的 callback 会看到**修改之后**的版本。这不是一个边缘细节:**`callbacks=[...]` 列表里的顺序是有实际语义的**,第 5 节验证 `EarlyStopping` 时会直接用到这个机制。

**可运行例子:**

```python
import tensorflow as tf
import numpy as np

x = np.random.rand(12, 4).astype("float32")
y = np.random.randint(0, 2, size=(12,)).astype("float32")

class LRLogger(tf.keras.callbacks.Callback):
    """记录每个epoch末尾的学习率——一个ModelCheckpoint/EarlyStopping都没有直接提供的自定义诊断。"""
    def __init__(self):
        super().__init__()
        self.lr_per_epoch = []

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate    # self.model 由框架自动注入,不是构造参数
        self.lr_per_epoch.append(float(lr.numpy()) if hasattr(lr, "numpy") else lr)

model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05), loss="binary_crossentropy")

cb = LRLogger()
assert cb.model is None                              # fit()调用之前,self.model还不存在
model.fit(x, y, batch_size=4, epochs=3, verbose=0, callbacks=[cb])
assert cb.model is model                              # fit()之后:同一个模型对象,不是拷贝
print("lr_per_epoch:", cb.lr_per_epoch)
# 实测: [0.05000000074505806, 0.05000000074505806, 0.05000000074505806]  (没配调度器,lr保持不变)
assert len(cb.lr_per_epoch) == 3

# 自定义callback也可以像内置EarlyStopping一样,通过 self.model.stop_training 中止训练
class StopAtEpoch(tf.keras.callbacks.Callback):
    def __init__(self, stop_epoch):
        super().__init__()
        self.stop_epoch = stop_epoch
    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.stop_epoch:
            self.model.stop_training = True           # fit()主循环第1节验证过:检查这个标志决定是否break

model2 = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="sigmoid")])
model2.compile(optimizer="sgd", loss="binary_crossentropy")
hist = model2.fit(x, y, batch_size=4, epochs=10, verbose=0, callbacks=[StopAtEpoch(2)])
print("请求10个epoch,实际跑了:", len(hist.history["loss"]))
assert len(hist.history["loss"]) == 3   # epoch 0,1,2跑完,在epoch 2的on_epoch_end里设了stop_training
```

**AI 研究/工程场景:** 训练过程中把每个 epoch 的关键指标同步推送到自建的实验跟踪系统(不用 TensorBoard 而是公司内部的监控平台)是自定义 callback 最常见的落地场景——`on_epoch_end` 里拿到 `logs` 字典直接转发即可,不需要碰训练循环本身;学习率没有现成 `LearningRateSchedule` 覆盖的自定义调整逻辑(比如"验证集连续 3 个 epoch 不提升就手动腰斩学习率"这种和 `EarlyStopping` 判断逻辑相似但动作不同的策略)通常也是继承 `Callback` 而不是改 `train_step`,因为这类逻辑天然是"epoch 级别的决策",不涉及单步计算,用 callback 表达更清晰、还能和其他 callback 自由组合。

**面试怎么问 + 追问链:**
- **Q:** "怎么写一个自定义 Keras callback,在训练过程中做一些 `fit()` 本身不提供的事情(比如推送指标到自己的监控系统)?" —— 期望说出"继承 `tf.keras.callbacks.Callback`,覆写关心的钩子方法(`on_epoch_end` 最常用),通过 `logs` 参数拿到当前的指标,通过 `self.model` 拿到模型引用"。
- **追问 1(容易问倒):** "`self.model` 是在 `__init__` 里赋值的吗?" —— **陷阱题**,期望答"不是,`self.model` 是 `fit()` 内部创建 `CallbackList` 时通过 `set_model()` 自动注入的,callback 实例化的时候还不知道自己会被用在哪个模型上,`__init__` 阶段访问 `self.model` 会是 `None`"。
- **追问 2(核心,考察对共享状态的理解):** "如果我同时传了两个 callback,都在 `on_epoch_end` 里处理 `logs`,它们之间会互相影响吗?" —— 期望能答"会,`CallbackList` 内部对同一个钩子,给列表里所有 callback 传的是**同一个** `logs` 字典对象,不是各自的副本;排在前面的 callback 修改 `logs` 的内容,排在后面的 callback 能看到修改后的结果——所以 `callbacks=[...]` 列表的顺序有实际语义,不是随便排的"。
- **追问 3(工程向):** "自定义 callback 想要提前结束训练,应该怎么做?" —— 期望提到 `self.model.stop_training = True`,并且知道这个标志是被 `fit()` 主循环在每个 batch/epoch 结束后检查的(第 1 节源码里的 `if self.stop_training: break`),`EarlyStopping` 内部用的也是同一个机制,不是什么特殊权限。

**常见坑:**
- 在 `__init__` 里试图访问 `self.model` 做初始化逻辑——这个阶段 `self.model` 恒为 `None`,任何需要模型引用的逻辑都必须放进 `on_train_begin` 或更晚的钩子里。
- 把关心的逻辑写进 `on_batch_end` 却忘记这是 `on_train_batch_end` 的旧别名,在需要区分训练/验证/预测阶段的场景下容易挂错钩子——`on_epoch_end`/`on_train_batch_end` 只在训练(`fit()`)时触发,`evaluate()`/`predict()` 走的是完全独立的 `on_test_*`/`on_predict_*` 钩子。
- 依赖多个 callback 之间通过修改 `logs` 字典"隐式通信",却没有显式控制 `callbacks=[...]` 列表的顺序——这在小项目里可能凑巧跑对,一旦 callback 数量变多或者顺序被后来者不小心调整,极难排查"为什么这个 callback 看到的指标不对"。

---

## 5. 内置 callbacks 机制——`ModelCheckpoint`/`EarlyStopping` 不是特殊权限,是第 4 节机制的两个具体实例

**是什么:**
```
tf.keras.callbacks.ModelCheckpoint(filepath="ckpt.weights.h5", save_weights_only=True,
                                     monitor="val_loss", save_best_only=True)
tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
```
*(以上是签名示意,省略了 `import tensorflow as tf`——完整可运行版本见本节"可运行例子")*

**一句话:** 这两个官方最常用的内置 callback,源码上就是继承 `tf.keras.callbacks.Callback`、覆写 `on_epoch_end`(以及 `ModelCheckpoint` 在 `save_freq` 是整数时还会用到的 `on_train_batch_end`)——没有任何绕开第 4 节机制的"内部特权通道",理解了第 4 节,这两个内置 callback 的行为就是可以从第一性原理推出来的,不需要死记硬背。

**底层机制/为什么这样设计:**

`EarlyStopping` 的核心逻辑本质上是一个状态机,挂在 `on_epoch_end` 上:每个 epoch 结束时读 `logs[monitor]`(默认 `monitor` 是 `"val_loss"`,没有验证集时退化成 `"loss"`),和历史最佳值比较——变好就刷新最佳值、`wait` 计数器清零;没变好,`wait` 计数器加一;`wait` 达到 `patience` 就把 `self.model.stop_training` 置 `True`(第 4 节验证过的同一个标志、同一个机制,`EarlyStopping` 没有走任何自定义 callback 拿不到的特殊路径)。`restore_best_weights=True` 时,`EarlyStopping` 还会在每次刷新"最佳值"的同时用 `get_weights()` 缓存一份权重快照,训练真正停止时再用 `set_weights()` 覆盖回去——这也是普通 Python 对象操作,不是什么框架黑魔法。

`ModelCheckpoint` 同理:`save_freq="epoch"`(默认)时挂在 `on_epoch_end` 上,每个 epoch 末尾检查是否要保存(`save_best_only=True` 时同样是读 `logs[monitor]` 和历史最佳值比较,逻辑和 `EarlyStopping` 的"变好判断"是同一套比较逻辑,只是动作从"计数+可能停止"换成"保存权重文件");`save_freq` 传整数时则挂在 `on_train_batch_end` 上,按训练 batch 数(不是 epoch 数)触发。

**可运行例子(用第 4 节验证过的"logs 是共享字典"机制,人为构造一个单调变差的 `val_loss`,确定性地触发 `EarlyStopping`):**

```python
import tensorflow as tf
import numpy as np
import os

x = np.random.rand(12, 4).astype("float32")
y = np.random.randint(0, 2, size=(12,)).astype("float32")

class BadValLoss(tf.keras.callbacks.Callback):
    """排在EarlyStopping前面,往共享的logs字典里注入一个每个epoch都变差的val_loss,
    让EarlyStopping的触发时机变得完全确定、可断言,不依赖真实训练的随机收敛过程。"""
    def on_epoch_end(self, epoch, logs=None):
        logs["val_loss"] = 1.0 + epoch * 0.1   # epoch 0: 1.0(当前最佳) epoch 1: 1.1(变差) epoch 2: 1.2(再变差)

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer="sgd", loss="binary_crossentropy")
# BadValLoss必须排在es前面 —— 靠第4节验证过的"同一个logs对象按顺序传递"机制,es才能读到被改过的值
hist = model.fit(x, y, batch_size=4, epochs=20, verbose=0,
                  validation_data=(x, y), callbacks=[BadValLoss(), es])

print("请求20个epoch,实际跑了:", len(hist.history["loss"]))
print("es.stopped_epoch:", es.stopped_epoch)
# 实测: 实际跑了3个epoch,stopped_epoch=2
# (epoch0: val_loss=1.0是目前最佳,wait=0; epoch1: 1.1变差,wait=1; epoch2: 1.2继续变差,wait=2=patience,停止)
assert len(hist.history["loss"]) == 3
assert es.stopped_epoch == 2

# ModelCheckpoint —— 同样是on_epoch_end驱动,验证文件确实被写出
ckpt_path = "/tmp/mc_demo.weights.h5"
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)
mc = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_weights_only=True,
                                          monitor="loss", save_best_only=True, verbose=0)
model2 = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="sigmoid")])
model2.compile(optimizer="sgd", loss="binary_crossentropy")
model2.fit(x, y, batch_size=4, epochs=3, verbose=0, callbacks=[mc])
assert os.path.exists(ckpt_path)
print("ModelCheckpoint写出的文件确实存在:", os.path.exists(ckpt_path))
```

**AI 研究/工程场景:** 长时间训练(几小时到几天量级)的任务,`ModelCheckpoint` 几乎是标配,不只是为了保存最终结果,更是为了应对训练中途机器重启、抢占式实例被回收这类真实的基础设施故障——`save_freq` 设成比较小的整数(而不是默认的每个 epoch)在单个 epoch 本身就要跑很久的大模型训练里更常见;`EarlyStopping` 配合 `restore_best_weights=True` 是防止"训练了很久但模型其实在中间某个 epoch 就已经过拟合、后面都在退步"这种情况的标准做法,`patience` 的取值本质上是"愿意为了确认'真的不会再变好了'multiply 等待几个 epoch"和"不愿意浪费太多算力在明显已经过拟合的模型上"之间的权衡。

**面试怎么问 + 追问链:**
- **Q:** "`EarlyStopping` 是怎么知道'该停止训练了'的?它和你自己写一个 callback 判断逻辑,有什么本质区别吗?" —— 期望答"没有本质区别,`EarlyStopping` 就是一个提前写好的 `Callback` 子类,同样是 `on_epoch_end` 读 `logs[monitor]`、和历史最佳值比较、`wait` 计数器过 `patience` 就设置 `self.model.stop_training = True`——和第 4 节自定义 `StopAtEpoch` 例子用的是完全相同的机制"。
- **追问 1(核心,考察是否真的理解"变好"是怎么判断的):** "`EarlyStopping` 怎么知道该看的指标是'越小越好'还是'越大越好'?" —— 期望知道有一个 `mode` 参数(`'min'`/`'max'`/`'auto'`),`'auto'` 时会根据 `monitor` 的名字猜(名字里带 `acc`/`accuracy` 等猜成 `max`,否则默认 `min`),这是实际使用中容易忽略、用错自定义 metric 名字时会踩的点。
- **追问 2(深挖,呼应第4节):** "如果我把 `EarlyStopping` 放在一个会修改 `logs['val_loss']` 的自定义 callback **前面**,会发生什么?" —— 期望能推理出"`EarlyStopping` 会读到没被修改过的原始 `val_loss`,因为它在 `callbacks` 列表里排得更早,`CallbackList` 是按列表顺序依次把同一个 `logs` 对象传给每个 callback 的",这是对第 4 节"logs 共享 + 顺序有语义"机制的直接应用,能不能答对是检验有没有真的理解、还是死记硬背了两个 API 的用法。
- **追问 3(工程向):** "`restore_best_weights=True` 具体是怎么实现"回到最佳权重"的?会不会很费显存/内存?" —— 期望提到"每次刷新最佳值时用 `get_weights()` 缓存一份快照(是普通 Python/numpy 数组,不是什么特殊机制),训练停止时用 `set_weights()` 覆盖回去;这确实意味着要多占一份完整模型权重的内存/显存,对于超大模型这是一个真实的资源取舍,不是免费的"。

**常见坑:**
- 忘记设置 `validation_data`/`validation_split` 却把 `monitor` 设成 `"val_loss"`——`EarlyStopping`/`ModelCheckpoint` 会因为 `logs` 里根本没有这个 key 而报警告或直接不生效(具体表现取决于版本,但"忘记配验证集却监控 val_ 开头的指标"是最常见的配置错误,没有之一)。
- 以为 `EarlyStopping` 触发后 `model` 里的权重自动是"最好的那次"——只有显式设置 `restore_best_weights=True` 才会在停止时回滚权重,默认值是 `False`,不设置的话,训练停止时模型权重停留在**最后一个 epoch**(可能已经过拟合退步过),不是历史最佳的那个 epoch。
- 把 `patience` 理解成"容忍多少个 epoch 变差"而不是"容忍多少个 epoch 没有刷新历史最佳"——两者在单调变差的极端情况下数值上恰好一样(本节例子就是这种简单情形),但真实训练曲线有噪声、有小幅震荡时,"没有刷新最佳"和"比上一个epoch差"是两个不同的判断标准,`EarlyStopping` 用的是前者。

---

## 6. metrics 状态累积机制——`update_state`/`result`/`reset_state` 三段式协议

**是什么:**
```
m = tf.keras.metrics.Mean()
m.update_state([1.0, 2.0, 3.0])   # 喂一批数据,更新内部状态
m.update_state([10.0])            # 再喂一批,状态继续累积,不是覆盖
m.result()                        # 读取"到目前为止"的聚合结果(这里是running mean)
m.reset_state()                   # 清空内部状态,回到初始值
```
*(以上是签名示意,省略了 `import tensorflow as tf`——完整可运行版本见本节"可运行例子")*

**一句话:** `tf.keras.metrics.Metric` 是一个**有状态的对象**,内部用 `tf.Variable` 保存"从上次 `reset_state()` 到现在,已经看过多少数据、聚合到什么程度",`update_state()` 只负责增量更新这份状态、`result()` 只负责读取当前状态算出的值——这和 PyTorch 代码里常见的"手动攒一个 Python list,每个 batch 的 loss/acc 塞进去,epoch 末尾 `sum(list)/len(list)` 算平均"是两种完全不同的实现思路,前者状态活在 `tf.Variable` 里、可以被 `tf.function` 追踪进图,后者活在普通 Python 对象里。

**底层机制/为什么这样设计:**

`update_state`/`result`/`reset_state` 这三个方法分别对应状态机的"写入"、"读取"、"清空"三个操作,拆成三个独立方法而不是一个"喂数据直接返回当前平均值"的函数,核心原因是**训练循环需要在三个不同的时间点,分别只做其中一个操作**:每个 batch 结束时只想"写入"(不一定要立刻读);`fit()` 的 `on_train_batch_end`/`on_epoch_end` 需要"读取"当前值给 callback/进度条看;每个 epoch 开始时只想"清空"上一个 epoch 的残留状态。把这三件事绑成一个函数,没法适配这种"写入频率(每 batch)"和"清空频率(每 epoch)"本身就不同步的场景。

状态本身是 `tf.Variable`,不是 Python 数字——这一点直接决定了这套机制能不能被塞进第 1 节验证过的 `tf.function` 编译图里:`train_step` 本身会被 `tf.function` 包裹反复执行,如果 metric 状态是一个普通 Python 变量(比如一个 list),每次 `update_state` 追加一个元素这种操作在图执行模式下根本不合法(图里没有"Python list"这种东西);用 `tf.Variable` 承载状态,`update_state` 内部做的是 `total.assign_add(...)`/`count.assign_add(...)` 这类图内合法的原地更新操作,可以被安全地反复执行在编译图里——这也是为什么 `compiled_metrics.update_state(...)` 可以直接放在 `train_step` 里,和第 1 节讲的"Python 副作用在 trace 之后不会真的执行"完全不冲突:因为它更新的根本不是 Python 状态,是图内的 `tf.Variable`。

**可运行例子(用最简单的 `Mean` 直接看到内部状态变量,并验证`fit()`真的在每个epoch边界重置它):**

```python
import tensorflow as tf

m = tf.keras.metrics.Mean(name="my_mean")
assert float(m.result()) == 0.0                 # 初始状态

m.update_state([1.0, 2.0, 3.0])                  # 这一批: 均值2.0
assert abs(float(m.result()) - 2.0) < 1e-6

m.update_state([10.0])                            # 再喂一批 —— 不是覆盖,是累积
# 实测: (1+2+3+10)/4 = 4.0,是"从上次reset到现在"看过的全部4个数的均值,不是这一批的10.0
assert abs(float(m.result()) - 4.0) < 1e-6

# 内部状态就是两个tf.Variable: total(累积和)和count(累积计数),此处结果可以直接手算验证
print("m.variables:", [(v.name, float(v.numpy())) for v in m.variables])
# 实测: [('total:0', 16.0), ('count:0', 4.0)]  -> 16.0/4.0 == result()的4.0,严丝合缝

m.reset_state()
assert float(m.result()) == 0.0                  # 清空回到初始状态,不是"保留但打折"

# sample_weight做加权平均:权重0的数据被完全排除,不是"权重小但仍参与"
m2 = tf.keras.metrics.Mean()
m2.update_state([1.0, 100.0], sample_weight=[1.0, 0.0])
assert abs(float(m2.result()) - 1.0) < 1e-6      # 100.0权重为0,对结果零贡献
```

```python
import tensorflow as tf
import numpy as np

# 直接验证"fit()真的在每个epoch开始前把metric状态清零",不是靠肉眼观察数值巧合
x = np.random.rand(12, 4).astype("float32")
y = np.random.randint(0, 3, size=(12,)).astype("int64")

class CountProbe(tf.keras.callbacks.Callback):
    """on_epoch_begin发生在fit()内部self.reset_metrics()之后(第1节源码顺序),
    在这里读accuracy metric的count状态变量,必须是0"""
    def __init__(self):
        super().__init__()
        self.counts_at_epoch_begin = []
    def on_epoch_begin(self, epoch, logs=None):
        candidates = [m for m in self.model.metrics if m.name == "accuracy"]
        if not candidates:
            self.counts_at_epoch_begin.append(None)   # epoch0: compiled_metrics容器还没懒构建(呼应第2节)
            return
        count_var = [v for v in candidates[0].variables if "count" in v.name][0]
        self.counts_at_epoch_begin.append(float(count_var.numpy()))

model = tf.keras.Sequential([tf.keras.layers.Dense(3, activation="softmax")])
model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
probe = CountProbe()
model.fit(x, y, batch_size=4, epochs=3, verbose=0, callbacks=[probe])

print("每个epoch开始时accuracy的count状态:", probe.counts_at_epoch_begin)
# 实测: [None, 0.0, 0.0] —— epoch0时容器还没建立;epoch1/2开始时都精确是0,
# 证明reset_metrics()确实每个epoch都真实清零了,不是只在第一次生效
assert probe.counts_at_epoch_begin == [None, 0.0, 0.0]
```

**AI 研究/工程场景:** 分布式训练(11 类要展开)下这套"状态活在 `tf.Variable` 里"的设计还有一层好处:每个副本(replica)各自维护一份 `total`/`count` 状态变量,`tf.distribute.Strategy` 可以用标准的变量同步机制(而不是某种专门为 Python 对象设计的跨进程通信)把多个副本的 metric 状态汇总——这是"metric 状态必须是图内变量"这个约束在分布式场景下的直接收益,不只是为了配合 `tf.function`;自定义训练诊断(比如按类别分别统计准确率、按数据来源分桶统计 loss)时,继承 `tf.keras.metrics.Metric` 自己实现 `update_state`/`result`/`reset_state` 三个方法,可以获得和内置 `accuracy`/`mean` 完全一样的生命周期管理(第 2 节验证过的"只要赋值成属性就被自动追踪、自动重置"),不需要重新发明一套状态管理逻辑。

**面试怎么问 + 追问链:**
- **Q:** "`tf.keras.metrics.Mean` 这类 metric 对象,调用两次 `update_state` 之后 `result()` 返回什么?" —— 期望明确说出"是从上次 `reset_state()` 到现在,所有喂进去的数据的聚合结果(运行中的均值),不是最后一次 `update_state` 那一批的值"——这是本节验证过的、最容易被想当然理解错的一点。
- **追问 1(核心):** "为什么不设计成 `update_state` 直接返回当前结果,而要拆成 `update_state`+`result` 两个方法?" —— 期望能说出"因为写入(每个 batch)和读取(`on_train_batch_end`/`on_epoch_end` 才需要)的频率在 `fit()` 编排逻辑里本来就不同步,拆开两个方法才能各自按自己的节奏被调用"。
- **追问 2(区分度高,考察和 `tf.function`/分布式的联系):** "这套状态为什么要用 `tf.Variable` 实现,不能是一个普通 Python 数字或者 list 吗?" —— 期望能联系到第 1 节"`train_step` 被 `tf.function` 包裹"的机制:图执行模式下不能对 Python 原生对象做增量更新,`tf.Variable` 是唯一能在编译图里安全做原地更新的载体;进阶答案能提到分布式场景下 `tf.Variable` 状态天然可以被 `Strategy` 的标准变量同步机制处理。
- **追问 3(对比 PyTorch,考察工程习惯的迁移):** "如果一个从 PyTorch 转过来的工程师,在 Keras 训练代码里手写一个 Python list 收集每个 batch 的 accuracy,epoch 末尾自己算平均,这样做有什么问题?" —— 期望指出"如果这段代码在 `train_step` 内部(会被 `tf.function` 追踪),这类 Python 副作用会撞上第 1 节验证过的'trace 之后不会每步都执行'的坑;即使放在 `fit()` 外层通过 callback 读 `logs` 来做,也是重新发明了 Keras 已经内置、且和分布式/图执行天然兼容的一套机制,没有必要"。

**常见坑:**
- 把 `result()` 当成"这一批(最后一次 `update_state` 传入的数据)的结果",而不是"自上次 reset 以来的累积结果"——本节例子里 `update_state([10.0])` 之后 `result()` 是 4.0 不是 10.0,这个心智错位是从"函数式、无状态"编程习惯迁移过来时最容易踩的一步。
- 在**已经被 `tf.function` 追踪的 `train_step` 内部**对 metric 的 `result()` 调用 `.numpy()`——`result()` 返回的是图内张量,`train_step` 运行在 `tf.function` 追踪的图执行上下文里,对图内张量调用 `.numpy()` 会直接报错(已现场触发,原样抄录):
  ```
  AttributeError: 'SymbolicTensor' object has no attribute 'numpy'
  ```
  正确写法是让 `result()` 的张量原样返回,`.numpy()`/打印这类"落地到具体数值"的操作留给 `fit()` 外层(`tf_utils.sync_to_numpy_or_python_type` 那一步,第 1 节提过)或者 callback 里处理。
- 混淆"`reset_metrics()`(`Model` 的方法,重置**这个模型全部** metrics)"和"某一个具体 metric 对象自己的 `reset_state()`"——两者都存在,前者是后者的批量封装(第 1 节验证过的源码 `for m in self.metrics: m.reset_state()`),手写 `GradientTape` 循环(第 3 节)里如果有多个自定义 metric,想要一次性全部重置,更适合参考这个"遍历 `model.metrics` 逐个调用"的模式,而不是一个个手写。

---

## 7. 混合精度与 `LossScaleOptimizer`——TF 版的 autocast + GradScaler

**是什么:**
```
tf.keras.mixed_precision.set_global_policy("mixed_float16")   # 全局策略,层默认继承
model.compile(optimizer="adam", loss="mse")     # compile()自动把optimizer包进LossScaleOptimizer

# 或者不走fit(),自己手动做scale/unscale(第2/3节train_step覆写/手写循环场景):
opt = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam())
```
*(以上是签名示意,`model` 代表已构造好的模型,不是可执行代码——完整可运行版本见本节"可运行例子")*

**一句话:** TF 处理的是和 [torch-deep-dive/07-training-loop-internals.md](../torch-deep-dive/07-training-loop-internals.md) 第 3/4 节(`autocast`+`GradScaler`)完全同一个问题——fp16 数值范围小,矩阵乘法这类算子用它算收益大,但直接拿 fp16 存权重/算 loss 容易丢精度或者梯度下溢成 0——只是集成方式不同:PyTorch 用两个独立组件手动配合(`with autocast(): ...` + `scaler.scale(loss).backward()`),TF 用一个全局 `Policy` 对象(决定哪些精度)+ 一个包装 optimizer 的类 `LossScaleOptimizer`(决定怎么做 loss scaling),而且走 `compile()`/`fit()` 路径时,这层包装是**自动发生**的,不需要用户手动配。

**底层机制/为什么这样设计:**

**Policy 机制,对应 `autocast` 的"分算子决定精度"。** `mixed_float16` 策略不是把整个模型的权重都转成 fp16——每个 `Layer` 有一个 `dtype_policy`,拆成两个独立的 dtype:`compute_dtype`(前向计算实际使用的精度,`mixed_float16` 下是 `float16`)和 `variable_dtype`(权重**存储**的精度,`mixed_float16` 下固定是 `float32`)。这和 torch07 验证过的"`autocast` 不改变模型参数本身的 dtype,只在计算过程中转精度"是同一个设计原则,只是 TF 把这个"该用什么精度算、该用什么精度存"的决定权做成了一个显式的 `Policy` 对象挂在每一层上,不是像 `autocast` 那样靠一个运行时上下文管理器按算子名单临时决定。

**为什么还需要 `LossScaleOptimizer`——和 torch07 第 4 节 `GradScaler` 一模一样的下溢问题。** fp16 能表示的最小正规数远大于 fp32,训练后期偏小的梯度直接用 fp16 算容易"下溢"成精确的 0;`LossScaleOptimizer` 的解法和 `GradScaler` 完全一样:loss 先乘一个远大于 1 的 `loss_scale` 再反向传播(链式法则下梯度也被同比例放大,`d(scale*L)/dx = scale * dL/dx` 精确成立,不是近似),更新参数前再除回来。`loss_scale` 同样是动态调整的,而且调整规则的参数名几乎是 `GradScaler` 的直接翻译:`dynamic_growth_steps`(对应 torch 的 `growth_interval`)——连续这么多步没有溢出,`loss_scale` 翻倍(乘数固定是 2,对应 `growth_factor=2.0`);一旦检测到梯度里出现 inf/nan,**这一步的参数更新会被跳过**,`loss_scale` 立刻减半。

**`compile()`/`fit()` 路径下这一切都是自动的,这一点直接关联第 8 节的两阶段设计。** `compile()` 发现当前全局 `Policy` 是 `mixed_float16` 且传入的 `optimizer` 还不是 `LossScaleOptimizer` 时,会自动包一层——已验证:传入普通 `SGD`,`compile()` 之后 `model.optimizer` 变成 `LossScaleOptimizerV3` 实例。第 2 节读过的默认 `train_step` 源码里那一行 `self.optimizer.minimize(loss, self.trainable_variables, tape=tape)`,当 `self.optimizer` 是 `LossScaleOptimizer` 时,`minimize()` 内部会自己完成"放大 loss、算梯度、缩小梯度、检测 inf/nan、决定要不要真的更新参数、调整 `loss_scale`"整套流程——这正是"全自动 `fit()`"这条路径(第 1 节)的价值所在:用户完全不需要手写 `get_scaled_loss`/`get_unscaled_gradients`。但如果是第 2 节 `train_step` 覆写或者第 3 节 `GradientTape` 手写循环,`optimizer.minimize()` 这一步便捷方法往往被拆开成 `tape.gradient()`+`apply_gradients()` 好插入自定义逻辑,这时候**必须手动**调用 `get_scaled_loss()`(backward 前)和 `get_unscaled_gradients()`(backward 后、`apply_gradients()` 前)——自动挡和手动挡的边界,恰好落在第 1-3 节讲过的"三条路"的边界上。

**可运行例子(Policy 的 compute/variable dtype 分离,以及 compile() 自动包装 optimizer):**

```python
import tensorflow as tf

print("默认policy:", tf.keras.mixed_precision.global_policy())
tf.keras.mixed_precision.set_global_policy("mixed_float16")

dense = tf.keras.layers.Dense(4)
x = tf.random.normal((2, 4))
out = dense(x)
print("compute_dtype:", dense.compute_dtype, " variable_dtype:", dense.variable_dtype)
print("out.dtype:", out.dtype, " dense.kernel.dtype:", dense.kernel.dtype)
assert dense.compute_dtype == "float16"       # 前向计算用fp16
assert dense.variable_dtype == "float32"      # 权重存储始终fp32,和torch autocast同一个原则
assert out.dtype == tf.float16
assert dense.kernel.dtype == tf.float32

# compile()在mixed_float16策略下,自动把普通optimizer包进LossScaleOptimizer
model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="sigmoid")])
raw_opt = tf.keras.optimizers.SGD()
print("compile()前optimizer类型:", type(raw_opt).__name__)
model.compile(optimizer=raw_opt, loss="binary_crossentropy")
print("compile()后model.optimizer类型:", type(model.optimizer).__name__)
assert isinstance(model.optimizer, tf.keras.mixed_precision.LossScaleOptimizer)

lso = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.SGD())
print("初始loss_scale:", float(lso.loss_scale), " dynamic:", lso.dynamic)
assert float(lso.loss_scale) == 32768.0    # 默认初始scale(2^15),和torch GradScaler默认的2^16同一量级
tf.keras.mixed_precision.set_global_policy("float32")   # 用完复位,避免影响其他代码
```

**一个真实会撞上的 dtype 报错,以及为什么手动模式下要显式 `tf.cast`(现场触发,原样抄录):**

```python
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("mixed_float16")

model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
x = tf.random.normal((4, 4))
y = tf.random.normal((4, 1))          # tf.random.normal默认fp32
pred = model(x, training=True)         # mixed_float16策略下,Dense输出是fp16
print("pred.dtype:", pred.dtype, " y.dtype:", y.dtype)

try:
    _ = pred - y                        # fp16和fp32直接相减,TF不会自动提升类型
except tf.errors.InvalidArgumentError as e:
    print(f"InvalidArgumentError: {e}")
# 实测报错: "cannot compute Sub as input #1(zero-based) was expected to be
#           a half tensor but is a float tensor [Op:Sub]"
# 这正是01类"dtype自动提升规则"这个知识点在混合精度场景下的真实代价:
# TF不会像numpy那样自动把fp16/fp32混合运算提升成fp32,必须显式tf.cast

pred_fixed = tf.cast(pred, tf.float32)   # 正确做法:算loss前把预测值cast回fp32,不是把标签转fp16
loss = tf.reduce_mean(tf.square(pred_fixed - y))
print("cast修复后正常计算 loss:", float(loss))
tf.keras.mixed_precision.set_global_policy("float32")
```

**动态 `loss_scale` 调整机制(和 torch07 `GradScaler` 完全对应的现场实测:连续无溢出涨倍、一遇溢出砍半+跳过该步):**

```python
import tensorflow as tf
import numpy as np

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# 场景1: 连续 dynamic_growth_steps 个干净step, loss_scale翻倍
model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(4,))])
opt = tf.keras.mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.SGD(0.01), initial_scale=8.0, dynamic_growth_steps=3)
scales_seen = [float(opt.loss_scale)]
for i in range(10):
    x = tf.random.normal((4, 4))
    with tf.GradientTape() as tape:
        pred = tf.cast(model(x, training=True), tf.float32)
        loss = tf.reduce_mean(tf.square(pred))
        scaled_loss = opt.get_scaled_loss(loss)          # backward前:放大
    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = opt.get_unscaled_gradients(scaled_grads)      # backward后、apply前:缩小回真实梯度
    opt.apply_gradients(zip(grads, model.trainable_variables))
    scales_seen.append(float(opt.loss_scale))
print("scale历史(initial=8, dynamic_growth_steps=3):", scales_seen)
# 实测: [8.0, 8.0, 8.0, 16.0, 16.0, 16.0, 32.0, 32.0, 32.0, 64.0, 64.0]
# 每凑够3个干净step,scale翻倍一次,精确对应dynamic_growth_steps=3
assert scales_seen[3] == 16.0 and scales_seen[6] == 32.0

# 场景2: 故意用超大输入制造真实溢出,验证"参数更新被跳过 + scale减半"
model2 = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(4,))])
opt2 = tf.keras.mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.SGD(0.1), initial_scale=2.0**14, dynamic_growth_steps=2000)
w_before = model2.layers[0].kernel.numpy().copy()
x2 = tf.random.normal((4, 4)) * 1e4
with tf.GradientTape() as tape:
    pred2 = tf.cast(model2(x2, training=True), tf.float32)
    loss2 = tf.reduce_mean(tf.square(pred2))
    scaled_loss2 = opt2.get_scaled_loss(loss2)
scaled_grads2 = tape.gradient(scaled_loss2, model2.trainable_variables)
grads2 = opt2.get_unscaled_gradients(scaled_grads2)
has_bad = any(tf.reduce_any(tf.math.logical_or(tf.math.is_inf(g), tf.math.is_nan(g))).numpy()
              for g in grads2 if g is not None)
assert has_bad   # 确认这一步梯度真的溢出了
opt2.apply_gradients(zip(grads2, model2.trainable_variables))
w_after = model2.layers[0].kernel.numpy()
assert np.array_equal(w_before, w_after)             # 权重完全没动:这一步被跳过
assert float(opt2.loss_scale) == (2.0**14) / 2        # scale立刻减半
print("溢出场景: 权重未更新, scale从", 2.0**14, "减半到", float(opt2.loss_scale))
tf.keras.mixed_precision.set_global_policy("float32")
```

**AI 研究/工程场景:** 和 torch07 的结论完全一致——混合精度是现代大模型训练的标配技术,不是可选优化;`tf.keras.mixed_precision` 同样支持 `mixed_bfloat16` 策略(`bfloat16` 指数位和 fp32 一样多,数值范围一致、不容易溢出,只是尾数更短精度更低),在 TPU 和较新 GPU 上训练大模型时,`mixed_bfloat16` 往往比 `mixed_float16` 更常见,原因和 torch07 讲过的一样:担心中间值超出 fp16 表示范围;`mixed_bfloat16` 策略下由于不容易溢出,`LossScaleOptimizer` 这层包装通常也不再需要(数值范围问题本身就小很多),这是实际选型时的一个直接权衡点。

**面试怎么问 + 追问链:**
- **Q:** "TF 怎么做混合精度训练?和 PyTorch 的 `autocast`+`GradScaler` 相比,设计上有什么不同?" —— 期望能说出"解决的是同一个问题(计算用低精度提速、权重保持高精度存储、loss scaling 防止梯度下溢),TF 用全局 `Policy` 对象决定精度、`LossScaleOptimizer` 包装 optimizer 决定 loss scaling,`compile()`/`fit()` 路径下两者都是自动集成的,不需要像 PyTorch 那样手动写 `with autocast()` 和 `scaler.scale().backward()`"。
- **追问 1(核心,呼应torch07):** "为什么需要 loss scaling,只转 fp16 计算不够吗?" —— 期望复用 torch07 第 4 节的结论:"fp16 表示范围小,训练后期偏小的梯度容易下溢成 0,loss scaling 通过放大 loss(进而放大梯度)避开这个问题,数学上是精确操作(链式法则下标量可以直接提出来),不是近似"。
- **追问 2(区分度高,考察对"自动挡/手动挡"边界的理解):** "如果我覆写了 `train_step`,还需要手动调用 `get_scaled_loss`/`get_unscaled_gradients` 吗?" —— 期望答"要看内部怎么算梯度:如果还是调用 `self.optimizer.minimize(loss, vars, tape=tape)` 这个便捷方法,`LossScaleOptimizer` 会自己处理整个 scale/unscale/inf检测流程;但如果像插入自定义梯度处理逻辑那样拆成 `tape.gradient()`+`apply_gradients()` 两步(第 2 节讨论过这个拆分的动机),就必须自己在 backward 前后分别调用这两个方法,`apply_gradients()` 本身不会替你做 scale/unscale"。
- **追问 3(工程向):** "`mixed_float16` 和 `mixed_bfloat16` 怎么选?" —— 期望提到"`bfloat16` 数值范围和 fp32 一致、不容易溢出,大模型训练/TPU 场景更常用;`mixed_bfloat16` 下 loss scaling 的必要性也相应降低",可以直接类比 torch07 里 `bfloat16` vs `float16` 的同一个结论。

**常见坑:**
- **`mixed_float16` 策略下,层输出是 fp16,但常见的标签/常量创建方式(`tf.random.normal`、普通 numpy 数组转 tensor)默认是 fp32,直接做逐元素运算会报错**——这是本节现场触发的真实报错,TF 不会像 numpy 那样自动把 fp16/fp32 混合运算提升成 fp32(呼应 01 类"dtype 自动提升规则"这个知识点),正确做法是在算 loss 之前把预测值 `tf.cast` 到 fp32(不是把标签转成 fp16——标签/loss 这类数值敏感的计算应该在更高精度上进行,这和 `autocast` 里"`sum`/`softmax`/loss 强制留在 fp32"是同一个考量)。
- 只设置了 `Policy` 却继续用普通 `optimizer`、自己手写 `GradientTape` 循环(第 3 节)时忘记调用 `get_scaled_loss`/`get_unscaled_gradients`——`compile()`/`fit()` 路径下这一步是自动的,容易让人误以为所有路径下都不需要手动处理,手写循环里漏掉不会报错,只是在训练后期梯度普遍偏小时悄悄损失一部分本该更新的信号。
- 忘记 `set_global_policy` 是**全局、进程级别**的设置,不是"只影响接下来新建的这一个模型"——同一个 Python 进程里后面创建的所有层,只要没有单独指定 `dtype` 参数覆盖,都会继承这个全局策略,调试时如果忘记在用完后调回 `"float32"`,容易在同一个脚本后续本不需要混合精度的部分意外引入 fp16 计算。

---

## 8. `compile()` 参数与 `fit()` 的关系——"声明配置"和"真正执行"的两阶段设计

**是什么:**
```
model.compile(optimizer="adam", loss="mse", metrics=["mae"])   # 阶段1: 只登记配置,不碰任何数据
model.fit(x, y, epochs=10)                                     # 阶段2: 真正拿数据跑
```
*(以上是签名示意,`model`/`x`/`y` 代表已构造好的模型和数据,不是可执行代码——完整可运行版本见本节"可运行例子")*

**一句话:** `compile()` 和 `fit()` 是两个职责严格分开的阶段——`compile()` 只是把"用什么 `optimizer`/`loss`/`metrics` 训练"这个**声明性配置**记录到模型对象上,不涉及任何一条真实数据;`fit()` 才是真正把数据接进来、驱动第 1 节讲的那套编排逻辑跑起来的阶段。这个划分不是随意的 API 设计选择,第 2 节"为什么能覆写 `train_step`"、第 1 节"`train_function` 为什么要缓存"这两件事,都是这个两阶段设计的直接推论。

**底层机制/为什么这样设计:**

**没有 `compile()`,`fit()` 会在第一行就拒绝执行。** `fit()` 内部第一步要做的事情之一,就是确认 `self.optimizer`/`self.compiled_loss` 这些"训练要用的规则"已经就绪——已验证,没调用过 `compile()` 直接 `fit()`,会现场报错(原样抄录):
```
RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.
```
这条报错本身就说明了 `compile()` 不是"可选的美化步骤",而是 `fit()`/`train_step` 覆写这两条路径(第 1/2 节)运行的**前提条件**——第 3 节验证过,只有完全抛开 `fit()` 的手写 `GradientTape` 循环才不需要它。

**`compile()` 阶段能立刻确定的东西,和需要真实数据才能确定的东西,是分开的。** `compile()` 调用完成的瞬间,`model.optimizer` 就已经是一个真实的 `Adam`/`SGD` 优化器实例,`model.compiled_loss` 也已经是一个真实的 `LossesContainer` 对象——这些东西不依赖数据的具体形状/数值,只依赖"用什么规则"这个声明,所以 `compile()` 可以在见到任何一条数据之前就把它们创建好。但第 2 节验证过 `compiled_metrics`/`compiled_loss` 内部的容器是**懒构建**的——需要看到真实的 `y_pred`/`y` 才能知道要创建哪些具体的 metric 对象,这部分工作必须推迟到第一次真正 `update_state()` 调用(也就是 `fit()` 跑起来之后)才发生。`compile()`/`fit()` 的边界,恰好就是"只依赖训练规则本身"和"依赖真实数据"这两类工作的边界。

**`train_function` 的缓存和失效,直接绑定在这个边界上。** 第 1 节验证过 `train_function` 会被缓存(避免每次 `fit()` 都重新 `trace`),但缓存不是永久的——`make_train_function` 的官方文档原话是"该函数在 `Model.fit` 或 `Model.train_on_batch` 第一次被调用时缓存,`Model.compile` 被调用时缓存会被清空"。已验证:同一次 `compile()` 之后连续调用两次 `fit()`,两次拿到的 `train_function` 是**同一个 Python 对象**(`is` 恒等,不只是行为相同);一旦中间插入一次新的 `compile()`,`train_function` 立刻变回 `None`,下次 `fit()` 会重新构建。这个设计是必要的,而不是保守起见的额外失效——重新 `compile()` 很可能换了不同的 `optimizer`/`loss` 对象,旧的编译图里"烧"进去的是旧对象的引用,继续复用旧图会指向已经不对的训练规则,必须重新 `trace`。

**这个两阶段设计,对第 2 节"为什么覆写 `train_step` 要用 `self.compiled_loss`/`self.optimizer` 而不是硬编码"这件事有直接影响。** 如果自定义 `train_step` 内部写死了一个具体的 `tf.keras.losses.MSE()` 或者绑定一个固定的 `optimizer` 实例,那么这个自定义 `Model` 子类就没法再通过换一次 `compile()` 参数来复用——`compile()` 存在的意义就是让用户在不碰 `train_step` 代码的前提下换训练规则,`train_step` 里读 `self.compiled_loss`/`self.optimizer` 而不是硬编码,才能把这个"两阶段"的约定继续兑现下去。

**可运行例子:**

```python
import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
x = np.random.rand(8, 4).astype("float32")
y = np.random.rand(8, 1).astype("float32")

# 阶段边界1: 没有compile()直接fit(),现场报错
try:
    model.fit(x, y, epochs=1, verbose=0)
    raise SystemExit("expected RuntimeError")
except RuntimeError as e:
    print("RuntimeError:", str(e))
    assert "must compile" in str(e).lower()

assert model.optimizer is None                    # compile()之前,这个属性不存在
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print("compile()后 optimizer:", model.optimizer)
print("compile()后 compiled_loss:", model.compiled_loss)
assert model.optimizer is not None                 # compile()瞬间就创建好了,不需要任何数据
assert model.train_function is None                 # 但train_function还没有 —— 要fit()第一次调用才构建

model.fit(x, y, epochs=1, verbose=0)
tf_fn_1 = model.train_function
model.fit(x, y, epochs=1, verbose=0)                # 再fit()一次,同一次compile()之下
tf_fn_2 = model.train_function
assert tf_fn_1 is tf_fn_2                            # 同一个Python对象 —— 复用缓存,没有重新trace

model.compile(optimizer="sgd", loss="mse")           # 重新compile() —— 换了新的optimizer
assert model.train_function is None                 # 缓存被清空:旧图绑定的是旧optimizer,必须作废重建
print("重新compile()后train_function缓存被清空,下次fit()会重新构建")
```

**AI 研究/工程场景:** 超参数扫描(hyperparameter sweep)场景下,同一个模型结构(甚至同一个已经训练过一部分的模型)反复 `compile()` 不同的 `optimizer`/学习率再 `fit()`,是这个两阶段设计最直接的收益——不需要重新定义模型结构代码;两阶段训练(比如先用较大学习率训一阵子,再重新 `compile()` 换更小学习率做 fine-tune)同样依赖"换训练规则不用换模型代码"这个前提,但要小心:重新 `compile()` 换了新的 `optimizer` 实例,像 Adam 这类有状态优化器积累的动量(`m`/`v`,07 类讲过的内容)也跟着旧对象一起被丢弃,不会自动迁移到新 `optimizer` 上,这是 fine-tune 场景真正切换学习率策略时需要留意的代价。

**面试怎么问 + 追问链:**
- **Q:** "`compile()` 和 `fit()` 分别做什么?为什么要分成两步,不能合并成一个函数吗?" —— 期望说出"`compile()` 是声明式配置(定下用什么 optimizer/loss/metrics 训练,不碰数据),`fit()` 才是真正执行;拆开是为了让'换训练规则'和'要不要重新定义模型/重新准备数据'解耦"。
- **追问 1(核心):** "`compile()` 时没有传入任何数据,`optimizer`/`loss` 这些对象是怎么就绪的?" —— 期望能区分"只依赖训练规则本身、不需要看数据就能确定"(`optimizer`/`compiled_loss` 在 `compile()` 时就创建)和"需要真实数据才能确定"(`compiled_metrics` 内部具体的 metric 对象要等第一次 `update_state()`,这是第 2 节验证过的懒构建)——能不能答出这个区分,是检验有没有真的理解两阶段边界画在哪里的关键。
- **追问 2(呼应第1节,考察源码理解的连贯性):** "连续两次 `fit()` 调用,中间没有 `compile()`,`train_function` 是每次都重新构建的吗?如果中间插入一次 `compile()` 呢?" —— 期望答"没重新 `compile()` 就是同一个缓存对象,复用编译好的图;一旦重新 `compile()`,缓存立刻失效,因为新的训练规则(可能是全新的 `optimizer`/`loss` 对象)不能继续用旧图"。
- **追问 3(连接第2节,考察设计意图的理解深度):** "覆写 `train_step` 时,为什么建议用 `self.compiled_loss`/`self.optimizer` 而不是在 `__init__` 里直接写死一个具体的 loss/optimizer?" —— 期望答"写死的话,这个自定义 `Model` 子类就没法再通过换 `compile()` 参数复用了——两阶段设计的意义就是'换训练规则不用碰 `train_step` 代码',`train_step` 里硬编码等于把这个设计承诺破坏掉了"。

**常见坑:**
- 以为每次 `fit()` 前都必须重新 `compile()`——不需要,只有真的要更换 `optimizer`/`loss`/`metrics` 配置才需要;习惯性地每次都重新 `compile()`,会导致 `train_function` 缓存被无谓清空,下一次 `fit()` 的第一步会重新触发第 1 节讲过的 trace 开销,是一个纯粹浪费、没有任何收益的性能坑。
- 重新 `compile()` 换成新的 `optimizer` 实例后,期望旧 `optimizer` 积累的动量状态被保留——不会,新对象是全新初始化的状态,旧状态随旧对象一起被丢弃,这在训练中途调整学习率策略、又误用"重新 `compile()`"而不是直接改 `optimizer.learning_rate` 的场景下容易被忽略。
- 误以为配置错误(比如 `metrics` 传了一个拼错名字的字符串)会在 `compile()` 这一步就报出来——实际上很多校验要等真实数据流过 `train_step` 才会触发,`compile()` 阶段本身几乎不接触数据,能提前发现的问题非常有限。

---

## 9. `validation_data`/`validation_split` 内部机制

**是什么:**
```
model.fit(x, y, epochs=10, validation_data=(x_val, y_val))   # 用户自己准备好的独立验证集
model.fit(x, y, epochs=10, validation_split=0.2)              # 让fit()自动从x/y切一部分出来当验证集
```
*(以上是签名示意,`model`/`x`/`y`/`x_val`/`y_val` 代表已构造好的模型和数据,不是可执行代码——完整可运行版本见本节"可运行例子")*

**一句话:** `validation_split` 不是随机抽样,而是直接从传入的训练数组**尾部**切一刀(不 shuffle、不重排);不管验证集是这样自动切出来的还是用户自己传的 `validation_data`,`fit()` 都是在每个 epoch 的训练 batch 全部跑完之后、`on_epoch_end` 回调触发之前,通过调用 `self.evaluate()`(内部走 `test_step`,`training=False`)来跑验证,得到的指标会加上 `val_` 前缀合并进当前 epoch 的 `logs` 里。

**底层机制/为什么这样设计:**

**`validation_split` 怎么切,直接决定了它是否适合你的数据。** `fit()` 内部处理 `validation_split` 用的是 `data_adapter.train_validation_split(...)`,已验证其行为:给一个 20 条数据的数组、`validation_split=0.2`,切出来训练集是索引 `0~15`(前 80%),验证集是索引 `16~19`(**最后** 20%),完全按原始顺序,没有任何 shuffle。这个"切尾部"的行为不是随手实现的细节,而是刻意为之的取舍:不 shuffle 意味着**可复现**(同样的数据、同样的 `validation_split`,每次切出来的验证集完全一样,不受随机种子影响);代价是如果原始数据本身是**有序**的(比如按类别拼接、按采集时间排列),直接用 `validation_split` 切出来的验证集分布可能和训练集有系统性差异,必须在调用 `fit()` 之前自己先 shuffle 整个数据集。

**验证发生的时机,复用的是第 1 节讲过的同一套编排逻辑,只是换了一个 `_step`。** 直接读第 1 节引用过的 `fit()` 主循环:一个 epoch 里所有训练 batch 跑完之后,`if validation_data and self._should_eval(...): val_logs = self.evaluate(...)`,紧接着才是 `callbacks.on_epoch_end(epoch, epoch_logs)`。这意味着:`on_epoch_end` 里能读到的 `logs`,是训练指标和验证指标**已经合并**之后的结果,`val_loss`/`val_accuracy` 这些 `val_` 前缀的 key 和普通训练指标同时存在于同一个字典里——第 4/5 节验证过的 `EarlyStopping`/`ModelCheckpoint` 之所以能直接监控 `monitor="val_loss"`,靠的就是这一步合并。而 `self.evaluate()` 内部走的是**另一套**方法:`test_step`(不是 `train_step`),`test_step` 里对模型的调用是 `self(x, training=False)`——这直接影响所有"行为随 `training` 参数改变"的层(`BatchNorm` 用 batch 统计量还是移动平均、`Dropout` 要不要真的丢弃),验证阶段和训练阶段用的是两条不同的前向路径,不只是"数据不同"这么简单。

**`validation_data` 和 `validation_split` 同时提供,不会报错,`validation_data` 静默胜出。** 已验证:两者同时传入 `fit()` 不会抛异常,实际生效的是 `validation_data`(通过验证阶段跑的 batch 数量精确对上 `validation_data` 的样本数,而不是 `validation_split` 会切出来的样本数,反推确认)——这是一个容易被忽略、没有任何警告提示的行为,混用时很容易以为两者是"叠加"或者以为传的是 `validation_split` 那一份。

**可运行例子(切分方式、训练/验证阶段 `training` 标志差异、`validation_freq` 控制频率):**

```python
import tensorflow as tf
import numpy as np

# --- validation_split 切的是"尾部",不shuffle ---
from tf_keras.src.engine import data_adapter

N = 20
x_idx = np.arange(N).astype("float32").reshape(-1, 1)
y_idx = np.arange(N).astype("float32").reshape(-1, 1)
(train_x, train_y), (val_x, val_y) = data_adapter.train_validation_split((x_idx, y_idx), validation_split=0.2)
train_x = train_x.numpy() if hasattr(train_x, "numpy") else np.asarray(train_x)
val_x = val_x.numpy() if hasattr(val_x, "numpy") else np.asarray(val_x)
print("训练集切到的索引范围:", train_x.min(), "~", train_x.max())
print("验证集切到的索引范围:", val_x.min(), "~", val_x.max())
assert train_x.tolist() == [[i] for i in range(16)]        # 前80%,原始顺序
assert val_x.tolist() == [[i] for i in range(16, 20)]       # 后20%,不是随机抽样

# --- 验证阶段走 training=False,和训练阶段的 training=True 是两条不同前向路径 ---
# 用tf.Variable计数,不用Python list——这里就是第1节验证过的教训本身:
# call()跑在被tf.function追踪的train_step/test_step内部,Python list.append只在trace时执行,
# 不代表真实batch数(现场验证:如果改用Python list,数出来是[False,True,True,False],
# count(True)=2、count(False)=2,精确等于trace次数而不是真实batch数,是本节踩过的真实坑)
class TrainFlagProbe(tf.keras.layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.true_count = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.false_count = tf.Variable(0, dtype=tf.int64, trainable=False)
    def call(self, inputs, training=None):
        if training:
            self.true_count.assign_add(1)
        else:
            self.false_count.assign_add(1)
        return inputs

probe = TrainFlagProbe()
inp = tf.keras.Input(shape=(1,))
out = tf.keras.layers.Dense(1)(probe(inp))
model = tf.keras.Model(inp, out)
model.compile(optimizer="sgd", loss="mse")
# 15条训练(15/4=4个batch: 4,4,4,3) + 5条验证(5/4=2个batch: 4,1),用validation_split=0.25精确对应
model.fit(x_idx, y_idx, batch_size=4, epochs=1, verbose=0, validation_split=0.25)
print("training=True(训练batch)次数:", int(probe.true_count.numpy()),
      " training=False(验证batch)次数:", int(probe.false_count.numpy()))
assert int(probe.true_count.numpy()) == 4     # 4个训练batch,和真实batch数精确对应
assert int(probe.false_count.numpy()) == 2    # 2个验证batch

# --- validation_freq 控制验证不必每个epoch都跑 ---
class ValEpochRecorder(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epochs_with_val = []
    def on_epoch_end(self, epoch, logs=None):
        if logs and "val_loss" in logs:
            self.epochs_with_val.append(epoch)

model2 = tf.keras.Sequential([tf.keras.layers.Dense(1)])
model2.compile(optimizer="sgd", loss="mse")
rec = ValEpochRecorder()
model2.fit(x_idx, y_idx, batch_size=4, epochs=6, verbose=0,
           validation_data=(x_idx, y_idx), validation_freq=2, callbacks=[rec])
print("跑了验证的epoch(0-indexed,validation_freq=2,共6个epoch):", rec.epochs_with_val)
assert rec.epochs_with_val == [1, 3, 5]   # 第2/4/6个epoch(0-indexed是1/3/5)才跑验证

# --- validation_data 和 validation_split 同时给:不报错,validation_data静默胜出 ---
class ValStepCounter(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.n = 0
    def on_test_batch_end(self, batch, logs=None):
        self.n += 1

model3 = tf.keras.Sequential([tf.keras.layers.Dense(1)])
model3.compile(optimizer="sgd", loss="mse")
vc = ValStepCounter()
# validation_data是5条(->2个batch); validation_split=0.2切出来的会是4条(->1个batch)
model3.fit(x_idx, y_idx, batch_size=4, epochs=1, verbose=0,
           validation_split=0.2, validation_data=(x_idx[:5], y_idx[:5]), callbacks=[vc])
print("实际验证跑的batch数:", vc.n, "(2说明用的是validation_data,不是validation_split)")
assert vc.n == 2
```

**AI 研究/工程场景:** 真实数据管线里,原始数据经常是按采集顺序或者按类别拼接存放的(比如先收集完类别 A 的所有样本再收集类别 B),这种情况下直接用 `validation_split` 而不先手动 shuffle,验证集很可能只覆盖某几个类别、和训练集分布严重不一致,评估出来的 `val_accuracy` 没有代表性——这是真实项目里"验证集指标异常好(或异常差)、和实际部署表现对不上"经常被追查到的原因之一,`validation_split` 的"确定性切尾部"特性必须和"数据本身有没有提前 shuffle"放在一起考虑;数据量大到不能整个放进内存时,`validation_data`/`validation_split` 这套面向 numpy 数组设计的机制通常也不再适用(`tf.data.Dataset` 输入下 `validation_split` 往往直接不支持),这时验证集需要作为独立的 `tf.data.Dataset` 提前手动划分好,通过 `validation_data` 传入,09 类 `tf.data` 输入管线会再展开这部分。

**面试怎么问 + 追问链:**
- **Q:** "`validation_split=0.2` 和你自己手动划分训练/验证集,效果一样吗?" —— 期望不满足于"差不多",能主动提到"切分方式不同——`validation_split` 是确定性地切原始数组的尾部,不 shuffle"。
- **追问 1(核心,陷阱题):** "`validation_split` 切分时会不会先打乱数据?" —— 期望明确答"不会,直接切最后 N%,原始顺序不变;如果数据本身有序(按类别/时间排列),必须自己先 shuffle 整个数据集再调用 `fit()`,否则切出来的验证集分布可能和训练集差异很大,这是很实际的坑,不是理论上的边界情况"。
- **追问 2(考察对 `training` 参数机制的理解,呼应04类):** "验证阶段的前向计算和训练阶段一样吗?" —— 期望答"不一样,验证走 `self.evaluate()` → `test_step()`,内部调用模型是 `training=False`;训练阶段是 `training=True`——这直接影响 `BatchNorm`/`Dropout` 这类行为随 `training` 参数变化的层,不只是'看的数据不同'"。
- **追问 3(工程向):** "如果验证集很大、每个 epoch 都跑一遍验证会明显拖慢训练,有什么办法?" —— 期望提到 `validation_freq` 参数(设成大于 1 的整数,或者传一个具体 epoch 编号的列表),本质是"验证信号及时性"和"训练总耗时"之间的权衡,不验证不代表训练本身变慢,只是少了实时反馈。
- **深挖追问:** "如果我同时传了 `validation_data` 和 `validation_split`,会发生什么?" —— 期望候选人诚实回答"不确定/需要查证"好过瞎猜,更好的回答是能推理出"大概率其中一个会被忽略而不是报错,因为两者语义上是同一个用途的两种不同数据来源方式";如果候选人能准确说出"`validation_data` 会静默胜出",说明这是真的深挖过源码或者踩过坑,是本节验证过的真实行为。

**常见坑:**
- **原始数据没有提前 shuffle,直接使用 `validation_split`**——这是本节反复强调、也是最容易被忽视的一点:已验证 `validation_split` 精确切的是数组尾部,不做任何重排;如果训练脚本里数据是"先加载类别A、再加载类别B……"这样拼起来的,不 shuffle 直接切 `validation_split`,验证集很可能只包含最后几个类别,`val_accuracy` 会是一个没有代表性、甚至完全没有意义的数字。
- **同时传 `validation_data` 和 `validation_split`,却不清楚哪个在生效**——已验证不会报错,`validation_data` 静默胜出,`validation_split` 被完全忽略,没有任何警告提示这一点,容易在重构代码时因为"两个参数都还留着"而产生这种静默的配置冲突。
- 以为 `validation_freq` 影响的是"验证集本身缩小了",实际上它控制的是"隔几个 epoch 才跑一次完整验证",没跑验证的那些 epoch,`logs` 里根本不会出现 `val_` 开头的 key(这也是第 4/5 节 `EarlyStopping`/`ModelCheckpoint` 如果监控 `val_loss`、又搭配 `validation_freq>1` 使用时需要小心的一点——那些没跑验证的 epoch,`monitor` 读不到值)。

---

## 小结:这一批 9 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | `model.fit()` 内部机制 | 三环节编排:`DataHandler` 取批次、`train_step` 被 `tf.function(reduce_retracing=True)` 包裹、`CallbackList` 在精确缝隙插钩子;已实测 trace 只发生2次(含形状泛化),Python副作用不代表每步真实执行 |
| 2 | `Model.train_step()` 覆写 | "全自动"和"手写"之间的第三条路,PyTorch核心生态没有直接对应物(Lightning是第三方复刻);覆写后fit()三块基础设施原样生效(已用ModelCheckpoint+自定义callback实测),自定义metric只需属性赋值即被自动追踪 |
| 3 | `GradientTape` 手写循环 | 和PyTorch心智模型最接近,compile()对这条路径不是必需品(已验证`model.optimizer`为None不影响训练);不加`@tf.function`比编译图慢3倍以上(已实测) |
| 4 | 自定义 Callback | `self.model`由`CallbackList.set_model()`自动注入,不是构造参数(已验证);`logs`是所有callback共享的同一个字典对象,顺序有实际语义 |
| 5 | 内置 callbacks 机制 | `ModelCheckpoint`/`EarlyStopping`就是普通`Callback`子类,同样挂`on_epoch_end`,没有特权通道;已用注入式`val_loss`确定性触发`EarlyStopping`(3/20 epoch,stopped_epoch=2) |
| 6 | metrics 状态协议 | `update_state`/`result`/`reset_state`三段式,状态活在`tf.Variable`里(已验证`total`/`count`);已直接证明`fit()`在每个epoch边界把状态精确清零(不是数值巧合) |
| 7 | 混合精度 + `LossScaleOptimizer` | 和torch07 autocast+GradScaler同一个问题,TF用全局Policy+optimizer包装、compile()下自动集成;已实测scale按`dynamic_growth_steps`翻倍、溢出时减半+跳过该步 |
| 8 | `compile()`/`fit()`两阶段设计 | compile()只声明配置不碰数据,`optimizer`/`compiled_loss`立即就绪但`compiled_metrics`懒构建;`train_function`跨多次fit()复用同一对象,重新compile()才失效(已验证对象恒等) |
| 9 | `validation_data`/`validation_split` | `validation_split`确定性切数组**尾部**、不shuffle(已验证);验证走`test_step`+`training=False`;两者同传时`validation_data`静默胜出(已用batch计数反推验证) |

**这一批和 torch07 最大的不同,回到开篇的框架:** torch 训练循环是"一条路走到黑"的手写 eager 循环,`torch.compile`(torch07第6节)是后来才补上的性能选项;TF/Keras 从设计之初就摆出三条并行的路,`fit()` 全自动 → `train_step` 覆写 → `GradientTape` 手写,厚度不同的自动化外壳包着同一套底层机制。真正面试会考察的,往往不是"这三条路分别怎么写"(这是文档层面的知识),而是"给定一个具体需求,你能不能准确判断该走哪一条、为什么"——这也是本篇花了接近一半篇幅在第 1-3 节反复对照这三条路径关系的原因。

下一批:[09-tf-data-pipeline.md](09-tf-data-pipeline.md) —— tf.data 输入管线机制(`Dataset.from_tensor_slices`、`map()`并行化、`batch`/`shuffle`/`prefetch`正确顺序、与 PyTorch `DataLoader` 的设计差异对比)。

---

*更新:2026-07-11*
