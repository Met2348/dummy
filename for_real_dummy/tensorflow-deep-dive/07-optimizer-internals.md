# 07 · 优化器内部机制深挖(Optimizer Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批讲"TF 的优化器 API 是怎么设计的、内部状态怎么管理的"——**不重新推导 SGD/Momentum/Adam/AdamW 的数学**,那一整套一阶矩、二阶矩、bias correction、AdamW 解耦权重衰减的推导,已经在 [torch-deep-dive/06-optimizer-internals.md](../torch-deep-dive/06-optimizer-internals.md) 里逐行读源码、逐步验证过一遍,数学本身不会因为换了个框架就变——本文默认你已经看过那一篇(至少第 1/4/5/6 节),遇到公式直接引用结论,不重复推导。本文要打开的是另一个黑箱:同一套数学,TF 的 API **长什么样**、优化器状态**存在哪、怎么查**、什么该由你手动调用、什么被框架自动做掉了——这些恰恰是两个框架里差异最大、最容易在"从 PyTorch 转 TF"或者"背题背串了"时露馅的地方。

**本文和 torch-deep-dive 06 的关系:** 完全同一套优化算法数学、两套框架实现。哪一节对应 torch06 哪一节,正文里会随时标注;凡是本文没有重新推导的公式,默认结论和 torch06 完全一致,不一致的地方(目前发现的只有第 4 节 weight decay 这一处)会明确指出"这里和 torch 不一样"。

**关于"AI 研究/工程场景"段落的诚实声明:** 仓库里没有可引用的 TF/Keras 代码,本文这部分是根据真实训练场景重构的例子,不是仓库引用(完整声明见 [00-roadmap.md](00-roadmap.md))。

**验证方法论:** 本文每一条"底层机制"结论都经过两步验证:①用 `inspect.getsource()` 现场读 TF 源码原文(不是转述文档或凭经验断言);②针对读到的每个关键行为写一段能现场断言的最小复现代码。所有代码已在 WSL2 `~/tf-venv`(TensorFlow **2.21.0**,`TF_USE_LEGACY_KERAS=1`,GPU 直通已验证可用)下实际跑通,环境细节见 [00-roadmap.md](00-roadmap.md) 第 0 节。

**本篇统一结构(与 00-roadmap.md 完全一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. `tf.keras.optimizers` 系列总览与 legacy/新 API 现状

**是什么:**
```
tf.keras.optimizers.Adam(learning_rate=0.001, ...)          # 当前默认实现(2.11 前后重写过的新架构)
tf.keras.optimizers.legacy.Adam(learning_rate=0.001, ...)   # 重写之前的实现,原样保留在 legacy 命名空间
tf.keras.optimizers.experimental.Adam(...)                  # 重写期间的过渡命名空间,历史遗留产物
```
*(以上是签名示意,`...` 代表省略的其余参数,不是可执行代码——完整可运行版本见本节"可运行例子")*

**一句话:** TF/Keras 的优化器**实现本身**在 TF 2.11 前后经历过一次和"Keras 2/3 分裂"完全独立的第二次架构重写——`tf.keras.optimizers.Adam` 这个名字默认指向的实现换过,旧实现没有被删除,而是原样搬进了 `tf.keras.optimizers.legacy` 命名空间留作后备。

**底层机制/为什么这样设计:**

必须先把两条独立的"legacy"轴分清楚,这是本节存在的第一个理由:
1. **Keras 2(tf_keras)vs Keras 3** —— 00 篇环境声明讲过,本系列全程锁定 Keras 2(`TF_USE_LEGACY_KERAS=1`),这条轴和"优化器实现"无关,是整个 `tf.keras` 命名空间指向哪个包的问题。
2. **本节要讲的这条轴,在 Keras 2(tf_keras)内部**:优化器的具体实现类,在 TF 2.11 前后又发生了一次独立重写。旧实现类叫 `OptimizerV2`(这个"V2"是相对更早的 TF1 `tf.train.Optimizer` 而言,和 Keras 2/3 的"2/3"没有任何关系),新实现类叫 `Optimizer`(内部继承自 `_BaseOptimizer`)。两次重写各自独立,只是巧合地都会被泛称"legacy vs 新版",面试时如果把这两条轴说串会显得没有真正搞清楚现状。

用继承链和源文件路径实测证实"新旧是并存关系,不是谁继承谁":

```python
import tensorflow as tf
print(tf.keras.optimizers.Adam.__module__)
print([c.__name__ for c in tf.keras.optimizers.Adam.__mro__])
print(tf.keras.optimizers.legacy.Adam.__module__)
print([c.__name__ for c in tf.keras.optimizers.legacy.Adam.__mro__])
```
实测输出:
```
tf_keras.src.optimizers.adam
['Adam', 'Optimizer', '_BaseOptimizer', 'AutoTrackable', 'Trackable', 'object']
tf_keras.src.optimizers.legacy.adam
['Adam', 'OptimizerV2', 'Trackable', 'object']
```
两条完全不同的继承链、不同的源文件(`optimizers/adam.py` vs `optimizers/legacy/adam.py`),`tf.keras.optimizers.legacy.Adam` 不是 `tf.keras.optimizers.Adam` 的父类也不是子类,是两套独立到底的实现。

重写带来的实际收益(能从代码结构直接读出来,不是转述发布记录):旧的 `OptimizerV2` 架构里,分布式梯度聚合、混合精度、EMA(权重指数滑动平均)、梯度裁剪这些"每个优化器都要有一份"的横切能力,基本靠每个子类各自实现或者靠继承层层叠加;新架构把这些全部收进 `_BaseOptimizer`/`Optimizer` 基类统一实现一次——直接证据是新版 `Adam.__init__` 的签名里,`weight_decay`、`clipnorm`/`clipvalue`/`global_clipnorm`、`use_ema`/`ema_momentum`、`jit_compile` 这些参数,SGD、RMSprop、Adam……**所有**新架构优化器都天然具备(本篇第 4/6 节要讲的能力,根源就在这里的"基类统一实现"),而 `tf.keras.optimizers.legacy.Adam` 的构造函数签名里完全没有 `weight_decay` 这一项——这不是疏忽,是旧架构压根没有在基类层面设计这个能力。

`tf.keras.optimizers.experimental` 命名空间是重写过程中的过渡产物——新架构在正式转正、成为 `tf.keras.optimizers.Xxx` 默认指向之前,曾经以"experimental"的身份和旧实现并行发布过一段时间。现在 `tf.keras.optimizers.Adam` 已经直接是转正后的新实现,`experimental` 命名空间是历史遗留,新代码不应该再从这里导入,看到别人代码这样写基本可以判断是早期迁移期间遗留下来的。

实测:在本系列锁定的 TF 2.21 环境下,构造或在 `model.fit()` 里使用 `tf.keras.optimizers.legacy.Adam` **都不会**触发任何 deprecation warning——用 `warnings.catch_warnings(record=True)` 包住构造和一次完整的 `fit()` 调用,捕获到的 warning 列表是空的。这说明至少现在这个版本,legacy 命名空间处于"完整可用、没有被强制警告或计划移除"的状态,不代表可以长期放心依赖——迁移方向很明确是"新代码用新 API",legacy 命名空间存在的唯一理由是给"新架构下极端情况数值和旧版本对不上"(比如分布式梯度聚合方式变化导致的浮点误差路径不同)的存量脚本留一条退路。

**AI 研究/工程场景:** 接手一份两三年前写的 TF 训练脚本,发现从 checkpoint 恢复后继续训练,loss 曲线和当年记录的形状对不上。排查会发现:当年这行 `tf.keras.optimizers.Adam(...)` 代码指向的还是旧 `OptimizerV2` 实现,现在同一行代码因为 TF 版本升级已经默认指向新架构,两套实现在分布式梯度聚合、EMA 等细节上不是逐位对齐的。唯一稳妥的处理方式是训练脚本里显式写清楚要用哪一套——需要严格复现旧结果就显式导入 `tf.keras.optimizers.legacy.Adam`,不要依赖"某个名字默认指向谁"这种会随 TF 版本悄悄变化的隐式行为。

**可运行例子:**
```python
import inspect
import tensorflow as tf

new_adam_mro = [c.__name__ for c in tf.keras.optimizers.Adam.__mro__]
legacy_adam_mro = [c.__name__ for c in tf.keras.optimizers.legacy.Adam.__mro__]

assert new_adam_mro == ["Adam", "Optimizer", "_BaseOptimizer", "AutoTrackable", "Trackable", "object"]
assert legacy_adam_mro == ["Adam", "OptimizerV2", "Trackable", "object"]
assert "Optimizer" not in legacy_adam_mro          # 两条继承链完全不相交
assert "OptimizerV2" not in new_adam_mro
assert tf.keras.optimizers.legacy.Adam is not tf.keras.optimizers.Adam

# weight_decay 是新架构基类通用能力,legacy 构造函数签名里完全没有这一项(第4节详细展开)
new_params = set(inspect.signature(tf.keras.optimizers.Adam.__init__).parameters)
legacy_params = set(inspect.signature(tf.keras.optimizers.legacy.Adam.__init__).parameters)
assert "weight_decay" in new_params
assert "weight_decay" not in legacy_params
print("新架构 Adam 构造参数个数:", len(new_params), " legacy Adam 构造参数个数:", len(legacy_params))
# 实测: 新架构 17 个、legacy 8 个(不含 **kwargs 展开),差距印证"横切能力收进基类"这条结论
```

**面试怎么问 + 追问链:**
- **Q:** "`tf.keras.optimizers` 现在默认是新版实现还是老版实现?" —— 期望能说出"新版(2.11 前后重写过),旧版完整保留在 `.legacy` 命名空间"。
- **追问 1(容易说串):** "这个新旧和 Keras 2/3 的分裂是同一件事吗?" —— 期望明确答"不是,是两条独立的历史分裂,只是都被泛称 legacy";答不出这一点说明只是背了个"有 legacy"的印象,没有真正搞清楚现状。
- **追问 2(深挖):** "新架构主要解决了旧架构的什么问题?" —— 期望提到"把分布式聚合、EMA、梯度裁剪这类横切能力从每个子类各自实现,收敛到基类统一实现",能举出 `weight_decay`/`clipnorm` 这些新基类通用参数作为具体证据是加分项。
- **追问 3(工程向):** "如果要保证和几年前训练出的 checkpoint 严格数值对齐,应该用哪一套?" —— `tf.keras.optimizers.legacy`。

**常见坑:** 把"`tf.keras.optimizers` 的新旧实现"和"Keras 2 vs Keras 3(tf_keras vs keras 3)"这两条独立的历史分裂混为一谈,面试时会显得没有真正理解、只是背了个"TF 有 legacy 概念"的模糊印象。另外容易忽略 `tf.keras.optimizers.experimental` 这个过渡命名空间的存在——如果在旧代码里看到从这里导入优化器,要知道这是重构期间的历史写法,不是还有第三套现役实现。

---

## 2. `apply_gradients` 机制——与 torch `optimizer.step()` 的 API 设计差异

**是什么:**
```
optimizer.apply_gradients(
    grads_and_vars,             # [(gradient, variable), (gradient, variable), ...] 的列表
    name=None,
    skip_gradients_aggregation=False,
    **kwargs,
)
```
*(以上是签名示意,`optimizer` 代表任意已构造好的优化器实例,不是可执行代码——完整可运行版本见本节"可运行例子")*

**一句话:** TF 优化器不会自己"知道"哪个梯度对应哪个变量——这个映射关系必须由调用者显式打包成 `(gradient, variable)` 元组的列表传进来;这和 PyTorch `optimizer.step()` 形成鲜明对比:`step()` 不接收任何梯度参数,因为梯度早在 `.backward()` 时就已经被 autograd 当作副作用写到了每个叶子 tensor 自己的 `.grad` 属性上,`step()` 只需要遍历构造时就记住的 `param_groups`、读各自的 `.grad`(torch06 第 7 节已经现场读过这段源码)。

**底层机制/为什么这样设计:**

差异的根源在于两边求梯度的 API 本身就不对称。PyTorch `loss.backward()` 没有返回值,是通过"副作用"把梯度写回每个 `requires_grad=True` 叶子 tensor 的 `.grad` 属性;TF `tape.gradient(loss, sources)` 是一次纯函数调用,**返回**一个和 `sources` 结构对应的梯度列表,不会自动写到任何地方——`tf.Variable` 本身没有 `.grad` 这种属性。既然梯度是"返回值"而不是"写到某个属性上的副作用",那"梯度和变量的配对关系"就必须靠返回值列表和输入列表的顺序对应,`apply_gradients` 收到的 `zip(grads, vars)` 正是把这份对应关系重新显式打包成数据传进去的方式——这不是 TF 优化器"设计得更啰嗦",而是 `GradientTape` 本身"记录一批 source、批量返回梯度"的求梯度模型的直接推论。

验证两种"梯度为 `None`"的边界情况——这是本节验证重点,也是实践中最容易踩的坑:

```python
import tensorflow as tf

# 场景A:部分变量的梯度是 None(这次前向根本没用到某个变量)
w2 = tf.Variable([5.0], dtype=tf.float32)
w3 = tf.Variable([9.0], dtype=tf.float32)   # 不参与本次 loss 计算
with tf.GradientTape() as tape:
    loss = tf.reduce_sum(w2 ** 2)
grads = tape.gradient(loss, [w2, w3])
assert grads[0] is not None and grads[1] is None

opt = tf.keras.optimizers.SGD(learning_rate=0.1)
opt.apply_gradients(zip(grads, [w2, w3]))   # 不报错,w3 被静默跳过(会有 WARNING,见"常见坑")
assert abs(float(w2.numpy()[0]) - 4.0) < 1e-6   # 5.0 - 0.1*10.0 = 4.0
assert float(w3.numpy()[0]) == 9.0              # 完全没变

# 场景B:全部变量的梯度都是 None
w4 = tf.Variable([1.0], dtype=tf.float32)
with tf.GradientTape() as tape:
    loss4 = tf.constant(5.0)   # 和 w4 毫无关系
grads4 = tape.gradient(loss4, [w4])
assert grads4[0] is None

opt4 = tf.keras.optimizers.SGD(learning_rate=0.1)
try:
    opt4.apply_gradients(zip(grads4, [w4]))
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "No gradients provided for any variable" in str(e)
    print("ValueError:", e)
```

两种情况走的是完全不同的两条代码路径,不能想当然认为"梯度是 `None` 就一律安全跳过"这一条规则能覆盖所有情况:**部分**变量梯度为 `None` 时静默跳过(只是打印一条 WARNING);**全部**变量梯度为 `None` 时直接抛 `ValueError`。对比 PyTorch:torch06 第 7 节验证过,`optimizer.step()` 的过滤条件是 `p.grad is not None`,不管是部分参数没梯度还是全部参数没梯度,都是同一条"静默跳过、不报错也不警告"的逻辑,不区分这两种边界情况——这是两套实现里一个具体、可验证的行为差异,不是"谁更好",是 TF 在"全部梯度都没有"这个大概率意味着用户写错代码的信号上,选择了更严格的防御性检查。

`minimize()` 是 `tape.gradient()` + `apply_gradients()` 的语法糖(签名 `minimize(self, loss, var_list, tape=None)`):没传 `tape` 就自己开一个 `GradientTape` 包住 `loss` 的求值,对 `var_list` 求梯度,再调用 `apply_gradients`。多数教程示例直接用 `minimize()`,但理解 `apply_gradients` 才是理解"TF 优化器到底需要什么输入"的关键——`Model.fit()`(08 篇会展开)内部的 `train_step` 最终也是落到 `apply_gradients` 这一层收口。

**AI 研究/工程场景:** 自定义训练循环(强化学习、GAN 交替优化、对抗训练这类不适合直接用 `model.fit()` 的场景)是 `apply_gradients` 的主战场。从 PyTorch 转 TF 写这类循环时,最容易踩的第一个坑就是这里:PyTorch 写惯了 `loss.backward(); optimizer.step()` 两行,完全不用操心"梯度对应哪个参数"这件事;搬到 TF 下意识忘记给 `tape.gradient` 传对 `sources` 列表,或者两个列表的顺序没对齐——这不会报语法错误,只会在运行时发现梯度是 `None`,或者更新到了错误的变量上,排查成本比一个直接的报错高得多。

**可运行例子:**
```python
import tensorflow as tf

w = tf.Variable([1.0, 2.0], dtype=tf.float32)
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
with tf.GradientTape() as tape:
    loss = tf.reduce_sum(w ** 2)          # d(loss)/dw = 2w = [2, 4]
grads = tape.gradient(loss, [w])
grads_and_vars = list(zip(grads, [w]))    # 显式打包成 (grad, var) 对的列表

assert grads_and_vars[0][0].numpy().tolist() == [2.0, 4.0]
assert grads_and_vars[0][1] is w   # 第二个元素就是变量本身的引用,不是拷贝

opt.apply_gradients(grads_and_vars)
assert tf.reduce_all(tf.abs(w - tf.constant([0.8, 1.6])) < 1e-6)   # [1,2] - 0.1*[2,4] = [0.8,1.6]
print("apply_gradients 之后 w =", w.numpy())
```

**面试怎么问 + 追问链:**
- **Q:** "TF 的 `apply_gradients` 和 PyTorch 的 `optimizer.step()` 设计上最大的区别是什么?" —— 期望答"显式传入 `(grad, var)` 对 vs 隐式读取 `.grad` 属性",并能说出根源在于 `tape.gradient()` 是返回值而不是写 `.grad` 的副作用。
- **追问 1:** "如果 `grads_and_vars` 里某个变量的梯度是 `None`,会发生什么?" —— 期望答"不报错,静默跳过,但有 WARNING";能进一步说出"全部是 `None` 才会报 `ValueError`"是加分项,说明真的验证过边界而不是猜。
- **追问 2(工程向):** "写自定义训练循环时,新手最容易在这一步踩什么坑?" —— `tape.gradient` 的 `sources` 列表和 `apply_gradients` 的 `vars` 列表顺序没对齐,或者漏传变量,且这类错误不会在语法层面暴露。
- **追问 3:** "`minimize()` 内部做了什么?" —— `tape.gradient` + `apply_gradients` 的封装,没传 `tape` 会自己开一个。

**常见坑:**
- 把"部分梯度是 `None` 会被静默跳过"错误推广成"`None` 梯度永远安全",忽略"全部是 `None` 会直接报 `ValueError`"这条不同的路径。
- 部分梯度为 `None` 时真实触发的 WARNING 原文(已现场触发抄录):`WARNING:tensorflow:Gradients do not exist for variables ['Variable:0'] when minimizing the loss. If you're using \`model.compile()\`, did you forget to provide a \`loss\` argument?`——这条文案的措辞历史上是给 `compile()`/`minimize()` 路径写的,即使你根本没用过 `compile()`,直接调 `apply_gradients` 触发的也是同一条通用文案,不要被"`model.compile()`"字样误导,以为自己一定是用错了别的 API。
- 全部梯度为 `None` 时真实触发的 `ValueError` 原文(已现场触发抄录):`ValueError: No gradients provided for any variable: (['Variable:0'],). Provided \`grads_and_vars\` is ((None, <tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([1.], dtype=float32)>),).`
- 忘记 `GradientTape` 默认只自动 watch **trainable 的 `tf.Variable`**(下一篇 GradientTape 专题详细展开),普通 `tf.constant` 或 `trainable=False` 的 Variable 不会被自动追踪,`tape.gradient` 对应位置会直接返回 `None`,容易和"变量确实没参与计算"这种情况混淆,排查方向完全不同。

---

## 3. `LearningRateSchedule`:可调用对象,而不是命令式 `.step()` 调度器

**是什么:**
```
class MySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __call__(self, step): ...   # 必须实现:根据 step 返回这一步的学习率
    def get_config(self): ...       # 必须实现:可序列化契约

optimizer = tf.keras.optimizers.Adam(learning_rate=MySchedule(...))
```
*(以上是签名示意,`...` 代表省略的方法体,不是可执行代码——完整可运行版本见本节"可运行例子")*

**一句话:** `LearningRateSchedule` 是一个"可调用对象"(实现了 `__call__(self, step)` 的类实例),直接作为 `learning_rate=` 参数传给优化器构造函数,优化器内部每一步用自己的 `iterations` 计数器去调用它、现场算出当前学习率——这是完全**声明式**的设计,你只描述"lr 是 step 的什么函数",不需要在训练循环里手动调用任何"推进一步"的方法。PyTorch 的 `lr_scheduler`(torch06 第 9 节)走的是**命令式**路线:调度器对象自己持有可变状态,必须在训练循环里手动调用 `scheduler.step()` 才能推进它的内部状态、进而修改 `optimizer.param_groups[i]["lr"]` 这个数字。

**底层机制/为什么这样设计:**

先用实测拆穿一个非常容易想当然的误解——很多人以为"把 schedule 传给 optimizer 之后,`optimizer.learning_rate` 就是这个 schedule 对象本身":

```python
import tensorflow as tf

schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1, decay_steps=10, decay_rate=0.5)
opt = tf.keras.optimizers.SGD(learning_rate=schedule)

assert (opt.learning_rate is schedule) == False   # 不是同一个对象!
assert type(opt.learning_rate).__name__ == "ResourceVariable"   # 拿到的是一个 Variable
print(type(opt.learning_rate).__name__, float(opt.learning_rate.numpy()))
```
实测:`opt.learning_rate is schedule` 是 `False`,`type(opt.learning_rate)` 是 `ResourceVariable`,不是 schedule 对象。

真相是优化器内部实际持有**两份**和学习率相关的状态,职责分开(通过读私有属性验证):
- `opt._learning_rate`:**原样保存你传入的 schedule 对象本身**——这是真正驱动计算的"事实来源",每一步都会重新调用 `self._learning_rate(self.iterations)` 现场求值。
- `opt._current_learning_rate`(公开属性 `opt.learning_rate` 访问到的就是它):一个 `ResourceVariable`,只是"最近一次实际生效的学习率数值"的**只读快照缓存**,存在的目的是给 checkpoint 保存、日志打印这类需要"读一个具体数字"的场景用,不是计算的驱动源。

```python
import tensorflow as tf

schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1, decay_steps=10, decay_rate=0.5)
opt = tf.keras.optimizers.SGD(learning_rate=schedule)

assert type(opt._learning_rate).__name__ == "ExponentialDecay"   # 私有属性里存的才是 schedule 本体

w = tf.Variable([1.0], dtype=tf.float32)
lrs_used = []
for step in range(3):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(w ** 2)
    grads = tape.gradient(loss, [w])
    lr_before_this_step = float(schedule(opt.iterations))   # 本次 apply_gradients 实际会用的 lr
    opt.apply_gradients(zip(grads, [w]))
    lrs_used.append(lr_before_this_step)
    # opt.learning_rate 读到的,正是"刚刚已经生效"的那个值——快照的更新时机紧跟在 apply_gradients 之后
    assert abs(float(opt.learning_rate.numpy()) - lr_before_this_step) < 1e-6

print("每步实际生效的 lr:", [round(x, 6) for x in lrs_used])
# 实测: [0.1, 0.093303, 0.087055] —— 严格按 ExponentialDecay(0.1, decay_steps=10, decay_rate=0.5) 的公式衰减
```

`opt.learning_rate` 在任意时刻读到的都是"刚刚已经生效的那个值",不是"下一步将要用的值"——这个"读到的是过去时"的语义如果不知道,在训练循环里读 `opt.learning_rate` 做日志/调试容易和预期对不上(尤其是在 `apply_gradients` **之前**读它,读到的其实是上一步的值,不是这一步即将使用的值)。

`iterations` 是一个 `int64` 的 `tf.Variable`,是 schedule 机制"自己知道现在第几步"的唯一依据——由优化器自己在每次 `apply_gradients` 里自增,不需要用户像 PyTorch scheduler 那样每个 epoch/batch 手动调用什么方法去推进它。也因为 `iterations` 本身就是一个 `tf.Variable`,它和模型参数一样会被自动纳入 `tf.train.Checkpoint`——"训练到第几步"这件事天然具备断点续训的可恢复性,不需要额外手动保存一个 step 计数再手动喂回调度器。

**两种设计哲学的差异,这是本节的核心结论:** PyTorch 的 `lr_scheduler` 之所以要命令式手动 `.step()`,是因为调度器本身不知道、也不关心训练循环的节奏(按 batch 还是按 epoch 调用由你决定——torch06 第 9 节的常见坑正是"StepLR/CosineAnnealingLR 该按 epoch 调、OneCycleLR 该按 batch 调,调用粒度搞混会导致曲线形状完全不对");TF 的 `LearningRateSchedule` 把"步数"直接做成 `__call__` 的显式参数,"当前第几步"这件事被优化器自己的 `iterations` 计数器接管,不再需要使用者在正确时机手动敲一下"前进"。**代价**是:如果训练节奏本身特殊(某些 step 会跳过 `apply_gradients`、或者一个逻辑"step"里会调用多次 `apply_gradients`),`iterations` 的自增节奏会和你脑子里的"训练进度"脱节,这时候反而不如 PyTorch scheduler 那种"你自己决定什么时候 `.step()`"来得直观可控。

**AI 研究/工程场景:** Transformer/LLM 训练里常见的"warmup + 余弦衰减"曲线(呼应 torch06 第 9 节),在 TF 下的标准写法就是继承 `LearningRateSchedule` 自己写一个 `__call__`(前 N 步线性插值、之后余弦衰减)。整个调度逻辑被封装成一个无状态的纯函数对象,可以脱离具体某一次训练直接单元测试(直接断言 `schedule(1000)` 应该等于多少),不需要真的搭一个训练循环跑一遍才能验证曲线对不对——调度逻辑和训练循环执行是完全解耦的两块代码,这是声明式设计在工程上的实际好处。

**可运行例子(自定义子类,验证声明式调用不需要手动推进状态):**
```python
import tensorflow as tf

class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.base_lr * tf.minimum(1.0, (step + 1) / self.warmup_steps)

    def get_config(self):
        return {"base_lr": self.base_lr, "warmup_steps": self.warmup_steps}

sched = LinearWarmup(base_lr=0.1, warmup_steps=5)
# 纯函数,不需要任何训练循环就能直接断言曲线形状
manual = [round(float(sched(s)), 4) for s in range(7)]
assert manual == [0.02, 0.04, 0.06, 0.08, 0.1, 0.1, 0.1]

# 传给 optimizer 之后,完全不需要手动调用任何"推进"方法
opt = tf.keras.optimizers.Adam(learning_rate=sched)
w = tf.Variable([1.0, 2.0], dtype=tf.float32)
for step in range(7):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(w ** 2)
    grads = tape.gradient(loss, [w])
    opt.apply_gradients(zip(grads, [w]))   # 没有任何 scheduler.step() 这一行

assert int(opt.iterations.numpy()) == 7   # iterations 全靠 apply_gradients 自己推进
print("final w:", w.numpy())
```

**面试怎么问 + 追问链:**
- **Q:** "TF 的学习率调度和 PyTorch 的 `lr_scheduler` 设计上有什么本质区别?" —— 期望答"声明式 callable(TF)vs 命令式 `.step()`(PyTorch)"。
- **追问 1(陷阱题,区分度很高):** "把 schedule 传给 optimizer 之后,`optimizer.learning_rate` 拿到的是不是这个 schedule 对象?" —— 大部分人会直觉回答"是",正确答案是"不是",拿到的是一个缓存了"最近一次生效学习率数值"的 `ResourceVariable` 快照;真正的 schedule 对象存在私有属性 `_learning_rate` 里——能说出这一点说明真的动手验证过,不是只看过文档示例。
- **追问 2:** "TF 这种设计下,'现在训练到第几步'是谁在维护?" —— optimizer 自己的 `iterations` 变量,随每次 `apply_gradients` 自动自增,并会被 checkpoint 一并保存,不需要用户额外管理。
- **追问 3(开放):** "这种声明式设计在什么场景下反而不如 PyTorch 的命令式调度器灵活?" —— 训练节奏特殊(不是每个逻辑 step 都恰好调用一次 `apply_gradients`)时,`iterations` 的自增节奏可能和你想要的"进度"语义脱节,这时候没法像 PyTorch scheduler 那样自己精确控制什么时候该推进。

**常见坑:** 以为 `optimizer.learning_rate` 就是传进去的 schedule 对象本身、可以直接当函数调用拿到"下一步"的 lr——实际拿到的是数值快照 Variable,而且是"上一次生效值"不是"即将生效值"。如果需要在训练循环里主动查看"下一步会用的 lr",应该自己调用 `schedule(optimizer.iterations)`,不要猜测 `optimizer.learning_rate` 的语义。另外,`LearningRateSchedule.get_config()` 是 `@abc.abstractmethod` 标记的方法,但基类本身没有用 `ABCMeta` 强制,子类不实现它**不会在实例化时报错**,也不影响正常训练——只有在真正需要序列化(比如把 schedule 存进 `get_config()` 链路)时才会在调用到那一刻报 `NotImplementedError`,这种"定义时不报错、用到才报错"的迟延特性容易让人误以为自己已经写对了。

---

## 4. weight decay 在 Adam / AdamW 中的实现差异

**是什么:**
```
tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=None)     # 新架构:Adam 也有 weight_decay,默认关闭
tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.004)   # 独立的 AdamW 类,weight_decay 默认开启且必填
```
*(以上是签名示意——两行都是合法调用,完整可运行验证见本节"可运行例子")*

**一句话:** torch06 第 6 节验证过,PyTorch 里 `Adam(weight_decay=)` 是把 weight decay 项混进梯度(L2 风格,会被 Adam 的自适应缩放"扭曲");`AdamW(weight_decay=)` 才是解耦、均匀收缩。**TF 完全不是这个格局**——实测证实 TF 新架构下 `Adam(weight_decay=)` 和 `AdamW(weight_decay=)` 用的是**同一套**在基类 `Optimizer` 里实现一次的解耦收缩逻辑,数值上完全等价;TF 里没有"weight_decay 参数会被自适应缩放污染"这种坑。

**底层机制/为什么这样设计(源码确认 + 隔离实验精确验证独立性):**

先读构造函数签名,确认三个类的 `weight_decay` 语义不对称:
```python
import inspect
import tensorflow as tf

print("Adam       :", inspect.signature(tf.keras.optimizers.Adam.__init__))
print("AdamW      :", inspect.signature(tf.keras.optimizers.AdamW.__init__))
print("legacy.Adam:", inspect.signature(tf.keras.optimizers.legacy.Adam.__init__))
print("AdamW 是 Adam 的子类吗:", issubclass(tf.keras.optimizers.AdamW, tf.keras.optimizers.Adam))
```
实测输出(节选关键差异):
```
Adam       : (..., weight_decay=None, clipnorm=None, ...)      # 默认 None = 关闭
AdamW      : (..., weight_decay=0.004, clipnorm=None, ...)     # 默认非 None = 开启,且不接受 None
legacy.Adam: (self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam', **kwargs)
AdamW 是 Adam 的子类吗: False
```
`AdamW` **不是** `Adam` 的子类——这一点和 torch06 读到的 PyTorch 源码结论完全相反:torch06 第 6 节明确验证过"PyTorch 的 `AdamW` 类本身没有独立实现,字面上就是 `Adam.__init__(..., decoupled_weight_decay=True)`",是父子关系;TF 里 `Adam` 和 `AdamW` 是**平级的兄弟类**,都直接继承自同一个 `Optimizer` 基类。而且 `legacy.Adam`(重写之前的旧实现)**压根没有 `weight_decay` 这个构造参数**——旧架构下想要 L2 风格的正则化,只能在层级别用 `kernel_regularizer=tf.keras.regularizers.l2(...)`(会被计入 loss、通过反向传播混进梯度,这才是 TF 里概念上最接近 PyTorch `Adam(weight_decay=)` L2 风格的东西,但它是一个完全独立的机制,不是优化器的构造参数)。

既然 `Adam` 和 `AdamW` 是平级兄弟类而不是父子,那 `weight_decay` 这个能力必然是在共同的父类 `Optimizer` 里实现的。读源码确认:

```python
import inspect
import tensorflow as tf
print(inspect.getsource(tf.keras.optimizers.Optimizer._apply_weight_decay))
```
实测输出(核心逻辑):
```python
def _apply_weight_decay(self, variables):
    if self.weight_decay is None:
        return
    def distributed_apply_weight_decay(distribution, variables, **kwargs):
        def weight_decay_fn(variable):
            if self._use_weight_decay(variable):
                lr = tf.cast(self.learning_rate, variable.dtype)
                wd = tf.cast(self.weight_decay, variable.dtype)
                variable.assign_sub(variable * wd * lr)   # 直接对变量做乘法收缩,不看 grad/m/v
        for variable in variables:
            distribution.extended.update(variable, weight_decay_fn, group=False)
    tf.__internal__.distribute.interim.maybe_merge_call(...)
```
`variable.assign_sub(variable * wd * lr)` 等价于 `param = param * (1 - lr*wd)`——这一行**完全不引用梯度、不引用一阶矩 `m`、不引用二阶矩 `v`**,和 torch06 第 6 节读到的 PyTorch `AdamW` 解耦公式 `param.mul_(1 - lr * weight_decay)` 是同一个闭式表达式,只是代码组织方式不同:PyTorch 是 `AdamW` 子类独有的分支,TF 是**所有**暴露 `weight_decay` 的优化器共享的基类方法——这意味着 TF 新架构里,任何优化器(不只是 Adam/AdamW,SGD、RMSprop 只要暴露了 `weight_decay` 参数)拿到的都是同一种解耦实现,不存在"某个优化器的 weight_decay 是 L2 风格、另一个是解耦风格"这种分裂。

**隔离实验,直接验证"和 v 无关"这个结论,而不是停留在读源码的文字描述:** 用 `optimizer.build()` 手动建好 slot 变量后,直接把二阶矩 `v` 改写成相差一百万倍的两个值,再做一次梯度恰好为 0 的纯 decay 步,看两者的衰减量是否一致:

```python
import tensorflow as tf

w_big_v = tf.Variable([1.0], dtype=tf.float32)
w_small_v = tf.Variable([1.0], dtype=tf.float32)
opt_a = tf.keras.optimizers.Adam(learning_rate=0.1, weight_decay=0.5, epsilon=1e-8)
opt_b = tf.keras.optimizers.Adam(learning_rate=0.1, weight_decay=0.5, epsilon=1e-8)
opt_a.build([w_big_v])
opt_b.build([w_small_v])

opt_a._velocities[0].assign([1e6])    # 人为把二阶矩 v 改成巨大值
opt_b._velocities[0].assign([1e-6])   # 人为把二阶矩 v 改成极小值(相差一百万倍)
assert float(opt_a._momentums[0].numpy()[0]) == 0.0   # 一阶矩 m 仍是 0(全程没有真实梯度流过)

# 施加一次梯度严格为 0 的更新 —— 此时 Adam 自身基于 m_hat 的更新项精确为 0(因为 m=0),
# 观测到的变量变化如果不为 0,只能来自 weight decay
opt_a.apply_gradients([(tf.constant([0.0]), w_big_v)])
opt_b.apply_gradients([(tf.constant([0.0]), w_small_v)])

delta_a = 1.0 - float(w_big_v.numpy()[0])
delta_b = 1.0 - float(w_small_v.numpy()[0])
expected = 1.0 * 0.1 * 0.5   # 闭式解:param * lr * wd

print(f"v=1e6  时 decay 量: {delta_a:.6f}")
print(f"v=1e-6 时 decay 量: {delta_b:.6f}")
print(f"闭式解 param*lr*wd = {expected}")

assert abs(delta_a - expected) < 1e-6
assert abs(delta_b - expected) < 1e-6
assert delta_a == delta_b   # v 相差一百万倍,decay 量完全相等,精确到浮点位
```
实测:`v=1e6` 和 `v=1e-6`(相差一百万倍)两种情况下,decay 量都精确等于 `0.05`,和二阶矩 `v` 的取值**完全无关**——这和 torch06 第 6 节对 PyTorch `Adam(weight_decay=)` 做的同类隔离实验形成直接对照:torch06 那边測得历史梯度大小不同的两个参数,L2 风格 weight decay 的实际力度能相差约 1000 倍;TF 这边不管 `Adam` 还是 `AdamW`,只要设了 `weight_decay`,力度都是均匀的,不受 `v` 影响。

`weight_decay` 还配了一个排除机制,呼应 torch06 第 8 节里"bias/LayerNorm 参数通常不做 weight decay"这条工程实践——PyTorch 靠手动拆分 `param_groups` 实现,TF 提供了一个内置方法:
```python
import inspect
import tensorflow as tf
print(inspect.signature(tf.keras.optimizers.Optimizer.exclude_from_weight_decay))
# (self, var_list=None, var_names=None) —— 可以传具体变量列表,也可以传变量名的正则
```

**AI 研究/工程场景:** Transformer/LLM 训练几乎清一色用 AdamW(torch06 第 6 节已经讲过原因:不同层梯度尺度差异巨大,L2 风格会被自适应缩放不均匀地"稀释")。把一个 PyTorch 训练脚本迁移到 TF 时,一个真实容易踩的坑恰恰和这里的发现相反方向——**在 PyTorch 里把 `Adam(weight_decay=0.01)` 改成 `AdamW(weight_decay=0.01)` 需要重新搜索超参**(torch06 第 6 节追问 3 已经讲过,两者实际衰减力度不同),但**在 TF 里做同样的替换,`Adam(weight_decay=0.01)` 换成 `AdamW(weight_decay=0.01)` 数值行为完全一致**(上面的隔离实验已验证),不需要重新调参——如果照搬"从 PyTorch 学到的经验"以为 TF 也需要重新搜索,纯属多余的工作量;反过来,如果从 TF 迁移到 PyTorch,直接照搬 `weight_decay` 数值给 `torch.optim.Adam` 而不是 `AdamW`,则会真实改变训练动态,这一步必须警惕。

**可运行例子(真实梯度、多步训练轨迹,而不是隔离实验里特意构造的单步零梯度):**
```python
import tensorflow as tf

x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
y = 2 * x

def loss_fn(w):
    return tf.reduce_mean((w * x - y) ** 2)

w_adam = tf.Variable([0.0], dtype=tf.float32)
w_adamw = tf.Variable([0.0], dtype=tf.float32)
opt_adam = tf.keras.optimizers.Adam(learning_rate=0.1, weight_decay=0.05)
opt_adamw = tf.keras.optimizers.AdamW(learning_rate=0.1, weight_decay=0.05)

for step in range(20):
    with tf.GradientTape() as tape:
        loss_a = loss_fn(w_adam)
    opt_adam.apply_gradients([(tape.gradient(loss_a, [w_adam])[0], w_adam)])

    with tf.GradientTape() as tape:
        loss_b = loss_fn(w_adamw)
    opt_adamw.apply_gradients([(tape.gradient(loss_b, [w_adamw])[0], w_adamw)])

    assert abs(float(w_adam.numpy()[0]) - float(w_adamw.numpy()[0])) < 1e-5, step

print("20 步真实训练后, Adam(weight_decay=0.05) 和 AdamW(weight_decay=0.05) 的参数值:",
      w_adam.numpy(), w_adamw.numpy())
# 实测:每一步的参数值逐位相等(diff 恒为 0.0),不只是隔离实验里的单步闭式解成立,
# 真实训练轨迹全程都成立
```

**面试怎么问 + 追问链:**
- **Q:** "TF 的 `Adam(weight_decay=)` 和 `AdamW(weight_decay=)` 有什么区别?" —— **这是一道容易被 PyTorch 经验带偏的题**,期望候选人不要直接照搬 torch06 的结论,而是能说出"TF 这两者数值完全一致,因为 weight decay 的解耦实现是在共同的 `Optimizer` 基类里做的,不是某个子类独有";如果候选人自信地照搬"Adam 是 L2、AdamW 才是解耦"的 PyTorch 结论,恰恰说明只会背题、没有意识到不同框架的实现细节不能想当然迁移。
- **追问 1(核心,必须能讲清楚机制):** "为什么 TF 里两者能做到完全一致?" —— 期望说出"AdamW 不是 Adam 的子类,是同级兄弟类,`weight_decay` 的实现在两者共同的父类 `Optimizer._apply_weight_decay` 里,是同一份代码,公式里不引用梯度/一阶矩/二阶矩"。
- **追问 2(工程向,承接场景题):** "把一个用了 `weight_decay` 的训练脚本从 PyTorch 迁移到 TF(或反过来),需要注意什么?" —— 期望说出"PyTorch 换 Adam/AdamW 数值不等价、需要重新搜索;TF 换 Adam/AdamW 数值等价、可以直接照搬——方向搞反了会引入一个隐蔽的超参数迁移 bug"。
- **追问 3:** "TF 的旧版 `tf.keras.optimizers.legacy.Adam` 有 `weight_decay` 参数吗?" —— 没有,旧架构下想要 L2 风格正则化只能用层级别的 `kernel_regularizer`,是完全独立的另一套机制。

**常见坑:** 把 torch06 第 6 节"Adam+L2 会被自适应缩放扭曲"的结论不加验证地平移到 TF,以为 TF 的 `Adam(weight_decay=)` 也存在同样的问题——本节的隔离实验已经精确证伪。另一个常见坑是 `AdamW` 的 `weight_decay` 参数**不接受 `None`**(和 `Adam` 不同,`Adam` 用 `None` 表示"关闭 weight decay"是合法默认值):
```python
import tensorflow as tf
try:
    tf.keras.optimizers.AdamW(learning_rate=0.1, weight_decay=None)
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "weight_decay" in str(e)
    print("ValueError:", e)
# 实测: ValueError: Missing value of `weight_decay` which is required and must be a float value.
```
如果想要"AdamW 类,但不做权重衰减",不能传 `weight_decay=None`,得显式传 `weight_decay=0.0`。

---

## 5. 优化器状态(slots)机制

**是什么:**
```
legacy_opt.get_slot_names()        # 旧架构:['m', 'v'] 这样的槽位名列表
legacy_opt.get_slot(var, "m")      # 旧架构:取出某个变量对应的动量/二阶矩状态
new_opt.variables                  # 新架构:一个包含所有状态变量的扁平列表
```
*(以上是签名示意,`legacy_opt`/`var`/`new_opt` 代表已构造好的优化器/变量实例,不是可执行代码——完整可运行版本见本节"可运行例子")*

**一句话:** TF 用"slot(槽位)"这个词描述每个可训练变量对应的优化器内部状态(动量、二阶矩估计等)——但这套以 `get_slot()`/`get_slot_names()` 为核心的公开查询 API,**只存在于 legacy 架构**;新架构(当前 `tf.keras.optimizers.Adam` 默认指向的实现)把它换成了一个扁平的 `optimizer.variables` 列表加内部按位置对齐的 Python 列表,不再提供"给我这个变量对应的动量"这种一步到位的公开方法。这和 PyTorch `optimizer.state`(一个直接以 `Parameter` 对象为键的字典)相比,是三种不同粒度的状态组织方式。

**底层机制/为什么这样设计(两套架构实测对比):**

先看 legacy 架构的 `get_slot` API,和"slots"这个术语名副其实:
```python
import tensorflow as tf

w = tf.Variable([1.0, 2.0], dtype=tf.float32, name="myvar")
legacy_opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.1)
with tf.GradientTape() as tape:
    loss = tf.reduce_sum(w ** 2)
grads = tape.gradient(loss, [w])
legacy_opt.apply_gradients(zip(grads, [w]))

assert legacy_opt.get_slot_names() == ["m", "v"]
m_slot = legacy_opt.get_slot(w, "m")
v_slot = legacy_opt.get_slot(w, "v")
print("m slot:", m_slot.numpy(), " v slot:", v_slot.numpy())
# 实测: m slot: [0.2  0.4] (=(1-beta1)*grad=(1-0.9)*[2,4)) v slot: [0.004 0.016] (≈(1-beta2)*grad^2)
assert abs(float(m_slot.numpy()[0]) - 0.2) < 1e-5
```
`get_slot(variable, slot_name)` 直接"以变量为键"取出它对应的某个状态张量,`get_slot_names()` 告诉你这个优化器每个变量身上挂了哪几种状态——这和 PyTorch `optimizer.state[param]["exp_avg"]`(torch06 里的 `m`)在心智模型上几乎一一对应,只是 PyTorch 用一个真正的 dict、TF (legacy) 用一对方法。

再看新架构——`get_slot`/`get_slot_names` 直接就不存在:
```python
import tensorflow as tf

w = tf.Variable([1.0, 2.0], dtype=tf.float32, name="myvar")
new_opt = tf.keras.optimizers.Adam(learning_rate=0.1)
assert hasattr(new_opt, "get_slot") == False
assert hasattr(new_opt, "get_slot_names") == False

with tf.GradientTape() as tape:
    loss = tf.reduce_sum(w ** 2)
grads = tape.gradient(loss, [w])
new_opt.apply_gradients(zip(grads, [w]))

for v in new_opt.variables:
    print(v.name, v.shape, v.numpy())
# 实测:
#   iteration:0        ()   1              <- 全局共享的 step 计数器,不属于任何一个变量
#   Adam/m/myvar:0      (2,) [0.2  0.4]     <- 命名里带上了 optimizer 名/槽位名/变量名
#   Adam/v/myvar:0      (2,) [0.004 0.016]
assert len(new_opt.variables) == 3
```
新架构把所有状态(包括那个全局共享的 `iterations` 计数器)摊平成一个列表,每个状态变量的 `name` 属性里编码了"属于哪个优化器/哪个槽位/哪个原始变量"这三层信息,但**不再提供**"给定一个 `tf.Variable`,直接查它的动量"这种一步到位的公开方法。读 `Adam.build()` 源码可以看到内部真正的存储结构:

```python
import inspect
import tensorflow as tf
print(inspect.getsource(tf.keras.optimizers.Adam.build))
```
关键片段:
```python
def build(self, var_list):
    ...
    self._momentums = []
    self._velocities = []
    for var in var_list:
        self._momentums.append(self.add_variable_from_reference(model_variable=var, variable_name="m"))
        self._velocities.append(self.add_variable_from_reference(model_variable=var, variable_name="v"))
```
`self._momentums`/`self._velocities` 是两个**按位置对齐**的 Python 列表,下标和 `var_list`(优化器第一次 `build` 时见到的可训练变量顺序)一一对应,内部靠 `self._index_dict`(把变量的唯一标识映射到列表下标)做查找,不是像 legacy 架构那样直接用"变量 → 状态张量"的字典。这是一个纯粹的内部实现选择(数组按位置索引通常比字典查找有更好的缓存局部性、也更方便做向量化的分布式聚合),代价是公开 API 层面"给定一个变量直接查它的状态"这件事从一等公民变成了需要自己翻 `.variables` 列表按命名约定去匹配。

**AI 研究/工程场景:** 调试一个训练不收敛的模型时,想现场检查"某一层的 Adam 二阶矩是不是已经炸了"(比如怀疑某层梯度长期偏大,`v` 异常大导致这层的有效学习率被压得过低)。用 legacy 优化器可以直接 `opt.get_slot(layer.kernel, "v")` 一步拿到;换成新架构默认的优化器,得自己遍历 `opt.variables`,按名字里的 `v/{variable_name}` 模式做字符串匹配,或者自己维护一份"变量 → 下标"的映射去对 `opt._velocities` 取值(后者是私有属性,官方不保证跨版本稳定)——这是一个从"新架构性能/实现更好"换来的"调试便利性下降"的真实取舍,做训练诊断工具时需要知道这个差异,不能想当然沿用老版本的 `get_slot` 写法。

**可运行例子(legacy 和新架构对同一个问题分别求解,验证 API 变了、数学没变):**
```python
import tensorflow as tf

w_legacy = tf.Variable([2.0, -1.0], dtype=tf.float32)
w_new = tf.Variable([2.0, -1.0], dtype=tf.float32)
grad_value = tf.constant([0.5, -0.3])

legacy_opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
new_opt = tf.keras.optimizers.Adam(learning_rate=0.01)
legacy_opt.apply_gradients([(grad_value, w_legacy)])
new_opt.apply_gradients([(grad_value, w_new)])

m_from_legacy = legacy_opt.get_slot(w_legacy, "m").numpy()
m_from_new = [v for v in new_opt.variables if "/m/" in v.name][0].numpy()

diff_m = abs(m_from_legacy - m_from_new).max()
diff_w = abs(w_legacy.numpy() - w_new.numpy()).max()
print("m 值差异:", diff_m, " 参数值差异:", diff_w)

# 两套完全独立重写的代码路径,agree 到 float32 精度但不是逐位相同(呼应本系列反复强调的
# "float32 别用 == 比较"这条教训)——用 1e-5 这种符合浮点误差量级的容差,而不是假设精确相等
assert diff_m < 1e-5
assert diff_w < 1e-5
```

**面试怎么问 + 追问链:**
- **Q:** "TF 的优化器状态是怎么组织的,'slots'这个词具体指什么?" —— 期望答出"每个可训练变量对应一份优化器内部状态(比如 Adam 的 m/v),这在 legacy 架构里叫 slot,有 `get_slot`/`get_slot_names` 两个查询方法"。
- **追问 1(区分度高,考察是否用过最新 API):** "现在默认的 `tf.keras.optimizers.Adam` 还有 `get_slot` 方法吗?" —— 期望知道"没有了,新架构换成了扁平的 `optimizer.variables` 列表,`get_slot`/`get_slot_names` 是 legacy 架构独有的";答"有"说明还停留在旧版本的知识,没有用当前版本验证过。
- **追问 2:** "新架构里,给定一个具体的 `tf.Variable`,怎么找到它对应的动量状态?" —— 期望能说出"没有一步到位的公开方法,需要自己按 `optimizer.variables` 里的命名约定匹配,或者依赖内部的 `_momentums`/`_index_dict`(非公开、不保证稳定)"。
- **追问 3(对比):** "这和 PyTorch 的 `optimizer.state` 有什么区别?" —— 期望说出"PyTorch `optimizer.state` 是一个真正以 `Parameter` 对象为键的字典,`optimizer.state[p]['exp_avg']` 一步到位;TF legacy 用方法调用模拟类似效果,新架构则退化成扁平列表 + 内部按位置索引,三种状态组织的公开程度依次递减"。

**常见坑:** 直接拿 legacy 架构的 `get_slot`/`get_slot_names` 代码去用在新架构默认的 `tf.keras.optimizers.Adam` 上,`AttributeError` 报出来才发现方法压根不存在——由于本篇第 1 节讲过的现状,`tf.keras.optimizers.Adam` 现在默认就是新架构,这个坑比想象的更容易踩。另外,`optimizer.variables` 列表里混着 `iterations` 这个全局计数器和每个变量各自的 `m`/`v`,不要把 `len(optimizer.variables)` 直接当成"变量数 × 状态种类数"来解读,得先减掉那些全局共享的状态。

---

## 6. gradient clipping:`clipnorm` / `clipvalue` / `global_clipnorm`

**是什么:**
```
tf.keras.optimizers.SGD(learning_rate=0.1, clipnorm=1.0)          # 每个梯度张量各自按范数裁剪
tf.keras.optimizers.SGD(learning_rate=0.1, global_clipnorm=1.0)   # 所有梯度拼在一起算一次全局范数裁剪
tf.keras.optimizers.SGD(learning_rate=0.1, clipvalue=0.5)         # 逐元素裁剪到 [-0.5, 0.5]
```
*(以上是签名示意——三行都是各自合法的独立调用,注意:三个参数实际能否共存有一条重要的优先级规则,本节"底层机制"和"可运行例子"会现场揭晓,不要想当然认为三者可以随意组合)*

**一句话:** TF 把梯度裁剪做成优化器构造函数的参数,`apply_gradients` 内部自动应用,是纯声明式的;PyTorch 的 `clip_grad_norm_`(torch06 第 10 节)是一个独立函数,必须在 `loss.backward()` 之后、`optimizer.step()` 之前手动插入一行调用。**更关键的陷阱在语义上**:TF 的 `clipnorm`(不带 `global_` 前缀)是**逐张量**裁剪——每个变量的梯度各自算自己的范数、各自裁剪,互不影响;真正和 PyTorch `clip_grad_norm_`(所有参数梯度拼成一个向量算**一次**全局范数)语义对等的,是名字很容易被忽略的 `global_clipnorm`。

**底层机制/为什么这样设计(实测三种裁剪模式的真实语义差异):**

**`clipnorm`:逐张量,不是全局。** 构造两个梯度、范数分别为 5.0 和 1.0 的变量,设 `clipnorm=2.0`:
```python
import tensorflow as tf

w1 = tf.Variable(tf.ones(3), dtype=tf.float32)
w2 = tf.Variable(tf.ones(5), dtype=tf.float32)
g1 = tf.constant([3.0, 4.0, 0.0])            # 范数 = 5.0,超过阈值,会被裁剪
g2 = tf.constant([1.0, 0.0, 0.0, 0.0, 0.0])  # 范数 = 1.0,低于阈值,不受影响

opt = tf.keras.optimizers.SGD(learning_rate=1.0, clipnorm=2.0)   # lr=1 方便直接从 delta 读出实际生效的梯度
before1, before2 = w1.numpy().copy(), w2.numpy().copy()
opt.apply_gradients([(g1, w1), (g2, w2)])
applied_g1 = before1 - w1.numpy()
applied_g2 = before2 - w2.numpy()

norm1 = float((applied_g1 ** 2).sum() ** 0.5)
norm2 = float((applied_g2 ** 2).sum() ** 0.5)
print(f"g1 实际生效范数={norm1:.4f}  g2 实际生效范数={norm2:.4f}")
assert abs(norm1 - 2.0) < 1e-4     # g1 被裁到阈值
assert abs(norm2 - 1.0) < 1e-4     # g2 完全没变 —— 证明 clipnorm 是逐张量独立判断的
```
实测:`g1`(原始范数 5.0)被单独裁到范数 2.0,`g2`(原始范数 1.0,本来就在阈值内)完全不受影响——如果 `clipnorm` 是全局语义,`g2` 应该也会因为"和 g1 合在一起算的总范数超标"被牵连裁剪,但它没有,证明每个梯度张量是各自独立判断、独立裁剪的。

**`global_clipnorm`:才是 PyTorch `clip_grad_norm_` 的真正对应物。** 同样两个梯度,换成 `global_clipnorm=2.0`:
```python
import tensorflow as tf

w3 = tf.Variable(tf.ones(3), dtype=tf.float32)
w4 = tf.Variable(tf.ones(5), dtype=tf.float32)
g3 = tf.constant([3.0, 4.0, 0.0])             # 范数 5.0
g4 = tf.constant([1.0, 0.0, 0.0, 0.0, 0.0])   # 范数 1.0
# 拼在一起的全局范数 = sqrt(5.0^2 + 1.0^2) = sqrt(26) ≈ 5.099

opt2 = tf.keras.optimizers.SGD(learning_rate=1.0, global_clipnorm=2.0)
before3, before4 = w3.numpy().copy(), w4.numpy().copy()
opt2.apply_gradients([(g3, w3), (g4, w4)])
applied_g3 = before3 - w3.numpy()
applied_g4 = before4 - w4.numpy()

combined_norm = float(((applied_g3 ** 2).sum() + (applied_g4 ** 2).sum()) ** 0.5)
print(f"两个梯度合并后的全局范数={combined_norm:.4f}")
assert abs(combined_norm - 2.0) < 1e-3   # 全局范数被裁到 2.0,而不是每个各自裁到 2.0

# 两个梯度应该被同一个缩放系数等比例缩小(保方向,呼应 torch06 第10节 clip_grad_norm_ 的"全局"定义)
ratio3 = float(applied_g3[0] / g3.numpy()[0])
ratio4 = float(applied_g4[0] / g4.numpy()[0])
assert abs(ratio3 - ratio4) < 1e-4
```
实测:合并后的全局范数被精确裁到 `2.0`,且两个梯度被同一个缩放系数等比例缩小——这才是和 torch06 第 10 节"把所有参数梯度拼成一个向量算一次全局 L2 范数,超标就按统一比例整体缩小"完全对等的语义。**`clipnorm` 和 `global_clipnorm` 命名上只差一个前缀,是实践中极容易读错、用错的一对参数——而且下面会现场证明,三个裁剪参数底层根本不是"各自独立、可以叠加"的关系,而是一条互斥的优先级链。**

**`clipvalue`:逐元素裁剪,和"范数"无关。**
```python
import tensorflow as tf

w5 = tf.Variable(tf.zeros(4), dtype=tf.float32)
g5 = tf.constant([5.0, -5.0, 0.3, -0.3])
opt3 = tf.keras.optimizers.SGD(learning_rate=1.0, clipvalue=1.0)
before5 = w5.numpy().copy()
opt3.apply_gradients([(g5, w5)])
applied_g5 = before5 - w5.numpy()
print("clipvalue=1.0:", applied_g5)
assert list(applied_g5) == [1.0, -1.0, 0.3, -0.3]   # 每个分量独立裁到 [-1, 1],不保证方向不变
```
`clipvalue` 是把梯度张量的**每一个分量**独立裁剪到 `[-clipvalue, clipvalue]`,和 `clipnorm`/`global_clipnorm` 完全是两种机制——这一点上 TF 和 PyTorch 是对齐的:PyTorch 也有对应的 `clip_grad_value_`(torch06 第 10 节面试追问里提到过,和 `clip_grad_norm_` 常被搞混的另一个函数),只是 TF 把它做成了优化器构造参数,PyTorch 是独立函数。

三者(`clipnorm`/`clipvalue`/`global_clipnorm`)都是在优化器构造时一次性声明好,`apply_gradients` 内部自动应用,不需要在训练循环里手动插入任何一行裁剪代码——这是本篇第 7 节要收拢的"声明式"设计哲学在梯度裁剪这个具体机制上的体现。

**AI 研究/工程场景:** RNN/Transformer 训练中梯度爆炸是经典问题,`global_clipnorm`(而不是 `clipnorm`)才是训练大模型时通常想要的行为——如果误用了没有 `global_` 前缀的 `clipnorm`,裁剪效果会比预期宽松得多(层数越多,"逐张量各自裁到阈值内"允许的总体梯度"能量"上限相当于随层数线性放大,这和 torch06 第 10 节追问 1 里讨论的"逐张量裁剪 vs 全局裁剪"是同一个问题,只是 TF 这边把这两种语义做成了两个名字相近的参数,而不是像 PyTorch 那样只提供全局版本、要全局裁剪就没有第二种选择)。把 PyTorch 训练脚本迁移到 TF 时,`clip_grad_norm_(model.parameters(), max_norm=1.0)` 应该翻译成 TF 优化器的 `global_clipnorm=1.0`,翻译成 `clipnorm=1.0` 是一个真实的语义偷换,只有在模型只有一个参数张量时两者才会碰巧等价。

**可运行例子(读源码 + 现场验证三个裁剪参数的真实优先级关系,推翻"三者可以自由组合叠加"的直觉):**

先读 `_clip_gradients` 源码,它是本节最关键的一段代码:
```python
import inspect
import tensorflow as tf
print(inspect.getsource(tf.keras.optimizers.Optimizer._clip_gradients))
```
实测输出(核心逻辑):
```python
def _clip_gradients(self, grads):
    clipped_grads = []
    if self.clipnorm and self.clipnorm > 0:
        for g in grads:
            ...
            clipped_grads.append(tf.clip_by_norm(g, self.clipnorm))
        return clipped_grads                                    # <- 命中就直接返回

    if self.global_clipnorm and self.global_clipnorm > 0:
        return tf.clip_by_global_norm(grads, self.global_clipnorm)[0]   # <- 命中就直接返回

    if self.clipvalue and self.clipvalue > 0:
        for g in grads:
            ...
            clipped_grads.append(tf.clip_by_value(g, -self.clipvalue, self.clipvalue))
        return clipped_grads

    return grads
```
这是一条 `if`/`if`/`if` **优先级链**,不是三种裁剪依次叠加应用的管道:`clipnorm` 命中就直接 `return`,`global_clipnorm`、`clipvalue` 根本没机会执行;`global_clipnorm` 命中同理直接 `return`,轮不到 `clipvalue`。真实优先级是 **`clipnorm` > `global_clipnorm` > `clipvalue`**,只有最高优先级里"设了且大于 0"的那一个会生效,其余的会被完全跳过——即使它们也被设置了非零值。

现场验证,包括最容易被忽视的静默失效场景:
```python
import tensorflow as tf

# 1) clipnorm 和 global_clipnorm 同时设置 —— 构造时直接报错(源码在 __init__ 里有显式检查)
try:
    tf.keras.optimizers.SGD(learning_rate=1.0, clipnorm=10.0, global_clipnorm=6.0)
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "At most one of" in str(e)
    print("ValueError:", e)

# 2) clipvalue 和 clipnorm 同时设置 —— 不报错!但 clipvalue 被静默完全忽略
opt1 = tf.keras.optimizers.SGD(learning_rate=1.0, clipvalue=1.0, clipnorm=10.0)
w1 = tf.Variable([1.0, 1.0])
before1 = w1.numpy().copy()
opt1.apply_gradients([(tf.constant([5.0, 0.0]), w1)])   # 梯度范数=5.0 < clipnorm 阈值10.0,不会被 clipnorm 裁剪
applied1 = before1 - w1.numpy()
print("clipvalue=1.0 + clipnorm=10.0 时,实际生效梯度:", applied1)
assert list(applied1) == [5.0, 0.0]   # 如果 clipvalue 真的生效,这里该是 [1.0, 0.0] —— 但它没有

# 3) 单独设置 clipvalue 时行为正常,证明不是 clipvalue 本身坏了,是被 clipnorm 的存在抢走了优先级
opt2 = tf.keras.optimizers.SGD(learning_rate=1.0, clipvalue=1.0)
w2 = tf.Variable([1.0, 1.0])
before2 = w2.numpy().copy()
opt2.apply_gradients([(tf.constant([5.0, 0.0]), w2)])
applied2 = before2 - w2.numpy()
assert list(applied2) == [1.0, 0.0]
print("单独设置 clipvalue=1.0 时,实际生效梯度:", applied2, "(证明 clipvalue 本身没问题)")
```
`clipnorm`/`global_clipnorm` 这一对互斥关系,TF 在构造函数里显式检查、直接报错,相对安全;但 `clipvalue` 和它们俩任意一个的互斥关系**没有**任何构造时检查——同时设置完全合法,只是 `clipvalue` 会在静默中彻底失效,没有报错也没有警告,只能靠现场对比裁剪前后的梯度数值才能发现。

**面试怎么问 + 追问链:**
- **Q:** "TF 优化器的 `clipnorm` 参数和 PyTorch 的 `clip_grad_norm_` 是同一个东西吗?" —— **这是本节的陷阱题**,期望答"不是,`clipnorm` 是逐张量独立裁剪;真正语义对等的是 `global_clipnorm`"。如果候选人直接说"是",说明没有意识到两个框架同名概念下藏着真实的语义差异。
- **追问 1(验证是否真的测过):** "怎么证明 `clipnorm` 是逐张量而不是全局的?" —— 期望能提出"构造两个范数不同的梯度,一个超阈值一个不超,分别观察裁剪后的结果——全局语义下没超标的那个也会被牵连,逐张量语义下它会保持原样",也就是上面验证代码的思路。
- **追问 2(工程向):** "把一个 PyTorch 训练脚本的 `clip_grad_norm_(params, max_norm=1.0)` 迁移到 TF,应该用 `clipnorm` 还是 `global_clipnorm`?" —— `global_clipnorm`,用 `clipnorm` 是语义偷换,只有模型只有一个参数张量时才碰巧等价。
- **追问 3:** "TF 的梯度裁剪需要像 PyTorch 那样手动在 `backward()` 和 `step()` 之间插一行吗?" —— 不需要,是优化器构造参数,`apply_gradients` 内部自动应用,声明式而非命令式。
- **追问 4(隐藏杀伤力最强,真实事故的原型):** "如果我在同一个优化器上同时设置了 `clipvalue` 和 `clipnorm`,会发生什么?" —— 大部分人会猜"两种裁剪都生效"或者"报错",正确答案是**都不对**:构造完全不报错,但 `clipvalue` 会被静默地完全忽略——源码 `_clip_gradients` 是一条 `if/if/if` 优先级链(`clipnorm` > `global_clipnorm` > `clipvalue`),`clipnorm` 分支命中就直接 `return` 了,根本走不到 `clipvalue` 那一段代码。只有真正读过源码或者亲手踩过这个坑的人才答得上来。

**常见坑:** 把 `clipnorm` 当成 PyTorch `clip_grad_norm_` 的直接替代品,忽略两者"逐张量 vs 全局"的语义差异——这是本节验证的核心陷阱之一,模型只有一层/一个参数张量的玩具例子里两者表现一致,一旦模型有多层就会不知不觉裁剪过松。**更隐蔽的第二个陷阱**:`clipvalue` 和 `clipnorm`/`global_clipnorm` 中的任意一个同时设置,构造时**不会报错**,`clipvalue` 会被静默地完全忽略(见上面"可运行例子"的源码验证)——只有 `clipnorm`+`global_clipnorm` 这一对互斥关系会在构造时被显式检查报错,`clipvalue` 那一侧的互斥没有任何保护,"以为自己同时开了范数裁剪和逐元素裁剪双保险"是一个真实存在、没有任何报错提示的坑。这几个参数和第 4 节 `weight_decay`、第 3 节 `LearningRateSchedule` 之间谁先谁后生效,如果没读过 `apply_gradients` 的调用链源码,单凭直觉猜容易猜错——不确定的时候,直接读源码或写一个和本节一样的隔离实验现场验证,比凭印象回答可靠。

---

## 7. 与 PyTorch 优化器机制对比(收尾总结)

**是什么:** 本节没有单独的 API 要介绍——把前 6 节各自验证过的具体机制差异收拢成一张"设计哲学对照表",作为这一批的收尾。

**一句话:** 把前 6 节的发现放在一起看,会得到一个一以贯之的模式:TF 倾向于把训练过程中要用到的一切配置(学习率怎么随 step 变、权重要不要衰减、梯度要不要裁剪)在优化器**构造**这一刻就以参数形式声明完,后续每一步只有一行 `apply_gradients`,该发生的事情由框架在这一行内部自动打理;PyTorch 倾向于把每一件事拆成训练循环里**独立、显式**的一行命令式调用(`backward()`、`clip_grad_norm_()`、`step()`、`scheduler.step()`、`zero_grad()`——各管一段,顺序和是否执行完全由你亲手安排)。

**底层机制/为什么这样设计(六节证据汇总成表):**

| 机制 | TF(本篇对应节) | PyTorch(torch06 对应节) | 差异的本质 |
|---|---|---|---|
| 梯度怎么送进优化器 | `apply_gradients([(grad, var), ...])`,显式配对(第2节) | `optimizer.step()` 直接读 `.grad` 属性(torch06 §7) | `tape.gradient` 是返回值,`.backward()` 是写属性的副作用 |
| 学习率调度 | `LearningRateSchedule` 可调用对象,传给构造函数,内部自动按 `iterations` 求值(第3节) | `lr_scheduler`,需要手动在正确粒度调用 `.step()`(torch06 §9) | TF 把"第几步"收进 optimizer 自己的状态;PyTorch 把"何时推进"的控制权交给你 |
| weight decay | 所有暴露该参数的优化器共享同一份基类解耦实现,`Adam`/`AdamW` 数值等价(第4节) | `Adam`(L2,混入梯度)和 `AdamW`(解耦)是两种不同实现,数值不等价(torch06 §6) | TF 新架构把横切能力收进基类;PyTorch 保留了历史上两条并存的实现 |
| 优化器内部状态怎么查 | legacy 用 `get_slot`;新架构只有扁平的 `optimizer.variables`(第5节) | `optimizer.state[param]` 直接是以 Parameter 为键的字典(torch06 隐含贯穿全文) | PyTorch 状态字典的公开程度从 legacy TF 到新 TF 是逐步降低的 |
| 梯度裁剪 | 优化器构造参数 `clipnorm`/`global_clipnorm`/`clipvalue`,自动应用(第6节) | 独立函数 `clip_grad_norm_`/`clip_grad_value_`,必须手动插入调用(torch06 §10) | TF 声明式配置一次;PyTorch 每次训练循环手动调用 |

这种差异不是两个团队审美不同这么简单,能看出一点历史动机:TF 从 1.x 时代"先画完整张计算图,再灌数据跑"的心智模型一路走过来,即使 TF2 已经默认 eager 执行,"训练要用到的东西尽量在构建阶段一次性配置好"这种倾向仍然渗透在 API 设计里(优化器构造函数几乎变成了一个小型配置对象);PyTorch 从第一天就是 eager、命令式,训练循环的每一步都是显式 Python 语句,自然长成"每件事都是独立一行代码"的风格。**没有哪一种绝对更好**:TF 的声明式换来"忘记调用某个步骤"这类坑基本不存在(第 6 节裁剪、第 3 节调度都是自动生效),代价是"内部到底按什么顺序做了什么"变得不直接可见,必须像本篇一样反复读源码才能确认;PyTorch 的命令式换来每一步都清晰可见、可以在中间插入任意自定义逻辑,代价是 torch06 通篇反复出现的"忘记调用"/"调用顺序或粒度搞错"类常见坑(scheduler.step() 的调用粒度、clip_grad_norm_ 的调用时机、zero_grad 是否遗漏……)。

**AI 研究/工程场景:** 团队同时维护一套 TF 训练代码和一套 PyTorch 训练代码(常见于"生产用 TF Serving、研究用 PyTorch"这类历史遗留的技术栈分裂),做 code review 或者把一个新训练技巧从一边移植到另一边时,本篇六节验证过的每一处差异都是真实的迁移风险点——尤其是第 4 节(weight decay 数值语义方向相反的迁移陷阱)和第 6 节(`clipnorm` vs `global_clipnorm` 的语义偷换)这两处,表面上"参数名字都叫 weight_decay/clipnorm,应该可以直接抄数值",实际抄过去数值相同但训练动态不同,这种"看起来对齐、实际没对齐"的坑比一眼就能看出来的语法错误更难排查,也是这类跨框架迁移工作里最值得建立 checklist 去逐项核对的地方。

**可运行例子(TF 一行声明式构造,对比 PyTorch 需要拆成几步命令式调用——仅作对照说明,PyTorch 一侧不在本系列 WSL2 环境执行):**
```python
import tensorflow as tf

# TF:调度、weight decay、梯度裁剪,三件事一次性在构造函数里声明完
schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9)
opt = tf.keras.optimizers.AdamW(
    learning_rate=schedule,
    weight_decay=0.01,
    global_clipnorm=1.0,
)

w = tf.Variable([1.0, 2.0], dtype=tf.float32)
for step in range(3):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(w ** 2)
    grads = tape.gradient(loss, [w])
    opt.apply_gradients(zip(grads, [w]))   # 调度 + weight decay + 裁剪,全部在这一行内部自动发生

assert int(opt.iterations.numpy()) == 3
print("TF: 3 步之后 w =", w.numpy())
```
等价的 PyTorch 训练循环需要拆成的独立语句(仅列出对照,不在本文环境执行——本系列 WSL2 `tf-venv` 是专门为 TF 独立建的 venv,不装 torch,这一点 00 篇已经声明过):
```
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
for step in range(3):
    optimizer.zero_grad()
    loss = ...
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # 必须手动插入,且必须在 backward 之后、step 之前
    optimizer.step()
    scheduler.step()   # 调用粒度必须自己保证正确(torch06 §9 的常见坑)
```
六行独立语句对一行 `apply_gradients`,不是"TF 代码更少更好",是两种设计哲学在代码行数上的直接体现——TF 那一行背后自动做的事情,恰恰是本篇前 6 节要一件件打开源码验证清楚的内容。

**面试怎么问 + 追问链:**
- **Q:** "如果要你用一句话总结 TF 和 PyTorch 优化器 API 设计哲学的核心差异,你会怎么说?" —— 期望"TF 声明式配置(构造时一次性声明,自动生效);PyTorch 命令式调用(训练循环里每一步显式调用)",并且能各举至少两个具体机制(调度/裁剪/weight decay 任选)支撑这个论断,而不是只会喊口号。
- **追问 1(检验是否真的理解,而不是背对照表):** "这两种哲学各自的代价是什么?" —— 声明式的代价是内部行为不直接可见、必须读源码/文档才能确认执行顺序;命令式的代价是容易漏调用某一步或调用粒度搞错(能举出 torch06 里具体的常见坑案例是加分项)。
- **追问 2(跨模块迁移):** "这种声明式 vs 命令式的差异,只在优化器这一个模块存在吗?" —— 开放题,期望能类比到别的模块可能也有类似对照(比如数据加载管道声明式的 `tf.data` 链式 API vs PyTorch `DataLoader` + 显式 for 循环,这属于 09 篇范畴,这里点到即可,不要求展开)。
- **追问 3(工程向,呼应本节场景):** "同时维护 TF 和 PyTorch 两套训练代码,你会怎么建立迁移 checklist 防止本篇这类'参数名一样、语义不一样'的坑?" —— 期望能提出"对每一个看起来同名的参数/函数,先写一个和本篇一样的最小隔离实验验证数值行为,而不是凭参数名字直接假设等价"这个方法论本身,这也是本篇乃至整个系列贯穿的验证方法论。

**常见坑:** 看到 TF 和 PyTorch 优化器里"名字相同或相近的参数"(`weight_decay`、`clipnorm`/`clip_grad_norm_`、`learning_rate`)就假设语义完全对等、可以直接照抄数值——本篇第 4 节和第 6 节已经用具体实验证明了至少这两处存在真实的语义差异,方向还相反(TF 的 weight_decay 两个类之间等价、PyTorch 不等价;TF 的 clipnorm 默认逐张量、PyTorch 默认全局)。跨框架迁移或者跨框架背题的时候,任何"名字一样"的地方都值得反过来多问一句"这两边的实现真的一样吗",而不是想当然。

---

## 小结:这一批 7 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | `tf.keras.optimizers` 总览与 legacy/新 API | 优化器实现本身在 2.11 前后独立重写过一次(与 Keras2/3 分裂无关),新旧并存不是继承关系,新架构把横切能力收进基类 |
| 2 | `apply_gradients` 机制 | 显式接收 `(grad, var)` 对,因为 `tape.gradient` 是返回值而非像 `.backward()` 那样写 `.grad` 属性;部分 None 静默跳过+警告,全部 None 直接报错 |
| 3 | `LearningRateSchedule` | 可调用对象声明式传入,内部靠 `iterations` 自动推进;`optimizer.learning_rate` 是"上一次生效值"的只读快照,不是 schedule 本体 |
| 4 | Adam/AdamW 的 weight decay | TF 里两者数值完全等价(共享同一份基类解耦实现),这一点和 PyTorch(两者不等价)是本篇发现的最大反直觉差异 |
| 5 | 优化器状态(slots) | legacy 有 `get_slot`/`get_slot_names`,新架构只有扁平 `optimizer.variables` + 内部按位置索引,不再有一步到位的公开查询方法 |
| 6 | 梯度裁剪 | `clipnorm` 是逐张量、`global_clipnorm` 才是 PyTorch `clip_grad_norm_` 的真正对应物,两者不能混用当同一件事 |
| 7 | 与 PyTorch 对比总结 | TF 声明式配置一次自动生效,PyTorch 命令式逐步显式调用;两种哲学各有代价,迁移时不能假设同名参数语义对等 |

下一批:[08-training-loop-internals.md](08-training-loop-internals.md) —— `fit()` 内核与自定义训练循环(`Model.train_step()` 覆写、GradientTape 手写训练循环、`LossScaleOptimizer` 混合精度)。

---

*更新:2026-07-09*
