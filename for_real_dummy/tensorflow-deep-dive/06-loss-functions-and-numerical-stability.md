# 06 · 损失函数与数值稳定性(Loss Functions and Numerical Stability)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批覆盖 5 个知识点,是 [torch-deep-dive/05-loss-functions-and-numerical-stability.md](../torch-deep-dive/05-loss-functions-and-numerical-stability.md)(以下简称"torch05")的 TensorFlow 对照篇。**数值不稳定性的数学根源——`exp` 上溢、`log(0)` 下溢、log-sum-exp trick 怎么同时堵死这两条路——是完全通用的数学,torch05 已经现场验证过,这里不重复推导。** 本篇的信息增量在别处:TF/Keras 用来"保护"你不掉进这些数值陷阱的**工程机制**,和 PyTorch 完全不是一套东西——`tf.keras.losses.Loss` 有自己独立的 `reduction` 体系和 `get_config()` 序列化契约,内置 loss 函数会在概率张量上偷偷挂一个 `_keras_logits` 属性来"看穿"你有没有正确设置 `from_logits`,`SparseCategoricalCrossentropy` 和 `CategoricalCrossentropy` 是两套完全独立的底层实现而不是互相包装的关系,`reduction` 在分布式训练自定义循环里甚至会被硬性拦截报错而不是任由你悄悄踩坑——这些都是 PyTorch 没有对应物、必须专门现场验证的 TF/Keras 机制。

**环境:** 本文所有代码在 [00-roadmap.md](00-roadmap.md) 声明的环境(WSL2、TF 2.21.0、`~/tf-venv`、GPU 可见、`TF_USE_LEGACY_KERAS=1`)下用 `source ~/tf-venv/bin/activate && python ...` 实际跑通验证,所有"实测"字样的数字都是现场跑出来的真实输出,不是转述文档或凭经验推算。"AI 研究/工程场景"段落按 00 篇声明,是根据真实训练场景重构的例子,不是仓库代码引用。

**本篇统一结构(与 00 篇模板一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计** —— 本系列的核心加深点
4. AI 研究/工程场景
5. 可运行例子(带 `assert`,能内省的地方现场打印内部状态)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. `SparseCategoricalCrossentropy` vs `CategoricalCrossentropy` —— 两套独立实现,不是互相包装

**是什么:**
```
tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, label_smoothing=0.0, axis=-1, name='categorical_crossentropy'
)
# y_true: one-hot,形状 (batch, num_classes),和 y_pred 同形状

tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, ignore_class=None, name='sparse_categorical_crossentropy'
)
# y_true: 整数类别下标,形状 (batch,);y_pred 形状 (batch, num_classes)
```

**一句话:** 两者算的是同一件事——多分类交叉熵——唯一"看得见"的差异是 label 编码方式(one-hot vs 整数下标);但翻源码会发现两者从 `from_logits=True` 那一刻起就分别路由到 TF 两个完全独立注册的原生 op(`tf.nn.softmax_cross_entropy_with_logits` vs `tf.nn.sparse_softmax_cross_entropy_with_logits`),不是"sparse 版本把 label 转成 one-hot 以后再调用 dense 版本的同一份计算"。

**底层机制/为什么这样设计:**

先给结论,再给证据。结论:`SparseCategoricalCrossentropy` 和 `CategoricalCrossentropy` 是两套**平行**的实现,各自针对自己的 label 编码方式做了专门优化,不存在谁包装谁的关系。

证据 1(源码 + 图追踪现场验证):`tf_keras/src/backend.py` 里,`categorical_crossentropy` 在 `from_logits=True` 时直接调 `tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output, axis=axis)`;`sparse_categorical_crossentropy` 在 `from_logits=True` 时调的是 `tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output)`——这不是"sparse"前缀的文字游戏,是 TF 底层两个独立注册的 op(`SoftmaxCrossEntropyWithLogits` vs `SparseSoftmaxCrossEntropyWithLogits`)。把两条路径用 `tf.function` 追踪成图后检查 `concrete_function.graph.get_operations()`,`CategoricalCrossentropy` 的图里只有 `SoftmaxCrossEntropyWithLogits`,`SparseCategoricalCrossentropy` 的图里只有 `SparseSoftmaxCrossEntropyWithLogits`,两者从不出现在对方的图里——这是"两个不同 kernel"最直接的证据。`sparse_categorical_crossentropy` 整个函数体从头到尾没有一次 `tf.one_hot` 调用:它拿到的整数下标全程就是整数下标,`SparseSoftmaxCrossEntropyWithLogits` 这个 op 内部是"按下标 gather 出目标类别对应的量"而不是"构造一份 one-hot 再逐元素相乘求和"——这才是 sparse 版本存在的核心工程动机:类别数巨大时(语言模型词表几万到十几万),构造 `(batch, vocab_size)` 的 one-hot 矩阵本身就是一笔昂贵到不现实的内存开销,sparse 版本从设计上就不允许这种中间产物出现。

证据 2(更反直觉,`from_logits=False` 概率输入分支,现场用**没有归一化**的"概率"实测验证):`categorical_crossentropy` 在 `from_logits=False` 时会先显式做 `output = output / tf.reduce_sum(output, axis, True)`——强制重新归一化,再 clip 到 `[epsilon, 1-epsilon]` 取 log。`sparse_categorical_crossentropy` 在同样场景下完全没有这一步除法,只 clip 然后取 `tf.math.log(output)`,再把这个"log 值"原样送进 `tf.nn.sparse_softmax_cross_entropy_with_logits` 当作 logits 用。乍看这是两条会给出不同答案的路径——但实测结果是**两者数值完全相同**:输入 `[0.5, 0.3, 0.3]`(故意让它加起来是 1.1,不是合法概率),`CategoricalCrossentropy` 给出 `0.7885`(等价于显式除以 1.1 归一化之后再取 `-log(0.5/1.1)`),`SparseCategoricalCrossentropy` 给出的**也是** `0.7885`,而不是没有归一化的 `-log(0.5)=0.6931`。原因是一个漂亮的恒等式:`softmax(log(p))_i = exp(log(p_i)) / Σ_j exp(log(p_j)) = p_i / Σ_j p_j`——只要把"概率"取一次 log 再喂给内部会重新做 softmax 的 op,原始输入到底有没有归一化这件事就被 softmax 自己**隐式抹平**了。`sparse_categorical_crossentropy` 看似没做归一化,但因为它把 `log(output)` 送进了一个内部会重新做 softmax 的 op,归一化在这个"取log再softmax"的往返里被免费完成了。这不是巧合,是 softmax 对输入整体缩放不敏感这条性质的直接推论,也解释了两个字面上完全不同的代码分支为什么在数值上总是重合。

**AI 研究/工程场景:** 语言模型的下一词预测训练,词表动辄几万到十几万,`SparseCategoricalCrossentropy` 是唯一现实的选择——先把 token id 转成 one-hot 再算交叉熵,单是这份 one-hot 张量的显存开销就是 `batch_size * seq_len * vocab_size` 量级,大模型训练里会直接把这一步变成显存瓶颈,这也是证据 1 里"sparse 版本全程不构造 one-hot"这一点的现实分量。`CategoricalCrossentropy` 则适合类别数不大、且 label 本来就该以概率分布形式存在的场景,比如知识蒸馏里学生模型要拟合教师模型给出的软标签分布,或者手工做了 label smoothing 之后 label 已经变成了非 0/1 的浮点分布——这时候整数下标表达不出"软"分布,只能用 dense 版本。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

# --- 基本等价性: 整数label vs one-hot label,同样的底层数学(都用from_logits=True的稳定路径) ---
y_true_int = tf.constant([0, 1])
y_true_onehot = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
y_pred = tf.constant([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])  # logits

scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true_int, y_pred)
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true_onehot, y_pred)
assert np.allclose(scce.numpy(), cce.numpy(), atol=1e-6)

# --- 底层证据:两者被tf.function追踪后,图里出现的是两个完全不同的原生op,不是"one-hot再调用同一个kernel" ---
@tf.function
def traced_scce(yt, yp):
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(yt, yp)

@tf.function
def traced_cce(yt, yp):
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(yt, yp)

concrete_scce = traced_scce.get_concrete_function(tf.TensorSpec([2], tf.int32), tf.TensorSpec([2, 3], tf.float32))
concrete_cce = traced_cce.get_concrete_function(tf.TensorSpec([2, 3], tf.float32), tf.TensorSpec([2, 3], tf.float32))

scce_ops = {op.type for op in concrete_scce.graph.get_operations()}
cce_ops = {op.type for op in concrete_cce.graph.get_operations()}
assert "SparseSoftmaxCrossEntropyWithLogits" in scce_ops
assert "SoftmaxCrossEntropyWithLogits" in cce_ops
assert "SparseSoftmaxCrossEntropyWithLogits" not in cce_ops
assert not any(t == "SoftmaxCrossEntropyWithLogits" for t in scce_ops)   # sparse版本不出现dense的op

# --- from_logits=False 分支:两者代码路径完全不同,但数学上意外重合(log->softmax往返自动做了归一化) ---
probs_not_normalized = tf.constant([[0.5, 0.3, 0.3]])   # 故意不归一化,sum=1.1
y_int_p1 = tf.constant([0])
y_oh_p1 = tf.constant([[1.0, 0.0, 0.0]])

loss_cce = tf.keras.losses.CategoricalCrossentropy()(y_oh_p1, probs_not_normalized)
loss_scce = tf.keras.losses.SparseCategoricalCrossentropy()(y_int_p1, probs_not_normalized)
assert np.allclose(loss_cce.numpy(), loss_scce.numpy(), atol=1e-5)
assert abs(loss_cce.numpy() - (-np.log(0.5 / 1.1))) < 1e-4     # 等价于"显式除以0.5+0.3+0.3=1.1做归一化"
assert abs(loss_cce.numpy() - (-np.log(0.5))) > 1e-3            # 而不是直接对0.5取log(未归一化)

print("point 1 all assertions passed")
```

实测(WSL2 `~/tf-venv`,TF 2.21.0):`concrete_scce.graph` 里出现的交叉熵相关 op 精确是 `['SparseSoftmaxCrossEntropyWithLogits']`,`concrete_cce.graph` 里精确是 `['SoftmaxCrossEntropyWithLogits']`;`probs=[0.5,0.3,0.3]`(sum=1.1)时 `CategoricalCrossentropy` 和 `SparseCategoricalCrossentropy` 给出完全相同的 `0.7884574`,与显式归一化后的 `-log(0.5/1.1)=0.78846` 吻合,而不是未归一化的 `-log(0.5)=0.69315`。

**面试怎么问 + 追问链:**
- **Q:** "`SparseCategoricalCrossentropy` 和 `CategoricalCrossentropy` 有什么区别?" —— 期望先说出 label 编码方式的差异(整数下标 vs one-hot),这是最基本的一层。
- **追问 1(核心):** "两者底层是共享同一个计算 kernel 吗,还是各自独立实现?" —— 期望准确说出"各自独立",分别路由到 `sparse_softmax_cross_entropy_with_logits` 和 `softmax_cross_entropy_with_logits` 两个不同的 op,而不是"sparse 版本内部把 label 转成 one-hot 再调用 dense 版本"这种想当然的猜测——这是一个筛选"是否真的看过源码,还是凭直觉编答案"的好问题。
- **追问 2(工程向):** "为什么大词表场景一定要用 sparse 版本?" —— 期望说出 one-hot 中间张量在词表巨大时的显存开销,而不只是含糊地说"sparse 更省内存"。
- **深挖追问(区分度很高):** "如果给两者都喂'概率'而不是 logits,但这份'概率'因为某种原因没有真正归一化(加起来不等于1),两者算出来的 loss 会一样吗?" —— 期望能推理出"数学上因为 softmax 对输入整体缩放不敏感,即便两条代码路径处理归一化的方式完全不同,最终结果仍然一致",这是一个几乎不会被常规教程覆盖、需要真正理解 softmax 性质才能答对的问题。

**常见坑:**
- 想当然地认为"sparse"代表整个计算都变快、变省内存——实际上只有 **label 的存储表示**更紧凑,`y_pred`(logits)本身依然是稠密的 `(batch, num_classes)`,softmax 内部的归一化求和这一步开销和 dense 版本完全一样,sparse 版本省下来的只是"不需要再额外分配一份 `(batch, num_classes)` 的 one-hot 空间去表示 label"。
- 混淆两种 label 格式(该传整数下标却传了 one-hot,或反过来)——实测这两个方向在形状明显不匹配时都会被 TF 直接拦截报出清晰的 `ValueError`(`labels.shape` 必须和 `logits.shape` 兼容),不是本系列常见的"静默算出错误结果"模式,但这个保护依赖形状检查,不能保证覆盖所有维度组合(比如更高秩的分割任务 label),排查这类问题时形状仍然是第一嫌疑对象。

---

## 2. `from_logits` 参数的数值稳定性意义 —— log-sum-exp trick 与一个被隐藏的安全网

**是什么:**
```
tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)  # 默认值:y_pred被当成"合法概率"
tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)   # y_pred被当成logits(原始未归一化分数)
tf.nn.log_softmax(logits, axis=-1)   # TF提供的、独立于任何loss类的数值稳定log-softmax实现
```

**一句话:** 教科书写法"先 `softmax` 变成概率、再取 `log`、再算负对数似然"在数学上和 `from_logits=True` 完全等价,但在浮点数上不安全——`from_logits=True` 内部走的是 log-sum-exp trick(先减最大值再在对数空间做减法),全程不产生任何"真实的概率数值",从根源上避开了 `exp` 上溢和"概率下溢成 0 再取 log"这两类风险;而 TF 的内置 loss 即便你传了 `from_logits=False`,也不会像手写公式那样直接吐出肉眼可见的 `nan`——它会用一个 `epsilon` clip 兜底,给你一个**有限但错误**的数字,这比 PyTorch 对应机制的报错/`nan` 更隐蔽。

**底层机制/为什么这样设计:**

数学恒等式和 log-sum-exp trick 的推导 torch05 第 1 节已经现场验证过(`log(softmax(x)_i) = (x_i - m) - log(Σ_j exp(x_j - m))`,`m = max_j(x_j)`),这里不重复,直接看 TF 这边两处独有的工程行为。

**行为 1:TF 内置 loss 的 `from_logits=False` 分支自带 `epsilon` clip,不会产生真 `nan`,但会给出一个悄悄被压小的错误数字。** `tf_keras/src/backend.py` 里 `sparse_categorical_crossentropy` 和 `categorical_crossentropy` 在 `from_logits=False` 时,都会先 `tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)` 再取 log,其中 `epsilon_` 默认是 `1e-7`。这意味着即便你手工构造出一个精确饱和到 `0.0`/`1.0` 的"概率"(比如把两个相差 200 的 logit 做 `tf.nn.softmax`),TF 内置 loss 也不会算出 `-inf`——`-log(1e-7)≈16.118` 是它能给出的最大值,一个**看起来完全正常的有限数字**。但这个数字是错的:用 `from_logits=True` 在同一组原始 logits 上算出的真实值是 `200.0`,`16.118` 只是 clip 阈值决定的"天花板",和这次预测到底错得多离谱毫无关系。这比手写公式直接抛 `nan` 更危险——`nan` 至少会让训练脚本或 `tf.debugging.check_numerics` 立刻报警,而一个"看起来正常"的 `16.118` 不会触发任何警报,只会让这一步的梯度信息被系统性地扭曲,且没有任何信号提示你这件事发生了。

**行为 2(全篇最反直觉的发现):Keras 的 `activation='softmax'`/`'sigmoid'` 会在输出张量上偷偷挂一个 `_keras_logits` 属性,让 `from_logits=False` 在这种场景下反而变得安全。** 源码 `tf_keras/src/backend.py` 的 `softmax()`/`sigmoid()` 函数(被 `tf.keras.activations.softmax`/`sigmoid` 以及任何 `activation='softmax'` 的 Keras 层调用)在算完结果后,都会执行一行 `output._keras_logits = x`——把激活前的原始输入(也就是真正的 logits)缓存成输出张量的一个隐藏 Python 属性。TF 内置 loss 函数在处理 `from_logits=False` 之前,都会先调用一个叫 `_get_logits` 的辅助函数,它做的第一件事就是检查 `hasattr(output, "_keras_logits")`:如果这个属性存在,直接读取它、**无视你传的 `from_logits=False`**、原地切换成数值稳定的 `from_logits=True` 路径。实测验证:同样是让 logit 差距达到 200 这种精确饱和场景,手写 `tf.nn.softmax()` 算出的概率喂给 `from_logits=False` 得到错误的 `16.118`;但换成 `tf.keras.activations.softmax()`(或者干脆用 `Dense(units, activation='softmax')` 这种标准 Keras 层写法)算出的、数值上完全相同的概率,喂给同一个 `from_logits=False` 的 loss,却能精确得到正确答案 `200.0`——因为 loss 内部根本没有真的用这份概率去算,而是透过 `_keras_logits` 这个隐藏通道直接抓回了原始 logits。这是一个只在"概率是通过 Keras 自己的 activation 机制产生"时才生效的安全网,手写 `tf.nn.softmax`/`tf.nn.sigmoid`,或者概率来自其它任何来源(numpy 数组、外部预处理管线),都不受这份保护。

**AI 研究/工程场景:** 知识蒸馏、对比学习这类需要"温度缩放 softmax"(`softmax(logits / T)`)的场景,工程师经常图方便手写一个 `tf.nn.softmax(logits / temperature)` 再喂给下游 loss——这段代码在正常温度、正常 logit 量级下完全看不出问题,一旦训练后期模型越来越自信、logit 差距变大,或者 `temperature` 被调得很小放大了这个差距,就会真实撞上"行为 1"描述的陷阱:loss 曲线不会出现 `nan` 这种明显报警信号,只会在某个阶段开始表现出"数值卡在一个奇怪的常数附近、梯度信息变得很弱"这种更难定位的异常,排查成本远高于一次直接的 `nan` 崩溃。这也是为什么工程上更推荐的做法是**无论如何都设 `from_logits=True`、让模型最后一层不带激活函数**,而不是依赖"行为 2"这个隐藏安全网——安全网只在特定条件下生效,`from_logits=True` 是唯一在任何构造路径下都成立的稳妥选择。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

# --- 完全手写、不经过任何 TF/Keras 内置保护的 log(softmax): 真实触发 nan ---
def naive_log_softmax(x):
    e = tf.exp(x)                       # 没有减最大值
    return tf.math.log(e / tf.reduce_sum(e, axis=-1, keepdims=True))

big_logits = tf.constant([[1000.0, 1001.0, 1002.0]])
naive = naive_log_softmax(big_logits)
assert tf.reduce_all(tf.math.is_nan(naive))          # exp上溢成inf,inf/inf=nan,log(nan)=nan

stable = tf.nn.log_softmax(big_logits)
assert tf.reduce_all(tf.math.is_finite(stable))

# --- TF内置 SparseCategoricalCrossentropy(from_logits=False) 不会真的产生nan/-inf,
#     但会给出一个"看似正常却是错的"数值 —— 这比直接报nan更隐蔽 ---
spread_logits = tf.constant([[0.0, 200.0]])
probs_raw = tf.nn.softmax(spread_logits)                 # 手工softmax,不带_keras_logits标记
assert probs_raw.numpy().tolist() == [[0.0, 1.0]]          # 精确饱和成0/1

y_true = tf.constant([0])   # 正确类别恰好是概率精确为0的那一个
wrong_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true, probs_raw)
correct_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, spread_logits)
assert np.isfinite(wrong_loss.numpy())                       # 不是nan/inf!TF内置loss有epsilon clip兜底
assert abs(correct_loss.numpy() - 200.0) < 0.5                # from_logits=True给出真实、正确的值(约200)
assert wrong_loss.numpy() < correct_loss.numpy() - 100         # wrong_loss被clip悄悄压小,远小于真实值

# --- Keras自己的activation='softmax'/'sigmoid'会在输出tensor上偷偷挂一个_keras_logits属性,
#     内置loss的from_logits=False会优先读取这个属性、自动"穿透"回softmax前的logits,让结果依然稳定正确 ---
probs_via_keras_activation = tf.keras.activations.softmax(spread_logits)
assert hasattr(probs_via_keras_activation, "_keras_logits")     # 秘密标记存在
assert not hasattr(probs_raw, "_keras_logits")                    # 手写tf.nn.softmax没有这个标记

protected_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true, probs_via_keras_activation)
assert abs(protected_loss.numpy() - 200.0) < 0.5      # 即便写的是from_logits=False,数值依然稳定、正确!

print("point 2 all assertions passed")
```

实测(WSL2 `~/tf-venv`,TF 2.21.0):`naive_log_softmax` 在 `[1000,1001,1002]` 上精确输出 `[nan,nan,nan]`;`tf.nn.log_softmax` 输出稳定的 `[-2.4076,-1.4076,-0.4076]`;`from_logits=False` 在手写 `tf.nn.softmax` 结果上算出的错误值精确是 `16.118095`(等于 `-log(1e-7)`),`from_logits=True` 给出正确值 `200.0`;而同样的 `from_logits=False`,只要概率来自 `tf.keras.activations.softmax`,结果同样精确是 `200.0`——两个 `from_logits=False` 的调用,唯一区别是概率的产生方式,数值结果却相差超过 12 倍。

**面试怎么问 + 追问链:**
- **Q(经典开场):** "不设置 `from_logits`、自己手写 softmax 再算交叉熵,为什么会有数值问题?" —— 期望答出 `exp` 上溢和"概率下溢成 0 再取 log"两类风险,以及 `from_logits=True` 内部用 log-sum-exp trick 全程避免产生真实概率值。
- **追问 1(区分度很高):** "如果我说 TF 内置的 loss 函数在 `from_logits=False` 时也做了 `epsilon` clip 保护,是不是意味着这条路径就安全了?" —— 期望答出"不安全,只是不会崩成 `nan`,给出的是一个被 clip 压小的、有限但错误的数字,这种失败模式比直接报错更难发现"。
- **深挖追问(本节最有信息量的问题):** "如果输出层用了 `activation='softmax'`,后面接 `from_logits=False` 的交叉熵,这样写到底安不安全?" —— 期望能提到 Keras 内部有一套通过 `_keras_logits` 属性"看穿"标准 activation 层输出、自动切换到稳定路径的机制,但这个机制**不是万能的**——只在概率确实是通过 Keras 自己的 activation 函数产生时才生效,手写 `tf.nn.softmax` 不受保护,因此工程上仍然建议无条件使用 `from_logits=True` + 无激活输出层。
- **追问 2(连接更大主题):** "你会用什么方法确认自己的训练脚本没有踩到这类隐蔽的数值问题?" —— 期望提到 `tf.debugging.check_numerics`、定期检查 loss 是否卡在某个可疑常数附近、或者干脆在代码审查时把"是否所有交叉熵调用都显式设置了 `from_logits=True`"作为强制检查项。

**常见坑:**
- 把"训练日志里从没出现过 `nan`"当成"数值稳定"的证据——TF 内置 loss 的 `epsilon` clip 会把本该是 `-inf` 的情况压成一个有限数字,`nan` 不出现不代表梯度信息没有被扭曲,这个假设本身就是不成立的。
- 误以为只要用了 Keras 的 `activation='softmax'`/`'sigmoid'` 层,`from_logits=False` 就永远安全——这份保护依赖 `_keras_logits` 这个隐藏属性能一路传递到 loss 函数被调用的那一刻,一旦中间插入了额外的自定义 `Lambda`/手写 `tf.nn.softmax` 等操作产出一个新的张量,这个属性不会自动继承,保护也就随之失效,而代码表面上完全看不出差异。
- 在知识蒸馏、温度缩放 softmax 等需要手工干预概率计算的场景里,图省事直接手写 `tf.nn.softmax`,却仍然默认相信"TF 会保护我"——这类手写路径恰恰是最容易脱离保护网的地方。

---

## 3. `reduction` 模式(`AUTO`/`SUM`/`SUM_OVER_BATCH_SIZE`/`NONE`)—— 分布式训练下,TF 选择硬拦截而不是让你悄悄踩坑

**是什么:**
```
tf.keras.losses.Reduction.NONE                 # 不归约,返回逐样本loss,形状=(batch,)
tf.keras.losses.Reduction.SUM                   # 对逐样本loss求和,返回标量
tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE    # 对逐样本loss求和再除以样本数,等价于mean,返回标量
tf.keras.losses.Reduction.AUTO                   # 默认值:绝大多数场景下等价于SUM_OVER_BATCH_SIZE
```

**一句话:** `SUM` 是 `NONE` 结果的 `.sum()`,`SUM_OVER_BATCH_SIZE` 是 `NONE` 结果的 `.mean()`,`AUTO` 在单机场景下就是 `SUM_OVER_BATCH_SIZE` 的别名——这几条关系简单到几乎不需要解释;真正值得深挖的是 TF 在这里做了一个 PyTorch 没有对应物的设计:**一旦检测到你在 `tf.distribute.Strategy` 的 `scope()` 内部、脱离 `Model.fit()` 直接调用一个 `reduction=AUTO`(或显式 `SUM_OVER_BATCH_SIZE`)的 loss,会直接抛 `ValueError` 拒绝执行**,而不是像 PyTorch 的 `reduction='mean'`/`'sum'` 那样安静地算出一个数字、把"这个数字到底对不对"完全交给你自己判断。

**底层机制/为什么这样设计:**

`sum`/`mean` 只差一个"除以样本数",这个差异会让反传的梯度相差 `batch_size` 倍——torch05 第 2 节已经用真实数字验证过这条结论对损失函数的影响机理是通用的,这里不重复推导。TF 这边的特殊之处在于分布式场景下"样本数"这个分母本身就有歧义:自定义训练循环搭配 `tf.distribute.Strategy`(比如 `MirroredStrategy`)时,每张卡各自拿到的是**本地一份** batch(比如全局 batch size 是 8、2 张卡各分 4),如果在每张卡上分别用 `SUM_OVER_BATCH_SIZE` 对本地这 4 个样本求平均,再简单地把两张卡的平均值加在一起或再平均一次,算出来的"总 loss"分母实际上是本地 batch size(4),不是真正应该除以的**全局** batch size(8)——这正是 torch05 第 2 节"换 reduction 不换学习率,等价于隐性改变有效学习率"那个坑的分布式版本,只不过这里被放大成了"分母该是本地 batch size 还是全局 batch size"的歧义,且这个错误极难通过读代码发现,因为单卡调试时一切正常(本地 batch size 就等于全局 batch size)。

TF 的选择是:**与其让这个歧义在多卡环境下悄悄产生一个"看起来对、实际上错"的数字,不如直接拒绝执行,逼你显式做出选择。** 源码 `tf_keras/src/losses.py` 里 `Loss._get_reduction()` 的逻辑是:如果当前 `reduction` 是 `AUTO` 或 `SUM_OVER_BATCH_SIZE`,并且 `tf.distribute.has_strategy()` 为真,并且这个 loss 对象没有被标记为"由 `Model.fit()` 内部豁免"(`_allow_sum_over_batch_size`),就直接抛出 `ValueError`,报错信息原文直接指路:"请改用 `Reduction.SUM` 或 `Reduction.NONE`"。实测:在 `strategy.scope()` 内部直接调用一个默认 `AUTO` 的 loss 对象,精确复现了这条报错;而 `reduction=SUM` 或 `NONE` 在同样的 scope 内调用完全不受影响,正常返回结果。正确的替代写法是官方推荐的模式:用 `reduction=NONE` 拿到逐样本 loss,再显式调用 `tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)`——这个函数内部就是简单的 `reduce_sum(per_example_loss) / global_batch_size`,但强制你把"全局 batch size 到底是多少"这件事写清楚,而不是隐式依赖某个自动推断。

那么 `Model.compile()` + `Model.fit()` 为什么不受这条限制?因为 `fit()` 内部的训练循环(`tf_keras/src/engine/compile_utils.py`)在把你传入的 loss 包装进内部使用之前,会执行一行 `loss._allow_sum_over_batch_size = True`,相当于显式告诉这个 loss 对象"接下来的归约不需要你操心,`fit()` 自己的梯度平均逻辑已经正确处理了全局 batch size,你可以放心用 `AUTO`"——这不是"`fit()` 场景下歧义不存在了",而是"`fit()` 已经在更上层用另一套机制正确解决了这个歧义,不需要在 loss 这一层再报错拦你"。这条豁免精确解释了为什么"分布式训练用 `fit()` 从来没遇到过这个报错,一换成自定义循环立刻报错"——不是自定义循环更容易出 bug,而是 `fit()` 替你把这类 bug 挡在了更上层。

**AI 研究/工程场景:** 团队里某个成员为了给单卡 `model.fit()` 训练脚本提速,重写成 `tf.distribute.MirroredStrategy` 下的手写训练循环(`GradientTape` + `strategy.run()`),第一次跑起来时,如果用了默认 `AUTO` 的内置 loss,会在第一步就被 `ValueError` 拦下——这其实是一种幸运:错误在开发阶段就被暴露,而不是留到多卡训练跑了几个小时后才发现指标不对。真正危险的是"绕过了这个报错、但没有理解报错原因"的场景:把 `reduction` 换成 `SUM`,然后自己手写 `total_loss / GLOBAL_BATCH_SIZE`,如果这里的 `GLOBAL_BATCH_SIZE` 手滑写成了 `per_replica_batch_size`(单卡开发调试阶段两者数值相同,掩盖了这个 bug),上了真正的多卡环境后,有效学习率会随着 GPU 数量的增加而不知不觉放大,训练变得不稳定甚至发散,且现象和"学习率设太高"完全一样,很难第一时间联想到 reduction 的分母算错了。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

y_true = tf.constant([1, 2])
y_pred = tf.constant([[0.05, 0.95, 0.0], [0.1, 0.8, 0.1]])

none_ = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
sum_ = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(y_true, y_pred)
sobs_ = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(y_true, y_pred)
auto_ = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)   # 默认就是 AUTO

assert np.allclose(sum_.numpy(), none_.numpy().sum())
assert np.allclose(sobs_.numpy(), none_.numpy().mean())
assert np.allclose(auto_.numpy(), sobs_.numpy())     # 单机场景下 AUTO 退化成 SUM_OVER_BATCH_SIZE

loss_default = tf.keras.losses.SparseCategoricalCrossentropy()
assert loss_default.reduction == "auto"
assert loss_default._allow_sum_over_batch_size is False

# --- 分布式场景下的真坑: 在 strategy.scope() 里手写训练循环直接调用AUTO/SUM_OVER_BATCH_SIZE,会被硬性拦截 ---
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    loss_auto_in_scope = tf.keras.losses.SparseCategoricalCrossentropy()   # AUTO
    try:
        loss_auto_in_scope(y_true, y_pred)
        raised = False
        err_msg = ""
    except ValueError as e:
        raised = True
        err_msg = str(e)
    assert raised
    assert "SUM" in err_msg and "NONE" in err_msg      # 报错信息直接指路:该用SUM或NONE

    # 正确写法: reduction=NONE, 手动用 compute_average_loss 除以"全局"batch size
    per_example = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
    global_batch_size = 8    # 假设真实分布式场景下,全局batch size是本地batch size(2)乘以副本数
    correct_avg = tf.nn.compute_average_loss(per_example, global_batch_size=global_batch_size)
    naive_local_mean = tf.reduce_mean(per_example)     # 常见手滑:直接对本地batch取mean
    assert not np.isclose(correct_avg.numpy(), naive_local_mean.numpy())
    assert np.isclose(correct_avg.numpy(), per_example.numpy().sum() / global_batch_size)
    # 如果全局batch size(8) != 本地batch size(2),两种算法给出完全不同的数字——
    # 这正是"每个副本各自对本地batch取mean、再简单平均汇总"为什么会隐性改变有效学习率的根源

# --- Model.compile()+fit() 是唯一被豁免的场景:Keras内部会把 _allow_sum_over_batch_size 悄悄设成True ---
with strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(4,))])
    model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy())  # 默认AUTO
assert model.loss.reduction == "auto"
xb = np.random.randn(8, 4).astype("float32")
yb = np.random.randint(0, 3, size=(8,))
history = model.fit(xb, yb, epochs=1, verbose=0)      # 不会报错
assert model.loss._allow_sum_over_batch_size is True    # compile()内部悄悄放行了AUTO

print("point 3 all assertions passed")
```

实测(WSL2 `~/tf-venv`,TF 2.21.0,单 GPU 环境 `MirroredStrategy.num_replicas_in_sync=1`):`strategy.scope()` 内直接调用 `AUTO` 的 loss 精确抛出 `ValueError`,报错信息包含"`Reduction.SUM`"和"`Reduction.NONE`"字样;`tf.nn.compute_average_loss` 按全局 batch size(8)算出 `0.2942`,朴素 `reduce_mean` 按本地 batch size(2)算出 `1.1769`,两者相差整整 4 倍;`model.compile()+fit()` 全程不受影响,且验证了 `compile()` 之后 loss 对象的 `_allow_sum_over_batch_size` 确实从默认的 `False` 被悄悄改写成了 `True`。

**面试怎么问 + 追问链:**
- **Q:** "`tf.keras.losses.Reduction` 有哪几种模式,分别是什么关系?" —— 期望准确说出 `SUM`=`NONE`求和、`SUM_OVER_BATCH_SIZE`=`NONE`求平均、`AUTO`绝大多数场景下等价于`SUM_OVER_BATCH_SIZE`。
- **追问 1(本节核心):** "为什么在 `tf.distribute.Strategy` 下的自定义训练循环里,直接用默认 `reduction` 的 loss 会报错?" —— 期望说出"默认的`SUM_OVER_BATCH_SIZE`会除以**本地**batch size,而不是全局batch size,在多卡场景下这个分母是错的;TF 选择直接拒绝执行,而不是让你算出一个隐蔽错误的数字"。
- **深挖追问(区分度很高):** "同样是默认 `AUTO`,为什么 `model.fit()` 从来不会触发这个报错?" —— 期望说出"`fit()`内部会把 loss 对象标记为'豁免',因为 `fit()` 自己的梯度平均逻辑已经在更上层正确处理了全局 batch size,不需要在 loss 这层再拦截",而不是简单地说"`fit()`比较智能"。
- **追问 2(工程向):** "正确的分布式自定义训练循环该怎么写 loss 归约?" —— 期望说出"用`reduction=NONE`拿到逐样本loss,再用`tf.nn.compute_average_loss(per_example_loss, global_batch_size=...)`显式指定全局batch size"。
- **追问 3(连接更大主题,可留给 11 类展开):** "这个'本地 batch size 还是全局 batch size'的歧义,在 PyTorch 的 `DistributedDataParallel` 里有没有类似的坑?" —— 开放题,期望能联想到 torch05 第 2 节"`reduction='sum'`等价于隐性放大学习率"是同一类问题在不同框架下的不同表现形式:TF 选择在 API 层面硬拦截,PyTorch 需要开发者自己在 all-reduce 梯度同步逻辑里保持清醒。

**常见坑:**
- 被 `ValueError` 拦截后,不理解报错的真正原因,直接把 `reduction` 从 `AUTO` 换成 `SUM`,然后手写 `total_loss / batch_size` 时用了本地 batch size 而不是全局 batch size——单卡调试时两者数值相同,这个 bug 完全不会暴露,一旦扩展到真正的多卡训练,有效学习率会随 GPU 数量增加而放大,且现象和"学习率设太高导致的不稳定"高度相似,容易被误诊。
- 以为只要没有报错就说明 reduction 用对了——`reduction=SUM`/`NONE` 在 `strategy.scope()` 内都不会报错,但如果后续手动归约的除数选错了,同样能算出一个"能跑、不报错、但数值错误"的结果,报错拦截只能防住 `AUTO`/`SUM_OVER_BATCH_SIZE` 这一种具体误用,不是分布式 loss 归约正确性的全面保证。

---

## 4. 自定义 loss 正确写法 —— 继承 `Loss` 类 vs 函数式,`get_config()` 是两者的真正分水岭

**是什么:**
```
# 函数式:一个普通函数,签名 fn(y_true, y_pred) -> 逐样本loss
def my_loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

# 类式:继承 tf.keras.losses.Loss,实现 call(),可选覆写 get_config()
class MyLoss(tf.keras.losses.Loss):
    def __init__(self, weight=1.0, name="my_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight = weight
    def call(self, y_true, y_pred):
        return self.weight * tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
```

**一句话:** 两种写法在 `model.compile(loss=...)` 里都能直接用、数学结果完全一致,表面上看只是代码风格差异;但只要这个 loss 带有除了 `y_true`/`y_pred` 之外的**额外可配置状态**(权重系数、超参数),两者的差距就会在"保存/重新加载模型"这一步暴露出来——`Loss` 子类通过 `get_config()`/`from_config()` 提供了一套显式的状态序列化契约,而这套契约**默认不会自动帮你捕获自定义的 `__init__` 参数**,忘记手动覆写 `get_config()` 是这个模式下最容易被忽略、且不会报错的一个坑;纯函数式写法则完全没有任何配置捕获机制,重新加载模型时能换回的只是"同一个函数对象",函数闭包/默认参数里绑定的具体数值无法被区分和保存。

**底层机制/为什么这样设计:**

`tf.keras.losses.Loss` 基类的 `get_config()` 默认实现只有一行:`return {"reduction": self.reduction, "name": self.name}`。这是刻意设计成"最小公共子集"——基类不可能预知子类会往 `__init__` 里塞什么样的自定义参数(可能是一个数、一个列表、甚至一个嵌套配置),所以基类只负责序列化自己确定拥有的两个字段,其余的必须由子类自己在覆写的 `get_config()` 里补充完整,再配合 `super().get_config()` 把两部分合并。这个设计把"状态完整性"的责任完全交给了子类作者,好处是灵活(任何可 JSON 序列化的东西都能塞进去),代价是**忘记覆写不会有任何报错或警告**——实测验证:一个 `__init__` 接收 `weight=2.0` 但没有覆写 `get_config()` 的 `Loss` 子类,调用 `get_config()` 拿到的字典里根本不包含 `"weight"` 这个键;用这个残缺的 config 走 `from_config()` 重新构造实例,`weight` 会静默地变回 `__init__` 签名里的默认值(`2.0`),不是保留原来实例化时传入的真实值(比如 `99.0`),整个过程没有一次异常抛出,新实例的 `weight` 属性看起来完全正常,只是值不对。这是"配置驱动"设计哲学的典型代价:它把正确性完全托付给"实现者有没有记得手动同步",而不是靠类型系统或运行时检查强制保证。

再看 `compile()` 这一层对两种写法的实际处理差异。往 `model.compile(loss=...)` 里传一个 `Loss` 子类的**实例**,`model.loss` 属性保存的就是这个货真价实的对象,它的自定义状态(比如 `weight`)可以在训练过程中随时通过 `model.loss.weight` 读到;传一个裸函数,`model.loss` 保存的是这个函数对象本身(`model.loss is my_loss_fn` 为真),没有被包装成任何带 `get_config()` 的中间对象——这意味着函数式写法从一开始就没有进入这套配置捕获体系,不是"配置捕获失败",而是"这条路径压根不涉及配置捕获"。如果通过闭包或 `functools.partial` 给这个函数绑定了额外参数(比如两个不同实验分别用 `partial(my_loss_fn, weight=2.0)` 和 `partial(my_loss_fn, weight=5.0)`),模型保存机制能记录下来的至多是"这里用了某个叫这个名字的函数",绑定的具体参数值完全落在序列化机制的视野之外——重新加载时必须由使用者通过 `custom_objects` 手动提供**原始的函数对象**,如果两个实验复用了同一个函数名却绑定了不同参数,`custom_objects` 没有任何办法区分它们,只能拿到同一个未参数化的函数,原来两个实验各自的 `weight` 选择在磁盘上根本没有留下痕迹。

**AI 研究/工程场景:** 研究阶段经常会给一个基础 loss 引入一两个可调超参数做消融实验(比如给某个正则项一个权重系数,反复尝试 `0.1`/`0.5`/`1.0` 看效果),如果这个 loss 是用闭包/`functools.partial` 包出来的函数,每次实验保存的 checkpoint 在事后复盘时会遇到一个尴尬的问题:模型文件本身不会告诉你"这次训练到底用的是哪个权重系数",只能依赖训练脚本的日志或者文件命名规范去交叉核对,一旦这些外部记录缺失或对不上,这个信息就永久丢失了。用 `Loss` 子类并且正确覆写 `get_config()` 能让这个超参数成为模型序列化产物的一部分,`from_config()` 重新构造出来的 loss 对象能精确复现当时的配置,这也是为什么正式的、需要长期维护和复现的项目里,官方文档和大多数成熟代码库都推荐用类式写法而不是函数式写法。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

# --- 函数式写法 ---
def weighted_mse_fn(y_true, y_pred, weight=2.0):
    return weight * tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

# --- 类式写法:继承 tf.keras.losses.Loss ---
class WeightedMSE(tf.keras.losses.Loss):
    def __init__(self, weight=2.0, name="weighted_mse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight = weight

    def call(self, y_true, y_pred):
        return self.weight * tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"weight": self.weight})
        return config

y_true = tf.constant([[1.0, 2.0]])
y_pred = tf.constant([[1.5, 1.5]])

# --- 数学上完全等价 ---
loss_fn_result = weighted_mse_fn(y_true, y_pred, weight=3.0)
loss_obj_result = WeightedMSE(weight=3.0)(y_true, y_pred)
assert np.allclose(loss_fn_result.numpy(), loss_obj_result.numpy())

# --- get_config()差异之一:类式写法能把自定义构造参数(weight)带进配置字典;
#     基类 Loss.get_config() 默认只认 reduction/name,自定义参数必须手动加进去 ---
loss_obj = WeightedMSE(weight=5.0)
config = loss_obj.get_config()
assert config == {"reduction": "auto", "name": "weighted_mse", "weight": 5.0}
recreated = WeightedMSE.from_config(config)
assert recreated.weight == 5.0     # 完整还原,包括weight这个自定义参数

# --- 常见坑实锤: 如果忘记覆写get_config(), 自定义参数会静默丢失,不会有任何报错或警告 ---
class WeightedMSENoConfig(tf.keras.losses.Loss):
    def __init__(self, weight=2.0, name="weighted_mse_noconfig", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight = weight

    def call(self, y_true, y_pred):
        return self.weight * tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    # 没有覆写 get_config()!

original = WeightedMSENoConfig(weight=99.0)
bad_config = original.get_config()
assert "weight" not in bad_config          # weight=99.0 从config里彻底消失了
recreated_bad = WeightedMSENoConfig.from_config(bad_config)
assert recreated_bad.weight == 2.0          # 静默变回了__init__的默认值2.0,不是99.0

# --- compile() 的实际差异:类式写法传入的是一个货真价实的Loss实例,自定义状态(weight)可以直接读到;
#     函数式写法 compile() 原样保留函数对象本身,不会包装成带config的Loss实例 ---
model_obj = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(2,))])
model_obj.compile(optimizer="sgd", loss=WeightedMSE(weight=7.0))
assert model_obj.loss.weight == 7.0

model_fn = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(2,))])
model_fn.compile(optimizer="sgd", loss=weighted_mse_fn)
assert model_fn.loss is weighted_mse_fn      # compile()原样保留了这个函数对象,没有包装成Loss实例

print("point 4 all assertions passed")
```

实测(WSL2 `~/tf-venv`,TF 2.21.0):正确覆写 `get_config()` 的 `WeightedMSE(weight=5.0)` 得到的配置精确是 `{'reduction': 'auto', 'name': 'weighted_mse', 'weight': 5.0}`;没有覆写的版本得到的配置精确是 `{'reduction': 'auto', 'name': 'weighted_mse_noconfig'}`——`weight` 键完全不存在,`from_config()` 重建出的实例 `weight` 精确回退到 `__init__` 默认值 `2.0`,原始的 `99.0` 无声无息地丢失。

**面试怎么问 + 追问链:**
- **Q:** "自定义 loss 应该写成函数还是继承 `tf.keras.losses.Loss`?两者有什么本质区别?" —— 期望先说"数学计算逻辑上没有区别,`compile()`都能直接用",再点出"真正的差异在于要不要序列化额外的配置状态"。
- **追问 1(本节核心):** "如果自定义 loss 有一个额外的超参数(比如权重系数),继承 `Loss` 类默认就能正确保存这个参数吗?" —— 期望准确答出"不能,基类 `get_config()` 默认只捕获 `reduction`/`name`,自定义参数必须手动覆写 `get_config()` 加进去,忘记加不会报错,只会在 `from_config()` 重建时静默丢失、退回默认值"。这是一个专门用来分辨"知道要继承 Loss 类"和"真正理解序列化契约"的问题。
- **追问 2(区分度高):** "如果自定义 loss 是通过 `functools.partial` 或闭包绑定参数的函数,这种写法在模型保存/重新加载时会遇到什么问题?" —— 期望说出"函数式写法完全没有配置捕获机制,重新加载依赖 `custom_objects` 提供原始函数对象,闭包里绑定的具体参数值不会被保存,不同参数化的同名函数在磁盘上无法区分"。
- **追问 3(开放题):** "什么场景下你会选择函数式写法而不是继承 `Loss` 类?" —— 期望能说出"loss 完全没有额外可配置状态、只是一个纯粹的计算逻辑,且不打算长期维护/复现"这类场景,函数式写法更简洁,不是"函数式写法一无是处"。

**常见坑:**
- 继承 `Loss` 类却忘记覆写 `get_config()`,以为"用了官方推荐的类式写法"就等于"序列化正确性有保障"——实测验证这是错觉,不覆写 `get_config()` 的自定义参数会在 `from_config()` 重建时静默回退到默认值,整个过程没有任何异常或警告,只有在事后对比模型行为差异时才可能被发现。
- 用 `functools.partial`/闭包给同一个基础 loss 函数在不同实验里绑定不同的超参数,却指望模型保存机制能区分它们——`custom_objects` 重新加载时能拿到的只是原始函数对象本身,不同实验绑定的具体参数值从未被真正保存过,长期看这是可复现性的隐患。
- 混淆"函数能不能被 `compile(loss=...)` 接受"和"函数式 loss 能不能被完整序列化"——前者答案是"能",后者答案是"不能捕获额外参数",两者是完全不同的问题,能跑不代表能被完整保存和复现。

---

## 5. `BinaryCrossentropy` 相关坑 —— 被夸大的 shape 恐慌,和被低估的饱和梯度归零

**是什么:**
```
tf.keras.losses.BinaryCrossentropy(from_logits=False)   # 默认:y_pred被当成sigmoid之后的概率
tf.keras.losses.BinaryCrossentropy(from_logits=True)     # y_pred被当成logits(无激活的原始输出)
# y_true / y_pred 常见形状: (batch,) 或 (batch, 1)
```

**一句话:** 网上流传最广的 `BinaryCrossentropy` 坑是"`y_true` 形状 `(N,)` 和 `y_pred` 形状 `(N,1)` 不匹配会导致隐式 broadcast 错误"——实测这个具体组合其实被 TF 内部的 `squeeze_or_expand_dimensions` 安全处理,两种写法数值完全一致,不是隐患;真正会静默产生错误结果、且更难被文档提前警告到的,是 `y_true` 形状 `(N,)` 配上 `y_pred` 形状 `(N, C)`(`C>1`,比如最后一层手滑写成 `Dense(2)` 而不是 `Dense(1)`)这种秩只差 1、但最后一维不是 1 的组合——这时候 squeeze 逻辑不生效,退化成普通 NumPy 广播规则,把 `y_true` 错误地当成"每一列的标签"广播到每一行,不报错,给出一个有限但语义完全错误的数字。

**底层机制/为什么这样设计:**

`BinaryCrossentropy` 是 [第 2 点](#2-from_logits-参数的数值稳定性意义--log-sum-exp-trick-与一个被隐藏的安全网)描述的 log-sum-exp 保护机制和 `_keras_logits` 隐藏标记在二分类场景下的应用——`backend.sigmoid()` 同样会在输出张量上挂 `output._keras_logits = x`,`from_logits=False` 遇到通过 Keras 标准 `activation='sigmoid'` 产生的概率时同样会被"看穿"、自动切换到稳定路径,这里不重复验证,直接看这一节独有的 shape 处理机制。

**机制 1(先纠正一个被过度渲染的"坑"):`(N,)` vs `(N,1)` 被 `LossFunctionWrapper.call()` 内部的 `squeeze_or_expand_dimensions` 安全处理。** 这个函数的核心逻辑是:当 `y_pred` 和 `y_true` 的**秩(rank)恰好相差 1**,或者 `y_pred` 的最后一维恰好是 `1` 时,自动调用 `remove_squeezable_dimensions` 去掉多余的那个大小为 1 的维度,让两者按预期的语义对齐。`y_true=(N,)`、`y_pred=(N,1)` 正好同时满足这两个条件(秩差 1,且 `y_pred` 最后一维是 1),实测验证:这种组合下算出的 loss 和显式把 `y_true` 也 reshape 成 `(N,1)` 完全是同一个数字,不存在任何隐式错误 broadcast。这个机制的存在,恰恰是因为 `(N,)` vs `(N,1)` 是 `Dense(1)` 最后一层加 `y_true` 是一维数组时**极其常见**的组合,TF/Keras 选择在框架层面主动兜底,而不是要求用户每次都手动对齐。

**机制 2(真正的坑,精确圈定生效边界):判定条件是"秩差 1 **且** (秩差不为 1 **或** 最后一维是 1)"这个组合条件,一旦秩差恰好是 1、但 `y_pred` 最后一维不是 1,两个子条件都不成立,squeeze 完全不会发生。`y_true=(N,)` 配 `y_pred=(N,C)`(`C>1`)正是这个盲区:秩差是 1(不满足"秩差不为1"),`y_pred` 最后一维是 `C`(不满足"最后一维是1")——两个条件都不触发,`y_true`、`y_pred` 原样传入后续计算,binary crossentropy 的逐元素运算会依照标准 NumPy/TF 广播规则展开:`y_true` 的唯一一维(大小 `N`)会和 `y_pred` **最后一维**(大小同样是 `N`,如果凑巧 `C==N`)或者报错(如果 `C!=N`)对齐,把 `y_true` 广播成 `y_pred` 的形状。实测一个 `N=C=2` 的例子:`y_true=[1.0, 0.0]`,`y_pred=[[0.7,0.2],[0.3,0.9]]`(2 个样本、每个样本手滑输出了 2 个"标签维度"),广播结果精确是 `[[1.0,0.0],[1.0,0.0]]`——`y_true` 的两个值被解读成"第 0 维标签是 1.0、第 1 维标签是 0.0",**同时套用到两个样本身上**,而不是"样本 0 的标签是 1.0、样本 1 的标签是 0.0"。这不是"形状不合法所以报错",是"形状凑巧合法,但语义已经错位",TF 没有办法、也没有职责去猜测你的真实意图是"每样本一个标签"还是"每样本两个独立标签",只能忠实执行摆在面前的这份广播规则。

**AI 研究/工程场景:** 把一个多分类模型(比如最后一层是 `Dense(num_classes)`)改造成二分类模型时,常见的重构路径是把最后一层从 `Dense(num_classes, activation='softmax')` 换成 `Dense(1, activation='sigmoid')`,loss 从 `SparseCategoricalCrossentropy` 换成 `BinaryCrossentropy`——如果这次重构没有全面覆盖到(比如某个分支还残留着 `Dense(2)` 或者复制粘贴时忘了把 `2` 改成 `1`),`y_pred` 的形状就会停留在 `(N,2)`,而上游 `tf.data` 管道产出的 `y_true` 大概率还是 `(N,)` 这种单标签整数/浮点形式——恰好撞进机制 2 描述的盲区,模型能正常训练、loss 曲线看起来"正常下降"(因为确实在优化某个数值,只是这个数值的含义不对),指标却怎么调都异常,这类问题的排查成本很高,因为第一直觉通常是怀疑学习率、数据质量,而不是 loss 输入的 shape 语义。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

# --- (N,) vs (N,1) 这个最常被警告的shape组合,其实被 squeeze_or_expand_dimensions 安全处理,结果完全一致 ---
y_true_1d = tf.constant([0.0, 1.0, 0.0, 1.0])            # shape (4,)
y_true_2d = tf.constant([[0.0], [1.0], [0.0], [1.0]])     # shape (4,1)
y_pred_2d = tf.constant([[0.1], [0.8], [0.3], [0.6]])      # shape (4,1),典型 Dense(1,activation='sigmoid') 输出

bce = tf.keras.losses.BinaryCrossentropy()
loss_with_1d_label = bce(y_true_1d, y_pred_2d)
loss_with_2d_label = bce(y_true_2d, y_pred_2d)
assert np.allclose(loss_with_1d_label.numpy(), loss_with_2d_label.numpy())    # 数值完全相同,这个组合不是隐患

# --- 真正危险的shape坑:y_true是(N,)而y_pred是(N,C>1)(比如误把Dense(1)写成Dense(2)),
#     rank只差1、但y_pred最后一维不是1,squeeze逻辑不生效,触发普通numpy广播规则,不报错但结果彻底错误 ---
y_true_bad = tf.constant([1.0, 0.0])                        # (2,) 本意是"2个样本各1个二分类标签"
y_pred_bad = tf.constant([[0.7, 0.2], [0.3, 0.9]])            # (2,2) 手误多输出了一列

bad_loss = bce(y_true_bad, y_pred_bad)                        # 不报错!
assert np.isfinite(bad_loss.numpy())
broadcast_true = np.broadcast_to(y_true_bad.numpy(), y_pred_bad.numpy().shape)
assert broadcast_true.tolist() == [[1.0, 0.0], [1.0, 0.0]]     # 两行被塞进同一组标签,不是"每样本一个标签"的语义
manual_bce_per_elem = -(broadcast_true * np.log(y_pred_bad.numpy() + 1e-7) +
                         (1 - broadcast_true) * np.log(1 - y_pred_bad.numpy() + 1e-7))
assert np.allclose(bad_loss.numpy(), manual_bce_per_elem.mean(), atol=1e-4)

# --- sigmoid饱和 + BCE(from_logits=False) 的梯度归零陷阱,是CrossEntropy篇log-sum-exp故事的二分类版本 ---
x_extreme = tf.Variable([[100.0], [-100.0]])
y_extreme = tf.constant([[0.0], [1.0]])

with tf.GradientTape() as tape:
    probs_manual = tf.sigmoid(x_extreme)          # 手写sigmoid,不带_keras_logits保护标记
    loss_probs = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_extreme, probs_manual)
grad_probs = tape.gradient(loss_probs, x_extreme)
assert grad_probs.numpy().flatten().tolist() == [0.0, 0.0]      # 梯度精确归零

x_extreme2 = tf.Variable([[100.0], [-100.0]])
with tf.GradientTape() as tape2:
    loss_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_extreme, x_extreme2)
grad_logits = tape2.gradient(loss_logits, x_extreme2)
assert not np.allclose(grad_logits.numpy(), 0.0)
assert np.allclose(sorted(grad_logits.numpy().flatten().tolist()), [-0.5, 0.5], atol=1e-4)

print("point 5 all assertions passed")
```

实测(WSL2 `~/tf-venv`,TF 2.21.0):`y_true=(4,)` 和 `y_true=(4,1)` 配同一个 `(4,1)` 的 `y_pred`,loss 精确都是 `0.29900098`;`(2,)` 标签配 `(2,2)` 预测这种危险组合不报错,精确算出 `1.0215937`,和手动按广播规则复现的公式完全吻合;手写 `tf.sigmoid` 在 `x=±100` 处饱和后,`from_logits=False` 反传梯度精确是 `[0.0, 0.0]`,`from_logits=True` 直接在 logits 上算则给出有信息量的梯度 `[0.5, -0.5]`。

**面试怎么问 + 追问链:**
- **Q:** "`BinaryCrossentropy` 的 `y_true` 形状是 `(N,)`、`y_pred` 形状是 `(N,1)`,这样写安全吗?" —— 期望能准确答出"安全,TF 内部的 `squeeze_or_expand_dimensions` 会自动对齐这个组合,不是隐患",而不是人云亦云地复述"这是一个经典坑"却从未实际验证过。这是一道专门筛选"背过网上的坑" vs "真正验证过"的问题。
- **追问 1(本节最有区分度的问题):** "那什么样的 shape 组合才是真正危险的?" —— 期望能说出"`y_true=(N,)` 配 `y_pred=(N,C)` 且 `C>1`"这种秩差 1 但最后一维不是 1 的组合,squeeze 逻辑不生效,会退化成普通广播规则,把标签向量误当成"每列的标签"广播到每一行。
- **深挖追问:** "这种情况 TF 为什么不直接报错?" —— 期望理解"广播规则本身在数学上是合法的(`(N,)`能和`(N,C)`的最后一维对齐,前提是C==N),TF 没有办法区分'这是故意的广播'还是'用户手滑传错了形状',只能忠实执行摆在面前的规则"。
- **追问 2(呼应第2点):** "sigmoid 饱和之后,`BinaryCrossentropy(from_logits=False)` 会有什么问题?" —— 期望连回第 2 点的机制:手写 `tf.sigmoid` 产生的概率不带 `_keras_logits` 保护标记,饱和处梯度会被 `sigmoid` 自身的导数(精确等于 0)拦腰截断,`from_logits=True` 直接在 logits 上用解析梯度公式计算,不会有这个问题。
- **追问 3(工程向):** "生产代码里应该怎么规避这类风险?" —— 期望给出"无条件使用 `from_logits=True` + 输出层不带激活函数"这个不依赖任何隐藏保护机制、在任何 shape/数据来源组合下都成立的稳妥实践。

**常见坑:**
- 花了大量精力去防范"网上说的" `(N,)` vs `(N,1)` 组合,却对真正危险的 `(N,)` vs `(N,C>1)` 组合毫无警惕——这种本末倒置来自于"传播最广的坑不一定是验证过的坑",经验结论需要用实测重新校准,不能直接照搬。
- 重构模型结构(比如多分类改二分类)时只改了 loss 函数和最后一层的 `units` 数值,没有同步检查 `y_true` 的来源管道是否也做了相应调整,残留的旧形状能在新 loss 函数下"合法但错误"地跑起来,不会有任何报错提示你哪里错了。
- 认为"这个模型训练时 loss 在下降,所以 loss 的输入形状/语义一定是对的"——[第 2 点](#2-from_logits-参数的数值稳定性意义--log-sum-exp-trick-与一个被隐藏的安全网)和本节都验证过,TF 的 loss 计算在语义错误的输入下依然能给出一个有限、看似合理、还会随训练下降的数字,"loss 在下降"不能作为"计算逻辑正确"的证据。

---

## 小结:这一批 5 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | `SparseCategoricalCrossentropy` vs `CategoricalCrossentropy` | 两套独立实现,分别路由到 `sparse_softmax_cross_entropy_with_logits`/`softmax_cross_entropy_with_logits` 两个不同 op,sparse 版本全程不构造 one-hot;`from_logits=False` 分支代码路径完全不同,但因 softmax(log(p))=p/sum(p) 这条恒等式而数值上意外重合 |
| 2 | `from_logits` 数值稳定性 | log-sum-exp trick 全程避免产生真实概率值;TF 内置 loss 的 `from_logits=False` 有 `epsilon` clip 兜底,不会报 `nan` 但会给出有限的错误数字;Keras `activation=` 产生的概率会被 `_keras_logits` 隐藏属性"看穿"自动保护,手写 `tf.nn.softmax`/`sigmoid` 不受此保护 |
| 3 | `reduction` 模式 | `AUTO`/`SUM_OVER_BATCH_SIZE` 在 `tf.distribute.Strategy` 自定义训练循环里会被硬性拦截报错(本地 batch size 和全局 batch size 歧义),必须显式用 `NONE`+`tf.nn.compute_average_loss`;`Model.fit()` 因内部标记 `_allow_sum_over_batch_size=True` 而被豁免 |
| 4 | 自定义 loss 写法 | 函数式和类式数学等价,但 `Loss` 基类 `get_config()` 默认不捕获自定义 `__init__` 参数,忘记覆写会静默丢失配置且无报错;函数式写法完全没有配置捕获机制,闭包绑定的参数值无法被序列化保存 |
| 5 | `BinaryCrossentropy` 相关坑 | `(N,)` vs `(N,1)` 被 `squeeze_or_expand_dimensions` 安全处理,不是隐患;真正危险的是 `(N,)` vs `(N,C>1)`,会静默触发普通广播、语义错位但不报错;手写 `sigmoid` 饱和处梯度精确归零,和第2点是同一套 `_keras_logits`/log-sum-exp 机制的二分类版本 |

**这一批的核心方法论,也是全系列的方法论:** 数值不稳定性的数学根源是通用的(torch05 已验证),但"框架具体怎么保护你、保护的边界在哪里"必须现场验证,不能凭直觉套用另一个框架的心智模型——本篇最有价值的两个发现(`_keras_logits` 隐藏属性带来的条件性安全网、`SparseCategoricalCrossentropy`"不归一化却数值一致"的隐式归一化)都不是从文档字面能读出来的,是翻源码 + 现场构造反例实测出来的。

下一批:[07-optimizer-internals.md](07-optimizer-internals.md) —— 优化器内部机制。

---

*更新:2026-07-11*
