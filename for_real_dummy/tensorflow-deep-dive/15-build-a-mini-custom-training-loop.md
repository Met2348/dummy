# 15 · 手把手实战:从零搭一个迷你自定义训练循环

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 15 个"知识点",不计入"100 个知识点"的统计——和 [14 类](14-advanced-interview-depth.md)是同一挂,但风格不一样:14 号文件里,你是**旁观者**,跟着调研出来的 5 条追问链把推理过程看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实的 loss 数值(不是文字描述"会下降"),最后独立组装出一个真实能跑的迷你自定义训练循环。这是仓库"教程体"内容形态第一次从 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 推广到其他系列——沿用已经试点验证过的格式,不是重新发明。

本文件全部代码在 WSL2 `~/tf-venv`(TensorFlow 2.21.0,`TF_USE_LEGACY_KERAS=1`,和 00-roadmap.md 第 0 节声明的系列统一环境完全一致)下真实跑通验证。数据规模刻意压到几百个样本、每段代码几秒内跑完,不依赖 GPU 特有的任何 API——换到纯 CPU 环境、装 CPU 版 TensorFlow 应该同样能跑。和 [dsa-deep-dive/21](../dsa-deep-dive/21-build-a-mini-search-engine.md) 同一个原因:本文件用仓库统一的 `_verify_md.py` 校验,校验方式是把每个 ` ```python ` 代码块单独拎出来起一个新的子进程执行——块与块之间**不共享任何变量**,所以后面阶段用到前面阶段的数据/变量时,会重新贴一遍构造代码,不是偷懒复制,是这套校验机制要求的。

## 为什么是"自定义训练循环"

**先说清楚和 [08 类第 3 节](08-training-loop-internals.md)的关系,不然会显得是重复内容。** 08.3 已经讲过"甩开 `fit()`、自己写 `GradientTape` 循环"这件事,但用的是一个 `tf.keras.Sequential` 模型,梯度是对 `model.trainable_variables` 求的——"创建变量"这一步已经被 `Dense` 层封装好了,训练循环本身是在一个现成的抽象之上写的。这一篇往下再退一层:**不用任何 Keras 模型,直接对着两个裸的 `tf.Variable`(`W` 和 `b`)手写前向、反向、参数更新**——这是 `GradientTape` 机制能覆盖到的最原始形态,08.3 的 `model.trainable_variables` 骨子里也是在对一堆这样的 `tf.Variable` 做同样的事,只是 Keras 把"变量从哪来"这一步替你包好了。退到这一层的价值是:如果能不借助任何框架封装,从零把训练循环的四个环节(数据、前向、反向、更新)独立拼出来,再回头看 08.3 的 `model.trainable_variables` 版本,会看得更清楚它到底替你自动做了哪一步、省了哪些代码。

不是要发明新知识点,是把三个你已经学过的知识点串成一个真实能跑的东西:

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 手写一步:用 `GradientTape` 算梯度、用 `assign_sub` 手动更新参数,并现场证明这和 `optimizer.apply_gradients` 是同一件事 | [02 类](02-gradienttape-internals.md) GradientTape 工作原理 + 自动 watch 规则、[07 类](07-optimizer-internals.md) `apply_gradients` 机制 |
| 阶段 2 | 把"一步"扩展成 15 轮完整训练,手工切分 batch,真实验证 loss 持续下降 | 阶段 1 + [08 类第 3 节](08-training-loop-internals.md) 手写训练循环骨架(这里换成最原始的裸 `tf.Variable`,不经过 Keras 模型层) |
| 阶段 3 | 用 `tf.data.Dataset` 管道(`from_tensor_slices`+`shuffle`+`batch`)替换手工切分 batch,验证训练效果一致 | [09 类第 1/3 节](09-tf-data-pipeline.md) `from_tensor_slices` 语义 + `shuffle`/`batch` 顺序 |
| 阶段 4 | 对照 `model.compile()`+`model.fit()` 跑同样数据、同样轮数,验证最终参数/loss 彼此接近 | [08 类第 1/8 节](08-training-loop-internals.md) `fit()` 内核与 `compile()` 的关系 |

每个阶段的代码都能独立运行,跑完后拿真实输出验证(不是先假设一个"应该"的结果再往上凑)。

---

## 阶段 1:手写一步——前向、反向、参数更新,一步都不能少

在写"训练循环"之前,先证明能写对**一步**——这是全系列反复强调的纪律(呼应 [02 类](02-gradienttape-internals.md)"所有数值结果都经过手算或多种方式交叉验证")在这篇教程里的第一次应用。问题选全系列最小的:一元线性回归,真实参数 `true_w=2.0`、`true_b=1.0`,数据是加了噪声的合成数据,起点是 `W=0.0, b=0.0`。

```python
import tensorflow as tf
import numpy as np

np.random.seed(0)
tf.random.set_seed(0)

N = 200
x = np.random.uniform(-5, 5, size=(N, 1)).astype("float32")
true_w, true_b = 2.0, 1.0
noise = np.random.normal(0, 0.5, size=(N, 1)).astype("float32")
y = (true_w * x + true_b + noise).astype("float32")

W = tf.Variable(0.0, name="W")
b = tf.Variable(0.0, name="b")

def forward(x_batch):
    return W * x_batch + b

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

loss_before = float(mse_loss(y, forward(x)))

learning_rate = 0.05
with tf.GradientTape() as tape:
    y_pred = forward(x)
    loss = mse_loss(y, y_pred)
dW, db = tape.gradient(loss, [W, b])

print("loss before:", loss_before)
print("dW:", float(dW), "db:", float(db))

W.assign_sub(learning_rate * dW)
b.assign_sub(learning_rate * db)

loss_after = float(mse_loss(y, forward(x)))
print("loss after one hand-written step:", loss_after)
print("W after:", float(W.numpy()), "b after:", float(b.numpy()))

# 一步之后loss应该明显下降——参数从(0,0)朝着真实值(2.0,1.0)迈出了正确方向的第一步
# 用宽松断言(只要求降到1/5以下),不卡实测的具体倍数
assert loss_after < loss_before / 5
assert 0.0 < float(W.numpy()) < true_w   # 还没到真值,但方向是对的
print("stage1 one-step ok")
```

实测输出:`loss before` 约 32.68(参数还在原点,离真实直线很远),一步梯度下降之后 `loss after` 降到约 2.14——`W` 从 0 走到约 1.595,`b` 从 0 走到约 0.095。`tape.gradient(loss, [W, b])` 这里没有任何 `tape.watch()` 调用——`W`、`b` 是默认 `trainable=True` 的 `tf.Variable`,在 tape 作用域内被 `forward(x)` 读取过,按 [02 类第 2 节](02-gradienttape-internals.md)"trainable Variable 自动被追踪"的规则,自动进入监听范围,这一点在写这一步之前就应该心里有数,不是巧合。

**这一步用的是 `assign_sub` 手动更新,不是 `optimizer.apply_gradients`——两者是不是同一件事,不能凭感觉,现场验证一下:**

```python
import tensorflow as tf

# 用真实算出来的梯度(不是手填的近似数字)验证:手写assign_sub 和 optimizer.apply_gradients
# 在普通SGD下是不是完全等价——如果等价,后面几个阶段改用optimizer.apply_gradients就有依据,不是抄近道
x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
y_true = tf.constant([[3.0], [5.0], [7.0], [9.0]])  # true_w=2, true_b=1,不加噪声,方便看清数值

def make_vars():
    return tf.Variable(0.0, name="W"), tf.Variable(0.0, name="b")

def one_grad_step(W, b):
    with tf.GradientTape() as tape:
        y_pred = W * x + b
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return tape.gradient(loss, [W, b])

lr = 0.05

# 路径A:手写assign_sub
W_a, b_a = make_vars()
dW, db = one_grad_step(W_a, b_a)
W_a.assign_sub(lr * dW)
b_a.assign_sub(lr * db)

# 路径B:optimizer.apply_gradients,相同初始值、相同的一次梯度计算
W_b, b_b = make_vars()
dW2, db2 = one_grad_step(W_b, b_b)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
optimizer.apply_gradients(zip([dW2, db2], [W_b, b_b]))

print("assign_sub     : W=%.10f b=%.10f" % (float(W_a.numpy()), float(b_a.numpy())))
print("apply_gradients: W=%.10f b=%.10f" % (float(W_b.numpy()), float(b_b.numpy())))
assert float(W_a.numpy()) == float(W_b.numpy())
assert float(b_a.numpy()) == float(b_b.numpy())
print("stage1 equivalence check ok")
```

实测两条路径给出完全相同的数值(`W=1.7500000000 b=0.6000000238`,精确到打印出的全部小数位)。这不是巧合:朴素 SGD 的 `apply_gradients` 内部做的就是 `variable.assign_sub(learning_rate * gradient)`,`tf.keras.optimizers.SGD` 只是把这一行代码包成了一个可复用的对象(外加 momentum、weight decay 这些朴素 SGD 用不到的可选机制,[07 类](07-optimizer-internals.md)有更完整的展开)。**有了这条等价性打底,后面的阶段 2/3/4 全部改用 `optimizer.apply_gradients`,不是从这里开始换了一套不透明的黑箱,只是换了个更少写样板代码的写法。**

---

## 阶段 2:手写完整多轮循环——从"一步"到"真实训练"

一步能证明方向对,但离"训练好"还差得远。真实训练要跑很多轮(epoch),每轮要把全部数据切成小批(batch)分别喂进去——这一步先用最朴素的方式手工切分 batch(`np.random.permutation` 打乱下标 + 切片),验证完整多轮循环下 loss 是不是真的持续走低。

```python
import tensorflow as tf
import numpy as np

np.random.seed(0)
tf.random.set_seed(0)

N = 200
x = np.random.uniform(-5, 5, size=(N, 1)).astype("float32")
true_w, true_b = 2.0, 1.0
noise = np.random.normal(0, 0.5, size=(N, 1)).astype("float32")
y = (true_w * x + true_b + noise).astype("float32")

W = tf.Variable(0.0, name="W")
b = tf.Variable(0.0, name="b")
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

def forward(x_batch):
    return W * x_batch + b

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

BATCH_SIZE = 20
EPOCHS = 15
epoch_losses = []

for epoch in range(EPOCHS):
    perm = np.random.permutation(N)          # 手工打乱:每轮重新生成一次下标排列
    x_shuffled = x[perm]
    y_shuffled = y[perm]
    batch_losses = []
    for start in range(0, N, BATCH_SIZE):     # 手工切分:每 BATCH_SIZE 个样本一批
        xb = x_shuffled[start:start + BATCH_SIZE]
        yb = y_shuffled[start:start + BATCH_SIZE]
        with tf.GradientTape() as tape:
            y_pred = forward(xb)
            loss = mse_loss(yb, y_pred)
        grads = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(grads, [W, b]))
        batch_losses.append(float(loss))
    epoch_loss = sum(batch_losses) / len(batch_losses)
    epoch_losses.append(epoch_loss)
    print("epoch %d: loss=%.4f" % (epoch, epoch_loss))

print("W:", float(W.numpy()), "b:", float(b.numpy()))
print("first epoch loss:", epoch_losses[0], "last epoch loss:", epoch_losses[-1])

# 宽松断言:只要求"首末轮相差足够大 + 最终稳定在低位 + 参数收敛到真值附近"，
# 不卡实测的具体倍数——mini-batch SGD每轮都重新洗牌，具体数值会因batch组合抖动
assert epoch_losses[0] > 3.0
assert epoch_losses[-1] < 0.5
assert epoch_losses[0] > epoch_losses[-1] * 5
assert abs(float(W.numpy()) - true_w) < 0.3
assert abs(float(b.numpy()) - true_b) < 0.3
print("stage2 ok")
```

实测 15 行里挑几行有代表性的:`epoch 0: loss=4.5004` → `epoch 1: loss=0.2889` → `epoch 3: loss=0.2322` → `epoch 12: loss=0.2420` → `epoch 13: loss=0.2300` → `epoch 14: loss=0.2416`,最终 `W≈1.9555`、`b≈0.9461`,离真实的 `(2.0, 1.0)` 已经很接近。

**一个不回避的真实现象:loss 不是每一轮都比上一轮低。** `epoch 3` 到 `epoch 4` 从 0.2322 涨到 0.2398,`epoch 12` 到 `epoch 13` 又从 0.2420 降到 0.2300——收敛之后 loss 在 0.23~0.24 附近来回小幅波动,不是严格单调下降到底。原因不是训练出了问题,是 mini-batch SGD 的正常方差:每轮重新打乱下标,每个 batch 抽到的具体 20 个样本不一样,算出来的梯度是"整批梯度"的一个有噪声的估计,批次组合不同,这一轮的平均 loss 自然有起伏。真正该看的是**整体趋势**(首轮 4.50 到末轮附近稳定在 0.23~0.24,相差接近 19 倍)和**收敛水平**,不是逐轮是否单调。这个 0.23~0.24 附近的平台期也不是巧合:数据里加的噪声是 `Normal(0, 0.5)`,方差是 0.25——只要参数已经学到了真实的 `(2.0, 1.0)`,MSE loss 的期望值理论上就是这个噪声方差本身,不可能无限逼近 0,实测的 0.23~0.24 已经非常接近这个理论下限,不该期望它继续往下走。

---

## 阶段 3:引入 `tf.data.Dataset` 输入管道——替换手工切 batch

阶段 2 的手工切分能跑,但两个问题:`np.random.permutation` 打乱整个数组、`for start in range(0, N, BATCH_SIZE)` 手动算切片下标,这些代码本身和"训练逻辑"没关系,纯粹是数据管理的样板代码;数据一旦大到内存放不下,`x[perm]` 这种整体打乱的写法直接失效。[09 类](09-tf-data-pipeline.md)讲过的 `tf.data.Dataset` 正是为解决这类问题设计的——这里把手工切分换成 `from_tensor_slices` + `shuffle` + `batch` 三个链式调用。

**先验证一个 09 类只给出签名、没有现场验证过的细节:`shuffle` 的 `reshuffle_each_iteration` 参数不显式传值时,默认到底是不是"每次重新迭代都重新打乱"。**

```python
import tensorflow as tf
import numpy as np

np.random.seed(0)
tf.random.set_seed(0)

N = 200
x = np.random.uniform(-5, 5, size=(N, 1)).astype("float32")
true_w, true_b = 2.0, 1.0
noise = np.random.normal(0, 0.5, size=(N, 1)).astype("float32")
y = (true_w * x + true_b + noise).astype("float32")

BATCH_SIZE = 20
dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=N, seed=0).batch(BATCH_SIZE)

# 同一个dataset对象，重新走一遍迭代(相当于训练循环里进入下一个epoch)，
# 第一个batch的内容是不是真的会变——不能假设，现场验证
first_pass_first_batch = next(iter(dataset))[0].numpy().flatten().tolist()
second_pass_first_batch = next(iter(dataset))[0].numpy().flatten().tolist()
print("first pass first batch x[:3]:", first_pass_first_batch[:3])
print("second pass first batch x[:3]:", second_pass_first_batch[:3])
print("are they identical:", first_pass_first_batch == second_pass_first_batch)

# reshuffle_each_iteration 不显式传参时默认是True:每次重新完整迭代这个dataset(每个epoch)，
# shuffle都会重新算一次顺序，不是构造时定死一份排列反复复用
assert first_pass_first_batch != second_pass_first_batch
print("stage3 reshuffle check ok")
```

实测两次拿到的首批次前 3 个值完全不同(`[-4.81, 4.28, -4.13]` vs `[-1.02, -3.62, -3.41]`),`are they identical: False`——证明默认行为确实是"每次完整迭代都重新打乱",这一点在训练循环里至关重要:如果默认行为反过来是"只打乱一次、之后每个 epoch 用同一个顺序",多轮训练看到的批次组合会完全固定,起不到 [09 类第 3 节](09-tf-data-pipeline.md)讲的"打乱粒度"应有的效果。

**确认了这一点,再把完整的多轮训练循环搬过来,唯一的改动是数据管道换成 `tf.data.Dataset`:**

```python
import tensorflow as tf
import numpy as np

np.random.seed(0)
tf.random.set_seed(0)

N = 200
x = np.random.uniform(-5, 5, size=(N, 1)).astype("float32")
true_w, true_b = 2.0, 1.0
noise = np.random.normal(0, 0.5, size=(N, 1)).astype("float32")
y = (true_w * x + true_b + noise).astype("float32")

W = tf.Variable(0.0, name="W")
b = tf.Variable(0.0, name="b")
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

def forward(x_batch):
    return W * x_batch + b

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

BATCH_SIZE = 20
EPOCHS = 15

# 替换掉阶段2里手工的 np.random.permutation + 切片:shuffle负责打乱,batch负责切分,
# 顺序是先shuffle后batch(09类第3节验证过的顺序——打乱粒度是单个样本,不是batch出场顺序)
dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=N, seed=0).batch(BATCH_SIZE)

epoch_losses = []
for epoch in range(EPOCHS):
    batch_losses = []
    n_batches = 0
    for xb, yb in dataset:                    # 直接for循环迭代Dataset,不再需要手写下标切片
        with tf.GradientTape() as tape:
            y_pred = forward(xb)
            loss = mse_loss(yb, y_pred)
        grads = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(grads, [W, b]))
        batch_losses.append(float(loss))
        n_batches += 1
    epoch_loss = sum(batch_losses) / len(batch_losses)
    epoch_losses.append(epoch_loss)
    print("epoch %d: loss=%.4f n_batches=%d" % (epoch, epoch_loss, n_batches))

print("W:", float(W.numpy()), "b:", float(b.numpy()))
print("first epoch loss:", epoch_losses[0], "last epoch loss:", epoch_losses[-1])

# 和阶段2同样的宽松断言:训练效果应该一致——同样的数据、同样的batch size、同样的epoch数，
# 只是batch的产生方式从手工切片换成了tf.data管道
assert n_batches == N // BATCH_SIZE        # 200/20=10个batch,不多不少
assert epoch_losses[0] > 3.0
assert epoch_losses[-1] < 0.5
assert epoch_losses[0] > epoch_losses[-1] * 5
assert abs(float(W.numpy()) - true_w) < 0.3
assert abs(float(b.numpy()) - true_b) < 0.3
print("stage3 training ok")
```

实测:`epoch 0: loss=3.9480` → `epoch 6: loss=0.2250`(全程最低点)→ `epoch 14: loss=0.2364`,`n_batches` 每轮都是 10(200 个样本 ÷ 20 一批),最终 `W≈1.9646`、`b≈0.9450`。和阶段 2 的手工切分版本(`W≈1.9555`、`b≈0.9461`,最终 loss 落在 0.23~0.24 区间)对比:两条路径用的是同一份合成数据、同样的 batch size、同样的 15 轮,最终参数和收敛水平落在同一量级、同样接近真实的 `(2.0, 1.0)`——**"手工切分 batch"换成"`tf.data.Dataset` 管道"不改变训练效果,只是把打乱和切分这两件和训练逻辑无关的事,交给了一套更标准、更容易扩展(后续接 `.map()`/`.prefetch()`/`.cache()` 都是加一行的事,[09 类第 4/5 节](09-tf-data-pipeline.md)有更完整的展开)的机制去做。**

---

## 阶段 4:对照 `model.compile()`+`model.fit()`——手写的和框架自动跑的,应该学到同一件事

前三个阶段全程没有出现过 `tf.keras.Sequential`、没有调用过 `compile()`/`fit()`。最后验证一件事:这条从裸 `tf.Variable` 手写出来的路径,和 [08 类第 1/8 节](08-training-loop-internals.md)讲过的"全自动" `fit()` 路径,喂同样的数据、跑同样的轮数,是不是学到了同一件事。

```python
import tensorflow as tf
import numpy as np

np.random.seed(0)
tf.random.set_seed(0)

N = 200
x = np.random.uniform(-5, 5, size=(N, 1)).astype("float32")
true_w, true_b = 2.0, 1.0
noise = np.random.normal(0, 0.5, size=(N, 1)).astype("float32")
y = (true_w * x + true_b + noise).astype("float32")

W = tf.Variable(0.0, name="W")
b = tf.Variable(0.0, name="b")
optimizer_manual = tf.keras.optimizers.SGD(learning_rate=0.05)

def forward(x_batch):
    return W * x_batch + b

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

BATCH_SIZE = 20
EPOCHS = 15
dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=N, seed=0).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
    for xb, yb in dataset:
        with tf.GradientTape() as tape:
            y_pred = forward(xb)
            loss = mse_loss(yb, y_pred)
        grads = tape.gradient(loss, [W, b])
        optimizer_manual.apply_gradients(zip(grads, [W, b]))

manual_final_loss = float(mse_loss(y, forward(x)))
print("hand-written loop: W=%.4f b=%.4f final_loss=%.4f" % (float(W.numpy()), float(b.numpy()), manual_final_loss))

# 换成 model.compile()+model.fit():同样的数据、同样的batch size、同样的epoch数、同样的学习率
tf.random.set_seed(0)
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05), loss="mse")
history = model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, verbose=0)

fit_final_loss = history.history["loss"][-1]
fit_W = float(model.layers[0].kernel.numpy()[0, 0])
fit_b = float(model.layers[0].bias.numpy()[0])
print("model.fit(): W=%.4f b=%.4f final_loss=%.4f" % (fit_W, fit_b, fit_final_loss))

diff_W = abs(float(W.numpy()) - fit_W)
diff_b = abs(float(b.numpy()) - fit_b)
print("diff W:", diff_W)
print("diff b:", diff_b)

# 宽松断言:不要求数值完全相同(两条路径shuffle的具体实现细节不保证逐比特一致)，
# 只要求最终学到的参数、loss落在同一量级、彼此接近，并且都接近真实的(2.0, 1.0)
assert diff_W < 0.2
assert diff_b < 0.2
assert abs(manual_final_loss - fit_final_loss) < 0.2
assert abs(fit_W - true_w) < 0.3
assert abs(fit_b - true_b) < 0.3
print("stage4 ok")
```

实测:手写路径 `W=1.9829 b=0.9506 final_loss=0.2285`,`model.fit()` 路径 `W=1.9879 b=0.9439 final_loss=0.2364`——两个参数的差异分别只有约 0.005 和 0.007,loss 差异约 0.008,全部落在同一量级,都收敛到了真实的 `(2.0, 1.0)` 附近。**注意不是完全相同**:两条路径虽然用同一个 `tf.random.set_seed(0)` 起点,但 `dataset.shuffle()` 的随机流和 `model.fit(..., shuffle=True)` 内部的打乱实现不是同一套随机数消费顺序,不同批次组合会带来真实但很小的浮点误差累积——**"手写训练循环"和"`fit()` 自动跑"在数学上做的是同一件事,不代表它们逐比特复现同一个随机过程**,这一点在诚实描述"结果接近"的时候需要说清楚,不能含糊地说成"完全一样"。

---

## 可以怎么继续扩展(只指方向,不在本文实现)

- **给单步训练函数加 `@tf.function`**:[08 类第 3 节](08-training-loop-internals.md)现场测过纯 eager 手写循环比 `@tf.function` 包装慢 3 倍以上,本文四个阶段全程是纯 eager 写法,套用同一个优化思路,把 `with tf.GradientTape()...optimizer.apply_gradients(...)` 那几行包进一个 `@tf.function` 修饰的函数,应该能拿到同等量级的加速,[03 类](03-tf-function-and-autograph.md)有完整的机制展开。
- **换成一个简单分类问题**:把 `forward` 换成 `tf.sigmoid(W * x + b)`,`mse_loss` 换成 `tf.keras.losses.BinaryCrossentropy()`(呼应 [06 类](06-loss-functions-and-numerical-stability.md)对 `from_logits` 数值稳定性的讨论),数据换成两类可分的合成点,训练循环骨架完全不用改。
- **接上更完整的 `tf.data` 管道**:阶段 3 只用了 `shuffle`+`batch`,`.prefetch(tf.data.AUTOTUNE)`(呼应 [09 类第 5 节](09-tf-data-pipeline.md))、`.cache()`(呼应第 4 节)都是加一行的事,数据规模变大之后收益会更明显。
- **补一段验证集 + 早停等价逻辑**:`fit()` 免费提供的 `validation_data`/`EarlyStopping`(呼应 [08 类第 4/5/9 节](08-training-loop-internals.md))在手写循环里都要自己写等价判断,是"手写路径要为灵活性付出的代价"这条 08.3 已经讲过的结论的一次具体实践。
- **从一元线性回归换成多层网络**:把 `W * x_batch + b` 换成 `tf.keras.layers.Dense` 或手写的多层矩阵乘法链,前向/反向数学本身可以直接对照 [05 类](05-layers-math-and-backward.md)的推导。

这几个方向都不实现,是为了让这篇教程聚焦在"手写训练循环的四个环节怎么独立拼出来、又怎么和已经学过的 `tf.data`/`fit()` 对上"这一件事上——真要继续做下去,每个方向单独展开都够写一整篇。

## 这篇教程展示的方法论

任何一条已完成的深挖系列,都可以用同样的模式产出"教程体"内容:挑几个关联的知识点 → 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实效果,而不是一次性甩出完整代码。这一篇额外验证了一件事:教程体不只能"横向拼接"不同能力(像 dsa-deep-dive/21 拼倒排索引+Trie+堆),也能"纵向下钻"——先退到一个已讲过的机制(08.3 的手写训练循环)能覆盖到的最原始形态,不借助任何框架封装从零搭一遍,再回头对照框架封装版本,验证两者确实是同一件事的不同抽象层级。这种"纵向下钻"的方式,对任何一条已经讲过"某个高层 API 内部在做什么"的系列(不只是 tensorflow-deep-dive)都应该同样适用。

---

*创建:2026-07-24*
