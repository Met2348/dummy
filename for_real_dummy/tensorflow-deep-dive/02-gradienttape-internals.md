# 02 · GradientTape 自动微分机制深挖(GradientTape Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)
> **这是全系列权重最高的一批之一。** 本篇和 03(`tf.function`/AutoGraph)合计占全系列五分之一权重——这不是平均分配出来的数字:TF2 相对 PyTorch 真正的心智负担,不是"自动微分"这个概念本身(两边概念上都是 reverse-mode AD,很像),而是"这行 eager 代码到底会不会被记录、记录之后这条路径能不能求出梯度"这件事在 TF2 里不是 tensor 与生俱来的属性,需要你主动理解 tape 的记录规则——这正是本篇要打开的黑箱。面试聊到"TensorFlow 底层机制",十有八九会落在 GradientTape 上,而且经常不是"`.gradient()` 怎么调"这种问法,而是"你能不能解释清楚,这里的梯度为什么是 `None`——是没被 watch,是路径断了,还是这个 op 本来就不可导"。

**关于 01 篇的前置知识:** 本系列 01(Tensor 基础与 tf.Variable)按路线图规划尚未成稿,但本篇要用到的前置知识很单薄,这里直接说清楚:`tf.Variable` 是可变的、持有存储句柄的"长期对象"(模型参数的标准载体,有 `trainable` 属性);`tf.Tensor`(包括 `tf.constant` 和任何运算产生的中间结果)是不可变的"一次性值对象",没有 `trainable` 这个概念。这个区别是第 2 节"自动 watch 规则"成立的物理基础——先记住这一句,后面会反复用到。

本文所有代码例子已在 WSL2 `~/tf-venv`(TensorFlow **2.21.0**,GPU 可用,环境细节见 [00-roadmap.md](00-roadmap.md) 第 0 节)下实际跑通验证。所有报错信息都是现场触发后原样抄录,不是转述文档或凭经验断言;所有数值结果都经过手算或多种方式交叉验证。"AI 研究/工程场景"段落如实写场景化例子,但仓库 `learning/` 目录下没有 TensorFlow 代码可引用——这是本系列统一声明过的诚实边界(见 00-roadmap.md),后文不再逐条重复。

**本篇统一结构(与 00-roadmap.md 定义的 7 段式一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计** —— 本系列的核心加深点,不停在"怎么用"
4. AI 研究/工程场景
5. 可运行例子(能内省的地方现场打印内部状态,不要求你相信文字描述)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. GradientTape 的工作原理 —— 记录哪些运算、怎么记录

**是什么:**
```text
with tf.GradientTape() as tape:
    ...   # 这个代码块内执行的TF运算,只要输入被tape"watch"着,就会被记录下来
grad = tape.gradient(target, sources)   # 之后据此从target出发反向遍历记录,算出对sources的梯度
```
（这是签名/用法示意,不是完整可运行代码——`target`/`sources` 是占位符,真正的可运行验证见本节"可运行例子"。后文每节"是什么"下的代码块同理,不再逐条注明。）

**一句话:** GradientTape 是 TF2 eager 模式下的运行时"录音带"——不是 tensor 自带的属性,而是一个有生命周期的外部监听者:只有一次运算恰好发生在某个 tape 的作用域内、且这次运算的输入恰好在这个 tape 当时的监听(watch)范围内,这次运算才会被记下来,留着以后反向用。

**底层机制/为什么这样设计:**

TF2 默认是 eager 执行:每一行代码都是普通 Python 语句,立即执行、立即返回具体数值(这一点第 10 节会和 PyTorch 详细对比,这里先只关注"记录"这一件事,不要和 TF1 静态图混为一谈——那是完全不同的话题,见 00-roadmap.md 对 03 篇的介绍)。eager 模式天然不带任何计算历史——一个 tensor 被产生出来之后并不会自动知道"我是怎么来的",不像 TF1 图模式那样图结构提前定义好、和真实数值分离。

TF 的解法是把"要不要构建反向路径"这件事,从"tensor 自己的属性"变成"当前有没有一个活跃的 tape 正在监听"。`tf.GradientTape()` 通过 Python 的上下文管理器协议(`__enter__`/`__exit__`),在进入时把自己注册为"当前活跃的 tape";这期间执行的每一个 TF 原生 op,内部都会检查自己的输入里有没有正在被某个活跃 tape 监听的 tensor——如果有,就把这次运算记一笔:op 类型是什么、反向公式真正需要的中间量是什么(不是无脑存全部输入输出——`y=x**2` 反向需要 x 的值,`y=exp(x)` 反向只需要 y 的值,这和 PyTorch `save_for_backward` 只存反向真正用得到的量是同一个道理)。反向阶段调用 `tape.gradient(target, sources)` 时,才真正从 target 出发反向遍历这些记录,对每条记录按 op 类型查表调用对应的反向公式——这张"op 名 → 反向函数"的注册表,是 TF 图模式和 eager 模式共用的同一套东西(第 8 节会展开这一点)。

这带来一个已经用代码验证过的直接推论:**"记录"检查的是"op 执行的那一刻",不是"`gradient()` 调用的那一刻"**——如果一次运算发生时,它的输入还没有进入任何 tape 的监听范围,这次运算就不会被记录,即使之后同一个 tape 里补调用了 `watch()`,这次已经"错过"的运算也补救不了。这不是什么反直觉的怪癖,而是"tape 是被动监听器,不是主动扫描器"这个设计的直接推论——tape 没有(也不需要)一个全局机制去"倒查"哪些变量在之前的某次运算里被用到过,它只能老老实实盯着每个 op 执行的瞬间做判断。这也解释了为什么"tape 只记录、不重放"——tape 内部维护的是一份和"这一次实际执行路径"精确对应的记录,而不是一份"这个 Python 函数理论上可能怎么执行"的静态描述,这一点第 10 节会看到和 PyTorch 完全同构(都是 define-by-run),只是"记录"的触发条件不同。

顺带一提:多个 tape 可以同时活跃、彼此独立地记录同一段代码(第 4 节的嵌套 tape 算高阶导数,正是利用这一点——内层 tape 和外层 tape 是两个完全独立的记录本,各自维护自己的监听范围)。

**AI 研究/工程场景:** TF2 自定义训练循环的标准写法——`with tf.GradientTape() as tape: logits = model(x, training=True); loss = loss_fn(y, logits)`,退出 `with` 块之后 `tape.gradient(loss, model.trainable_variables)`;可解释性研究里对输入求梯度(saliency map:哪些输入像素对输出影响大,梯度绝对值大的地方就是关键区域)、对抗样本生成(FGSM 沿着 loss 对输入图像的梯度符号方向做一步扰动)都要求先对输入显式 `tape.watch()`,因为输入通常是普通 tensor 而不是 Variable(第 2 节详细讲这个区别)。

**可运行例子:**
```python
import tensorflow as tf

x = tf.Variable(3.0, name="x")
with tf.GradientTape() as tape:
    y = x ** 2
grad = tape.gradient(y, x)
print("grad:", grad.numpy())
assert grad.numpy() == 6.0

# tape作用域外执行的运算,完全不产生任何记录 —— 不是"记录了但没保留",而是压根没发生过记录
z = x ** 2
print("tape之外的z:", type(z).__name__)
assert type(z).__name__ == "EagerTensor"   # 就是个普通eager tensor,没有任何autograd相关的额外信息
```

```python
import tensorflow as tf

# 验证"记录"检查的是op执行的瞬间,不是gradient()调用的瞬间:
# watch()补调用之前发生的运算,不会被追溯记录
xm = tf.Variable(4.0, name="xm")
with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
    tmp = xm * 2        # 此刻xm还没被watch,这次运算不会被记录
    tape.watch(xm)       # 从这一刻起,xm才进入监听范围
    ym = xm ** 2          # 这次运算发生在watch()之后,会被记录

grad_tmp = tape.gradient(tmp, xm)
print("watch()之前的运算,事后也求不出梯度:", grad_tmp)
assert grad_tmp is None

grad_ym = tape.gradient(ym, xm)
print("watch()之后的运算,梯度正常:", grad_ym.numpy())
assert grad_ym.numpy() == 8.0   # d(x^2)/dx = 2x = 8
del tape
```

**面试怎么问 + 追问链:**
- **Q:** "`tf.GradientTape` 是怎么知道该给哪些运算构建反向路径的?"—— 期望答出"不是 tensor 自带属性,而是看这次运算执行时,有没有一个活跃的 tape 正在监听它的输入",而不是含糊地说"tape 会自动追踪"。
- **追问 1:** "如果一次运算发生在没有任何活跃 tape 的情况下,之后还能不能想办法补一个梯度出来?"—— 期望明确回答"不能,没有 tape 监听的时候压根没有产生任何记录,不是'记录丢失了',而是'这条记录从没存在过',唯一的办法是把这次运算重新放到一个活跃的 tape 里再跑一遍"。
- **深挖追问(区分度很高):** "`tape.watch(x)` 补调用之后,`x` 之前已经参与过的运算能不能被追溯记录进来?"—— 期望候选人能想到用代码验证(就是上面的例子),而不是猜;能说出"不能,tape 检查的是 op 执行那一刻的监听状态,watch() 只影响它被调用之后发生的运算"是加分项。
- **追问 2:** "两个 `GradientTape` 可以同时嵌套/并存吗?它们的记录是共享的还是独立的?"—— 期望说出"完全独立,各自维护自己的记录和监听范围",并能联系到第 4 节嵌套 tape 算高阶导数正是利用这个独立性。

**常见坑:** 把"这段代码写在 `with tf.GradientTape()` 缩进块里"和"这次运算一定会被记录"划等号——还要看输入是否被监听(第 2 节);在 tape 外面先算好一个中间结果,再拿进 tape 里当输入用,这个中间结果自己的产生过程(它是怎么由更早的 tensor 算出来的)完全不在任何记录里,即使它本身在 tape 内被使用;`watch()` 调用得太晚,已经执行过的运算无法被追溯,这是调试"为什么明明 watch 了梯度还是 None"时最容易被忽略的时序问题。

---

## 2. `watch()` 与自动 watch 规则 —— trainable Variable 自动被追踪,其余需要显式声明

**是什么:**
```text
tape.watch(tensor)   # 显式把一个tensor(或Variable)加入tape的监听范围,
                       # 之后对它求梯度才有意义

tf.GradientTape(watch_accessed_variables=True)  # 默认值:trainable=True的Variable,
                       # 只要在tape作用域内被读取过,自动进入监听范围,不需要手动watch()
```

**一句话:** `trainable=True` 的 `tf.Variable`,只要在 tape 作用域内被**读取**过,就自动被监听,不需要手动 `watch()`;除此之外的一切——普通 `tf.Tensor`(包括 `tf.constant` 和任何运算的中间结果)、`trainable=False` 的 `Variable`——都必须显式调用 `tape.watch()` 才能求出梯度,否则 `tape.gradient()` 默默返回 `None`(不报错、不警告)。

**底层机制/为什么这样设计:**

为什么 Variable 和普通 Tensor 待遇不同?—— Variable 在 TF 里代表的是"模型的可训练参数"这样一个长期存在、语义明确"这就是需要被优化的对象"的实体。绝大多数训练代码的核心诉求就是"对所有 trainable variables 求梯度",如果每次都要对着几十上百个变量手写 `tape.watch()`,是纯粹的样板代码负担,所以 TF 选择默认自动化这条最常见路径。但普通 Tensor 是"一次性"的——每次运算都产生新的 tensor 对象,如果默认监视所有 tensor,tape 的记录量会不受控制地膨胀,失去"tape 只记录真正需要反向的路径"这个性能设计的意义,而且绝大多数中间 tensor 根本不需要对它求梯度(通常只关心最终对 trainable variables 的梯度,不是对每一个中间激活值求梯度)。所以 TF 选的是"保守默认 + 显式声明"策略:除了 trainable Variable 这一个最常见场景,其余一律要求你明确告诉 tape "这个我要"。

"自动被 watch"的触发时机是"该 Variable 在 tape 作用域内被**读取**(访问其值参与运算)的那一刻",而不是"tape 一进入 `with` 块就扫描全局揪出所有 trainable variable"。这是因为 tape 采取的是第 1 节讲的被动监听模式,它没有(也不需要)一份全局变量清单可供主动扫描。直接推论:如果一个 trainable Variable 在 tape 内部从未被用到,它不会出现在 `tape.watched_variables()` 里,对它求梯度得到 `None`,这个 `None` 和"故意不想要它的梯度"在表现上完全一样——不会有任何提示告诉你"这个变量你是不是忘记用了"。

`watch_accessed_variables=False` 这个开关存在的意义:某些场景你不想要"自动"这个行为——比如只想对模型上百个参数里的少数几个求梯度,显式列出这几个能让代码意图更清晰,也避免"不小心"读到了一个不想要梯度的变量却被自动记录、产生无用的记录开销。这个模式下,即使是 `trainable=True` 的 Variable 也**不会**被自动监听,必须无差别地全部显式 `watch()`。

**AI 研究/工程场景:** 冻结部分网络参数做迁移学习(只训练最后几层)时,如果不小心让 `tape` 记录了不该训练的那部分 Variable(比如它们的 `trainable` 忘了设成 `False`),自动 watch 规则会让它们"顺手"被求出梯度、参与更新,这是一个隐蔽的正确性 bug;对输入图像做梯度分析(saliency/FGSM,呼应第 1 节)时,输入通常是普通 `tf.Tensor`(来自 `tf.data` 管道或 `tf.constant` 包装的一批图片),必须显式 `tape.watch(input_tensor)`,这是新手最常在这类代码里漏掉的一步,表现为"梯度全是 None,报错都没有,一脸懵"。

**可运行例子:**
```python
import tensorflow as tf

# trainable Variable: 只要在tape内被读取,自动被watch
x = tf.Variable(3.0, name="x")
with tf.GradientTape() as tape:
    y = x ** 2
watched_names = [v.name for v in tape.watched_variables()]
print("watched_variables:", watched_names)
assert "x:0" in watched_names
grad = tape.gradient(y, x)
assert grad.numpy() == 6.0

# 从未在tape内被读取的trainable Variable —— 不会自动被watch,梯度是None(不报错不警告)
x2 = tf.Variable(3.0, name="x2")
w_untouched = tf.Variable(5.0, name="w_untouched")
with tf.GradientTape() as tape2:
    y2 = x2 * 3   # w_untouched全程没被用到
grad_w = tape2.gradient(y2, w_untouched)
print("从未被读取的trainable变量,梯度:", grad_w)
assert grad_w is None
```

```python
import tensorflow as tf

# 普通常量tensor: 默认不被watch
c = tf.constant(2.0)
with tf.GradientTape() as tape:
    yc = c ** 2
grad_c = tape.gradient(yc, c)
print("常量未显式watch,梯度:", grad_c)
assert grad_c is None

with tf.GradientTape() as tape2:
    tape2.watch(c)
    yc2 = c ** 2
grad_c2 = tape2.gradient(yc2, c)
print("常量显式watch后,梯度:", grad_c2.numpy())
assert grad_c2.numpy() == 4.0

# trainable=False的Variable: 同样不会被自动watch
v_ntr = tf.Variable(3.0, trainable=False, name="v_ntr")
with tf.GradientTape() as tape3:
    y3 = v_ntr ** 2
assert tape3.gradient(y3, v_ntr) is None

with tf.GradientTape() as tape4:
    tape4.watch(v_ntr)
    y4 = v_ntr ** 2
grad4 = tape4.gradient(y4, v_ntr)
assert grad4.numpy() == 6.0

# watch_accessed_variables=False: 连trainable=True也不再自动watch
xw = tf.Variable(5.0, name="xw")
with tf.GradientTape(watch_accessed_variables=False) as tape5:
    yw = xw ** 2
assert tape5.gradient(yw, xw) is None    # 自动watch被关闭,即使xw本身trainable=True

with tf.GradientTape(watch_accessed_variables=False) as tape6:
    tape6.watch(xw)
    yw2 = xw ** 2
grad6 = tape6.gradient(yw2, xw)
assert grad6.numpy() == 10.0
```

**面试怎么问 + 追问链:**
- **Q:** "`GradientTape` 默认会追踪哪些 tensor 的梯度?普通的 `tf.constant` 需要额外做什么才能对它求梯度?"—— 期望答出"trainable Variable 在 tape 内被读取会自动 watch;普通 tensor 一律要显式 `tape.watch()`"。
- **追问 1:** "如果我漏掉了一个该 watch 的 tensor,`tape.gradient()` 会报错提醒我吗?"—— 期望"不会,默默返回 `None`,这是实践中排查'梯度消失'类 bug 时第一个要检查的疑点"。
- **深挖追问(区分度很高):** "一个 `trainable=True` 的 Variable,如果在 tape 作用域内从未被读取过,它会不会出现在 `tape.watched_variables()` 里?为什么?"—— 期望理解自动 watch 的触发条件是"被读取的那一刻",不是"tape 进入时的一次性全局扫描",这个变量因为从未被访问,tape 根本不知道它的存在,所以不会出现在监听列表里——能用 `tape.watched_variables()` 现场验证是加分项。
- **追问 2:** "`watch_accessed_variables=False` 存在的意义是什么,什么场景会用到?"—— 期望说出"想显式控制求梯度的范围,避免默认自动 watch 带来的'意外'记录,常见于只想训练模型一部分参数、需要精确控制的场景"。

**常见坑:** 对着一个普通 `tf.Tensor`(尤其是输入数据、或者某个中间计算结果)直接求梯度却忘了 `watch()`,得到 `None` 却当成"这条路径本来就不可导"来排查,方向完全错了;以为"变量存在于 tape 作用域内的某个地方"就等于"被 watch 了"——决定性因素是"有没有被读取",不是"是否在同一个 `with` 块的某处出现过";混淆"没被 watch"和"路径断开"这两种都会导致 `None` 的不同原因(第 9 节会把三种 `None` 的来源系统对比清楚)。

---

## 3. `persistent=True` 与 tape 的生命周期 —— 默认 tape 只能用一次

**是什么:**
```text
tape = tf.GradientTape()                 # 默认: persistent=False,只能调用一次.gradient()
tape = tf.GradientTape(persistent=True)   # 允许对同一个tape多次调用.gradient()/.jacobian()
```

**一句话:** 默认(`persistent=False`)一个 tape 只允许调用一次 `gradient()`(或 `jacobian()`),调用完之后 tape 内部记录占用的资源就被释放;如果需要对同一份记录多次求梯度(比如多个不同 target 分别求梯度),必须传 `persistent=True`,用完后官方建议手动 `del tape` 尽早释放。

**底层机制/为什么这样设计:**

tape 记录期间,为了让反向阶段能正确算出梯度,必须持有前向过程中一些中间结果的引用(和第 1 节讲的"反向公式真正需要的中间量"是同一份东西,例如 `y=exp(x)` 反向要用到 `y` 本身)。这些引用不会随着 Python 变量离开作用域就自动消失,是 tape 对象自己拿着的。默认情况下,一次 `gradient()` 调用完成后,TF 会主动释放这些资源——这是"多数场景只需要一次反向"的默认优化,不用用户操心显存管理。用代码验证这个限制,报错是:

```
A non-persistent GradientTape can only be used to compute one set of gradients (or jacobians)
```

`persistent=True` 关闭这个自动释放行为,让同一份记录能被多次复用去调用 `gradient()`——这在"一个前向过程被多个独立的 loss 分别求梯度"这类场景是刚需(比如想分别观察几个子 loss 各自对参数的独立贡献,而不是先加权求和成一个标量再统一求一次梯度)。因为 `persistent=True` 意味着资源不会被自动释放,官方文档明确建议用完后手动 `del tape`,靠 Python 垃圾回收显式尽快清理,不要等它超出作用域被动收集——这是显存管理的最佳实践,不是可有可无的建议。

有两个容易被忽略的生命周期细节,都已用代码验证过:第一,**退出 `with` 块本身并不会让非 persistent 的 tape "作废"**——`with` 块只是停止继续记录新的运算,tape 对象在块外依然可以正常调用一次 `gradient()`;第二,tape 还提供了两个精细控制生命周期的方法:`tape.reset()` 会清空这个 tape 目前为止记录的全部历史(但 tape 对象本身继续存活、可以继续记录新的运算,`reset()` 之前的旧 target 求梯度会得到 `None`,因为记录已经不在了);`tape.stop_recording()` 是一个可以嵌套在 `with tf.GradientTape()` 内部的上下文管理器,进入这个子块之后发生的运算**不会**被记录,退出子块后又恢复正常记录——适合用来在训练循环里顺手算一些不需要梯度的监控指标,而不用为此专门跳出 tape 作用域。

**AI 研究/工程场景:** 多任务学习里一个共享 encoder 输出被多个任务头分别计算 loss,想要分别看每个任务 loss 对 encoder 参数的独立梯度范数(而不是先加权求和),需要 `persistent=True` 对同一次前向的记录反复取梯度;强化学习里在同一个 rollout 的 tape 作用域内,除了要反传的 loss,还想顺手打印一些不参与梯度更新的诊断量(比如某个中间 value 的统计量),用 `tape.stop_recording()` 包一下能避免这些计算被无谓地记录进反向路径。

**可运行例子:**
```python
import tensorflow as tf

# 默认非persistent: 第二次gradient()报错(现场触发,原样抄录)
x = tf.Variable(3.0, name="x")
with tf.GradientTape() as tape:
    y = x ** 2
    z = x ** 3
g1 = tape.gradient(y, x)
print("g1:", g1.numpy())
try:
    g2 = tape.gradient(z, x)
    assert False
except RuntimeError as e:
    print("第二次gradient()报错:", str(e))
    assert "A non-persistent GradientTape can only be used to compute one set of gradients" in str(e)

# persistent=True: 可以多次调用
x2 = tf.Variable(3.0, name="x2")
with tf.GradientTape(persistent=True) as tape2:
    y2 = x2 ** 2
    z2 = x2 ** 3
g1b = tape2.gradient(y2, x2)
g2b = tape2.gradient(z2, x2)
print("persistent, g1b:", g1b.numpy(), " g2b:", g2b.numpy())
assert g1b.numpy() == 6.0
assert g2b.numpy() == 27.0
del tape2   # 官方建议: persistent tape用完手动del,尽早释放

# 退出with块本身不会让非persistent tape失效,块外依然能调用一次gradient()
x3 = tf.Variable(5.0, name="x3")
with tf.GradientTape() as tape3:
    y3 = x3 ** 2
g3 = tape3.gradient(y3, x3)   # 在with块外调用,依然是"这个tape的第一次也是唯一一次"
print("with块外调用gradient():", g3.numpy())
assert g3.numpy() == 10.0
```

```python
import tensorflow as tf

# reset(): 清空已记录的历史,tape对象本身可以继续用
x = tf.Variable(2.0, name="x")
with tf.GradientTape(persistent=True) as tape:
    y_before = x ** 2
    tape.reset()             # 忘掉目前为止的一切记录
    y_after = x ** 3
g_after = tape.gradient(y_after, x)
print("reset后, 新target的梯度:", g_after.numpy())
assert g_after.numpy() == 12.0   # d(x^3)/dx = 3x^2 = 12

g_before = tape.gradient(y_before, x)   # 旧target的记录已经被reset()清空
print("reset后, 旧target的梯度:", g_before)
assert g_before is None
del tape

# stop_recording(): 临时暂停记录,退出子块后恢复
x2 = tf.Variable(2.0, name="x2")
with tf.GradientTape() as tape2:
    y2 = x2 * 3
    with tape2.stop_recording():
        untracked = y2 * 100   # 这次运算不会被记录
    final = y2 * 5              # 恢复正常记录
g_final = tape2.gradient(final, x2)
print("stop_recording前后正常路径的梯度:", g_final.numpy())
assert g_final.numpy() == 15.0   # d(x*3*5)/dx = 15

x3 = tf.Variable(2.0, name="x3")
with tf.GradientTape() as tape3:
    y3 = x3 * 3
    with tape3.stop_recording():
        untracked3 = y3 * 100
g_untracked = tape3.gradient(untracked3, x3)   # untracked3这个tensor自身的产生过程没被记录
print("对stop_recording内部产生的tensor求梯度:", g_untracked)
assert g_untracked is None
```

**面试怎么问 + 追问链:**
- **Q:** "为什么 `GradientTape` 默认只能调用一次 `gradient()`?什么场景需要 `persistent=True`?"—— 期望答出"默认释放中间结果省显存,多个独立 loss 分别对同一次前向求梯度时需要 persistent"。
- **追问 1:** "`persistent=True` 用完之后要注意什么?"—— 期望"资源不会自动释放,官方建议手动 `del tape` 尽早回收,不然中间结果会一直占着显存直到 tape 对象被垃圾回收"。
- **深挖追问(区分度很高):** "退出 `with tf.GradientTape() as tape:` 代码块之后,这个 tape 是不是就不能用了?"—— 期望能推理或验证出"退出 `with` 块只是停止记录新运算,tape 对象依然存活,在块外调用一次 `gradient()` 完全没问题——'用一次就废'指的是调用 `gradient()` 这个动作,不是离开代码块这个动作",这是一个专门筛掉"死记规则、没理解机制"候选人的问题。
- **追问 2:** "`tape.reset()` 和创建一个全新的 `GradientTape()` 有什么区别?"—— 期望说出"`reset()` 是同一个 tape 对象清空历史记录、继续复用(通常配合 `persistent=True` 在循环里避免反复创建新对象),效果上接近但不是"新建",tape 本身的配置(比如 `persistent`)不变"。

**常见坑:** 忘记 `persistent=True` 就想对同一个 tape 多次求梯度,拿到一脸懵的 `RuntimeError`;`persistent=True` 用完不清理,在循环里反复创建 persistent tape 却不 `del`,导致显存缓慢泄漏(这是"能跑但迟早 OOM"类问题里比较隐蔽的一种);混淆 `tape.reset()`(清空记录、tape 本身还活着)和"tape 已经被释放不能用"(非 persistent 调用完 `gradient()` 之后的状态)——两者表现都是"旧记录查不到了",但一个是主动清空、可以继续用,一个是资源已经释放、整个 tape 报废。

---

## 4. 高阶导数 —— nested GradientTape 计算二阶及以上导数

**是什么:**
```text
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        y = f(x)
    dy_dx = inner.gradient(y, x)
d2y_dx2 = outer.gradient(dy_dx, x)   # 对"一阶导数"这个tensor再求一次梯度,得到二阶导数
```

**一句话:** 把一个 `GradientTape` 嵌套在另一个 `GradientTape` 内部,只要内层求梯度这个动作本身也发生在外层 tape 的记录范围内,外层就能把"求梯度"这件事也当成一次可以再求导的运算,对着一阶导数再来一次 `gradient()`,得到二阶导数;嵌套多少层,就能求多少阶导数。

**底层机制/为什么这样设计:**

`inner.gradient(y, x)` 内部执行的一系列 TF 运算(反向传播本质上也是一堆 tensor 运算:乘法、加法……不是什么脱离计算之外的魔法)。如果这些运算恰好发生在外层 tape 的监听范围内——外层 tape 会像记录任何普通前向运算一样,把这些"用来算梯度的运算"也记录下来。这正是第 1 节"tape 只关心 op 执行那一刻是否处于监听状态"这条规则的直接应用:tape 并不区分"这是用户业务逻辑的前向计算"还是"这是另一个 tape 内部在算反向传播"——对它来说都只是发生在自己监听范围内的普通运算。这和 PyTorch `create_graph=True` 的原理是同一件事(让"求梯度"这个过程本身变得可微),只是 PyTorch 用一个参数开关表达,TF 用"这段代码是否物理地被套在另一层 tape 里"表达。

有一个已验证的关键约束:**两层 tape 都需要各自"看到" `x`**。如果 `x` 是 trainable Variable,两层会各自自动 watch(第 2 节的自动 watch 规则是逐个 tape 独立生效的,不会"传染"给外层或内层);如果 `x` 是普通 tensor,内层和外层都要各自显式 `watch(x)`——只在内层 watch、忘了外层,会导致外层 `gradient()` 拿到 `None`(因为外层的角度看,`dy_dx` 虽然是从 `x` 算出来的,但外层从来没把 `x` 纳入监听范围,反向遍历不到这条路径的终点)。这是实践中最容易踩的坑,已经用代码触发验证过。

**AI 研究/工程场景:** 元学习(MAML 等算法,呼应 torch-deep-dive/02 第 8 节讲的同一类场景)需要对"内层任务适应一步梯度更新"这个过程本身再求梯度,来更新外层的元参数;物理仿真/PINN(physics-informed neural network)里损失函数直接包含对输入的二阶偏导数(比如 PDE 里的拉普拉斯算子 ∂²u/∂x²),需要嵌套 tape 现算;二阶优化方法(比如需要 Hessian 或 Hessian-vector product 的场景)同样以此为基础(第 7 节 `jacobian` 配合嵌套 tape 就能求出完整 Hessian 矩阵)。

**可运行例子:**
```python
import tensorflow as tf

# y=x^3: dy/dx=3x^2, d2y/dx2=6x, 在x=2处两者数值恰好都是12(用来交叉验证不是巧合算错)
x = tf.Variable(2.0, name="x")
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        y = x ** 3
    dy_dx = inner.gradient(y, x)
d2y_dx2 = outer.gradient(dy_dx, x)
print("Variable自动watch两层: dy_dx =", dy_dx.numpy(), " d2y_dx2 =", d2y_dx2.numpy())
assert dy_dx.numpy() == 12.0
assert d2y_dx2.numpy() == 12.0

# 普通常量tensor: 内外两层都要显式watch
c = tf.constant(2.0)
with tf.GradientTape() as outer2:
    outer2.watch(c)
    with tf.GradientTape() as inner2:
        inner2.watch(c)
        y2 = c ** 3
    dy_dx2 = inner2.gradient(y2, c)
d2y_dx2_2 = outer2.gradient(dy_dx2, c)
print("常量两层都watch: dy_dx =", dy_dx2.numpy(), " d2y_dx2 =", d2y_dx2_2.numpy())
assert d2y_dx2_2.numpy() == 12.0
```

```python
import tensorflow as tf

# 常见坑现场验证: 只在内层watch,外层忘记watch -> 外层梯度是None
c3 = tf.constant(2.0)
with tf.GradientTape() as outer3:
    with tf.GradientTape() as inner3:
        inner3.watch(c3)
        y3 = c3 ** 3
    dy_dx3 = inner3.gradient(y3, c3)
    # outer3从未调用outer3.watch(c3)!
d2_missing = outer3.gradient(dy_dx3, c3)
print("外层忘记watch, 二阶导数:", d2_missing)
assert d2_missing is None

# 三重嵌套求三阶导数: y=x^4, y'=4x^3, y''=12x^2, y'''=24x, 在x=2处: 32, 48, 48
x4 = tf.Variable(2.0, name="x4")
with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t3:
            y4 = x4 ** 4
        d1 = t3.gradient(y4, x4)
    d2 = t2.gradient(d1, x4)
d3 = t1.gradient(d2, x4)
print("三重嵌套(4次方): d1=", d1.numpy(), " d2=", d2.numpy(), " d3=", d3.numpy())
assert d1.numpy() == 32.0
assert d2.numpy() == 48.0
assert d3.numpy() == 48.0
```

**面试怎么问 + 追问链:**
- **Q:** "TensorFlow 里怎么算二阶导数?"—— 期望答出"嵌套 GradientTape,内层求一阶导,外层再对这个一阶导求一次梯度"。
- **追问 1:** "外层 tape 是怎么'看懂'内层 `gradient()` 这个调用的?它不是一个普通的赋值语句吗?"—— 期望理解"`gradient()` 内部本身也是一系列 TF 运算(反向传播的具体计算步骤),只要这些运算发生在外层 tape 的监听范围内,就会像任何其他运算一样被正常记录"。
- **深挖追问(区分度很高):** "如果 `x` 是一个普通 `tf.constant` 而不是 `Variable`,嵌套 tape 求二阶导数的时候,`watch()` 要调用几次、分别在哪里调用?"—— 期望准确答出"内层和外层都要各自调用一次 `watch(x)`,只 watch 一层会导致另一层的梯度是 `None`",最好能现场解释原因(每层 tape 的监听范围互相独立,不会自动传递)而不是死记"要 watch 两次"这条规则。
- **追问 2:** "求 n 阶导数需要嵌套几层 tape?"—— 期望"n 层",并能用三重嵌套求三阶导数的例子现场佐证。

**常见坑:** 只在内层 `watch()`,忘记外层也要 `watch()` 同一个 tensor(trainable Variable 不受此坑影响,因为两层都会各自自动 watch);把"二阶导数"和"对`.gradient()`的返回值重复调用两次"搞混——必须是**嵌套的 tape 结构**,而不是对同一个 tape 连续调用两次 `gradient()`(后者第二次调用还会撞上第 3 节讲的"非 persistent tape 只能用一次"的限制);嵌套层数和要求的导数阶数对不上,少一层就少求一阶。

---

## 5. `tf.stop_gradient` —— 阻断指定路径的梯度传播

**是什么:**
```text
y_stopped = tf.stop_gradient(y)   # 前向数值上y_stopped完全等于y(恒等映射),
                                    # 但反向传播时,任何经过y_stopped的路径到此为止,
                                    # 不会继续往y的输入回传
```

**一句话:** `tf.stop_gradient` 是 TF 里"这条路径到此为止"的显式声明——前向恒等、反向截断,概念上等价于 PyTorch 的 `.detach()`,但是函数式用法(包一层调用,不是 tensor 的方法),常见于图/eager 通用的代码路径里。

**底层机制/为什么这样设计:**

`tf.stop_gradient` 本质上是一个特殊 op:前向计算是恒等映射(输出数值 = 输入数值,验证过完全相等),但它在"op 类型 → 反向函数"这张注册表里(第 1 节提到的那张表),对应的反向函数被显式定义成"不再往输入传"。这意味着从 tape 记录的角度看,`stop_gradient` 这个 op 本身**照样会被记录**(它确实执行了、确实产生了一个新 tensor),只是它自己的反向定义就是"此路不通"——这和"tensor 根本没被 watch、无路可走"是两种表现相似但成因不同的 `None`(第 2 节的 `None` 是"压根没有记录",这里的 `None` 是"有记录,但记录本身规定了在这里截断"),第 9 节会把这几种 `None` 的来源系统对比。

**AI 研究/工程场景:** Straight-Through Estimator(STE)——量化训练/离散化操作(比如把连续值 round 成整数)前向必须用不可导的操作(`tf.round`),但反向想让梯度"假装"是恒等映射穿过去继续更新参数,标准写法是 `x + tf.stop_gradient(round(x) - x)`(前向数值等于 `round(x)`,反向梯度等于对 `x` 直接求导的结果,即 1);EMA/target network(强化学习 DQN、自监督学习 BYOL 这类算法)在计算 target 相关的量时,需要确保梯度不会意外从这条路径回传去污染另一部分参数,常见做法是显式 `stop_gradient` 或者干脆用 `trainable=False` 的影子变量(第 2 节的场景)。

**可运行例子:**
```python
import tensorflow as tf

x = tf.Variable(3.0, name="x")

# 基线: 不加stop_gradient
with tf.GradientTape() as tape:
    y = x ** 2
    z = y * 3
grad = tape.gradient(z, x)
print("无stop_gradient, grad:", grad.numpy())
assert grad.numpy() == 18.0   # dz/dx = 3 * 2x = 18

# 加stop_gradient: 经过y的路径被切断
with tf.GradientTape() as tape2:
    y2 = x ** 2
    y2_stopped = tf.stop_gradient(y2)
    z2 = y2_stopped * 3
grad2 = tape2.gradient(z2, x)
print("有stop_gradient, grad:", grad2)
assert grad2 is None

# 前向数值完全不受影响,只是反向被截断
print("y2 vs y2_stopped 数值:", y2.numpy(), y2_stopped.numpy())
assert y2.numpy() == y2_stopped.numpy() == 9.0
```

```python
import tensorflow as tf

# straight-through estimator: 前向像round(),反向像identity
x = tf.Variable(2.7, name="x")
with tf.GradientTape() as tape:
    rounded = tf.round(x)
    ste = x + tf.stop_gradient(rounded - x)   # 前向数值 = rounded,反向梯度 = 对x直接求导(=1)
grad = tape.gradient(ste, x)
print("STE前向值:", ste.numpy(), " STE反向梯度:", grad.numpy())
assert ste.numpy() == 3.0
assert grad.numpy() == 1.0

# 对比: 直接对round()求梯度 —— 这个op被注册为不可导, 梯度是None(不是0, 是None)
x2 = tf.Variable(2.7, name="x2")
with tf.GradientTape() as tape2:
    rounded2 = tf.round(x2)
grad2 = tape2.gradient(rounded2, x2)
print("naive round()的梯度:", grad2)
assert grad2 is None

# stop_gradient只切断"它包住的那一条"路径,同级的其他路径不受影响
x3 = tf.Variable(3.0, name="x3")
w = tf.Variable(5.0, name="w")
with tf.GradientTape() as tape3:
    a = x3 ** 2
    b = w * 2
    combined = tf.stop_gradient(a) + b
grads3 = tape3.gradient(combined, [x3, w])
print("stop_gradient只影响x3路径, w路径正常:", grads3[0], grads3[1].numpy())
assert grads3[0] is None
assert grads3[1].numpy() == 2.0
```

**面试怎么问 + 追问链:**
- **Q:** "`tf.stop_gradient` 是做什么的?和 PyTorch 的什么概念最接近?"—— 期望答出"前向恒等、反向截断,概念上接近 `.detach()`",能指出用法差异(函数式包一层 vs tensor 方法)是加分项。
- **追问 1:** "`stop_gradient` 之后,前向计算出来的数值会变吗?"—— 期望"完全不变,`stop_gradient` 只影响反向传播,不影响任何前向数值,这是它和'把这段计算直接删掉'的本质区别"。
- **深挖追问(区分度很高):** "对一个被 `stop_gradient` 包住的 tensor 求梯度,和对一个从未被 `watch()` 的 tensor 求梯度,两者都返回 `None`,这两种 `None` 是同一回事吗?"—— 期望能区分"前者是 tape 确实记录了这次运算,但这个 op 的反向定义就是截断;后者是 tape 压根没有关于这个 tensor 的任何记录"——这是一个专门用来区分"背结论"和"真理解"的好问题,呼应第 9 节的系统梳理。
- **追问 2:** "Straight-Through Estimator 这个经典写法 `x + tf.stop_gradient(round(x) - x)` 是怎么做到'前向是 round,反向是 identity'的?"—— 期望能拆解:前向数值上 `stop_gradient(round(x)-x)` 就是 `round(x)-x`,加上 `x` 恰好等于 `round(x)`;反向上 `stop_gradient` 部分整体贡献 0 梯度,只剩下 `+x` 这一项对 `x` 的梯度,也就是 1。

**常见坑:** 以为 `stop_gradient` 会改变前向数值或者等价于"删掉这段计算"——两者都不对,它只影响反向;把 `stop_gradient` 造成的 `None` 和"没被 watch"造成的 `None` 混为一谈,排查思路会跑偏(该去检查 watch 状态时却在找一个根本不存在的 `stop_gradient`,反之亦然);忘记 `stop_gradient` 只切断它包住的那一条路径,误以为会连带影响同一个表达式里其他独立的分支。

---

## 6. `@tf.custom_gradient` 装饰器 —— 手写前向与反向逻辑

**是什么:**
```text
@tf.custom_gradient
def my_op(x):
    y = ...              # 前向计算
    def grad(dy):
        return ...        # 自定义反向逻辑, dy是上游传入的梯度
    return y, grad

y = my_op(x)   # tape记录时把整个my_op当成一个原子op,反向直接调用你写的grad函数
```

**一句话:** `tf.custom_gradient` 让你手写一个函数的前向和反向逻辑,`GradientTape` 记录到这个函数调用时不会展开成内部逐算子的自动微分链条,而是把它当成一个"整体的黑箱 op",反向传播时直接调用你提供的 `grad`——概念上和 PyTorch 的 `torch.autograd.Function` 是同一类需求(呼应 torch-deep-dive/02 第 9 节)。

**底层机制/为什么这样设计:**

`tf.custom_gradient` 装饰的函数被调用、且发生在活跃 tape 的记录范围内时,tape 记录的不是函数体内部一步步展开的普通运算序列,而是把这一整次调用当成一个原子操作,并把你在函数体内 `return` 的那个 `grad` 闭包保存为这个"op"对应的反向计算函数——这本质上是往第 1 节提到的"op 类型 → 反向函数"注册表里,动态注册了一条只在这次调用生效的自定义规则,而不是走默认的逐算子自动微分展开路径。

三类典型用途,分别有对应的实测证据:

1. **数值稳定性** —— 某些函数如果按自动微分逐步展开中间会产生数值问题。经典例子 softplus:`log(1+exp(x))` 在 `x` 很大时 `exp(x)` 会直接溢出成 `inf`,`log(1+inf)` 还是 `inf`,继续对它求梯度会得到 `nan`——手写一个数值稳定的等价公式作为前向,再手写解析形式的反向(`sigmoid(x)`),两头都不会有溢出问题。
2. **自定义/近似反向逻辑** —— 前向是不可微操作(比如取整),想要一个近似梯度(straight-through,和第 5 节 `stop_gradient` 拼出来的效果一样,但 `custom_gradient` 写法更直接、更不容易出错)。
3. **性能/复用前向中间量** —— 手写反向有时比自动微分展开更快,或者能直接复用前向已经算出的某个中间量,不用重复计算。

一个容易被忽略、已经用代码触发验证过的细节:如果 `my_op` 的前向逻辑里直接引用了一个**闭包捕获的 `tf.Variable`**(不是通过参数传进来的),而不只是显式传入的位置参数,那么 `grad` 函数必须多接受一个 `variables` 关键字参数,并且返回值要变成 `(对位置参数的梯度, 对这些variables的梯度列表)` 这样一个二元组——如果漏掉,TF 会现场抛出 `TypeError`,明确指出"函数用到了哪些 variable,但 grad_fn 没有声明 `variables` 参数"。这个规则的存在是因为 `custom_gradient` 的原始设计目标是"对位置参数求梯度",但真实场景里前向逻辑经常会顺手用到模型自己的权重(比如自定义层内部),TF 选择用运行时检测强制你显式处理这种情况,而不是静默漏算某些权重的梯度。

**AI 研究/工程场景:** 自定义 Keras 层内部实现数值不稳定的激活/损失函数(如自定义的 softplus、logsumexp 变体)时,手写反向比依赖自动微分展开更可靠;量化感知训练(QAT)里权重/激活的量化操作前向不可导,`custom_gradient` 是比拼凑 `stop_gradient` 表达式更清晰、更不容易在复杂场景下出错的标准写法;某些自定义层如果前向直接用到层自己的可训练权重(而不是把权重当函数参数传入),就会触发上面讲的 `variables` 关键字参数规则,这是自定义层开发中一个真实会踩到的坑。

**可运行例子:**
```python
import tensorflow as tf

# 数值稳定的softplus: log(1+exp(x))的稳定实现
@tf.custom_gradient
def stable_softplus(x):
    y = tf.math.log(1.0 + tf.exp(-tf.abs(x))) + tf.maximum(x, 0.0)
    def grad(dy):
        return dy * tf.sigmoid(x)
    return y, grad

x = tf.Variable(100.0, name="x")
with tf.GradientTape() as tape:
    y = stable_softplus(x)
g = tape.gradient(y, x)
print("stable_softplus(100):", y.numpy(), " grad:", g.numpy())
assert y.numpy() == 100.0
assert abs(g.numpy() - 1.0) < 1e-6   # sigmoid(100) ≈ 1

# 对比: 朴素实现在x=100直接溢出,梯度是nan(不是不准,是彻底算错)
xn = tf.Variable(100.0, name="xn")
with tf.GradientTape() as tapen:
    yn = tf.math.log(1.0 + tf.exp(xn))
gn = tapen.gradient(yn, xn)
print("朴素实现: 前向值 =", yn.numpy(), " 梯度 =", gn.numpy())
import math
assert math.isinf(yn.numpy())
assert math.isnan(gn.numpy())
```

```python
import tensorflow as tf

# straight-through round, 用custom_gradient写(比stop_gradient拼表达式更直接)
@tf.custom_gradient
def round_ste(x):
    y = tf.round(x)
    def grad(dy):
        return dy   # 反向直接原样透传,假装这个op是identity
    return y, grad

x = tf.Variable(2.7, name="x")
with tf.GradientTape() as tape:
    y = round_ste(x)
g = tape.gradient(y, x)
print("round_ste前向:", y.numpy(), " 反向:", g.numpy())
assert y.numpy() == 3.0
assert g.numpy() == 1.0

# custom_gradient的函数可以正常参与链式法则(不是一个"求梯度到此为止"的黑箱终点)
x2 = tf.Variable(3.0, name="x2")

@tf.custom_gradient
def stable_softplus2(x):
    y = tf.math.log(1.0 + tf.exp(-tf.abs(x))) + tf.maximum(x, 0.0)
    def grad(dy):
        return dy * tf.sigmoid(x)
    return y, grad

with tf.GradientTape() as tape2:
    y2 = stable_softplus2(x2) * 2   # 外面还乘了2, 检验链式法则能不能正确穿过custom_gradient
g2 = tape2.gradient(y2, x2)
import math
expected = 2 / (1 + math.exp(-3.0))
print("链式法则穿过custom_gradient:", g2.numpy(), " 期望:", expected)
assert abs(g2.numpy() - expected) < 1e-5
```

```python
import tensorflow as tf

# 前向闭包捕获了一个trainable Variable, grad函数必须声明variables参数, 否则现场TypeError
w = tf.Variable(2.0, name="w")

@tf.custom_gradient
def scale_by_w_broken(x):
    y = x * w
    def grad(dy):          # 没有声明variables参数
        return dy * w
    return y, grad

x = tf.Variable(3.0, name="x")
try:
    with tf.GradientTape() as tape:
        y = scale_by_w_broken(x)
    _ = tape.gradient(y, [x, w])
    assert False
except TypeError as e:
    print("闭包捕获Variable但grad漏了variables参数, 报错:", str(e))
    assert "must accept keyword argument 'variables'" in str(e)

# 正确写法: grad函数声明variables参数, 返回(对位置参数的梯度, 对variables的梯度列表)
@tf.custom_gradient
def scale_by_w_correct(x):
    y = x * w
    def grad(dy, variables=None):
        grad_x = dy * w
        grad_vars = [dy * x] if variables else []
        return grad_x, grad_vars
    return y, grad

with tf.GradientTape() as tape2:
    y2 = scale_by_w_correct(x)
gx, gw = tape2.gradient(y2, [x, w])
print("正确写法梯度: dx =", gx.numpy(), " dw =", gw.numpy())
assert gx.numpy() == 2.0   # d(x*w)/dx = w = 2
assert gw.numpy() == 3.0   # d(x*w)/dw = x = 3
```

**面试怎么问 + 追问链:**
- **Q:** "什么时候需要用 `tf.custom_gradient`,而不是让自动微分自己展开?"—— 期望举出"内置组合数值不稳定(softplus 类)""前向不可导需要自定义近似反向(STE)""手写反向更高效/能复用前向中间量"这几类场景。
- **追问 1(区分度很高):** "为什么不直接用自动微分展开 `log(1+exp(x))`,而要手写反向?"—— 期望能现场举出"`x` 很大时朴素实现前向就已经溢出成 `inf`,梯度直接是 `nan`,手写的稳定版本前向不溢出、反向是解析形式的 `sigmoid(x)`,数学上等价但数值上天差地别"。
- **深挖追问:** "如果 `custom_gradient` 装饰的函数内部直接用到了一个 `tf.Variable`(不是通过参数传入,而是闭包捕获),会发生什么?"—— 期望知道"`grad` 函数必须多接受一个 `variables` 关键字参数,并且返回值要变成 `(位置参数梯度, variables梯度列表)` 的二元组,否则会在运行时被 TF 检测出来并抛 `TypeError`,而不是静默漏算这部分梯度"——能说出这是运行时检测而非编译期检查,是理解扎实的体现。
- **追问 2:** "`custom_gradient` 包起来的函数,能不能和外面其他正常运算一起参与链式法则?"—— 期望"能,`custom_gradient` 只是替换了'这一个 op'的反向公式,不是切断了整条反向路径的终点,继续往外层复合完全正常"。

**常见坑:** 忘记 `grad` 函数的返回值个数/顺序要和前向的位置参数一一对应(这里只有一个参数还好,多参数场景容易错位);闭包捕获了 Variable 却没有在 `grad` 里声明 `variables` 参数,线上遇到才会现场报错(至少这个是硬报错,不是静默算错,比某些"看似正常实则错误"的坑要安全);把"手写反向"当成默认应该做的事——大多数场景自动微分展开完全够用且更不容易出 bug,`custom_gradient` 应该是"遇到具体数值稳定性或不可导操作问题时"的针对性方案,不是习惯性的做法。

---

## 7. `jacobian` 与 `batch_jacobian` —— 完整雅可比矩阵

**是什么:**
```text
tape.jacobian(target, sources)
# target可以是非标量,返回"每个target元素对每个source元素"的完整雅可比矩阵
# shape = target.shape + source.shape

tape.batch_jacobian(target, source)
# 假设target和source的第0维都是batch维,且batch间彼此独立
# (每个样本的输出只依赖该样本自己的输入),利用这个假设省掉批次间必然为0的交叉项
```

**一句话:** `tape.gradient()` 本质上只能算 vector-Jacobian product(标量 target,或者退化成标量等价的场景);`jacobian()` 才是老老实实把完整的雅可比矩阵——每一个输出分量对每一个输入分量的偏导数——都摆出来,代价是比一次 `gradient()` 贵得多;`batch_jacobian()` 是专门为"逐样本独立"场景做的优化版本,跳过必然是 0 的批次间交叉项。

**底层机制/为什么这样设计:**

`gradient()` 对非标量 target 内部做的事情,是隐式地假设一个全 1 的上游梯度(等价于先对 target 求和再反传),所以它给出的是"每个 target 分量的贡献汇总到每个 source 分量"的**一个向量**,不是矩阵——已验证:对 `y = x**2`(`x` 是长度 3 的向量),`gradient(y, x)` 得到 `[2,4,6]`,恰好等于完整雅可比矩阵(对角阵 `diag([2,4,6])`)按行求和的结果,这不是巧合,而是"隐式全 1 上游向量做 VJP"这套机制的直接体现。

如果你需要的不是"汇总后的向量"而是"每一个输出分量单独对每一个输入分量的偏导数",就必须用 `jacobian()`。它的实现方式概念上是"对 target 的每一个标量分量分别做一次 vector-Jacobian product"(用一个 one-hot 的上游向量,一次探出雅可比矩阵的一行),所以调用成本大致是 `target` 元素个数倍于一次普通 `gradient()` 调用——这是"雅可比"操作明显比"梯度"更贵的根本原因,面试常会追问这个复杂度差异。

`batch_jacobian` 存在的意义:如果你有一个 `(batch, n)` 的输入和 `(batch, m)` 的输出,且能保证"每个样本的输出只依赖它自己那一份输入"(常见于批量前向,比如普通 Dense 层批量处理一批独立样本),那么完整的 `jacobian()` 会算出一个 `(batch, m, batch, n)` 的张量,其中"批次不等于批次"的那部分交叉项在数学上必然全是 0(样本 i 的输出不可能依赖样本 j 的输入)——已验证:`(2,2)` 输入算出的完整 `jacobian` 形状是 `(2,2,2,2)`,而 `batch_jacobian` 直接给出更紧凑的 `(2,2,2)`(`batch, out, in`),跳过了那些必然为零、纯粹浪费算力和显存的交叉项。

**AI 研究/工程场景:** 输入敏感性分析/物理仿真里需要完整的雅可比矩阵(比如控制系统里线性化一个非线性动力学模型,需要状态对每个输入分量的完整偏导矩阵,不能只要一个汇总梯度);二阶优化方法里,`jacobian` 配合嵌套 tape(第 4 节)是计算 Hessian 矩阵的标准手段;`batch_jacobian` 在逐样本独立的批量推理场景(比如批量计算每个样本输出对该样本输入的敏感度,像 batch 版本的 saliency map)是比强行调用完整 `jacobian` 再手动抽取对角块高效得多的正确姿势。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

# jacobian: 完整偏导矩阵。y_i = x_i^2, dy_i/dx_j = 2*x_i (i==j) 否则 0
x = tf.Variable([1.0, 2.0, 3.0], name="x")
with tf.GradientTape() as tape:
    y = x ** 2
jac = tape.jacobian(y, x)
print("jacobian shape:", jac.shape)
print("jacobian:\n", jac.numpy())
assert jac.shape == (3, 3)
assert np.allclose(jac.numpy(), np.diag([2.0, 4.0, 6.0]))

# 对比plain gradient(): 只给"汇总后的向量",恰好等于jacobian按行求和(因为非对角项全是0)
with tf.GradientTape() as tape_g:
    y_g = x ** 2
grad_sum = tape_g.gradient(y_g, x)
print("plain gradient() (相当于全1上游向量做VJP):", grad_sum.numpy())
assert np.allclose(grad_sum.numpy(), jac.numpy().sum(axis=0))
```

```python
import tensorflow as tf
import numpy as np

# batch_jacobian: 逐样本独立假设下的紧凑版本
xb = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name="xb")   # shape (batch=2, n=2)
with tf.GradientTape() as tape:
    yb = xb ** 2   # shape (2,2), 每个样本的输出只依赖自己的输入
bjac = tape.batch_jacobian(yb, xb)
print("batch_jacobian shape:", bjac.shape)
assert bjac.shape == (2, 2, 2)   # (batch, out_dim, in_dim), 不是(batch,out,batch,in)

for b in range(2):
    expected_b = np.diag(2 * xb.numpy()[b])
    assert np.allclose(bjac.numpy()[b], expected_b)
print("每个batch的雅可比都是diag(2x), 验证通过")

# 完整jacobian在同样的批量输入上, 会算出大量必然为0的批次间交叉项, 形状明显更大更浪费
with tf.GradientTape() as tape_f:
    yf = xb ** 2
full_jac = tape_f.jacobian(yf, xb)
print("完整jacobian在批量输入上的shape:", full_jac.shape)
assert full_jac.shape == (2, 2, 2, 2)   # (batch, out, batch, in) —— batch_jacobian跳过的就是这多出来的一维
```

**面试怎么问 + 追问链:**
- **Q:** "`tape.gradient()` 和 `tape.jacobian()` 有什么区别?"—— 期望答出"`gradient()` 对非标量 target 隐式做全 1 上游向量的 VJP,只给一个汇总后的向量;`jacobian()` 给出完整的、每个输出分量对每个输入分量的偏导矩阵"。
- **追问 1:** "`jacobian()` 的计算成本大概是 `gradient()` 的多少倍?为什么?"—— 期望答出"大致是 target 元素个数倍,因为概念上相当于对每一个输出分量分别做一次独立的 VJP(one-hot 上游向量),不是一次调用能顺带算完的"。
- **深挖追问(区分度很高):** "`batch_jacobian` 相比直接调用 `jacobian` 再手动抽取对角块,省的是什么?"—— 期望理解"不只是省了'手动抽取'这一步代码,而是从计算层面根本不去计算那些批次间必然是 0 的交叉项——`jacobian` 会先老老实实把 `(batch,out,batch,in)` 全部算出来再让你抽对角块,`batch_jacobian` 是利用'逐样本独立'这个先验从一开始就不算那部分",能说出这是计算量层面的优化、不只是接口层面的便利,是加分项。
- **追问 2:** "什么时候 `batch_jacobian` 的'逐样本独立'假设会不成立,导致结果错误?"—— 期望能想到"如果网络里有 batch 内样本之间产生交互的操作(比如 BatchNorm 训练模式下用到了整个 batch 的统计量,或者某种样本间的注意力/池化),这个假设就被破坏了,`batch_jacobian` 会给出错误(遗漏交叉项)的结果而不是报错提醒你"。

**常见坑:** 在不需要完整雅可比矩阵的场景滥用 `jacobian()`(比如只是想要模型参数的梯度做更新,这种场景 `gradient()` 完全够用还更快),白白付出成倍的计算代价;对"逐样本独立"假设不成立的模型(比如内部有跨样本交互的层)使用 `batch_jacobian`,得到的是一个看起来形状正确、实际上悄悄丢失了交叉项的错误结果;混淆 `jacobian` 返回的 shape 顺序(`target.shape + source.shape`,不是反过来),对着结果做后续矩阵运算时排错维度。

---

## 8. `tape.gradient()` vs `tf.gradients()` —— TF2 eager API 与 TF1 遗留静态图 API

**是什么:**
```text
tf.gradients(ys, xs)   # TF1时代的图级别API: 在一个由session.run驱动的静态图上,
                        # 构建"计算ys对xs梯度"的反向子图,依赖隐式的全局默认图,
                        # 不是给eager tensor用的
```

**一句话:** `tf.gradients()` 是 TF1 静态图时代的遗留 API,依赖"图已经完整定义好、还没有真实数值"这个前提;TF2 默认 eager 执行下没有这样一张预先存在的图可供它去构建反向子图,官方明确要求改用 `GradientTape`——这两个 API 不是"新旧两种写法随便选",在 eager 模式下 `tf.gradients()` 直接报错,不能用。

**底层机制/历史背景(简要,不展开):** TF1 的自动微分是"图级别"的操作——`tf.gradients` 在图构建阶段,遍历已经定义好的静态图(此时还没有任何真实数值在流动),依据每个 op 类型查同一张"op 名 → 反向函数"注册表(第 1 节提到过,这张表 TF1 图模式和 TF2 eager+tape 模式是共用的),生成一整套新的"梯度子图" op 节点,插回原来的图里;等 `session.run` 的时候,这套梯度子图和前向图一起被执行,拿到真实的梯度数值。这套机制的前提是"图是预先完整定义好、可以被遍历的静态结构"——GradientTape 恰恰是为 eager 模式设计的替代方案,因为 eager 下没有这样一个可供预先遍历的图对象,只能靠运行时主动录制。

**AI 研究/工程场景:** 这个对比本身更多是"排查历史遗留代码报错"的场景——接手一份老旧的、混杂 TF1 风格代码的仓库(尤其是用了 `tf.compat.v1` 兼容层却又跑在默认 eager 模式下),看到 `tf.gradients` 相关的报错,需要能立刻反应过来"这是图模式 API 混进了 eager 环境",而不是去纠结这个函数本身的用法。

**可运行例子:**
```python
import tensorflow as tf

x = tf.Variable(2.0, name="x")
y = x ** 2

print("当前是否eager执行:", tf.executing_eagerly())
assert tf.executing_eagerly() is True

# eager模式下调用tf.gradients(), 现场触发报错(原样抄录, 不是转述文档)
try:
    g = tf.gradients(y, x)
    assert False
except RuntimeError as e:
    print("tf.gradients()在eager模式下报错:", str(e))
    assert "tf.gradients is not supported when eager execution is enabled" in str(e)
    assert "Use tf.GradientTape instead" in str(e)

# 正确写法: GradientTape
with tf.GradientTape() as tape:
    y2 = x ** 2
g2 = tape.gradient(y2, x)
print("GradientTape正常工作:", g2.numpy())
assert g2.numpy() == 4.0
```

**面试怎么问 + 追问链:**
- **Q:** "`tf.gradients()` 和 `tape.gradient()` 是什么关系,能不能混用?"—— 期望答出"`tf.gradients()` 是 TF1 静态图时代的 API,依赖预先构建好的图,eager 模式下直接不可用,会报错;TF2 一律用 `GradientTape`"。
- **追问 1:** "如果在自己维护的老代码里看到 `tf.gradients` 报错,应该怎么理解这个报错、怎么改?"—— 期望"识别出这是图模式遗留 API 混进了 eager 环境,按官方报错提示直接换成 `GradientTape` 改写这部分逻辑,而不是尝试去'修好' `tf.gradients` 本身的调用方式"。
- **深挖追问:** "`tf.gradients()` 在 TF1 图模式下和 `GradientTape` 在 TF2 下,反向传播用的是不是同一套'每个 op 怎么求导'的底层规则?"—— 期望理解"是同一张 op 级别的梯度注册表,两边共用,区别只在于'怎么触发、什么时候构建反向路径'这个外层机制(图构建期一次性生成 vs 运行时按需录制),不是两套互相独立、可能给出不同结果的求导逻辑"。

**常见坑:** 在 TF2 eager 环境下沿用 TF1 教程/老代码里的 `tf.gradients()` 写法,得到一个看似陌生的 `RuntimeError` 就以为是自己的计算图写错了,实际上换成 `GradientTape` 才是唯一正确的方向;误以为两个 API 是"性能优化版 vs 简单版"这种可以按场景挑选的关系——它们服务于两种完全不同的执行模型,eager 模式下压根没有第二个选项。

---

## 9. 多输出多输入梯度计算与 `unconnected_gradients` —— 断开连接时的行为

**是什么:**
```text
tape.gradient(target, sources)
# target和sources都可以是单个tensor,也可以是list/tuple/dict这类嵌套结构
# target是list时: 实际计算的是"对所有target元素求和"之后再对每个source求梯度
#                 (VJP视角: 每个target各自的隐式上游梯度都是1, 走同一次反向遍历汇总)

unconnected_gradients=tf.UnconnectedGradients.NONE   # 默认: source如果完全没参与target的计算, 梯度是None
unconnected_gradients=tf.UnconnectedGradients.ZERO    # source未连接时, 改成返回同形状的全0 tensor
```

**一句话:** `target` 是多个 tensor 组成的列表时,`tape.gradient()` 算的是"这些 target 各自梯度贡献的加总",不是分别返回每个 target 单独的梯度;`unconnected_gradients` 决定的是"source 如果压根没参与这次计算,该返回 `None` 还是 `0`"这一个具体行为的开关。

**底层机制/为什么这样设计:**

`target` 为 list 时,`tape.gradient()` 内部的反向传播,概念上等价于先把每个 target 各自的隐式上游梯度设成全 1(标量就是 1,张量就是全 1 同形张量),这些贡献在同一次反向拓扑遍历里,汇入同一个 source 的路径是**相加**的——这和 torch-deep-dive/02 第 4 节讲的"同一个 leaf 被多路径依赖时梯度是累加不是覆盖"是同一套底层道理,只是 TF 把"多个 target"和"同一条路径内部的多次汇入"统一用同一套加总逻辑处理,不做区分。已验证:`x` 同时参与 `a=x**2`(梯度贡献 6)和 `b=x**3`(梯度贡献 27)两个 target,`gradient([a,b], x)` 得到 `33.0`,和用 persistent tape 分别求出 `6.0`、`27.0` 再手动相加完全一致。

`unconnected_gradients` 这个参数存在的原因:某些场景(尤其是批量、自动化地对一组 sources 统一处理梯度,比如写一个通用的梯度诊断工具循环遍历所有参数)不想因为"某个 source 这次压根没参与计算"就得到一个 `None`,然后代码里到处要判断 `is not None` 才能继续往下算(比如想直接把梯度累加进一个统计量,`None` 没法参与数值运算,`0` 可以)。TF 选择让默认值是 `NONE` 而不是直接默认 `0`,是刻意保留"未定义/不可导"和"确实是 0"这两种语义上不完全等价的信息——`None` 代表"这条路径根本不存在,没法谈梯度是多少";`0` 代表"这条路径存在,只是这次算出来的值恰好是 0"。默认选择"精确但麻烦"的 `None`,把"图省事但抹掉这层语义区分"的 `0` 作为一个需要显式打开的选项,这是 TF 在这个 API 上的设计取舍。

**串联三种 `None` 的来源(呼应第 2、5 节):** 走到这里,同样表现为 `tape.gradient()` 返回 `None`,已经现场验证过三种截然不同的成因,值得放在一起区分清楚,这也是面试里区分度很高的问题:
1. **source 从未被 tape 记录/watch**(第 2 节)—— tape 对这个 tensor 压根没有任何记录,无路可走;
2. **source 和 target 之间确实没有计算路径**(本节的"unconnected")—— tape 有完整记录,但从 target 出发反向遍历,走不到这个 source;
3. **路径上经过了一个前向有定义、反向被显式注册为不可导的 op**(第 5 节的 `tf.round` 就是例子)—— tape 记录了这次运算,但这个 op 自己的反向公式就是"不产生梯度"。

从用户角度看,这三者都表现为一个安静的 `None`,没有任何报错或警告替你区分是哪一种——能不能讲清楚这个区别,以及分别对应什么排查手段,是判断"真的懂 GradientTape"还是"记住了几条使用规则"的一个好指标。

**AI 研究/工程场景:** 写通用的训练诊断工具(比如遍历 `model.trainable_variables` 统一打印每层的梯度范数)时,如果某些变量这一个 batch 恰好没被用到(比如某个条件分支这次没走到、或者是一个尚未接入计算图的新增层),用 `unconnected_gradients=ZERO` 能让统计代码不用到处加 `None` 判断;多任务学习里想知道"总 loss"对共享参数的梯度,直接把多个子 loss 放进一个 list 传给 `gradient()`,比手动加权求和成一个标量再求梯度更直接,还能少一步"选权重"的决定(虽然本质上默认权重就是全 1)。

**可运行例子:**
```python
import tensorflow as tf

x = tf.Variable(2.0, name="x")
y = tf.Variable(3.0, name="y")
z_unused = tf.Variable(5.0, name="z_unused")

# 多target多source, 默认unconnected_gradients=NONE
with tf.GradientTape() as tape:
    out1 = x ** 2
    out2 = y ** 2
grads = tape.gradient([out1, out2], [x, y, z_unused])
print("grads:", [None if g is None else g.numpy() for g in grads])
assert grads[0].numpy() == 4.0    # d(out1)/dx = 2x = 4
assert grads[1].numpy() == 6.0    # d(out2)/dy = 2y = 6
assert grads[2] is None            # z_unused完全没参与计算

# unconnected_gradients=ZERO: 把上面的None换成0
with tf.GradientTape() as tape2:
    out1b = x ** 2
    out2b = y ** 2
grads2 = tape2.gradient([out1b, out2b], [x, y, z_unused],
                          unconnected_gradients=tf.UnconnectedGradients.ZERO)
print("grads2 (ZERO):", [g.numpy() for g in grads2])
assert grads2[2].numpy() == 0.0
```

```python
import tensorflow as tf

# 多target语义验证: gradient([a,b], x) 等于分别求梯度再相加, 不是"分别返回"
x = tf.Variable(3.0, name="x")
with tf.GradientTape() as tape:
    a = x ** 2   # da/dx = 2x = 6
    b = x ** 3   # db/dx = 3x^2 = 27
combined_grad = tape.gradient([a, b], x)
print("gradient([a,b], x):", combined_grad.numpy())
assert combined_grad.numpy() == 33.0   # 6+27, 是加总不是分别返回

with tf.GradientTape(persistent=True) as tape2:
    a2 = x ** 2
    b2 = x ** 3
ga = tape2.gradient(a2, x)
gb = tape2.gradient(b2, x)
print("分别求梯度再手动相加:", (ga + gb).numpy())
assert (ga + gb).numpy() == combined_grad.numpy()
del tape2

# sources支持嵌套结构(dict), 返回值保持相同结构
xd = tf.Variable(2.0, name="xd")
yd = tf.Variable(3.0, name="yd")
with tf.GradientTape() as tape3:
    out = xd * yd
grad_dict = tape3.gradient(out, {"a": xd, "b": yd})
print("dict结构的sources, 返回也是dict:", {k: v.numpy() for k, v in grad_dict.items()})
assert grad_dict["a"].numpy() == 3.0   # d(xd*yd)/dxd = yd = 3
assert grad_dict["b"].numpy() == 2.0   # d(xd*yd)/dyd = xd = 2

# target本身完全不依赖任何watched source: 同样是unconnected, 默认None, ZERO变成0
with tf.GradientTape(persistent=True) as tape4:
    _ = xd * 2
    const_target = tf.constant(42.0)   # 和xd毫无关系的独立常量
g_default = tape4.gradient(const_target, xd)
g_zero = tape4.gradient(const_target, xd, unconnected_gradients=tf.UnconnectedGradients.ZERO)
print("target完全不依赖source: 默认 =", g_default, " ZERO =", g_zero.numpy())
assert g_default is None
assert g_zero.numpy() == 0.0
del tape4
```

**面试怎么问 + 追问链:**
- **Q:** "`tape.gradient()` 的 `target` 传一个 list 进去,算出来的是什么?"—— 期望答出"是所有 target 各自梯度贡献的加总,不是分别返回每个 target 单独的梯度",最好能举出和"手动分别求梯度再相加"结果一致这个验证方式。
- **追问 1:** "`unconnected_gradients` 参数解决的是什么问题?为什么默认不是直接返回 0?"—— 期望答出"某个 source 完全没参与这次计算时,默认返回 `None` 而不是自动判断成 `0`,是因为'没有梯度'和'梯度恰好是0'在语义上不是一回事,`None` 更精确但用起来麻烦,`ZERO` 是为了图方便可以显式打开的选项"。
- **深挖追问(区分度很高):** "`tape.gradient()` 返回 `None`,可能是哪几种原因造成的?能不能分别举例?"—— 期望能完整说出"没被 watch / 路径确实不连通(unconnected) / 经过了一个反向被注册为不可导的 op"这三种,并能各自举出具体例子(而不是笼统地说"因为没有梯度")——这是本节最核心的区分度问题,直接决定候选人是不是真的理解 GradientTape 的运作机制,还是只记住了几条零散的使用规则。
- **追问 2:** "`gradient()` 的 `sources` 参数能不能传一个字典或者嵌套结构进去?"—— 期望"可以,TF 会保持返回值和输入结构一致(dict 进 dict 出),这在批量处理多个命名变量时比自己手动拼 list 更方便"。

**常见坑:** 把 `gradient([a,b], x)` 误解成"能同时拿到 a 和 b 各自独立的梯度"——实际上算出来的是加总后的**一个**结果,如果确实需要分开,要么用 `persistent=True` 分别调用,要么自己在数学上确认加总就是想要的语义;不清楚 `unconnected_gradients=ZERO` 改变的只是"表现形式"(`None` 变 `0`),这条路径"没有梯度贡献"这个事实本身没有变,不代表用了 `ZERO` 之后这个 source 就突然开始被优化;混淆本节"unconnected"造成的 `None` 和第 2 节"没被 watch"、第 5 节"op 本身不可导"造成的 `None`,排查方向南辕北辙。

---

## 10. 与 PyTorch autograd 设计差异对比 —— tape-based 显式记录 vs 全自动追踪

**是什么:** 对比 TF2 `GradientTape` 和 PyTorch autograd 在"什么时候开始记录一次运算的反向路径"这件事上的设计差异——注意这**不是**"TF2 eager vs TF1 静态图"那个话题(torch-deep-dive/02 第 1 节已经讲透 TF1 静态图 `tf.placeholder`+`session.run` 和 PyTorch define-by-run 的本质区别,那是完全不同的对比维度)。TF2 默认执行模式本身就是 eager 的,和 PyTorch 一样,每一行代码立即算出真实数值、可以随时 `print` 中间结果、可以用原生 Python `if`/`for` 写模型——这些"define-by-run 家族"的共同特征,TF2 和 PyTorch 都具备,第 1 节和第 4 节的例子已经反复印证过这一点(分支/循环次数在运行时决定,记录的运算路径跟着实际走过的代码走)。本节要讲清楚的是另一个维度的差异:**同样是 eager,"要不要把这次运算记下来留给反向传播用"这件事,两边由什么来决定。**

**一句话:** PyTorch 的 autograd 记录是**tensor 自带的持久属性**在驱动(`requires_grad=True` 的 tensor,只要参与运算就自动挂 `grad_fn`,不需要一个外部的"录制中"上下文,直到你主动用 `no_grad()`/`detach()` 关掉);TF2 GradientTape 的记录是**一个有生命周期的外部作用域**在驱动(只有 op 执行时恰好处于某个活跃 tape 的监听范围内,才会被记录,tape 退出、gradient 用完,记录就没了)——同一段计算逻辑,是否产生反向记录,在 PyTorch 里取决于"这个 tensor 自己的属性",在 TF2 里取决于"这次调用发生时,外部有没有一双眼睛在看"。

**底层机制/为什么这样设计:**

先说两者相通的部分,避免和 TF1 的对比混淆:PyTorch 和 TF2+GradientTape 本质上都是在做同一件事——reverse-mode automatic differentiation(反向模式自动微分),都需要在前向执行的同时或之后,构建出一个能支撑链式法则反向遍历的 DAG(有向无环图)结构。这个 DAG 的表示形式确实不同:PyTorch 是一张由 `Node` 对象和 `next_functions` 指针组成的、公开暴露给用户的图结构(torch-deep-dive/02 第 2、3 节反复强调"这是一个真实的、可以用代码遍历的数据结构");TF 用的是"tape"(磁带)这个说法,这个术语来自自动微分文献里更早的传统——经典做法是把前向执行过程中依次发生的每一步基本运算,按时间顺序记录成一份线性的记录表(文献里常称为 Wengert list,得名于 R. E. Wengert 1964 年那篇提出这个想法的论文),反向阶段把这份记录表倒着"播放"一遍,对每一步应用链式法则。这只是**记录用的数据结构不同**(节点+指针的图,还是按执行顺序排列、彼此可以互相引用的记录表),两者能表达的信息本质上是同一张 DAG,不是两种不同的算法。

真正的差异在于**触发记录的机制**,这一点已经用代码反复验证过:

- **PyTorch:** `requires_grad` 是 tensor 对象上一个持久的、跟着这个 tensor 走的字段。只要参与运算的输入里有一个 `requires_grad=True`(torch-deep-dive/02 第 12 节讲的传播规则),这次运算的输出就自动 `requires_grad=True` 并挂上 `grad_fn`——不需要任何额外的"现在开始录制"的声明,默认行为就是"追踪",想要关掉需要主动 `no_grad()`/`.detach()`/`requires_grad_(False)`。
- **TF2 + GradientTape:** 记录与否,首要看"这次 op 执行时,有没有一个活跃的 tape 正在监听它的输入"——已验证 `tf.Tensor` 上根本没有 `requires_grad` 这样的属性(`hasattr` 直接是 `False`),`tf.Variable` 有的是 `trainable`,但 `trainable=True` 只影响"在 tape 内被读取时是否自动 watch"这一件事,不影响"tape 本身存不存在"。完全相同的一行代码 `xc * 3`,在没有任何 tape 活跃时执行,是一次彻头彻尾的普通数值运算,不产生任何 autograd 相关的额外信息,永远不可能事后补出一个梯度;把同一行代码搬到一个活跃 tape 内部执行,才会被记录、才能求出梯度——默认行为是"不追踪",想要则需要主动用 `with tf.GradientTape()` 圈出范围,再加上 watch 规则(第 2 节)决定这个范围内具体追踪谁。

这个差异带来一个实际后果:PyTorch 写推理代码必须记得套一层 `torch.no_grad()`(否则默认会白白构建一张永远用不上的反向图,浪费显存和调度开销);TF2 eager 写推理代码什么都不用做——只要不主动开一个 tape,任何计算天然零 autograd 开销,这是"默认关闭、按需开启"设计带来的直接好处。反过来,PyTorch 的图结构因为持久挂在每个 tensor 上,才能有 `y.grad_fn`、`grad_fn.next_functions` 这样一套面向用户公开、可以拿着一个 tensor 现场往回遍历的自省 API(torch-deep-dive/02 第 2、3 节整整两节都在利用这套 API 做验证);已经用代码验证过,`tf.GradientTape` 的公开方法只有 `watch`、`gradient`、`jacobian`、`batch_jacobian`、`reset`、`stop_recording`、`watched_variables` 这几个,产生的 `EagerTensor` 本身也没有类似 `grad_fn`/`next_function` 的属性——TF 没有对外暴露一个"这个 tensor 是怎么算出来的,能不能让我顺着走回它的输入"这样的图遍历接口,想知道"某个变量有没有被追踪",能拿到的只有 `watched_variables()` 这一份监听名单,拿不到完整的、可以手动遍历节点和边的图对象。这不是 TF 技术上做不到,而是 tape 从设计上就是一个"用完即焚"的临时记录本,不是一个准备长期暴露给用户内省的公开数据结构。

**AI 研究/工程场景:** 从 PyTorch 转 TF2(或者反过来)最容易水土不服的地方正是这里——PyTorch 习惯了"只要 `requires_grad=True` 就一直追踪,想关掉才需要主动声明",带着这个心智模型写 TF2 代码,会不理解为什么一个训练好的 `trainable=True` 权重在某段代码里死活求不出梯度(因为压根没被任何 tape 圈住);反过来 TF2 老手写 PyTorch,可能会习惯性地想找"tape 在哪"或者"这个变量有没有被 watch",而不去检查 `requires_grad`/`no_grad()` 这条完全不同的判断路径。团队里同时维护两个框架代码库时,这种"记录触发条件"的差异比"两边梯度计算数学上是不是一回事"更容易在代码审查里被忽略,却是实际踩坑频率最高的地方之一。

**可运行例子:**
```python
import tensorflow as tf

# TF2 eager+tape 同样是define-by-run: 分支/循环走哪条路径, 记录就跟着走哪条(不是TF1那一套)
def f(x, flag):
    if flag:
        y = x * 2
    else:
        y = x ** 2
    return y

x1 = tf.Variable(3.0, name="x1")
with tf.GradientTape() as tape1:
    y1 = f(x1, True)
g1 = tape1.gradient(y1, x1)
print("分支True: grad =", g1.numpy())
assert g1.numpy() == 2.0    # d(2x)/dx = 2

x2 = tf.Variable(3.0, name="x2")
with tf.GradientTape() as tape2:
    y2 = f(x2, False)
g2 = tape2.gradient(y2, x2)
print("分支False: grad =", g2.numpy())
assert g2.numpy() == 6.0    # d(x^2)/dx = 2x = 6
```

```python
import tensorflow as tf

# 核心差异验证: 完全相同的一行代码, 有没有活跃tape决定了它能不能求出梯度
# (PyTorch是tensor自带requires_grad持久属性驱动, TF2是外部tape作用域驱动)
xc = tf.Variable(5.0, name="xc")

op_outside = xc * 3   # 没有任何tape活跃 —— 永远不可能补出梯度, 不是报错, 是压根没记录
print("tape外执行的op, 类型:", type(op_outside).__name__)

with tf.GradientTape() as tape_in:
    op_inside = xc * 3   # 同一行代码, 这次tape活跃且xc被自动watch —— 正常记录
g_in = tape_in.gradient(op_inside, xc)
print("同一行代码, tape内执行, grad =", g_in.numpy())
assert g_in.numpy() == 3.0

# tf.Tensor没有requires_grad这种持久属性, tf.Variable也没有grad_fn这种图遍历指针
print("EagerTensor有requires_grad属性?", hasattr(op_outside, "requires_grad"))
assert hasattr(op_outside, "requires_grad") is False
print("EagerTensor有grad_fn属性?", hasattr(op_outside, "grad_fn"))
assert hasattr(op_outside, "grad_fn") is False

# GradientTape公开方法里没有任何图遍历API(对比torch grad_fn/next_functions那一整套自省能力)
with tf.GradientTape() as tape_pub:
    yy = xc ** 2
public_methods = sorted(a for a in dir(tape_pub) if not a.startswith("_"))
print("GradientTape公开方法:", public_methods)
assert public_methods == ["batch_jacobian", "gradient", "jacobian", "reset", "stop_recording", "watch", "watched_variables"]
graph_walk_attrs = [a for a in dir(yy) if "grad_fn" in a or "next_function" in a]
print("产生的tensor本身有没有图遍历相关属性:", graph_walk_attrs)
assert graph_walk_attrs == []
```

**面试怎么问 + 追问链:**
- **Q:** "TensorFlow 2 的 GradientTape 和 PyTorch 的 autograd,都是 eager 模式,那到底有什么本质区别?"—— 期望候选人第一步先澄清"这不是在问 TF1 静态图 vs PyTorch 那个经典对比,TF2 本身就是 eager 的",然后准确说出"PyTorch 靠 tensor 自带的 `requires_grad` 持久属性自动追踪,TF2 靠外部 tape 作用域 + watch 规则决定要不要记录"——很多候选人会不小心把这题答成 TF1 vs PyTorch 那道经典题,这本身就是一个筛选点。
- **追问 1:** "这个设计差异,会给写推理代码带来什么不同的注意事项?"—— 期望答出"PyTorch 推理代码要主动套 `torch.no_grad()`,否则默认会白白构建反向图;TF2 eager 推理代码什么都不用做,只要不主动开 tape,天然零 autograd 开销——一个是'默认追踪、按需关闭',一个是'默认不追踪、按需开启'"。
- **深挖追问(区分度很高):** "PyTorch 可以拿一个 tensor 的 `grad_fn` 现场遍历 `next_functions` 走完整张反向图(呼应 torch-deep-dive/02 第 2、3 节),TF2 的 `GradientTape` 能不能提供类似的、面向用户公开的图遍历能力?"—— 期望候选人能想到直接查 `dir(tape)` 或者检查产生的 tensor 有没有类似属性去验证,而不是凭印象回答;最好能说出"`GradientTape` 只暴露 `watched_variables()` 这样的监听名单,没有对外公开节点+边级别的可遍历图对象,这和 tape 本身'用完即焚、不打算长期暴露给用户'的设计定位是一致的",这是本节乃至全篇最能体现"真的动手对比过"而不是"背了两句概念"的问题。
- **追问 2:** "如果要写一个'不管框架是 TF2 还是 PyTorch,写起来心智模型尽量一致'的工具函数库,这个记录触发机制的差异会带来什么麻烦?"—— 期望能联系到"AI 研究/工程场景"里讲的团队协作场景,说出"'为什么这个变量的梯度是 None'背后的排查路径在两个框架里完全不同(一个查 `requires_grad`/`no_grad` 上下文,一个查 tape 作用域和 watch 状态),没法直接套用同一套排查直觉"。

**常见坑:** 把本节的对比和"TF1 静态图 vs PyTorch 动态图"那道经典面试题搞混,答着答着就滑到 `tf.placeholder`/`session.run` 上去了——TF2 默认执行模式本身就是 eager,这里比的是"eager 之下,两个框架各自怎么决定要不要记录"这个更细的维度;误以为 TF2 因为是"tape 显式记录"就不支持原生 Python 控制流写模型——这是错的,第 1、4 节的例子已经证明 TF2 eager+tape 一样是 define-by-run,分支和循环怎么走完全由运行时决定,TF2 和 TF1 静态图在这一点上是天壤之别;以为"两边都是 eager"就意味着两边的自动微分能力/API 设计没有值得深究的差异——记录触发机制的不同,直接决定了两边关于"这个梯度为什么是 None/为什么图不见了"的排查思路完全不能照搬。

---

## 小结:这一批 10 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|------|
| 1 | tape 工作原理 | 记录不是 tensor 的属性,是"op 执行时是否处于活跃 tape 监听范围"决定的;记录检查的是执行瞬间,不能事后追溯 |
| 2 | `watch()` 与自动 watch 规则 | trainable Variable 被读取时自动 watch;普通 tensor/`trainable=False` 变量必须显式 `watch()`,否则安静返回 `None` |
| 3 | `persistent=True` 与生命周期 | 默认 tape 只能 `gradient()` 一次,报错文案现场验证;`persistent=True` 可复用但要记得手动 `del`;`reset()`/`stop_recording()` 提供更细粒度的生命周期控制 |
| 4 | 高阶导数(nested tape) | 内层求梯度这个过程本身若被外层 tape 记录,即可对梯度再求梯度;两层都要各自 watch 同一个非 Variable 目标,少一层就断链 |
| 5 | `tf.stop_gradient` | 前向恒等、反向截断,等价于 PyTorch `.detach()` 的语义但函数式调用;是 straight-through estimator 的经典构件 |
| 6 | `@tf.custom_gradient` | 手写前向+反向,数值稳定性/自定义近似梯度的标准工具;闭包捕获 Variable 时 `grad` 函数必须声明 `variables` 参数,否则现场 `TypeError` |
| 7 | `jacobian`/`batch_jacobian` | `gradient()` 只给隐式全 1 上游向量的汇总结果;`jacobian()` 给完整矩阵但代价是 target 元素个数倍;`batch_jacobian` 利用逐样本独立假设跳过必然为零的批次交叉项 |
| 8 | `tape.gradient()` vs `tf.gradients()` | 后者是 TF1 图模式遗留 API,eager 下直接报错;两者共用同一张 op 级别梯度注册表,只是触发机制不同 |
| 9 | 多输出多输入梯度 与 `unconnected_gradients` | `target` 为 list 时梯度是加总不是分别返回;`unconnected_gradients` 决定路径不连通时 `None` 还是 `0`;三种 `None` 来源(未 watch / 未连接 / op 不可导)在此系统区分 |
| 10 | 与 PyTorch autograd 差异 | 两边都是 define-by-run 的 eager 执行,差异在"记录触发机制":PyTorch 靠 tensor 持久属性 `requires_grad`,TF2 靠外部 tape 作用域 + watch 规则;TF 没有对外暴露 PyTorch 那种 `grad_fn`/`next_functions` 级别的可遍历图 API |

下一批:[03-tf-function-and-autograph.md](03-tf-function-and-autograph.md) —— `tf.function` 与 AutoGraph 计算图机制。

---

*更新:2026-07-09*
