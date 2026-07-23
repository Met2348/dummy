# 14 · 进阶深度追加:5 个真实二面级别的多级追问链案例

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是知识点,不计入统计。

## 为什么需要这篇追加内容

`01-13` 全部完成并自查通过之后,用户转达了一位有经验从业者的反馈:现有材料没有达到 **2026 年大厂技术二面** 的深度。这篇追加内容基于一次真实的调研(WebSearch 检索中国大厂面经、西方大厂面经、面试官视角的元讨论,而不是凭训练数据里的印象去猜),调研结论完整存档在项目 memory 里,而且已经在 [dsa-deep-dive/20-advanced-interview-depth.md](../dsa-deep-dive/20-advanced-interview-depth.md) 和 [torch-deep-dive/12-advanced-interview-depth.md](../torch-deep-dive/12-advanced-interview-depth.md) 里验证过一遍呈现格式——本篇是这套格式在 tensorflow-deep-dive 系列的落地,和 torch-deep-dive 12 号文件是"同一套心智模型,两个框架"的姊妹篇,结构上刻意保持一致,方便对照读。核心发现是:真实的追问不是"正确性 → 复杂度 → 能不能优化"这一条线性链,而是至少沿着 **5 条独立轴线** 展开,并且经常在同一道题里综合出现:

1. **规模递增轴**——数据/负载规模一级一级往上跳,原方案在更大规模下会失效,需要换思路。
2. **工程约束递增轴(并发/分布式)**——单机正确 → 并发/架构约束 → 分布式扩展。
3. **方案批判迭代轴**——面试官不深挖同一个方案的复杂度,而是连续指出具体的工程缺陷,逼你换方案(不是"不够好"这种空话,是可验证的具体缺陷)。
4. **决策依据追问轴**——不纠错,只逼问"你是怎么考虑选这个不选那个的"。
5. **真实性验证轴**——把"做了优化""自动选择了最优配置""环境配对了"这类抽象表述,追问压向具体数字、具体验证过程,而且要能验证"权威表述本身"是不是全称成立,不是"某一个例子恰好成立"。

下面 5 个案例,每个都明确标注建立在哪个已有知识点之上,包含完整还原的多级追问链(带参考答案)和真实验证过的可运行例子,并且标注各自主要挂在哪条轴线上。**这是方法论范例,不是把 100 个知识点全部重写**——读者应该能把同样的思路自己套用到任何一个已有知识点上练习追问。

**关键环境差异,必须先说清楚:** 本系列(tensorflow-deep-dive)和仓库其它系列不同,不在 Windows 原生 `.venv` 里验证,而是在 **WSL2(Ubuntu 24.04 LTS)的独立 venv `~/tf-venv`** 里验证,详见 [00-roadmap.md](00-roadmap.md) 开头"0. 环境声明"一节。本文所有代码已在这个环境(TensorFlow **2.21.0**,GPU:NVIDIA GeForce RTX 3080 Ti Laptop GPU,WSL2 GPU 直通,总容量 16384 MiB,已用 `nvidia-smi` 现场核实;`TF_USE_LEGACY_KERAS=1` 已生效)下实际跑通验证,给出的每一个数字都是本机实测,不是转述或凭经验编造。案例 3 涉及的分布式部分,如实沿用 [11 类](11-distributed-training-basics.md)已经声明过的验证边界:本机只有一张物理 GPU,虚拟设备/多进程模拟能验证机制正确性,不能验证真实多卡/多机吞吐——凡是触及这条边界的地方,本文都会像 11 类一样明确标注,不含糊带过。

---

## 案例 1:retracing 性能陷阱的诊断链——"加一个参数应该就够了"经不起现场验证(全新题型,兼方案批判迭代轴)

建立在 [03 类](03-tf-function-and-autograph.md) 第 2 节(retracing 触发条件:Tensor 的 shape/dtype 变化,和非 Tensor 的 Python 原生值变化,两者被对待的方式完全不同)和 [13 类](13-debugging-and-common-errors.md) 第 6 节(retracing 性能陷阱的识别方法——TF 自动打的 WARNING,以及在函数体里埋计数器对比调用次数和 trace 次数)之上。这是这条系列相比 torch-deep-dive 12 号文件最有差异化价值的一个案例方向——PyTorch 没有"tracing"这个概念,这整条追问链是 TF 独有的心智负担。13 类第 6 节已经列出三条修复手段(把 Python 数字包成 `tf.constant`/`tf.Variable`、`input_signature`、`reduce_retracing=True`),但没有细究这三条手段各自真正针对哪一种根因——这个案例把这个"看起来是平级的三个选项,其实各管一段"的边界现场挖出来,而且顺带发现了一个 13 类没有覆盖的独立现象。

**追问链条完整还原:**

- **Q(基础):** "这个用 `@tf.function` 包装的训练 step 函数,你发现它跑起来几乎和不加 `@tf.function` 一样慢,你怎么判断问题出在哪?"—— 期望答出 03/13 类的方法论:看 TF 自动打的 retracing WARNING,以及自己在函数体里埋一个只在 trace 时才会自增的计数器,对比"调用次数"和"trace 次数"。
- **追问 1(锁定根因):** "假设你已经确认是 retrace 频繁触发,而且打印了每次触发 retrace 时的参数,发现每次都在变的是一个 Python `float`(比如一个逐 step 变化的 loss 权重系数),你的第一个修复方案是什么?"—— 这里故意等一个条件反射式的回答:"WARNING 文本里不是给了 `reduce_retracing=True` 这个选项吗,加上就行。"
- **追问 2(方案批判,不是深挖同一方案,要求现场验证而不是相信文本):** "你怎么证明加了 `reduce_retracing=True` 之后,retrace 真的减少了?"—— 期望候选人现场验证,而不是看到 WARNING 文本里提过这个参数就认为一定管用。现场验证的结果是:**`trace_count` 依然是 10/10,和完全不加这个参数没有任何区别**——这个"方案"压根没有解决问题。
- **深挖追问(反直觉核心,不能把 `reduce_retracing` 一棍子打死):** "既然 `reduce_retracing=True` 对这种情况没用,是不是说明 WARNING 文本给的这条建议本身是错的、过时的?"—— 期望候选人不要因为一次失败就否定整个机制,而是设计对照实验:把"每次变化的东西"换成 **Tensor 的 shape**(而不是 Python `float`),这次 `reduce_retracing=True` 让 `trace_count` 从 10 降到 2,证明它确实是真实生效的——只是它的生效对象是"多个 Tensor 输入的 shape 能不能被泛化合并成一张更宽的图",和"要不要让一个 Python 原生值参与决定图结构"是完全不同的机制,根本不是同一件事的两种说法。
- **深挖追问·续(换正确方案):** "那真正对 Python 标量这种 retrace 生效的手段是什么?"—— 期望答出:把这个标量包成 `tf.constant`(或 `tf.Variable`),让它在图里变成一个只看 shape/dtype 的 Placeholder,不再被当成编译期常量烧进图里。
- **深挖追问(真实性验证,考验候选人会不会自己吓自己):** "验证完之后你发现,修好的版本 `trace_count` 依然不是理论最小值 1,而是 2,是不是修复不彻底?"—— 期望候选人不要慌,先隔离变量:现场发现这是**另一个完全独立的现象**——`tf.function` 内如果在第一次调用时创建了新的 `tf.Variable`(比如 optimizer 的 slot 变量在第一次 `apply_gradients` 时才被创建),这一次调用本身就会触发两次 trace,和传入的参数值是否变化毫无关系。能诚实、准确地把"这是我的修复没做对"和"这是一个独立于我的改动之外、documented 的机制"分开,是这道题真正的区分点,而不是含糊地把两件事混着解释。

**可运行例子(1/2):完整复现"`reduce_retracing=True` 治不好 Python 标量退化"这条陷阱,以及它真正对什么生效:**

```python
import tensorflow as tf

# --- Part 1: the bug -- a python float that changes every call, passed directly (not as a Tensor) ---
@tf.function
def f_baseline(x, scale):
    return x * scale

x = tf.ones((8,))
for i in range(10):
    f_baseline(x, float(i) + 0.001)   # a different python float every call
n_baseline = f_baseline.experimental_get_tracing_count()

# --- Part 2: candidate's first-instinct fix -- "just add reduce_retracing=True" ---
@tf.function(reduce_retracing=True)
def f_reduce(x, scale):
    return x * scale

for i in range(10):
    f_reduce(x, float(i) + 0.001)     # SAME bug pattern: a different python float every call
n_reduce_pyfloat = f_reduce.experimental_get_tracing_count()

# --- Part 3: prove reduce_retracing=True is NOT broken in general -- it genuinely helps when the
# thing that varies is a Tensor's SHAPE, which is what it was actually designed for ---
@tf.function(reduce_retracing=True)
def g_reduce_shape(x):
    return tf.reduce_sum(x)

for n in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
    g_reduce_shape(tf.ones((n,)))     # a different TENSOR SHAPE every call
n_reduce_shape = g_reduce_shape.experimental_get_tracing_count()

@tf.function(reduce_retracing=False)
def g_noreduce_shape(x):
    return tf.reduce_sum(x)

for n in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
    g_noreduce_shape(tf.ones((n,)))
n_noreduce_shape = g_noreduce_shape.experimental_get_tracing_count()

# --- Part 4: the correct fix -- wrap the python scalar as a Tensor so it becomes a Placeholder
# (graph input) instead of a baked-in constant ---
@tf.function
def f_fixed(x, scale):
    return x * scale

for i in range(10):
    f_fixed(x, tf.constant(float(i) + 0.001))
n_fixed = f_fixed.experimental_get_tracing_count()

print(f"baseline (python float scale, 10 distinct values):        trace_count = {n_baseline} / 10 calls")
print(f"reduce_retracing=True, SAME python-float bug:               trace_count = {n_reduce_pyfloat} / 10 calls")
print(f"reduce_retracing=True, varying TENSOR SHAPE instead:        trace_count = {n_reduce_shape} / 10 calls")
print(f"reduce_retracing=False, varying TENSOR SHAPE (control):     trace_count = {n_noreduce_shape} / 10 calls")
print(f"fixed (scale wrapped as tf.constant):                       trace_count = {n_fixed} / 10 calls")

assert n_baseline == 10, "every distinct python float should retrace"
assert n_reduce_pyfloat == 10, "reduce_retracing=True does NOT fix the python-scalar case"
assert n_reduce_shape < 10, "reduce_retracing=True DOES generalize across tensor shapes"
assert n_noreduce_shape == 10, "without reduce_retracing, shape changes retrace every time (control)"
assert n_fixed < n_baseline, "wrapping as tf.constant should cut retracing dramatically"

# --- Part 5: why did an earlier run of "the fix" show trace_count==2, not 1? Isolate whether this
# is a residual flaw, or an unrelated phenomenon -- creating optimizer slot variables INSIDE a
# tf.function on its first call is documented to trace twice, independent of argument values. ---
model = tf.keras.Sequential([tf.keras.layers.Dense(8, activation='relu'), tf.keras.layers.Dense(4)])
opt = tf.keras.optimizers.SGD(0.01)

@tf.function
def step_with_var_creation(a, b, coef):
    with tf.GradientTape() as tape:
        logits = model(a, training=True)
        loss = coef * tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(b, logits, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))   # creates optimizer slot vars on call 1
    return loss

a = tf.random.normal((16, 8))
b = tf.random.uniform((16,), maxval=4, dtype=tf.int32)
fixed_coef = tf.constant(1.0)          # NEVER changes across any of these calls
for _ in range(5):
    step_with_var_creation(a, b, fixed_coef)
n_with_var_creation = step_with_var_creation.experimental_get_tracing_count()

w = tf.Variable(tf.random.normal((8, 4)))   # variable created OUTSIDE the tf.function, up front

@tf.function
def step_no_var_creation(a, coef):
    return coef * tf.matmul(a, w)

for _ in range(5):
    step_no_var_creation(a, fixed_coef)
n_no_var_creation = step_no_var_creation.experimental_get_tracing_count()

print(f"\nfunction that creates optimizer vars on call 1, coef NEVER changes: trace_count = {n_with_var_creation} / 5 calls")
print(f"function with no variable creation inside, coef NEVER changes:      trace_count = {n_no_var_creation} / 5 calls")

assert n_with_var_creation == 2, "creating variables inside tf.function on first call traces twice"
assert n_no_var_creation == 1, "with no variable creation inside, a truly constant input traces once"
print("\nOK: a fix's trace_count=2 (instead of the theoretical minimum 1) can be fully explained by "
      "optimizer-slot-variable creation on the first call -- a separate, well-known tf.function quirk -- "
      "not a residual flaw in wrapping the scalar as a Tensor. Misattributing this would send the "
      "diagnosis down the wrong path.")
```

本机实测(WSL2 `~/tf-venv`,TensorFlow 2.21.0):Part 1-4 的五个 `trace_count` 依次是 `10, 10, 2, 10, 1`,精确对应"Python 标量退化对 `reduce_retracing` 免疫、Tensor shape 变化对 `reduce_retracing` 敏感、正确修复后一步到位"这条结论;Part 5 里,函数体内创建了 optimizer slot 变量的版本,即使传入的 `coef` 全程是同一个 `tf.constant(1.0)` 从未变化,`trace_count` 依然是 2(不是 1),而没有变量创建的对照组是 1——精确验证了"第一次调用创建变量会额外多 trace 一次"这条与参数值无关的独立机制。

**可运行例子(2/2):量化"方案批判"真正省下来多少——25 步真实训练 step(3 层 Dense,1024 维,batch=256)的实测耗时对比:**

```python
import tensorflow as tf
import time

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10),
    ])

BATCH, IN_DIM, N_CALLS = 256, 1024, 25

# --- Scenario A (bug): a python float "coef" (e.g. a per-call loss-weighting coefficient) changes
# every call, gets passed directly (not wrapped in tf.constant) into a @tf.function train step.
model_a = build_model()
opt_a = tf.keras.optimizers.SGD(0.01)

@tf.function
def step_bug(x, y, coef):
    with tf.GradientTape() as tape:
        logits = model_a(x, training=True)
        loss = coef * tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
    grads = tape.gradient(loss, model_a.trainable_variables)
    opt_a.apply_gradients(zip(grads, model_a.trainable_variables))
    return loss

x = tf.random.normal((BATCH, IN_DIM))
y = tf.random.uniform((BATCH,), maxval=10, dtype=tf.int32)

t0 = time.perf_counter()
for i in range(N_CALLS):
    step_bug(x, y, 1.0 + i * 0.001)   # a different python float every call -> retrace every call
t_bug = time.perf_counter() - t0
n_trace_bug = step_bug.experimental_get_tracing_count()

# --- Scenario B (fixed): same coefficient, but wrapped as tf.constant before being passed in
model_b = build_model()
opt_b = tf.keras.optimizers.SGD(0.01)

@tf.function
def step_fixed(x, y, coef):
    with tf.GradientTape() as tape:
        logits = model_b(x, training=True)
        loss = coef * tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
    grads = tape.gradient(loss, model_b.trainable_variables)
    opt_b.apply_gradients(zip(grads, model_b.trainable_variables))
    return loss

t0 = time.perf_counter()
for i in range(N_CALLS):
    step_fixed(x, y, tf.constant(1.0 + i * 0.001))   # same varying value, but as a Tensor -> traced once
t_fixed = time.perf_counter() - t0
n_trace_fixed = step_fixed.experimental_get_tracing_count()

print(f"Scenario A (bug, python float each call): {N_CALLS} calls in {t_bug*1000:.1f}ms total, "
      f"{t_bug/N_CALLS*1000:.2f}ms/call avg, trace_count={n_trace_bug}")
print(f"Scenario B (fixed, tf.constant each call): {N_CALLS} calls in {t_fixed*1000:.1f}ms total, "
      f"{t_fixed/N_CALLS*1000:.2f}ms/call avg, trace_count={n_trace_fixed}")
print(f"speedup = {t_bug/t_fixed:.2f}x, trace_count ratio = {n_trace_bug}/{n_trace_fixed}")

assert n_trace_bug > n_trace_fixed * 5
assert t_bug > t_fixed * 2
```

本机实测(3 层 Dense、1024 维、batch=256、25 次调用):Scenario A(bug)总耗时 4222.6ms,平均 168.90ms/call,`trace_count=26`;Scenario B(fixed)总耗时 779.9ms,平均 31.20ms/call,`trace_count=2`——速度提升 **5.41 倍**。`trace_count` 从 26 降到 2 而不是从 25 降到 1,原因和例 1 Part 5 揭示的机制完全一致:第 0 次调用因为 optimizer 创建 slot 变量额外多 trace 一次(贡献 2),后续 24 个不同的 Python `float` 值在 Scenario A 里各自触发一次新 trace(贡献 24,合计 26),而 Scenario B 里这 24 次不再触发任何新 trace(维持在 2)。

**常见坑:** 把 WARNING 文本列出的三条建议(包成 Tensor、`input_signature`、`reduce_retracing=True`)当成"三选一都能解决同一个问题"的等价选项,不去分辨每条各自针对哪一种根因;看到"`trace_count` 从一个大数字降到一个小数字"就直接下结论"修好了",不继续深挖"这个小数字为什么不是理论最小值 1",把一个独立现象(变量创建导致的首次多 trace)误判成自己方案的残留缺陷,浪费时间去"修复"一个根本不存在的问题。

---

## 案例 2:GPU 显存贪婪分配的"防御性配置"到底防住了什么——一次双进程实测颠覆预期(真实性验证轴,兼工程约束递增轴)

建立在 [10 类](10-memory-and-performance.md) 第 1 节(`set_memory_growth`/`TF_FORCE_GPU_ALLOW_GROWTH` 的贪婪分配机制——已用单进程实测出默认贪婪模式一次极小运算就吃掉 13873MiB、约 84.7% 整卡容量)之上。10 类第 1 节给出的动机是"避免抢占其它共享同一张卡的进程的显存",这个案例把这句话当成一个需要现场验证的具体断言,而不是直接照抄——用两个真正独立的 OS 进程(不是虚拟设备,是两个进程真实共享同一张物理卡)去验证它,而且验证过程中现场发现了一个和直觉不完全一致、需要如实报告的真实现象。

**追问链条完整还原:**

- **Q(基础):** "为什么这个仓库的 `~/tf-venv/bin/activate` 要把 `TF_FORCE_GPU_ALLOW_GROWTH=true` 写死在激活脚本里,而不是让每个训练脚本自己决定要不要开?"—— 期望答出:默认贪婪分配会一次性抢占几乎整张卡,如果多个进程/脚本共享同一张物理卡(开发机上开着 Jupyter 同时跑单测,这正是本系列自己的真实验证场景),写在环境变量层面能保证"只要用这个 venv,就不会有人不小心变成那个贪婪进程",比指望每个脚本作者自己记得调用 `set_memory_growth()` 更可靠。
- **追问 1(决策依据,不是纠错):** "如果我的训练脚本自己已经规规矩矩开了 growth 模式,还有必要要求整个团队/所有共享这张卡的脚本都统一配置吗?自己做对是不是就够了?"—— 期望候选人能想到:自己开 growth 模式只保证"我自己不是那个贪婪进程",不能保证"我不会被别的贪婪进程影响"——这是两件不同的事;决策依据应该是"在 venv/镜像层面统一配置,覆盖所有共享者",而不是依赖每个使用者自觉。
- **追问 2(真实性验证,核心反转):** "你能不能现场证明,一个默认贪婪的邻居进程确实会让你这边配置正确(growth 模式)的进程 OOM?"—— 期望候选人现场搭一个真实的双进程实验:一个模拟"贪婪邻居"(默认分配),一个模拟"自己"(growth 模式,之后尝试申请一大块张量),看是否真的复现"被挤爆"。
- **深挖追问(如果现场结果和预期不一致,考验候选人诚实面对而不是慌乱圆场):** 现场验证的真实结果是:即使邻居已经按 `nvidia-smi` 读数抓了 13873MiB(84.7% 的卡),自己这边随后申请 8192MiB、甚至 13312MiB **仍然成功**——不是预期中的 OOM。候选人这时候的第一反应应该是什么?—— 期望不要立刻怀疑"实验搭错了"就放弃,也不要不假思索地宣布"growth mode 果然没用",而是先重新审视自己的测量方法本身有没有混淆变量(这里确实先踩过一个真实的坑:一开始用 `tf.random.normal` 而不是 `tf.zeros` 做占位分配,现场测出前者生成期间的峰值显存能到最终请求量的 **3 倍左右**(申请 2048MiB 峰值到 6144MiB、申请 4096MiB 峰值到 12288MiB),必须先换成峰值可预测的 `tf.zeros`,排除这个混淆变量之后,再把观察到的现象当成需要如实报告、需要进一步定界的真实发现,而不是随手解释掉。
- **深挖追问(继续定界,不能只满足于"发现了一个反直觉现象"):** "这种'看起来能超发'的情况是没有上限的吗?"—— 期望候选人继续设计实验找边界:一是在两个进程都还存活、都还持有各自张量的那一刻,用 `nvidia-smi` 测组合总占用;二是把请求量推到明显超过卡物理容量,看是不是还能一直成功。现场验证出:组合总占用被收在物理 16384MiB 容量之内(实测 16109MiB,两次独立重跑数字完全一致),且把请求量推到 15000MiB 以上确实会真实触发 `ResourceExhaustedError`——说明这不是无限超发,是一个比"卡容量减去邻居用量"这种朴素算术更宽松、但仍然存在真实边界的动态机制。机制的具体细节(WSL2 这一层的 GPU 显存虚拟化到底怎么在两个进程间协调)超出了用户态 Python 脚本能确认的范围,如实标注这是这次验证的边界,不强行给出一个没有把握的具体解释。
- **深挖追问(工程含义收尾):** "结合这些发现,你还会不会推荐'统一在 venv/镜像层面配置 growth 模式'这个做法?"—— 期望:会,这仍然是唯一对自己没有代价、只有好处的配置(至少保证自己不会变成那个贪婪邻居);但候选人应该诚实说明:不能把它包装成"配了就不会被挤爆"的保证,尤其是在这类虚拟化/共享显存环境里,真正可靠的容量规划必须依赖实际测量(peak memory + 真实压测),而不是心算"卡的总容量减去别人报告的用量"。

**可运行例子(1/2):双进程实测——默认贪婪 vs growth 模式,以及"贪婪邻居"是否真的能挤爆一个配置正确的进程:**

```python
import multiprocessing as mp
import os
import subprocess


def nvidia_smi_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


def hog_process(allow_growth, ready_evt, release_evt, grabbed_q):
    """Simulates a co-located, independently-owned process on a shared GPU machine
    (a colleague's notebook kernel, another team's job) that we do not control the code of."""
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" if allow_growth else "false"
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.constant([1.0, 2.0, 3.0])
        _ = (x * 2).numpy()   # first real GPU op -> triggers (or not) the greedy pre-allocation
    grabbed_q.put(nvidia_smi_used_mib())
    ready_evt.set()                 # "I have already grabbed my share, go ahead"
    release_evt.wait(timeout=90)    # keep holding the memory until told to exit


def victim_process(alloc_mb, ready_evt, result_q):
    """The well-behaved process: sets growth mode itself, waits for the hog to grab memory
    first, then tries to allocate alloc_mb MiB and reports whether it actually succeeded."""
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf
    ready_evt.wait(timeout=90)
    n_floats = alloc_mb * 1024 * 1024 // 4
    try:
        with tf.device('/GPU:0'):
            big = tf.zeros((n_floats,), dtype=tf.float32)   # tf.zeros: peak == requested size,
            s = float(tf.reduce_sum(big).numpy())             # unlike tf.random.normal (~3x peak)
        result_q.put(f"SUCCESS sum={s}")
    except tf.errors.ResourceExhaustedError as e:
        result_q.put(f"OOM {str(e)[:160]}")


def run_round(hog_growth, victim_alloc_mb):
    ctx = mp.get_context("fork")   # fork, not spawn: no real __main__ file for a spawned child to
    ready_evt, release_evt = ctx.Event(), ctx.Event()   # re-import (same precedent as 11 类)
    grabbed_q, result_q = ctx.Queue(), ctx.Queue()
    p_hog = ctx.Process(target=hog_process, args=(hog_growth, ready_evt, release_evt, grabbed_q))
    p_victim = ctx.Process(target=victim_process, args=(victim_alloc_mb, ready_evt, result_q))
    p_hog.start(); p_victim.start()
    p_victim.join(timeout=100)
    release_evt.set()
    p_hog.join(timeout=30)
    assert not p_victim.is_alive() and not p_hog.is_alive(), "a worker process hung"
    grabbed_mib = grabbed_q.get(timeout=5)
    result = result_q.get(timeout=5)
    return grabbed_mib, result


if __name__ == "__main__":
    VICTIM_ALLOC_MB = 8192   # ~1/2 of the 16384MiB card -- clearly bigger than the ~2511MiB that
                             # "capacity minus default-greedy-hog" arithmetic would predict as free

    grabbed_default, result_default = run_round(hog_growth=False, victim_alloc_mb=VICTIM_ALLOC_MB)
    grabbed_growth, result_growth = run_round(hog_growth=True, victim_alloc_mb=VICTIM_ALLOC_MB)

    print(f"hog uses TF's default (greedy) allocation: grabbed {grabbed_default} MiB (nvidia-smi)")
    print(f"  -> a SEPARATE growth-mode process requesting {VICTIM_ALLOC_MB} MiB afterwards: {result_default}")
    print(f"hog explicitly uses growth mode: grabbed only {grabbed_growth} MiB (nvidia-smi)")
    print(f"  -> a SEPARATE growth-mode process requesting {VICTIM_ALLOC_MB} MiB afterwards: {result_growth}")

    # Solid, reproducible facts: default grabs ~85% of the card, growth grabs only a sliver --
    # this part exactly reproduces 10-class section 1's finding in a genuine 2-OS-process setting.
    assert grabbed_default > 10 * 1024, "default (greedy) hog should grab most of the 16GiB card"
    assert grabbed_growth < 2 * 1024, "growth-mode hog should only grab a small, need-based amount"

    # The surprising part, stated as an observation rather than an assumption: on THIS WSL2 setup,
    # the victim's later request for 8192MiB succeeds in BOTH rounds -- including the one where the
    # hog already holds 13873MiB and naive arithmetic (16384-13873=2511MiB) says it should not fit.
    print(f"\nOK: hog default-greedy grab={grabbed_default}MiB vs growth grab={grabbed_growth}MiB "
          f"(the difference itself is exactly as expected). But the victim's {VICTIM_ALLOC_MB}MiB "
          f"request result was '{result_default.split()[0]}' when sharing with the default-greedy hog -- "
          f"if that says SUCCESS rather than OOM, it means 'another process's nvidia-smi-reported usage' "
          f"is not a reliable stand-in for 'how much room is actually left for me', at least not on this "
          f"WSL2 GPU setup. See the next example for how far this tolerance actually goes.")
```

本机实测(两个真实独立的 OS 进程,`fork` 启动):默认贪婪 hog 抓了 **13873 MiB**(和 10 类第 1 节单进程测出的数字完全一致),growth 模式 hog 只抓了 **191 MiB**;但令人意外的是,**两种情况下** victim 后续申请 8192MiB 都成功了——包括邻居已经抓走 84.7% 显存的那一轮。

**可运行例子(2/2):定界这个"超预期成功"的真实边界——组合总占用与更大请求量的对照:**

```python
import multiprocessing as mp
import os
import subprocess


def nvidia_smi_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


def hog_process(ready_evt, release_evt):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
    import tensorflow as tf
    with tf.device('/GPU:0'):
        x = tf.constant([1.0, 2.0, 3.0])
        _ = (x * 2).numpy()
    ready_evt.set()
    release_evt.wait(timeout=90)


def victim_process(alloc_mb, ready_evt, both_alive_evt, release_evt, result_q):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf
    ready_evt.wait(timeout=90)
    n_floats = alloc_mb * 1024 * 1024 // 4
    try:
        with tf.device('/GPU:0'):
            big = tf.zeros((n_floats,), dtype=tf.float32)
            s = float(tf.reduce_sum(big).numpy())
        result_q.put(f"SUCCESS sum={s}")
        both_alive_evt.set()          # still holding the tensor -> safe to measure combined usage now
        release_evt.wait(timeout=30)
    except tf.errors.ResourceExhaustedError as e:
        result_q.put(f"OOM {str(e)[:160]}")
        both_alive_evt.set()


def run_round(victim_alloc_mb, measure_overlap):
    ctx = mp.get_context("fork")
    ready_evt, both_alive_evt, release_evt = ctx.Event(), ctx.Event(), ctx.Event()
    result_q = ctx.Queue()
    p_hog = ctx.Process(target=hog_process, args=(ready_evt, release_evt))
    p_victim = ctx.Process(target=victim_process,
                            args=(victim_alloc_mb, ready_evt, both_alive_evt, release_evt, result_q))
    p_hog.start(); p_victim.start()

    combined_mib = None
    if measure_overlap:
        got = both_alive_evt.wait(timeout=100)
        if got:
            combined_mib = nvidia_smi_used_mib()   # both processes' tensors are alive right now

    release_evt.set()
    p_victim.join(timeout=100)
    p_hog.join(timeout=30)
    assert not p_victim.is_alive() and not p_hog.is_alive(), "a worker process hung"
    result = result_q.get(timeout=5) if not result_q.empty() else "<no result>"
    return combined_mib, result


if __name__ == "__main__":
    CARD_CAPACITY_MIB = 16384   # this machine's card (see 00-roadmap.md environment declaration)

    # Part 1: while a default-greedy hog (grabs ~13873MiB alone) and a growth-mode victim asking for
    # 13312 MiB (the largest size a lone growth-mode process could get -- verified separately: 14336MiB
    # alone already OOMs) are BOTH simultaneously holding their tensors, what does nvidia-smi actually
    # report as the combined total? Naive arithmetic (13873+13312=27185MiB) is 1.66x the physical card.
    combined_mib, result_13312 = run_round(victim_alloc_mb=13312, measure_overlap=True)
    print(f"victim requests 13312 MiB while hog(default-greedy) holds its share: {result_13312}")
    print(f"nvidia-smi TOTAL usage while BOTH are simultaneously alive: {combined_mib} MiB "
          f"(nominal card capacity: {CARD_CAPACITY_MIB} MiB; naive 'hog+victim' sum would be 27185 MiB)")

    # Part 2: this tolerance is NOT unlimited -- push the victim's request further and confirm a real,
    # reproducible ceiling still exists under the exact same contention setup.
    _, result_15000 = run_round(victim_alloc_mb=15000, measure_overlap=False)
    print(f"\nvictim requests 15000 MiB while hog(default-greedy) holds its share: {result_15000[:120]}")

    assert result_13312.startswith("SUCCESS"), "13312MiB alongside the hog was observed to succeed"
    assert combined_mib is not None and combined_mib <= CARD_CAPACITY_MIB, \
        "the combined simultaneous usage should still be reconciled to fit under the physical card"
    assert combined_mib > 13312, "the combined reading should still reflect real, substantial usage"
    assert result_15000.startswith("OOM"), "a large enough request should still genuinely fail"

    print(f"\nOK: combined usage while both alive ({combined_mib}MiB) fits under the {CARD_CAPACITY_MIB}MiB "
          f"physical card -- so this is NOT limitless overcommit, but a real, reproducible ceiling that "
          f"sits well ABOVE what 'card_capacity - hog_reported_usage' arithmetic predicts. The mechanism "
          f"(how the two processes' allocations get reconciled under one physical card) is not something "
          f"this script can pin down from user-space alone -- what IS verified is that nvidia-smi's "
          f"'how much did the other process grab' number is not a safe way to compute 'how much is left "
          f"for me' on this WSL2 setup, and the only reliable check is trying the real allocation.")
```

本机实测(两次独立重跑,数字完全一致):两个进程同时存活时,`nvidia-smi` 测得的组合总占用是 **16109 MiB**,稳稳落在 16384 MiB 的物理容量之内,而不是"贪婪邻居的 13873 + victim 的 13312 = 27185"这种朴素算术给出的数字;但把 victim 的请求量推到 15000 MiB 时,在完全相同的邻居贪婪条件下,**真实触发了 `ResourceExhaustedError`**——证明这确实是一个有真实边界、动态协调出来的结果,不是没有上限的超发。

**常见坑:** 只做单进程测量就直接下结论"growth 模式能保护我不被挤爆",没有用真正独立的第二个进程验证过这条推论;发现实验结果和预期不一致时,第一反应是怀疑代码写错了、而不是先怀疑自己的测量方法本身(这里的真实教训是 `tf.random.normal` 生成期间的峰值显存开销,不能拿来测"占用了多少");把一次意外发现直接当成普适结论到处传播("WSL2 上显存能随便超发"),而不去进一步定界这个现象的真实边界在哪里——本文已经验证到"总量被收在物理容量之内,且推得足够大依然会真实 OOM",不是无条件成立。

---

## 案例 3:训练规模递增下的分布式决策——从单卡到 MirroredStrategy,该不该无脑上(规模递增轴,兼决策依据追问轴与工程约束递增轴)

建立在 [11 类](11-distributed-training-basics.md) 第 1 节(`MirroredStrategy` 机制:`scope()` 镜像变量创建 + `run()` 分发计算 + 梯度 all-reduce)、第 5 节(与 PyTorch DDP 的对比,已用 `time.sleep` 实验验证 eager 模式下副本线程严格串行)和第 6 节(和 `distributed-inference` 模块的分工——切数据 vs 切模型)之上,也呼应 [torch-deep-dive 12 号文件案例 3](../torch-deep-dive/12-advanced-interview-depth.md)(`DataParallel` 被 DDP 取代的决策依据)和案例 5(训练规模递增下的分布式决策)。11 类第 5 节的串行验证用的是 `time.sleep` 模拟,这个案例换成真实计算(Dense 堆叠的前向+反向)重新量化一遍同一个论点,同时补上 11 类没有正面回答过的一个问题:虚拟设备到底能不能验证"多卡更快"这件事本身。

**追问链条完整还原:**

- **Q(基础):** "单卡训练数据处理不过来,想加卡,TF 里怎么做?"—— 期望答出 `MirroredStrategy`:`scope()` 拦截变量创建做镜像复制,`run()` 把计算分发到各设备,`apply_gradients` 内部做梯度 all-reduce 同步(11 类第 1、2 节)。
- **追问 1(决策依据):** "只有一张物理卡的时候,你怎么验证这套多卡逻辑本身是对的,而不是等有第二张卡了才敢用?"—— 期望答出 `tf.config.set_logical_device_configuration` 切虚拟设备(11 类已验证的标准做法),但候选人应该主动补一句:这只能验证机制对不对,不能验证真实的多卡吞吐收益——追问 3 会把这条边界的具体后果现场测出来。
- **追问 2(规模递增,决策依据结合,区分度高):** "如果数据量再往上翻,一张卡的显存实在放不下了,你会不会直接上 `MirroredStrategy` 解决?"—— 期望候选人不要不假思索地说"会",而是先反问"放不下"具体指什么:如果是**模型本身**(参数+激活值)已经超过单卡显存,`MirroredStrategy` 没有用——它是"每张卡放一份完整模型副本,只切数据",不切模型本身,模型该放不下还是放不下(这是 11 类第 6 节已经讲清楚的"切数据 vs 切模型"分工,需要的是模型并行);只有当"模型本身一张卡放得下,但吞吐/数据量要求超过单卡算力"时,`MirroredStrategy` 才是对的工具。
- **追问 3(反直觉,要求现场量化,呼应"只有 2 张卡值不值得改造"这个开放问题):** "假设确实是吞吐问题,不是模型放不下——那你去现场测一下,用虚拟设备切出的 2 个'副本'训练,吞吐是不是真的接近单卡的 2 倍?"—— 期望候选人现场测出:虚拟 2 副本(同一张物理卡)不但没有接近 2 倍,反而比同样有效 batch 直接在单个逻辑设备上跑还**更慢**(实测 0.48x)。期望候选人能诚实、准确地解读这个数字:这不代表 `MirroredStrategy` 没用,而是暴露了虚拟设备测试的先天局限——两个副本抢占同一批物理 SM(Streaming Multiprocessor,GPU 上真正执行计算的硬件单元)和显存带宽,不可能有真实并行收益,只是把镜像同步的开销如实体现了出来;真正的吞吐收益必须要有第二张物理卡才能验证,这里如实标注"本机条件不允许"。
- **深挖追问(工程约束递增轴,呼应案例 2 的进程/调度对照):** "`MirroredStrategy` 单进程会不会重蹈 `DataParallel` 的覆辙?"—— 期望答出:eager 模式下确实有真实、可测的调度开销(11 类用 `sleep` 测过,这里用真实计算重新量化一遍:5 倍以上差距),但机制不是 GIL(TF 是单进程但用多线程调度,问题是副本线程被硬编码成严格串行执行,不是被 GIL 卡住),而是"没有编译成一张图、只能靠 Python 线程一步步牵着走";一旦用 `tf.function` 包裹,执行下沉到 C++ runtime,这部分开销大幅消失。
- **深挖追问(规模再递增到分布式,直接引用已验证内容,不重新做实验):** "如果数据/吞吐需求已经超过单机所有卡的总和呢?"—— 期望提出 `MultiWorkerMirroredStrategy`,靠 `TF_CONFIG` 环境变量发现集群,底层复用同一套 `CollectiveAllReduceStrategy`——这一步直接引用 [11 类第 4 节](11-distributed-training-basics.md)已经验证过的"2 个独立 OS 进程模拟 2-worker、`TF_CONFIG` 指向 `localhost` 不同端口"的跨进程 all-reduce 结果,不需要重新做实验;候选人应该能说清楚这一步和"虚拟设备"是不同性质的验证(虚拟设备是"同一张卡切分",这里是"真正独立的多个 OS 进程,原则上可以跨机器"),机制上更可信,这也是本案例的收尾——不是每一层新增的规模压力都需要从头验证,能准确指出"这一点已经在哪个案例/哪个知识点被验证过、边界在哪里",本身就是候选人可信度的一部分。

**可运行例子(1/2):MirroredStrategy 下,eager 直接调用 vs `tf.function` 包裹的真实计算耗时对比(2 个虚拟逻辑设备,真实前向+反向,不是 `sleep` 模拟):**

```python
import tensorflow as tf
import time

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=2048),
     tf.config.LogicalDeviceConfiguration(memory_limit=2048)],
)
strategy = tf.distribute.MirroredStrategy()
assert strategy.num_replicas_in_sync == 2

DIM, BATCH_PER_REPLICA, N_STEPS = 1024, 256, 15

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(DIM, activation='relu'),
        tf.keras.layers.Dense(DIM, activation='relu'),
        tf.keras.layers.Dense(DIM, activation='relu'),
        tf.keras.layers.Dense(10),
    ])
    opt = tf.keras.optimizers.SGD(0.01)

def make_step_fn():
    def step_fn(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    return step_fn

per_replica_x = strategy.experimental_distribute_values_from_function(
    lambda ctx: tf.random.normal((BATCH_PER_REPLICA, DIM)))
per_replica_y = strategy.experimental_distribute_values_from_function(
    lambda ctx: tf.random.uniform((BATCH_PER_REPLICA,), maxval=10, dtype=tf.int32))

# --- (a) eager: strategy.run(step_fn) called directly, no tf.function ---
step_eager = make_step_fn()
for _ in range(3):   # warmup (build layers, optimizer slots)
    strategy.run(step_eager, args=(per_replica_x, per_replica_y))
t0 = time.perf_counter()
for _ in range(N_STEPS):
    strategy.run(step_eager, args=(per_replica_x, per_replica_y))
t_eager = time.perf_counter() - t0

# --- (b) same computation, wrapped in tf.function ---
step_traced = tf.function(make_step_fn())
for _ in range(3):
    strategy.run(step_traced, args=(per_replica_x, per_replica_y))
t0 = time.perf_counter()
for _ in range(N_STEPS):
    strategy.run(step_traced, args=(per_replica_x, per_replica_y))
t_traced = time.perf_counter() - t0

print(f"eager strategy.run:      {N_STEPS} steps in {t_eager*1000:.1f}ms total, {t_eager/N_STEPS*1000:.2f}ms/step")
print(f"tf.function strategy.run: {N_STEPS} steps in {t_traced*1000:.1f}ms total, {t_traced/N_STEPS*1000:.2f}ms/step")
print(f"speedup = {t_eager/t_traced:.2f}x")

assert t_eager > t_traced * 2, "eager MirroredStrategy.run should carry real, measurable overhead"
```

本机实测(2 个虚拟逻辑设备,3 层 Dense、1024 维、每副本 batch=256,15 步):eager 方式 `798.3ms` 总耗时(`53.22ms/step`),`tf.function` 包裹后 `151.3ms`(`10.08ms/step`)——**5.28 倍**的真实速度差异,和 11 类用 `sleep` 测出的"严格串行"结论方向完全一致,这次换成了真实计算量化。

**可运行例子(2/2):量化虚拟设备测试的先天局限——同样有效 batch,单逻辑设备 vs 2 虚拟副本的吞吐对比:**

```python
import tensorflow as tf
import time

DIM, BATCH_PER_REPLICA, N_STEPS = 1024, 256, 20

# Split into 2 virtual devices FIRST (must happen before any GPU op, 11-class precedent) -- the
# "single-device baseline" below will just pin everything to logical device 0 and ignore device 1.
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=2048),
     tf.config.LogicalDeviceConfiguration(memory_limit=2048)],
)

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(DIM, activation='relu'),
        tf.keras.layers.Dense(DIM, activation='relu'),
        tf.keras.layers.Dense(DIM, activation='relu'),
        tf.keras.layers.Dense(10),
    ])

# --- (a) plain baseline pinned to ONE logical device: total batch = BATCH_PER_REPLICA ("1 replica") ---
with tf.device('/GPU:0'):
    model1 = build_model()
    opt1 = tf.keras.optimizers.SGD(0.01)

    @tf.function
    def step1(x, y):
        with tf.GradientTape() as tape:
            logits = model1(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
        grads = tape.gradient(loss, model1.trainable_variables)
        opt1.apply_gradients(zip(grads, model1.trainable_variables))
        return loss

    x1 = tf.random.normal((BATCH_PER_REPLICA, DIM))
    y1 = tf.random.uniform((BATCH_PER_REPLICA,), maxval=10, dtype=tf.int32)
    for _ in range(5):
        step1(x1, y1)
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        step1(x1, y1)
    t_single = time.perf_counter() - t0
samples_per_sec_single = (N_STEPS * BATCH_PER_REPLICA) / t_single

# --- (b) MirroredStrategy across BOTH logical devices (still the SAME physical GPU);
# total batch = 2 * BATCH_PER_REPLICA ("2 replicas") ---
strategy = tf.distribute.MirroredStrategy()
assert strategy.num_replicas_in_sync == 2

with strategy.scope():
    model2 = build_model()
    opt2 = tf.keras.optimizers.SGD(0.01)

def step_fn(x, y):
    with tf.GradientTape() as tape:
        logits = model2(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
    grads = tape.gradient(loss, model2.trainable_variables)
    opt2.apply_gradients(zip(grads, model2.trainable_variables))
    return loss

step2 = tf.function(step_fn)
px = strategy.experimental_distribute_values_from_function(
    lambda ctx: tf.random.normal((BATCH_PER_REPLICA, DIM)))
py = strategy.experimental_distribute_values_from_function(
    lambda ctx: tf.random.uniform((BATCH_PER_REPLICA,), maxval=10, dtype=tf.int32))
for _ in range(5):
    strategy.run(step2, args=(px, py))
t0 = time.perf_counter()
for _ in range(N_STEPS):
    strategy.run(step2, args=(px, py))
t_mirrored = time.perf_counter() - t0
samples_per_sec_mirrored = (N_STEPS * BATCH_PER_REPLICA * 2) / t_mirrored

print(f"single logical device, batch={BATCH_PER_REPLICA}: {t_single/N_STEPS*1000:.2f}ms/step, "
      f"{samples_per_sec_single:.0f} samples/sec")
print(f"MirroredStrategy 2 VIRTUAL replicas (same physical GPU), total batch={BATCH_PER_REPLICA*2}: "
      f"{t_mirrored/N_STEPS*1000:.2f}ms/step, {samples_per_sec_mirrored:.0f} samples/sec")
ratio = samples_per_sec_mirrored / samples_per_sec_single
print(f"throughput ratio (mirrored/single) = {ratio:.2f}x "
      f"(NOT expected to approach 2x -- both replicas share the SAME physical SMs/memory bus, "
      f"this only measures overhead, not real parallel compute capacity)")

assert ratio < 1.5, "2 virtual replicas on ONE physical GPU should NOT show anything close to 2x throughput"
```

本机实测:单逻辑设备(batch=256)`2.25ms/step`,吞吐 `113575 样本/秒`;2 个虚拟副本(总 batch=512)`9.37ms/step`,吞吐 `54641 样本/秒`——比值 **0.48x**,不但没有翻倍,反而比单设备直接跑同样有效 batch 更慢。这精确证明了"虚拟设备只能验证机制正确性,不能用来评估真实多卡吞吐收益"这条边界不是一句空洞的免责声明,而是能被数字直接量出来的真实限制。

**常见坑:** 一听到"显存不够/放不下"就无条件反射式地上 `MirroredStrategy`,不区分"模型本身太大"和"数据吞吐太大"两种完全不同的"放不下",前者 `MirroredStrategy` 无能为力;在只有一张物理卡的机器上,把虚拟设备测出的吞吐数字直接当成"多卡有没有收益"的证据(不管这个数字好看还是难看),而不去追问"虚拟设备的物理限制决定了这个数字压根不能回答这个问题";把 eager 模式下的调度开销问题简单归因成"GIL",和 PyTorch `DataParallel` 的真实瓶颈机制混为一谈,答不出 TF 这边"没编译成图、靠 Python 线程牵着走"这个具体机制。

---

## 案例 4:模型部署格式选型的方案批判迭代——从 SavedModel 到 .keras 到 TFLite,每一步都被戳一个新坑(方案批判迭代轴)

建立在 [12 类](12-serialization-and-deployment.md) 第 1 节(SavedModel,内嵌追踪图,即使自定义 Layer 没注册,`tf.keras.models.load_model()` 也能"复活"出一个能跑但 `isinstance` 失败的代理对象)、第 3 节(`.keras`,纯 config 驱动,类解析不了就硬报错,没有中间地带)和第 6 节(TFLite 转换与动态范围量化)之上。这个案例延续 [torch-deep-dive 12 号文件案例 4](../torch-deep-dive/12-advanced-interview-depth.md)"方案连续被指出具体缺陷,逼你换方案"的追问模式,但取材换成 TF 这条系列独有的真实工程决策——一个训练好的模型到底该存成哪种格式,而不是 torch 系列已经覆盖过的话题。

**追问链条完整还原(方案批判迭代,不是深挖同一方案):**

- **面试官给约束:** "你训练好一个用了自定义 Layer 的模型,现在要满足三件事:①这份文件要能被同事的 Python 训练脚本继续 fine-tune;②未来要能部署给一个不跑 Python 的 C++/Java 推理服务;③几个月后哪怕这几个自定义 Layer 的源码不小心丢了、没同步,至少还能加载出来做推理。给一个方案。"
- **候选人方案 1(最直觉的第一反应):** "都存成 SavedModel 不就行了,一个格式全占了。"
- **面试官指出这个方案的具体缺陷(不是"不够好",是"漏了一条硬要求"):** "SavedModel 确实满足②和③(12 类第 1 节验证过,即使自定义 Layer 没注册,`tf.saved_model.load()` 这条框架级路径完全不受影响,`tf.keras.models.load_model()` 也能'复活'出能跑的对象)。但你考虑过①吗——同事的 fine-tuning 脚本如果依赖 `isinstance()` 判断哪些层要冻结,SavedModel 复活出来的代理类根本不是原来那个类,`isinstance` 检查会**静默失败**,这不是'能不能跑'的问题,是'跑出来的结果不对'的问题,而且不会有任何报错提示你。"
- **候选人方案 2(换方案):** "那训练迭代场景专门用 `.keras`,把所有自定义 Layer 都用 `register_keras_serializable()` 注册好,这样 `isinstance` 不会有问题;部署场景另外单独导出一份 SavedModel。"
- **面试官指出新方案的代价:** "这样确实解决了①,但你怎么保证③?如果几个月后这几个自定义 Layer 真的没注册好(比如换了个新同事,漏掉了这个装饰器),`.keras` 会发生什么?"—— 期望候选人现场验证:`.keras` 对未注册的类是硬 `TypeError`,没有 SavedModel 那种"至少还能跑"的后备路径——③这条约束天生就不该指望 `.keras` 来满足,必须由 SavedModel 兜底。
- **候选人方案 3(整合,给出最终决策):** "那这两个格式各司其职,不是互相替代:训练迭代阶段用 `.keras`(+ 严格要求自定义 Layer 都注册)服务①;每次真正要交付部署的时候,从最新 checkpoint 重新单独导出一份 SavedModel 服务②③;如果还要塞进移动端,再从这份 SavedModel 转一层 TFLite。"
- **面试官追问这个方案的代价(决策依据,要求量化,不是空对空的"两份文件不好维护"):** "维护两份格式,团队真的能接受吗?具体到磁盘/传输成本上,这些格式的体积差异大不大?TFLite 那一步的量化又能省多少?"—— 期望候选人能用实测数字而不是含糊说"应该差不多"来回答,并且能诚实说明"两份真理来源"为什么不是真风险:SavedModel 应该是"训练完成后的产物",不是训练过程中的中间态,只要构建流程明确"交付部署前重新从最新 checkpoint 导出",就不存在不同步的风险。

**可运行例子(1/2):同一个训练好的模型,五种交付形态的磁盘体积实测对比:**

```python
import os
import tempfile
import shutil
import numpy as np
import tensorflow as tf


def dir_size_bytes(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def kb(n):
    return n / 1024


tmpdir = tempfile.mkdtemp()

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10),
])
x = tf.constant(np.random.RandomState(0).randn(4, 32).astype("float32"))
y_ref = model(x).numpy()

sm_dir = os.path.join(tmpdir, "sm")
h5_path = os.path.join(tmpdir, "model.h5")
keras_path = os.path.join(tmpdir, "model.keras")
model.save(sm_dir)
model.save(h5_path)
model.save(keras_path)

converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
tflite_plain = converter.convert()
converter2 = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant = converter2.convert()
tflite_plain_path = os.path.join(tmpdir, "model_plain.tflite")
tflite_quant_path = os.path.join(tmpdir, "model_quant.tflite")
with open(tflite_plain_path, "wb") as f:
    f.write(tflite_plain)
with open(tflite_quant_path, "wb") as f:
    f.write(tflite_quant)

sizes = {
    "SavedModel (dir)": dir_size_bytes(sm_dir),
    ".h5 (single file)": dir_size_bytes(h5_path),
    ".keras (single file)": dir_size_bytes(keras_path),
    ".tflite (no quant)": dir_size_bytes(tflite_plain_path),
    ".tflite (dynamic-range quant)": dir_size_bytes(tflite_quant_path),
}
for name, sz in sizes.items():
    print(f"{name:32s}: {kb(sz):8.1f} KB")

# Sanity: all formats reproduce the same numeric output (this comparison is only meaningful if
# it's comparing genuinely equivalent artifacts, not accidentally comparing a broken export)
assert np.allclose(tf.keras.models.load_model(sm_dir)(x).numpy(), y_ref, atol=1e-5)
assert np.allclose(tf.keras.models.load_model(h5_path)(x).numpy(), y_ref, atol=1e-5)
assert np.allclose(tf.keras.models.load_model(keras_path)(x).numpy(), y_ref, atol=1e-5)

assert sizes["SavedModel (dir)"] > sizes[".keras (single file)"], \
    "SavedModel carries a traced graph on top of weights+config, so it should be larger"
assert sizes[".tflite (dynamic-range quant)"] < sizes[".tflite (no quant)"] * 0.6, \
    "dynamic-range weight quantization should meaningfully shrink the .tflite file"
print(f"\nOK: SavedModel({kb(sizes['SavedModel (dir)']):.0f}KB) is the biggest (graph+weights+keras "
      f"metadata all bundled); .keras/.h5 are smaller (weights+config JSON only); quantized .tflite "
      f"({kb(sizes['.tflite (dynamic-range quant)']):.0f}KB) is smallest -- "
      f"{sizes['.tflite (no quant)']/sizes['.tflite (dynamic-range quant)']:.2f}x bigger before quantization.")

shutil.rmtree(tmpdir, ignore_errors=True)
```

本机实测(3 层 Dense、32 维输入的小模型):SavedModel 目录 `150.3 KB`,`.h5` `102.6 KB`,`.keras` `99.1 KB`,未量化 TFLite `86.5 KB`,动态范围量化后 TFLite `26.0 KB`——量化前后差 **3.33 倍**。体积排序精确对应"SavedModel 多带一份追踪图和 Keras 元信息 → `.keras`/`.h5` 只有权重+config → TFLite 精简算子集 → 量化后权重压缩成 int8"这条逐层裁剪的逻辑链。

**可运行例子(2/2):"`isinstance` 失败"不是学究式的洁癖,而是真实会让 fine-tuning 脚本悄悄跑错的 bug:**

```python
import os
import tempfile
import numpy as np
import tensorflow as tf


class FrozenScaleLayer(tf.keras.layers.Layer):
    """A custom layer a fine-tuning script needs to recognize (e.g. to keep it frozen)."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units)

    def call(self, x):
        return self.dense(x) * 0.5

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.dense.units})
        return cfg


def freeze_all_frozen_scale_layers(model):
    """Realistic downstream fine-tuning utility: freeze every FrozenScaleLayer, train everything
    else. This is exactly the kind of code a colleague's fine-tuning script would actually run --
    not a pedantic isinstance() check for its own sake."""
    frozen_count = 0
    for layer in model.layers:
        if isinstance(layer, FrozenScaleLayer):
            layer.trainable = False
            frozen_count += 1
    return frozen_count


tmpdir = tempfile.mkdtemp()
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8,)),
    FrozenScaleLayer(16, name="frozen_scale"),
    tf.keras.layers.Dense(4, name="trainable_head"),
])
x = tf.constant(np.random.RandomState(0).randn(2, 8).astype("float32"))
y_ref = model(x).numpy()
sm_dir = os.path.join(tmpdir, "sm_model")
model.save(sm_dir)

# Scenario A: same process, same session -- freezing works exactly as intended (baseline / control)
n_frozen_same_process = freeze_all_frozen_scale_layers(model)
assert n_frozen_same_process == 1
assert model.get_layer("frozen_scale").trainable is False
print(f"[same-process control] froze {n_frozen_same_process} FrozenScaleLayer(s) -- as expected")

# Scenario B: a colleague's fine-tuning script, in a FRESH process/session, reloads the SavedModel
# via tf.keras.models.load_model() -- exactly the "①能被同事的训练脚本继续fine-tune" requirement.
# The class is NOT registered with register_keras_serializable(), simulating the common case where
# a quick custom layer was never set up for that (12-class already showed this "revives but isn't
# isinstance-true" fact in isolation; this example shows the downstream fine-tuning consequence).
reloaded = tf.keras.models.load_model(sm_dir)
assert np.allclose(reloaded(x).numpy(), y_ref, atol=1e-5), "numerically the reload is fine"

n_frozen_reloaded = freeze_all_frozen_scale_layers(reloaded)
revived_layer = reloaded.get_layer("frozen_scale")
print(f"[reloaded via tf.keras.models.load_model] type(revived_layer) = {type(revived_layer)}")
print(f"[reloaded via tf.keras.models.load_model] froze {n_frozen_reloaded} FrozenScaleLayer(s)")

assert type(revived_layer).__name__ == "FrozenScaleLayer"       # looks right by name
assert isinstance(revived_layer, FrozenScaleLayer) is False       # but is NOT the real class
assert n_frozen_reloaded == 0                                       # so the freeze silently does nothing
assert revived_layer.trainable is True                              # this layer will get fine-tuned
                                                                       # by mistake, with no error anywhere

print("\nOK: the fine-tuning script's freeze_all_frozen_scale_layers() runs without any error or "
      "warning, reports freezing 0 layers instead of 1, and the layer that was supposed to stay "
      "frozen goes on to be fine-tuned by mistake -- this is the real, functional cost of "
      "'isinstance fails after reload', not a pedantic type-purity complaint.")

# The actual fix: pass custom_objects= explicitly so the real class is used on reload.
reloaded_fixed = tf.keras.models.load_model(sm_dir, custom_objects={"FrozenScaleLayer": FrozenScaleLayer})
n_frozen_fixed = freeze_all_frozen_scale_layers(reloaded_fixed)
assert n_frozen_fixed == 1
assert isinstance(reloaded_fixed.get_layer("frozen_scale"), FrozenScaleLayer) is True
print("Fixed via custom_objects=: froze", n_frozen_fixed, "layer(s) correctly.")
```

本机实测:同进程控制组正确冻结 1 层;通过 `tf.keras.models.load_model()` 重新加载后,`type(revived_layer)` 显示为 `<class 'tf_keras.src.saving.legacy.saved_model.load.FrozenScaleLayer'>`——名字对得上,但 `isinstance` 检查返回 `False`,`freeze_all_frozen_scale_layers()` 在没有任何报错或警告的情况下,实际冻结数从 1 变成 **0**,这个本该被冻结的层会在后续 fine-tuning 里被意外训练;传入 `custom_objects=` 后恢复正确,冻结数回到 1。

**常见坑:** 把"能不能存下来、能不能加载"当成格式选型唯一的判断标准,忽视了"加载出来的对象在下游代码里的类型行为是否符合预期"这个更隐蔽的维度;面对"未注册自定义类"这种边界情况,想当然地假设所有格式的失败模式都一样(都是能跑但类型有问题,或者都是硬报错),而不亲自验证每种格式各自的真实行为;讨论"两份格式维护成本"时给不出具体数字,只会重复"维护两份东西比较麻烦"这种没有说服力的空话。

---

## 案例 5:Keras 2/3 环境声明的真实性验证边界——"装对了环境"这句话到底保证了什么(真实性验证轴)

建立在 [00-roadmap.md](00-roadmap.md) 环境声明一节(本系列显式安装 `tf_keras` 并设置 `TF_USE_LEGACY_KERAS=1`,让 `tf.keras` 解析回经典实现)、[12 类第 8 节](12-serialization-and-deployment.md)(已经验证"包装了不代表默认生效,环境变量才是真正的开关",并验证过 `tf.keras` 这一条访问路径在两种"没配对"情况下各自的具体报错)和 [13 类第 8 节](13-debugging-and-common-errors.md)(Keras 2/3 版本冲突报错排查)之上。12 类第 8 节已经把"`tf.keras` 这条路径解析到哪"验证得很彻底,这个案例问一个 12/13 类都还没有正面回答过的问题:这句环境声明,对"`tf.keras`"这个属性访问路径**之外**的东西,还成立吗?

**追问链条完整还原:**

- **Q(基础):** "`00-roadmap.md` 的环境声明说'装了 `tf_keras` 并设置 `TF_USE_LEGACY_KERAS=1`,让 `tf.keras` 解析回经典实现'——这句话具体保证了什么?"—— 期望答出:`tf.keras.*` 这条访问路径解析到 Keras 2(`tf_keras` 包),12 类第 8 节已经反复验证过。
- **追问 1(真实性验证,反直觉,这是本案例的核心):** "如果同一个进程里,除了用 `tf.keras`,还单独写了一行 `import keras`(不通过 `tf.keras` 这个属性,直接导入顶层包),这个 `keras` 解析到的是 Keras 2 还是 Keras 3?"—— 期望候选人不要想当然地认为"反正这个环境配置对了,所有 `keras` 相关的东西肯定都是 Keras 2",而是能推理出:环境变量 + `tf_keras` 包解决的只是"`tf.keras` 这个属性访问路径"的重定向逻辑,不代表"`keras`"这个顶层包名本身被换掉了或者不存在了——这是两条完全独立的 import 路径,必须现场验证,不能靠"环境配对了就该都对"这种推广式的信心去回答。
- **追问 2(工程后果,不是空对空的"两个不一样"):** "如果不小心在同一个模型里混用了 `tf.keras.layers.Dense` 和裸 `keras.layers.Dense`,会发生什么?"—— 期望候选人现场验证:立刻被 `tf_keras` 的 `Sequential.add()` 拒绝,报 `TypeError`("must be an instance of class Layer"),报错本身还算清楚,直接点出了类型不对。
- **深挖追问(第二层反直觉,考验"不要因为一个例子报错清楚,就假设所有场景都一样清楚"):** "如果混用的不是 layer,而是 callback——比如某个第三方库内部用裸 `import keras` 写了一个 `Callback` 子类,你把它传给 `tf.keras` 的 `model.fit(callbacks=[...])` 会怎样?"—— 期望候选人不要预设"和上面 layer 的情况一样,会有个清楚的报错",而是现场验证:这次失败的方式完全不同——一个和"Keras 2/3"毫无关联字眼的 `AttributeError: '...' object has no attribute '_implements_train_batch_hooks'`,如果不知道背景,几乎不可能从这条报错文本本身推断出真正原因是"这个 callback 来自另一个 Keras 大版本"。这个反差本身就是本案例最核心的论点:同一个根因(裸 `import keras` 绕过了环境声明的保护范围),在不同的混用位置,失败的"可诊断程度"完全不同——"权威表述"(环境配对了)不能被当作在所有位置都同样安全的全称保证。
- **深挖追问·续(工程落地,收尾):** "那你会怎么把这条容易被忽视的边界,变成一个能被自动检查出来的东西,而不是靠肉眼记住'别裸 `import keras`'这条规则?"—— 期望呼应 13 类第 8 节"CI 阶段检查 `type(model).__module__`"的建议,但候选人应该主动扩展这个建议的覆盖范围:不能只查一条路径就假设整个环境都对,需要同时检查 `tf.keras` 和裸 `keras` 两条路径(比如一条简单的静态检查规则:代码里只要出现不通过 `tf.keras` 的裸 `import keras`/`from keras import` 就报警),从源头上让"混用"这件事根本没有发生的机会,而不是被动等报错(不管报错是清楚还是隐晦)才发现。

**可运行例子(1/2):裸 `import keras` 的真实解析结果,以及混用 Layer 的真实后果:**

```python
import os
import numpy as np
import tensorflow as tf

assert os.environ.get("TF_USE_LEGACY_KERAS") == "1"   # 本进程继承自 ~/tf-venv/bin/activate

m_tfkeras = tf.keras.Sequential([tf.keras.layers.Dense(4)])
print("tf.keras.Sequential(...) module:", type(m_tfkeras).__module__)
assert type(m_tfkeras).__module__.startswith("tf_keras")

import keras   # bare import -- NOT routed through the tf.keras redirection layer at all
print("\nbare `import keras` -- keras.__version__:", keras.__version__)
m_bare = keras.Sequential([keras.layers.Dense(4)])
print("bare keras.Sequential(...) module:", type(m_bare).__module__)
assert type(m_bare).__module__.startswith("keras.")          # Keras 3, NOT tf_keras
assert not type(m_bare).__module__.startswith("tf_keras")

same_class = tf.keras.layers.Dense is keras.layers.Dense
print("\nAre tf.keras.layers.Dense and keras.layers.Dense the SAME class object?", same_class)
assert same_class is False   # two genuinely different classes, not two names for the same thing

# The interesting question: mix the two sources of "Dense" in the same model
x = tf.constant(np.random.RandomState(0).randn(2, 4).astype("float32"))
try:
    mixed_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(8, name="from_tf_keras"),   # Keras 2 (tf_keras) Dense
        keras.layers.Dense(3, name="from_bare_keras"),     # Keras 3 (bare keras) Dense
    ])
    mixed_model(x)
    raise SystemExit("UNEXPECTED: mixing tf_keras and bare-keras layers did not fail")
except TypeError as e:
    print("\nMixing tf.keras.layers.Dense with bare keras.layers.Dense in one Sequential:")
    print(f"  {type(e).__name__}: {str(e)[:200]}")
    assert "must be an instance of class Layer" in str(e)
    print("OK: fails loudly and specifically -- easy to diagnose.")
```

本机实测:`keras.__version__` 显示 `3.15.0`(不是 tf_keras 对应的 2.x 语义),`tf.keras.Sequential(...)` 的模块路径落在 `tf_keras.src.engine.sequential`,而裸 `keras.Sequential(...)` 落在 `keras.src.models.sequential`——两者是**不同的类**(`tf.keras.layers.Dense is keras.layers.Dense` 为 `False`),完全独立于环境变量是否设置。混用两者的 `Dense` 到同一个 `Sequential` 里,立刻触发 `TypeError: The added layer must be an instance of class Layer. Received: layer=<Dense name=from_bare_keras, built=False> of type <class 'keras.src.layers.core.dense.Dense'>`——报错清楚地点出了类型不对。

**可运行例子(2/2):同一个根因,换成 Callback,失败方式完全不同、诊断难度陡增:**

```python
import numpy as np
import tensorflow as tf
import keras  # bare import -- resolves to Keras 3 regardless of TF_USE_LEGACY_KERAS (verified in example 1)

same_class = tf.keras.callbacks.Callback is keras.callbacks.Callback
print("tf.keras.callbacks.Callback is keras.callbacks.Callback:", same_class)
assert same_class is False


class BareKerasCallback(keras.callbacks.Callback):
    """A callback written against the bare (Keras 3) callbacks.Callback base class -- e.g. copied
    from a tutorial, or coming from a third-party library that does `import keras` internally."""
    def __init__(self):
        super().__init__()
        self.on_epoch_end_calls = 0

    def on_epoch_end(self, epoch, logs=None):
        self.on_epoch_end_calls += 1


model = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(2)])
model.compile(optimizer="sgd", loss="mse")
x = np.random.RandomState(0).randn(8, 4).astype("float32")
y = np.random.RandomState(1).randn(8, 2).astype("float32")

cb = BareKerasCallback()
try:
    model.fit(x, y, epochs=2, verbose=0, callbacks=[cb])
    raise SystemExit("UNEXPECTED: a bare-Keras-3 callback did not fail inside tf.keras fit()")
except AttributeError as e:
    print(f"model.fit() with a bare-Keras-3 callback FAILED: {type(e).__name__}: {str(e)[:200]}")
    assert "_implements_train_batch_hooks" in str(e)
    print("\nOK: this error message contains NOTHING that says 'Keras 2 vs 3' or 'wrong base class' --")
    print("compare this to example 1's clear 'must be an instance of class Layer' message. Same root")
    print("cause (bare `import keras`), wildly different diagnostic clarity depending on WHERE it leaks in.")
```

本机实测:`tf.keras.callbacks.Callback is keras.callbacks.Callback` 同样为 `False`;把一个继承自裸 `keras.callbacks.Callback` 的回调传给 `tf.keras` 的 `model.fit()`,触发的是 `AttributeError: 'BareKerasCallback' object has no attribute '_implements_train_batch_hooks'`——这条报错文本里没有任何字眼提示"Keras 2/3 不兼容"或者"基类错了",和例 1 里"must be an instance of class Layer"这种一眼看出问题的报错相比,诊断难度完全不在一个量级。

**常见坑:** 把"`tf.keras` 解析正确"当成"这个环境里所有 Keras 相关的东西都正确"的充分证据,不去检查有没有其它绕过 `tf.keras` 这条路径的裸导入;遇到一次"报错很清楚"的混用场景,就把这个经验泛化成"反正混用了肯定会有清楚的报错提醒我",结果被同一个根因在另一个位置产生的隐晦报错坑到;只做静态的"这次跑起来没报错"检查,而不是主动、显式地用 `__module__` 或者身份检查(`is`)去确认关键类到底来自哪个包。

---

## 小结:5 个案例对应调研发现的哪些轴线

| 案例 | 规模递增轴 | 工程约束递增轴(并发/分布式) | 方案批判迭代轴 | 决策依据追问轴 | 真实性验证轴 | 全新题型(TF独有机制/日志诊断) |
|---|---|---|---|---|---|---|
| 1. retracing 性能陷阱诊断链 | | | ✅(方案被指出没解决问题) | | ✅(方案有效性需现场验证) | ✅ 核心(TF独有tracing心智负担) |
| 2. GPU 显存贪婪分配决策 | | ✅(多进程共享一张卡) | | ✅(为什么要统一在环境层配置) | ✅ 核心(双进程实测颠覆预期) | |
| 3. 训练规模递增分布式决策 | ✅ 核心 | ✅(单卡→虚拟多卡→引用多机) | | ✅ 核心(该不该无脑上多卡) | | |
| 4. 部署格式选型方案批判 | | | ✅ 核心 | ✅(两份格式的取舍依据) | | |
| 5. Keras 2/3 真实性验证边界 | | | | | ✅ 核心 | |

这 5 个案例不是要覆盖 100 个知识点里的每一个——它们演示的是**方法论本身**:拿到任何一个已经掌握的知识点,都可以自己追问"面试官给的这个'标准修复方案',我有没有现场验证过它真的解决了问题,还是只是听起来应该有效""这个决策依据除了'官方推荐'之外,我有没有办法用一次真实实验把利弊量化出来""如果我的实验结果和预期不一致,我是老老实实报告这个反直觉的发现、并且想办法定界它,还是下意识地怀疑自己代码写错了、或者反过来不假思索地把这一次的意外当成放之四海而皆准的新结论"。真正的二面深度,是能不能对着一个自己没准备过的知识点,现场把这几条轴线走一遍——包括诚实地说出"这一步我验证到这里为止,更深的机制细节超出了我这次能确认的范围"这种边界声明本身,也是候选人可信度的一部分。
