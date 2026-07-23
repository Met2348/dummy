# 02 · Autograd 核心机制深挖(Autograd Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)
> **这是全系列权重最高的一批。** PyTorch 面试只要聊到"底层机制",十有八九会落在 autograd 上——而且往往不是"`.backward()`怎么用"这种问法,而是"你能不能现场证明给我看,计算图是真实存在的数据结构,不是一个比喻"。本文要做的,就是把这个"黑箱"打开,让你能亲手遍历一张计算图、亲手验证链式法则、亲手触发并读懂每一类经典报错。

**本文和 01 篇的关系:** [01-tensor-memory-model.md](01-tensor-memory-model.md) 建立的是"地基"——storage/stride/版本计数器这些内存层面的事实。本篇要在这个地基上回答一个更本质的问题:PyTorch 怎么在你写的、看起来就是普通 Python 代码的 `forward` 函数背后,"织"出一张可以反向遍历、自动应用链式法则的计算图?这张图到底是不是一个真实的、可以用代码遍历的数据结构?`backward()`执行的每一步到底在算什么?01 篇第 7 节讲的"in-place + 版本计数器"、第 8 节讲的 `.detach()`,本篇会反复回来引用——那些内存层面的事实,恰恰是本篇要讲的 autograd 机制得以正确工作的前提。

本文所有代码例子已在仓库 `.venv`(torch 2.11.0+cu128,CUDA 可用)下实际跑通验证。所有报错信息、警告文案都是现场触发后原样抄录(用 `assert 子串 in str(e)` 的方式核验,不是转述文档或凭经验断言),所有数值结果(尤其是高阶导数、hook 修改梯度后的值)都经过手算和 `.backward()` 结果交叉验证。

**本篇统一结构(和 01 篇一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计** —— 本系列的核心加深点,不停在"怎么用"
4. AI 研究场景
5. 可运行例子(能内省的地方现场打印 `grad_fn`/`next_functions`/`is_leaf`/`.grad` 等,不要求你相信文字描述)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. 计算图的动态构建原理(define-by-run)—— 和 TensorFlow 1.x 静态图的本质区别

**是什么:** 每次执行一行会产生新 tensor 的运算代码(不管这行代码在不在循环里、在不在 `if` 分支里),只要有输入 `requires_grad=True`,PyTorch 就会在这一刻"现场"于 C++ 层实例化一个新的反向计算节点,并把它接到计算图里。这张图只在"这一次调用"期间存在,不是提前定义好、可以反复复用的静态结构。

**一句话:** PyTorch 是 **define-by-run**(动态图):图是"跑一次 forward 现造一次";TensorFlow 1.x 是 **define-and-run**(静态图):先把整张图的结构定义完(此时没有任何真实数值),再开一个 `Session` 反复灌数据去跑。

**底层机制/为什么这样设计:**

TF1.x 的心智模型是"先画图纸,再按图纸生产":用 `tf.placeholder`(占位符)、`tf.Variable` 搭出一整套符号化的计算图对象,这一步 Python 代码只是在**描述**图,不产生任何真实数值;图搭好之后,通过 `session.run(fetches, feed_dict={...})` 才真正把数据"喂"进去执行。图的拓扑结构在第一次 `session.run` 之前就已经完全固定死了——不管你 run 多少次、喂什么数据,图的形状都不会变。这带来一个直接后果:你没法用原生 Python 的 `if`/`for` 依据 tensor 的**运行时值**去决定"这次要不要多算一层",因为写 `if` 的时候图还没跑,根本不知道值是多少;必须用 `tf.cond`/`tf.while_loop` 这类专门的"图内控制流"算子把分支逻辑也编码进图里,写法别扭,调试时也没法直接 `print` 中间张量(打印出来的只是一个"尚无值"的符号 op 描述)。

PyTorch 反过来:每个 tensor 操作在 Python 这一行代码被**实际执行**的瞬间,同时做两件事——1)eager 地算出前向数值(所以你随时可以 `print` 中间结果);2)如果输入 `requires_grad=True`,同步在 C++ 层生成一个新的 `Node`(即 `grad_fn`,下一节详讲),通过 `next_functions`(第 3 节)接到输入的 `grad_fn` 上。图的构建和前向计算是同一个动作,不分先后。这意味着"能用原生 Python 的 `if`/`for`/递归写模型"不是一个孤立的语法糖特性,而是"图在运行时才被现场构建"这个设计的**直接推论**——因为图本来就是跟着代码实际走过的路径长出来的,你的 `if` 走了哪条分支,图就长成哪个形状。

这个设计不是没有代价:动态图放弃了"预先看到完整图结构、做全局图优化(算子融合、内存复用规划)"的机会,每次调用都要重新构建、有真实的 Python/C++ 调度开销。这正是 `torch.compile`(PyTorch 2.0 引入,后面章节会展开)存在的意义——想让你继续写动态图风格的 Python 代码,同时通过 TorchDynamo 在背后把没有数据依赖分支的部分"捕获"成静态子图去编译优化,试图两头都要。

**AI 研究场景:** RL 里 agent 根据 environment 运行时反馈决定要不要多展开一步 rollout;NLP 变长序列不需要 pad 到固定长度就能直接写 for 循环的 RNN/树形结构网络;MoE(Mixture-of-Experts)路由根据每个 token 的 gating 结果动态选择激活哪几个专家子网络,这条路径每个 batch、甚至每个 token 都可能不同,静态图几乎没法自然表达这种"数据依赖的结构";日常调试一个新模型结构时,直接在 `forward` 中间打断点或者 `print` 张量数值——这是 TF1.x 时代做不到的(必须整张图定义完才能跑,断点处根本没有值)。

**可运行例子:**
```python
import torch

# --- 同一个函数,靠python原生if控制流走不同分支 -> 产生完全不同类型的grad_fn ---
def f(x, flag):
    if flag:
        y = x * 2
    else:
        y = x ** 2
    return y

x1 = torch.tensor(3.0, requires_grad=True)
y1 = f(x1, True)
print("branch True  -> grad_fn:", type(y1.grad_fn).__name__)   # MulBackward0

x2 = torch.tensor(3.0, requires_grad=True)
y2 = f(x2, False)
print("branch False -> grad_fn:", type(y2.grad_fn).__name__)   # PowBackward0

assert type(y1.grad_fn).__name__ == "MulBackward0"
assert type(y2.grad_fn).__name__ == "PowBackward0"
assert type(y1.grad_fn).__name__ != type(y2.grad_fn).__name__   # 同一个函数,图结构却不同

# --- 循环次数不同(运行时才决定)-> 图的"深度"跟着变 ---
def build(loop_n):
    x = torch.tensor(2.0, requires_grad=True)
    h = x
    for _ in range(loop_n):
        h = h * 2
    return h

for loop_n in (1, 3, 5):
    h = build(loop_n)
    depth = 0
    node = h.grad_fn
    while node is not None and type(node).__name__ != "AccumulateGrad":
        depth += 1
        nxt = next((n for n, _ in node.next_functions if n is not None), None)
        node = nxt
    print(f"loop_n={loop_n} -> MulBackward0 链条深度={depth}")
    assert depth == loop_n   # 图的深度精确等于本次调用的循环次数,不是固定结构
```

**面试怎么问 + 追问链:**
- **Q:** "PyTorch 和 TensorFlow 1.x 最核心的设计差异是什么?为什么 PyTorch 在研究界迅速反超?"—— 期望答出 define-by-run vs define-and-run,并能举出"能用原生 Python 控制流写模型、能随时打断点调试"这类具体好处,而不是只会说"PyTorch 更好用"。
- **追问 1:** "动态图每次都要重新构建,不担心这个开销吗?"—— 期望提到"eager 执行确实有额外的 Python/C++ 调度开销,这也是 `torch.compile` 存在的意义——想要动态语法的同时,拿到接近静态图的执行效率"。
- **深挖追问(区分度很高):** "你怎么向我证明 PyTorch 的计算图是'现场生成'的,而不是'碰巧看起来和 python 代码结构一样,其实背后还是某种固定结构'?"—— 期望候选人能想到"用同一个函数、不同的运行时输入,现场对比 grad_fn 类型或者 next_functions 链条长度是否不同",而不是停留在背概念。

**常见坑:** 把"能用 if/for 写模型"当成动态图的**定义本身**——这只是表现之一,真正的区别是"图在什么时候被构建"(运行时 vs 编译前);另外容易误以为 TF1.x 完全没法调试,实际上配合一些专门工具也能做到,只是心智负担和 PyTorch 的所见即所得完全不是一个量级。

---

## 2. `grad_fn` —— 每个非叶子 tensor 背后挂的 Function 节点

**是什么:**
```python
y.grad_fn   # 一个反向计算节点(Node/Function)对象的引用,记录"y是通过什么运算、
            # 从哪些输入算出来的"。只有 requires_grad=True 且不是用户直接创建的
            # tensor(non-leaf,第5节详讲)才有 grad_fn;leaf tensor 的 grad_fn 永远是 None。
```

**一句话:** `grad_fn` 是反向传播时"这一步该调用谁来算梯度"的入口指针——每完成一次可微分运算,PyTorch 就在结果 tensor 背后挂一个对应类型的反向节点,节点类型和前向运算一一对应(`x*3` 挂 `MulBackward0`,`x+1` 挂 `AddBackward0`,`x**2` 挂 `PowBackward0`……)。

**底层机制/为什么这样设计:** PyTorch 用的是 "tape-based"(磁带式)autograd:每次调用一个可微分 op,前向计算和反向图的构建是**同一时刻**发生的两件事——前向算出数值的同时,C++ 层实例化一个对应的 `Node` 子类,这个 Node 内部保存了反向传播这一步需要的信息(比如乘法的反向 `d(a*b)/da = b` 需要知道"另一个乘数是多少",这类值通过 `save_for_backward` 机制保存,并受 01 篇第 7 节讲的版本计数器保护),以及指向"制造出当前输入"的下一层节点的边(`next_functions`,下一节详细讲)。这个 Node 对象通过 `grad_fn` 属性暴露给 Python 层——所以 `y.grad_fn` 本身就是"y 是怎么来的"这段历史的真实记录,不是转述或比喻。

**AI 研究场景:** 调试一个由多个子 loss 加权拼起来的复杂目标函数时,如果某处代码不小心引入了一次不该有的 `detach()` 或 `no_grad`,某条路径的梯度会在那里"悄悄"断掉而不报错;直接查看最终 loss 或某个中间量的 `grad_fn` 类型是不是符合预期的最外层运算(比如你以为最后一步是加权求和,结果发现 `grad_fn` 根本不是 `AddBackward0`),是比"逐个打印中间张量的值再猜哪里错了"精确得多的排查手段。

**可运行例子:**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
assert x.grad_fn is None                      # leaf tensor,没有grad_fn
print("x.grad_fn:", x.grad_fn)

y = x * 3
print("y.grad_fn:", y.grad_fn)                  # 形如 <MulBackward0 object at 0x...>(地址每次运行都不同,正常)
print("type(y.grad_fn).__name__:", type(y.grad_fn).__name__)
assert type(y.grad_fn).__name__ == "MulBackward0"

z = y + 1
print("z.grad_fn:", z.grad_fn)
assert type(z.grad_fn).__name__ == "AddBackward0"

w = z ** 2
print("w.grad_fn:", w.grad_fn)
assert type(w.grad_fn).__name__ == "PowBackward0"

# requires_grad=False的输入,不会产生grad_fn —— 没必要为它构建反向路径
a = torch.tensor(2.0, requires_grad=False)
b = a * 3
print("b.grad_fn:", b.grad_fn, " b.requires_grad:", b.requires_grad)
assert b.grad_fn is None
assert b.requires_grad is False
```

**面试怎么问 + 追问链:**
- **Q:** "`y = x * 3` 之后,`y.grad_fn` 是什么?如果 `x.requires_grad=False` 呢?"—— 期望"是一个 `MulBackward0` 节点;如果 x 不需要梯度,y.grad_fn 是 None,因为没必要为它构建反向计算路径"。
- **追问:** "`grad_fn` 类名后面那个 `0` 后缀是什么意思,是不是所有 `grad_fn` 都有这个后缀?"—— 期望答出这类似"该 op 对应第几号输出"的内部实现记号,内置 op 基本都有;但第 9 节会看到,自定义 `torch.autograd.Function` 产生的节点是**没有**这个数字后缀的(比如 `MyReLUBackward`),说明这更像内置 kernel 的实现细节,不是一个必须理解语义的强制规则——能连起来答说明真的动手验证过,不是背的。

**常见坑:** 以为"每个 tensor 都有 `grad_fn`"——leaf tensor 永远是 `None`(不管 `requires_grad` 是不是 `True`);把 `grad_fn` 误解成"这个 tensor 的值是怎么算出来的"的前向函数——它其实是**反向**计算节点,存的是"怎么把梯度往回传",前向的数值早就在 eager 执行时算完并存在 tensor 里了。

---

## 3. `next_functions` —— 计算图真正的"边",一条可以手动遍历的链

**是什么:**
```python
grad_fn.next_functions
# 一个tuple,每个元素是 (下一个Node对象 或 None, 该输入是产生自哪个Node的第几个输出)
# —— 这是计算图真正的"边",从当前节点连向"制造出当前某个输入"的那个节点
```

**一句话:** 如果说 `grad_fn` 是图上的"点",`next_functions` 就是"边"——顺着它一路往前遍历,能走到每一个参与计算的 leaf tensor,这就是 `backward()` "知道该往哪传"的底层依据:不是某种全局记录表,而是每个节点自己知道"我的输入是谁生成的"。

**底层机制/为什么这样设计:** 每个非叶子输入,在 `next_functions` 里对应一个真实的 Node;如果某路输入不需要梯度,对应位置是 `(None, 0)`(第 12 节详细验证)。如果某个输入是 leaf 且 `requires_grad=True`,对应位置指向一个特殊的终端节点—— **`AccumulateGrad`**。它不是"某种梯度值",而是一个**执行动作的节点**:它的 `.variable` 属性指回原始 leaf tensor 对象本身,它的 `next_functions` 是空 tuple `()`(已验证,是图遍历的终点);backward 执行到这里时,会把传入的梯度**累加**写入 `variable.grad`(这正是第 10 节梯度累加机制的直接实现)。值得注意:如果同一个 leaf 在图里被用了多次(比如 `z = x + x`),两条边指向的是**同一个** `AccumulateGrad` 对象(不是两个不同实例)——这保证了多条路径的梯度汇入同一个累加目标,而不是被分散记录、最后再合并。

**AI 研究场景:** 排查"为什么这个 loss 反传不到某个参数"时,一种可靠(虽然略繁琐)的手段是从 `loss.grad_fn` 开始写脚本递归遍历 `next_functions`,检查目标参数对应的 `AccumulateGrad` 节点在不在这棵树里;如果发现某条分支中途变成 `(None, 0)` 或者压根走不到,说明这条路径在某处被 `detach()`/`no_grad`/或一个 `requires_grad=False` 的中间量切断了——这比对着代码肉眼找往往更可靠。

**可运行例子:**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(4.0, requires_grad=True)

z = (x + y) * w   # z = (2+3)*4 = 20

print("z.grad_fn:", type(z.grad_fn).__name__)
print("z.grad_fn.next_functions:", z.grad_fn.next_functions)

add_node = z.grad_fn.next_functions[0][0]
w_acc_node = z.grad_fn.next_functions[1][0]
assert type(add_node).__name__ == "AddBackward0"
assert type(w_acc_node).__name__ == "AccumulateGrad"
assert w_acc_node.variable is w        # AccumulateGrad.variable直接指回原始leaf tensor对象

# 继续往下遍历: AddBackward0的next_functions指向x和y各自的AccumulateGrad
x_acc = add_node.next_functions[0][0]
y_acc = add_node.next_functions[1][0]
assert x_acc.variable is x
assert y_acc.variable is y
assert x_acc.next_functions == ()       # AccumulateGrad是终点,不再往下传

# 深挖: 同一个leaf被用两次,next_functions两条边指向同一个AccumulateGrad对象吗?
x2 = torch.tensor(3.0, requires_grad=True)
z2 = x2 + x2
fn0 = z2.grad_fn.next_functions[0][0]
fn1 = z2.grad_fn.next_functions[1][0]
print("两条边是否指向同一个AccumulateGrad对象:", fn0 is fn1)
assert fn0 is fn1                        # 是同一个对象,不是两个独立实例
assert fn0.variable is x2

z2.backward()
assert x2.grad.item() == 2.0             # 两条路径的梯度(各1.0)汇入同一个AccumulateGrad累加成2.0
```

**用一张图看懂这个例子——`z = (x + y) * w` 的计算图长什么样:** 前两小节一直在用代码打印 `grad_fn`/`next_functions`,但打印出来的元组终究要在脑子里拼成一张图才有用。这里把上面"可运行例子"里 `z = (x + y) * w` 那段代码(已经现场验证过)画出来——`grad_fn` 是图上的"点",`next_functions` 是"边",箭头方向就是 `backward()` 真正遍历、传梯度的方向(从 `z` 往 leaf 走,和前向计算方向相反):

```text
z.grad_fn = MulBackward0                        ← 对应 (x+y)*w 这一步乘法
  │
  ├─ next_functions[0] ──► AddBackward0           ← 对应 x+y 这一步加法
  │                          │
  │                          ├─ next_functions[0] ──► AccumulateGrad   .variable is x,  next_functions == ()
  │                          │
  │                          └─ next_functions[1] ──► AccumulateGrad   .variable is y,  next_functions == ()
  │
  └─ next_functions[1] ──► AccumulateGrad         .variable is w,  next_functions == ()
```

对照着读:`z.grad_fn` 就是 `MulBackward0`——这是"点";它的 `next_functions` 有两条"边",一条通向 `AddBackward0`(`x+y` 这一步),另一条直接通向 `w` 的 `AccumulateGrad`(因为 `w` 是叶子,不需要再经过任何中间运算节点);再往下一层,`AddBackward0` 的 `next_functions` 两条边分别通向 `x`、`y` 各自的 `AccumulateGrad`。三个 `AccumulateGrad` 都是图的终点(`next_functions == ()`,已在上面代码里验证过)。`backward()` 从最上面的 `MulBackward0` 出发,顺着这些边一路往下走,每走到一个 `AccumulateGrad` 就把梯度累加写进对应 leaf tensor 的 `.grad`——这就是"计算图是一个真实的、可以用代码遍历的数据结构"这句话具体长什么样,不是比喻,上面代码里的 `add_node`/`w_acc_node`/`x_acc`/`y_acc` 这几个变量,对应的正是这张图里的四个节点。

**面试怎么问 + 追问链:**
- **Q:** "`backward()` 内部到底是怎么知道'该往哪传梯度'的?"—— 期望能提到 `next_functions` 这个链式结构,而不是含糊说"框架自动处理"。
- **追问 1:** "`AccumulateGrad` 是什么?为什么 leaf 节点的 `next_functions` 要指向一个'节点',而不是直接是 `None` 表示到头了?"—— 期望理解 `AccumulateGrad` 是一个"执行'写入 `.grad`'这个副作用"的真实节点,不是空指针;这样设计让"图的终点"和"图断开的地方(不需要梯度的分支)"统一用 `next_functions` 表达,只是前者指向一个真实节点,后者是 `None`。
- **深挖追问(区分度很高):** "如果一个 leaf tensor 在计算里被用了两次(比如 `z = x + x`),`next_functions` 里对应的两条边,是不是指向同一个 `AccumulateGrad` 对象?"—— 期望候选人能想到用 `is` 现场比较验证,而不是猜;能进一步说出"这保证了多路径梯度汇入同一个累加目标"就是加分项。

**常见坑:** 以为 `next_functions` 只有一层,看一次就以为看完了整个图——必须递归/循环遍历才能看到完整结构;把 `AccumulateGrad` 误认为是"梯度值本身",它是"负责把梯度累加写入 `.grad`"的**节点**,不是数值。

---

## 4. `backward()` 的实际执行过程 —— 反向拓扑遍历 + 链式法则(VJP)

**是什么:** 从最终 output 出发,沿着计算图按反向拓扑序(每个节点必须等它下游所有汇入的梯度都到齐才能继续往上传),依次调用每个 Node 的反向计算,把梯度值链式传给它的 `next_functions`,直到所有参与计算的 leaf 的 `AccumulateGrad` 被触发为止。

**一句话:** `backward()` = 反向拓扑遍历 + 每一步做一次 **vector-Jacobian product(VJP)**——不是符号推导出一个梯度表达式,也不是老老实实算出完整的雅可比矩阵,而是拿"上游传来的梯度向量"直接和当前这一步的雅可比矩阵做乘法,一步到位得到"往下游传的梯度向量"。

**底层机制/为什么这样设计:**

**先从零定义:雅可比矩阵(Jacobian matrix)是什么——这是理解 VJP 的前提。** 如果一个函数 `y=f(x)` 的输入 `x` 是长度 `n` 的向量、输出 `y` 是长度 `m` 的向量(每个 `y_i` 都可能同时依赖 `x` 的所有分量),雅可比矩阵 `J` 就是把"每一个输出分量对每一个输入分量的偏导数"整整齐齐排成一个 `(m,n)` 的矩阵:

```text
J[i, j] = ∂y_i / ∂x_j
```

`J` 的第 `i` 行是"第 `i` 个输出分量对整个输入向量的梯度";第 `j` 列是"第 `j` 个输入分量对整个输出向量各分量的影响"。这是一个纯数学定义,和 PyTorch/autograd 本身没有任何关系——`backward()` 到底怎么用它、要不要把它老老实实算出来,是接下来要讲的事。

**具体到能验证的数值例子(和 [tensorflow-deep-dive/02-gradienttape-internals.md 第 7 节](../tensorflow-deep-dive/02-gradienttape-internals.md) 用的是同一个例子,方便两个框架对照着看):** 取 `y=x**2`(逐元素平方),`x=[1,2,3]`。因为是逐元素运算,`y_i` 只依赖 `x_i` 自己,和其他分量 `x_j`(`j≠i`)完全无关,所以 `∂y_i/∂x_j = 2*x_i`(当 `i==j`)、`=0`(当 `i≠j`)——写成矩阵就是一个对角阵 `J = diag([2*x_0, 2*x_1, 2*x_2]) = diag([2,4,6])`,对角线之外全是 0,因为 `y_i` 和 `x_j`(`i≠j`)之间根本没有依赖关系。PyTorch 没有把 `jacobian` 做成 tensor 自带的方法(`.backward()`/`torch.autograd.grad()` 默认只做 VJP,不会老实把整个矩阵铺开——这正是下面要讲的重点),但独立函数 `torch.autograd.functional.jacobian` 能直接把这整个矩阵算出来,现场和手推的 `diag([2,4,6])` 对比:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])

def f(x):
    return x ** 2

jac = torch.autograd.functional.jacobian(f, x)
print("jacobian:\n", jac)
assert torch.equal(jac, torch.diag(torch.tensor([2.0, 4.0, 6.0])))
```

实测 `jac` 恰好等于 `[[2,0,0],[0,4,0],[0,0,6]]`,和手推结果完全一致——这和 TensorFlow 那边 `tape.jacobian(y,x)` 对同一个 `y=x**2` 例子算出来的 `diag([2,4,6])` 是同一个数学对象,只是两个框架挂 API 的地方不同:TF 的 `tape.jacobian()` 是 `GradientTape` 自带的方法,PyTorch 要单独调用 `torch.autograd.functional.jacobian`。

有了这个具体例子做参照,再看 `backward()` 和 VJP 的关系就清楚多了。关键认知是 `backward()` 每一步算的不是"雅可比矩阵"本身,而是 `grad_output`(维度和这一步的**输出**一样)与雅可比矩阵的乘积,直接得到 `grad_input`(维度和这一步的**输入**一样)。这就是为什么调用 `.backward()` 时,如果最终 output 不是标量,必须显式传入一个和 output 同形状的 `gradient` 参数(也就是最初的 `grad_output`,VJP 里的那个"上游向量")——因为如果 output 是高维的,"对它求梯度"这件事本身没有唯一定义(梯度是对标量而言的),必须由你告诉 PyTorch "从这个方向往回传的量是多少"。这个设计避免了对高维 tensor 显式构造和存储完整雅可比矩阵(对于一个 `(1000,)` 输入 `(1000,)` 输出的 op,完整雅可比矩阵是 `1000x1000`,而 VJP 全程只需要 `O(n)` 的向量)。反向拓扑遍历还必须处理"多条路径汇入同一节点"的情况(第 3 节 `z=x+x` 的例子):必须等所有上游分支的梯度都传到、加总之后,才能继续往更上游传——这也是为什么"同一个 tensor 在图里被多次使用"和"多次调用 `.backward()`"最终都归结到同一套累加机制(第 10 节)。

**可运行例子:**
```python
import torch

# --- 手写链式法则(不调用.backward()),和真正的backward()结果对比 ---
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2      # node 1: y = x^2
z = y * 3       # node 2(最终节点): z = 3y

dz_dy = 3.0                    # z = 3y  =>  dz/dy = 3(常数)
dy_dx = 2 * x.item()            # y = x^2 =>  dy/dx = 2x
manual_dz_dx = dz_dy * dy_dx    # 链式法则: dz/dx = dz/dy * dy/dx = 3 * 4 = 12

z.backward()
print("手算 dz/dx:", manual_dz_dx, " autograd 算出的 x.grad:", x.grad.item())
assert torch.allclose(x.grad, torch.tensor(manual_dz_dx))   # 完全一致

# --- 多路径汇入: 同一个leaf被用两次,梯度是两条路径的和(反向拓扑遍历的直接体现) ---
x2 = torch.tensor(3.0, requires_grad=True)
z2 = x2 + x2
z2.backward()
assert x2.grad.item() == 2.0    # dz2/dx2 = 1(第一条路径) + 1(第二条路径) = 2

# --- vector-Jacobian product: 非标量输出必须显式传入grad_output(上游向量) ---
v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
out = v * 2
try:
    out.backward()          # 非标量,没告诉PyTorch"往回传的量是多少"
    assert False
except RuntimeError as e:
    print("非标量backward报错:", str(e))
    assert "grad can be implicitly created only for scalar outputs" in str(e)

grad_outputs = torch.tensor([1.0, 1.0, 1.0])   # 上游grad_output(VJP里的"v")
out.backward(gradient=grad_outputs)
assert v.grad.tolist() == [2.0, 2.0, 2.0]        # d(out)/d(v)=2(逐元素),乘以全1的grad_output

# 换一个不是全1的grad_output,证明VJP确实是"直接乘出来的向量",不是雅可比矩阵本身
v2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
out2 = v2 * 2
out2.backward(gradient=torch.tensor([1.0, 0.0, 1.0]))
assert v2.grad.tolist() == [2.0, 0.0, 2.0]       # 第二个分量的上游梯度是0,直接体现在结果里
```

**面试怎么问 + 追问链:**
- **Q:** "`backward()` 内部做了什么?"—— 期望答出"反向拓扑遍历 + 链式法则逐层传",最好能说出"每步实际算的是 VJP,不是显式的雅可比矩阵"。
- **追问 1:** "如果计算图不是一条链,而是有分叉又汇合(比如 `y` 被用了两次),`backward` 怎么处理?"—— 期望说出"必须等一个节点的所有上游梯度都到齐才能继续往上传,汇合处的梯度是相加"。
- **深挖追问(区分度很高):** "为什么对一个非标量 tensor 调用 `.backward()` 必须传入 `gradient` 参数?这个参数在数学上对应什么?"—— 期望理解"梯度本来就是对标量定义的,非标量 output 要往回传必须先指定'从哪个方向、以什么权重往回传',这个参数就是 VJP 公式里的上游向量,不是一个可有可无的形状占位符"。

**常见坑:** 以为 `backward()` 会显式算出并存储雅可比矩阵——实际上全程只有向量,这也是 autograd 能高效处理高维 tensor(比如动辄百万参数的模型)的关键;以为反向执行顺序严格是"时间戳倒序"——大方向确实和前向相反,但遇到分叉汇合时,真正决定顺序的是"依赖是否满足"(拓扑序),不是单纯按时间倒放录像。

---

## 5. leaf tensor vs non-leaf tensor —— `is_leaf`

**是什么:** `x.is_leaf` 判断一个 tensor 是不是计算图的"起点"。leaf tensor 包括:1)用户直接创建、`requires_grad=True` 的 tensor(模型参数、手动设置 `requires_grad=True` 的输入);2)**所有 `requires_grad=False` 的 tensor,不管它是不是由别的 tensor 运算出来的**。non-leaf 是任何由其他 tensor 通过可微分运算计算出来的、`requires_grad=True` 的 tensor。

**一句话:** leaf 是你(或者"当前 `requires_grad` 上下文")亲手标记的图的"起点";non-leaf 是计算过程中产生的"中间站"——区分标准的本质是"有没有 `grad_fn`",不是"是不是我自己敲代码创建的"。

**底层机制/为什么这样设计:** PyTorch 默认只给 leaf tensor 保留 `.grad`,原因是显存优化:多数场景下你只需要模型参数(leaf)的梯度去做更新,中间激活值的梯度只是链式法则"传话过程"里的临时产物——backward 结束后,这些中间梯度被计算出来、传给下一层之后就可以立刻丢弃,而不是额外占用一份内存长期保留。想象一个 100 层网络,如果每一层的中间激活都要保留梯度,显存开销会显著增加。leaf 节点的梯度写入靠的正是第 3 节讲的 `AccumulateGrad` 机制;非 leaf 节点默认没有对应的持久化存储位置。

**可运行例子:**
```python
import torch
import warnings

x = torch.tensor(2.0, requires_grad=True)
y = x * 3   # non-leaf

print("x.is_leaf:", x.is_leaf, " y.is_leaf:", y.is_leaf)
assert x.is_leaf is True
assert y.is_leaf is False

y.backward()
assert x.grad is not None
print("x.grad:", x.grad.item())

# 访问非leaf的.grad,得到None并触发真实的UserWarning(现场抓取,不是转述)
y2 = x * 3
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    g = y2.grad
    assert g is None
    assert len(w) == 1
    msg = str(w[0].message)
    print("warning:", msg[:120], "...")
    assert "is not a leaf Tensor" in msg
    assert "retain_grad()" in msg          # 警告文案里直接提示了下一节的解法

# 反直觉细节: requires_grad=False的tensor,即使是"运算出来的",也算leaf!
a = torch.tensor(2.0, requires_grad=False)
b = a * 3
print("b.requires_grad:", b.requires_grad, " b.is_leaf:", b.is_leaf)
assert b.requires_grad is False
assert b.is_leaf is True    # 反直觉点: 判断标准是requires_grad=False,不是"是不是计算出来的"

# 深挖: 在no_grad块里算出来的tensor,即使是别的requires_grad=True的tensor算出来的,也是leaf
x3 = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    y3 = x3 * 3        # y3是x3算出来的,但因为在no_grad里,y3.requires_grad变成False
assert y3.requires_grad is False
assert y3.is_leaf is True     # 是leaf! 尽管y3是"计算出来的"

y3.requires_grad_(True)        # 把y3变成一个"需要梯度的leaf"(常见于reparameterization场景)
assert y3.is_leaf is True       # 依然是leaf,因为它从始至终都没有grad_fn
z3 = y3 ** 2
z3.backward()
assert y3.grad.item() == 12.0   # 完全像一个真正的叶子参数一样正常工作: dz3/dy3=2*6=12
```

**面试怎么问 + 追问链:**
- **Q:** "什么是 leaf tensor?模型参数为什么都是 leaf?"—— 期望答出"用户直接创建、requires_grad=True 的 tensor,或者所有 requires_grad=False 的 tensor"。
- **追问 1:** "一个 `requires_grad=False` 的 tensor,如果它是由另一个 `requires_grad=True` 的 tensor 计算出来的(比如在 `no_grad` 块里算的),它是 leaf 吗?"—— 期望能推理出"是,因为判断标准是有没有 `grad_fn`,不是这个 tensor 的'血统'"。
- **深挖追问(区分度很高):** "能不能构造一个'看起来是计算出来的,但其实是 leaf'的 tensor,并且让它像模型参数一样正常参与训练?"—— 这正是上面例子里 `no_grad` 块 + `requires_grad_(True)` 的组合,是很多 reparameterization/量化感知训练代码里真实会用到的模式,能现场写出来说明理解是扎实的。

**常见坑:** 直接访问一个非 leaf tensor 的 `.grad` 期望拿到值——默认拿到的是 `None`(还会有警告),需要 `retain_grad()`(下一节);想当然地认为"只有我自己写 `torch.tensor(...)` 创建的才是 leaf"——`requires_grad=False` 这个条件本身就足以让任何 tensor(不管来源)被视为 leaf。

---

## 6. `retain_grad()` —— 让非叶子节点也能在 backward 后查看 `.grad`

**是什么:** 对一个 non-leaf tensor 调用 `.retain_grad()`,声明"backward 之后我要看这个中间量的梯度"——之后它的 `.grad` 属性会被正常填充,而不是默认丢弃。

**一句话:** `retain_grad()` 是"我知道你不是 leaf,但这次我就是要看你的梯度"的显式声明,本质上是往这个 tensor 对应的反向路径上挂了一个"顺手存一份"的钩子(和第 13 节 `register_hook` 是同一套底层机制的两种表现)。

**底层机制/为什么这样设计:** 调用 `retain_grad()` 后,PyTorch 会在这个 tensor 的反向路径上注册一个类似 `AccumulateGrad` 行为的回调——backward 流经这里时,顺手把当前这一步的梯度值写入 `tensor.grad`,但这个 tensor 本身依然不是图的终点,梯度还会继续往上游传,不影响主计算路径。这是研究/调试用的"侧写"能力,不改变任何计算结果,只是多留一份"备份"。**它只对"接下来要执行的 backward"生效**——调用的时机必须在对应的 `backward()` 之前,但和这个 tensor 是什么时候被创建的没有关系,哪怕是在两次 `backward()` 之间补调用也完全来得及。

**可运行例子:**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3      # non-leaf, y = 6
y.retain_grad()
z = y ** 2     # z = 36

z.backward()
assert y.grad is not None
print("y.grad:", y.grad.item())
assert y.grad.item() == 12.0     # dz/dy = 2y = 2*6 = 12
print("x.grad:", x.grad.item())
assert x.grad.item() == 36.0     # z=9x^2, dz/dx=18x=18*2=36

# --- retain_grad()只要在"那次要生效的backward"之前调用即可,和tensor创建的时机无关 ---
x2 = torch.tensor(2.0, requires_grad=True)
y2 = x2 * 3
z2 = y2 ** 2

z2.backward(retain_graph=True)   # 第一次backward: 还没调用retain_grad,y2.grad是None
assert y2.grad is None
assert x2.grad.item() == 36.0

y2.retain_grad()                  # 补调用,为紧接着的第二次backward生效
z2.backward()                     # 对同一张图再backward一次(第一次retain_graph=True保留了图)
print("补调用retain_grad后, y2.grad:", y2.grad.item(), " x2.grad:", x2.grad.item())
assert y2.grad.item() == 12.0     # 这次生效了,y2.grad被填充
assert x2.grad.item() == 72.0     # x2.grad是两次backward的累加: 36+36=72(呼应第10节)
```

**面试怎么问 + 追问链:**
- **Q:** "`retain_grad()` 和 leaf tensor 默认保留梯度,有什么本质区别?"—— 期望答出"leaf 靠 `AccumulateGrad` 终端节点天然保留;non-leaf 默认没有这个持久化位置,`retain_grad()` 是显式地'借用'一份类似的机制"。
- **追问:** "大量调用 `retain_grad()` 对显存有什么影响?"—— 期望答"每个被 retain 的中间量都会额外占用一份和它同形状的显存来存 `.grad`,如果在深层网络的每一层都加,累积开销不小,通常只在调试/研究阶段临时用,不建议留在训练主循环里"。
- **深挖追问:** "`retain_grad()` 的效果和你自己用 `register_hook` 手动把梯度存到一个变量里,是不是本质上是同一件事?"—— 期望理解两者共享同一套"在梯度流经某节点时插入回调"的机制,`retain_grad()` 可以看成一个内置的、专门写入 `.grad` 属性的 hook(第 13 节会现场用 `register_hook` 手动实现出一样的效果来验证这个说法)。

**常见坑:** 以为 `retain_grad()` 必须在 tensor 刚创建时就调用——其实只要赶在"这次要用到它的 backward"发生之前调用即可;忘记 `retain_grad()` 只是"多留一份备份",不会改变任何计算结果或者梯度数值本身。

---

## 7. `retain_graph=True` —— 为什么默认 backward 后计算图会被释放

**是什么:** `.backward(retain_graph=True)` 传入这个参数后,backward 完成时不会释放计算图里保存的中间 buffer,从而允许对同一张图再调用一次 backward。

**一句话:** 默认情况下,`backward()` 之后图会"用后即焚"(省显存);`retain_graph=True` 是"这一次先别烧,我马上还要用"的声明——它控制的是**这一次调用**之后要不要释放,不是一个"从此以后图永久保留"的全局开关。

**底层机制/为什么这样设计:** backward 执行时,每个 Node 在算完梯度、把结果传给下游之后,默认会释放它在 forward 阶段通过 `save_for_backward` 保存的中间 tensor——这是一次性省显存的默认行为,也是为什么大多数图只能 backward 一次。`retain_graph=True` 就是跳过这一步释放动作。需要注意:这个参数**只对"这一次 backward 调用之后"的释放行为生效**——如果你连续三次对同一张图调用 backward,前两次都要传 `retain_graph=True`,第三次(最后一次)可以不传,让它正常释放;如果三次都不传,第二次就会报错。什么场景需要:一个共享的中间结果(比如多任务学习里共享的 encoder 输出)被两个不同的 loss 分别调用 backward(而不是先把两个 loss 加起来再一起 backward 一次)。

**可运行例子:**
```python
import torch

# --- 不设retain_graph: 第二次backward报错(现场抓取真实报错文案) ---
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y ** 2
z.backward()
try:
    z.backward()
    assert False
except RuntimeError as e:
    print("第二次backward报错:", str(e))
    assert "Trying to backward through the graph a second time" in str(e)
    assert "Specify retain_graph=True" in str(e)

# --- retain_graph=True只对"这一次调用之后"生效,不是永久开关 ---
x2 = torch.tensor(2.0, requires_grad=True)
y2 = x2 * 3
z2 = y2 ** 2
z2.backward(retain_graph=True)   # 第1次: 之后不释放
assert x2.grad.item() == 36.0
z2.backward()                     # 第2次: 没传retain_graph,这一次之后就释放了
assert x2.grad.item() == 72.0     # 两次backward梯度累加(第10节)
try:
    z2.backward()                 # 第3次: 图已经被第2次调用释放
    assert False
except RuntimeError as e:
    assert "Trying to backward through the graph a second time" in str(e)

# --- 真实场景: 共享一个前向结果,两个不同的loss各自backward一次 ---
shared = torch.tensor(2.0, requires_grad=True)
feat = shared * 3                     # 假装是共享encoder的输出, feat=6
loss_a = feat ** 2                     # 假装是任务A的loss, = 36
loss_b = feat ** 3                     # 假装是任务B的loss, = 216

loss_a.backward(retain_graph=True)     # loss_a先backward,保留图给loss_b用
assert shared.grad.item() == 36.0       # d(loss_a)/d(shared)=d(feat^2)/d(feat)*d(feat)/d(shared)=2*6*3=36

loss_b.backward()                       # loss_b再backward,复用同一个feat节点
# d(loss_b)/d(shared)=d(feat^3)/d(feat)*d(feat)/d(shared)=3*36*3=324; 累加: 36+324=360
assert shared.grad.item() == 360.0
```

**面试怎么问 + 追问链:**
- **Q:** "什么场景需要 `retain_graph=True`?"—— 期望举出"同一个前向结果被多个独立的 loss 分别 backward"这类真实场景。
- **追问(容易被搞混的两个相似名字):** "`retain_graph=True` 和 `retain_grad()` 有什么区别?"—— 期望清楚区分:`retain_graph` 决定的是"整张图的中间 buffer 要不要保留以便再 backward 一次",作用域是**整张图**、生效范围是**这一次调用**;`retain_grad()` 决定的是"某个特定 non-leaf tensor 的 `.grad` 要不要在 backward 后保留",作用域是**单个 tensor**、一旦调用会一直生效到该 tensor 所在的图被释放为止——这是两个完全不同维度的东西,名字像纯属巧合。
- **深挖追问:** "如果我要连续对同一张图 backward 5 次,`retain_graph=True` 要传几次?"—— 期望能推理出"前 4 次都要传,第 5 次(最后一次)可以不传",而不是笼统地说"传一次就永久有效"。

**常见坑:** 以为 `retain_graph=True` 能解决所有"两次 backward 报错"的问题——它只解决"图被释放"这一个问题,梯度依然会**累加**(第 10 节),如果不配合清零,多次 backward 会让 `.grad` 越加越大,这是另一个独立的坑。

---

## 8. `create_graph=True` 与高阶导数

**是什么:** `torch.autograd.grad(..., create_graph=True)`(或 `.backward(create_graph=True)`)让反向传播计算梯度这个过程本身也被记录进一张新的计算图里,这样对得到的梯度再做一次反向传播,就能算出二阶导数。

**一句话:** 正常 backward 算出的梯度是"数值",本身脱离计算图,不能再往回传;`create_graph=True` 让"算梯度"这件事本身变得可微,得到的梯度 tensor 仍然带着 `requires_grad=True` 和自己的 `grad_fn`,可以继续参与新一轮 backward。

**底层机制/为什么这样设计:** 正常情况下,backward 内部各个 Node 的反向计算(比如 `MulBackward0.apply()` 内部实际做的那次乘法)是在"不记录梯度"的模式下用 C++ 直接跑数值的——这是默认的性能优化,没人会平白无故对着梯度计算再去多维护一份计算图。`create_graph=True` 会让这些反向传播过程中产生的运算**也被 autograd 追踪**(构建一张新图),于是拿到的梯度 tensor `requires_grad=True` 且有自己的 `grad_fn`,可以继续被 `backward()`/`torch.autograd.grad()`。**推荐用 `torch.autograd.grad()` 而不是 `.backward()` 来做这件事**——这不是一条凭经验的建议,而是官方在 `.backward(create_graph=True)` 时会**现场触发一条 UserWarning** 明确指出的(下面例子会验证):用 `.backward()` 做高阶导数会在参数和它的梯度之间形成一个引用环,可能导致内存泄漏;`torch.autograd.grad()` 直接返回梯度值、不写入 `.grad` 属性,天然不会有这个环。

**可运行例子:**
```python
import torch

# y = x^3 => dy/dx = 3x^2, d2y/dx2 = 6x。在x=2处: 一阶导=12,二阶导=12(数值刚好相等,用来交叉验证)

# --- 推荐写法: torch.autograd.grad + create_graph=True ---
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print("一阶导 dy/dx:", grad1.item())
assert grad1.item() == 12.0
assert grad1.requires_grad is True     # grad1本身仍带着计算图,不是纯数值
print("grad1.grad_fn:", grad1.grad_fn)
assert grad1.grad_fn is not None

grad2 = torch.autograd.grad(grad1, x)[0]   # 对"一阶导"这个tensor再求一次导
print("二阶导 d2y/dx2:", grad2.item())
assert grad2.item() == 12.0                 # 6*2=12,和理论值吻合
```

```python
import torch
import warnings

# --- 对比: .backward(create_graph=True) 会现场触发内存泄漏警告(实测,不是猜的) ---
x2 = torch.tensor(2.0, requires_grad=True)
y2 = x2 ** 3
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    y2.backward(create_graph=True)
    assert len(w) == 1
    msg = str(w[0].message)
    print("警告:", msg)
    assert "reference cycle" in msg
    assert "memory leak" in msg
    assert "autograd.grad" in msg     # 警告文案里官方直接建议改用autograd.grad

assert x2.grad.item() == 12.0
assert x2.grad.requires_grad is True   # x2.grad本身也在新图里

# --- 陷阱: 不清空.grad直接对它再backward,一阶和二阶梯度会"混"在一起(累加机制,呼应第10节) ---
first_order_grad = x2.grad
first_order_grad.backward()             # 没有清空x2.grad就再backward
print("不清空直接再backward:", x2.grad.item())
assert x2.grad.item() == 24.0            # 12(残留的一阶导数) + 12(真正的二阶导数) = 24,被污染了!

# 正确做法: backward前先清空,拿到的才是"干净"的二阶导
x3 = torch.tensor(2.0, requires_grad=True)
y3 = x3 ** 3
y3.backward(create_graph=True)
first_order_grad3 = x3.grad
x3.grad = None                          # 必须先清空
first_order_grad3.backward()
print("清空后再backward:", x3.grad.item())
assert x3.grad.item() == 12.0            # 干净的二阶导,和理论值/autograd.grad结果一致
```

**面试怎么问 + 追问链:**
- **Q:** "怎么用 PyTorch 算二阶导数?"—— 期望答出 `create_graph=True` 让梯度计算本身可微,进而能再 backward 一次。
- **追问:** "为什么 MAML 这类需要二阶梯度的元学习算法,内部实现通常用 `torch.autograd.grad` 而不是 `.backward()`?"—— 期望连回"`.backward(create_graph=True)` 会在参数和梯度之间形成引用环导致内存泄漏,`autograd.grad` 不写 `.grad` 属性天然没有这个问题",最好能提到这是官方警告文案里明确写出来的,不是社区经验之谈。
- **深挖追问(容易翻车的细节):** "如果我用 `.backward(create_graph=True)`,然后不清空 `.grad` 直接对它再 `backward()` 一次,会发生什么?"—— 期望理解"二阶梯度会**累加**到已经存有一阶梯度值的 `.grad` 上",能现场算出这个被污染的数值(一阶+二阶的和)说明是真的懂了累加机制,不是背了个"要清空"的规则。

**常见坑:** 忘记 `create_graph=True` 情况下算出的梯度默认还是脱离计算图的(除非显式开启),对它再 `backward()` 会报"没有 `grad_fn`"之类的错误;用 `.backward(create_graph=True)` 后不清空 `.grad` 就复用,会让高阶梯度和低阶梯度累加混在一起,数值"看起来正常"实际是错的,比直接报错更危险。

---

## 9. 自定义 `torch.autograd.Function` —— `forward`/`backward`/`ctx.save_for_backward`

**是什么:**
```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)     # 把backward要用到的tensor存进ctx
        return ...                    # 前向计算逻辑

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors        # 取回forward保存的tensor
        return ...                    # 返回"对每个前向输入的梯度",顺序和forward的入参一一对应

y = MyOp.apply(x)   # 用.apply(...)调用,不要直接调用MyOp(x)
```

**一句话:** 当内置 op 组合无法表达、或者你想要一个数值更稳定/更高效的自定义反向逻辑时,`torch.autograd.Function` 让你手动"挂"一个自己的 `grad_fn` 节点到计算图里,`ctx` 是 forward 和 backward 之间传递信息的桥梁。

**底层机制/为什么这样设计:** `.apply()` 内部会把你的 `forward` 输出包装成一个新的 non-leaf tensor,并把这个 tensor 的 `grad_fn` 设成一个基于你的 `backward` 实现自动生成的节点(类名是 `<你的类名>Backward`,注意第 2 节提到过,这类自定义节点**没有**内置 op 常见的数字后缀)。`ctx`("context")是必须存在的,因为 `forward`/`backward` 是两个独立的 `staticmethod`(没有 `self`),没法用普通的闭包变量在两次调用之间传递状态,必须靠这个显式的上下文对象。`ctx.save_for_backward(...)` 专门用来保存"数据类"的 tensor,而不是简单地写 `ctx.x = x`——区别是**前者会受 01 篇第 7 节讲的版本计数器保护**,一旦这个 tensor 在 backward 真正需要它之前被原地修改过,会正确报错提醒你;直接挂 `ctx.x = x` 这种写法完全绕开了这层保护,即使值被污染,`backward` 也会静默地用错误的值继续算下去,不会有任何报错——下面的例子会现场对比这两种写法的真实差异,而不是停留在"官方建议这样做"的说教。

**可运行例子:**
```python
import torch
import torch.nn.functional as F

# --- 自定义ReLU: 验证前向/反向都和内置F.relu完全一致 ---
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input

x1 = torch.randn(5, requires_grad=True)
x2 = x1.detach().clone().requires_grad_(True)   # 独立副本,用于交叉对比

y1 = MyReLU.apply(x1)
y2 = F.relu(x2)
assert torch.equal(y1, y2)                        # 前向完全一致
print("自定义grad_fn类名(没有数字后缀):", type(y1.grad_fn).__name__)
assert type(y1.grad_fn).__name__ == "MyReLUBackward"

y1.sum().backward()
y2.sum().backward()
assert torch.equal(x1.grad, x2.grad)               # 反向也完全一致

# --- save_for_backward vs 直接挂ctx属性: 版本计数器保护的真实差异 ---

class SafeExp(torch.autograd.Function):        # 正确写法: 用save_for_backward
    @staticmethod
    def forward(ctx, x):
        y = x.exp()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        return grad_output * y

xs = torch.tensor([1.0, 2.0], requires_grad=True)
ys = SafeExp.apply(xs)
ys.add_(100)                                      # 原地修改forward的输出
try:
    ys.sum().backward()
    assert False
except RuntimeError as e:
    print("SafeExp(受保护)报错:", str(e))
    assert "modified by an inplace operation" in str(e)
    assert "SafeExpBackward" in str(e)
    assert "is at version 1; expected version 0" in str(e)

class UnsafeExp(torch.autograd.Function):        # 错误示范: 直接挂ctx属性,没有版本保护
    @staticmethod
    def forward(ctx, x):
        y = x.exp()
        ctx.y_unsafe = y            # 没用save_for_backward
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.y_unsafe
        return grad_output * y

xu = torch.tensor([1.0, 2.0], requires_grad=True)
yu = UnsafeExp.apply(xu)
yu.add_(100)                                       # 同样原地修改
yu.sum().backward()                                 # 不报错!没有版本保护,静默放行
wrong_grad = xu.grad.tolist()
correct_grad = xu.exp().tolist()
print("UnsafeExp算出的(被污染的)梯度:", wrong_grad)
print("理论正确梯度 e^x:", correct_grad)
assert torch.allclose(xu.grad, xu.exp() + 100)      # 梯度精确等于"被+100污染后的e^x",不是随机错误
assert not torch.allclose(xu.grad, xu.exp(), atol=1.0)  # 明显偏离正确值
```

**面试怎么问 + 追问链:**
- **Q:** "什么时候需要自定义 `torch.autograd.Function`,而不是直接用内置 op 组合出来?"—— 期望举出"内置组合数值不稳定(比如某些函数直接算比拆开算精度更好)""想要一个比自动微分组合出来更高效的手写反向""某个操作根本没法用现有可微分 op 表达"这几类场景。
- **追问 1(区分度很高):** "`ctx.save_for_backward(x)` 和直接 `ctx.x = x` 有什么区别,为什么官方文档强调要用前者?"—— 期望答出"前者受版本计数器保护,原地修改会被正确检测并报错;后者完全绕开这层保护,backward 会静默使用被污染的值算出一个'看起来正常但其实错误'的梯度",最好能现场举出具体污染后的数值。
- **深挖追问:** "自定义 `Function` 的 `backward` 返回值,和 `forward` 的输入参数是什么关系?如果 `forward` 有 3 个输入,其中 1 个不需要梯度,`backward` 要返回几个值?"—— 期望知道"`backward` 返回值数量、顺序必须和 `forward` 的位置参数一一对应,不需要梯度的位置可以返回 `None`",这是实践中很容易踩的一个位置错位坑。

**常见坑:** 用 `MyOp(x)` 而不是 `MyOp.apply(x)` 调用(直接实例化不会正确接入 autograd 机制);在 `forward` 里保存了 tensor 却忘了用 `save_for_backward`,当时测试"岁月静好"(因为测试代码没有恰好在 backward 前做原地修改),一旦上线遇到某处不巧的原地操作,梯度会静默出错却没有任何报错信号,比直接崩溃更难排查。

---

## 10. 梯度累加机制 —— 为什么训练循环必须 `zero_grad()`

**是什么:** 多次调用 `.backward()` 而不清零 `.grad`,梯度是**累加**而不是覆盖——这正是第 3 节 `AccumulateGrad` 这个名字里 "Accumulate" 的字面含义。

**一句话:** `AccumulateGrad` 节点被触发时执行的逻辑大致是"如果 `.grad` 还是空的就直接赋值,否则加到已有值上",不是"覆盖成最新一次的结果"——这是为什么训练循环里 `optimizer.zero_grad()` 是刚需,而不是可有可无的防御性代码。

**底层机制/为什么这样设计:** 如果 `AccumulateGrad` 默认做的是"覆盖"而不是"累加",像第 4 节验证过的"同一个 leaf 在一次前向里被用了两次"这种场景(`z=x+x`)就没法正确工作——两条路径的梯度会互相覆盖而不是相加,链式法则直接算错。累加是保证"一次 backward 内部多路径汇入正确"这件事的**必要条件**;而"多次 backward 之间也累加"只是同一套底层逻辑的另一种表现——`AccumulateGrad` 并不区分"这次梯度来自同一次 backward 的另一条路径"还是"来自另一次完全独立的 backward 调用",它只知道"又来了一份要加进去的梯度"。这个设计顺带带来一个有用的副作用:**梯度累加训练技巧**——显存不够跑不了大 batch 时,把一个大 batch 拆成多个小 batch,分别 forward + backward(不 `zero_grad`、不 `step`),梯度自然逐份累加,最后一次性 `step` 再清零,数学上等价于直接用大 batch 训练。

**可运行例子:**
```python
import torch

# --- 基础验证: 两次backward,梯度是累加不是覆盖 ---
x = torch.tensor(2.0, requires_grad=True)
y1 = x ** 2
y1.backward()
assert x.grad.item() == 4.0                # dy1/dx = 2x = 4

y2 = x ** 2
y2.backward()                                # 不清零
assert x.grad.item() == 8.0                 # 4+4=8,不是覆盖成4

x.grad = None                                 # 手动清零,证明"覆盖"和"累加"的区别
y3 = x ** 2
y3.backward()
assert x.grad.item() == 4.0                 # 清零后就是干净的单次结果

# --- 真实场景: 梯度累加训练技巧(用多个小batch模拟一个大batch,不调用zero_grad/step) ---
w = torch.tensor([1.0, 2.0], requires_grad=True)
micro_batches = [torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0]), torch.tensor([3.0, 3.0])]

for batch in micro_batches:
    loss = (w * batch).sum()
    loss.backward()          # 不zero_grad,不step,让梯度自然累加

expected = sum(micro_batches)                # d(loss_i)/dw = batch_i,理论上三次累加等于三个batch求和
assert torch.equal(w.grad, expected)
print("累加后的w.grad:", w.grad.tolist(), " 等价于把3个micro-batch的梯度加起来:", expected.tolist())
```

**面试怎么问 + 追问链:**
- **Q:** "为什么训练循环里必须调用 `optimizer.zero_grad()`?"—— 期望答出"梯度默认是累加的,不清零会把历史多个 step 的梯度混在一起"。
- **追问:** "如果忘记 `zero_grad()`,训练会直接报错吗?具体会出现什么后果?"—— 期望答"不会报错,而是梯度值不断变大(累积了历史 step 的贡献),导致参数更新步子越来越大,训练发散或者不收敛——这是一类'不报错但结果错'、比直接崩溃更难排查的 bug"。
- **深挖追问(反向利用这个机制):** "能不能举一个'不清零反而是故意的、有用的'场景?"—— 期望答出梯度累加训练技巧(显存不够时用多个小 batch 模拟大 batch),说明理解的是机制本身,而不是死记"训练循环必须 zero_grad"这一条规则。

**常见坑:** 以为 `.backward()` 内部会自动清零(不会,这是 `optimizer.zero_grad()`/手动清零单独要做的事);只在"多次调用 `.backward()`"这个场景下理解累加,忽略了"同一次 backward 内部,一个 leaf 被多路径依赖"(第 4 节 `z=x+x`)本质上是同一套 `AccumulateGrad` 机制的另一种触发方式。

---

## 11. `torch.no_grad()` vs `requires_grad_(False)` vs `.detach()` —— 三者的精确区别

**是什么:**
```python
with torch.no_grad():
    ...    # 代码块内新产生的所有tensor都不记录grad_fn/不构建计算图,不管输入是不是requires_grad

x.requires_grad_(False)   # 原地修改x自己的requires_grad属性,还是同一个python对象,不新建tensor

y = x.detach()             # 从x切出一个新tensor,共享同一块storage,但requires_grad=False、grad_fn=None
                            # (01篇第8节已经讲过detach和clone/.data的区别,这里聚焦它在"控制梯度追踪"这个维度的定位)
```

**一句话:** 三者解决的是同一类问题("不要梯度追踪")但作用维度完全不同——`no_grad` 是**代码块范围**的临时开关(影响这段代码里所有新计算);`requires_grad_(False)` 是**改一个 tensor 自己的持久状态**(不新建 tensor,离开任何上下文依然生效);`detach()` 是**新建一个脱离计算图的独立对象**(原 tensor 完全不受影响)。

**底层机制/为什么这样设计:** `no_grad` 通过一个线程局部的"梯度模式"标志位实现——进入代码块时把这个标志位设为 False,任何 op 执行时都会检查它,标志位为 False 时**无论输入 `requires_grad` 是什么**,都不创建 `grad_fn`、不往图里挂节点,纯粹当成普通数值运算跑;退出代码块后标志位恢复。这个"检查全局标志位"的逻辑在优先级上**高于**"输入是否 requires_grad"——所以哪怕块内的输入 `requires_grad=True`,输出照样是 `requires_grad=False`(下面会现场验证)。`requires_grad_(False)` 只是直接改 tensor 对象自己的一个字段,和线程局部标志位是完全不同的两套机制,互不干扰:即使在 `no_grad` 块内调用它,这个改动也会在离开代码块之后继续生效(因为它压根不依赖那个临时标志位)。`detach()` 则是创建一个新的 tensor 对象,共享同一份 storage(01 篇第 8 节验证过),但这个新对象的 autograd 相关字段是全新的"干净"初始状态,和原 tensor 的计算历史彻底断开。

**可运行例子:**
```python
import torch

# 1. no_grad: 块内所有新tensor不记录计算图
x = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    y = x * 3
print("no_grad块内: y.requires_grad =", y.requires_grad, " y.grad_fn =", y.grad_fn)
assert y.requires_grad is False
assert y.grad_fn is None

y_outside = x * 3   # 对比: 块外同样的操作
assert y_outside.requires_grad is True
assert y_outside.grad_fn is not None

# 2. requires_grad_(False): 原地修改,同一个python对象,不新建tensor
x2 = torch.tensor(2.0, requires_grad=True)
id_before, ptr_before = id(x2), x2.data_ptr()
ret = x2.requires_grad_(False)
assert id(x2) == id_before and x2.data_ptr() == ptr_before   # 还是同一个对象、同一块内存
assert ret is x2                                                # 返回值就是self本身
assert x2.requires_grad is False

# 3. detach(): 新对象,共享内存,不带计算历史,原tensor完全不受影响
x3 = torch.tensor(2.0, requires_grad=True)
y3 = x3 * 3
d3 = y3.detach()
assert d3 is not y3 and d3.data_ptr() == y3.data_ptr()   # 新对象,但共享storage
assert d3.requires_grad is False and d3.grad_fn is None
assert y3.requires_grad is True and y3.grad_fn is not None  # y3本身毫发无损

# 4. 优先级验证: no_grad块内,即使输入requires_grad=True,输出也强制是False
x4 = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    y4 = x4 * 3
    print("no_grad优先级验证: 输入requires_grad=True, 块内输出requires_grad =", y4.requires_grad)
    assert y4.requires_grad is False   # no_grad的优先级高于"输入是否需要梯度"

# 5. requires_grad_(False)是持久状态改变,离开no_grad块依然生效(和no_grad的"临时性"不同)
x5 = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    x5.requires_grad_(False)
print("离开no_grad块后, x5.requires_grad =", x5.requires_grad)
assert x5.requires_grad is False    # 这个改动不是"临时"的,不会随着离开代码块而复原
```

**面试怎么问 + 追问链:**
- **Q:** "`torch.no_grad()`、`requires_grad_(False)`、`.detach()` 三者有什么区别,分别在什么场景用?"—— 期望答出"临时上下文开关 / 原地改自身状态 / 新建脱离计算图的对象"这三个不同维度,并举出典型场景:推理阶段用 `no_grad` 包住整个前向省显存;冻结部分网络参数(迁移学习只训练最后一层)用 `requires_grad_(False)`;GAN 训练判别器时需要生成器输出参与前向但不希望梯度传回生成器,用 `detach()`。
- **追问 1(容易漏答):** "在 `no_grad` 块里,对一个 `requires_grad=True` 的 tensor 做运算,输出的 `requires_grad` 是什么?"—— 期望现场推理或验证出"是 `False`,因为 `no_grad` 的线程局部标志位优先级高于输入本身的 `requires_grad`",而不是含糊地说"不确定,没试过"。
- **深挖追问:** "如果在 `no_grad` 块里调用某个 tensor 的 `requires_grad_(False)`,离开代码块之后这个改动还在不在?"—— 期望理解"在,因为 `requires_grad_()` 是直接修改 tensor 自身的持久字段,和 `no_grad` 那个临时性的线程局部标志位是两套完全独立的机制,不会互相影响生命周期"。

**常见坑:** 把三者当成"同一件事的三种写法"随手替换——`requires_grad_(False)` 是**原地**操作,会影响所有持有同一个 tensor 对象引用的代码路径(不像 `detach()` 那样创建新对象天然隔离),在一个被多处共享的 tensor 上调用它可能产生意料之外的副作用;混淆"`no_grad` 的临时性"和"`requires_grad_(False)` 的持久性",错误地以为离开 `no_grad` 块后所有块内做的修改都会"复原"。

---

## 12. `requires_grad` 的传播规则

**是什么:** 一个运算的多个输入 tensor 里,只要有**一个** `requires_grad=True`,输出 tensor 就自动 `requires_grad=True`(在 `no_grad` 上下文之外)——不需要所有输入都要求梯度。

**一句话:** `requires_grad` 是"传染性"的——一堆输入里只要有一个"needs grad",这次运算产生的输出就会被标记为需要梯度,哪怕其余输入全都是普通常量。

**底层机制/为什么这样设计:** autograd 引擎决定"要不要为这次运算创建 `grad_fn` 节点"时的判断逻辑是"检查所有输入的 `requires_grad`,只要有一个为 `True` 就构建"。这个规则设计成"任一为真即为真"而不是"全部为真才为真",本质原因是**反向传播需要完整链路**——如果只有部分输入被追踪、其余被忽略,一旦某条路径"漏标"了 `requires_grad`,这条路径的梯度就传不下去了,所以规则设计成宁可保守也不能漏标。但这不代表"所有输入都会被计算梯度"——`next_functions` 里,对应"不需要梯度"的那些输入位置依然是 `(None, 0)`(第 3 节讲过的空位),autograd 不会为它们做多余的反向计算,这是性能层面的优化:只追踪真正需要的那部分路径。

**可运行例子:**
```python
import torch

# --- 基础传播规则 ---
a = torch.tensor(1.0, requires_grad=False)
b = torch.tensor(2.0, requires_grad=True)
c = a + b
assert c.requires_grad is True     # 只要有一个输入requires_grad=True,输出就是True
assert a.requires_grad is False    # a自己不受影响,不会被"传染"

d = a * a                            # 两个输入都是False
assert d.requires_grad is False

# --- next_functions里,不需要梯度的输入对应位置是(None, 0) ---
e = a + b
print("e.grad_fn.next_functions:", e.grad_fn.next_functions)
assert e.grad_fn.next_functions[0][0] is None          # 对应a的位置: 没有节点
assert type(e.grad_fn.next_functions[1][0]).__name__ == "AccumulateGrad"  # 对应b的位置: 真实节点

# --- 5个输入,只有1个requires_grad=True: 验证只有对应位置是真实节点,其余全是None ---
t_no1 = torch.tensor([1.0])
t_no2 = torch.tensor([2.0])
t_yes = torch.tensor([3.0], requires_grad=True)
t_no3 = torch.tensor([4.0])
t_no4 = torch.tensor([5.0])

out = torch.cat([t_no1, t_no2, t_yes, t_no3, t_no4])
assert out.requires_grad is True
print("out.grad_fn:", type(out.grad_fn).__name__)
print("out.grad_fn.next_functions:", out.grad_fn.next_functions)

non_none = [(i, fn) for i, (fn, _) in enumerate(out.grad_fn.next_functions) if fn is not None]
assert len(non_none) == 1                    # 5个输入,只有1个位置是真实节点
idx, fn = non_none[0]
assert idx == 2                                # 恰好是t_yes所在的第3个位置(index=2)
assert fn.variable is t_yes                    # 而且这个节点确实指回t_yes
```

**面试怎么问 + 追问链:**
- **Q:** "一个 tensor 由多个输入运算而来,它的 `requires_grad` 怎么决定?"—— 期望答出"只要有一个输入 `requires_grad=True`,输出就是 `True`"。
- **追问:** "如果一个函数有 5 个输入,其中 1 个 `requires_grad=True`,其余 4 个是普通常量,输出的 `grad_fn.next_functions` 会有几个非 `None` 的位置?"—— 期望答出"只有 1 个,对应那个 `True` 的输入;其余 4 个位置是 `(None, 0)`",最好能说出这是性能优化(不追踪不需要梯度的分支能省掉多余的反向计算)。
- **深挖追问:** "既然输出的 `requires_grad` 是 `True`,是不是意味着这 5 个输入都会'收到'梯度?"—— 期望明确否定:"不是,只有 `requires_grad=True` 的那个输入才会在 backward 时真正被计算并累加梯度,其余常量输入即使参与了运算,也不会出现在这条反向路径上,连计算都不会做"——这是一个专门用来分辨"背结论"和"真理解"的好问题。

**常见坑:** 把"输出 requires_grad=True"误解成"所有输入都会被求梯度"——只有真正 `requires_grad=True` 的输入才会被追踪和计算,其余输入即使数值上参与了这次运算,也完全不出现在反向路径里;忘记这个规则在 `no_grad` 上下文里会被整体覆盖(第 11 节),`no_grad` 块内不管输入是什么,输出一律 `requires_grad=False`。

---

## 13. `register_hook` —— 反向传播过程中查看/修改中间梯度

**是什么:** `tensor.register_hook(fn)` 给一个 tensor 注册一个回调函数,在反向传播**流经这个 tensor 的那一刻**被调用,传入当前这一步的梯度;如果 `fn` 的返回值不是 `None`,这个返回值会**替代**原本的梯度继续往下游(更上游的输入)传。

**一句话:** `register_hook` 是"在梯度反向流经某个 tensor 的瞬间"插入自定义逻辑的钩子——常用于调试(打印/记录中间梯度,排查梯度消失/爆炸)或者直接修改流经该处的梯度(比如局部梯度裁剪)。

**底层机制/为什么这样设计:** hook 本质上是往这个 tensor 对应的反向节点(`grad_fn` 或者叶子的 `AccumulateGrad`)上注册了一个回调:该节点执行反向计算、即将把梯度往更下游传之前,会先依次调用所有注册在这个 tensor 上的 hook,把当前的梯度值传进去;如果某个 hook 返回了一个新 tensor,后续继续往下游传播的梯度就被**替换**成这个返回值(返回 `None` 则不修改,继续用原值)。第 6 节提到的 `retain_grad()`,本质上可以理解为一个内置的、专门"偷看一眼、原样存到 `.grad`、不修改"的 hook——下面的例子会现场用 `register_hook` 手动实现出一样的效果来验证这个说法。

**可运行例子:**
```python
import torch

# --- 基础用法: hook能看到"流经这里"的梯度,还能替换它 ---
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2

captured = []
def hook(grad):
    captured.append(grad.clone())
    return grad * 0.5     # 把继续往下游(往x)传的梯度改成一半

h = y.register_hook(hook)
z = y.sum()
z.backward()

print("hook看到的原始梯度(dz/dy):", captured[0].tolist())
print("x.grad(hook修改后继续往下传的结果):", x.grad.tolist())
assert captured[0].tolist() == [1.0, 1.0, 1.0]    # dz/dy=1,hook看到的是"修改前"的原始值
assert x.grad.tolist() == [1.0, 1.0, 1.0]          # 修改后的0.5,乘以dy/dx=2,得到1.0
h.remove()                                            # 用完及时移除,避免影响后续复用

# --- 真实场景: 对某个中间量的梯度做局部裁剪(不同于对全部参数整体裁剪的clip_grad_norm_) ---
x2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y2 = x2 * 10
h2 = y2.register_hook(lambda grad: grad.clamp(min=-1.5, max=1.5))
loss2 = (y2 ** 2).sum()
loss2.backward()
# d(loss2)/d(y2) = 2*y2 = [20,40,60],被clamp到1.5; 再乘dy2/dx2=10 -> x2.grad全是15.0
print("clip hook后的x2.grad:", x2.grad.tolist())
assert x2.grad.tolist() == [15.0, 15.0, 15.0]
h2.remove()

# --- 用register_hook手动实现retain_grad()的效果,验证两者是同一套机制 ---
x3 = torch.tensor(2.0, requires_grad=True)
y3 = x3 * 3
manual_store = {}
h3 = y3.register_hook(lambda grad: manual_store.setdefault("grad", grad.clone()))  # 不return,不修改梯度
z3 = y3 ** 2
z3.backward()
print("手动hook存下来的梯度:", manual_store["grad"].item())
assert manual_store["grad"].item() == 12.0    # 和第6节retain_grad()拿到的y.grad完全一致(dz3/dy3=2*6=12)
h3.remove()

# --- hook返回形状不匹配的梯度,会现场报错(不是静默出错) ---
x4 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y4 = x4 * 2
h4 = y4.register_hook(lambda grad: torch.tensor([1.0, 2.0]))   # 故意返回错误形状(3 -> 2)
try:
    y4.sum().backward()
    assert False
except RuntimeError as e:
    print("hook形状不匹配报错:", str(e))
    assert "has changed the size of value" in str(e)
h4.remove()
```

**面试怎么问 + 追问链:**
- **Q:** "`register_hook` 有哪些实际应用场景?"—— 期望举出"诊断梯度消失/爆炸(在每层输出注册 hook 打印梯度范数)""自定义局部梯度裁剪"这类具体场景,而不是空泛地说"能修改梯度"。
- **追问 1:** "hook 函数如果返回一个新 tensor,对下游反向传播的影响是什么?"—— 期望答"替代原有梯度继续往前传,不是'追加'也不是'只读旁路'——后续所有依赖这个梯度的计算都会用修改后的值"。
- **深挖追问(区分度很高):** "`register_hook` 和 `retain_grad()` 有什么关系?"—— 期望理解 `retain_grad()` 本质上是一个内置的、"存一份不修改"的特殊 hook,能现场用 `register_hook` 手动复现出一样的效果(拿到完全相同的数值)是很有说服力的验证,而不是只会背"它们不一样"或者"它们差不多"这种模糊说法。

**常见坑:** hook 返回的 tensor 形状/dtype 和原梯度不匹配,会在 backward 时直接报错中断(不是静默产生错误结果,这点比"忘记 `save_for_backward`"那类坑更安全,至少会报错提醒);忘记调用 `.remove()` 清理不再需要的 hook——尤其是在循环/多次实验中反复对同一个长期存在的 tensor 注册 hook,会导致同一份梯度被多个残留的旧 hook 重复处理,产生难以察觉的错误结果。

---

## 14. `.backward()` vs `torch.autograd.grad()` —— 副作用式 vs 函数式

**是什么:**
```python
loss.backward()                                # 副作用式: 梯度写入每个leaf的.grad属性,不返回任何值
grads = torch.autograd.grad(outputs, inputs)    # 函数式: 直接返回一个tuple,每个元素对应inputs里每个tensor的梯度
```

**一句话:** `backward()` 是"把梯度写到 `.grad` 里,后面 `optimizer.step()` 自己去读";`torch.autograd.grad()` 是"我现在就要这个梯度的数值,直接给我,不要有任何副作用"。

**底层机制/为什么这样设计:** 两者共享同一套底层反向传播引擎——同样是从 output 出发反向拓扑遍历、调用各 Node 的反向计算,唯一的区别在于"到达叶子节点之后怎么处理梯度":`backward()` 到达叶子后触发 `AccumulateGrad`(第 3 节),把值写入 `.grad`(累加,第 10 节);`torch.autograd.grad()` 则是直接在引擎内部收集你指定的 `inputs` 对应的梯度值,包成一个 tuple 直接返回给你,**完全不经过 `.grad` 这条路径**,不会有任何累加副作用——每次调用都是"干净"的一次性结果,不管调用几次都不会互相污染。这也是第 8 节高阶导数场景更推荐 `torch.autograd.grad()` 的原因:MAML 这类元学习算法内部需要对"梯度"这个中间量本身继续做计算,却不想让这个中间过程污染任何模型参数的 `.grad`——用 `backward()` 做不到这种干净的隔离。需要特别注意:**"函数式、无副作用"只是指不碰 `.grad`,不代表它不受计算图释放规则约束**——`torch.autograd.grad()` 默认同样会在调用后释放计算图,多次调用同一张图同样需要 `retain_graph=True`,这一点和 `backward()` 完全对称。

**可运行例子:**
```python
import torch

# --- backward(): 写入.grad,有副作用 ---
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
assert x.grad.item() == 6.0
x.grad = None

# --- autograd.grad(): 函数式返回,完全不碰.grad ---
y2 = x ** 2
g = torch.autograd.grad(y2, x)
print("autograd.grad返回:", g, " 而x.grad依然是:", x.grad)
assert x.grad is None           # 完全没被touch过
assert g[0].item() == 6.0        # 但返回值是对的

# --- autograd.grad同样遵循图释放规则,不是"绝对安全可以调用任意多次" ---
y3 = x ** 2
g1 = torch.autograd.grad(y3, x)
try:
    torch.autograd.grad(y3, x)    # 第二次,图已经释放
    assert False
except RuntimeError as e:
    print("第二次调用报错:", str(e))
    assert "Trying to backward through the graph a second time" in str(e)

y3b = x ** 2                       # 换一张全新的图,配合retain_graph=True验证可以连续调用
g1b = torch.autograd.grad(y3b, x, retain_graph=True)
g2b = torch.autograd.grad(y3b, x)
assert g1b[0].item() == g2b[0].item() == 6.0

# --- 不在inputs列表里的leaf,即使requires_grad=True且参与了计算,.grad也完全不会被动 ---
x2 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(3.0, requires_grad=True)
y4 = x2 * w2
g3 = torch.autograd.grad(y4, x2)     # 只要x2的梯度
print("g3(只对x2求梯度):", g3, " w2.grad(完全没被碰过):", w2.grad)
assert w2.grad is None
assert g3[0].item() == 3.0             # dy4/dx2 = w2 = 3
```

**面试怎么问 + 追问链:**
- **Q:** "`.backward()` 和 `torch.autograd.grad()` 的区别是什么?"—— 期望答出"前者副作用式写入 `.grad`,后者函数式直接返回梯度值",能补充"底层是同一套引擎,区别只在到达叶子后的处理方式"是加分项。
- **追问:** "为什么 MAML 这类需要二阶梯度的元学习算法,内部实现通常用 `torch.autograd.grad` 而不是 `.backward()`?"—— 呼应第 8 节,期望答出"避免把中间梯度污染进任何参数的 `.grad`,以及避免 `.backward(create_graph=True)` 那个引用环内存泄漏问题"。
- **深挖追问(容易漏答的对称性):** "`torch.autograd.grad()` 完全不碰 `.grad`,是不是意味着可以对同一张图无限次调用它,不用担心 `retain_graph`?"—— 期望能明确回答"不是,它和 `backward()` 共享同一套图释放规则,默认调用一次之后图就被释放了,第二次同样会报错,需要 `retain_graph=True`——'函数式无副作用'只是指不碰 `.grad` 这一件事,不等于图的生命周期规则也变了"。这是一个专门筛掉"只记住结论不理解机制"候选人的问题。

**常见坑:** 以为 "函数式、不修改任何东西" 等于 "可以毫无顾虑地调用任意多次"——图释放规则和 `backward()` 完全对称,一样需要 `retain_graph=True` 才能对同一张图多次取梯度;混淆 "`inputs` 参数只是过滤返回值" 和 "只计算 `inputs` 对应的梯度"——虽然效果上确实只返回你要的那部分,但背后的反向传播仍然是沿着完整依赖图走的,不是"只从 output 到 inputs 抄近路"。

---

## 小结:这一批 14 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|------|
| 1 | 计算图动态构建(define-by-run) | 图是"跑一次 forward 现造一次",不是预先编译;能用原生 Python 控制流写模型是这个设计的直接推论 |
| 2 | `grad_fn` | 非叶子 tensor 背后挂的反向计算节点,leaf / requires_grad=False 的 tensor 都是 `None` |
| 3 | `next_functions` | 计算图真正的边,递归遍历能走到每个 leaf 的 `AccumulateGrad`,证明计算图是真实可遍历的数据结构 |
| 4 | `backward()` 执行过程 | 反向拓扑遍历 + 链式法则(每步是 vector-Jacobian product),手算结果与 `.backward()` 完全一致 |
| 5 | leaf vs non-leaf(`is_leaf`) | 判断标准是"有没有 `grad_fn`"而非"是不是用户创建";`requires_grad=False` 的 tensor 永远是 leaf |
| 6 | `retain_grad()` | 给非 leaf tensor 显式声明"backward 后我要看你的 `.grad`",本质是挂一个写入型 hook,只要在对应 backward 前调用即可生效 |
| 7 | `retain_graph=True` | 控制"这一次 backward 之后要不要释放中间 buffer",按调用次数逐次生效,不是全局/永久开关 |
| 8 | `create_graph=True` 与高阶导数 | 让反向传播本身被记录进新图,可对梯度再求梯度;`.backward()` 版本有官方实测的内存泄漏警告,推荐 `autograd.grad` |
| 9 | 自定义 `autograd.Function` | `forward`/`backward` 两个 staticmethod;`save_for_backward` 比直接挂 `ctx` 属性多一层版本计数器保护(实测污染后果) |
| 10 | 梯度累加机制 | `AccumulateGrad` 是累加不是覆盖,这是 `zero_grad()` 刚需的根本原因,也是梯度累加训练技巧的基础 |
| 11 | `no_grad`/`requires_grad_(False)`/`.detach()` | 临时上下文开关 vs 原地改自身持久状态 vs 新建脱离计算图的对象;`no_grad` 优先级高于输入本身的 `requires_grad` |
| 12 | `requires_grad` 传播规则 | 多输入只要一个 `True`,输出就 `True`;`next_functions` 里不需要梯度的输入对应位置是 `(None, 0)`,不做多余计算 |
| 13 | `register_hook` | 在梯度流经某 tensor 时插入回调,可查看/替换梯度;`retain_grad()` 本质是内置的一种特殊 hook |
| 14 | `.backward()` vs `torch.autograd.grad()` | 副作用式写入 `.grad` vs 函数式直接返回,但两者共享同一套图释放规则,同样需要 `retain_graph=True` 才能重复取梯度 |

下一批:[03-nn-module-internals.md](03-nn-module-internals.md) —— nn.Module 系统内核。

---

*更新:2026-07-07*
