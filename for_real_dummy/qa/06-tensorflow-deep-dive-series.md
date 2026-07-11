# Q&A —— TensorFlow 深挖系列(对标 torch-deep-dive 深度)的建立与验证(2026-07-08 ~ 07-11)

## Q:TensorFlow 也做一遍同等深度的 100 项计划,格式对标 torch-deep-dive

延续 [04](04-torch-deep-dive-series.md) 建立的面试深度方法论(签名 → 一句话 → 底层机制/为什么这样设计 → AI 场景 → 可运行例子 → 面试怎么问+追问链 → 常见坑),但换到 TensorFlow。先解决一个前置问题:仓库 `.venv` 里完全没装 TensorFlow/Keras/JAX,而且 TensorFlow 自 2.11 起官方砍掉了 Windows 原生 GPU 支持——要对齐 torch 系列"GPU 上真跑过"的验证深度,必须先在 WSL2 里单独搭一套环境,这是独立于内容写作的前置工程,当时和 python-idioms 系列([05](05-python-idioms-series.md))一起排期,选择先做完 python-idioms 这条纯 Python、无环境依赖的系列,再回头单独处理 TF 的环境搭建。

## 做了什么

建立了第五条深挖系列:**[tensorflow-deep-dive/](../tensorflow-deep-dive/00-roadmap.md)** —— 100 个知识点,13 个分类,覆盖 tensor 基础 → GradientTape 自动微分 → tf.function/AutoGraph 计算图(★TF 独有的两大重中之重,合计 20 项,占全系列五分之一)→ Keras 三套 API 内核 → 层的数学推导 → 损失函数 → 优化器 → fit() 内核与自定义训练循环 → tf.data 输入管道(★TF 独有)→ 内存性能 → 分布式训练 → 序列化部署 → 调试报错精解。

**环境:** WSL2(Ubuntu 24.04)+ 独立 venv `~/tf-venv`,TensorFlow 2.21.0,GPU 为 RTX 3080 Ti Laptop GPU(WSL2 直通,已验证 `list_physical_devices('GPU')` 非空)。做了一个主动的版本策略决定:TF 2.16 起 `tf.keras` 默认指向 Keras 3(多后端架构),但面试题库和大多数教程的心智模型还是经典 Keras 2 语义,所以显式装了 `tf_keras` 包并设置 `TF_USE_LEGACY_KERAS=1`,把 `tf.keras` 锁回经典实现——这个决定本身在 04/12 两个类目里各有一条知识点专门讲清楚,面试被问到"Keras 2 和 3 什么关系"能答明白。

## 怎么做的

- **分工:** 13 个分类一次性并行派发给 13 个 subagent,一对一分类,不像 torch 系列那样先打样再铺开——因为格式是照抄 torch-deep-dive 现成模板,不需要重新摸索。
- **中途反复遭遇 API 限额,横跨好几天:** 和 torch 系列那次"一次性打断、我自己接手写完"不同,这次是断断续续、多次反复的限额中断,前后拖了三四天才让 13 个分类全部落地。应对方式沿用同一条原则——不因为"agent 还没回应"就假设它失败或重新派发,而是等限额窗口过去后用 `SendMessage` 恢复被打断的 agent(带着它自己完整的历史上下文继续,不丢进度);批量恢复前先单点测试一个 agent 探路,确认限额真的缓过来了再批量恢复其余的,避免在限额仍然生效时盲目重试浪费效果。
- **两个独立 agent 分别撞见同一个 AutoGraph 限制,是这次最有方法论价值的意外发现:** 09(tf.data)和 13(调试排错)两个类目的 agent 各自独立发现,`_verify_md.py` 用 `python -c "<code>"` 执行代码块时,AutoGraph 对 `if`/`for` 控制流的转换依赖 `inspect.getsource()` 读取真实源码文件——`-c` 执行的代码没有真实文件支撑,AutoGraph 会静默降级不转换,导致产出的报错类型和真实 `.py` 文件运行完全不同(比如非标量条件预期是 `ValueError`,`-c` 下变成 `OperatorNotAllowedInGraphError`)。09 先解决:把待测函数写入真实临时 `.py` 文件、用 `importlib` 导入,让 AutoGraph 总能读到真实 source。13 后来撞见同一个问题时,直接复用了这个已验证的解法,没有重新摸索。
- **08(训练循环内核)复验阶段抓到的问题最多,不是走流程:**
  - 最初用朴素 Python 字典计数器验证 `train_step` 每步真的执行了,`assert call_count==6` 失败,实际只有 2——顺着这个"假设被证伪"深挖下去,才精确定位到 `make_train_function` 把 `train_step` 包进 `tf.function(reduce_retracing=True)`,只在"精确形状 trace 一次 + 泛化形状 trace 一次"时才有 Python 副作用,之后的真实调用不会再执行 Python 代码。这个发现被写成了 08 篇第 1 节的核心论证,不是简单绕过。
  - 同一个坑在第 9 节验证阶段又踩了一次:`TrainFlagProbe` 层用 Python list 记录 call() 次数,整篇跑验证脚本时数值对不上(2/2 而不是真实的 4/2 训练/验证 batch 数)——本质和第 1 节是同一个机制,改用 `tf.Variable` 计数器后精确对上,顺手把这次真实复现过程写成了正文里的教学注记。
  - 一处原本以为是坑、实测发现"比预期更省心"的反例:自定义 metric 不需要显式覆写 `metrics` 这个 property 才能被 `fit()` 自动 reset——Keras 对 `self.xxx = tf.keras.metrics.Xxx()` 这种属性赋值本身就自动追踪,和 `tf.Variable` 走同一套机制,原计划设计的"leaky metric"陷阱测试反而验证失败,如实改成记录这个真实行为而不是保留错误预设。
- **我自己的独立复验(不只信任 agent 自我报告):** 全部 13 个文件、195 个代码块,用 `_verify_md.py` 在 WSL2 里逐文件重新跑一遍,13/13 文件全部 195/195 代码块通过。其中 05(层的数学推导)是唯一一个此前只有 agent 自我报告、从未被我亲自复验过的文件,这次补上了(7/7 通过)。复验过程中额外清理了一处分类 11(分布式训练)遗留的 `_mwms_tmp/`(MultiWorkerMirroredStrategy 测试产生的 rank0.txt/rank1.txt 临时文件,确认是验证副产物而非正文内容后删除)。

## 结论

13 个文件全部完成并逐条独立验证,合计精确 100 个知识点(8+10+10+7+7+5+7+9+8+6+6+8+9),195 个代码块 100% 通过。这条系列最大的方法论价值不是单个 TF 知识点本身,是**大量和 torch 系列直接对照出的框架设计分歧**,这些分歧本身就是很好的面试追问素材:

- **梯度聚合:** TF 默认多设备梯度聚合是 **SUM**,PyTorch DDP 默认是等效 **mean**——同样的多卡设置,有效学习率会因为这个默认值差异悄悄乘上卡数倍。
- **混合精度:** TF 的 `LossScaleOptimizer` 在 `compile()` + `mixed_float16` 策略下**全自动包装**,PyTorch 的 `GradScaler` 需要**显式实例化**——同样的功能,两个框架对"要不要让用户知道"的设计哲学正好相反。
- **自动微分:** TF 的 `GradientTape` 是显式 tape-based(手动划定记录范围),PyTorch autograd 是隐式 define-by-run(默认全程记录)——03/13 两个类目都从这个根本差异出发解释了各自的常见坑。
- **设备放置:** TF2 默认开启软设备放置,CPU/GPU tensor 混用大多数情况下自动插入拷贝、不报错,PyTorch 对跨设备运算是硬报错——13 篇专门验证了这个反差比预想的更大。

三个"从底层机制讲到面试追问链"的深挖系列(torch-deep-dive 100 项 + tensorflow-deep-dive 100 项)现在互为镜像,加上 numpy-deep-dive + python-advanced + python-idioms,五条系列合计约 366 个知识点,全部完成并验证。
