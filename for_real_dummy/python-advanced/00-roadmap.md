# Python 高级知识补完计划 —— 课堂不教但 AI 研究代码里到处都是

> 目标:20 个 Python 中高级知识点,由浅入深,一个知识点一个小节,分批次完成。
> 背景:你的 Python 课只教了基本语法,没讲这些内容——但博士学长仓库里的代码(尤其 PyTorch/HuggingFace 相关)几乎每个文件都会用到装饰器、生成器、类型注解这些东西,不补上这块看代码会经常卡壳。
> 这里和 [numpy-deep-dive/](../numpy-deep-dive/) 系列是并行的两条线:一条补 numpy/AI 数值计算,一条补 Python 语言本身的中高级特性。两者互不依赖,可以交替看。

---

## 每个知识点的固定讲解结构

1. **是什么**——一句话
2. **为什么课堂不教但很重要**——具体关联到 AI 研究代码里的哪类写法
3. **从最笨的写法讲起**——先给出不用这个语法糖的"笨办法",再引入正式写法,讲清楚它解决了什么问题(不是凭空出现的魔法)
4. **AI 研究代码里的真实例子**——不是玩具例子,是这类代码库里真实会出现的写法
5. **可运行例子**——能直接跑,关键结论用 assert/print 验证
6. **常见坑**

---

## 进度表

| # | 分类 | 文件 | 知识点数 | 状态 |
|---|------|------|---------|------|
| 01 | 函数与闭包进阶 | [01-functions-and-closures.md](01-functions-and-closures.md) | 6 | ✅ 已完成(已验证;2026-07-24 P/T/L/V/D 可读性审计:闭包一节原文只用文字+代码说明"捕获的是变量引用",没有状态示意图,补了一张"格子(cell)"逐次调用状态表 + `__closure__`/`cell_contents` 现场验证,新增代码块已独立跑通) |
| 02 | 迭代器与生成器 | [02-iterators-and-generators.md](02-iterators-and-generators.md) | 4 | ✅ 已完成(已验证;2026-07-24 P/T/L/V/D 可读性审计:`yield` 一节原文只用 print 佐证暂停/恢复,没有状态机示意图,补了一张逐次 `next()` 状态表 + `gi_frame`/`f_locals` 现场验证,新增代码块已独立跑通) |
| 03 | 面向对象进阶 | [03-oop-advanced.md](03-oop-advanced.md) | 5 | ✅ 已完成(已验证;2026-07-24 P/T/L/V/D 可读性审计:通读未发现需修复的 gap,未改动——本篇范围只到单继承 + `super()`,不含多重继承/MRO,故无需补菱形继承图) |
| 04 | 类型注解、上下文管理与并发 | [04-typing-context-and-concurrency.md](04-typing-context-and-concurrency.md) | 5 | ✅ 已完成(已验证;2026-07-24 P/T/L/V/D 可读性审计:补了"事件循环"实际调度机制的说明(原文只说"创建循环、执行、关闭"没解释怎么调度)、补了 `threading.Thread` 的 `.start()`/`.join()` 语义说明(原文直接使用未定义)) |
| 05 | 进阶深度追加:5 个多级追问链案例 | [05-advanced-interview-depth.md](05-advanced-interview-depth.md) | 5案例(不计入20) | ✅ 已完成(已验证,10/10代码块独立进程复验全部通过;基于dsa-deep-dive/python-idioms已验证的5条追问轴线撰写——①threading vs multiprocessing任务粒度决定加速还是拖累+GIL前瞻(工程约束递增轴核心;实测细粒度400个任务multiprocessing比顺序慢约5000-6000倍、粗粒度4个600万次循环任务multiprocessing比顺序快约1.5-2倍,并用sys._is_gil_enabled()确认当前venv为GIL启用的常规构建、混合CPU+IO线程实测总耗时明显小于顺序相加证明sleep期间GIL被释放)、②装饰器叠加顺序导致审计日志被缓存吞掉(方案批判迭代轴核心;lru_cache叠外层时3次相同成功调用审计日志只留1条、audit叠外层时留3条,并验证该bug只影响成功重复调用、不影响失败路径这一更隐蔽性质)、③descriptor协议与property底层机制(决策依据追问轴核心;现场复现朴素描述符状态被跨实例覆盖的结构性bug,用weakref.WeakKeyDictionary修复并用weakref.finalize精确验证无内存泄漏)、④类型注解运行时校验器自建到碰壁(真实性验证轴核心;验证FastAPI式运行时校验的真相是库代码自己读__annotations__实现的,自建校验器遇到list[int]参数化泛型isinstance直接崩溃,用get_origin修复后仍暴露只查外壳不查元素类型的真实缺口)、⑤IterableDataset生成器只能用一次陷阱(规模递增轴核心;用真实torch.utils.data.IterableDataset+DataLoader复现第二个epoch静默返回空批次的真实bug,验证两种修法——新建生成器表达式/__iter__自身写成yield from生成器函数——均可正确支持多epoch,呼应02类"这里先点一下不展开"的伏笔));2026-07-24 P/T/L/V/D 可读性审计:通读未发现需修复的 gap,未改动——进阶案例定位本就假设读者已掌握 01-04,信息密度符合该定位 |

**合计:20 个知识点,4 篇 + 1 篇进阶深度追加(5 个案例,不计入 20),全部完成并独立验证。**

---

## 明细

### 01 函数与闭包进阶
1. `lambda` 表达式
2. `*args` / `**kwargs`
3. 闭包(closure)——函数记住外部变量
4. 装饰器基础(`@decorator` 语法糖,本质是"函数包函数")
5. 带参数的装饰器(装饰器工厂,两层包裹)
6. `functools` 常用工具:`wraps`/`partial`/`lru_cache`

### 02 迭代器与生成器
1. 迭代器协议(`__iter__`/`__next__`,for 循环的本质)
2. `yield` 生成器基础(暂停与恢复执行)
3. 生成器表达式 vs 列表推导(惰性求值,内存差异——大数据集/流式处理为什么要用生成器)
4. `yield from` 委托生成器

### 03 面向对象进阶
1. `@property` / `@staticmethod` / `@classmethod`
2. 魔法方法(dunder methods):`__init__`/`__repr__`/`__eq__`/`__len__`/`__getitem__`
3. 继承与 `super()`
4. `@dataclass`(自动生成 `__init__`/`__repr__`,配置类的标配写法)
5. 抽象基类 `abc.ABC`(定义"子类必须实现哪些方法"的契约)

### 04 类型注解、上下文管理与并发
1. 类型注解(type hints)与 `typing` 模块
2. `with` 语句与上下文管理器(`__enter__`/`__exit__`、`contextlib.contextmanager`——呼应 `torch.no_grad()` 的实现原理)
3. `async`/`await` 协程基础
4. `asyncio` 事件循环基础用法
5. `threading` vs `multiprocessing`(GIL 的影响,I/O 密集 vs CPU 密集该选哪个)

---

*更新:2026-07-13(新增 05 进阶深度追加)*
