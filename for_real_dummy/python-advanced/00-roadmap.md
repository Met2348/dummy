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
| 01 | 函数与闭包进阶 | [01-functions-and-closures.md](01-functions-and-closures.md) | 6 | ✅ 已完成(已验证) |
| 02 | 迭代器与生成器 | [02-iterators-and-generators.md](02-iterators-and-generators.md) | 4 | ✅ 已完成(已验证) |
| 03 | 面向对象进阶 | [03-oop-advanced.md](03-oop-advanced.md) | 5 | ✅ 已完成(已验证) |
| 04 | 类型注解、上下文管理与并发 | [04-typing-context-and-concurrency.md](04-typing-context-and-concurrency.md) | 5 | ✅ 已完成(已验证) |

**合计:20 个知识点。**

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

*更新:2026-07-07*
