# Pythonic 写法逐点精讲 —— 路线图与进度表

> 目标:26 个 Pythonic 写法/惯用法知识点,由浅入深,分批次完成。
> 背景:课堂教的是"能跑就行"的写法,但地道 Python(pythonic)有一套自己的表达习惯——推导式、解包、EAFP、one-liner 的取舍——这些东西没人系统教过,但博士学长的代码里到处都在用。
> 定位:和 [python-advanced/](../python-advanced/00-roadmap.md) 是姊妹篇,那边补"语言特性"(装饰器/生成器/OOP/类型注解/并发),这里补"表达习惯"(用什么写法更 pythonic、什么时候一行更清晰、什么时候是炫技)。两者不重复,交叉引用的地方会明确标注。
> 格式沿用 python-advanced 系列的模板(见下),不新造格式。

---

## 每个知识点的固定讲解结构

1. **是什么**——一句话
2. **为什么课堂不教但很重要**——具体关联到 AI 研究代码里的哪类写法
3. **从最笨的写法讲起**——先给出不用这个写法的"笨办法",再引入 pythonic 写法,讲清楚它解决了什么问题
4. **AI 研究代码里的真实例子**——优先挖仓库里博士学长自己写的代码(不是 vendor 进来的第三方仓库);挖不到真实例子的知识点会如实标注"示例性质",不编造
5. **可运行例子**——能直接跑,关键结论用 assert/print 验证
6. **常见坑**

---

## 进度表(由浅入深)

| # | 分类 | 文件 | 知识点数 | 状态 |
|---|------|------|---------|------|
| 01 | 推导式与函数式基础 | [01-comprehensions-and-functional.md](01-comprehensions-and-functional.md) | 7 | ✅ 已完成(已验证,44个代码块独立子进程复验全部通过) |
| 02 | 解包与迭代惯用法 | [02-unpacking-and-iteration.md](02-unpacking-and-iteration.md) | 7 | ✅ 已完成(已验证,53个代码块独立子进程复验全部通过) |
| 03 | 容器与标准库惯用法 | [03-containers-and-stdlib.md](03-containers-and-stdlib.md) | 7 | ✅ 已完成(已验证,含1处多级排序手算修正+6处非连续引用改text围栏) |
| 04 | 字符串与现代语法惯用法 | [04-strings-and-modern-syntax.md](04-strings-and-modern-syntax.md) | 5 | ✅ 已完成(已验证,含1处match-case编译期陷阱发现) |

**合计:26 个知识点,4 篇全部完成并独立验证。**

---

## 每一批具体覆盖哪些知识点(明细)

### 01 推导式与函数式基础
1. 列表推导式语法与嵌套(常见坑吸收"条件表达式:filter 部分 vs 三元表达式部分"这一常见混淆点)
2. 字典推导式 / 集合推导式
3. `map`/`filter` vs 推导式——该用哪个
4. `functools.reduce`——什么时候真的需要(以及为什么大部分场景不推荐)
5. 三元表达式 `x if cond else y`
6. 海象运算符 `:=`(推导式/循环条件里怎么用,避免重复计算)
7. **one-liner 的取舍**——什么时候更清晰、什么时候是炫技,综合前 6 项做正反例对照

> 交叉引用:[python-advanced/02-iterators-and-generators.md](../python-advanced/02-iterators-and-generators.md) 已讲过"生成器表达式 vs 列表推导"的惰性求值/内存对比,本系列第 1 点只讲推导式**语法本身**,不重复那部分。

### 02 解包与迭代惯用法
1. 序列解包与多重赋值(`a, b = b, a` 交换原理)
2. 星号解包(`a, *rest = lst`;调用时 `*args`/`**kwargs` 解包与定义时的区别)
3. `enumerate`——为什么不要手写计数器
4. `zip`(含 `zip(*matrix)` 转置技巧、`strict=True`)
5. `itertools` 核心工具:`combinations`/`chain`/`pairwise`
6. `any`/`all` 与短路求值
7. 链式比较 `a < b < c`

### 03 容器与标准库惯用法
1. `collections.Counter`
2. `collections.defaultdict`
3. `collections.namedtuple`(与 dataclass 的取舍)
4. `collections.deque`
5. `dict.get`/`setdefault` vs 手写 if-else
6. EAFP vs LBYL 哲学
7. `sorted`/`.sort()` 的 `key=` 参数(`operator.itemgetter`/`attrgetter`、多级排序)

### 04 字符串与现代语法惯用法
1. f-string 高级用法
2. `str.join`(以及为什么不要在循环里用 `+` 拼接字符串)
3. `str.split`/`strip`/`partition` 系列
4. `pathlib.Path` 面向对象路径操作
5. `match`-`case` 结构化模式匹配(3.10+)

---

*更新:2026-07-08*
