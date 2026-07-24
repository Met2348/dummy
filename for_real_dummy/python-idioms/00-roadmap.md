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
| 01 | 推导式与函数式基础 | [01-comprehensions-and-functional.md](01-comprehensions-and-functional.md) | 7 | ✅ 已完成(已验证,44个代码块独立子进程复验全部通过;2026-07-24 P/T/L/V/D 可读性审计:通读未发现需修复的 gap,未改动) |
| 02 | 解包与迭代惯用法 | [02-unpacking-and-iteration.md](02-unpacking-and-iteration.md) | 7 | ✅ 已完成(已验证,53个代码块独立子进程复验全部通过;2026-07-24 P/T/L/V/D 可读性审计:通读未发现需修复的 gap,未改动) |
| 03 | 容器与标准库惯用法 | [03-containers-and-stdlib.md](03-containers-and-stdlib.md) | 7 | ✅ 已完成(已验证,含1处多级排序手算修正+6处非连续引用改text围栏;2026-07-24 P/T/L/V/D 可读性审计:通读未发现需修复的 gap,未改动) |
| 04 | 字符串与现代语法惯用法 | [04-strings-and-modern-syntax.md](04-strings-and-modern-syntax.md) | 5 | ✅ 已完成(已验证,含1处match-case编译期陷阱发现;2026-07-24 P/T/L/V/D 可读性审计:通读未发现需修复的 gap,未改动) |
| 05 | 进阶深度追加:5 个多级追问链案例 | [05-advanced-interview-depth.md](05-advanced-interview-depth.md) | 5案例(不计入26) | ✅ 已完成(已验证,10/10代码块独立进程复验全部通过;基于dsa-deep-dive/peft-deep-dive已验证的5条追问轴线撰写——①EAFP vs LBYL到TOCTOU竞态真实复现(工程约束递增轴核心;用threading.Event精确卡时机分别复现文件版和dict版竞态,LBYL检查通过后被并发删除真实崩溃,EAFP始终优雅处理)、②推导式vs生成器表达式的规模与复用取舍(规模递增轴核心;tracemalloc实测250,000行日志场景list比生成器多耗398.7倍峰值内存,并验证生成器二次消费静默返回空列表这一决策依据)、③namedtuple vs dataclass"更省内存"结论的前提(决策依据追问轴核心;现场证伪单实例sys.getsizeof朴素外推的2.25倍预期,200,000条记录实测聚合差距仅13.7%,溯源到CPython key-sharing dict优化(PEP 412)并现场构造"打破key-sharing"对照组验证内存反弹2.98倍,另附dataclass(slots=True)反超namedtuple 9.1%、可哈希性与可变性互斥两条新发现)、④"pythonic化性能声称"诚实拆解(真实性验证轴核心;timeit分环节实测deque/join两处为12.7x/23.7x数量级差异,推导式/defaultdict两处仅1.11x/1.33x基本不变,证明"pythonic不等于fast")、⑤sorted key=多级排序到functools.cmp_to_key(方案批判迭代轴核心;现场构造两趟排序顺序搞反的具体错误结果、验证key=函数签名无法表达头对头这类成对关系、实测cmp_to_key比key=慢7.3倍的真实代价));2026-07-24 P/T/L/V/D 可读性审计:通读未发现需修复的 gap,未改动——进阶案例定位本就假设读者已掌握 01-04,信息密度符合该定位 |
| 06 | 手把手实战:从零搭一个迷你日志分析 CLI | [06-build-a-mini-log-analyzer-cli.md](06-build-a-mini-log-analyzer-cli.md) | 4阶段(不计入26) | ✅ 已完成(已验证,8/8代码块独立子进程复验全部通过;仿照 dsa-deep-dive/21 的教程体格式试点推广——串联 03类EAFP vs LBYL哲学→单行日志解析、01类列表推导式+python-advanced/02类生成器表达式→批量清洗管道、02类提到但未展开的itertools.groupby+03类Counter/defaultdict→按级别分组统计,四阶段组装成一个可持续ingest()的`MiniLogAnalyzer`;阶段1现场实测LBYL版本因校验函数内部仍需try/except导致同一日期被解析2次,EAFP版本只解析1次;阶段3现场复现`itertools.groupby`真实坑——对着按时间顺序(未排序)的15条记录直接分组,产出12个破碎小组而不是3个,而不是回避这个反直觉结果,而是解释清楚"groupby只认连续key"这个根因,再用sorted()+groupby、Counter、defaultdict三条独立路径互相印证得到一致的正确统计(INFO 8/WARNING 3/ERROR 4)) |

**合计:26 个知识点,4 篇 + 1 篇进阶深度追加(5 个案例,不计入 26)+ 1 篇教程体试点(4 阶段,不计入 26),全部完成并独立验证。**

**关于 06 类的方法论说明:** dsa-deep-dive 系列试点了"教程体"这一新内容形态(见 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md)"关于 21 类的方法论说明"):19/20 类是"读者旁观、跟着叙事看一遍推理链条",21 类要的是"读者动手、从空文件开始一步步敲代码搭出一个东西"。这一试点被要求推广到其余系列,06 类是 python-idioms 这一条的落地——挑选和"表达习惯"关系最近的三类知识点(EAFP 哲学、推导式/生成器表达式、容器标准库惯用法)串成一个迷你日志分析器。**06 类不计入本系列"26 个知识点"的统计**,和 05 类一样是正式知识点之外的追加内容,只是风格从"叙事追问"换成了"从零动手搭建"。撰写过程中额外发现:[02 类第 5.3 节](02-unpacking-and-iteration.md)此前只是在"相关工具一览"里提了一句 `itertools.groupby` 的存在、没有真实用例——06 类第一次真正用它,顺带验证了它"只认连续 key"这个容易被忽视的行为,补上了之前系列里一直缺失的真实使用场景。

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

*更新:2026-07-13(新增 05 进阶深度追加)*
