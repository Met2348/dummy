# numpy 逐函数精讲 —— 路线图与进度表

> 目标:至少 100 个 numpy 函数,面向 AI 科研场景,由浅入深,分批次完成(约 120 个)。
> 定位:这是"精读一遍建立认知"的参考资料,不是要求背下来——别忘了 [03-how-to-look-up-not-memorize.md](../03-how-to-look-up-not-memorize.md) 的心态。这份文档和 03 不冲突:03 教你日常怎么查、怎么记核心的十几个;这里是把 AI 研究会遇到的 numpy 面系统性地过一遍,建立"看到就认得出、知道去哪查"的识别感。
> 完成后,同一套模式会延伸到 torch(衔接 [02-pytorch-basics.md](../02-pytorch-basics.md))。

---

## 每个函数的固定讲解结构

1. **签名** —— 参数人话翻译,不是抄文档
2. **一句话** —— 这个函数是干什么的
3. **AI 研究场景** —— 具体在论文/研究代码里怎么用、为什么必须用它
4. **可运行例子** —— 带 `assert` 验证,而且都实际跑过,不是凭空写的
5. **常见坑** —— 易混淆点/踩雷点

---

## 进度表(由浅入深)

| # | 分类 | 文件 | 函数数(约) | 状态 |
|---|------|------|-----------|------|
| 01 | 创建与初始化 | [01-creation-and-init.md](01-creation-and-init.md) | 15 | ✅ 已完成(已验证) |
| 02 | 形状与结构操作 | [02-shape-and-structure.md](02-shape-and-structure.md) | 15 | ✅ 已完成(已验证;2026-07-24 P/T/L/V/D 可读性审计通过,在第1节 reshape 处补充 `.strides`/内存布局图示——此前"reshape 是 view 还是 copy 取决于内存是否连续"只给结论没给机制,现在用字节级 strides 计算+转置后 reshape 必须复制的跳跃顺序,把结论变成可推导的) |
| 03 | 索引与选择 | [03-indexing-and-selection.md](03-indexing-and-selection.md) | 12 | ✅ 已完成(已验证) |
| 04 | 数学与逐元素运算 | [04-elementwise-math.md](04-elementwise-math.md) | 15 | ✅ 已完成(已验证) |
| 05 | 归约与统计 | [05-reduction-and-statistics.md](05-reduction-and-statistics.md) | 16 | ✅ 已完成(已验证) |
| 06 | 线性代数 | [06-linear-algebra.md](06-linear-algebra.md) | 17 | ✅ 已完成(已验证) |
| 07 | 排序与集合运算 | [07-sorting-and-set-ops.md](07-sorting-and-set-ops.md) | 8 | ✅ 已完成(已验证) |
| 08 | 广播与 ufunc 机制 | [08-broadcasting-and-ufunc.md](08-broadcasting-and-ufunc.md) | 6 | ✅ 已完成(已验证) |
| 09 | 随机数进阶与可复现 | [09-advanced-random.md](09-advanced-random.md) | 6 | ✅ 已完成(已验证) |
| 10 | IO 与验证工具 | [10-io-and-verification.md](10-io-and-verification.md) | 10 | ✅ 已完成(已验证;2026-07-24 P/T/L/V/D 审计给 `.view()` 补了一句 C 的 union/指针强转类比) |
| 11 | 进阶深度追加:3 个多级追问链案例 | [11-advanced-interview-depth.md](11-advanced-interview-depth.md) | 3案例(不计入约120) | ✅ 已完成(已验证,6/6代码块独立进程复验全部通过,含timing/GIL/多进程三类环境敏感代码块的重复运行确认;基于dsa-deep-dive/python-idioms已验证的5条追问轴线撰写,诚实收敛到3个案例(numpy系列5步模板没有"底层机制"这一步,材料相对最薄弱)——①np.vectorize假向量化(真实性验证轴+工程约束递增轴核心;跨20万/200万两个数量级验证差距不被摊薄(16-19倍持平),再用threading实测GIL释放情况:纯ufunc链路双线程加速1.5-1.7倍,vectorize链路仅1.1倍,证明它连"被多线程救一下"的退路都没有)、②视图共享内存导致的累积数据腐蚀(规模递增轴核心;cutout数据增强就地改写切片视图,小测试只断言返回值全绿,6个epoch后母数据集被累积清零15%→90%,用shares_memory/.base证明因果,再实测"无脑copy整个数据集"比"只copy当前batch"贵近2倍,揭示拷贝粒度这一决策依据)、③default_rng跨进程种子管理(规模递增轴+工程约束递增轴核心;09类原文从未真正跨进程验证过"独立Generator对象"的假设,这里用真实multiprocessing.Pool实测4个worker同一seed产出完全相同的"随机"数据,再用SeedSequence.spawn()层级展开"2机器x2worker"结构验证4条流互不相同且整体可复现)) |

**合计:约 120 个函数,10 篇 + 1 篇进阶深度追加(3 个案例,不计入约 120),全部完成并独立验证。**

---

## 每一批具体覆盖哪些函数(明细)

### 01 创建与初始化
`array` `zeros`/`ones` `full` `empty` `zeros_like`/`ones_like` `arange` `linspace` `eye`/`identity` `random.seed` `random.randn` `random.normal` `random.uniform` `random.randint` `random.choice` `meshgrid`

### 02 形状与结构操作
`.shape`/`.reshape` `.T`/`transpose` `swapaxes`/`moveaxis` `.flatten`/`.ravel` `squeeze` `expand_dims`/`newaxis` `concatenate` `stack` `hstack`/`vstack` `split`/`array_split` `tile` `repeat` `pad` `broadcast_to` `flip`/`fliplr`/`flipud`

### 03 索引与选择
基础切片 布尔索引 花式索引 `where`(选择器用法) `take` `take_along_axis` `argwhere` `nonzero` `isin` `diag`/`diagonal` `triu`/`tril` `select`

### 04 数学与逐元素运算
四则运算的 ufunc 本质 `exp`/`log` `log2`/`log1p` `sqrt`/`power` `abs` `sign` `round`/`floor`/`ceil` `maximum`/`minimum` `clip` `nan_to_num` `sin`/`cos` `mod`/`remainder` 比较运算符 `logical_and`/`or`/`not` `where`(三元表达式用法)

### 05 归约与统计
`sum` `mean` `std`/`var` `max`/`min` `argmax`/`argmin` `median` `percentile`/`quantile` `cumsum`/`cumprod` `all`/`any` `count_nonzero` `unique` `bincount` `histogram` `average` `ptp` `corrcoef`/`cov`

### 06 线性代数
`dot` `@`/`matmul` `outer`/`inner` `linalg.norm` `linalg.inv` `linalg.pinv` `linalg.det` `linalg.solve` `linalg.eig` `linalg.eigh` `linalg.svd` `linalg.qr` `linalg.cholesky` `trace` `einsum` `kron` `linalg.matrix_rank`

### 07 排序与集合运算
`sort` `argsort` `partition`/`argpartition` `intersect1d` `union1d` `setdiff1d` `in1d` `searchsorted`

### 08 广播与 ufunc 机制
广播规则专题 `vectorize` `apply_along_axis` `apply_over_axes` ufunc 的 `.reduce`/`.accumulate`

### 09 随机数进阶与可复现
`random.shuffle` `random.permutation` `random.default_rng`(新 Generator API) `rng.random`/`rng.integers` `random.binomial`/`random.poisson` 训练/验证集划分实践

### 10 IO 与验证工具
`save`/`load` `savez` `savetxt`/`loadtxt` `allclose`/`isclose` `array_equal`/`array_equiv` `.copy()` vs 视图(view) `.astype` `errstate` `may_share_memory` `set_printoptions`

---

*更新:2026-07-13(新增 11 进阶深度追加)*
