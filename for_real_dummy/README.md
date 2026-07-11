# For Real Dummy — 我的学习笔记库

> 这里记录我(大二本科)学习这个博士级 AI/ML 仓库过程中的真实问题和收获。
> 没有傻问题,只有没搞懂的知识点。

## 我的背景(2026-07-02 更新)

| 项目 | 情况 |
|------|------|
| 编程经验 | C(有基础),Python(刚学完基本语法,无中高级) |
| 数学基础 | 高等数学 ✅ 线性代数 ✅ 概率论 ✅ 随机过程 ❌(未学) |
| PyTorch/DL | 零基础 |
| 目标 | 读懂并使用这个博士学长留下的 AI/ML 研究仓库 |

## 我需要补的核心技能(路线图)

```
现在的我                         需要到达的地方
─────────────────────────────────────────────────
C 基础                 →  理解内存/性能(已有,有帮助)
Python 基本语法        →  中级 Python(类、装饰器、numpy)
                       →  PyTorch 基础
数学三件套             →  理解 ML 中的数学(梯度、矩阵运算)
                       →  不需要随机过程也能看很多论文
```

## 入门教程(按顺序读)

| 文件 | 内容 | 前提 |
|------|------|------|
| [01-numpy-for-c-programmers.md](01-numpy-for-c-programmers.md) | numpy 入门,C 视角讲解 | 会 C 即可 |
| [02-pytorch-basics.md](02-pytorch-basics.md) | PyTorch Tensor、autograd、nn.Module、训练循环 | 看完 01 |
| [03-how-to-look-up-not-memorize.md](03-how-to-look-up-not-memorize.md) | 别死记函数签名——核心记忆清单 + 意图索引表 + 常见地雷 + 查阅技巧 | 看完 01、02 |
| [04-how-to-practice-with-jupyter.md](04-how-to-practice-with-jupyter.md) | 怎么在 VSCode 里用 Jupyter 跑代码、用 assert 验证结果、把练习存成笔记 | 看完 01、02 |

## 深挖系列(逐函数 / 逐知识点,由浅入深)

入门教程看完之后,这五条系列负责"系统性地过一遍"——不是要求背下来,是精读一遍建立"看到就认得出、知道去哪查"的识别感(呼应 [03](03-how-to-look-up-not-memorize.md) 的心态)。五条系列互不依赖,可以交替看,但建议顺序是 numpy → python-advanced → torch → tensorflow(torch 系列默认你已经具备前两条系列建立的心智模型;tensorflow 系列和 torch 是"同一套心智模型,两个框架"的关系,建议紧接着 torch 看,概念相通的地方两边会交叉引用);python-idioms 是 python-advanced 的姊妹篇(那边讲语言特性,这边讲表达习惯),看完 python-advanced 之后接着看比较顺,和框架系列互不冲突,随时插进去看也可以。

| 系列 | 内容 | 规模 | 状态 |
|------|------|------|------|
| [numpy-deep-dive/](numpy-deep-dive/00-roadmap.md) | AI 科研场景里会用到的 numpy 函数,10 个分类(创建初始化→形状结构→索引选择→逐元素数学→归约统计→线性代数→排序集合→广播ufunc机制→随机数进阶→IO验证工具) | 约 120 个函数 | ✅ 全部完成并验证 |
| [python-advanced/](python-advanced/00-roadmap.md) | 课堂没讲、但仓库代码里到处都是的 Python 中高级语法(装饰器/闭包/生成器/OOP进阶/类型注解/async 并发等),4 个分类 | 20 个知识点 | ✅ 全部完成并验证 |
| [torch-deep-dive/](torch-deep-dive/00-roadmap.md) | torch 独有、面试重灾区的底层机制:tensor内存模型→autograd→nn.Module内核→层的数学推导→损失函数→优化器→训练循环(混合精度/梯度累加/checkpoint)→内存性能→分布式→序列化部署→调试报错精解,11 个分类。比前两个系列深度更高,每个知识点都讲到"为什么这样设计"+"面试追问链",目标是扛住大厂技术面试二三面的深挖 | 100 个知识点 | ✅ 全部完成并验证 |
| [tensorflow-deep-dive/](tensorflow-deep-dive/00-roadmap.md) | torch-deep-dive 的 TensorFlow 版,同等深度对标:tensor基础→GradientTape自动微分→tf.function/AutoGraph计算图(TF独有的两大重中之重)→Keras三套API内核→层的数学推导→损失函数→优化器→fit()内核与自定义训练循环→tf.data输入管道(TF独有)→内存性能→分布式→序列化部署→调试报错精解,13 个分类。运行环境为 WSL2(Windows 原生不支持 GPU) | 100 个知识点 | ✅ 全部完成并验证 |
| [python-idioms/](python-idioms/00-roadmap.md) | python-advanced 的姊妹篇,讲"表达习惯"而不是语言特性:推导式/解包/itertools/容器与标准库惯用法/EAFP哲学/字符串与现代语法,4 个分类,收尾一节专门讲 one-liner 的取舍(什么时候一行更清晰、什么时候是炫技) | 26 个知识点 | ✅ 全部完成并验证 |

每个函数/知识点都固定同一套结构(torch/tensorflow 两个框架系列额外多两块:底层机制/为什么这样设计、面试怎么问+追问链):签名(人话翻译)→ 一句话是什么 → AI 研究场景(具体用在哪) → 可运行例子(带 assert,真的在仓库 `.venv` 里跑过) → 常见坑。五条系列合计约 366 个知识点,均已验证完成。

## 自维护工具

- [my-cheatsheet.md](my-cheatsheet.md) —— 你自己的速查表。规则:同一个函数查了 3 次文档,就记一条进去。自己写的记得最牢。
- [practice/](practice/) —— 你自己动手跑的练习 notebook,一个练习一个 `.ipynb` 文件,代码 + assert 验证为主。从 [practice/00-getting-started.ipynb](practice/00-getting-started.ipynb) 开始。

## 目录结构

```
for_real_dummy/
├── README.md                          ← 这个文件,总览 + 路线图
├── roadmap.md                         ← 详细学习路线和优先级
├── 01-numpy-for-c-programmers.md      ← numpy 入门教程
├── 02-pytorch-basics.md               ← PyTorch 入门教程
├── 03-how-to-look-up-not-memorize.md  ← 查阅策略 + 意图索引表 + 常见地雷
├── 04-how-to-practice-with-jupyter.md ← Jupyter 怎么用 + 怎么验证结果 + 怎么留笔记
├── numpy-deep-dive/                   ← numpy 逐函数精讲系列(约120个函数,10批)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表
│   └── 01~10-*.md                     ← 每批一个文件
├── python-advanced/                   ← Python 中高级语法补完系列(20个知识点,4批)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表
│   └── 01~04-*.md                     ← 每批一个文件
├── torch-deep-dive/                   ← torch逐机制精讲系列(100个知识点,面试深度,11批)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表
│   └── 01~11-*.md                     ← 每批一个文件
├── tensorflow-deep-dive/              ← TF逐机制精讲系列(100个知识点,面试深度,13批,WSL2 GPU环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 环境声明
│   └── 01~13-*.md                     ← 每批一个文件
├── python-idioms/                     ← Pythonic写法惯用法系列(26个知识点,4批,python-advanced姊妹篇)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表
│   └── 01~04-*.md                     ← 每批一个文件
├── my-cheatsheet.md                   ← 你自己维护的速查表(持续增长)
├── practice/                          ← 你自己动手写代码验证的 notebook
│   └── 00-getting-started.ipynb       ← 现成的起步文件,含环境自检 + 练习模板
└── qa/                                ← 每次有价值的问答,按主题归档
    └── ...(随问随加)
```

## 使用方式

每次问了有价值的问题,Claude 会把问题和答案整理到 `qa/` 对应的文件里。

---

*最后更新:2026-07-11*
