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

入门教程看完之后,这六条系列负责"系统性地过一遍"——不是要求背下来,是精读一遍建立"看到就认得出、知道去哪查"的识别感(呼应 [03](03-how-to-look-up-not-memorize.md) 的心态)。六条系列互不依赖,可以交替看,但建议顺序是 numpy → python-advanced → torch → huggingface-deep-dive(依赖 torch 系列建立的框架内核心智模型,是从"理解PyTorch"到"能实际用HuggingFace生态做微调"的下一步)→ tensorflow(和 torch 是"同一套心智模型,两个框架"的关系,概念相通的地方两边会交叉引用);python-idioms 是 python-advanced 的姊妹篇(那边讲语言特性,这边讲表达习惯),看完 python-advanced 之后接着看比较顺,和框架系列互不冲突,随时插进去看也可以。

| 系列 | 内容 | 规模 | 状态 |
|------|------|------|------|
| [numpy-deep-dive/](numpy-deep-dive/00-roadmap.md) | AI 科研场景里会用到的 numpy 函数,10 个分类(创建初始化→形状结构→索引选择→逐元素数学→归约统计→线性代数→排序集合→广播ufunc机制→随机数进阶→IO验证工具) | 约 120 个函数 | ✅ 全部完成并验证 |
| [python-advanced/](python-advanced/00-roadmap.md) | 课堂没讲、但仓库代码里到处都是的 Python 中高级语法(装饰器/闭包/生成器/OOP进阶/类型注解/async 并发等),4 个分类 | 20 个知识点 | ✅ 全部完成并验证 |
| [torch-deep-dive/](torch-deep-dive/00-roadmap.md) | torch 独有、面试重灾区的底层机制:tensor内存模型→autograd→nn.Module内核→层的数学推导→损失函数→优化器→训练循环(混合精度/梯度累加/checkpoint)→内存性能→分布式→序列化部署→调试报错精解,11 个分类。比前两个系列深度更高,每个知识点都讲到"为什么这样设计"+"面试追问链",目标是扛住大厂技术面试二三面的深挖 | 100 个知识点 | ✅ 全部完成并验证 |
| [huggingface-deep-dive/](huggingface-deep-dive/00-roadmap.md) | 建在 torch 之上的 HuggingFace 生态工程内核(不是算法论文):tokenizer机制→模型加载与AutoClass→pipeline→datasets库→Trainer内核→accelerate分布式→PEFT库工程机制→量化机制bitsandbytes→**微调实战对比(全参/LoRA/QLoRA真实端到端训练闭环)**→TRL训练器抽象→Hub与模型分享→推理优化→调试报错精解,13 个分类。验证环境为仓库根目录 `.venv`(Windows原生,PyTorch原生支持CUDA不需要WSL2),真实跑通多组微调训练对比(核心发现:QLoRA显存最省但训练耗时反而最长) | 101 个知识点 | ✅ 全部完成并验证 |
| [tensorflow-deep-dive/](tensorflow-deep-dive/00-roadmap.md) | torch-deep-dive 的 TensorFlow 版,同等深度对标:tensor基础→GradientTape自动微分→tf.function/AutoGraph计算图(TF独有的两大重中之重)→Keras三套API内核→层的数学推导→损失函数→优化器→fit()内核与自定义训练循环→tf.data输入管道(TF独有)→内存性能→分布式→序列化部署→调试报错精解,13 个分类。运行环境为 WSL2(Windows 原生不支持 GPU) | 100 个知识点 | ✅ 全部完成并验证 |
| [python-idioms/](python-idioms/00-roadmap.md) | python-advanced 的姊妹篇,讲"表达习惯"而不是语言特性:推导式/解包/itertools/容器与标准库惯用法/EAFP哲学/字符串与现代语法,4 个分类,收尾一节专门讲 one-liner 的取舍(什么时候一行更清晰、什么时候是炫技) | 26 个知识点 | ✅ 全部完成并验证 |

每个函数/知识点都固定同一套结构(torch/tensorflow/huggingface 三个框架/生态系列额外多两块:底层机制/为什么这样设计、面试怎么问+追问链):签名(人话翻译)→ 一句话是什么 → AI 研究场景(具体用在哪) → 可运行例子(带 assert,真的在仓库 `.venv` 里跑过) → 常见坑。六条系列合计约 467 个知识点,均已验证完成。

## 独立技能系列(不是 Python/ML,完全独立的新领域)

| 系列 | 内容 | 规模 | 状态 |
|------|------|------|------|
| [rhcsa-bash-deep-dive/](rhcsa-bash-deep-dive/00-roadmap.md) | Linux 系统管理 + bash 脚本编程,对标 Red Hat 官方 RHCSA(EX200,RHEL 10 基准)认证范围,9 个分类:必备工具与文本处理→进程与系统运行→本地存储与LVM→文件系统与权限→软件与系统部署→用户组管理→网络配置→安全(SELinux/防火墙)→bash脚本编程本身。验证环境为 WSL2 Rocky Linux 10.2(真实 systemd + root),涉及改网络/权限的知识点全部在隔离沙箱(dummy网卡/loop device)里操作,不影响宿主环境 | 100 个知识点 | ✅ 全部完成并验证 |
| [dsa-deep-dive/](dsa-deep-dive/00-roadmap.md) | 数据结构与算法,对标技术终面 1 小时持续深挖难度,18 个分类:复杂度分析与Python内置容器内核→数组字符串技巧→链表→二分查找→栈队列(含LRU/LFU设计)→排序从零实现→堆/优先队列→树→回溯→DP基础→贪心算法→Trie与字符串匹配→位运算与数学→图论基础→图论进阶(最短路/MST/强连通分量/网络流/二分图匹配)→DP进阶(区间/状压/数位/树形/概率期望/博弈)→线段树与树状数组→面试方法论与代码规范,收尾 1 篇 [1小时模拟终面capstone](dsa-deep-dive/19-mock-interview-capstone.md)(不是知识点列表,是完整还原终面节奏的场景叙事),再收尾 1 篇 [进阶深度追加](dsa-deep-dive/20-advanced-interview-depth.md)(基于真实调研2026大厂二面追问模式撰写的5个多级追问链案例:LRU并发/分布式、TopK海量数据、限流方案批判迭代、日志诊断真实系统行为、项目真实性验证追问)。验证环境为仓库根目录 `.venv`,纯 Python 标准库,不需要任何第三方包 | 140 个知识点 + 1 篇 capstone + 1 篇进阶深度追加 | ✅ 全部完成并验证 |

rhcsa-bash-deep-dive 每个知识点固定七步结构(命令/配置 → 一句话是什么 → 为什么RHCSA真考/生产会用到 → 从最容易犯错的做法讲起 → 真实场景例子 → 可运行例子 → 常见坑),不采用 torch/tensorflow/huggingface 系列"面试怎么问"环节(RHCSA 是纯上机操作考试)。和"深挖系列"表格里六条系列的关键差异:本仓库没有 Linux 系统管理场景可挖,"真实场景例子"如实标注为典型运维/RHCSA 考试场景,不冒充仓库代码里挖出来的;部分知识点受 WSL2 内核结构性限制(GRUB引导链路不存在、dm-vdo模块缺失、selinuxfs接口残缺)无法完整验证真实效果,均已在对应小节诚实标注验证颗粒度,不冒充"已完整验证"。

dsa-deep-dive 的知识点结构和"深挖系列"六条完全一致的七步(签名/是什么→一句话→底层机制/为什么这样设计→AI研究/工程场景→可运行例子→面试怎么问+追问链→常见坑),"AI研究/工程场景"步骤改写为"这个技巧在真实系统里出现在哪",优先引用 torch-deep-dive/huggingface-deep-dive 已验证内容(如 Trie→tokenizer 的 BPE 训练、堆→beam search/top-k 采样、并查集→分布式一致性),没有真实关联的如实引用通用工程场景。本系列独有的方法论:复杂度不是断言出来的,是真的用 `time.perf_counter()` 在多组输入规模上测出来的(容差足够宽,只验证增长趋势的量级方向,不追求精确复现理论系数),这条纪律和 huggingface-deep-dive 系列"显存测量必须交叉核对 `torch.cuda.max_memory_allocated()` 和 `nvidia-smi`"同源,只是验证对象从"库的真实行为"换成了"算法的真实复杂度表现"。

## 专题精读系列(直接对应 `learning/` 下具体专题模块的精读伴读笔记)

和上面"深挖系列"的关键差异:上面六条补的是"读懂 `learning/` 代码需要的通用框架/语言技能"(numpy/torch/HF 库机制本身,不绑定具体研究专题);这里每一条都**直接对应 `learning/` 下一个具体专题模块**,讲的是**同一份代码**,只是换成更适合初学者/面试备考的讲解深度(7 步结构,从最笨的想法讲起,面试怎么问+追问链),不是重复造轮子——定位类似"这份 PhD 级代码,配一份大二学生看得懂、还能扛住面试的精读笔记"。

| 系列 | 对应 `learning/` 专题 | 内容 | 规模 | 状态 |
|------|---------------------|------|------|------|
| [long-context-deep-dive/](long-context-deep-dive/00-roadmap.md) | [`learning/long-context/`](../learning/long-context/) | RoPE 外推家族(vanilla→PI→NTK→YaRN→3D-RoPE)→ 长上下文 Attention 架构(Ring/Striped/Infini-Attention)→ 长上下文评测方法论(NIAH/RULER/Lost-in-the-Middle)→ 数据工程与 Capstone(文档打包/课程学习/YaRN+LoRA capstone/KV-cache 显存核算),4 个分类。全系列纯 CPU、零 GPU 依赖 | 17 个知识点 | ✅ 全部完成并验证 |
| [kernel-gpu-deep-dive/](kernel-gpu-deep-dive/00-roadmap.md) | [`learning/gpu-architecture/`](../learning/gpu-architecture/) + [`learning/kernel-engineering/`](../learning/kernel-engineering/) | GPU 硬件与存储层次(规格表/tier推荐/Tensor Core/SM Occupancy/NVLink)→ Roofline 性能建模(arithmetic intensity/ridge point/反直觉发现)→ Kernel 设计语言(Triton autotune/CUTLASS layout代数/swizzle stub方法论)→ FlashAttention与Kernel Fusion(online softmax/HBM流量核算/128k capstone),4 个分类。全系列纯 CPU、零第三方依赖 | 19 个知识点 | ✅ 全部完成并验证 |
| [alignment-algorithms-deep-dive/](alignment-algorithms-deep-dive/00-roadmap.md) | [`learning/dpo-family/`](../learning/dpo-family/) | DPO 基础与推导(RLHF闭式解→Bradley-Terry代换→loss逐项对照→真训练脚本→零margin=log2边界测试)→ PO 变体家族(IPO/KTO/ORPO/SimPO/CPO/DPOP六变体+8算法横向对比表)→ RainbowPO 统一视角与 Capstone(4轴统一框架的真实验证成色→6变体50步benchmark→zero trl import架构选择),3 个分类。全系列纯 CPU(`dpo_minimal.py` 真训练部分标注为可选进阶验证)。**亮点**:独立复验发现 `rainbowpo.py::unified_po_loss` 只精确复现 `dpo` 一个配置——`dpop`/`kto` 配置字段和 `dpo` 逐字段相同、静默退化成纯 DPO,其余变体均与各自独立实现存在结构性数值偏差 | 15 个知识点 | ✅ 全部完成并验证 |

每个知识点固定七步结构(与 torch/tensorflow/huggingface 系列完全一致:签名/是什么→一句话→**底层机制/为什么这样设计**→AI研究场景→可运行例子(assert验证)→**面试怎么问+追问链**→常见坑)。这套模板比"深挖系列"的轻量六步(numpy/python-advanced/python-idioms)多两块,因为这里讲的是算法/框架机制而非通用语言技能。

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
├── huggingface-deep-dive/             ← HuggingFace生态工程内核系列(101个知识点,面试深度,13批,含真实多组微调训练对比)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 环境声明 + 模型数据集选型
│   ├── 01~13-*.md                     ← 每批一个文件
│   ├── 09-lab-artifacts/              ← 09类微调对比的真实训练产物(loss曲线/显存数字等JSON记录)
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本
├── tensorflow-deep-dive/              ← TF逐机制精讲系列(100个知识点,面试深度,13批,WSL2 GPU环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 环境声明
│   └── 01~13-*.md                     ← 每批一个文件
├── python-idioms/                     ← Pythonic写法惯用法系列(26个知识点,4批,python-advanced姊妹篇)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表
│   └── 01~04-*.md                     ← 每批一个文件
├── rhcsa-bash-deep-dive/              ← Linux系统管理+bash脚本系列(100个知识点,对标RHCSA/EX200,9批,WSL2 Rocky Linux环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 环境声明
│   └── 01~09-*.md                     ← 每批一个文件
├── dsa-deep-dive/                     ← 数据结构与算法系列(140个知识点+1篇capstone+1篇进阶深度追加,面试深度,18批,纯Python标准库环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 复杂度验证方法论 + 经典题单映射表附录
│   ├── 01~18-*.md                     ← 每批一个文件
│   ├── 19-mock-interview-capstone.md  ← 1小时模拟终面capstone(场景叙事,非知识点列表)
│   ├── 20-advanced-interview-depth.md ← 进阶深度追加(基于真实调研的5个多级追问链案例,非知识点列表)
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本
├── long-context-deep-dive/            ← 长上下文技术精读系列(17个知识点,面试深度,4批,对应learning/long-context/,纯CPU环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 差异化声明 + 环境声明
│   └── 01~04-*.md                     ← 每批一个文件
├── kernel-gpu-deep-dive/              ← GPU架构+Kernel工程精读系列(19个知识点,面试深度,4批,对应learning/gpu-architecture+kernel-engineering,纯CPU环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 差异化声明 + 环境声明
│   └── 01~04-*.md                     ← 每批一个文件
├── alignment-algorithms-deep-dive/    ← 偏好优化算法精读系列(15个知识点,面试深度,3批,对应learning/dpo-family,纯CPU环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 差异化声明 + 环境声明
│   └── 01~03-*.md                     ← 每批一个文件
├── my-cheatsheet.md                   ← 你自己维护的速查表(持续增长)
├── practice/                          ← 你自己动手写代码验证的 notebook
│   └── 00-getting-started.ipynb       ← 现成的起步文件,含环境自检 + 练习模板
└── qa/                                ← 每次有价值的问答,按主题归档
    └── ...(随问随加)
```

## 使用方式

每次问了有价值的问题,Claude 会把问题和答案整理到 `qa/` 对应的文件里。

---

*最后更新:2026-07-13*
