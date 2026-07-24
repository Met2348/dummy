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
| [numpy-deep-dive/](numpy-deep-dive/00-roadmap.md) | AI 科研场景里会用到的 numpy 函数,10 个分类(创建初始化→形状结构→索引选择→逐元素数学→归约统计→线性代数→排序集合→广播ufunc机制→随机数进阶→IO验证工具)+ 1 篇[进阶深度追加](numpy-deep-dive/11-advanced-interview-depth.md)(3 个案例,材料相对最薄弱、诚实收敛不强行凑 5 个)+ 1 篇[手把手教程体](numpy-deep-dive/12-build-a-mini-image-pipeline.md)(迷你图像处理管线:灰度化→卷积模糊→Sobel边缘检测,串联strides/广播/线性代数) | 约 120 个函数 | ✅ 全部完成并验证 |
| [python-advanced/](python-advanced/00-roadmap.md) | 课堂没讲、但仓库代码里到处都是的 Python 中高级语法(装饰器/闭包/生成器/OOP进阶/类型注解/async 并发等),4 个分类 + 1 篇[进阶深度追加](python-advanced/05-advanced-interview-depth.md)(5 个案例)+ 1 篇[手把手教程体](python-advanced/06-build-a-mini-event-bus.md)(迷你事件总线:装饰器注册回调+闭包保存状态+生成器惰性回放) | 20 个知识点 | ✅ 全部完成并验证 |
| [torch-deep-dive/](torch-deep-dive/00-roadmap.md) | torch 独有、面试重灾区的底层机制:tensor内存模型→autograd→nn.Module内核→层的数学推导→损失函数→优化器→训练循环(混合精度/梯度累加/checkpoint)→内存性能→分布式→序列化部署→调试报错精解,11 个分类 + 1 篇[进阶深度追加](torch-deep-dive/12-advanced-interview-depth.md)(5 个案例)+ 1 篇[手把手教程体](torch-deep-dive/13-build-a-mini-autograd-engine.md)(全系列唯一一处脱离torch独立重建的内容:从零搭一个迷你autograd引擎,不import torch,只在最后一步交叉验证)。比前两个系列深度更高,每个知识点都讲到"为什么这样设计"+"面试追问链",目标是扛住大厂技术面试二三面的深挖 | 100 个知识点 | ✅ 全部完成并验证 |
| [huggingface-deep-dive/](huggingface-deep-dive/00-roadmap.md) | 建在 torch 之上的 HuggingFace 生态工程内核(不是算法论文):tokenizer机制→模型加载与AutoClass→pipeline→datasets库→Trainer内核→accelerate分布式→PEFT库工程机制→量化机制bitsandbytes→**微调实战对比(全参/LoRA/QLoRA真实端到端训练闭环)**→TRL训练器抽象→Hub与模型分享→推理优化→调试报错精解,13 个分类 + 1 篇[进阶深度追加](huggingface-deep-dive/14-advanced-interview-depth.md)(5 个案例)+ 1 篇[手把手教程体](huggingface-deep-dive/15-build-a-mini-bpe-tokenizer.md)(从零训练一个迷你BPE tokenizer,训练+推理两阶段都做完整)。验证环境为仓库根目录 `.venv`(Windows原生,PyTorch原生支持CUDA不需要WSL2),真实跑通多组微调训练对比(核心发现:QLoRA显存最省但训练耗时反而最长) | 101 个知识点 | ✅ 全部完成并验证 |
| [tensorflow-deep-dive/](tensorflow-deep-dive/00-roadmap.md) | torch-deep-dive 的 TensorFlow 版,同等深度对标:tensor基础→GradientTape自动微分→tf.function/AutoGraph计算图(TF独有的两大重中之重)→Keras三套API内核→层的数学推导→损失函数→优化器→fit()内核与自定义训练循环→tf.data输入管道(TF独有)→内存性能→分布式→序列化部署→调试报错精解,13 个分类 + 1 篇[进阶深度追加](tensorflow-deep-dive/14-advanced-interview-depth.md)(5 个案例)+ 1 篇[手把手教程体](tensorflow-deep-dive/15-build-a-mini-custom-training-loop.md)(裸`tf.Variable`+GradientTape手写训练循环,不经Keras抽象,再回头对照`model.fit()`验证是同一件事)。运行环境为 WSL2(Windows 原生不支持 GPU) | 100 个知识点 | ✅ 全部完成并验证 |
| [python-idioms/](python-idioms/00-roadmap.md) | python-advanced 的姊妹篇,讲"表达习惯"而不是语言特性:推导式/解包/itertools/容器与标准库惯用法/EAFP哲学/字符串与现代语法,4 个分类,收尾一节专门讲 one-liner 的取舍(什么时候一行更清晰、什么时候是炫技)+ 1 篇[进阶深度追加](python-idioms/05-advanced-interview-depth.md)(5 个案例)+ 1 篇[手把手教程体](python-idioms/06-build-a-mini-log-analyzer-cli.md)(迷你日志分析CLI,现场发现`itertools.groupby`只合并连续key这个系列内此前从未演示过的坑) | 26 个知识点 | ✅ 全部完成并验证 |

每个函数/知识点都固定同一套结构(torch/tensorflow/huggingface 三个框架/生态系列额外多两块:底层机制/为什么这样设计、面试怎么问+追问链):签名(人话翻译)→ 一句话是什么 → AI 研究场景(具体用在哪) → 可运行例子(带 assert,真的在仓库 `.venv` 里跑过) → 常见坑。六条系列合计约 467 个知识点,均已验证完成。**2026-07-13 更新:六条系列现在各自都新增了一篇独立的"进阶深度追加"文件**(不计入上面的知识点统计),延续 dsa-deep-dive 率先验证的方法论——用户转达"现有材料没有达到 2026 年大厂技术二面深度"的反馈后,经过真实 WebSearch 调研(中国大厂面经/西方大厂面经/面试官视角元讨论)确认真实追问沿 5 条独立轴线展开(规模递增/工程约束递增即并发分布式/方案批判迭代/决策依据追问/真实性验证),每篇追加文件用几个跨知识点的范例案例演示这套追问框架怎么用,而不是把已有知识点重写一遍。

## 独立技能系列(不是 Python/ML,完全独立的新领域)

| 系列 | 内容 | 规模 | 状态 |
|------|------|------|------|
| [rhcsa-bash-deep-dive/](rhcsa-bash-deep-dive/00-roadmap.md) | Linux 系统管理 + bash 脚本编程,对标 Red Hat 官方 RHCSA(EX200,RHEL 10 基准)认证范围,9 个分类:必备工具与文本处理→进程与系统运行→本地存储与LVM→文件系统与权限→软件与系统部署→用户组管理→网络配置→安全(SELinux/防火墙)→bash脚本编程本身,收尾 1 篇[进阶深度追加](rhcsa-bash-deep-dive/10-advanced-interview-depth.md)(5 个真实故障排查链案例;该系列本身不采用面试对话体,改用"故障现象→排查动作→发现→根因→修复与验证"的运维排障语序)+ 1 篇[手把手教程体](rhcsa-bash-deep-dive/11-build-a-mini-system-healthcheck.md)(迷你系统巡检脚本,格式第一次从 Python 场景迁移到 bash/系统操作场景,现场记录僵尸进程构造两次失败才成功、`set -e`+`grep -c`零匹配陷阱两处真实坑)。验证环境为 WSL2 Rocky Linux 10.2(真实 systemd + root),涉及改网络/权限的知识点全部在隔离沙箱(dummy网卡/loop device)里操作,不影响宿主环境 | 100 个知识点 | ✅ 全部完成并验证 |
| [dsa-deep-dive/](dsa-deep-dive/00-roadmap.md) | 数据结构与算法,对标技术终面 1 小时持续深挖难度,18 个分类:复杂度分析与Python内置容器内核→数组字符串技巧→链表→二分查找→栈队列(含LRU/LFU设计)→排序从零实现→堆/优先队列→树→回溯→DP基础→贪心算法→Trie与字符串匹配→位运算与数学→图论基础→图论进阶(最短路/MST/强连通分量/网络流/二分图匹配)→DP进阶(区间/状压/数位/树形/概率期望/博弈)→线段树与树状数组→面试方法论与代码规范,收尾 1 篇 [1小时模拟终面capstone](dsa-deep-dive/19-mock-interview-capstone.md)(不是知识点列表,是完整还原终面节奏的场景叙事),再收尾 1 篇 [进阶深度追加](dsa-deep-dive/20-advanced-interview-depth.md)(基于真实调研2026大厂二面追问模式撰写的5个多级追问链案例:LRU并发/分布式、TopK海量数据、限流方案批判迭代、日志诊断真实系统行为、项目真实性验证追问)。**2026-07-24 新增试点** 1 篇[手把手实战教程](dsa-deep-dive/21-build-a-mini-search-engine.md)(不是知识点参考卡片也不是场景叙事,是读者从空文件开始跟着敲代码、分 4 阶段亲手搭出一个真实能用的迷你搜索引擎:倒排索引→前缀自动补全→相关度排序→组装成完整类,详见下方"教程体"说明)。验证环境为仓库根目录 `.venv`,纯 Python 标准库,不需要任何第三方包 | 140 个知识点 + 1 篇 capstone + 1 篇进阶深度追加 + 1 篇教程体试点 | ✅ 全部完成并验证 |
| [statistics-deep-dive/](statistics-deep-dive/00-roadmap.md) | 统计学(应用统计 + AI/ML场景专属统计,不是纯数理统计基础复习),对标技术终面 1 小时持续深挖难度,5 大板块 20 个分类:概率论回顾与描述统计→点估计理论→区间估计与假设检验框架→经典检验方法→多重检验与回归推断→A/B测试设计与功效分析→现代实验方法(序贯检验/mSPRT/CUPED)→因果推断基础→观察性因果推断方法(DID/IV/PSM/RDD)→真实陷阱案例集(Simpson悖论/幸存者偏差/真实事故复盘)→贝叶斯推断基础→MCMC基础(手写Metropolis-Hastings/Gibbs采样)→贝叶斯应用→模型评测统计→排位系统(Bradley-Terry/Elo,呼应 `learning/llm-judge-arena` 已验证源码)→Scaling law与外推→分布漂移与监控→标注一致性与分析方法论→时间序列基础→简单预测方法,收尾 1 篇 [模拟终面capstone](statistics-deep-dive/21-mock-interview-capstone.md)(frontier lab训练改动评估场景,把t检验/功效分析/多重比较/随机化混淆/贝叶斯概率化/诊断真实数据6处知识点串成一次连续追问)+ 1 篇[手把手教程体](statistics-deep-dive/22-build-an-ab-test-pipeline.md)(完整A/B测试分析流水线,现场验证双样本比例z检验的z²精确等于卡方独立性检验统计量,诚实记录一次"功效不足但恰好显著"的真实抽样结果)。验证环境为仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1),`statsmodels` 未安装,核心机制(MLE/bootstrap/置换检验/MCMC/2SLS/倾向得分匹配/Bradley-Terry/KL散度/Cohen's kappa/AR(1)最小二乘等)全部手写实现 | 116 个知识点 + 1 篇 capstone | ✅ 全部完成并验证 |
| [os-concurrency-deep-dive/](os-concurrency-deep-dive/00-roadmap.md) | 操作系统与并发("职业发展与需求"四部曲第 1 部:操作系统与并发→计算机网络→数据库原理与实战→系统设计,替代原"算法/系统设计/统计三选一"里笼统的"系统设计"一项),对标技术终面深挖难度,5 大板块 11 个分类:进程与线程基础→CPU调度算法→基础同步原语→高级并发模式与无锁编程→死锁→虚拟内存与地址转换→内存分配与页面置换→IO模型演进→进程间通信IPC→文件系统基础→现代系统专题(namespaces/cgroups/eBPF/NUMA/GPU调度与OS调度器关系/协程有栈无栈深化/云原生资源隔离),收尾 1 篇[模拟终面capstone](os-concurrency-deep-dive/12-mock-interview-capstone.md)(缓冲区池两条路径加锁顺序相反引发偶发死锁、进而被误诊为"内存泄漏"的场景,基于真实生成的线程事件日志诊断根因)+ 1 篇[手把手教程体](os-concurrency-deep-dive/13-build-a-mini-thread-pool.md)(迷你线程池,4阶段从单worker到优雅关闭全是同一个类不断长能力而非独立组件拼装,8任务8worker连续测出7.98~8.00倍加速)。验证环境双轨:仓库根目录 `.venv`(纯算法/数据结构模拟,如调度算法、页面置换算法、锁与竞态的 `threading` 演示)+ 复用 rhcsa-bash-deep-dive 已装好的 WSL2 Rocky Linux(真实 `fork`/`epoll`/`signal`/namespaces/cgroups 等 Linux 专属语义,每个知识点逐条标注用的是哪个环境) | 79 个知识点 + 1 篇 capstone | ✅ 全部完成并验证 |
| [computer-networking-deep-dive/](computer-networking-deep-dive/00-roadmap.md) | 计算机网络("职业发展与需求"四部曲第 2 部:操作系统与并发→计算机网络→数据库原理与实战→系统设计),对标技术终面深挖难度,全栈覆盖(链路层→网络层→传输层→应用层,应用层加重),5 大板块 12 个分类:分层模型与链路层基础→IP编址与子网划分→路由与网络层机制→TCP连接管理→TCP可靠传输与流量控制→TCP拥塞控制与UDP→HTTP演进→HTTPS与TLS→DNS域名解析→现代应用协议与API(WebSocket/gRPC/GraphQL/CDN/负载均衡)→网络编程与IO模型→现代网络专题(服务发现/API网关/Service Mesh/DDoS/容器网络),收尾 1 篇[模拟终面capstone](computer-networking-deep-dive/13-mock-interview-capstone.md)(API网关P99延迟陡增被误判为网络丢包、实为连接池打满排队的场景,基于真实生成的P50/P99延迟拆分数据诊断根因)+ 1 篇[手把手教程体](computer-networking-deep-dive/14-build-a-mini-http-server.md)(从原始socket手写HTTP服务器,不用http.server/Flask,阶段4用标准库客户端从另一端真实验证全链路)。验证环境双轨:仓库根目录 `.venv`(socket编程/HTTP手工构造/拥塞控制模拟/路由算法模拟/DNS报文构造)+ WSL2 Rocky Linux(真实 `tcpdump` 抓包、`tc netem` 注入延迟/丢包、`openssl s_client` 真实TLS握手、network namespace+veth pair真实容器网络隔离、`strace` 追踪 `sendfile()` 真实零拷贝系统调用)。**亮点**:用 `ssl.MemoryBIO` 手动驱动握手状态机真实测出 TLS1.2 需要 2 次网络往返、TLS1.3 只需 1 次(不是背书,是真实统计 socket 往返次数);真实构造 4KB 小缓冲区在 WSL2 复现 TCP 背压完整生效与解除过程(Windows loopback 自动调优会掩盖这个效果,已如实标注环境限制) | 80 个知识点 + 1 篇 capstone | ✅ 全部完成并验证 |
| [database-deep-dive/](database-deep-dive/00-roadmap.md) | 数据库原理与实战("职业发展与需求"四部曲第 3 部:操作系统与并发→计算机网络→数据库原理与实战→系统设计),对标技术终面深挖难度,关系数据库加重(分布式数据库与NoSQL点到面试常考程度),6 大板块 8 个分类:关系模型与基础SQL→范式与进阶SQL→索引结构与查询优化器→事务与隔离级别→MVCC与锁机制→存储引擎内部机制→复制与分布式基础→NoSQL与缓存模式,收尾 1 篇[模拟终面capstone](database-deep-dive/09-mock-interview-capstone.md)(限时优惠券超发+活动预览页慢查询雪崩两条独立线索同时发生的事故复盘场景叙事)+ 1 篇[手把手教程体](database-deep-dive/10-build-a-mini-kv-store.md)(迷你KV存储引擎:WAL持久化+崩溃重放+损坏记录识别,只用标准库文件I/O,呼应dsa-deep-dive/21结尾提到的持久化方向)。验证环境双轨:仓库根目录 `.venv`(stdlib `sqlite3`,覆盖关系模型/DDL/DML/JOIN/范式反例/聚合窗口函数/CTE/纯算法模拟)+ WSL2 Rocky Linux 真实双引擎(PostgreSQL 16.14 + MariaDB 10.11.15,隔离级别/MVCC/锁/复制类知识点双引擎真实对比,不满足于单引擎教科书描述)+ Valkey 8.0.9(NoSQL板块,Redis协议兼容分支)。**亮点**:真实验证出 PostgreSQL 的 REPEATABLE READ 能阻止丢失更新(`SerializationFailure`),MariaDB 的 REPEATABLE READ 做不到(要到 SERIALIZABLE 才行,且报错是 `Deadlock` 不是序列化失败)——全系列最有价值的单条技术发现;`Index Only Scan` 真实需要 `VACUUM` 更新可见性图,只 `ANALYZE` 不够 | 65 个知识点 + 1 篇 capstone | ✅ 全部完成并验证 |

rhcsa-bash-deep-dive 每个知识点固定七步结构(命令/配置 → 一句话是什么 → 为什么RHCSA真考/生产会用到 → 从最容易犯错的做法讲起 → 真实场景例子 → 可运行例子 → 常见坑),不采用 torch/tensorflow/huggingface 系列"面试怎么问"环节(RHCSA 是纯上机操作考试)。和"深挖系列"表格里六条系列的关键差异:本仓库没有 Linux 系统管理场景可挖,"真实场景例子"如实标注为典型运维/RHCSA 考试场景,不冒充仓库代码里挖出来的;部分知识点受 WSL2 内核结构性限制(GRUB引导链路不存在、dm-vdo模块缺失、selinuxfs接口残缺)无法完整验证真实效果,均已在对应小节诚实标注验证颗粒度,不冒充"已完整验证"。

dsa-deep-dive 的知识点结构和"深挖系列"六条完全一致的七步(签名/是什么→一句话→底层机制/为什么这样设计→AI研究/工程场景→可运行例子→面试怎么问+追问链→常见坑),"AI研究/工程场景"步骤改写为"这个技巧在真实系统里出现在哪",优先引用 torch-deep-dive/huggingface-deep-dive 已验证内容(如 Trie→tokenizer 的 BPE 训练、堆→beam search/top-k 采样、并查集→分布式一致性),没有真实关联的如实引用通用工程场景。本系列独有的方法论:复杂度不是断言出来的,是真的用 `time.perf_counter()` 在多组输入规模上测出来的(容差足够宽,只验证增长趋势的量级方向,不追求精确复现理论系数),这条纪律和 huggingface-deep-dive 系列"显存测量必须交叉核对 `torch.cuda.max_memory_allocated()` 和 `nvidia-smi`"同源,只是验证对象从"库的真实行为"换成了"算法的真实复杂度表现"。

**"教程体"新内容形态(2026-07-24,试点 → 全面推广):** 两轮新手可读性审计把 22 条系列现有内容修到文件级穷尽标准后,用户提出想要"面对新手的详细手把手教程"。这不是第三次可读性审计,而是新增一种仓库里原本没有的内容形态——和 19/20 类"读者旁观、跟着场景叙事看一遍推理链条"不同,教程体是"读者动手、从空文件开始跟着一步步敲代码、每写一段就跑一次看到真实效果",最终独立组装出一个真实能用的小工具。先在 [dsa-deep-dive/21-build-a-mini-search-engine.md](dsa-deep-dive/21-build-a-mini-search-engine.md) 试点(串联 01/02/07/12 类知识点搭一个迷你搜索引擎),用户确认"其余推广"后,同一天内推广到全部其余 16 条内容系列(数据/独立技能/专题精读三张表里除 dsa-deep-dive 自己之外的全部系列),`daily-toolkit-deep-dive` 刻意排除在外(见该系列表格下方单独说明)。

推广过程中期撞上过一次 API **日**限额中断(比这仓库历史上更常见的 5 小时窗口中断更severe)——17 路里 2 路完整跑完,其余 15 路当场失败;按 [[feedback-api-ratelimit-recovery-protocol]] 记录的既有协议逐个核实真实完成度(不是直接重派),发现大部分"失败"的 agent 其实已经写完真实内容(甚至留下了未清理但完全可复用的验证脚本),只有 4 条系列是真正零进度;用 1 个小规模探针 agent 确认限额已恢复后,批量补齐了剩余全部系列,过程零工作丢失。每一篇教程体文件都:①不修改所在系列任何已有的知识点卡片文件(纯新增);②交叉引用前先实地核实对应知识点文件的真实编号,不凭空猜测;③每段可运行代码在真实环境(`.venv`/WSL2,按各系列既有约定)里跑出真实输出后才誊写成断言;④诚实记录过程中撞见的真实坑或反直觉结果,不为了叙事干净而回避(比如 torch-deep-dive 现场复现拓扑序错误导致梯度算错、python-idioms 现场发现 `itertools.groupby` 只合并连续 key 这个系列内此前从未演示过的坑、rhcsa-bash 现场记录僵尸进程构造两次失败才成功)。

statistics-deep-dive 在"深挖系列"七步模板基础上新增独立的第 3 步,变成八步(定义与记号→一句话→**数学推导**→底层机制/为什么这样设计→AI研究/工程场景→可运行例子→面试怎么问+追问链→常见坑)——把七步版"底层机制"里混在一起的"真推公式"和"设计动机"拆开分别讲透,响应"要有数学"这条明确要求;全篇不用 LaTeX 语法,数学公式用 unicode 符号(μ σ θ Σ √ ≈ ≤ ≥ ∂ ∑ ∫)直接写在正文里,保证任何 markdown 渲染器下都能正确显示。面试追问链从第一天起就采用 dsa-deep-dive 20 类"进阶深度追加"事后调研验证过的五轴方法论(规模递增/工程约束递增/方案批判迭代/决策依据追问/真实性验证)+"诊断真实数据"新题型,不是像 dsa-deep-dive 那样事后再补一篇独立文件。撰写过程中数值验证发现并如实修正了多处真实问题而不是回避或掩盖:06类功效曲线公式的单样本/双样本噪声项混淆(漏了一个√2)、13类"贝叶斯方法天然免疫窥探问题"这一常见说法被数值证伪(properly threshold-matched后贝叶斯朴素peeking假阳性率≈频率派朴素peeking,只有mSPRT的鞅性质才有真实保护效果)、18类"Fleiss' kappa在n=2时精确等于Cohen's kappa"被修正为"仅当两名标注者边际分布相同时成立"(用AM-GM不等式证明恒有Fleiss≤Cohen)——这条纪律和 dsa-deep-dive"复杂度是真的测出来的"同源,只是验证对象从"算法的真实复杂度表现"换成了"数学结论的真实数值成立性"。

os-concurrency-deep-dive 是"职业发展与需求"四部曲(替代原三选一里的"系统设计")第 1 部,2026-07-13 真实调研(国内校招/实习后端八股文体系、欧美 new grad 面试趋势、招聘市场需求信号)确认操作系统/计算机网络/数据库这"计算机基础三件套"和系统设计权重相近、同样必考,才把笼统的"系统设计"拆成四个独立子项目依次做。知识点结构沿用 dsa-deep-dive 的七步模板,不像 statistics-deep-dive 那样加"数学推导"独立步骤(OS 的数学——Amdahl定律/Little's Law/Belady异常证明——是零星出现而非每点必需,直接写进"底层机制"步骤),面试追问链同样从第一天起融入五轴方法论,不事后补文件。**本系列独有的方法论:双环境验证策略**——纯算法/数据结构类知识点(调度算法模拟、页面置换算法模拟、锁与竞态的 `threading` 演示)在仓库 `.venv` 验证,真实 Linux 语义强相关的知识点(`fork` 写时复制、`epoll`、命名管道、`signal`、namespaces、cgroups)复用 rhcsa-bash-deep-dive 已装好的 WSL2 Rocky Linux 环境验证,每个知识点显式标注用的是哪个环境,不笼统带过;为此把 `_verify_md.py` 拓展了一个 `python-wsl2` 围栏标记(这一份独立拷贝,不影响其他系列用的原版脚本),让脚本正确跳过 Windows 原生 Python 跑不了的代码块而不是报错崩溃。撰写过程中真实发现并修正多处问题而不是回避:`multiprocessing.Process` 被 `_verify_md.py` 以 `-c` 字符串方式驱动时,因 spawn 机制找不到可 `import` 的 `__main__` 模块而导致父进程在 `queue.get()` 上无限挂起,改为把每个代码块写成临时文件再执行修复;孤儿进程验证最初在 `wsl.exe` 会话生命周期内被意外连带终止,加入忽略 `SIGHUP` 才稳定复现;cgroups 内存限制的真实行为比"一超限就立刻杀死"这个朴素预期更精细(有 swap 缓冲时不一定立刻 OOM kill,禁用 swap 后才稳定、确定地复现);03类知识点1竞态条件断言在全库自查阶段被独立重跑真实抓到过一次偶发失败,改为"多轮试验取至少一次命中"的聚合断言修复,这个偶发失败本身恰好印证了该知识点讨论的"竞态条件是概率性现象"这句话。

computer-networking-deep-dive 是"职业发展与需求"四部曲第 2 部,范围广度(链路层→网络层→传输层→应用层全栈覆盖,应用层加重)在 brainstorming 阶段用 AskUserQuestion 向用户核实过("通用软件工程方向"日常面试/工作里 HTTP/HTTPS/DNS/WebSocket/gRPC 这类应用层协议出现频率不低于甚至高于 TCP 底层细节,值得比传统教科书更重的篇幅)。七步模板与五轴追问链方法论延续 os-concurrency-deep-dive 不变,双环境验证策略按网络学科重新分工:`.venv` 覆盖 socket 编程/协议报文手工构造解析/拥塞控制与路由算法数值模拟,WSL2 Rocky Linux 覆盖真实 Linux 网络语义(`tcpdump` 抓包、`tc netem` 流量控制、`openssl` 真实 TLS 握手、network namespace+veth pair、`strace`)。**本系列新增纪律:不引入外部网络依赖**——所有可运行例子只连本地回环地址或 WSL2 内部构造的虚拟网络拓扑,不对真实互联网发起请求(DNS 类知识点用手工构造的虚拟域名层级验证,不查询真实公网 DNS 服务器),这是相对 os-concurrency-deep-dive 新增的一条纪律(OS 系列没有"外部依赖不可控"这类风险)。撰写过程中真实发现并如实处理多处问题而不是回避:①"签名/是什么"纯示意代码块两次被误标成可执行的 ` ```python ` 围栏(应为纯 ` ``` `),第二次之后做了全目录 grep 排查确认无其余遗漏;②TLS 相关知识点用 WSL2 `openssl req` 生成自签名证书时,初始 `-days 2` 有效期太短,会在读者未来运行代码时导致证书过期验证失败,改为 `-days 3650`;③KP6"网络编程背压"知识点最初尝试在 `.venv` 用小 `SO_RCVBUF`/`SO_SNDBUF` 复现背压,发现 Windows loopback 接口的缓冲区自动调优会覆盖这个显式设置、无法真实复现阻塞,改用 `python-wsl2` 围栏在真实 Linux 环境复现(真实测得发送方在接收方暂停期间确实阻塞、恢复后确实解除阻塞且全部数据完整送达);④用 `ssl.MemoryBIO` 手动驱动 TLS 握手状态机时发现 Python 标准库 `SSLSocket` 不会通过传入的 socket 对象的 Python 层 `send`/`recv` 方法做 IO(直接操作底层 fd),最初尝试的"包装 socket 计数收发次数"方案完全拿不到数据,改用 `MemoryBIO` 手动驱动握手状态机才真实测出 TLS1.2 需要 2 次网络往返、TLS1.3 只需 1 次。

database-deep-dive 是"职业发展与需求"四部曲第 3 部,范围广度(关系数据库加重,分布式数据库与NoSQL点到面试常考程度)和验证引擎选择(两个真实引擎都装,而不是只装一个更省事的)在 brainstorming 阶段用 AskUserQuestion 向用户核实过。七步模板与五轴追问链方法论延续前两部不变(数据库和 OS/网络一样是"系统学科"而非"数学学科",不像 statistics-deep-dive 那样加"数学推导"独立步骤,B+树磁盘IO复杂度推导/ARIES恢复算法思想/MVCC可见性判断规则这类内容直接写进"底层机制"步骤)。**本系列核心新决策:双真实引擎验证策略**——涉及隔离级别/MVCC/锁/复制的知识点不满足于单一引擎的行为描述,而是同时连接 WSL2 内真实安装的 PostgreSQL 16.14 和 MariaDB 10.11.15 做双引擎对比,专门暴露"教科书概念在不同引擎里的真实实现差异"(比单引擎教科书描述有更高的面试信息量);NoSQL板块因为 Rocky Linux 10 的 dnf 仓库在 2024 年 Redis Inc. 改许可证后已经没有 `redis` 包,如实记录并改用 Linux Foundation 主导的协议兼容分支 Valkey 8.0.9。撰写过程中真实发现并如实记录多处反直觉结论而不是回避:①`Index Only Scan` 需要 `VACUUM` 更新可见性图,只 `ANALYZE` 不够;②**PostgreSQL 的 REPEATABLE READ 真实能阻止丢失更新(`SerializationFailure`),MariaDB 的 REPEATABLE READ 做不到(要到 SERIALIZABLE 才行,且报错是 `Deadlock` 不是序列化失败)——全系列最有价值的单条技术发现**;③next-key lock 真实锁定范围比教科书定义更宽(目标值前后两侧间隙都被锁,不止通常描述的"前面"一侧);④现代 Redis/Valkey 的 AOF 持久化已经是多部分文件格式(`manifest`+`.base.rdb`+`.incr.aof`),不是教科书描述的单一 `appendonly.aof` 文件。WSL2 环境本身也踩过一个反复复发的坑:镜像网络模式合成的 `loopback0` 网卡被 firewalld 默认划进限制性 zone,导致所有到 `127.0.0.1` 的连接报 `No route to host`,基于接口的修复不够durable(复发 4 次),最终改用基于源地址(`--zone=trusted --add-source=127.0.0.0/8`)的规则才稳固。全库自查回归阶段还发现并修复了一个真实的并发同步 bug:capstone 里 5 个线程模拟"同时抢购最后一张优惠券"若不加显式 `threading.Barrier` 同步点,连接建立耗时不均会导致偶发复现失败——用起 5 个线程冒充"5 个人真的同时查询",时序噪声掩盖了要复现的真实 bug,这个发现本身也印证了 04/05 类反复强调的"并发时序必须用显式同步点控制,不能靠 sleep 或天然调度去猜"这条纪律。

## 操作类系列(手把手教工具怎么用,不是概念深挖)

和上面所有系列的关键差异:深挖系列/独立技能系列回答"这个函数/机制/命令是什么、为什么这样设计";这条系列回答"这件事具体怎么操作、做的时候该看到什么、卡住了怎么办"——类比:前面像教科书,这条系列像一位坐在你旁边、一步步带你操作的学长。

| 系列 | 内容 | 规模 | 状态 |
|------|------|------|------|
| [daily-toolkit-deep-dive/](daily-toolkit-deep-dive/00-roadmap.md) | 2026-07-23 老师反馈"学生可能连IDE/VSCode/Linux/SSH远程服务器/TeX这些操作性工具都不会用"驱动的新系列:VSCode日常开发操作(不含Jupyter,见[04](04-how-to-practice-with-jupyter.md))→Linux日常生产力操作(区别于rhcsa-bash的RHCSA考试视角)→SSH与远程GPU服务器实操→Git/GitHub日常协作实操→Python环境管理实操→命令行监控与调试实操(nvidia-smi/htop/日志/pdb)→LaTeX论文写作实操(直接服务ICLR投稿),7个分类,6步操作类模板(为什么需要/环境要求/一步步跟着做/背后发生了什么/常见坑/自测清单)。验证环境多轨:仓库根目录`.venv`+Git Bash(git/ssh本地部分/Python环境管理)、本机MiKTeX(LaTeX真实编译)、复用rhcsa-bash-deep-dive已装好的WSL2 Rocky Linux(Linux日常操作/SSH"远程服务器"角色扮演)。**亮点**:Git/Python环境管理章节所有操作全部在隔离的临时仓库/临时目录完成,真实构造并解决过merge conflict,收尾用git status/字节比对确认真实仓库和真实`.venv`零改动;LaTeX章节几乎每一节都真实编译出PDF(含用学生自己已学过的DPO loss公式排版验证、BibTeX完整4遍编译、3种常见报错真实触发);SSH章节撰写时发现并安全化解一次真实风险——Bash工具的`$HOME`和Read/Write/PowerShell工具解析到的真实Windows用户目录不是同一路径,一度险些用Write覆盖真实`~/.ssh/config`,被工具"改前必读"保护拦下(零写入),该发现已写入roadmap供后续操作`~/.ssh`类路径时留意 | 7个分类,操作类模板 | ✅ 全部完成并验证 |

**关于"教程体"没有推广到本系列**:下方"教程体"说明提到的新内容形态(2026-07-24 从 dsa-deep-dive 试点推广到其余 16 条系列)刻意跳过了 daily-toolkit-deep-dive——这条系列本身就是"一步步带你操作"的操作类模板(见本节开头的定位说明),已经具备"手把手"的核心特质,再叠加一篇同类的教程体文件价值有限,属于刻意排除而不是遗漏。

---

## 论文发表系列(从写作到社区回响)

2026-07-25,导师(Weikai Lin)转达两轮反馈:第一轮要求补充科研写作/学术绘图/论文投递相关的手把手教程;第二轮明确点名要再加 rebuttal(重点强调)、参会指南、oral、camera ready、release code、poster、presentation、hugface release、interact with community、deal with reject。这是 for_real_dummy 里第一次出现"研究判断力/学术生存技能"类内容——和"深挖系列"(函数/机制是什么)、"操作类系列"(工具怎么操作)都不同,回答的是"论文这件事从写到发表到被社区看见,应该怎么做"。设计文档见 [`docs/superpowers/specs/2026-07-25-paper-publication-series-design.md`](../docs/superpowers/specs/2026-07-25-paper-publication-series-design.md)。

按内容性质拆成 5 个独立系列(不是揉进一个大文件集,原因是写作/视觉设计/流程策略/临场表达/发布运营是差异很大的技能类型,合并会互相稀释深度),每个系列自己"由浅入深",不强行报一个大知识点总数——这类内容天然是原则+范例结构,统一按"分类数"呈现,延续 numpy-deep-dive"材料相对薄弱、诚实收敛不强行凑"的先例。

| 系列 | 内容 | 规模 | 状态 |
|------|------|------|------|
| [research-writing-deep-dive/](research-writing-deep-dive/00-roadmap.md) | 科研写作:叙事结构与电梯演讲(C-C-C/Motivation-Gap-Contribution、SPJ"先写论文再做研究"哲学)→Introduction段落写作与Related Work定位→Method/实验呈现逻辑(结果先行/Figure 1该放什么/消融三件套/bake-off陷阱)→句子层面学术英语(被动语态/zombie nouns/引用是脚注不是名词)→局限性诚实自曝→多轮修改方法论与AI辅助边界→审稿人视角红旗拆解→**Rebuttal写作技巧**(响应导师点名强调),8个分类,收尾1篇[模拟审稿+rebuttal攻防capstone](research-writing-deep-dive/09-mock-review-rebuttal-capstone.md)。新设计"六步写作判断力模板"(常见误区反例→逐处修改对照→可操作检查清单→能量化的部分写真实代码验证→审稿人会怎么挑刺+反驳链→常见坑),复用五轴追问链方法论的评审对抗版。**亮点**:能量化的判断力(被动语态占比/句长分布/关键词重叠度等)全部写真实Python代码验证而不是空谈;真实撞见并诚实记录多处坑,如`et al.`污染朴素句子切分器、`Figure 1:`图号数字污染正则规则误判有量化结果 | 8个分类+1篇capstone | ✅ 全部完成并验证 |
| [research-figures-deep-dive/](research-figures-deep-dive/00-roadmap.md) | 学术绘图:图表类型选择→颜色与感知设计(Okabe-Ito色盲友好调色板/为什么不能用jet colormap)→多面板排版工程(matplotlib gridspec/字体线宽DPI三件套/矢量图vs位图)→架构图与流程图(TikZ/draw.io/PowerPoint适用场景判断)→图注写作规范→常见反模式(Tufte chartjunk/双y轴陷阱/3D饼图),6个分类,复用深挖系列标准七步模板(面试怎么问换成"审稿人/读者会怎么挑刺"),收尾1篇[手把手教程体](research-figures-deep-dive/07-build-a-mini-publication-figure.md)(画一张完整投稿级核心结果图)。**验证方法论差异**(仓库第一次系统性产出图片文件而非纯文本assert):检查文件真实生成+尺寸/DPI元数据符合预期,能算的设计判断(如jet/viridis感知亮度单调性)真算,不假装能对图片内容做像素级assert。**亮点**:真实测出`fig.savefig(dpi=300)`读回DPI元数据是`299.9994`不是精确`300.0`(PNG格式浮点圆整误差);真实测出"墨水像素占比"和"信息密度"是两件不同的事,一个64pt大数字反而比八行小字占用更多像素面积 | 6个分类+1篇教程体 | ✅ 全部完成并验证 |
| [paper-submission-deep-dive/](paper-submission-deep-dive/00-roadmap.md) | 投稿与评审:Venue选择与fit度判断(JMLR vs IEEE TNNLS影响因子迷思)→双盲匿名化与OpenReview投稿系统操作(含真实可用的[匿名化检查脚本](paper-submission-deep-dive/_assets/02-anon-checker/check_anonymity.py),8类信号检测)→Rebuttal时间与协作管理(ICLR真实数据开篇:分数提升论文录用率55.7%-57.6%,不变的只有7.8%-12.4%)→**Deal with Reject深挖**(响应导师点名强调,独立分类不被"decision分支"笼统带过)→Camera-ready与Supplementary组织,5个分类,复用daily-toolkit已验证的6步操作类模板,收尾1篇[投稿全流程复盘capstone](paper-submission-deep-dive/06-mock-submission-journey-capstone.md)(首投被拒→冷静拆解→调整重投→录用)。**亮点**:匿名化检查脚本真实检出15类身份泄露信号,0误报;引用2025-11-27 OpenReview平台真实发生的匿名信息泄露事故(利用泄露信息=直接拒稿+多年禁令) | 5个分类+1篇capstone | ✅ 全部完成并验证 |
| [academic-presentation-deep-dive/](academic-presentation-deep-dive/00-roadmap.md) | 学术演讲与参会:Oral Presentation结构与讲述节奏(SPJ姊妹讲座"How to Give a Great Research Talk")→Slides设计原则(区别于论文图表,slide是给耳朵听的)→Poster设计与摆摊讲解(`#betterposter`运动)→现场Q&A应对技巧(CAP结构:Acknowledge-Answer-Perspective)→会议参会实用指南(日程冲突检测/networking分层策略),5个分类,复用写作系列的六步判断力模板(第⑤步换成"听众/评委会怎么问"),收尾1篇[会议日capstone](academic-presentation-deep-dive/06-conference-day-capstone.md)(5分钟spotlight准备→连续4轮追问当场应对)。**亮点**:用matplotlib真实渲染slide/poster版式好坏对照(`get_window_extent`真实测量文字包围盒,不是画风格示意图);真实测出"墨水像素占比"这一项和直觉不完全一致(好例子的64pt大数字占用像素面积反而略高于坏例子,但认知负担明显更低) | 5个分类+1篇capstone | ✅ 全部完成并验证 |
| [research-release-deep-dive/](research-release-deep-dive/00-roadmap.md) | 成果发布与社区:开源代码发布规范(README/依赖锁定/随机种子方差/License/DOI,和daily-toolkit的Git协作章节划清边界)→HuggingFace Release(Model Card/Dataset Card/上传前检查)→学术社区互动(挂号渠道/宣传帖/GitHub issue/邮件回应),3个分类(规模诚实偏小,不强行凑),收尾1篇[发布清单串讲](research-release-deep-dive/04-release-checklist-walkthrough.md)(拿真实项目`research/world-model-imagination-controller/`走一遍真实审计,不是空对空复述清单)。**亮点**:本机确认无真实HuggingFace账号/token,诚实分三层验证颗粒度(本地可验证/一次性真实网络校验不纳入自动化范围/完全没做的真实上传);真实发现`ModelCardData`的`tags`/`datasets`两个字段传入裸字符串时静默失败方式不一样(读源码确认) | 3个分类+1篇清单串讲 | ✅ 全部完成并验证 |

**5 个系列并行构建过程中期撞上一次 API 5 小时限额中断**(4/5 路成功,1 路[research-figures-deep-dive]在收尾自查阶段被切断)——按 [[feedback-api-ratelimit-recovery-protocol]] 既有协议核实真实完成度,发现该系列实际已经写完全部内容(7 篇正文+全部图片资产),只差最后一步"重新扫描确认干净"未完成,不需要重派,直接由主会话接手完成收尾自查。**这次自查额外发现一个跨 3 个系列共 90 余处的真实问题**:部分可运行代码块的 `print()` 语句字符串字面量混用了中文,这类内容按仓库既定纪律("Windows-GBK `print()` 必须纯 ASCII")本该在撰写阶段避免——`_verify_md.py` 只检查有没有 Python 异常、不检查文字是否可读,所以即使输出会在这台机器的 GBK 控制台编码下显示成乱码,验证脚本依然会判定"[ok]"。已逐一核实并修复(数值断言逻辑完全不变,只改字符串字面量),修复后全部代码块重新验证通过。

5 个系列均已独立核实(git diff 核对范围、重跑 `_verify_md.py`、抽查内容质量),不是单纯采信 agent 自报。

---

## 专题精读系列(直接对应 `learning/` 下具体专题模块的精读伴读笔记)

和上面"深挖系列"的关键差异:上面六条补的是"读懂 `learning/` 代码需要的通用框架/语言技能"(numpy/torch/HF 库机制本身,不绑定具体研究专题);这里每一条都**直接对应 `learning/` 下一个具体专题模块**,讲的是**同一份代码**,只是换成更适合初学者/面试备考的讲解深度(7 步结构,从最笨的想法讲起,面试怎么问+追问链),不是重复造轮子——定位类似"这份 PhD 级代码,配一份大二学生看得懂、还能扛住面试的精读笔记"。

| 系列 | 对应 `learning/` 专题 | 内容 | 规模 | 状态 |
|------|---------------------|------|------|------|
| [long-context-deep-dive/](long-context-deep-dive/00-roadmap.md) | [`learning/long-context/`](../learning/long-context/) | RoPE 外推家族(vanilla→PI→NTK→YaRN→3D-RoPE)→ 长上下文 Attention 架构(Ring/Striped/Infini-Attention)→ 长上下文评测方法论(NIAH/RULER/Lost-in-the-Middle)→ 数据工程与 Capstone(文档打包/课程学习/YaRN+LoRA capstone/KV-cache 显存核算),4 个分类 + 1 篇[进阶深度追加](long-context-deep-dive/05-advanced-interview-depth.md)(4 个案例)+ 1 篇[手把手教程体](long-context-deep-dive/06-build-a-mini-niah-evaluator.md)(迷你NIAH评测器,诚实标注复现的Lost-in-the-Middle U型曲线是判分函数里人为设计的衰减权重,不是接了真实模型跑出来的)。全系列纯 CPU、零 GPU 依赖 | 17 个知识点 | ✅ 全部完成并验证 |
| [kernel-gpu-deep-dive/](kernel-gpu-deep-dive/00-roadmap.md) | [`learning/gpu-architecture/`](../learning/gpu-architecture/) + [`learning/kernel-engineering/`](../learning/kernel-engineering/) | GPU 硬件与存储层次(规格表/tier推荐/Tensor Core/SM Occupancy/NVLink)→ Roofline 性能建模(arithmetic intensity/ridge point/反直觉发现)→ Kernel 设计语言(Triton autotune/CUTLASS layout代数/swizzle stub方法论)→ FlashAttention与Kernel Fusion(online softmax/HBM流量核算/128k capstone),4 个分类 + 1 篇[进阶深度追加](kernel-gpu-deep-dive/05-advanced-interview-depth.md)(4 个案例)+ 1 篇[手把手教程体](kernel-gpu-deep-dive/06-build-a-mini-roofline-calculator.md)(迷你Roofline计算器+简化版online softmax,向量加法/大矩阵乘法在A100/H100两块GPU上分类结果稳定符合直觉)。全系列纯 CPU、零第三方依赖 | 19 个知识点 | ✅ 全部完成并验证 |
| [alignment-algorithms-deep-dive/](alignment-algorithms-deep-dive/00-roadmap.md) | [`learning/dpo-family/`](../learning/dpo-family/) | DPO 基础与推导(RLHF闭式解→Bradley-Terry代换→loss逐项对照→真训练脚本→零margin=log2边界测试)→ PO 变体家族(IPO/KTO/ORPO/SimPO/CPO/DPOP六变体+8算法横向对比表)→ RainbowPO 统一视角与 Capstone(4轴统一框架的真实验证成色→6变体50步benchmark→zero trl import架构选择),3 个分类 + 1 篇[进阶深度追加](alignment-algorithms-deep-dive/04-advanced-interview-depth.md)(4 个案例)+ 1 篇[手把手教程体](alignment-algorithms-deep-dive/05-build-a-mini-dpo-loop.md)(迷你DPO训练循环,复用01号文件已推导的loss函数,200步训练现场验证margin真实拉开,诚实记录独立偏好对轨迹精确重合这一玩具设定的结构性产物)。全系列纯 CPU(`dpo_minimal.py` 真训练部分标注为可选进阶验证)。**亮点**:独立复验发现 `rainbowpo.py::unified_po_loss` 只精确复现 `dpo` 一个配置——`dpop`/`kto` 配置字段和 `dpo` 逐字段相同、静默退化成纯 DPO,其余变体均与各自独立实现存在结构性数值偏差 | 15 个知识点 | ✅ 全部完成并验证 |
| [peft-deep-dive/](peft-deep-dive/00-roadmap.md) | [`learning/lora-family/`](../learning/lora-family/) + [`learning/adapter-tuning-family/`](../learning/adapter-tuning-family/) | LoRA 核心与初始化变体(LoRA数学→rsLoRA/LoRA+→merge_and_unload→PiSSA/OLoRA→VeRA→LoHa/LoKr→AdaLoRA)→ 量化+LoRA(NF4→QLoRA fake-quant训练循环→QLoRA真4bit路径→LoftQ→DoRA→真4bit训练与真bitsandbytes从未同时出现的精确边界)→ Adapter家族核心(原始bottleneck adapter→Houlsby vs Pfeiffer→AdapterFusion→Compacter→Parallel Adapter→IA3)→ Adapter进阶与统一视角(AdapterDrop→MAM→K-Adapter/MAD-X→AdaMix→Prompt+LoRA+Adapter三线统一公式),4 个分类 + 1 篇[进阶深度追加](peft-deep-dive/05-advanced-interview-depth.md)(5 个案例)+ 1 篇[手把手教程体](peft-deep-dive/06-build-a-mini-lora-layer.md)(不调用peft库、从零实现迷你LoRA层,起点输出与冻结base逐位相等torch.equal验证,现场撞见lr=0.05导致loss飙升的真实不稳定案例,刻意压到CPU玩具规模),四条新系列里规模最大的一条,唯一真实用到本机 GPU(RTX 3080 Ti)验证。**亮点**:独立复验发现 IA3 手写与 peft 两条实现路径参数总量相等(55,296)系 GPT-2 具体维度巧合,直接读 peft 源码确认两者缩放的是完全不同的张量;MAD-X 的 `InvertibleAdapter` 被创建、计入参数量,但用 forward hook 确认从未接入前向计算图;`merge_weights()` docstring 与实现不符导致误用会双重计数 | 24 个知识点 | ✅ 全部完成并验证 |
| [inference-serving-deep-dive/](inference-serving-deep-dive/00-roadmap.md) | `learning/` Module 5《用大模型》7 个专题([`inference-engine-core`](../learning/inference-engine-core/)/[`sglang-radixattention`](../learning/sglang-radixattention/)/[`speculative-decoding`](../learning/speculative-decoding/)/[`quantization-deploy`](../learning/quantization-deploy/)/[`distributed-inference`](../learning/distributed-inference/)/[`production-serving`](../learning/production-serving/)/[`serving-graduation`](../learning/serving-graduation/)) | 2026-07-13 真实市场调研("大厂 AI 岗二三面权重转向系统设计/LLM 推理服务"信号)驱动的第 5 条新系列:mini-vLLM 骨架复刻(PagedAttention/Continuous Batching/CUDA Graph)→ RadixAttention/Agent 推理→投机解码全家族(Medusa/EAGLE系列/Lookahead)→推理期量化部署(GPTQ/AWQ/SmoothQuant/FP8)→分布式推理(TP/PP/EP/Disaggregated)→生产级部署(TensorRT-LLM/Triton/Ollama/OpenAI兼容API,**含真实部署 bonus**)→毕业专题常规知识点(Agent场景缓存/Thinking Budget/多模型路由/服务工程5原则)→毕业顶点capstone(叙事体,Module 5 毕业答辩),7 篇正文 + 1 篇 capstone + 1 篇[手把手教程体](inference-serving-deep-dive/09-build-a-mini-kv-cache-scheduler.md)(迷你KV-cache调度器,串联01号文件PagedAttention与continuous batching机制,阶段4复现并修复一个真实撰写前scratch阶段触发的head-of-line blocking调度器hang)。验证环境为仓库根目录 `.venv`(源模块 `requirements.txt` 列出的 vllm/sglang/triton 等库在 `src/*.py` 里几乎零真实 import,可跑内容全部是诚实标注的纯 Python/torch 算法复现),唯一例外是 06 号文件的真实部署 bonus 知识点,真实用 WSL2 + `vllm serve Qwen/Qwen2.5-0.5B-Instruct --quantization bitsandbytes` 跑通,过程连续踩中 UVA/pinned memory、gcc/Python.h 缺失、FlashInfer 需要完整 CUDA Toolkit、宿主机 VPN TUN 模式劫持 WSL2 回环路由四类真实环境阻塞并逐一诊断修复。**亮点**:独立复验发现 `openai_api_server.py` 的 `/v1/chat/completions` 因 `from __future__ import annotations` + 函数局部 `Request` import 组合,真实起服务后对任何请求统一 422(全系列最重要发现,已定位根因+验证修复);`pp_demo.py::gpipe_bubble()` 公式仅在 `n_stages=2` 时凑巧正确,18 组参数扫描证实其余场景系统性低估;`thinking_budget.py::generate_with_budget()` 强制关闭后直接丢弃真实 answer,3 组独立参数复现;`MockR1Model.stream()` 完全不读 `prompt` 参数,5 道题恒定输出同一答案,capstone 自己的验收条件从未被真正检验过;真实部署 bonus 测出 TTFT 异常但 TPOT 正常,用 `nvidia-smi` 逐秒采样 GPU 利用率证明延迟不在计算侧,而是本机 VPN 残留环境影响 | 71 个知识点 | ✅ 全部完成并验证 |
| [pretraining-infra-deep-dive/](pretraining-infra-deep-dive/00-roadmap.md) | `learning/` Module 3《造大模型》剩余 4 站([`data-curation`](../learning/data-curation/)/[`scaling-infra`](../learning/scaling-infra/)/[`pretraining-recipe`](../learning/pretraining-recipe/)/[`small-model-graduation`](../learning/small-model-graduation/))+ Module 8《系统与Infra》剩余 5 站([`cuda-essentials`](../learning/cuda-essentials/)/[`cluster-networking`](../learning/cluster-networking/)/[`storage-dataops`](../learning/storage-dataops/)/[`training-orchestration`](../learning/training-orchestration/)/[`infra-graduation`](../learning/infra-graduation/)) | 第 6 条新系列,覆盖"从数据到千卡集群"预训练规模化基建全链路:数据处理全流水(CommonCrawl抽取→MinHash/SemDeDup去重→质量过滤→BPE/SentencePiece分词→Magpie合成)→训练规模化(Chinchilla scaling law→FSDP/DeepSpeed/Megatron-LM并行→混合精度→L08-L11与inference-serving-deep-dive去重交叉引用)→预训练配方(WSD/cosine LR→Phi-tiny架构GQA+RoPE+SwiGLU→知识蒸馏→Llama-3 vs DeepSeek-V3配方对照,**含真实GPU训练bonus**)→Module 3毕业五部曲(VanillaGPT2→Phi-tiny→YaRN长上下文→课程学习,叙事体capstone,**含真实GPU训练bonus**)→CUDA执行模型(Warp原语→Bank Conflict→Coalescing→Tiled GEMM→Online Softmax)→集群网络(All-Reduce算法家族→Fat-Tree/Dragonfly拓扑→NCCL五大Collective→SHARP交换机内聚合)→存储与数据管线(五层存储BW/IOPS/延迟→Dataloader流水线→Sharding策略→Checkpoint三代→WebDataset)→训练编排(Slurm FIFO+Backfill→Gang Scheduling→MTBF+Young's Formula→Ray Actor模型→Elastic Training)→Module 8毕业(Mini-Cluster模拟器→TCO模型→Topology Selector→MLPerf对比,叙事体capstone,链[kernel-gpu-deep-dive](kernel-gpu-deep-dive/00-roadmap.md)),7 篇正文 + 2 篇叙事体capstone + 1 篇[手把手教程体](pretraining-infra-deep-dive/10-build-a-mini-cluster-scheduler.md)(迷你集群调度模拟器,FIFO队头阻塞真实复现→Backfill预约窗口机制,12job队列真实测出1.38倍利用率提升)。全系列纯 CPU、零第三方依赖(两处例外:03/04 号文件各有一次真实 GPU bf16 训练 bonus,本机 RTX 3080 Ti 直接跑)。**亮点**:独立发现VanillaGPT2 vs PhiTiny在有缺陷的mock标签方案(`y=x.clone()`)下初始化loss差一个数量级(≈1.0 vs ≈9.6,3种子验证,且labels正确shift后差异消失),该发现未见于任一源模块文档;独立验证SHARP相对ring的加速比精确收敛到`n_gpus-1`(跨16/128/256-GPU×1MB/10GB共6组合);独立验证B200/H100训练速度比精确等于两代GPU的`bf16_tflops`比值2250/989=2.275025(用全新128GPU+1B模型配置验证到1e-9精度,证实非预设场景巧合);独立发现小集群(8GPU)固定storage capex占3年TCO高达76.9%而4096GPU集群仅0.7% | 62 个知识点 | ✅ 全部完成并验证 |

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
├── numpy-deep-dive/                   ← numpy 逐函数精讲系列(约120个函数,10批+1篇进阶深度追加+1篇教程体)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表
│   ├── 01~10-*.md                     ← 每批一个文件
│   └── 11-advanced-interview-depth.md ← 进阶深度追加(3个案例,非知识点列表)
├── python-advanced/                   ← Python 中高级语法补完系列(20个知识点,4批+1篇进阶深度追加+1篇教程体)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表
│   ├── 01~04-*.md                     ← 每批一个文件
│   └── 05-advanced-interview-depth.md ← 进阶深度追加(5个案例,非知识点列表)
├── torch-deep-dive/                   ← torch逐机制精讲系列(100个知识点,面试深度,11批+1篇进阶深度追加+1篇教程体)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表
│   ├── 01~11-*.md                     ← 每批一个文件
│   └── 12-advanced-interview-depth.md ← 进阶深度追加(5个案例,非知识点列表)
├── huggingface-deep-dive/             ← HuggingFace生态工程内核系列(101个知识点,面试深度,13批+1篇进阶深度追加+1篇教程体,含真实多组微调训练对比)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 环境声明 + 模型数据集选型
│   ├── 01~13-*.md                     ← 每批一个文件
│   ├── 14-advanced-interview-depth.md ← 进阶深度追加(5个案例,非知识点列表)
│   ├── 09-lab-artifacts/              ← 09类微调对比的真实训练产物(loss曲线/显存数字等JSON记录)
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本
├── tensorflow-deep-dive/              ← TF逐机制精讲系列(100个知识点,面试深度,13批+1篇进阶深度追加+1篇教程体,WSL2 GPU环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 环境声明
│   ├── 01~13-*.md                     ← 每批一个文件
│   └── 14-advanced-interview-depth.md ← 进阶深度追加(5个案例,非知识点列表)
├── python-idioms/                     ← Pythonic写法惯用法系列(26个知识点,4批+1篇进阶深度追加+1篇教程体,python-advanced姊妹篇)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表
│   ├── 01~04-*.md                     ← 每批一个文件
│   └── 05-advanced-interview-depth.md ← 进阶深度追加(5个案例,非知识点列表)
├── rhcsa-bash-deep-dive/              ← Linux系统管理+bash脚本系列(100个知识点,对标RHCSA/EX200,9批+1篇进阶深度追加+1篇教程体,WSL2 Rocky Linux环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 环境声明
│   ├── 01~09-*.md                     ← 每批一个文件
│   └── 10-advanced-interview-depth.md ← 进阶深度追加(5个故障排查链案例,非知识点列表,不采用面试对话体)
├── dsa-deep-dive/                     ← 数据结构与算法系列(140个知识点+1篇capstone+1篇进阶深度追加+1篇教程体试点,面试深度,18批,纯Python标准库环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 复杂度验证方法论 + 经典题单映射表附录
│   ├── 01~18-*.md                     ← 每批一个文件
│   ├── 19-mock-interview-capstone.md  ← 1小时模拟终面capstone(场景叙事,非知识点列表)
│   ├── 20-advanced-interview-depth.md ← 进阶深度追加(基于真实调研的5个多级追问链案例,非知识点列表)
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本
├── statistics-deep-dive/              ← 统计学系列(116个知识点+1篇capstone+1篇教程体,面试深度,5大板块20批,仓库根目录.venv环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 八步模板说明 + 五轴追问链方法论
│   ├── 01~20-*.md                     ← 每批一个文件
│   ├── 21-mock-interview-capstone.md  ← 模拟终面capstone(frontier lab训练改动评估场景叙事,非知识点列表)
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本
├── os-concurrency-deep-dive/          ← 操作系统与并发系列("职业发展与需求"四部曲第1部,79个知识点+1篇capstone+1篇教程体,面试深度,5大板块11批,.venv+WSL2 Rocky Linux双环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 七步模板说明 + 五轴追问链方法论 + 双环境验证声明
│   ├── 01~11-*.md                     ← 每批一个文件,涉及真实Linux语义的代码块用python-wsl2围栏标记
│   ├── 12-mock-interview-capstone.md  ← 模拟终面capstone(缓冲区池偶发死锁引发"内存泄漏"假警报场景叙事,非知识点列表)
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本(拓展了python-wsl2围栏识别,拷贝自dsa-deep-dive但已针对本系列需求增量修改)
├── computer-networking-deep-dive/     ← 计算机网络系列("职业发展与需求"四部曲第2部,80个知识点+1篇capstone+1篇教程体,面试深度,5大板块12批,.venv+WSL2 Rocky Linux双环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 七步模板说明 + 五轴追问链方法论 + 双环境验证声明 + 不引入外部网络依赖纪律
│   ├── 01~12-*.md                     ← 每批一个文件,涉及真实Linux网络语义的代码块用python-wsl2围栏标记
│   ├── 13-mock-interview-capstone.md  ← 模拟终面capstone(API网关P99延迟陡增,连接池打满被误判为网络丢包的场景叙事,非知识点列表)
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本(拷贝自os-concurrency-deep-dive,未修改)
├── database-deep-dive/                ← 数据库原理与实战系列("职业发展与需求"四部曲第3部,65个知识点+1篇capstone+1篇教程体,面试深度,6大板块8批,.venv+WSL2双真实引擎[PostgreSQL 16.14+MariaDB 10.11.15]+Valkey 8.0.9)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 七步模板说明 + 五轴追问链方法论 + 双真实引擎验证声明 + firewalld坑记录
│   ├── 01~08-*.md                     ← 每批一个文件,涉及真实数据库连接的代码块用python-wsl2围栏标记
│   ├── 09-mock-interview-capstone.md  ← 模拟终面capstone(限时优惠券超发+活动预览页慢查询雪崩事故复盘场景叙事,非知识点列表)
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本(拷贝自computer-networking-deep-dive,未修改)
├── long-context-deep-dive/            ← 长上下文技术精读系列(17个知识点,面试深度,4批+1篇进阶深度追加+1篇教程体,对应learning/long-context/,纯CPU环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 差异化声明 + 环境声明
│   ├── 01~04-*.md                     ← 每批一个文件
│   └── 05-advanced-interview-depth.md ← 进阶深度追加(4个案例,非知识点列表)
├── kernel-gpu-deep-dive/              ← GPU架构+Kernel工程精读系列(19个知识点,面试深度,4批+1篇进阶深度追加+1篇教程体,对应learning/gpu-architecture+kernel-engineering,纯CPU环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 差异化声明 + 环境声明
│   ├── 01~04-*.md                     ← 每批一个文件
│   └── 05-advanced-interview-depth.md ← 进阶深度追加(4个案例,非知识点列表)
├── alignment-algorithms-deep-dive/    ← 偏好优化算法精读系列(15个知识点,面试深度,3批+1篇进阶深度追加+1篇教程体,对应learning/dpo-family,纯CPU环境)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 差异化声明 + 环境声明
│   ├── 01~03-*.md                     ← 每批一个文件
│   └── 04-advanced-interview-depth.md ← 进阶深度追加(4个案例,非知识点列表)
├── peft-deep-dive/                    ← PEFT技术精读系列(24个知识点,面试深度,4批+1篇进阶深度追加+1篇教程体,对应learning/lora-family+adapter-tuning-family,唯一用到本机GPU验证)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 差异化声明 + 环境声明
│   ├── 01~04-*.md                     ← 每批一个文件
│   └── 05-advanced-interview-depth.md ← 进阶深度追加(5个案例,非知识点列表)
├── inference-serving-deep-dive/       ← LLM推理服务精读系列(71个知识点+1篇capstone+1篇教程体,面试深度,7批,对应learning/ Module5《用大模型》7个专题,.venv环境+06号文件含WSL2真实部署bonus,全部完成)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 差异化声明 + 环境声明(含WSL2真实部署环境说明)
│   ├── 01~07-*.md                     ← 每批一个文件
│   ├── 08-serving-graduation-capstone.md ← Module 5毕业答辩capstone(叙事体,非知识点列表)
│   └── _verify_md.py                  ← 独立提取"可运行例子"标签代码块并执行的验证脚本
├── pretraining-infra-deep-dive/       ← 预训练规模化基建精读系列(62个知识点+2篇capstone+1篇教程体,面试深度,5批,对应learning/ Module3《造大模型》剩余4站+Module8《系统与Infra》剩余5站,纯CPU环境,03/04号文件各含1次真实GPU训练bonus,全部完成)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 差异化声明(kernel-gpu-deep-dive覆盖范围订正) + 环境声明
│   ├── 01-04-*.md                     ← Module 3部分:数据处理→训练规模化→预训练配方→Module3毕业(叙事体capstone)
│   ├── 05-09-*.md                     ← Module 8部分:CUDA执行模型→集群网络→存储管线→训练编排→Module8毕业(叙事体capstone,链kernel-gpu-deep-dive)
│   └── _verify_md.py                  ← 独立提取"可运行例子"标签代码块并执行的验证脚本
├── daily-toolkit-deep-dive/           ← 日常工具与科研工作流实操(操作类系列,非概念深挖,7个分类,响应"操作性工具都不会用"反馈,.venv+MiKTeX+WSL2多轨验证,全部完成)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 6步操作类模板说明 + 验证环境声明 + 完成小结
│   ├── 01~07-*.md                     ← VSCode→Linux日常→SSH远程→Git协作→Python环境管理→监控调试→LaTeX论文写作
│   └── _assets/                       ← 各章节真实验证用的辅助脚本/LaTeX编译产物(含真实生成的PDF)
├── research-writing-deep-dive/        ← 论文发表系列:科研写作(8分类+1篇capstone,六步写作判断力模板,含Rebuttal写作技巧,全部完成)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 六步模板说明 + 验证纪律
│   ├── 01~08-*.md                     ← 叙事结构→Introduction/Related Work→Method/结果呈现→学术英语→局限性→修改方法论→审稿人视角→Rebuttal写作
│   ├── 09-mock-review-rebuttal-capstone.md ← 模拟审稿+rebuttal攻防capstone(非知识点列表)
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本
├── research-figures-deep-dive/        ← 论文发表系列:学术绘图(6分类+1篇教程体,深挖系列标准七步模板,全部完成)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 验证方法论差异声明(图片文件验证) + 环境声明
│   ├── 01~06-*.md                     ← 图表类型→颜色感知→多面板排版→架构图工具→图注规范→常见反模式
│   ├── 07-build-a-mini-publication-figure.md ← 手把手教程体:画一张完整投稿级核心结果图
│   ├── _assets/                       ← 各知识点真实渲染的图片资产(PNG/PDF/SVG)
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本
├── paper-submission-deep-dive/        ← 论文发表系列:投稿与评审(5分类+1篇capstone,6步操作类模板,含真实匿名化检查脚本,全部完成)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 验证声明
│   ├── 01~05-*.md                     ← Venue选择→双盲匿名化与OpenReview→Rebuttal时间协作管理→Deal with Reject→Camera-ready
│   ├── 06-mock-submission-journey-capstone.md ← 投稿全流程复盘capstone(首投被拒→重投→录用,非知识点列表)
│   ├── _assets/02-anon-checker/       ← 真实可用的双盲匿名化检查脚本 check_anonymity.py + 测试用.tex样例
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本
├── academic-presentation-deep-dive/   ← 论文发表系列:学术演讲与参会(5分类+1篇capstone,六步判断力模板,全部完成)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 验证声明
│   ├── 01~05-*.md                     ← Oral结构节奏→Slides设计→Poster设计摆摊→现场Q&A→会议参会指南
│   ├── 06-conference-day-capstone.md  ← 会议日capstone(spotlight准备→现场追问应对,非知识点列表)
│   ├── _assets/                       ← 真实渲染的slide/poster版式对照图片
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本
├── research-release-deep-dive/        ← 论文发表系列:成果发布与社区(3分类+1篇清单串讲,6步操作类模板,全部完成)
│   ├── 00-roadmap.md                  ← 总规划 + 进度表 + 验证环境的诚实分级声明
│   ├── 01~03-*.md                     ← 开源代码发布规范→HuggingFace Release→学术社区互动
│   ├── 04-release-checklist-walkthrough.md ← 发布清单串讲(拿真实项目world-model-imagination-controller走一遍审计)
│   └── _verify_md.py                  ← 独立提取并执行每个代码块的验证脚本
├── my-cheatsheet.md                   ← 你自己维护的速查表(持续增长)
├── practice/                          ← 你自己动手写代码验证的 notebook
│   └── 00-getting-started.ipynb       ← 现成的起步文件,含环境自检 + 练习模板
└── qa/                                ← 每次有价值的问答,按主题归档
    └── ...(随问随加)
```

## 使用方式

每次问了有价值的问题,Claude 会把问题和答案整理到 `qa/` 对应的文件里。

---

*最后更新:2026-07-25*
