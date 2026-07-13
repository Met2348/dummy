# 操作系统与并发深挖 —— 路线图与进度表

> 目标:约 80 个操作系统与并发知识点,由浅入深,深度对标 [dsa-deep-dive/](../dsa-deep-dive/00-roadmap.md)/[statistics-deep-dive/](../statistics-deep-dive/00-roadmap.md)(面试二三面深度,不是"这个命令怎么用")。
> 定位:仓库"职业发展与需求"四部曲第 1 部——操作系统与并发 → 计算机网络 → 数据库原理与实战 → 系统设计。四部曲替代了原先"算法/系统设计/统计三选一"里笼统的"系统设计"一项:2026-07-13 真实调研(国内校招/实习后端八股文体系、欧美 new grad 面试趋势、招聘市场需求信号)显示,操作系统/计算机网络/数据库这"计算机基础三件套"和系统设计权重相近、同样覆盖面广、同样必考,值得各自成篇,不是系统设计的附属内容。
> 设计文档:[docs/superpowers/specs/2026-07-13-os-concurrency-deep-dive-design.md](../../docs/superpowers/specs/2026-07-13-os-concurrency-deep-dive-design.md)、实施计划:[docs/superpowers/plans/2026-07-13-os-concurrency-deep-dive.md](../../docs/superpowers/plans/2026-07-13-os-concurrency-deep-dive.md)。

---

## 和 rhcsa-bash-deep-dive 的边界(已核实,不重复)

已逐行核对 [rhcsa-bash-deep-dive/00-roadmap.md](../rhcsa-bash-deep-dive/00-roadmap.md) 全部知识点列表(尤其 02 类"进程与系统运行"、07 类"网络配置"):那一条是**操作层面**——`systemctl`/`nmcli`/`ps`/`kill` 这些命令怎么用,面向 RHCSA(EX200)上机操作考试,该系列明确声明不采用"面试怎么问"框架。这一条是**机制/理论层面**——进程调度算法为什么这样设计、并发原语的底层机制是什么、技术面试会怎么追问,和操作层面互不重复,可以并存参照。

## 知识点结构模板(七步,复用 dsa-deep-dive 模板)

1. **签名/是什么** 2. **一句话** 3. **底层机制/为什么这样设计**(零星数学——Amdahl定律、Little's Law、Belady异常证明、LRU竞争比——直接写在这一步,不单独拆"数学推导"步骤;和 statistics-deep-dive 不同,OS 的数学是零星点缀而非每点必需) 4. **AI研究/工程场景** 5. **可运行例子**(assert验证,**必须显式标注验证环境**:`.venv` 或 `WSL2 Rocky Linux`) 6. **面试怎么问+追问链**(五轴方法论,见下表) 7. **常见坑**

## 五轴追问链方法论(从第一天融入,不事后补——延续 statistics-deep-dive 已验证更优的做法)

| 轴线 | 含义 | OS/并发学科例子 |
|------|------|------|
| 规模递增轴 | 小规模→大规模→极限行为 | 单线程→多线程→NUMA多路服务器/海量连接C10M |
| 工程约束递增轴 | 单机→并发→分布式 | 本地锁→跨进程锁→分布式锁(为系统设计系列埋伏笔) |
| 方案批判迭代轴 | 面试官连续指出具体缺陷逼换方案 | 全局锁→细粒度锁→无锁结构,每次因具体性能/正确性缺陷被打回 |
| 决策依据追问轴 | 不纠错,只逼问选择依据 | "为什么用epoll不用多线程阻塞IO" |
| 真实性验证轴 | 简历"做了性能优化"被追问到具体数字 | "具体解决了什么并发问题,怎么测的,改动前后数字是多少" |
| 诊断真实数据(新题型) | 给日志/监控指标,要求诊断而非套公式 | 服务偶发卡顿,给thread dump要求定位死锁/锁竞争根因 |

每个知识点挑 1~2 条最自然的轴线走 2~3 层深,不强行凑满 5 轴。12 类模拟终面 capstone 是唯一要求同时用 3 条以上轴线的文件。

## 环境声明:双环境验证策略(本系列相对此前所有系列的新模式)

- **默认环境**:仓库根目录 `.venv`(Windows 原生 Python),用于纯算法/数据结构类模拟——调度算法模拟、页面置换算法模拟、锁与竞态的 `threading` 演示、IPC 的 `multiprocessing`/`shared_memory` 演示、内存分配器模拟等,不依赖真实内核语义。
- **WSL2 Rocky Linux**(`Rocky Linux 10.2`,`Python 3.12.13`,2026-07-13 已确认可用,复用 rhcsa-bash-deep-dive 已装好并验证过的环境,不新建):用于真实 Linux 语义强相关的知识点——`fork()` 真实写时复制行为、`epoll`(Windows 无此系统调用)、命名管道 `os.mkfifo`、真实 `signal`(如 `SIGCHLD`)、`namespaces`/`cgroups` 探查等。调用方式:Git Bash 调 `wsl.exe` 前需 `export MSYS_NO_PATHCONV=1`;`wsl.exe` 本身的路径参数用原生 Windows 反斜杠路径。
- 每个知识点的"可运行例子"步骤开头**必须**显式标注验证环境,不笼统一句"已验证"带过——延续 rhcsa-bash-deep-dive"逐条如实标注验证颗粒度"的纪律。
- `_verify_md.py` 只能驱动 `.venv` 里能跑的代码块;涉及 WSL2 Rocky Linux 验证的代码块,`_verify_md.py` 会因为 Windows Python 不支持对应系统调用(如 `os.fork()`/`select.epoll`)而报错,这类代码块需要单独用 `wsl.exe -d RockyLinux -- python3 -c "..."` 验证,并在文中如实记录"已在 WSL2 Rocky Linux 单独验证通过"。

## 并发/计时/死锁类特有验证纪律

- 计时对比断言用 `best_of(fn, *args, trials=N)` 取多次采样最小值,规模不能小到被系统调度噪声淹没(dsa-deep-dive 01 类教训)。
- **竞态条件/死锁复现类断言**:写作阶段至少重复运行 5~10 次确认现象稳定复现,不满足于运气好复现一次。
- **死锁 demo 必须带超时安全网**(`threading.Timer` 或进程级 timeout wrapper 或 daemon thread + 有限等待):真实构造死锁是教学价值所在,但验证脚本本身必须能确定性退出,不能永久挂起。

## 进度表

| # | 板块 | 分类 | 文件 | 知识点数(约) | 状态 |
|---|------|------|------|------------|------|
| 01 | I 进程、线程与调度 | 进程与线程基础 | [01-processes-and-threads.md](01-processes-and-threads.md) | 8 | ✅ 已完成(6个`.venv`代码块+3个WSL2真实系统调用验证,均通过) |
| 02 | I | CPU调度算法 | [02-cpu-scheduling.md](02-cpu-scheduling.md) | 8 | ✅ 已完成(8个`.venv`代码块全部通过,含`multiprocessing`真实多核加速测量) |
| 03 | II 并发同步与死锁 | 基础同步原语 | [03-synchronization-primitives.md](03-synchronization-primitives.md) | 8 | ✅ 已完成(8个`.venv`代码块,竞态类断言独立重跑5次全部稳定通过) |
| 04 | II | 高级并发模式与无锁编程 | [04-advanced-concurrency-patterns.md](04-advanced-concurrency-patterns.md) | 6 | ✅ 已完成(6个`.venv`代码块,含CAS/ABA/无锁栈/乐观锁,独立重跑3次稳定通过) |
| 05 | II | 死锁 | [05-deadlock.md](05-deadlock.md) | 6 | ✅ 已完成(6个`.venv`代码块,真实死锁复现独立重跑5次全部稳定,均带超时安全网) |
| 06 | III 内存管理 | 虚拟内存与地址转换 | [06-virtual-memory-and-address-translation.md](06-virtual-memory-and-address-translation.md) | 8 | ✅ 已完成(6个`.venv`+2个WSL2真实fork验证,均通过) |
| 07 | III | 内存分配与页面置换 | [07-memory-allocation-and-page-replacement.md](07-memory-allocation-and-page-replacement.md) | 7 | ✅ 已完成(7个`.venv`代码块全部通过) |
| 08 | IV IO模型与进程间通信 | IO模型演进 | [08-io-models.md](08-io-models.md) | 8 | ✅ 已完成(6个`.venv`+3个WSL2真实select/poll/epoll验证,均通过) |
| 09 | IV | 进程间通信IPC | [09-ipc.md](09-ipc.md) | 7 | ⏳ 未开始 |
| 10 | V 文件系统与现代系统专题 | 文件系统基础 | [10-filesystem-foundations.md](10-filesystem-foundations.md) | 6 | ⏳ 未开始 |
| 11 | V | 现代系统专题 | [11-modern-systems-topics.md](11-modern-systems-topics.md) | 7 | ⏳ 未开始 |
| 12 | 收尾 | 模拟终面capstone | [12-mock-interview-capstone.md](12-mock-interview-capstone.md) | —(不计入合计) | ⏳ 未开始 |

**预计合计:约 80 个知识点(±10~15% 浮动),11 个分类文件 + 1 篇模拟终面 capstone。** 数字标"约",最终以全库自查的精确计数为准(参照 statistics-deep-dive"先估后核"先例)。

---

## 验证纪律

- 验证脚本 `_verify_md.py`(regex 提取 ` ```python ` 代码块,每块独立 subprocess 执行)直接拷贝自 `dsa-deep-dive/_verify_md.py`,不重新设计。
- 数学/机制结论必须用 `assert` 数值验证,不能只摆公式描述。
- print() 语句必须纯 ASCII(历史教训:Windows GBK locale 下 `_verify_md.py` 子进程 reader 线程遇到非 ASCII 字符会 UnicodeDecodeError);正文 markdown 数学符号(μ σ θ Σ √ ≈ ≤ ≥ ∂ ∑ ∫)不受此限制。
- 不新增任何第三方 Python 依赖,标准库优先。
- 设计文档与实施计划见上方链接。

---

*创建:2026-07-13*
