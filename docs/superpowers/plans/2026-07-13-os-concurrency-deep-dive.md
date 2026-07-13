# 操作系统与并发深挖 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (内联执行模式,不用subagent-driven-development——用户已明确授权批量推进,任务间不暂停等待确认)。Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `for_real_dummy/os-concurrency-deep-dive/` 下产出一套约 80 个知识点、5 大板块 11 个分类文件 + 1 篇模拟终面 capstone 的操作系统与并发深挖系列,深度对标本仓库已完成的 dsa-deep-dive(140点)/statistics-deep-dive(116点),达到技术终面级别深度广度。

**Architecture:** 每个分类文件独立成篇,遵循统一的七步知识点模板(签名/是什么→一句话→底层机制/为什么这样设计→AI研究/工程场景→可运行例子→面试怎么问+追问链→常见坑),五轴追问链方法论从第一天融入"面试怎么问"步骤。验证采用双环境策略:纯算法/数据结构类模拟用仓库 `.venv`(Windows 原生 Python),真实 Linux 语义强相关的知识点(fork/epoll/signal/namespaces)用已确认可复用的 WSL2 Rocky Linux(`Python 3.12.13`,和 rhcsa-bash-deep-dive 共享同一套已装好的环境)。

**Tech Stack:** Python 3(仓库根目录 `.venv`,标准库为主:`threading`/`multiprocessing`/`select`/`mmap`/`os`/`signal`,不新增第三方依赖)+ WSL2 Rocky Linux 10.2(`/usr/bin/python3`,3.12.13,通过 `wsl.exe -d RockyLinux` 调用)。

## Global Constraints

- 环境:默认仓库根目录 `.venv`(Windows 原生);真实 Linux 语义(fork 写时复制/epoll/signal/namespaces/cgroups)用 WSL2 Rocky Linux(已确认可用,复用 rhcsa-bash-deep-dive 环境,不新建)。Git Bash 调 `wsl.exe` 前必须 `export MSYS_NO_PATHCONV=1`。
- 每个知识点的"可运行例子"步骤开头必须显式标注验证环境(`.venv` 或 `WSL2 Rocky Linux`)。
- 不新增任何第三方 Python 依赖,标准库优先。
- 知识点模板固定七步(签名/是什么→一句话→底层机制/为什么这样设计→AI研究/工程场景→可运行例子→面试怎么问+追问链→常见坑),不单独拆"数学推导"步骤——零星数学(Amdahl定律/Little's Law/Belady异常证明/LRU竞争比)直接写进"底层机制"步骤。
- 五轴追问链方法论(规模递增/工程约束递增/方案批判迭代/决策依据追问/真实性验证 + "诊断真实数据"新题型)从第一天融入"面试怎么问+追问链"步骤,每点挑 1~2 条最自然的轴线,不强行凑满 5 轴。
- 数学/机制结论必须用 `assert` 数值验证,不能只摆公式描述。
- print() 语句必须纯 ASCII(历史教训:Windows GBK locale 下 `_verify_md.py` 子进程 reader 线程遇到非 ASCII 字符如 "≈" 会 UnicodeDecodeError)。正文 markdown 里的数学符号(μ σ θ Σ √ ≈ ≤ ≥ ∂ ∑ ∫)不受此限制,只有 CODE 块内的 print() 输出受限。
- 竞态条件/死锁复现类断言:写作阶段至少重复运行 5~10 次确认现象稳定复现,不满足于复现一次。
- 死锁 demo 必须带超时安全网(`threading.Timer` 或进程级 timeout wrapper),保证脚本能确定性退出,不永久挂起。
- 计时对比断言用 `best_of(fn, *args, trials=N)` 取多次采样最小值,规模不能小到被系统调度噪声淹没。
- 每个板块(不是每个文件)验证通过后可以独立 git commit,但每个文件至少一次 commit;`git add` 必须显式列出文件路径,不用 `git add -A`/`.`。
- 涉及跨文件引用其他分类知识点,用established的"NN类知识点M"纯文本格式,不用 markdown 链接锚点。
- 11 类"现代系统专题"里提及 kernel-gpu-deep-dive 关联前,必须先实际读取该系列相关文件确认真实技术关联,不能凭听起来相关就编造联系。
- 08 类"IO模型演进"只讲 OS 怎么实现 IO 事件通知这个机制角度(select/poll/epoll 的内部差异),不讲 TCP/IP 协议细节、HTTP、DNS、拥塞控制——这些留给后续"计算机网络"子系列,不在本系列展开。

---

### Task 1: 脚手架

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/00-roadmap.md`
- Create: `for_real_dummy/os-concurrency-deep-dive/_verify_md.py`(从 `for_real_dummy/dsa-deep-dive/_verify_md.py` 原样拷贝)

**Interfaces:**
- Produces:`00-roadmap.md` 含 5 板块 11 文件 + capstone 的进度表(初始全部标"⏳ 未开始"),供 Task 2-13 逐行更新为 "✅ 已完成"。`_verify_md.py` 供 Task 2-13 每篇验证时调用:`python _verify_md.py <path/to/file.md>`。

- [ ] **Step 1: 拷贝验证脚本**

```bash
cp for_real_dummy/dsa-deep-dive/_verify_md.py for_real_dummy/os-concurrency-deep-dive/_verify_md.py
```

- [ ] **Step 2: 确认 WSL2 Rocky Linux 环境可用**

```bash
export MSYS_NO_PATHCONV=1 && wsl.exe -d RockyLinux -- python3 --version
```
Expected: `Python 3.12.13`(已在 brainstorming 阶段确认过一次,这里是执行阶段的正式复核)。

- [ ] **Step 3: 撰写 `00-roadmap.md`**

内容包含:目标声明(约80个知识点,5板块11文件+capstone,对标dsa-deep-dive/statistics-deep-dive深度)、与 rhcsa-bash-deep-dive 的边界声明(操作层面vs机制层面,已核实无重叠)、七步知识点模板说明、五轴追问链方法论表格(照抄 spec §4)、双环境验证声明(`.venv` 默认 + WSL2 Rocky Linux 真实语义,逐点标注)、死锁/竞态特有验证纪律(多次复现确认+超时安全网)、进度表(下表,初始状态全部"⏳ 未开始"):

| # | 板块 | 分类 | 文件 | 知识点数(约) | 状态 |
|---|------|------|------|------------|------|
| 01 | I 进程、线程与调度 | 进程与线程基础 | 01-processes-and-threads.md | 8 | ⏳ |
| 02 | I | CPU调度算法 | 02-cpu-scheduling.md | 8 | ⏳ |
| 03 | II 并发同步与死锁 | 基础同步原语 | 03-synchronization-primitives.md | 8 | ⏳ |
| 04 | II | 高级并发模式与无锁编程 | 04-advanced-concurrency-patterns.md | 6 | ⏳ |
| 05 | II | 死锁 | 05-deadlock.md | 6 | ⏳ |
| 06 | III 内存管理 | 虚拟内存与地址转换 | 06-virtual-memory-and-address-translation.md | 8 | ⏳ |
| 07 | III | 内存分配与页面置换 | 07-memory-allocation-and-page-replacement.md | 7 | ⏳ |
| 08 | IV IO模型与进程间通信 | IO模型演进 | 08-io-models.md | 8 | ⏳ |
| 09 | IV | 进程间通信IPC | 09-ipc.md | 7 | ⏳ |
| 10 | V 文件系统与现代系统专题 | 文件系统基础 | 10-filesystem-foundations.md | 6 | ⏳ |
| 11 | V | 现代系统专题 | 11-modern-systems-topics.md | 7 | ⏳ |
| 12 | 收尾 | 模拟终面capstone | 12-mock-interview-capstone.md | —(不计入合计) | ⏳ |

- [ ] **Step 4: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/00-roadmap.md for_real_dummy/os-concurrency-deep-dive/_verify_md.py
git commit -m "docs(os-concurrency): 脚手架 - roadmap + 验证脚本"
```

---

### Task 2: 01-processes-and-threads.md(进程与线程基础)

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/01-processes-and-threads.md`
- Modify: `for_real_dummy/os-concurrency-deep-dive/00-roadmap.md`(第01行状态改✅)

**Interfaces:**
- Consumes: Task 1 产出的 `_verify_md.py`
- Produces: 完成后续文件可引用的"01类知识点N"编号体系(本文件内 8 个知识点从 1 到 8 编号)

知识点范围(8个,均为设计边界,允许在撰写时合理合并/拆分同类概念但总数保持在 7~9 区间):PCB与进程状态转换、fork/exec/wait语义与写时复制COW(`.venv`用`multiprocessing`演示跨平台语义,WSL2 Rocky Linux用真实`os.fork()`演示COW)、线程模型(1:1/N:1/M:N)与Python GIL的位置、协程与用户态调度(有栈vs无栈)、线程私有数据TLS(`threading.local`)、进程vs线程vs协程选型决策、僵尸进程与孤儿进程(WSL2 Rocky Linux真实复现)、Windows进程创建模型对比(`CreateProcess`/spawn vs Unix `fork`)。

- [ ] **Step 1: 设计知识点并逐点数值/系统调用验证**

用 Bash 在 `.venv` 和 WSL2 Rocky Linux 分别验证核心结论(如:`fork()`后子进程修改变量不影响父进程,COW实际测量写时才复制;僵尸进程在`wait()`前`ps`能看到`<defunct>`状态等),迭代到断言稳定,记录每点用的验证环境。

- [ ] **Step 2: 撰写完整 markdown**

按七步模板撰写全部 8 个知识点,"面试怎么问+追问链"融入五轴方法论(如"决策依据追问轴:为什么这里用进程不用线程隔离"),每个可运行例子开头标注验证环境。

- [ ] **Step 3: 运行验证脚本**

```bash
cd for_real_dummy/os-concurrency-deep-dive && python _verify_md.py 01-processes-and-threads.md
```
Expected: 全部代码块 PASS。WSL2 相关代码块需额外用 `wsl.exe -d RockyLinux -- python3 -c "..."` 单独验证(`_verify_md.py`本身跑在`.venv`,不会自动路由到WSL,涉及WSL验证的代码块需要在文中如实标注"以下例子需在WSL2 Rocky Linux验证,已单独确认通过"并附验证记录,不能让`_verify_md.py`对这类块直接执行——因为Windows Python解释器不支持`os.fork()`,直接跑会报`AttributeError`)。

- [ ] **Step 4: 检查 print() 语句纯 ASCII**

```bash
grep -P '[^\x00-\x7F]' for_real_dummy/os-concurrency-deep-dive/01-processes-and-threads.md | grep 'print('
```
Expected: 无匹配(有匹配则修正为纯 ASCII)。

- [ ] **Step 5: 更新 roadmap 第01行状态为 ✅ 已完成**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/01-processes-and-threads.md for_real_dummy/os-concurrency-deep-dive/00-roadmap.md
git commit -m "docs(os-concurrency): 01类 进程与线程基础(8点)"
```

---

### Task 3: 02-cpu-scheduling.md(CPU调度算法)

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/02-cpu-scheduling.md`
- Modify: `00-roadmap.md`(第02行)

知识点范围(8个):调度目标与指标(吞吐量/周转时间/响应时间/公平性)、FCFS、SJF与饥饿、时间片轮转RR、多级反馈队列MLFQ、CFS完全公平调度器(红黑树+vruntime思想)、实时调度与优先级反转/优先级继承、多核调度与CPU亲和性/cache一致性代价。

- [ ] **Step 1: 设计知识点并数值验证**

调度算法本身用 `.venv` 纯 Python 模拟实现(FCFS/SJF/RR/MLFQ 各自实现为调度器类,构造一批任务,验证平均等待时间/周转时间符合理论公式,比如 SJF 在无饥饿场景下平均等待时间严格 ≤ FCFS)。CPU 亲和性用 `os.sched_getaffinity`/`os.sched_setaffinity`(注意:Windows 无此 API,需要用 WSL2 Rocky Linux 验证;或改用 `psutil` 前先确认是否违反"不新增第三方依赖"——若违反则该知识点的"可运行例子"改为纯概念性模拟,不追求调用真实 affinity API,如实标注)。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本并确认 PASS**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第02行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/02-cpu-scheduling.md for_real_dummy/os-concurrency-deep-dive/00-roadmap.md
git commit -m "docs(os-concurrency): 02类 CPU调度算法(8点) - 板块I完成"
```

---

### Task 4: 03-synchronization-primitives.md(基础同步原语)

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/03-synchronization-primitives.md`
- Modify: `00-roadmap.md`(第03行)

知识点范围(8个):竞态条件本质与检测、临界区与互斥锁、自旋锁vs互斥锁的忙等待代价、信号量与生产者消费者问题、读写锁(读者优先vs写者优先)、条件变量与虚假唤醒、屏障Barrier同步、内存屏障与可见性(happens-before初步)。

- [ ] **Step 1: 设计知识点并数值验证**

用 `.venv` 的 `threading` 模块**真实复现**竞态条件(多线程无锁并发自增共享计数器,验证最终值 < 理论值,重复运行 5~10 次确认现象稳定复现——不是运气好复现一次);再展示加锁后严格等于理论值。生产者消费者用 `threading.Semaphore` 实现并验证缓冲区不越界。读写锁需要自己实现(标准库无内置读写锁),验证读者优先策略下写者可能饥饿的现象。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本,竞态类代码块额外重复运行 5~10 次确认稳定**

```bash
for i in $(seq 1 8); do python _verify_md.py 03-synchronization-primitives.md; done
```

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第03行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/03-synchronization-primitives.md for_real_dummy/os-concurrency-deep-dive/00-roadmap.md
git commit -m "docs(os-concurrency): 03类 基础同步原语(8点)"
```

---

### Task 5: 04-advanced-concurrency-patterns.md(高级并发模式与无锁编程)

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/04-advanced-concurrency-patterns.md`
- Modify: `00-roadmap.md`(第04行)

知识点范围(6个):CAS与ABA问题、无锁队列初步、线程池设计(核心线程数/队列策略/拒绝策略)、协程调度器实现思想、并发容器设计模式、乐观锁到数据库场景的映射(为后续数据库系列埋伏笔,如实标注"这是为后续数据库系列埋下的伏笔,当前不展开数据库specific内容")。

- [ ] **Step 1: 设计知识点并数值验证**

CAS 用 `ctypes`/或 `threading` + 循环重试模拟实现,构造真实 ABA 场景(值被改回原值但中间状态已变化,单纯值比较无法检测)。线程池用 `concurrent.futures.ThreadPoolExecutor` 实测不同核心线程数下的吞吐量对比(用 `best_of` 纪律)。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本并确认 PASS**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第04行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/04-advanced-concurrency-patterns.md for_real_dummy/os-concurrency-deep-dive/00-roadmap.md
git commit -m "docs(os-concurrency): 04类 高级并发模式与无锁编程(6点)"
```

---

### Task 6: 05-deadlock.md(死锁)

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/05-deadlock.md`
- Modify: `00-roadmap.md`(第05行)

知识点范围(6个):死锁四个必要条件、资源分配图与检测算法、银行家算法、死锁预防策略(破坏四条件)、死锁避免vs预防vs检测的取舍、活锁与饥饿。

- [ ] **Step 1: 设计知识点并数值验证(重点:死锁真实复现 + 超时安全网)**

用 `.venv` 的 `threading` + 两把锁构造**真实死锁**(线程A持锁1求锁2,线程B持锁2求锁1),用 `threading.Timer` 或者对 `join(timeout=N)` 判断线程是否仍存活来证明死锁确实发生(而非碰巧调度顺序躲过),同时保证脚本本身能在断言完成后正常退出(不能让死锁线程真的挂起整个验证进程——用 daemon thread + 主线程只等待有限时间的方式处理)。银行家算法手写实现,构造安全/不安全两种资源分配序列各验证一次。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本,死锁复现代码块额外重复运行 5~10 次确认稳定触发**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第05行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/05-deadlock.md for_real_dummy/os-concurrency-deep-dive/00-roadmap.md
git commit -m "docs(os-concurrency): 05类 死锁(6点) - 板块II完成"
```

---

### Task 7: 06-virtual-memory-and-address-translation.md(虚拟内存与地址转换)

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/06-virtual-memory-and-address-translation.md`
- Modify: `00-roadmap.md`(第06行)

知识点范围(8个):虚拟地址空间与逻辑/物理地址、MMU与地址转换机制、多级页表与空间开销权衡、TLB与地址转换加速、分段机制、分段与分页对比、写时复制COW的实际应用(和01类知识点2呼应,不重复展开只做cross-reference)、大页Huge Page。

- [ ] **Step 1: 设计知识点并数值验证**

多级页表的空间开销用 Python 数值计算验证(比如对比单级页表 vs 二级页表在稀疏地址空间下的实际占用字节数)。TLB 命中率用模拟内存访问序列(时间局部性 vs 随机访问)对比命中率数值差异。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本并确认 PASS**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第06行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/06-virtual-memory-and-address-translation.md for_real_dummy/os-concurrency-deep-dive/00-roadmap.md
git commit -m "docs(os-concurrency): 06类 虚拟内存与地址转换(8点)"
```

---

### Task 8: 07-memory-allocation-and-page-replacement.md(内存分配与页面置换)

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/07-memory-allocation-and-page-replacement.md`
- Modify: `00-roadmap.md`(第07行)

知识点范围(7个):malloc内部机制(空闲链表法)、伙伴系统与slab分配器、页面置换算法FIFO/LRU/Clock、最优置换Belady定理与Belady异常(构造真实的FIFO增加页框数反而缺页率上升的反例序列)、mmap内存映射、Swap与抖动Thrashing、内存碎片(外部碎片vs内部碎片)。

- [ ] **Step 1: 设计知识点并数值验证**

页面置换算法(FIFO/LRU/Clock/OPT)全部手写实现,用同一组访问序列跑全部算法对比缺页次数,验证 OPT ≤ 其他所有算法(最优性下界)。Belady 异常需要专门构造一个已知的经典反例序列(如 1,2,3,4,1,2,5,1,2,3,4,5 用 3 帧 vs 4 帧对比 FIFO 缺页数),验证反直觉的"帧数增加缺页反而变多"现象真实成立。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本并确认 PASS**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第07行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/07-memory-allocation-and-page-replacement.md for_real_dummy/os-concurrency-deep-dive/00-roadmap.md
git commit -m "docs(os-concurrency): 07类 内存分配与页面置换(7点) - 板块III完成"
```

---

### Task 9: 08-io-models.md(IO模型演进)

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/08-io-models.md`
- Modify: `00-roadmap.md`(第08行)

知识点范围(8个):阻塞IO vs 非阻塞IO、select多路复用与局限(fd_set大小限制)、poll改进(无fd上限但仍O(n)扫描)、epoll与水平触发vs边缘触发(WSL2 Rocky Linux验证,Windows无epoll)、异步IO(AIO)与io_uring简介(概念性介绍,不强求完整实现)、Reactor模式、Proactor模式、C10K/C10M问题的历史与解法演进。

- [ ] **Step 1: 设计知识点并数值验证**

`select`/`poll` 用 `.venv` 标准库 `select` 模块对真实 socket 对做跨平台验证。`epoll`(`select.epoll`,Linux-only)必须在 WSL2 Rocky Linux 用真实 Python3 验证,构造水平触发(不读完数据下次仍通知)vs 边缘触发(只在状态变化时通知一次)的真实行为差异,用真实 socket 收发数据验证两种触发模式通知次数不同。Reactor 模式用 `select` 手写一个简化事件循环验证。

- [ ] **Step 2: 撰写完整 markdown**

epoll 相关例子明确标注"以下例子需在WSL2 Rocky Linux验证(Windows无epoll系统调用)"。

- [ ] **Step 3: 运行验证脚本(`.venv`部分用`_verify_md.py`,WSL2部分单独用`wsl.exe -d RockyLinux -- python3 -c "..."`验证并在文中记录结果)**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第08行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/08-io-models.md for_real_dummy/os-concurrency-deep-dive/00-roadmap.md
git commit -m "docs(os-concurrency): 08类 IO模型演进(8点)"
```

---

### Task 10: 09-ipc.md(进程间通信IPC)

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/09-ipc.md`
- Modify: `00-roadmap.md`(第09行)

知识点范围(7个):匿名管道Pipe、命名管道FIFO(WSL2 Rocky Linux验证,Windows无`os.mkfifo`)、System V消息队列(概念性,Python标准库无直接封装,如实标注)、POSIX共享内存及同步问题(`multiprocessing.shared_memory`跨平台验证)、Socket作为IPC手段(`AF_UNIX`,WSL2验证;`AF_INET` localhost 作为Windows替代方案)、信号Signal机制(WSL2 Rocky Linux验证真实`SIGCHLD`等)、IPC方式选型对比(7种方式的适用场景,纯对比表格不需要单独代码验证)。

- [ ] **Step 1: 设计知识点并数值验证**

匿名管道、共享内存用 `.venv` 的 `os.pipe()`/`multiprocessing.shared_memory` 验证。命名管道/`AF_UNIX` socket/真实signal 用 WSL2 Rocky Linux 验证。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本(双环境分别验证)**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第09行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/09-ipc.md for_real_dummy/os-concurrency-deep-dive/00-roadmap.md
git commit -m "docs(os-concurrency): 09类 进程间通信IPC(7点) - 板块IV完成"
```

---

### Task 11: 10-filesystem-foundations.md(文件系统基础)

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/10-filesystem-foundations.md`
- Modify: `00-roadmap.md`(第10行)

知识点范围(6个):inode/超级块/目录项结构、日志文件系统journaling与崩溃一致性(概念性讲解为主)、page cache与fsync语义(`.venv`能测:写文件后不`fsync`直接读能读到刚写的内容,说明page cache的存在,但断电持久性本身无法在单元测试里验证,如实标注这个局限)、硬链接软链接的文件系统层本质(WSL2 Rocky Linux验证真实inode共享,呼应rhcsa-bash-deep-dive 01类但角度不同——那边是操作层面`ln -s`怎么用,这里是inode共享机制本身)、文件系统一致性保证机制、稀疏文件与文件空洞(`.venv`可验证:seek后写入产生的文件`st_size`与实际磁盘占用块数不同)。

- [ ] **Step 1: 设计知识点并数值验证**

inode 共享用 WSL2 Rocky Linux 验证(硬链接后 `stat` 显示相同 inode 号,链接数变化)。稀疏文件用 `.venv` 验证(`os.stat().st_size` vs 实际占用,Windows NTFS 也支持稀疏文件语义可以验证,但需要确认实际行为,如不确定则同样用 WSL2 兜底)。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第10行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/10-filesystem-foundations.md for_real_dummy/os-concurrency-deep-dive/00-roadmap.md
git commit -m "docs(os-concurrency): 10类 文件系统基础(6点)"
```

---

### Task 12: 11-modern-systems-topics.md(现代系统专题)

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/11-modern-systems-topics.md`
- Modify: `00-roadmap.md`(第11行)

知识点范围(7个):容器隔离namespaces原理(WSL2 Rocky Linux,`unshare`命令或`/proc/[pid]/ns/`查看)、cgroups资源限制原理(WSL2 Rocky Linux,`/sys/fs/cgroup/`真实探查)、eBPF基础原理(概念性为主,真实eBPF程序编写超出本系列范围,如实标注边界)、NUMA架构对高性能/训练任务的影响、GPU任务调度与OS调度器的关系、协程实现原理深化(有栈vs无栈性能对比,`.venv`可用`greenlet`风格手写栈切换 vs `asyncio`协程对比,注意不新增第三方依赖,`greenlet`需要用标准库方式替代或明确标注为概念性讲解)、云原生场景下的资源隔离与调度。

- [ ] **Step 1: 核实 kernel-gpu-deep-dive 真实关联(前置步骤,写作前必须完成)**

```bash
grep -rn "调度\|scheduling\|scheduler" for_real_dummy/kernel-gpu-deep-dive/*.md
```
读取匹配结果,确认 GPU 任务调度与 OS 调度器的真实技术关联点(比如 CUDA stream 调度、MPS、GPU 时间片和 OS 进程调度的类比与差异),只写真实核实过的关联,如果核实后发现关联很弱,如实缩小这个知识点的"AI研究/工程场景"步骤篇幅,不夸大。

- [ ] **Step 2: 设计知识点并数值验证**

namespaces/cgroups 用 WSL2 Rocky Linux 真实探查验证(比如创建一个新 PID namespace 内进程 PID 从 1 开始,cgroups 限制内存后实际测试超限行为)。

- [ ] **Step 3: 撰写完整 markdown**

- [ ] **Step 4: 运行验证脚本**

- [ ] **Step 5: 检查 print() 纯 ASCII**

- [ ] **Step 6: 更新 roadmap 第11行状态为 ✅**

- [ ] **Step 7: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/11-modern-systems-topics.md for_real_dummy/os-concurrency-deep-dive/00-roadmap.md
git commit -m "docs(os-concurrency): 11类 现代系统专题(7点) - 板块V完成,全部11个分类文件完成"
```

---

### Task 13: 12-mock-interview-capstone.md(模拟终面capstone)

**Files:**
- Create: `for_real_dummy/os-concurrency-deep-dive/12-mock-interview-capstone.md`
- Modify: `00-roadmap.md`(第12行)

**Interfaces:**
- Consumes: 01-11 类全部知识点编号(用于追问链里的 cross-reference,格式"NN类知识点M")

场景设定:高并发服务出现偶发死锁 + 内存暴涨,给一段真实模拟日志(带时间戳的事件序列,由 Python 脚本真实生成而非手写编造)要求诊断根因。结构参照 statistics-deep-dive 21 类capstone(候选人初版汇报→面试官多轮追问→最终诊断真实异常数据),覆盖至少 3 条五轴追问链轴线,cross-reference 至少 5 处不同类别的知识点(比如 03类竞态/05类死锁/07类内存/08类IO模型/02类调度)。

- [ ] **Step 1: 设计capstone叙事结构与验证场景**

用 `.venv` 的 `threading` 真实构造一个"看起来是内存泄漏、实际是死锁导致连接池耗尽后不断重试"的场景,生成真实的模拟日志(时间戳+事件),验证日志里确实能看到"请求堆积"+"同一资源反复获取失败"这类可诊断的具体异常特征(不是编的故事)。

- [ ] **Step 2: 撰写完整 markdown(候选人初版汇报→多轮追问→最终诊断)**

- [ ] **Step 3: 运行验证脚本,并独立重跑至少 3 次确认稳定(全部随机性固定种子)**

```bash
cd for_real_dummy/os-concurrency-deep-dive
python _verify_md.py 12-mock-interview-capstone.md
python _verify_md.py 12-mock-interview-capstone.md
python _verify_md.py 12-mock-interview-capstone.md
```

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第12行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/os-concurrency-deep-dive/12-mock-interview-capstone.md for_real_dummy/os-concurrency-deep-dive/00-roadmap.md
git commit -m "docs(os-concurrency): 12类 模拟终面capstone"
```

---

### Task 14: 全库自查回归 + README + memory 更新 + 最终提交

**Files:**
- Modify: `for_real_dummy/os-concurrency-deep-dive/00-roadmap.md`(合计行)
- Modify: `for_real_dummy/README.md`
- Create/Modify: `C:\Users\ericp\.claude\projects\e--Workspace-dummy\memory\os-concurrency-deep-dive-complete.md`
- Modify: `C:\Users\ericp\.claude\projects\e--Workspace-dummy\memory\MEMORY.md`
- Modify: `C:\Users\ericp\.claude\projects\e--Workspace-dummy\memory\feedback-function-by-function-teaching.md`

- [ ] **Step 1: 逐文件独立子进程重跑验证**

```bash
cd for_real_dummy/os-concurrency-deep-dive
for f in 01-processes-and-threads.md 02-cpu-scheduling.md 03-synchronization-primitives.md 04-advanced-concurrency-patterns.md 05-deadlock.md 06-virtual-memory-and-address-translation.md 07-memory-allocation-and-page-replacement.md 08-io-models.md 09-ipc.md 10-filesystem-foundations.md 11-modern-systems-topics.md 12-mock-interview-capstone.md; do
  echo "=== $f ==="
  python _verify_md.py "$f"
done
```
Expected: 全部 PASS。

- [ ] **Step 2: 结构标记计数核对**

```bash
grep -c "^## [0-9]" for_real_dummy/os-concurrency-deep-dive/0*.md for_real_dummy/os-concurrency-deep-dive/1[01].md
```
逐文件核对实际知识点数和 roadmap 表格声明的数字是否一致,不一致则修正 roadmap 表格为精确值(参照 statistics-deep-dive"先估后核"先例,允许最终精确数字和最初"约80"有合理浮动)。

- [ ] **Step 3: 死锁/竞态/epoll 类知识点额外复测稳定性**

```bash
for i in $(seq 1 5); do python _verify_md.py 03-synchronization-primitives.md && python _verify_md.py 05-deadlock.md; done
```

- [ ] **Step 4: 更新 `00-roadmap.md` 合计行为精确数字**

- [ ] **Step 5: 更新 `for_real_dummy/README.md`**

在"独立技能系列"表格新增一行(参照 dsa-deep-dive/statistics-deep-dive 行的格式),补一段说明本系列是"职业发展与需求四部曲"第1部、和 rhcsa-bash-deep-dive 的边界、双环境验证策略,目录树新增 `os-concurrency-deep-dive/` 条目。

- [ ] **Step 6: 创建 memory 文件 `os-concurrency-deep-dive-complete.md`**

frontmatter `type: project`,内容参照 `statistics-deep-dive-complete.md` 的结构:精确知识点数/板块文件数、七步模板(不同于统计的八步,注明原因)、双环境验证策略(和之前任何系列都不同的新模式)、和 rhcsa-bash-deep-dive 的边界、四部曲进度更新(操作系统与并发✅完成,计算机网络/数据库/系统设计排队中)。

- [ ] **Step 7: 更新 `MEMORY.md` 索引**

新增一行指向 `os-concurrency-deep-dive-complete.md`。

- [ ] **Step 8: 更新 `feedback-function-by-function-teaching.md`**

追加一段简短说明:四部曲第1部完成,后续排队项明确。

- [ ] **Step 9: 最终提交(确认 `git status`/`git diff` 只涉及本系列相关文件)**

```bash
git status --short for_real_dummy/os-concurrency-deep-dive/ for_real_dummy/README.md
git add for_real_dummy/os-concurrency-deep-dive/00-roadmap.md for_real_dummy/README.md
git commit -m "docs(os-concurrency): 全库自查回归 + README集成 + 收尾提交"
```

---

*创建:2026-07-13*
