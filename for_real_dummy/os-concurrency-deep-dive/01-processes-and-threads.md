# 01 进程与线程基础

> 板块 I:进程、线程与调度。本类是全系列的地基——后面调度、同步、内存管理几乎每一类都会用到这里建立的进程/线程/协程模型。

---

## 1. PCB与进程状态转换

**签名/是什么**

在讲 PCB 之前,先建立"进程"本身是什么——这是全系列第一次正式用到这个词,后面几乎每一类都会默认你已经理解它。**程序(Program)**是磁盘上一份静止的可执行文件,里面是编译好的指令和数据,不占用 CPU、不占用运行时内存,双击它或者在命令行敲它的名字之前,它永远只是那一份躺在磁盘上的文件,什么都不会做。**进程(Process)**是程序被启动、运行起来之后的产物:操作系统把这份静止的文件加载进内存,为它分配一块独立的内存空间、赋予它独立的执行状态(比如唯一的 PID、记录"现在执行到哪条指令"的程序计数器),这些东西合在一起构成一个正在运行的、动态的实例。一个具体的直觉例子:把同一份程序(比如同一个 `.exe`,或者同一份 Python 脚本)双击/启动两次,操作系统会创建出**两个独立的进程**——它们各自拿到自己的 PID、各自的内存空间,一个进程往自己的内存里写数据完全不会影响另一个进程看到的内容,彼此互不干扰。"同一份程序"和"运行这份程序产生的进程实例"是两个完全不同层次的概念,类似"图纸"和"按同一张图纸盖出来的房子"——图纸只有一份,但可以盖出很多栋互不相干的房子。

PCB(Process Control Block,进程控制块)就是操作系统内核为**每一个进程实例**(不是每一份程序)维护的一块数据结构,记录该进程的全部运行时元信息:PID、状态、程序计数器、寄存器快照、内存管理信息(**页表指针**——这个字段涉及虚拟内存机制,06 号文件会详细展开,这里只需要知道 PCB 记录了这样一个指针,指向该进程自己的页表)、打开文件表、**调度优先级**(决定这个进程在就绪队列里被调度器选中的先后顺序,02 号文件的 CPU 调度算法会详细展开优先级具体怎么起作用,这里只需要知道 PCB 上有这个字段)等。前面例子里"同一份程序运行出的两个独立进程各有自己的 PID 和内存"这件事,底层落地的方式正是操作系统为每个进程分配了独立的 PCB——两个进程实例对应两块独立的 PCB,互不共享任何字段。进程状态转换,就是这块数据结构里"状态"字段随事件驱动变化的过程。

**一句话**

PCB 是内核用来"认识"一个进程的唯一凭证——没有 PCB,进程在内核眼里就不存在。

**底层机制/为什么这样设计**

经典进程状态机有 5 个核心状态:新建(New)→就绪(Ready)→运行(Running)→阻塞/等待(Waiting)→终止(Terminated),就绪和运行之间可以来回切换(时间片用完/被抢占→就绪;被调度器选中→运行),运行和阻塞之间也可以来回切换(发起阻塞式系统调用如 `read()`→阻塞;等待的资源就绪→就绪)。这样设计的核心原因是**关注点分离**:调度器只需要维护"就绪队列"这一个数据结构就能决定"下一个跑谁",不需要感知每个进程具体在等什么;而"阻塞"状态把"暂时不需要 CPU 但还没死"这个语义单独区分出来,避免调度器把明明在等 IO 的进程也拿去轮询浪费 CPU。

把这几个状态之间的转换关系画出来会更直观——不是"新建→就绪→运行→阻塞→终止"这种教科书里常见的单向线性箭头,真实的转换关系是有来有回的:

```
主干(从创建到终止的单向主线):

┌─────┐       ┌───────┐       ┌─────────┐       ┌────────────┐
│ New │ ────▶ │ Ready │ ────▶ │ Running │ ────▶ │ Terminated │
└─────┘       └───────┘       └─────────┘       └────────────┘

Ready 和 Running 之间还有一条反向边,不是单向的:

┌───────┐       ┌─────────┐
│ Ready │ ◀──── │ Running │
└───────┘       └─────────┘
   时间片用完/被抢占(preempted)—— Running 掉回 Ready 重新排队,不是终点

Running 会先进入 Waiting,不会直接消失也不会直接回到 Ready:

┌─────────┐       ┌─────────┐
│ Running │ ────▶ │ Waiting │
└─────────┘       └─────────┘
   发起阻塞式系统调用,如 read()(blocks on I/O)

Waiting 结束后回到的是 Ready,不是直接回到 Running:

┌─────────┐       ┌───────┐
│ Waiting │ ────▶ │ Ready │
└─────────┘       └───────┘
   等待的资源就绪 / I/O 完成(event completes)—— 回去重新排队,要再被调度器选中一次才会运行
```

把四条边合在一起看:Ready ⇄ Running 是真正双向的(dispatch 和 preempted 各占一个方向);Running → Waiting → Ready 是一条"绕路"而不是原路返回(阻塞醒来不会直接变成"正在运行",要先回就绪队列重新排队等调度);只有 Running → Terminated 这一条边不会再流回任何状态,是整个状态机唯一的出口。

**AI研究/工程场景**

训练任务调度系统(如 Slurm、Kubernetes Job/Pod)本质上是同一个状态机思想在更粗粒度上的复刻:Pending(新建/排队)→Running(运行)→Succeeded/Failed(终止),中间还有类似"阻塞"语义的状态(比如等待 GPU 资源分配、等待数据挂载完成)。理解 PCB 状态机能直接理解为什么 K8s Job 会有 `Pending`/`Running`/`Failed` 这类状态设计,以及为什么"资源不够导致 Pending"和"程序本身在跑但在等 IO"是两个完全不同层面的问题。

**可运行例子**(验证环境:`.venv`)

```python
import subprocess
import time

p = subprocess.Popen(['python', '-c', 'import time; time.sleep(0.3)'])

state_right_after_start = p.poll()  # None 表示仍在运行(就绪/运行态的外部观测)
time.sleep(0.5)
state_after_done = p.poll()  # 非None 表示已终止,值是退出码

print('state_right_after_start=' + str(state_right_after_start))
print('state_after_done=' + str(state_after_done))

assert state_right_after_start is None, "process should still be running right after start (poll() returns None)"
assert state_after_done == 0, "process should have terminated with exit code 0 after its sleep(0.3) completes"
print("STATE_TRANSITION_TEST=PASS")
```

**面试怎么问+追问链**

- **决策依据追问轴**:"就绪"和"运行"为什么要分成两个状态,合并成一个不行吗?——追问:如果合并,单核 CPU 上"就绪"就等于"正在跑",但多个进程同时"就绪"意味着它们都在跑,这在单核上不可能;分开是为了让调度器能维护一个"候选池"(就绪队列)而不破坏"某一时刻单核只有一个进程在运行"这个物理约束。
- **诊断真实数据(新题型)**:给一段 `ps aux` 输出,某进程状态是 `D`(不可中断睡眠),问这和 `S`(可中断睡眠)有什么区别、什么场景会导致大量进程卡在 `D` 状态——追问:`D` 状态通常是在等待磁盘 IO 完成且内核不允许这类等待被信号打断(避免磁盘操作留下不一致状态),大量进程卡 `D` 态往往指向存储子系统故障或严重过载,`kill -9` 对 `D` 态进程是无效的(这是很多人不知道的坑)。

**常见坑**

- 把"进程状态"和"CPU 使用率"混为一谈:一个进程即使处于"运行"态,也可能因为在死循环里空转而看起来"很忙"但什么有用的事都没做;反过来一个进程"阻塞"在网络 IO 上,CPU 使用率是 0% 但进程完全健康。
- 以为 `kill` 一定能立刻终止进程:处于 `D`(不可中断睡眠)状态的进程无法响应包括 `SIGKILL` 在内的大多数信号,必须等它自己从不可中断的内核态调用中返回。

---

## 2. fork/exec/wait语义与写时复制COW

**签名/是什么**

`fork()` 是 Unix/Linux 创建新进程的核心系统调用:调用一次,返回两次——父进程里返回子进程 PID,子进程里返回 0。子进程是父进程在 `fork()` 那一刻的完整复制品(几乎所有内存、文件描述符表等)。`exec()` 系列系统调用(`execve` 等)用一个新程序完全替换当前进程的地址空间(PID 不变,代码/数据全部换掉)。`wait()`/`waitpid()` 让父进程阻塞等待子进程终止并回收其退出状态。COW(Copy-On-Write,写时复制)是 `fork()` 的实现优化:`fork()` 时不真的复制父进程全部内存,而是让父子进程共享同一块物理内存并标记为只读,只有当某一方真的尝试**写入**时,内核才为那一页分配新的物理内存并复制,是"惰性复制"。

**一句话**

`fork` 复制自己,`exec` 变成别人,`wait` 收尸,COW 是让 `fork` 不至于慢到没法用的关键优化。

**底层机制/为什么这样设计**

如果 `fork()` 真的立刻物理复制父进程全部内存(可能几百 MB 到几十 GB),那么每次 `fork()` 都会有巨大的延迟和内存开销,而现实中绝大多数 `fork()` 之后紧跟着就是 `exec()`(比如 shell 执行一条命令:先 `fork` 出子进程,子进程立刻 `exec` 成新程序,父进程刚复制出来的内存转眼就被整个丢弃),那份"复制"的工作完全是浪费。COW 让 `fork()` 只做最轻量的元数据操作(复制页表并把页表项标记为只读、共享),把真正的物理内存复制推迟到"确实有人要写"的那一刻才发生,而且往往连那一刻都不会到来(因为紧跟着 `exec` 直接换了地址空间)。这是操作系统设计里"惰性求值"思想的经典应用。

**AI研究/工程场景**

大模型推理服务用 `fork()` 启动 worker 进程是一个真实的性能技巧:主进程先把几十 GB 的模型权重加载进内存,再 `fork()` 出多个 worker 进程处理并发请求——因为 COW,子进程"拥有"完整模型权重的内存视图,但这些内存物理上仍然是共享的一份,不需要每个 worker 都重新反序列化加载一次模型(那可能要多花几十秒甚至几分钟)。只有当某个 worker 真的需要修改自己的一部分状态(比如 KV cache)时,对应那部分内存才会被复制,不会影响权重本身共享的部分。这也是为什么第 8 点会讲到 Windows 用 spawn(没有 COW)做多进程数据加载时,每个 worker 进程启动都明显更慢更吃内存。

**可运行例子**(验证环境:`WSL2 Rocky Linux`,Windows 无 `os.fork()`)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证(已单独确认通过,Windows 原生 Python 没有 os.fork())
import os

data = [1, 2, 3]
r, w = os.pipe()
pid = os.fork()
if pid == 0:
    # 子进程:修改 data 不应该影响父进程看到的 data
    os.close(r)
    data.append(999)
    os.write(w, str(data).encode())
    os.close(w)
    os._exit(0)
else:
    os.close(w)
    os.waitpid(pid, 0)  # 回收子进程,避免变成僵尸(见第7点)
    child_result = os.read(r, 1024).decode()
    os.close(r)
    print('parent_data=' + str(data))
    print('child_result=' + child_result)
    assert data == [1, 2, 3], "parent's data must be unaffected by child's modification (COW isolation)"
    assert child_result == '[1, 2, 3, 999]', "child must see its own independent modification"
    print("COW_TEST=PASS")
```

验证记录:2026-07-13 在 WSL2 Rocky Linux 10.2(`Python 3.12.13`)实测通过,输出 `parent_data=[1, 2, 3]` / `child_result=[1, 2, 3, 999]`,证明父子进程在 `fork()` 后确实拥有独立、互不影响的数据副本。

**面试怎么问+追问链**

- **底层机制追问轴(工程约束递增)**:`fork()` 之后子进程会继承父进程的哪些东西、不会继承哪些?——追问:继承打开的文件描述符(共享同一个文件偏移量,这是很多人会忽略的细节——父子进程对同一个已打开文件的读写会互相影响游标位置)、继承内存地址空间(COW)、继承信号处理方式;不继承的典型例子是文件锁(部分锁类型不继承)、定时器、未决信号。深挖到:如果父进程 `fork()` 前打开了一个文件写了一半,子进程和父进程之后各自再写,文件里的内容顺序会是什么样——这需要理解共享文件偏移量这一点才能正确回答。
- **真实性验证轴**:"我们用 fork 启动 worker 加速了模型加载"这种说法怎么验证是不是真的理解了 COW,而不是背了个词——追问:具体是怎么测的?fork 前后内存实际增长了多少(用 `/proc/[pid]/status` 里的 `VmRSS` 而不是 `VmSize` 才能看出物理内存真实增长,因为 COW 页面在 `VmSize` 里会被算进去但物理页面是共享的)。

**常见坑**

- 以为 `fork()` 之后父子进程完全独立、互不影响——文件描述符的偏移量是共享的,这是一个真实的坑:父进程 `fork()` 前打开文件写了一部分,`fork()` 后父子进程都往同一个 fd 写,会互相影响对方写入的位置,不是"各写各的"。
- 忘记 `wait()`/`waitpid()` 回收子进程导致僵尸进程堆积(见第 7 点),这是长期运行的服务里非常常见的资源泄漏。
- 把 COW 理解成"永远不会真正复制内存"——只要任意一方发生写操作,那一页就会被真实复制,如果父子进程都在大量随机写内存,COW 的省内存优势会迅速消失,反而因为额外的缺页中断处理增加开销。

---

## 3. 线程模型(1:1/N:1/M:N)与Python GIL的位置

**签名/是什么**

在讲三种线程模型之前,先建立"线程"本身最基础的定义(这是全系列第一次正式讲线程是什么)。**线程(Thread)**是进程内部的一条独立执行流:一个进程可以只有一条执行流(单线程),也可以同时拥有多条(多线程)。同属一个进程的多个线程,**共享这个进程的内存地址空间、打开文件表等资源**,但每个线程各自拥有独立的栈(stack)和独立的执行状态(比如自己的程序计数器、寄存器快照)。这是线程和进程最核心的区别:**进程之间默认相互隔离**(见第 2 点,一个进程看不到、碰不到另一个进程的内存),**同一进程内的线程默认相互共享内存**——这也是为什么线程间通信天然比进程间通信(见 09 类 IPC)简单直接(共享变量直接读写就行,不需要专门的通信机制),但代价是多个线程同时读写同一块共享内存如果不加保护就会出问题,这正是 03 类整整一类内容要解决的问题。

在这个基础定义之上,线程模型描述"用户态线程"和"内核态线程(可被内核调度器直接调度的执行单元)"之间的映射关系:1:1(每个用户线程对应一个内核线程,Linux/Windows 现代实现的主流方式)、N:1(多个用户线程映射到一个内核线程,内核完全看不到线程的存在,调度全在用户态完成)、M:N(M 个用户线程映射到 N 个内核线程,试图兼顾两者优点,Go 语言的 goroutine 调度器是典型例子)。GIL(Global Interpreter Lock,全局解释器锁)是 CPython 解释器的一把全局互斥锁,保证同一时刻只有一个线程在执行 Python 字节码。

**一句话**

线程模型决定"用户线程"能不能被操作系统真正并行调度,GIL 是 CPython 在"已经拿到了真内核线程"之后又人为加的一把锁,把并行度重新收窄成"同一时刻只有一个线程真的在跑 Python 代码"。

**底层机制/为什么这样设计**

CPython 的线程是标准的 1:1 内核线程(`threading` 模块底层就是操作系统原生线程),所以 CPython 线程能获得多核并行调度的资格——但 GIL 又把这份并行度锁死了。GIL 存在的根本原因是 CPython 的内存管理(尤其是引用计数)不是线程安全的:如果不加全局锁,多个线程同时对同一个对象做引用计数加减,会产生竞态条件(见 03 类知识点1),要么内存提前被释放要么永久泄漏。GIL 用一把粗粒度的锁换来了"不用给每个对象都加锁"的实现简单性,代价是 CPU 密集型任务无法用多线程获得真正的多核加速。但 GIL 会在执行阻塞式系统调用(如网络 IO、磁盘 IO、`time.sleep`)时被显式释放,所以 IO 密集型任务用多线程仍然能获得真实的并发收益。

**AI研究/工程场景**

这直接解释了为什么 PyTorch `DataLoader` 的 `num_workers` 参数默认用多进程而不是多线程做数据预处理:tokenization、图像增强这类 CPU 密集型操作如果放在多线程里做,会被 GIL 卡死在单核性能上,完全发挥不出机器的多核算力,必须用多进程(每个进程有自己独立的 GIL)才能真正并行。反过来,如果 `DataLoader` 的瓶颈是纯 IO(比如从远程对象存储读取数据文件),多线程反而是更轻量的选择,因为 IO 期间 GIL 会被释放。

**可运行例子**(验证环境:`.venv`)

```python
import threading
import time

def cpu_bound(n):
    x = 0
    for i in range(n):
        x += i * i
    return x

def io_bound():
    time.sleep(0.1)

def best_of(fn, trials):
    best = None
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best

N = 3_000_000

# CPU密集:多线程不应有明显加速(GIL卡死并行度)
t_cpu_single = best_of(lambda: (cpu_bound(N), cpu_bound(N)), trials=3)
def cpu_threaded():
    t1 = threading.Thread(target=cpu_bound, args=(N,))
    t2 = threading.Thread(target=cpu_bound, args=(N,))
    t1.start(); t2.start(); t1.join(); t2.join()
t_cpu_threaded = best_of(cpu_threaded, trials=3)
print('cpu_single=%.4f cpu_threaded=%.4f' % (t_cpu_single, t_cpu_threaded))
assert t_cpu_threaded >= t_cpu_single * 0.85, "GIL should prevent meaningful speedup for CPU-bound threading"

# IO密集:多线程应有明显加速(GIL在IO等待时释放)
t_io_single = best_of(lambda: [io_bound() for _ in range(4)], trials=2)
def io_threaded():
    threads = [threading.Thread(target=io_bound) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
t_io_threaded = best_of(io_threaded, trials=2)
print('io_single=%.3f io_threaded=%.3f speedup=%.2fx' % (t_io_single, t_io_threaded, t_io_single / t_io_threaded))
assert t_io_threaded < t_io_single * 0.5, "IO-bound threading should show real speedup since GIL releases during blocking IO"
print("GIL_TEST=PASS")
```

**面试怎么问+追问链**

- **方案批判迭代轴**:"我们的数据预处理慢,我加了多线程"——追问1:预处理具体是 CPU 密集(tokenize/resize/augment)还是 IO 密集(读文件)?候选人如果说"CPU 密集但确实用多线程,而且我测过确实变快了",继续追问:是不是预处理里其实调用了 numpy/Pillow 这类底层用 C 实现、会主动释放 GIL 的库?——很多"多线程加速了 CPU 密集任务"的真实案例,其实是因为热点代码在 GIL 之外的 C 扩展里跑,不是 Python 字节码本身被并行了,这是一个需要精确定位加速来源的经典追问。
- **决策依据追问轴**:Python 3.13 引入了可选的"无 GIL"构建(PEP 703),为什么不是所有人立刻切过去?——追问:去掉 GIL 意味着所有 C 扩展都需要重新审视线程安全性(很多现有 C 扩展隐式依赖 GIL 提供的保护),生态迁移成本极高,短期内单线程性能也可能因为需要更细粒度的锁而略有下降。

**常见坑**

- 以为"用了 `threading` 模块"就等于"用了多核" —— CPython 的多线程只在 IO 密集场景下有真实并发收益,CPU 密集场景需要 `multiprocessing` 或者依赖底层 C 扩展释放 GIL。
- 混淆"线程模型的 1:1/N:1/M:N"和"GIL"是同一回事——GIL 是 CPython 解释器实现的产物,和操作系统线程调度模型是两个独立的层次,即便线程是货真价实的 1:1 内核线程(能被多核调度),GIL 依然可以把并行度锁掉。

---

## 4. 协程与用户态调度(有栈vs无栈)

**签名/是什么**

协程(coroutine)是一种用户态(不需要内核参与)的协作式调度执行单元:协程之间的切换由程序自己在明确的"让出点"主动触发,而不是像线程那样由操作系统随时抢占。"有栈协程"(如 Lua 的 coroutine、Go 的 goroutine)每个协程拥有独立的调用栈,可以在调用栈的任意深度挂起;"无栈协程"(如 Python 的 `async`/`await`、C++20 coroutine)没有独立调用栈,状态机由编译器/解释器改写成显式的状态保存结构,只能在特定语法标记的点(`await`)挂起。

**一句话**

协程用"我自己说了算什么时候让出 CPU"替代了线程"操作系统随时可能把我切走",换来极低的切换开销和可预测性,代价是一旦某个协程忘记让出就会卡死整条事件循环。

**底层机制/为什么这样设计**

线程切换需要陷入内核态(保存/恢复寄存器、页表切换、可能的缓存失效),开销通常在微秒级;协程切换纯粹是用户态的函数调用/跳转,开销在几十到几百纳秒级,快 1-2 个数量级,这使得单机支持成千上万并发协程成为可能(而线程数超过几千就会因为切换开销和内存占用而不现实,这也是 08 类"C10K 问题"的核心动机之一)。协作式调度(不被随时抢占)反过来意味着两点:好处是"临界区内不需要担心被随时打断",很多协程框架里操作共享状态不需要加锁;坏处是一旦某个协程执行了阻塞操作或者陷入死循环而不让出,整条事件循环上的所有其他协程都会被饿死,这是抢占式的线程/进程调度完全不会遇到的问题类别。

**AI研究/工程场景**

大规模 LLM 推理服务(如 vLLM、TGI 这类框架)的请求调度大量使用 `asyncio`:每个到达的推理请求是一个协程,在等待 GPU 计算的间隙(比如等待批处理凑够一批、等待网络 IO 把结果传回客户端)主动让出控制权,让调度器去处理下一个请求的到达或响应发送,而不是给每个并发请求开一个操作系统线程(几千个并发连接开几千个线程,内存和切换开销都扛不住)。

**可运行例子**(验证环境:`.venv`)

```python
import asyncio

order = []

async def worker(name, steps):
    for i in range(steps):
        order.append((name, i))
        await asyncio.sleep(0)  # 显式让出控制权(协作式调度的核心)

async def main():
    await asyncio.gather(worker('A', 3), worker('B', 3))

asyncio.run(main())
print(order)
# 协作式调度:每次显式await都会交替执行
assert order == [('A', 0), ('B', 0), ('A', 1), ('B', 1), ('A', 2), ('B', 2)], \
    "cooperative scheduling should interleave exactly at each explicit await point"
print("COOP_TEST=PASS")

# 对比:循环内部没有await点,协程会一次性跑完自己的同步部分才轮到下一个(不像抢占式线程可能随时被切走)
order2 = []
async def worker_no_yield(name, steps):
    for i in range(steps):
        order2.append((name, i))
    await asyncio.sleep(0)

async def main2():
    await asyncio.gather(worker_no_yield('A', 3), worker_no_yield('B', 3))

asyncio.run(main2())
print(order2)
assert order2 == [('A', 0), ('A', 1), ('A', 2), ('B', 0), ('B', 1), ('B', 2)], \
    "without an await point inside the loop, one coroutine runs its synchronous portion to completion before the other starts"
print("NO_YIELD_TEST=PASS")
```

**面试怎么问+追问链**

- **诊断真实数据(新题型)**:线上服务用 `asyncio` 写的,监控显示某个时间段所有请求延迟突然一起飙升然后一起恢复,不是逐渐劣化——追问:这种"整体同步卡顿"的模式高度提示某个协程在事件循环里执行了同步阻塞操作(比如误用了同步的 `requests` 库而不是异步 HTTP 客户端,或者做了一次没有 `await` 的重计算),把整条事件循环独占了,直到那个操作结束所有协程才恢复——这是协作式调度特有的故障模式,线程池不会有"一个请求卡住导致全部请求一起卡住"这种表现。
- **规模递增轴**:同样是"支持大量并发连接",为什么不直接开等量的操作系统线程?——追问:估算一下 1 万个并发连接分别用线程(每个线程栈通常几百 KB 到几 MB,还有内核调度开销)和协程(每个协程状态通常几 KB)大致需要多少内存,量级差异有多大。

**常见坑**

- 以为协程"自动"就是并发/并行的——协程只有在正确使用 `await` 主动让出的前提下才能实现并发,一旦某处误用了同步阻塞调用,协程模型反而会表现得比线程更差(整体卡死而不是单个变慢)。
- 混淆"协程"和"生成器"——Python 的 `async def` 函数和普通生成器(`yield`)有相似的底层机制(都是可挂起的执行帧),但语义定位不同,`asyncio` 的协程是为了并发调度设计的,普通生成器主要是为了惰性求值/迭代器协议设计的。

---

## 5. 线程私有数据TLS

**签名/是什么**

TLS(Thread-Local Storage,线程私有存储)让同一个变量名在不同线程里各自持有独立的值,互不干扰。Python 用 `threading.local()` 提供这一机制:创建一个 `threading.local()` 实例后,每个线程对它的属性读写都是该线程私有的,即使多个线程访问的是同一个 Python 对象。

**一句话**

TLS 是"同一个变量名,每个线程各拿一份自己的存储格子",不需要每次访问都显式传参或加锁。

**底层机制/为什么这样设计**

如果不用 TLS,想让每个线程维护"只属于自己"的状态(比如数据库连接、随机数生成器、事务上下文),要么得显式地在函数调用链的每一层传递这个状态(侵入性强,容易漏传),要么得用一个全局字典以线程 ID 为 key 存取(容易忘记清理导致内存泄漏,且没有语言层面的保护)。TLS 把"按线程隔离存储"这件事做成语言/运行时层面的一等公民:底层实现上,`threading.local()` 实例其实是维护了一个"每个线程 ID → 独立属性字典"的映射,对该实例的属性访问会自动被路由到当前线程对应的那份字典,业务代码完全感知不到这层路由,写起来就像操作一个普通对象。

**AI研究/工程场景**

多线程模型服务/训练场景下,每个线程如果需要维护自己独立的随机数生成器状态(避免多线程共享同一个 RNG 导致的竞态和不可复现)、或者独立的推理/训练上下文(比如某些框架的线程本地 CUDA stream 句柄),TLS 是标准做法——numpy 的旧版全局随机状态在多线程下就有过这类问题,后来推荐显式用 `numpy.random.default_rng()` 创建每个线程/进程独立的生成器实例,思路上和 TLS 是同一类"避免共享可变状态"的解决方案。

**可运行例子**(验证环境:`.venv`)

```python
import threading
import time

local_data = threading.local()
results = {}

def worker(name, value):
    local_data.x = value
    time.sleep(0.05)  # 故意留出时间窗口,如果TLS隔离失败,其他线程的写入会在这里"污染"当前线程看到的值
    results[name] = local_data.x

threads = [threading.Thread(target=worker, args=('t%d' % i, i)) for i in range(5)]
for t in threads: t.start()
for t in threads: t.join()

print(results)
for i in range(5):
    assert results['t%d' % i] == i, "each thread must see only its own value stored via threading.local(), not another thread's write"
print("TLS_TEST=PASS")
```

**面试怎么问+追问链**

- **底层机制追问轴**:`threading.local()` 是怎么做到"同一个对象,不同线程看到不同值"的,是不是每个线程有自己的一份对象拷贝?——追问:不是拷贝,是同一个 `threading.local()` 实例内部维护了一个按线程 ID 索引的存储结构,对象本身在内存里只有一份,但它的属性访问被动态路由到"当前调用线程"对应的那格存储,这也是为什么把 `threading.local()` 实例通过参数传给另一个线程也没用——真正起作用的是"访问它的是哪个线程",不是"引用它的变量在哪个线程创建的"。
- **真实性验证轴**:"我们用 TLS 解决了多线程下的连接池问题"——追问:具体是每个线程一个数据库连接,还是线程池复用连接?如果是线程池(线程会被复用处理不同任务),TLS 存的连接什么时候需要被清理/归还,会不会导致连接泄漏或者被不同任务误用同一个仍处于事务中的连接。

**常见坑**

- 在使用了线程池(线程会被反复复用执行不同任务)的场景下,忘记 TLS 数据是"跟线程走"不是"跟任务走"——上一个任务在某个线程里设置的 TLS 值,如果没有清理,会被复用同一线程的下一个任务意外看到。
- 以为 `threading.local()` 能保护"跨线程共享的可变对象"——TLS 解决的是"同名变量各线程独立"问题,如果多个线程本来就是要读写同一个共享对象(比如同一个列表),TLS 完全不适用,该用锁的地方还是要用锁(见 03 类)。

---

## 6. 进程vs线程vs协程选型决策

**签名/是什么**

在并发执行方案里,进程、线程、协程三者是不同粒度的执行单元,选型决策就是根据任务的 CPU/IO 特性、隔离性要求、规模量级,决定用哪一种(或者组合)。

**一句话**

CPU 密集选进程,IO 密集且规模不大选线程,IO 密集且规模巨大选协程,需要故障隔离/资源限制选进程。

**底层机制/为什么这样设计**

回顾第 3 点已经建立的基础定义:线程共享所属进程的地址空间,进程之间则默认相互隔离——这里把这条基础特征代入选型决策,看它具体怎么变成"隔离性 vs 开销"这条光谱上的权衡。三者的核心差异来自这条光谱的三个点:进程有独立地址空间(一个进程崩溃不会拖垮另一个,天然隔离,配合操作系统的资源限制机制如 cgroups 还能做硬性资源配额),但创建/切换开销最大;线程共享地址空间(通信方便,不需要 IPC),创建/切换开销中等,但一个线程的野指针可以破坏整个进程;协程开销最小(用户态调度,见第 4 点),但完全没有隔离性,且只在 IO 密集、单核也能应付的场景下有意义(CPU 密集任务丢给协程完全没用,因为协程不解决"谁在用 CPU"的问题,只解决"IO 等待期间谁在用 CPU"的问题)。

**AI研究/工程场景**

一个典型的数据处理 pipeline 选型决策链:从远程对象存储下载数据文件(IO 密集,量级可能上千并发)用协程或者线程池;下载完成后做 tokenization/图像增强(CPU 密集,受 GIL 限制,见第 3 点)必须用多进程;如果流水线某一环节是调用一个不稳定的第三方模型服务、经常超时或崩溃,把它隔离到独立进程里(甚至独立容器)避免拖垮主流程,是隔离性优先于性能的选型决策。

**可运行例子**(验证环境:`.venv`)

```python
import threading
import asyncio
import multiprocessing as mp
import time

N = 8

def io_task_toplevel():
    time.sleep(0.1)

def thread_version():
    threads = [threading.Thread(target=io_task_toplevel) for _ in range(N)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    return time.perf_counter() - t0

async def coro_version_inner():
    await asyncio.gather(*[asyncio.sleep(0.1) for _ in range(N)])

def coro_version():
    t0 = time.perf_counter()
    asyncio.run(coro_version_inner())
    return time.perf_counter() - t0

def process_version():
    procs = [mp.Process(target=io_task_toplevel) for _ in range(N)]
    t0 = time.perf_counter()
    for p in procs: p.start()
    for p in procs: p.join()
    return time.perf_counter() - t0

if __name__ == "__main__":
    t_thread = min(thread_version() for _ in range(3))
    t_coro = min(coro_version() for _ in range(3))
    t_proc = min(process_version() for _ in range(3))
    print('thread=%.3f coro=%.3f process=%.3f' % (t_thread, t_coro, t_proc))
    # 线程和协程在这种轻量IO负载下都接近0.1s的理论下限,开销差异被噪声淹没,不适合断言二者严格大小关系;
    # 稳健、有教学意义的结论是:进程创建开销显著高于二者(隔离性是有代价的)。
    assert t_proc > t_thread * 3, "process creation overhead should be dramatically higher than thread for this lightweight IO task"
    assert t_proc > t_coro * 3, "process creation overhead should be dramatically higher than coroutine for this lightweight IO task"
    assert t_thread < 0.2 and t_coro < 0.2, "thread and coroutine overhead should both stay close to the fundamental 0.1s IO-wait floor"
    print("SELECTION_BENCHMARK_TEST=PASS")
```

**面试怎么问+追问链**

- **方案批判迭代轴**:候选人说"IO 密集场景我全部用协程"——追问1:如果这个 IO 密集任务里混了一小段 CPU 密集的同步计算(比如响应体要做一次 JSON 大对象的深度校验),会发生什么?候选人如果说"没关系,反正很快"——追问2:多快算快?这一小段同步计算执行期间,整条事件循环上其他所有协程都会被阻塞(见第 4 点常见坑),如果这段计算恰好在高并发时被放大成明显的尾延迟,协程模型的"单点卡顿全局遭殃"特性会比线程模型更容易被这类隐藏的同步代码点拖垮。
- **规模递增轴**:10 个并发请求、1000 个并发请求、10 万个并发连接,选型会不会变?——追问:10 个规模下几乎选什么都无所谓;1000 左右线程池仍然可行但开始有明显的内存/切换开销;10 万级别几乎只有协程(或者进一步用多进程+每进程内协程的组合架构)能扛住,这条追问链本质是在检验候选人是不是真的理解"量级"如何改变最优解,而不是死记"IO 用协程"这句结论。

**常见坑**

- 把"协程比线程快"当成普适真理不加条件地使用——协程只在"IO 等待"场景下体现优势,CPU 密集任务里协程和单线程没有本质区别(甚至因为额外的调度开销更慢)。
- 选型只考虑性能不考虑故障隔离——把一个不稳定的第三方依赖用最快的协程/线程方式集成进主服务,一旦它崩溃或内存泄漏,会直接拖垮整个进程,这时候用进程级隔离(哪怕慢一点)可能是更负责任的工程决策。

---

## 7. 僵尸进程与孤儿进程

**签名/是什么**

僵尸进程(Zombie):子进程已经终止,但父进程还没有调用 `wait()`/`waitpid()` 回收它的退出状态,内核会保留这个已终止进程的 PCB(主要是退出码等少量信息)不释放,`ps` 里显示为 `Z` 或 `<defunct>`。孤儿进程(Orphan):父进程先于子进程退出,子进程会被"过继"给一个 reaper 进程(现代 Linux 上通常最终是 PID 1 / systemd,但也可能是设置了 `PR_SET_CHILD_SUBREAPER` 的中间进程先接手)继续运行,不会因为原父进程消失而受影响。

**一句话**

僵尸进程是"死了但没人来收尸",孤儿进程是"爹没了但自己还活得好好的、换了个新监护人"。

**底层机制/为什么这样设计**

子进程终止后,内核不能立刻彻底销毁它的全部记录,因为父进程可能还需要通过 `wait()` 查询它的退出码(这是 `fork`/`wait` 编程模型的基本约定——父进程有权知道子进程是怎么结束的)。所以内核把"进程彻底消失"拆成两步:进程终止(资源大部分释放,但 PCB 里的退出信息保留)→ 父进程 `wait()` 回收(PCB 彻底释放)。如果父进程一直不 `wait()`,子进程就一直卡在"僵尸"这个中间态,虽然不占用内存/CPU 等实质资源,但会占用进程表里的一个 PID 槽位,大量堆积会导致系统无法创建新进程。孤儿进程被重新过继的设计,是为了保证"任何进程终止时都应该有一个进程负责 `wait()` 它"这一不变量始终成立——如果不这样设计,父进程先退出的子进程会永远没有人来给它"收尸",就会永久变成僵尸。

**AI研究/工程场景**

长时间运行的训练脚本如果通过 `fork()`/`subprocess` 派生数据加载或预处理子进程,一旦父进程的清理逻辑有 bug(比如异常路径下跳过了 `wait()`),僵尸进程会随着训练轮次的推进不断堆积,虽然不直接吃内存,但会逐渐耗尽系统的 PID 空间,最终表现为"莫名其妙无法创建新进程/新线程"这种难以第一时间联想到根因的故障——这是一个真实、容易被误诊为"内存泄漏"的运维陷阱。

**可运行例子**(验证环境:`WSL2 Rocky Linux`)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证(已单独确认通过)
import os
import time

# 第一部分:僵尸进程 - 子进程退出但父进程故意不wait
pid = os.fork()
if pid == 0:
    os._exit(0)
else:
    time.sleep(0.5)  # 给子进程时间退出,父进程故意还不wait
    with open('/proc/%d/stat' % pid) as f:
        stat = f.read()
    state_char = stat.split(') ')[1].split(' ')[0]
    print('zombie_state_char=' + state_char)
    assert state_char == 'Z', "child should show as zombie (Z) before parent reaps it with wait()"
    os.waitpid(pid, 0)  # 现在才回收,僵尸消失
    print("ZOMBIE_TEST=PASS")
```

验证记录(第二部分,孤儿进程):用另一段脚本验证——子进程 fork 后先忽略 `SIGHUP`(否则父进程退出时子进程可能被连带终止,这是调试过程中真实发现的坑,见"常见坑"),然后轮询 `os.getppid()` 直到其发生变化;父进程 fork 后立刻 `os._exit(0)`。2026-07-13 在 WSL2 Rocky Linux 10.2 实测:原父进程 PID 576,父进程退出后子进程观测到自己的 `ppid` 变为 574(`changed=True`),证明孤儿进程确实被重新挂到了别的进程下继续运行,不会因为原父进程消失而被杀死。

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证(已单独确认通过)
import os
import time
import signal

original_parent = os.getpid()
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # 见"常见坑":不忽略SIGHUP,子进程可能提前被杀
    start = time.time()
    new_ppid = os.getppid()
    while new_ppid == original_parent and time.time() - start < 4:
        time.sleep(0.05)
        new_ppid = os.getppid()
    changed = new_ppid != original_parent
    print('original_parent=%d new_ppid=%d changed=%s' % (original_parent, new_ppid, changed))
    assert changed, "child's ppid must change after the original parent exits (orphan reparenting)"
    print("ORPHAN_TEST=PASS")
else:
    os._exit(0)
```

**面试怎么问+追问链**

- **诊断真实数据(新题型)**:`ps aux` 显示几百个 `<defunct>` 进程,同一个父 PID,怎么排查、怎么修?——追问:先确认父进程代码里是不是有路径 `fork()` 之后没有配对 `wait()`(常见于异常处理分支遗漏);修复思路除了补全 `wait()`,还可以让父进程忽略 `SIGCHLD` 信号(`signal.signal(signal.SIGCHLD, signal.SIG_IGN)`,这会让内核自动回收子进程无需显式 wait,是一个很多人不知道的技巧)或者用双重 `fork()` 让孙进程直接被 init 收养从而父进程完全不需要关心它的生命周期。
- **规模递增/工程约束递增轴**:单机场景下漏 `wait()` 只是慢慢堆积僵尸,如果这是一个用 `fork()` 派生 worker 的分布式服务,大规模部署下会有什么额外风险?——追问:每台机器的 PID 上限是有限资源(Linux 默认几万到十几万,可配置),集群规模越大、单机僵尸堆积速度越快,加上容器化场景下容器内 PID 1 进程通常没有正确实现"收养并回收孤儿/僵尸"这个 init 系统该做的事(这是很多人踩过的真实容器坑,现在有 `tini`/`dumb-init` 这类专门解决这个问题的极简 init 工具)。

**常见坑**

- 以为孤儿进程一定会被过继给 PID 1(systemd/init):现代 Linux 允许任意设置了 `PR_SET_CHILD_SUBREAPER` 的祖先进程先接手当 reaper,不一定是 PID 1(本知识点验证时实测过继给的是调用链上的中间进程,不是 PID 1 本身)。
- 以为父进程退出后,还没退出的子进程一定能继续独立运行下去:如果子进程没有显式忽略或处理 `SIGHUP`,在某些会话/终端管理场景下(比如本例最初在非交互式 `wsl.exe` 调用链里复现时)子进程可能在父进程退出的同时被连带终止,这不是"孤儿进程机制本身"的问题,而是"控制终端/会话关闭时是否向前台进程组发送 SIGHUP"这一独立机制叠加造成的——真实调试中先忽略这一步导致验证失败了三次,后来显式 `signal.signal(signal.SIGHUP, signal.SIG_IGN)` 才稳定复现,这本身就是一个值得记住的真实坑:孤儿化(reparenting)本身不会杀死子进程,但会话终止时的 SIGHUP 广播可能会。
- 容器场景里以 PID 1 运行的应用进程,如果没有正确处理"收养孤儿/回收僵尸"的职责(大部分应用程序本身根本不是为了当 init 系统设计的),容器内会持续堆积僵尸进程,这是"为什么容器最佳实践建议用 tini 之类的极简 init 作为 PID 1"这一问题的真实答案。

---

## 8. Windows进程创建模型对比(CreateProcess/spawn vs fork)

**签名/是什么**

Windows 没有 `fork()` 系统调用,创建新进程统一通过 `CreateProcess()` API:总是从磁盘上的可执行文件重新启动一个全新进程,不存在"复制当前进程内存"这个概念。Python 的 `multiprocessing` 模块在 Windows 上默认使用 `spawn` 启动方式,原理上对应 `CreateProcess`:子进程是一个全新启动的 Python 解释器,重新 `import` 目标模块,不会继承父进程当前的运行时状态,只会拿到显式通过参数传递(且必须可被 `pickle` 序列化)的数据。

**一句话**

`fork` 是"复制正在跑的自己接着跑",`CreateProcess`/`spawn` 是"重新启动一个全新的自己,只带走你明确塞给它的行李"。

**底层机制/为什么这样设计**

Windows 的进程模型设计哲学和 Unix 不同:Unix 的 `fork`+`exec` 两阶段模型源于"先复制再替换"的历史设计(复制的开销后来用 COW 解决,见第 2 点),而 Windows 从一开始就选择了"一步到位直接启动新程序"的 `CreateProcess`,没有"先克隆自己再变成别人"这个中间阶段,自然也没有对应的 COW 优化空间。这导致两个直接后果:第一,Windows 上创建进程天然比 Linux 上 `fork()` 更慢(需要重新初始化解释器、重新执行模块顶层代码),第二,子进程需要的任何数据都必须显式序列化传递(不能像 fork 那样"顺手"就带着父进程当前的内存状态),这也是为什么 Python `multiprocessing` 在 Windows 上要求目标函数必须是模块顶层可 `import` 的(可 `pickle`)对象,不能是闭包里的局部函数。

**AI研究/工程场景**

这个仓库本身就跑在 Windows 上,是一个会被真实踩到的坑:PyTorch `DataLoader(num_workers>0)` 在 Windows 上用 `spawn` 启动 worker 进程,如果不把 `DataLoader` 的创建和训练循环放在 `if __name__ == "__main__":` 保护之下,每个新 spawn 出来的 worker 进程会重新执行整个脚本顶层代码,又会看到 `DataLoader(num_workers>0)` 这行代码从而再次尝试 spawn 更多 worker——导致无限递归启动进程直到系统资源耗尽,这是 Windows 用户几乎必然会遇到一次的经典报错,而 Linux 用户因为默认用 `fork`(不重新执行脚本)几乎不会遇到。

**可运行例子**(验证环境:`.venv`,本例专门在 Windows 原生 Python 下运行以体现 spawn 语义)

```python
import multiprocessing as mp
import time

counter = [1, 2, 3]

def child_fn(q):
    # spawn 会重新 import 这个模块,子进程看到的 counter 是模块顶层的初始值,
    # 不会看到 __main__ 里对 counter 做的运行时修改(这一点和 fork 的 COW 语义完全不同)
    q.put(list(counter))

if __name__ == "__main__":
    counter.append(999)  # 在 spawn 子进程创建之前,对 counter 做运行时修改
    q = mp.Queue()
    p = mp.Process(target=child_fn, args=(q,))
    t0 = time.perf_counter()
    p.start()
    result = q.get()
    p.join()
    spawn_overhead = time.perf_counter() - t0
    print('parent_counter=' + str(counter))
    print('child_saw=' + str(result))
    print('spawn_overhead_sec=%.4f' % spawn_overhead)
    assert result == [1, 2, 3], \
        "spawn re-imports the module fresh; child must NOT see the parent's runtime mutation (999) made before start(), unlike fork's COW snapshot"
    assert counter == [1, 2, 3, 999], "the parent's own list is of course still mutated in the parent process"
    print("SPAWN_SEMANTICS_TEST=PASS")
```

验证记录:2026-07-13 在仓库 `.venv`(Windows 原生)实测,`child_saw=[1, 2, 3]`(没有看到父进程运行时追加的 999),`spawn_overhead_sec` 实测约 0.17 秒——对比第 6 点里同样是"创建一个子执行单元"但用协程/线程只需要几毫秒量级,进程创建的开销差了近两个数量级,这也是第 6 点"进程创建开销显著更高"这一结论的具体数字来源之一。

**面试怎么问+追问链**

- **真实性验证轴**:候选人说"在 Windows 上用多进程做数据加载踩过坑"——追问:具体报的是什么错、根因是什么?正确回答需要提到 spawn 会重新执行脚本顶层代码导致的递归启动问题,以及 `if __name__ == "__main__":` 保护为什么能解决这个问题(保护住的是"只有直接运行这个脚本时才创建 DataLoader/启动多进程",被 spawn 重新 import 时 `__name__` 不等于 `"__main__"`,那段代码不会再被执行)。
- **决策依据追问轴**:同样是"用 fork 还是 spawn",Python 的 `multiprocessing` 在 Linux 上其实也可以手动指定用 `spawn`(而不是默认的 `fork`),什么场景下即使在 Linux 上也应该主动选 `spawn`?——追问:如果父进程持有一些不适合被子进程"意外继承"的状态(比如打开的网络连接、锁、线程),用 fork 的 COW 语义会让子进程"莫名其妙"拿到一份这些状态的快照,可能引发难以调试的 bug(比如子进程继承了父进程一个已经加锁的 mutex,但没有对应释放它的线程,导致死锁);`spawn` 强制"显式传递你需要的一切",虽然慢,但更安全、更可预测,Python 官方文档也建议在混用了线程和多进程的复杂程序里优先用 `spawn` 而不是 `fork`。

**常见坑**

- 在 Windows 上用 `multiprocessing` 时,把目标函数写成闭包里的局部函数(不是模块顶层定义)——`spawn` 需要 `pickle` 序列化目标函数,局部函数无法被 `pickle`,会报 `AttributeError: Can't get local object`(这是撰写本知识点验证代码时真实踩到并修正的坑,不是编造的场景)。
- 忘记给 Windows 上的 `multiprocessing` 入口代码加 `if __name__ == "__main__":` 保护,导致 spawn 出的子进程重新执行整个脚本,引发无限递归启动子进程。
- 把 `spawn` 的"更慢"简单理解为"更差",忽视它"更安全/更可预测"(不会意外继承父进程不该继承的状态)这个真实优势——这也是为什么 Python 3.14 起 Linux 上 `multiprocessing` 的默认启动方式也从 `fork` 改成了 `forkserver`/更安全的方式,不是 Windows "落后",而是 fork 语义本身在复杂多线程程序里有真实的安全隐患。

---

*本文件 8 个知识点,验证环境:`.venv`(1,3,4,5,6,8 共 6 点)+ `WSL2 Rocky Linux`(2,7 共 2 点)。*
