"""操作系统深水区（终面深水区，约 12 个 DeepPoint）。

覆盖：死锁生产环境定位（jstack/gdb）、内存泄漏定位思路（复现-二分-对拍）、上下文切换开销测量与
过度并发降低吞吐、缺页中断真实性能影响、CFS 调度公平性权衡，以及 OOM killer、syscall 开销、
虚拟内存 thrashing、自旋锁/futex、epoll ET/LT、false sharing 等相邻真实生产topic。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import DeepPoint, categories, drill, grade_chain  # noqa: E402

CAT = "操作系统深水"

BANK: list[DeepPoint] = [
    DeepPoint(
        id="dp-be-os-01", cat=CAT,
        trigger="你说线上出现过死锁，具体怎么定位到的？",
        chain=(
            ("死锁的四个必要条件是什么？",
             "互斥条件、请求与保持条件、不可剥夺条件、循环等待条件，四者同时成立才会发生死锁。",
             ("互斥", "请求与保持", "不可剥夺", "循环等待")),
            ("光知道四个条件，线上出问题时你怎么去确认真的是死锁，而不是普通的慢查询/GC停顿？",
             "用jstack对Java进程连续dump两三次线程栈，对比是否有线程始终处于BLOCKED状态且持有锁的"
             "对象一致；如果只是偶尔卡顿波动应该先排除GC(用gc log/jstat)和IO等待。",
             ("jstack", "BLOCKED", "线程栈", "GC")),
            ("jstack dump出来一堆线程都在等锁，你怎么从里面找出真正的死锁环，而不是只是锁竞争激烈？",
             "jstack实际会自动在输出末尾标注\"Found one Java-level deadlock\"，扫描"
             "\"waiting to lock <0x...>\" 和 \"which is held by\" 配对形成环；如果没有自动识别"
             "（比如跨JVM/native锁），要手工画图：每个线程持有什么锁、在等什么锁，构造有向图找环。",
             ("Found one Java-level deadlock", "waiting to lock", "held by", "有向图")),
            ("如果是C++/native进程没有jstack这种工具，死锁了怎么办？",
             "用gdb attach上去，thread apply all bt 打印所有线程调用栈，找出卡在"
             "pthread_mutex_lock/futex wait的线程，再结合/proc/<pid>/task/*/stack反推持有关系；"
             "容器场景下要先进容器namespace再gdb，或用crictl/nsenter定位真实pid。",
             ("gdb", "attach", "thread apply all bt", "futex", "nsenter")),
        ),
        pitfall="很多人只会说'用jstack看'，说不清楚怎么从dump里精确定位死锁环，也答不出native进程没有"
                "jstack时怎么办",
        real_world_link=""),

    DeepPoint(
        id="dp-be-os-02", cat=CAT,
        trigger="你刚提到破坏循环等待条件可以预防死锁，实际系统里怎么落地？",
        chain=(
            ("破坏循环等待具体怎么做？",
             "给所有锁资源全局排序编号，规定任何线程加锁时必须按编号递增顺序申请，不允许交叉持有后"
             "申请一个编号更小的锁。",
             ("全局排序", "编号递增", "递增顺序")),
            ("如果系统里锁很多、来自不同模块，全局排序谁来维护？这个方案有什么代价？",
             "需要在代码规范/CI层面强制检查（比如静态分析工具扫描加锁顺序），代价是牺牲了模块间的"
             "独立性——原本B模块不需要知道A模块的锁编号，现在耦合到了一起；新增锁时要给全局排序表打"
             "补丁，容易遗漏。",
             ("静态分析", "独立性", "新增锁")),
            ("如果两个锁分别来自两个不同的第三方库，你没法改它们的加锁顺序，这时候还能怎么破坏死锁"
             "条件？",
             "改破坏\"不可剥夺\"或\"请求与保持\"——用tryLock+超时代替阻塞式lock，拿不到锁就主动释放"
             "已持有的锁再重试(back off)，本质是把无限等待变成有限重试，避免循环等待固化成死锁。",
             ("tryLock", "超时", "back off", "请求与保持")),
            ("生产环境里用tryLock+超时重试，你怎么判断超时时间设多少合适？设短了/长了分别有什么"
             "问题？",
             "超时太短会导致大量线程空转重试，CPU浪费、可能形成活锁(livelock)；太长则死锁发生时业务"
             "卡顿时间长、影响可用性；实践中通常结合指数退避+抖动(jitter)设置基础超时，并配合监控"
             "告警锁等待时长的P99，超过阈值触发人工介入。",
             ("指数退避", "抖动", "活锁", "P99")),
        ),
        pitfall="第2层容易漏说'跨第三方库没法排序'这种真实约束，第3层很多人想不到用tryLock+超时来"
                "替代全局排序",
        real_world_link=""),

    DeepPoint(
        id="dp-be-os-03", cat=CAT,
        trigger="你说排查过内存泄漏，具体怎么定位的，不是说说'用valgrind'就完了吧？",
        chain=(
            ("内存泄漏的基本定位思路是什么？",
             "先确认能稳定复现(观察内存曲线是否单调上升、是否和特定业务操作相关)，复现路径越短越好；"
             "然后用二分法逐步排除模块，定位到具体是哪个组件/接口触发的；最后对比正常和异常两种情况下"
             "的内存快照做'对拍'，找出差异对象。",
             ("复现", "二分", "对拍", "内存快照")),
            ("'对拍'具体怎么做？两次内存快照怎么比出差异？",
             "用MAT/jmap生成heap dump，或native的massif/valgrind --leak-check=full生成报告，在同一"
             "状态下打两次快照，中间执行若干次可疑操作，然后做diff——如果某类对象数量持续净增长"
             "(而不是先增后降)，说明没有被正确释放，顺着它的GC Roots引用链往回找是谁一直持有它。",
             ("heap dump", "GC Roots", "diff", "净增长")),
            ("如果GC Roots链路显示是一个缓存类持有着这些对象，但这个缓存看起来'应该'会过期清理，"
             "为什么还是泄漏了？",
             "常见原因是缓存用了强引用而不是弱引用/软引用，或者过期清理线程本身挂了/被异常中断；也"
             "可能是key的equals/hashCode实现有问题导致同一个逻辑key每次都生成新的entry，永远命中不"
             "到应该被替换的旧entry，造成条目无限累积。",
             ("强引用", "弱引用", "hashCode", "无限累积")),
            ("这类问题在生产环境很难复现（可能要跑几天才涨爆内存），你怎么在不影响线上服务的前提下"
             "定位？",
             "上线前先用压测环境放大QPS或用更小的堆促使问题更快复现；线上必须现场定位时，用非侵入式"
             "的采样profiler(如async-profiler做alloc profiling)持续采样对象分配，配合按频率自动"
             "dump一次heap(heap dump会引起较长时间STW，需控制频率、选在低峰期)；也可以先加监控指标"
             "(比如某个Map/Cache的size曲线)辅助锁定可疑组件，减少直接dump大堆的次数。",
             ("压测", "async-profiler", "STW", "监控指标")),
        ),
        pitfall="很多人一开始就说'用valgrind/MAT分析'，但说不清楚复现-二分-对拍这个系统性排查流程，"
                "也答不出线上大堆dump代价大时该怎么办",
        real_world_link=""),

    DeepPoint(
        id="dp-be-os-04", cat=CAT,
        trigger="你提到线程池调大之后吞吐反而下降了，具体是怎么回事？",
        chain=(
            ("怎么测量一次上下文切换的开销？",
             "用vmstat看cs列(每秒上下文切换次数)，或者pidstat -w看具体进程的上下文切换次数；更精确"
             "的用perf stat记录context-switches事件，也可以写微基准测试(如lmbench的lat_ctx)用管道"
             "让两个进程/线程乒乓传递token强制触发切换并计时。",
             ("vmstat", "pidstat", "perf stat", "lmbench")),
            ("上下文切换本身耗时不算长(微秒级)，为什么线程数太多还是会明显拖累吞吐？",
             "除了切换本身的寄存器保存恢复开销，更大的隐性成本是cache locality被破坏——线程数超过"
             "CPU核数后频繁抢占调度，每次切换后L1/L2 cache里缓存的都是上一个线程的数据，新调度上来的"
             "线程要重新'热身'，大量cache miss会显著拖慢实际执行效率；就绪队列变长，调度器本身的调度"
             "决策开销也会增加。",
             ("cache locality", "cache miss", "就绪队列", "决策开销")),
            ("如果这是CPU密集型任务，把线程数从等于核数继续往上加，吞吐为什么会不增反降，而不只是"
             "'增长变缓'？",
             "多余的线程并不能获得额外的并行度，只会增加调度竞争；如果这些线程还共享锁，线程数越多"
             "锁竞争概率越高，多个线程在锁上自旋或者被唤醒又立刻阻塞，产生额外的无效上下文切换"
             "(thundering herd式唤醒)，CPU真正花在有效计算上的时间比例反而下降，这是典型的过度并发"
             "(over-subscription)现象。",
             ("过度并发", "锁竞争", "thundering herd", "有效计算")),
            ("生产环境里你怎么确定一个线程池的'合理线程数'，具体用什么方法验证？",
             "先分析任务的IO等待占比(比如用arthas trace或perf看线程在等待IO vs CPU计算的时间比例)，"
             "CPU密集型任务线程数大致设为核数或核数+1；但更可靠的做法是做压测——固定其他条件，横向"
             "扫描不同线程数下的吞吐和P99延迟，画出吞吐-线程数曲线，找到拐点作为实际配置依据，而不是"
             "死记公式。",
             ("IO等待占比", "压测", "拐点", "P99延迟")),
        ),
        pitfall="第2层很多人只会说'切换开销大'，答不出cache miss才是主因；第3层容易解释不清为什么"
                "CPU密集型场景会'降'而不只是'不增'",
        real_world_link=""),

    DeepPoint(
        id="dp-be-os-05", cat=CAT,
        trigger="你提到用mmap读文件比read/write快，这和缺页中断有什么关系？",
        chain=(
            ("缺页中断(page fault)是怎么发生的，major fault和minor fault有什么区别？",
             "进程访问一个虚拟地址，如果对应页表项显示该页不在物理内存中，CPU会触发缺页中断陷入内核；"
             "如果所需数据已经在内存(如在page cache中，只是没建立该进程的页表映射)，只需建立映射，这"
             "是minor fault，代价小；如果数据根本不在内存中，需要从磁盘读入，这是major fault，涉及"
             "磁盘IO，代价大得多。",
             ("major fault", "minor fault", "page cache", "磁盘IO")),
            ("mmap读文件本质也会触发缺页中断，为什么还被认为比read系统调用快？",
             "read需要把数据从内核page cache拷贝到用户buffer，每次读都有一次数据拷贝加系统调用开销；"
             "mmap把文件页直接映射进用户地址空间，后续访问是纯粹的内存访问、接近零拷贝，不需要再拷贝"
             "数据也不需要系统调用，对多次随机访问同一文件区域的场景优势明显；但对只顺序读一遍就丢弃"
             "的场景，mmap反而可能因为缺页中断开销比一次性read的批量拷贝更慢。",
             ("零拷贝", "缺页中断开销", "随机访问", "顺序读")),
            ("如果一个进程物理内存不够，缺页中断会怎么进一步影响性能，什么时候会发展成'抖动'"
             "(thrashing)？",
             "内存不足时操作系统要换出页面，如果被换出的页面很快又被访问，就要重新触发major fault"
             "换入，如果这种换入换出频繁发生形成'工作集'远大于可用物理内存的情况，系统会陷入"
             "thrashing——CPU大部分时间花在处理缺页中断和磁盘IO上，吞吐急剧下降；可以通过vmstat的"
             "si/so列或pgmajfault计数持续增长判断是否发生thrashing。",
             ("thrashing", "工作集", "si/so", "pgmajfault")),
            ("生产环境里怀疑某个服务因为缺页中断/thrashing导致性能抖动，你具体怎么排查确认？",
             "先用vmstat 1观察si/so是否非零且持续，用sar -B或/proc/vmstat看pgfault/pgmajfault增长"
             "速率；用perf stat -e page-faults,major-faults跟踪具体进程；确认major fault过多后用"
             "perf record -e major-faults结合火焰图定位代码位置；对策通常是增加物理内存、调小工作集、"
             "或禁用swap(很多低延迟服务直接vm.swappiness=0)。",
             ("vmstat", "perf record", "火焰图", "swappiness")),
        ),
        pitfall="很多人分不清minor/major fault，也说不出为什么mmap顺序读一次性场景反而可能更慢；"
                "第4层常答不出具体用什么命令/工具确认thrashing",
        real_world_link=""),

    DeepPoint(
        id="dp-be-os-06", cat=CAT,
        trigger="Linux默认的CFS调度器具体是怎么决定该运行哪个线程的？",
        chain=(
            ("CFS调度算法的核心思想是什么？",
             "CFS(Completely Fair Scheduler)用vruntime(虚拟运行时间)衡量每个任务'应得'的CPU时间，"
             "每次调度都选择vruntime最小的任务运行；所有可运行任务按vruntime组织在一棵红黑树里，最"
             "左边的节点就是vruntime最小、最应该被调度的任务，调度器直接取树的最左节点。",
             ("vruntime", "红黑树", "最左节点")),
            ("nice值是怎么影响vruntime增长速度，从而实现'不同优先级但仍然公平'的？",
             "nice值转换成一个权重(weight)，nice值越低权重越大；vruntime的增长速度和权重成反比——"
             "权重大的任务(高优先级)跑相同的实际时间，vruntime增长得慢，因此会更频繁地被调度到，从而"
             "获得更多CPU时间份额，但仍然是在同一个'公平'框架下按比例分配，而不是像传统优先级调度那样"
             "直接抢占。",
             ("nice", "权重", "vruntime的增长速度", "比例分配")),
            ("如果任务数量非常多(比如上千个线程都在等CPU)，CFS的公平性会遇到什么问题？",
             "CFS会给每个任务分配一个最小调度粒度保证的时间片，但当任务数远超CPU核数时，为了保证每"
             "个任务都能在一个调度周期(sched_latency_ns)内至少运行一次，实际分给每个任务的时间片会"
             "被压缩得非常小，导致上下文切换频率暴涨，调度开销占比上升，也会影响单个任务的响应延迟。",
             ("调度粒度", "调度周期", "调度开销", "响应延迟")),
            ("生产环境里发现一个多线程服务的延迟P99很高，怀疑是CFS调度延迟导致的，具体怎么验证和"
             "缓解？",
             "用perf sched record/perf sched latency分析每个线程实际等待被调度的延迟分布，或者用"
             "schedstat看任务在运行队列里等待的时间；确认后缓解手段包括给关键线程设置更高优先级或直接"
             "用SCHED_FIFO等实时调度类脱离CFS管理、减少同时竞争CPU的线程数、用cpuset/taskset绑核减少"
             "调度竞争。",
             ("perf sched", "schedstat", "SCHED_FIFO", "cpuset")),
        ),
        pitfall="很多人只知道'CFS用红黑树+vruntime'这个表层描述，说不清楚nice值具体怎么映射到"
                "vruntime增长速度，也答不出高负载下调度延迟怎么用工具验证",
        real_world_link=""),

    DeepPoint(
        id="dp-be-os-07", cat=CAT,
        trigger="你提到线上进程被系统杀掉了，怀疑是OOM killer，这个机制具体怎么工作的？",
        chain=(
            ("Linux的OOM killer是什么时候触发、怎么选择杀哪个进程的？",
             "当系统物理内存和swap都耗尽、内核无法满足新的内存分配请求时，触发OOM killer；它会给"
             "每个进程计算一个oom_score，综合考虑进程占用的内存比例(主要因素)、运行时长，以及用户通过"
             "/proc/<pid>/oom_score_adj设置的调整值，选择oom_score最高的进程杀掉。",
             ("oom_score", "占用的内存比例", "oom_score_adj")),
            ("为什么有时候杀掉的不是真正占用内存最多的'元凶'进程，反而是别的服务被误杀？",
             "oom_score是综合评分而不是单纯按内存占用排序，一个内存占用不算最大但没有调整"
             "oom_score_adj的进程可能比真正的内存大户评分更高；同一台机器混部多个服务时，容器/cgroup"
             "级别的内存核算和进程级别核算的口径不完全一致，导致'谁该死'的判断和直觉不符。",
             ("综合评分", "混部", "cgroup", "口径不完全一致")),
            ("如果是Kubernetes环境，容器被OOMKilled和宿主机层面的OOM killer有什么不同？",
             "容器场景下更常见的是cgroup级别的内存限制触发——当容器内存使用超过memory.limit"
             "(cgroup v1)或memory.max(cgroup v2)时，内核只在该cgroup范围内触发OOM killer，不影响"
             "宿主机上其他容器；Kubernetes自己还有基于QoS等级(Guaranteed/Burstable/BestEffort)的"
             "驱逐机制，在宿主机整体内存压力大时kubelet会先驱逐BestEffort的pod，这一层发生在内核OOM"
             "killer之前。",
             ("内存限制", "memory.limit", "QoS", "驱逐机制")),
            ("生产环境里怎么防止一个关键服务被OOM killer误杀，具体怎么配置？",
             "对关键进程设置更低的oom_score_adj(但要谨慎因为可能导致系统真正内存危险时无法回收内存)；"
             "容器场景下给关键pod设置合理的requests/limits让它落入Guaranteed QoS，降低被kubelet优先"
             "驱逐的概率；给cgroup内存限制加监控告警，从根源通过压测确定合理的资源配额；也可以开启"
             "earlyoom这类用户态守护进程在真正OOM之前提前做更可控的清理。",
             ("oom_score_adj", "Guaranteed QoS", "资源配额", "earlyoom")),
        ),
        pitfall="很多人只答得出'内存不够杀进程'，说不清oom_score的综合评分逻辑，也分不清宿主机OOM"
                "killer和K8s cgroup限制/QoS驱逐是两层不同机制",
        real_world_link=""),

    DeepPoint(
        id="dp-be-os-08", cat=CAT,
        trigger="你提到用io_uring替换epoll能减少syscall开销，具体开销体现在哪？",
        chain=(
            ("一次系统调用(syscall)的开销具体来自哪里？",
             "用户态发起syscall会触发CPU从用户态(ring 3)切换到内核态(ring 0)，需要保存用户态寄存器"
             "上下文、切换到内核栈、执行处理函数再切换回用户态；这个过程本身有固定的CPU周期开销，还会"
             "破坏用户态代码正在使用的cache/TLB局部性。",
             ("ring 3", "ring 0", "内核栈", "TLB局部性")),
            ("既然syscall有固定开销，为什么Linux后来搞了vDSO这种机制？",
             "vDSO把一些不需要真正陷入内核的调用(比如gettimeofday、clock_gettime)改造成用户态可以"
             "直接读取的函数，完全避免了ring切换的开销，因为这类调用只是读取一个内核周期性更新的共享"
             "只读页面，vDSO本质是用共享内存替代了syscall陷入。",
             ("vDSO", "共享只读页面", "ring切换")),
            ("epoll已经是IO多路复用的高效方案了，为什么高性能场景还要进一步换成io_uring？",
             "epoll每次实际的读写操作仍然是独立的、各自陷入内核一次的系统调用，高并发场景下这些独立"
             "syscall次数依然很多；io_uring用两个环形缓冲区(SQ/CQ)在用户态和内核态之间共享，应用可以"
             "批量提交多个IO请求而不需要为每个请求单独陷入内核，完成结果也由用户态轮询CQ读取，甚至可"
             "以做到SQPOLL模式完全不需要用户态发起syscall，本质是靠批量提交减少了syscall次数。",
             ("批量提交", "环形缓冲区", "SQPOLL", "减少了syscall次数")),
            ("怎么在生产环境实际测量一个服务的syscall开销占比，判断值不值得迁移到io_uring？",
             "用strace -c线下诊断统计各类syscall调用次数和耗时占比(strace开销大不适合线上长期跑)；"
             "线上更适合用perf trace/bpftrace挂载syscall入口/出口的tracepoint统计频率和延迟分布，也"
             "可以用perf top看是否有大量时间花在entry_SYSCALL_64等内核符号上；确认syscall确实是瓶颈"
             "后再评估io_uring改造，否则盲目迁移收益有限还增加复杂度。",
             ("strace -c", "bpftrace", "perf top", "entry_SYSCALL_64")),
        ),
        pitfall="很多人知道io_uring'更快'但说不清具体机制上是怎么减少syscall次数的(SQ/CQ批量)，也"
                "想不到用bpftrace/perf量化syscall占比再决定是否值得迁移",
        real_world_link=""),

    DeepPoint(
        id="dp-be-os-09", cat=CAT,
        trigger="你提到虚拟内存，具体page替换算法在真实系统里是怎么实现的？",
        chain=(
            ("理论上LRU是最优的页面置换策略之一，Linux实际是怎么近似实现的？",
             "Linux用的是近似LRU——Clock算法(二次机会算法)的变种，维护active list和inactive list两"
             "个链表，每个页表项有一个accessed位，页面被访问时硬件自动置位，内核定期扫描把accessed位"
             "为1的页面挪到active(或清除标志给它'第二次机会')，真正需要回收内存时优先从inactive list"
             "尾部换出。",
             ("active list", "inactive list", "accessed位", "二次机会")),
            ("这种近似LRU和严格LRU相比，在什么场景下会做出'错误'的置换决策？",
             "如果发生一次大范围的顺序扫描(比如遍历一个很大的文件)，会把大量原本'热'的页面短时间内"
             "标记为'刚访问过'冲到active list前面，而这些页面之后可能再也不会被访问，等于把真正的热点"
             "数据挤出了active list——这是经典的'顺序扫描污染缓存'问题，Linux为此提供了"
             "madvise(MADV_SEQUENTIAL)提示内核尽快回收。",
             ("顺序扫描污染", "MADV_SEQUENTIAL", "热点数据")),
            ("如果物理内存不足，页面置换和swap是什么关系，什么条件下会触发大范围thrashing而不是"
             "正常的按需换页？",
             "当系统可用内存持续低于水位线，kswapd后台线程被唤醒异步回收内存；如果内存压力增长速度"
             "超过kswapd回收速度，会触发直接回收(direct reclaim)——分配内存的进程自己被迫陷入内核同步"
             "等待回收完成，延迟急剧上升；如果所有进程的工作集总和持续大于物理内存，换出去的页面很快"
             "又被换回来形成恶性循环，就是系统级thrashing。",
             ("kswapd", "水位线", "direct reclaim", "恶性循环")),
            ("生产环境怀疑发生了thrashing，你具体怎么用工具确认，以及怎么应急处理？",
             "用vmstat 1观察si/so是否持续非零、free是否逼近0；用sar -B看pgscank/pgscand，pgscand远"
             "大于pgscank说明大量走代价更高的直接回收；用top看wa(iowait)比例是否异常高；应急手段包括"
             "临时kill非核心进程释放内存、禁用swap，长期方案是给容器设置合理的memory.high做提前节流。",
             ("si/so", "pgscand", "iowait", "memory.high")),
        ),
        pitfall="很多人只答得出Clock算法的名字，说不清顺序扫描会污染缓存这个反直觉的细节；第3层容易"
                "把'换页'和'thrashing'混为一谈，说不清direct reclaim和kswapd异步回收的区别",
        real_world_link=""),

    DeepPoint(
        id="dp-be-os-10", cat=CAT,
        trigger="你提到高并发场景用自旋锁代替互斥量，这两者具体怎么选？",
        chain=(
            ("自旋锁和互斥量(mutex)的核心区别是什么？",
             "自旋锁获取不到锁时，线程原地忙等，不释放CPU也不发生上下文切换；互斥量获取不到锁时会让"
             "线程睡眠、放弃CPU，等锁被释放后由内核唤醒。自旋锁适合临界区极短的场景，互斥量适合临界区"
             "较长、持锁时间不确定的场景。",
             ("忙等", "上下文切换", "临界区", "内核唤醒")),
            ("Linux用户态的mutex(比如pthread_mutex)实际实现是纯睡眠等待吗？",
             "不是，pthread_mutex基于futex(fast userspace mutex)实现，是'乐观自旋+悲观睡眠'的混合"
             "策略：无竞争时纯粹是用户态的CAS原子操作完成加锁解锁，完全不进入内核；只有发生竞争时才"
             "通过futex系统调用陷入内核让线程睡眠等待。",
             ("futex", "CAS", "无竞争时", "陷入内核")),
            ("如果在单核CPU上使用自旋锁会发生什么问题？为什么内核态的自旋锁要求关闭抢占？",
             "单核CPU上，如果线程A持有自旋锁后被抢占，线程B开始自旋等待，但只有一个核，线程A永远没"
             "机会被重新调度来释放锁，会一直空转；因此内核自旋锁在获取锁的同时会关闭当前CPU的抢占"
             "(preempt_disable)，保证持锁线程不会被抢占，这也是为什么内核自旋锁的临界区不能睡眠。",
             ("单核", "抢占", "preempt_disable", "临界区不能睡眠")),
            ("生产环境里怎么判断一个高并发热点该用自旋锁思路优化还是应该用无锁数据结构，具体怎么"
             "验证收益？",
             "先用perf lock或火焰图看到底是在mutex_lock/futex_wait上花了大量时间还是CPU真的在自旋；"
             "如果竞争频繁但临界区确实很短(比如只是更新计数器)，先尝试无锁化(原子操作/CAS循环重试)；"
             "实际验证收益要靠压测对比不同实现在真实并发度下的吞吐和延迟分布，因为自旋在核数不够、"
             "竞争线程数远超核数时反而会比mutex更差。",
             ("perf lock", "火焰图", "无锁化", "压测对比")),
        ),
        pitfall="很多人分不清用户态pthread_mutex其实是futex实现的乐观自旋+悲观睡眠混合体，第3层常"
                "答不出单核场景下自旋锁为什么必须关闭抢占",
        real_world_link=""),

    DeepPoint(
        id="dp-be-os-11", cat=CAT,
        trigger="你提到用epoll的边缘触发(ET)模式做性能优化，具体是怎么回事？",
        chain=(
            ("epoll的水平触发(LT)和边缘触发(ET)有什么区别？",
             "水平触发(LT)模式下，只要fd的缓冲区里还有未读数据/还可写，每次epoll_wait都会返回该事件；"
             "边缘触发(ET)模式下只有当fd的状态发生'变化'时才通知一次，如果这次没有把数据读完，之后"
             "不会再收到重复通知，除非又有新数据到达触发新的状态变化。",
             ("水平触发", "边缘触发", "状态变化", "重复通知")),
            ("ET模式下如果没有把数据一次性读完会有什么后果？为什么ET通常要求配合非阻塞IO？",
             "ET只在状态变化时通知一次，如果只读了一部分数据就不再处理，剩下的数据会一直留在内核"
             "缓冲区里且不会再收到通知，看起来像连接卡住了；因此ET模式要求每次通知后必须循环read直到"
             "返回EAGAIN为止，这必须搭配非阻塞IO，否则最后一次read在没有更多数据时会一直阻塞。",
             ("EAGAIN", "非阻塞IO", "循环read", "内核缓冲区")),
            ("既然ET更容易踩坑，为什么高性能服务器(如Nginx)还是倾向于用ET？",
             "LT模式下只要缓冲区还有剩余数据，每次epoll_wait都会重复上报该fd，浪费内核和应用层来回"
             "交互的开销；ET模式每个状态变化只通知一次，配合读到EAGAIN的模式能显著减少重复的事件通知"
             "次数，尤其在高并发大量fd都有数据时减少epoll_wait被无谓唤醒的次数，代价是编程复杂度上升。",
             ("重复上报", "唤醒的次数", "编程复杂度")),
            ("生产环境里一个用ET模式的服务出现了连接'假死'(数据发过去但服务端没反应)，可能是什么"
             "原因，怎么排查？",
             "最常见原因是应用层在某次事件通知后没有把socket缓冲区读空就退出了读循环，导致内核不会"
             "再通知，数据卡在内核缓冲区；排查时用ss -tnp看该连接的Recv-Q是否有堆积字节数没被应用读"
             "走，也可以用strace attach观察对该fd的read调用序列，确认是否真的读到了EAGAIN；修复方式"
             "是审查读循环逻辑确保严格读到EAGAIN为止，或者退回LT模式降低出错概率。",
             ("Recv-Q", "ss -tnp", "strace", "读到EAGAIN为止")),
        ),
        pitfall="很多人只知道'ET更高效'的结论，说不清楚为什么必须配合非阻塞IO+循环读到EAGAIN，也"
                "答不出线上'假死'时具体用什么命令验证是内核缓冲区堆积",
        real_world_link=""),

    DeepPoint(
        id="dp-be-os-12", cat=CAT,
        trigger="你提到多核场景下要注意false sharing，这个具体是怎么发生、怎么排查的？",
        chain=(
            ("false sharing是什么？",
             "CPU cache以cache line为单位(通常64字节)在核间同步，如果两个逻辑上无关的变量恰好落在"
             "同一条cache line里，被不同CPU核上的线程分别频繁读写，硬件的cache一致性协议(如MESI)也"
             "会把整条cache line标记为失效并在核间来回同步，造成性能显著下降，这种'伪共享'现象叫"
             "false sharing。",
             ("cache line", "MESI", "伪共享", "核间同步")),
            ("MESI协议具体是怎么导致这个开销的？",
             "当核A要写自己cache里的一条cache line时，如果这条line在核B的cache里也存在(哪怕核B读写"
             "的是line里的另一个变量)，核A必须先通知核B把该line标记为Invalid，核B下次访问自己那部分"
             "数据时该cache line已失效，需要重新同步——这个'失效通知+重新同步'涉及跨核通信，比单纯"
             "的L1 cache命中慢一到两个数量级。",
             ("Invalid", "跨核通信", "数量级", "重新同步")),
            ("如果只是把两个变量物理上隔开(padding)就能避免false sharing，为什么不是所有场景都直接"
             "把所有变量都padding到独立cache line？",
             "padding会显著增加内存占用，在变量数量巨大的场景(比如大数组每个元素都要独立统计)padding"
             "会造成内存膨胀好几十倍，更大的内存占用还会降低cache的整体命中率，所以只应该对'确定会被"
             "不同核高频并发写'的热点变量做针对性padding，而不是无差别处理所有共享数据。",
             ("padding", "内存膨胀", "命中率", "热点变量")),
            ("生产环境里怎么确认一个多线程程序的性能问题是false sharing导致的，而不是普通的锁竞争？",
             "用perf c2c(cache-to-cache，专门检测false sharing的perf子命令)记录采样，它会报告哪些"
             "内存地址在不同核之间发生了频繁的cache line迁移和HITM(hit modified)事件，直接定位到具体"
             "变量/字段；也可以对比优化前后的cache-misses/LLC-load-misses；确认后用alignas(64)等"
             "手段给高频独立写的变量插入padding，修复后应看到cache-misses显著下降。",
             ("perf c2c", "HITM", "cache-misses", "alignas")),
        ),
        pitfall="很多人知道'padding能解决false sharing'但说不清MESI协议层面到底是什么在跨核同步，也"
                "说不出用perf c2c这种专门工具去定位，容易和普通锁竞争混淆",
        real_world_link=""),
]


def _self_test() -> None:
    assert 11 <= len(BANK) <= 13, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [dp.id for dp in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("dp-be-os-") for i in ids), "id 前缀不一致"
    for dp in BANK:
        assert len(dp.chain) >= 3, f"{dp.id} chain层数不足"
        assert dp.trigger and dp.pitfall, f"{dp.id} 缺少trigger/pitfall"
        for q, ref, keys in dp.chain:
            assert q and ref and keys, f"{dp.id} 存在字段缺失的层"
        # 自洽性：把参考答案本身当作作答，每层理应满分命中自己列的采分关键词
        answers = [ref for _, ref, _ in dp.chain]
        scores = grade_chain(dp, answers)
        assert all(s == 1.0 for s in scores), f"{dp.id} 自洽性检查失败: {scores}"
    assert drill(BANK, cat=CAT, n=3) == BANK[:3]
    print(f"[PASS] dp_os_internals: {len(BANK)} 个 DeepPoint 通过自检")


if __name__ == "__main__":
    _self_test()
