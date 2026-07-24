# 10 · 手把手实战:从零搭一个迷你集群调度模拟器

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 63 个"知识点",不计入"62 个知识点"的统计——和 [04 类](04-small-model-graduation.md)、[09 类](09-infra-graduation.md)是同一挂,但风格不一样:04/09 号文件里,你是**旁观者**,跟着一条五部曲/端到端系统设计的叙事线看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立搭出一个真实能跑的迷你集群调度模拟器。这个"教程体"格式最早在 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 试点验证,这是它在本系列的第一次落地。

## 为什么是"集群调度模拟器"

不是要发明新知识点,是把 [08 类知识点 1](08-training-orchestration.md) 讲过的 Slurm 风格 FIFO+Backfill 调度,从"静态的单次分配快照"往前推一步,变成一个真的有时间在流动、job 会陆续到达和结束的动态模拟器。08 类知识点 1 的 `fifo_with_backfill` 函数本身有一处诚实的自我限定——它的追问 1 直接问过:"如果要让这份代码支持真实 Slurm 那种'持续轮询'行为,需要在现有函数基础上加什么?"期望答案提到两件事:①一个持续往前走的外层循环;②一套真正建模"预留"(reservation)的机制。这一篇就是把这两件事真正动手写成代码。

和 08 类知识点 1/2 有一个刻意的不同:那边的 `try_assign`/`gang_assign` 都要操心"具体分配到哪几个节点"(多节点拓扑放置),这一篇先把这层复杂度拿掉,把整个集群简化成一个"总共还有多少张空闲 GPU"的单一数字池。这不是不知道拓扑放置的存在(08 类知识点 2 已经用一整节讲拓扑放置的坑),而是刻意的取舍:先把"调度策略"(先来后到,还是见缝插针)这一个维度看清楚,不需要同时处理"具体分配到哪个节点"这第二个维度。真实 Slurm 两个维度都要处理,这篇教程只处理第一个。

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 严格按到达顺序调度,现场验证"队头一个大 job 排不下,会卡住后面明明能立刻跑的小 job" | [08 类知识点 1](08-training-orchestration.md) Slurm FIFO 排队规则 |
| 阶段 2 | 给队首 job 算一个"预约时间",允许不影响这个预约的小 job 插队 | 08 类知识点 1 的 backfill 思想 + 追问链里提到但源码没写出的 reservation 逻辑 |
| 阶段 3 | 用同一份更大的合成 job 队列,量化比较 FIFO 和 Backfill 的整体利用率/平均等待时间 | 阶段 1+2 的直接应用 |
| 阶段 4 | 封装成一个可复用的 `MiniClusterScheduler` 类,一次端到端跑完两种策略并打印对比报告 | 阶段 1-3 全部组装 |

每个阶段的代码都能独立运行(本文件用仓库统一的 `_verify_md.py` 校验,校验方式是把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行——块与块之间**不共享任何变量**,所以后面阶段用到前面阶段的代码时会重新贴一遍,不是偷懒复制,是这套校验机制要求的)。

---

## 阶段 1:严格 FIFO——现场验证"队头阻塞"

这个调度器用"离散时间片(tick)"的方式往前推进——每一步(tick)固定做四件事:①把这一刻应该结束的 job 释放掉,腾出它们占用的 GPU;②把这一刻应该到达的新 job 塞进等待队列;③尝试调度;④时间前进一步。"调度器每一步检查队首 job 需要的 GPU 数"这句话字面对应的就是第③步——严格 FIFO 版本里,第③步只做一件事:看等待队列最前面的 job 需要的 GPU 数是否 `<=` 当前空闲 GPU 数,够就分配、然后继续检查新的队首,不够就整个队列原地不动——**即使排在后面的某个 job 本来自己就能塞进当前的空闲量,严格 FIFO 也不会去看它一眼**,因为规则里根本没有"跳过队首往后找"这一步。

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Job:
    job_id: int
    n_gpus: int
    duration_s: float
    submitted_at: float
    started_at: float | None = None

    def waited_s(self):
        if self.started_at is None:
            return None
        return self.started_at - self.submitted_at


def run_fifo(jobs, total_gpus, max_tick=200):
    """Strict FIFO: every tick, only the queue head is ever considered.
    If it does not fit, the whole queue waits -- even a job behind it
    that would fit right now never gets a chance."""
    queue = sorted(jobs, key=lambda j: (j.submitted_at, j.job_id))
    waiting, running = [], []          # running: list of (job, finish_tick)
    free_gpus, arrived_idx, finished, t = total_gpus, 0, [], 0
    while len(finished) < len(jobs) and t <= max_tick:
        still_running = []
        for job, finish_t in running:
            if finish_t <= t:
                free_gpus += job.n_gpus
                finished.append(job)
            else:
                still_running.append((job, finish_t))
        running = still_running

        while arrived_idx < len(queue) and queue[arrived_idx].submitted_at <= t:
            waiting.append(queue[arrived_idx])
            arrived_idx += 1

        while waiting and waiting[0].n_gpus <= free_gpus:
            job = waiting.pop(0)
            job.started_at = t
            free_gpus -= job.n_gpus
            running.append((job, t + job.duration_s))

        t += 1
    return finished


# 8-GPU 玩具集群: alice 先到且占用大头, bob 紧接着到达但需求超过剩余量,
# carol 最后到达但只要 1 张卡 -- alice 留下的 2 张空闲卡本来绰绰有余。
jobs = [
    Job(1, 6, 10, submitted_at=0),   # alice: 6 gpus, 10 ticks, 于 t=0 到达
    Job(2, 4, 3,  submitted_at=1),   # bob:   4 gpus, 3 ticks,  于 t=1 到达
    Job(3, 1, 1,  submitted_at=2),   # carol: 1 gpu,  1 tick,   于 t=2 到达
]
result = run_fifo(jobs, total_gpus=8)
by_id = {j.job_id: j for j in result}

assert by_id[1].started_at == 0                  # alice 立刻开始
assert by_id[2].started_at == 10                 # bob 被卡到 alice 结束才开始
assert by_id[3].started_at == 10                 # carol 也被卡到同一时刻 -- 明明她一直有位置
assert by_id[3].waited_s() == 8                   # 白白多等 8 个 tick

for j in sorted(result, key=lambda j: j.job_id):
    print(f"job {j.job_id}: n_gpus={j.n_gpus} duration={j.duration_s} submitted={j.submitted_at} started={j.started_at} waited={j.waited_s()}")
print("stage1 fifo head-of-line blocking reproduced ok")
```

**实测(`.venv` 真跑):** alice 在 t=0 立刻拿到 6 张卡开跑,剩 2 张空闲。bob 在 t=1 到达时需要 4 张卡,只有 2 张空闲,排不下,成为队首、卡住。carol 在 t=2 到达,只需要 1 张卡,2 张空闲卡随便一张都够用——但因为她排在被卡住的 bob 后面,严格 FIFO 规则下调度器根本不会去看她,只会死盯着队首的 bob。直到 t=10 alice 训练结束、释放 6 张卡,bob 才终于开始(此时空闲量变成 8),bob 让位之后 carol 紧接着在同一个 tick 被检查到、立刻开始。carol 从 t=2 到 t=10,整整 8 个 tick 里,眼前一直摆着 2 张空闲卡,却因为排队顺序一张都用不上——这就是"队头阻塞"(head-of-line blocking),不是这份代码写错了,是严格 FIFO 这个策略本身的定义决定的。

## 阶段 2:预约时间——Backfill 怎么保证"不耽误队首"

Backfill 要解决阶段 1 暴露的问题,但不能简单地说"谁能塞进去就先让谁跑"——如果这样做,原本排在最前面的大 job 可能被后面源源不断到达的小 job 无限期插队,永远轮不到自己(这正是 08 类知识点 1 追问链提到的隐患)。真正的 backfill 需要先回答一个问题:"如果什么都不做,队首这个 job 最快什么时候能凑够资源开始跑?"——这个时间点叫**预约时间**(reservation),只根据"当前正在跑的 job 会在什么时候陆续结束"来算,不去管等待队列里还有谁(因为排队的 job 还没拿到任何 GPU,不会影响这个计算)。算出预约时间之后,后面排队的小 job 只要同时满足两个条件就可以插队:①现在的空闲量够它跑;②它自己的运行时长足够短,能在预约时间到达之前就跑完、把 GPU 吐出来。第二个条件是关键:只要插队的 job 保证在预约时间之前就已经让出资源,队首 job 到了预约时间那一刻看到的空闲量,和"没有任何人插队"时完全一样——队首完全不知道、也不受影响。

先把"预约时间怎么算"单独拎出来,脱离完整调度器验证一遍:

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Job:
    job_id: int
    n_gpus: int
    duration_s: float
    submitted_at: float
    started_at: float | None = None


def compute_reservation(head, running, free_gpus_now, now):
    """Earliest tick at which >= head.n_gpus GPUs will be free, assuming
    only the jobs already running finish on schedule (the wait queue is
    not considered -- queued jobs hold no GPUs yet)."""
    if head.n_gpus <= free_gpus_now:
        return now
    avail = free_gpus_now
    for job, finish_t in sorted(running, key=lambda x: x[1]):
        avail += job.n_gpus
        if avail >= head.n_gpus:
            return finish_t
    return float("inf")


# 队首需要 4 张卡,当前空闲 0 张；3 个正在跑的 job 会分别在 t=5/9/9 释放 1/2/3 张卡
head = Job(99, 4, 0, submitted_at=0)
running = [
    (Job(1, 1, 5, 0), 5),
    (Job(2, 2, 9, 0), 9),
    (Job(3, 3, 9, 0), 9),
]
# t=5 只回收 1 张(总共 1 张),不够 4 张；要等到 t=9 两个 job 一起结束(2+3=5 张)才够
assert compute_reservation(head, running, free_gpus_now=0, now=0) == 9

# 如果队首只需要 1 张卡,t=5 那次回收已经够了
head_small = Job(98, 1, 0, submitted_at=0)
assert compute_reservation(head_small, running, free_gpus_now=0, now=0) == 5

# 如果当前空闲量已经够,预约时间就是"现在"
assert compute_reservation(head, running, free_gpus_now=4, now=3) == 3

print("stage2 reservation math verified in isolation")
```

**实测(`.venv` 真跑):** 三组断言全部一次通过,对应三种真实场景:①需求较大、要等两个 job 一起结束才够;②需求较小、第一个结束的 job 就已经够;③当前空闲量已经满足需求,预约时间退化成"现在"(这一分支在完整调度器里其实不会被触发,因为队首排不下才会走到 backfill 逻辑——写在这里是为了让这个函数本身的边界条件独立可验证)。

把 `compute_reservation` 接进完整的调度循环,重新跑一遍阶段 1 的 alice/bob/carol 场景:

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Job:
    job_id: int
    n_gpus: int
    duration_s: float
    submitted_at: float
    started_at: float | None = None

    def waited_s(self):
        if self.started_at is None:
            return None
        return self.started_at - self.submitted_at


def compute_reservation(head, running, free_gpus_now, now):
    if head.n_gpus <= free_gpus_now:
        return now
    avail = free_gpus_now
    for job, finish_t in sorted(running, key=lambda x: x[1]):
        avail += job.n_gpus
        if avail >= head.n_gpus:
            return finish_t
    return float("inf")


def run_backfill(jobs, total_gpus, max_tick=200):
    queue = sorted(jobs, key=lambda j: (j.submitted_at, j.job_id))
    waiting, running = [], []
    free_gpus, arrived_idx, finished, t = total_gpus, 0, [], 0
    while len(finished) < len(jobs) and t <= max_tick:
        still_running = []
        for job, finish_t in running:
            if finish_t <= t:
                free_gpus += job.n_gpus
                finished.append(job)
            else:
                still_running.append((job, finish_t))
        running = still_running

        while arrived_idx < len(queue) and queue[arrived_idx].submitted_at <= t:
            waiting.append(queue[arrived_idx])
            arrived_idx += 1

        while waiting and waiting[0].n_gpus <= free_gpus:
            job = waiting.pop(0)
            job.started_at = t
            free_gpus -= job.n_gpus
            running.append((job, t + job.duration_s))

        if waiting:                      # 队首被卡住了，尝试 backfill
            head = waiting[0]
            reservation = compute_reservation(head, running, free_gpus, t)
            i = 1
            while i < len(waiting):
                cand = waiting[i]
                if cand.n_gpus <= free_gpus and t + cand.duration_s <= reservation:
                    waiting.pop(i)
                    cand.started_at = t
                    free_gpus -= cand.n_gpus
                    running.append((cand, t + cand.duration_s))
                else:
                    i += 1

        t += 1
    return finished


jobs = [
    Job(1, 6, 10, submitted_at=0),
    Job(2, 4, 3,  submitted_at=1),
    Job(3, 1, 1,  submitted_at=2),
]
result = run_backfill(jobs, total_gpus=8)
by_id = {j.job_id: j for j in result}

assert by_id[1].started_at == 0
assert by_id[2].started_at == 10          # bob: 和阶段 1 的纯 FIFO 结果完全一样 -- 没有被耽误
assert by_id[3].started_at == 2           # carol: 一到达就立刻开始
assert by_id[3].waited_s() == 0            # 8 -> 0，这就是 backfill 真正生效的地方

for j in sorted(result, key=lambda j: j.job_id):
    print(f"job {j.job_id}: n_gpus={j.n_gpus} duration={j.duration_s} submitted={j.submitted_at} started={j.started_at} waited={j.waited_s()}")
print("stage2 backfill ok: carol jumps the queue, bob start time is untouched")
```

**实测(`.venv` 真跑):** carol 在 t=2 一到达就检测到"当前空闲 2 张卡够用,且她 1 个 tick 的运行时长远远短于 bob 的预约时间(t=10)",立刻插队开跑,t=3 就结束,等待时间从阶段 1 的 8 变成 0。更关键的是 bob 的开始时间——两次跑出来**分毫不差,都是 t=10**——carol 插队没有偷走 bob 的任何东西,因为她在 bob 的预约时间到来之前就已经主动让出了 GPU。这正是 backfill 设计要保证的核心约束:允许后面的小 job 见缝插针,但绝不能让队首多等一刻。

## 阶段 3:大规模合成队列——量化比较整体效率

阶段 1/2 的 3-job 场景足够说明"backfill 好在哪",但看不出量级上的差异。这一步用同一份更大的合成 job 队列(12 个 job,GPU 需求/运行时长/到达时间各不相同),同时跑 FIFO 和 Backfill 两种策略,比较整体资源利用率和平均等待时间。

第一次尝试用了 16 张卡的宽松集群配置,结果 FIFO 和 Backfill 跑出来的每个 job 开始时间完全一样——集群太宽松,几乎没有 job 需要排队,backfill 根本没有用武之地。这是一个有用的提醒:backfill 是"资源紧张时才生效"的机制,想现场验证它的效果,测试场景必须真的紧张,不能随手选一个宽松的规模就断言"两者没区别"。下面把集群缩小到 8 张卡,真实制造出排队和资源竞争。

```python
from __future__ import annotations
from dataclasses import dataclass
import random


@dataclass
class Job:
    job_id: int
    n_gpus: int
    duration_s: float
    submitted_at: float
    started_at: float | None = None

    def waited_s(self):
        if self.started_at is None:
            return None
        return self.started_at - self.submitted_at

    def finished_at(self):
        if self.started_at is None:
            return None
        return self.started_at + self.duration_s


def make_synthetic_jobs(n=12, seed=7):
    rng = random.Random(seed)
    jobs, t = [], 0.0
    for i in range(1, n + 1):
        t += rng.choice([0, 0, 0, 1, 1])
        n_gpus = rng.choice([1, 1, 2, 2, 4, 6])
        duration = rng.choice([2, 3, 4, 5, 8, 10])
        jobs.append(Job(i, n_gpus, duration, t))
    return jobs


def run_fifo(jobs, total_gpus, max_tick=500):
    queue = sorted(jobs, key=lambda j: (j.submitted_at, j.job_id))
    waiting, running, free_gpus, arrived_idx, finished, t = [], [], total_gpus, 0, [], 0
    while len(finished) < len(jobs) and t <= max_tick:
        still_running = []
        for job, finish_t in running:
            if finish_t <= t:
                free_gpus += job.n_gpus
                finished.append(job)
            else:
                still_running.append((job, finish_t))
        running = still_running
        while arrived_idx < len(queue) and queue[arrived_idx].submitted_at <= t:
            waiting.append(queue[arrived_idx])
            arrived_idx += 1
        while waiting and waiting[0].n_gpus <= free_gpus:
            job = waiting.pop(0)
            job.started_at = t
            free_gpus -= job.n_gpus
            running.append((job, t + job.duration_s))
        t += 1
    return finished


def compute_reservation(head, running, free_gpus_now, now):
    if head.n_gpus <= free_gpus_now:
        return now
    avail = free_gpus_now
    for job, finish_t in sorted(running, key=lambda x: x[1]):
        avail += job.n_gpus
        if avail >= head.n_gpus:
            return finish_t
    return float("inf")


def run_backfill(jobs, total_gpus, max_tick=500):
    queue = sorted(jobs, key=lambda j: (j.submitted_at, j.job_id))
    waiting, running, free_gpus, arrived_idx, finished, t = [], [], total_gpus, 0, [], 0
    while len(finished) < len(jobs) and t <= max_tick:
        still_running = []
        for job, finish_t in running:
            if finish_t <= t:
                free_gpus += job.n_gpus
                finished.append(job)
            else:
                still_running.append((job, finish_t))
        running = still_running
        while arrived_idx < len(queue) and queue[arrived_idx].submitted_at <= t:
            waiting.append(queue[arrived_idx])
            arrived_idx += 1
        while waiting and waiting[0].n_gpus <= free_gpus:
            job = waiting.pop(0)
            job.started_at = t
            free_gpus -= job.n_gpus
            running.append((job, t + job.duration_s))
        if waiting:
            head = waiting[0]
            reservation = compute_reservation(head, running, free_gpus, t)
            i = 1
            while i < len(waiting):
                cand = waiting[i]
                if cand.n_gpus <= free_gpus and t + cand.duration_s <= reservation:
                    waiting.pop(i)
                    cand.started_at = t
                    free_gpus -= cand.n_gpus
                    running.append((cand, t + cand.duration_s))
                else:
                    i += 1
        t += 1
    return finished


TOTAL_GPUS = 8
fifo_jobs = run_fifo(make_synthetic_jobs(12, seed=7), TOTAL_GPUS)
backfill_jobs = run_backfill(make_synthetic_jobs(12, seed=7), TOTAL_GPUS)
assert len(fifo_jobs) == 12 and len(backfill_jobs) == 12   # 12 个 job 全部跑完，没有卡在 max_tick

fifo_by_id = {j.job_id: j for j in fifo_jobs}
bf_by_id = {j.job_id: j for j in backfill_jobs}

fifo_avg_wait = sum(j.waited_s() for j in fifo_jobs) / len(fifo_jobs)
bf_avg_wait = sum(j.waited_s() for j in backfill_jobs) / len(backfill_jobs)
fifo_makespan = max(j.finished_at() for j in fifo_jobs)
bf_makespan = max(j.finished_at() for j in backfill_jobs)
work = sum(j.n_gpus * j.duration_s for j in fifo_jobs)      # 两种策略总工作量完全一样
fifo_util = work / (TOTAL_GPUS * fifo_makespan)
bf_util = work / (TOTAL_GPUS * bf_makespan)

assert fifo_makespan == 29 and bf_makespan == 21
assert bf_avg_wait < fifo_avg_wait
assert bf_util > fifo_util

# 核心约束: backfill 不会让任何一个 job 比 fifo 更晚开始
violations = [jid for jid in fifo_by_id if bf_by_id[jid].started_at > fifo_by_id[jid].started_at]
assert violations == []

# job 7 是这份合成队列里真实卡过队的"队首"之一 -- 验证它的开始时间两边完全一致
assert fifo_by_id[7].started_at == bf_by_id[7].started_at == 6

# job 11 是被直接 backfill 插队的例子: 等待时间从 16 骤降到 0
assert fifo_by_id[11].waited_s() == 16 and bf_by_id[11].waited_s() == 0

# job 10 需求只有 1 张卡，看起来应该很容易被 backfill，但它的 duration=8 太长，
# 在每一次有机会插队的时刻都无法保证在预约时间之前结束，所以自始至终没被 backfill，
# 两种策略下的开始时间完全相同 -- 不是所有"看起来很小"的 job 都能插队
assert fifo_by_id[10].started_at == bf_by_id[10].started_at == 11

print(f"FIFO:     avg_wait={fifo_avg_wait:.3f} utilization={fifo_util:.4f} makespan={fifo_makespan}")
print(f"Backfill: avg_wait={bf_avg_wait:.3f} utilization={bf_util:.4f} makespan={bf_makespan}")
print("stage3 aggregate comparison ok, zero violations of the no-delay guarantee")
```

**实测(`.venv` 真跑):** 12 个 job、8 张卡的合成场景下,FIFO 平均等待 **5.833** tick、集群利用率 **0.6767**、总耗时(makespan)**29** tick;Backfill 平均等待降到 **3.667** tick、利用率提升到 **0.9345**、总耗时缩短到 **21** tick——makespan 缩短约 1.38 倍。逐个 job 核对"backfill 是否曾让任何人比 FIFO 更晚开始"这条核心约束,**12 个 job 里零违反**。这份数据里还有两个真实跑出来才发现的细节:①真正被判定为走"backfill"路径插队的其实只有 job 11 一个,但 job 12 的开始时间也从 t=21 提前到 t=11——它并不是被直接插队的,而是沾了 job 11 提前让出资源的光,资源腾出来的时间点整体往前挪了,job 12 排到队首的时刻自然也跟着提前,这是 backfill 的收益会向后"传导"给排在更后面的 job 的一个真实例子,不是巧合;②job 10 只需要 1 张卡,直觉上应该很容易被插队,但它 8 个 tick 的运行时长在每一次检查时都无法保证在当时的预约时间之前结束,所以自始至终一次都没被 backfill,两种策略下开始时间完全一样(都是 t=11)——这说明能不能被 backfill 不是只看"需要的 GPU 数够不够小",运行时长相对预约窗口是否足够短同样是硬条件,两个条件缺一不可。

## 阶段 4:封装成 `MiniClusterScheduler`,端到端跑一次完整对比

把前三阶段的逻辑收进一个类,一次调用同时产出 FIFO 和 Backfill 两份完整报告。

```python
from __future__ import annotations
from dataclasses import dataclass
import random


@dataclass
class Job:
    job_id: int
    n_gpus: int
    duration_s: float
    submitted_at: float
    started_at: float | None = None

    def waited_s(self):
        if self.started_at is None:
            return None
        return self.started_at - self.submitted_at

    def finished_at(self):
        if self.started_at is None:
            return None
        return self.started_at + self.duration_s


class MiniClusterScheduler:
    """Stage 1-3 的逻辑打包成一个类: total_gpus 固定，backfill 开关决定
    走严格 FIFO 还是 FIFO+Backfill。"""

    def __init__(self, total_gpus, backfill=True):
        self.total_gpus = total_gpus
        self.backfill = backfill

    @staticmethod
    def _compute_reservation(head, running, free_gpus_now, now):
        if head.n_gpus <= free_gpus_now:
            return now
        avail = free_gpus_now
        for job, finish_t in sorted(running, key=lambda x: x[1]):
            avail += job.n_gpus
            if avail >= head.n_gpus:
                return finish_t
        return float("inf")

    def run(self, jobs, max_tick=1000):
        queue = sorted(jobs, key=lambda j: (j.submitted_at, j.job_id))
        waiting, running = [], []
        free_gpus, arrived_idx, finished, t = self.total_gpus, 0, [], 0
        while len(finished) < len(jobs) and t <= max_tick:
            still_running = []
            for job, finish_t in running:
                if finish_t <= t:
                    free_gpus += job.n_gpus
                    finished.append(job)
                else:
                    still_running.append((job, finish_t))
            running = still_running
            while arrived_idx < len(queue) and queue[arrived_idx].submitted_at <= t:
                waiting.append(queue[arrived_idx])
                arrived_idx += 1
            while waiting and waiting[0].n_gpus <= free_gpus:
                job = waiting.pop(0)
                job.started_at = t
                free_gpus -= job.n_gpus
                running.append((job, t + job.duration_s))
            if self.backfill and waiting:
                head = waiting[0]
                reservation = self._compute_reservation(head, running, free_gpus, t)
                i = 1
                while i < len(waiting):
                    cand = waiting[i]
                    if cand.n_gpus <= free_gpus and t + cand.duration_s <= reservation:
                        waiting.pop(i)
                        cand.started_at = t
                        free_gpus -= cand.n_gpus
                        running.append((cand, t + cand.duration_s))
                    else:
                        i += 1
            t += 1
        finished.sort(key=lambda j: j.job_id)
        return finished

    @staticmethod
    def report(jobs, total_gpus):
        work = sum(j.n_gpus * j.duration_s for j in jobs)
        makespan = max(j.finished_at() for j in jobs)
        return {
            "avg_wait": sum(j.waited_s() for j in jobs) / len(jobs),
            "utilization": work / (total_gpus * makespan),
            "makespan": makespan,
        }


def make_synthetic_jobs(n=12, seed=7):
    rng = random.Random(seed)
    jobs, t = [], 0.0
    for i in range(1, n + 1):
        t += rng.choice([0, 0, 0, 1, 1])
        n_gpus = rng.choice([1, 1, 2, 2, 4, 6])
        duration = rng.choice([2, 3, 4, 5, 8, 10])
        jobs.append(Job(i, n_gpus, duration, t))
    return jobs


TOTAL_GPUS = 8
fifo_jobs = MiniClusterScheduler(TOTAL_GPUS, backfill=False).run(make_synthetic_jobs(12, seed=7))
bf_jobs = MiniClusterScheduler(TOTAL_GPUS, backfill=True).run(make_synthetic_jobs(12, seed=7))

fifo_report = MiniClusterScheduler.report(fifo_jobs, TOTAL_GPUS)
bf_report = MiniClusterScheduler.report(bf_jobs, TOTAL_GPUS)

# 用类重新跑一遍，数字必须和阶段 3 手写函数版本的结果完全一致 -- 重构没有改变行为
assert abs(fifo_report["avg_wait"] - 5.833333333333333) < 1e-9
assert abs(bf_report["avg_wait"] - 3.6666666666666665) < 1e-9
assert fifo_report["makespan"] == 29 and bf_report["makespan"] == 21

speedup = fifo_report["makespan"] / bf_report["makespan"]
assert 1.3 < speedup < 1.5

fifo_by_id = {j.job_id: j for j in fifo_jobs}
bf_by_id = {j.job_id: j for j in bf_jobs}
assert all(bf_by_id[jid].started_at <= fifo_by_id[jid].started_at for jid in fifo_by_id)

print(f"FIFO     report: {fifo_report}")
print(f"Backfill report: {bf_report}")
print(f"makespan speedup: {speedup:.3f}x")
print("stage4 end-to-end MiniClusterScheduler ok, matches stage3 numbers exactly")
```

**实测(`.venv` 真跑):** 类封装后重新跑出来的数字和阶段 3 手写函数版本逐位对齐(`avg_wait`/`utilization`/`makespan` 全部一致),证明"从散装函数收进一个类"这次重构没有偷偷改变任何行为。makespan 加速比 **1.381x**,`MiniClusterScheduler` 现在是一个真实能用的小工具:给它一份 job 列表和总 GPU 数,它能同时告诉你"如果用严格 FIFO 会怎样"和"如果用 backfill 会怎样",并且保证 backfill 这条路径永远不会让任何 job 比 FIFO 更晚开始。

## 可以怎么继续扩展(只指方向,不在本文实现)

- **多节点拓扑感知放置**:这篇教程把集群简化成一个"空闲 GPU 总数"的单一数字池,真实调度还要决定"具体分配到哪几个节点"——[08 类知识点 1/2](08-training-orchestration.md) 的 `try_assign`/`gang_assign` 已经在做这件事(贪心按节点空闲量降序装箱),把这两套逻辑合并起来,才是更完整的真实调度器。
- **Gang scheduling 语义**:分布式训练的多个 rank 必须同时启动、同时失败,不能只有一部分 rank 抢到资源——[08 类知识点 2](08-training-orchestration.md) 的 `gang_assign` 已经实现了这个"要么全部、要么一个都不"的原子分配,和本文的 backfill 逻辑结合需要额外处理"gang job 的 backfill 安全性判断要同时保证所有 rank 都能在预约时间前完成"这一层复杂度。
- **duration 是估计值,不是真实值**:这份模拟器里 duration 是"上帝视角"精确已知的,预约时间的计算完全准确。真实 HPC 系统里,duration 是用户提交时自己填的 walltime 估计——填少了任务跑到一半会被系统强制杀掉,填多了会让 backfill 的安全窗口过度保守、白白空出本可以利用的资源,这是真实 backfill 调度器要处理、而这篇教程完全没有建模的一大类复杂性。
- **故障容忍的耦合**:如果一个正在跑的 job 中途失败,不仅要重新排队,还会打乱所有已经算好的 reservation——这和 [08 类知识点 3](08-training-orchestration.md) 的 Young's Formula/MTBF 模型是同一个系统的两个侧面,值得合起来看。
- **优先级与 fairshare**:目前的调度完全按到达顺序,没有考虑用户优先级或历史用量(`learning/training-orchestration/README.md` 提到的 `fairshare = (used_recently / fair_share_limit)^-1`),真实生产集群几乎都会在 FIFO 之上叠加这一层。

这几个方向都不实现,是为了让这篇教程聚焦在"时间维度上的调度策略"这一件事上——真要继续做下去,每一个方向单独展开都够写一整篇。

## 这篇教程展示的方法论

和 dsa-deep-dive/21 横向拼接三个不同文件的知识点不同,这一篇的教程体内容来自另一个方向:把 [08 类知识点 1](08-training-orchestration.md) 追问链里已经提出、但源代码没有真正实现的两处延伸——"持续轮询的时间循环"和"保护队首预约窗口的 backfill 判断"——真正动手写出来、跑起来、验证清楚。这提示了教程体的另一种可能来源:不一定是横向拼接多个知识点,也可以是把某一个知识点后面追问链里已经问出来、但没有写代码回答的问题,选一个动手做到底。两种来源殊途同归:读者都得到一个从空文件开始、真实跑得起来、把已经学过的东西用一种新方式重新组装了一遍的小工具。

---

*创建:2026-07-24*
