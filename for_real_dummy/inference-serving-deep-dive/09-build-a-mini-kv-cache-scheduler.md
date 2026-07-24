# 09 · 手把手实战:从零搭一个迷你 KV-cache 调度器

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 9 个"知识点",不计入本系列约 71 个知识点的统计——和 [07 号文件](07-serving-graduation-topics.md)/[08 号文件](08-serving-graduation-capstone.md)不是同一种形态:07 号文件是标准的七步模板知识点,08 号文件是"毕业答辩"叙事体,读者在这两篇里都是**旁观者**,跟着 lecture 讲解或答辩场景把一条推理链条看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立搭出一个能跑的迷你调度器。这是 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 试点过的"教程体"格式,第一次推广到本系列。

## 为什么是"KV-cache 调度器"

不是要发明新知识点,是把 [01 号文件](01-inference-engine-core.md)已经讲过的几个机制串成一个真实能跑、能现场量出数字差异的小工具:PagedAttention 的"固定容量物理块池"抽象(知识点 3)、Continuous Batching 的"迭代级准入"调度循环(知识点 5),外加调度策略里"pending 队列先挑谁"的准入顺序(知识点 8)。**全篇是纯 Python 模拟,不接真实模型推理**——用"一个请求需要占用多少个 cache slot""每步还要跑多少步才会生成完"这类抽象数值代表真实开销,这和 01 号文件 `Engine.kv_budget`/`_can_admit()` 的抽象粒度是同一个精神:提前用一个数字代表这个请求一辈子最多要用多少资源,不逐 token 动态模拟。

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 有限个 cache slot 怎么被安全地借出去、还回来——不多分配、不重复分配,也不能悄悄泄漏 | [01 号文件](01-inference-engine-core.md)知识点 3 PagedAttention 的物理块池抽象(`PagedKvPool.alloc_block`/`free_block`) |
| 阶段 2 | 先实现最笨的调度:一批请求必须全部跑完,才能开始收下一批 | 01 号文件知识点 5 对"静态批处理为什么浪费"的描述 |
| 阶段 3 | 换一条准入规则,请求随时能在有空位时插进来,现场量出两种策略的平均等待时间差多少 | 01 号文件知识点 5 `Engine.step()`/`_can_admit()` 的"迭代级调度" |
| 阶段 4 | 发现一个"天生不可能被满足"的请求会把排在它后面的所有请求一起拖死,补上边界检查,组装成最终的调度器类 | 01 号文件知识点 6 Chunked Prefill 讨论的 head-of-line blocking 同类问题 + 知识点 8 调度策略"pending 队列先挑谁"的准入顺序 |

本文件的代码块用和 [08 号文件](08-serving-graduation-capstone.md)相同的独立校验方式(细节见 00-roadmap.md 里关于本文件的说明):每个代码块单独起一个新的 Python 子进程执行,块与块之间**不共享任何变量**,所以后面阶段用到前面阶段的 `SlotPool`/`SimRequest` 时会重新贴一遍,不是偷懒复制。

---

## 阶段 1:slot 池——有限资源怎么被安全地借出去、还回来

真实的 PagedAttention([01 号文件](01-inference-engine-core.md)知识点 3 `PagedKvPool`)把 KV cache 切成大小统一的物理块,`alloc_block()`/`free_block()` 管理一个"空闲块 id 队列"。这篇教程把这个思路简化到"调度器"这一层:不关心 block 内部怎么存 K/V 张量,只关心"某个请求现在占用了哪几个编号的 slot、还剩多少个空闲",用一个整数 `capacity` 代表这次 KV cache 池子总共能装多少个"槽位"。每个请求提交时声明自己需要多少个 slot(`needed_slots`)——这和 01 号文件知识点 5 `Engine._can_admit()` 用 `need = len(prompt_ids) + max_new_tokens` 提前估算"这个请求一辈子最多要用多少预算"是同一种简化:用一个固定数字代表资源需求上限,不逐 token 动态调整。

先实现最基本的借用/归还,并且要经得起边界情况的考验:池子满了必须明确拒绝,不能分配到一半就出错,也不能"蹭"走别人的 slot。

```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class SlotPool:
    """固定容量的 KV-cache slot 池，slot 编号 0..capacity-1。"""
    capacity: int
    free_ids: deque = field(default_factory=deque)
    owner: dict = field(default_factory=dict)  # slot_id -> 当前占用它的请求 id

    def __post_init__(self):
        self.free_ids = deque(range(self.capacity))

    def n_free(self):
        return len(self.free_ids)

    def try_allocate(self, request_id, n_slots):
        """一次性尝试拿到 n_slots 个空闲 slot。成功返回 slot id 列表；
        暂时不够就返回 None——这是一个正常、常见的情况（调用方应该让请求排队等待），
        不是异常。"""
        if n_slots > self.n_free():
            return None
        allocated = [self.free_ids.popleft() for _ in range(n_slots)]
        for sid in allocated:
            self.owner[sid] = request_id
        return allocated

    def release(self, slot_ids):
        for sid in slot_ids:
            del self.owner[sid]
            self.free_ids.append(sid)


pool = SlotPool(capacity=8)
assert pool.n_free() == 8
assert pool.owner == {}

a = pool.try_allocate("reqA", 3)
assert a is not None and len(a) == 3
assert pool.n_free() == 5
assert all(pool.owner[s] == "reqA" for s in a)

b = pool.try_allocate("reqB", 5)
assert b is not None and len(b) == 5
assert pool.n_free() == 0

# 池子已经完全占满：第三个请求必须被明确告知"现在没地方"
c = pool.try_allocate("reqC", 1)
assert c is None
assert pool.n_free() == 0  # 被拒绝的分配不能"部分消耗"掉任何 slot——要么全给，要么一个都不给

pool.release(a)
assert pool.n_free() == 3
assert all(s not in pool.owner for s in a)

c = pool.try_allocate("reqC", 3)
assert c is not None
assert pool.n_free() == 0

pool.release(b)
pool.release(c)
assert pool.n_free() == 8
assert pool.owner == {}

print("stage1 slotpool basic alloc/release ok")
```

**实测(`.venv` 真跑):** `capacity=8` 的池子里,先后借出 3 个和 5 个 slot 后恰好占满;第三次请求 1 个 slot 被正确拒绝(`try_allocate` 返回 `None`),且这次被拒绝的请求没有拿到、也没有消耗任何 slot(`n_free()` 前后不变)——分配失败必须是"全有或全无",不能出现"分配到一半"的中间状态。释放 3 个 slot 后,新提交的 3-slot 请求能立刻拿到刚释放出来的那批 slot。

**一个真实存在的坑:** 如果调度器在请求结束时忘了调用 `release()`,后果不是立刻报错,而是 slot 会被"悄悄吃掉"——表面上每个请求都"成功完成"了,池子却在不知不觉中被耗尽。

```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class SlotPool:
    capacity: int
    free_ids: deque = field(default_factory=deque)
    owner: dict = field(default_factory=dict)

    def __post_init__(self):
        self.free_ids = deque(range(self.capacity))

    def n_free(self):
        return len(self.free_ids)

    def try_allocate(self, request_id, n_slots):
        if n_slots > self.n_free():
            return None
        allocated = [self.free_ids.popleft() for _ in range(n_slots)]
        for sid in allocated:
            self.owner[sid] = request_id
        return allocated

    def release(self, slot_ids):
        for sid in slot_ids:
            del self.owner[sid]
            self.free_ids.append(sid)


# 复现一个真实存在的坑：调度器"忘记"在请求结束时调用 release()
pool = SlotPool(capacity=8)

def buggy_finish_request(pool, request_id, n_slots):
    """bug：分配、"跑完"、标记完成——但忘了调用 release()。"""
    slots = pool.try_allocate(request_id, n_slots)
    if slots is None:
        return False
    # ...这里本该有真实生成 token 的过程，生成完之后本该 release(slots)...
    return True  # 忘记了：pool.release(slots)

admitted = 0
for i in range(4):
    if buggy_finish_request(pool, f"req{i}", n_slots=2):
        admitted += 1

# 四个请求全部"成功完成"，但池子早就被吃光了
assert admitted == 4
assert pool.n_free() == 0
print("after 4 finished requests, free slots:", pool.n_free())

# 一个全新请求会被拒绝，即便此刻没有任何请求真的还在运行——
# 这是一类真实存在的症状："看起来像是满了"，真正原因是泄漏，不是负载高
victim = pool.try_allocate("req_new", 1)
assert victim is None
print("leak reproduced: new request rejected even though 0 requests are still actually running")
```

**实测:** 4 个请求各借 2 个 slot、全部"完成"(但代码里漏调 `release()`)后,`n_free()` 精确降到 0;这时哪怕系统里其实没有任何请求真的还在运行,一个全新请求依然会被拒绝——这是生产系统里一类真实存在、容易被误诊的症状:表面看起来像"负载太高、cache 不够用",真正病因却是资源泄漏,两者需要用不同手段排查(前者该扩容或限流,后者该找泄漏点),不能一概而论。

---

## 阶段 2:最笨的调度循环——静态 batching,一批必须整批出清才能收下一批

有了 slot 池,接下来实现"调度循环":每个抽象的"时间步"里,决定谁能拿到 slot 开始跑、谁的生成又往前推进了一步。最直观的做法是 01 号文件知识点 5 描述的静态批处理:一批请求一起开始,必须等这一批**全部**结束才能开始下一批——即便这一批里有请求提前完成、腾出了 slot,那些空出来的 slot 在这批彻底出清之前也只能闲置。

用一个可以手算验证的小例子:3 个请求都在第 0 步到达,`capacity=4`。req0 要跑 6 步、req1 只要跑 2 步,两者各占 2 个 slot,恰好把 4 个 slot 占满;req2 也要 2 个 slot,只能先排队。

```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class SlotPool:
    capacity: int
    free_ids: deque = field(default_factory=deque)
    owner: dict = field(default_factory=dict)

    def __post_init__(self):
        self.free_ids = deque(range(self.capacity))

    def n_free(self):
        return len(self.free_ids)

    def try_allocate(self, request_id, n_slots):
        if n_slots > self.n_free():
            return None
        allocated = [self.free_ids.popleft() for _ in range(n_slots)]
        for sid in allocated:
            self.owner[sid] = request_id
        return allocated

    def release(self, slot_ids):
        for sid in slot_ids:
            del self.owner[sid]
            self.free_ids.append(sid)


@dataclass
class SimRequest:
    rid: int
    arrival_step: int
    needed_slots: int
    remaining_steps: int          # 抽象的"还要跑多少步才会生成完"，代表真实的 decode 时长
    admitted_step: int = None
    finish_step: int = None


def simulate_static(specs, capacity, max_steps=10_000):
    """specs: (rid, arrival_step, needed_slots, duration) 元组列表。
    静态 batching：只要还有任何请求在跑，就不允许新请求加入——
    必须等当前这一批全部跑完，才能开始下一批。"""
    reqs = sorted(
        (SimRequest(rid=rid, arrival_step=arr, needed_slots=needed, remaining_steps=dur)
         for rid, arr, needed, dur in specs),
        key=lambda r: r.arrival_step,
    )
    pool = SlotPool(capacity)
    pending, running, finished = deque(), [], []
    next_idx, step = 0, 0
    free_log = []
    while (next_idx < len(reqs) or pending or running) and step < max_steps:
        while next_idx < len(reqs) and reqs[next_idx].arrival_step <= step:
            pending.append(reqs[next_idx])
            next_idx += 1

        if not running:  # 和 continuous batching 唯一的区别：准入条件多了"当前必须没人在跑"
            while pending and pending[0].needed_slots <= pool.n_free():
                r = pending.popleft()
                slots = pool.try_allocate(r.rid, r.needed_slots)
                r.admitted_step = step
                running.append([r, slots])

        free_log.append(pool.n_free())
        still_running = []
        for r, slots in running:
            r.remaining_steps -= 1
            if r.remaining_steps <= 0:
                pool.release(slots)
                r.finish_step = step + 1
                finished.append(r)
            else:
                still_running.append([r, slots])
        running = still_running
        step += 1

    assert step < max_steps, "simulation did not terminate -- likely a bug, not real workload"
    return finished, free_log


capacity = 4
# req0 要跑很久（6 步），req1 很快跑完（2 步），req2 一直在排队
specs = [(0, 0, 2, 6), (1, 0, 2, 2), (2, 0, 2, 2)]
finished, free_log = simulate_static(specs, capacity)
by_rid = {r.rid: r for r in finished}

print("admitted_step:", {rid: r.admitted_step for rid, r in by_rid.items()})
print("finish_step:", {rid: r.finish_step for rid, r in by_rid.items()})
print("free_log:", free_log)

assert by_rid[0].admitted_step == 0
assert by_rid[1].admitted_step == 0
assert by_rid[2].admitted_step == 6   # req2 必须等 req0 和 req1 都跑完才能进场
assert by_rid[0].finish_step == 6
assert by_rid[1].finish_step == 2
assert by_rid[2].finish_step == 8

# 第 2~5 步：req1 已经跑完、腾出了 2 个 slot，但 req0 还在跑——
# 静态 batching 下这 2 个空闲 slot 只能干等着，因为 req2 不能中途插队
idle_window = free_log[2:6]
assert idle_window == [2, 2, 2, 2]
print("static batching wasted", sum(idle_window), "slot-steps of idle capacity while req2 waited")
```

**实测(`.venv` 真跑):** req0/req1 在第 0 步一起被准入;req1 在第 2 步就跑完并释放了它的 2 个 slot,但 req2 在静态批处理下必须等 req0 也在第 6 步跑完才能进场——第 2~5 步之间,池子里明明有 2 个空闲 slot(`free_log` 精确记录为 `[2, 2, 2, 2]`),却因为"这一批还没走完"而干等着,一共浪费了 8 个 slot-步的闲置容量。这正是 01 号文件知识点 5 说的"已完成请求腾出的槽位在这期间只能空转"在这个简化模型上的真实复现,不是转述。

---

## 阶段 3:只改一条准入规则——continuous batching,现场量出等待时间差多少

阶段 2 的问题出在准入条件"只要还有人在跑,就不能收新请求"太保守。[01 号文件](01-inference-engine-core.md)知识点 5 `Engine.step()` 的做法是每个 iteration 都重新检查:只要队首的等待请求现在能塞进空闲 slot,就立刻放行,不需要等其他还在跑的请求也全部结束——这就是"迭代级调度"。把阶段 2 的准入条件从"当前必须没人在跑"换成"看队首请求能不能塞进去",就是 continuous batching。

```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class SlotPool:
    capacity: int
    free_ids: deque = field(default_factory=deque)
    owner: dict = field(default_factory=dict)

    def __post_init__(self):
        self.free_ids = deque(range(self.capacity))

    def n_free(self):
        return len(self.free_ids)

    def try_allocate(self, request_id, n_slots):
        if n_slots > self.n_free():
            return None
        allocated = [self.free_ids.popleft() for _ in range(n_slots)]
        for sid in allocated:
            self.owner[sid] = request_id
        return allocated

    def release(self, slot_ids):
        for sid in slot_ids:
            del self.owner[sid]
            self.free_ids.append(sid)


@dataclass
class SimRequest:
    rid: int
    arrival_step: int
    needed_slots: int
    remaining_steps: int
    admitted_step: int = None
    finish_step: int = None


def simulate(specs, capacity, policy, max_steps=10_000):
    """policy="static"：只有 running 为空才准入；
    policy="continuous"：只要队首请求现在能塞进去就准入——
    和 01 号文件知识点 5 Engine._can_admit()/step() 的"迭代级调度"是同一条规则。"""
    reqs = sorted(
        (SimRequest(rid=rid, arrival_step=arr, needed_slots=needed, remaining_steps=dur)
         for rid, arr, needed, dur in specs),
        key=lambda r: r.arrival_step,
    )
    pool = SlotPool(capacity)
    pending, running, finished = deque(), [], []
    next_idx, step = 0, 0
    while (next_idx < len(reqs) or pending or running) and step < max_steps:
        while next_idx < len(reqs) and reqs[next_idx].arrival_step <= step:
            pending.append(reqs[next_idx])
            next_idx += 1

        can_admit_now = (policy == "continuous") or (policy == "static" and not running)
        if can_admit_now:
            while pending and pending[0].needed_slots <= pool.n_free():
                r = pending.popleft()
                slots = pool.try_allocate(r.rid, r.needed_slots)
                r.admitted_step = step
                running.append([r, slots])

        still_running = []
        for r, slots in running:
            r.remaining_steps -= 1
            if r.remaining_steps <= 0:
                pool.release(slots)
                r.finish_step = step + 1
                finished.append(r)
            else:
                still_running.append([r, slots])
        running = still_running
        step += 1

    assert step < max_steps, "simulation did not terminate -- likely a bug, not real workload"
    return finished


# --- Part A：和静态 batching 完全同一组请求，只是换成 continuous 准入规则 ---
capacity = 4
specs = [(0, 0, 2, 6), (1, 0, 2, 2), (2, 0, 2, 2)]
finished = simulate(specs, capacity, "continuous")
by_rid = {r.rid: r for r in finished}
assert by_rid[2].admitted_step == 2   # req1 一在第 2 步腾出 2 个 slot，req2 立刻能插进来
assert by_rid[2].finish_step == 4     # 比静态 batching 下的第 8 步提前完成了 4 步
print("continuous: req2 admitted at step", by_rid[2].admitted_step, "(was step 6 under static)")

# --- Part B：更大规模、随机生成的工作负载，直接量出两种策略的平均等待时间差异 ---
import random
import statistics

def make_workload(n, seed):
    rng = random.Random(seed)
    specs = []
    t = 0
    for i in range(n):
        t += rng.randint(0, 1)
        needed = rng.randint(1, 3)
        duration = rng.randint(15, 25) if rng.random() < 0.25 else rng.randint(2, 5)
        specs.append((i, t, needed, duration))
    return specs

capacity = 6
workload = make_workload(n=18, seed=7)
assert all(needed <= capacity for _, _, needed, _ in workload)  # 这一批里没有"天生不可能"的请求

static_result = simulate(workload, capacity, "static")
cont_result = simulate(workload, capacity, "continuous")
assert len(static_result) == len(cont_result) == len(workload)

static_waits = [r.admitted_step - r.arrival_step for r in static_result]
cont_waits = [r.admitted_step - r.arrival_step for r in cont_result]
assert all(w >= 0 for w in static_waits) and all(w >= 0 for w in cont_waits)

static_mean = statistics.mean(static_waits)
cont_mean = statistics.mean(cont_waits)
print("static avg wait:", static_mean, "continuous avg wait:", cont_mean)
assert cont_mean < static_mean
print("continuous batching has a real, measured lower average wait time")
```

**实测(`.venv` 真跑):** 完全相同的 3 请求场景下,req2 在 continuous batching 下第 2 步就被放行(静态批处理下要等到第 6 步),提前 4 步完成。换成 18 个请求的随机工作负载(固定种子,`capacity=6`)重新测量,static 策略平均等待 `40.11` 步,continuous 策略平均等待 `24.89` 步——continuous 只有 static 的约 62%,和阶段 2 手算例子展示的方向完全一致,只是这次是在更大规模、非精心构造的随机数据上量出来的。

---

## 阶段 4:一个真实的边界 bug——单个请求超过总容量会拖死整条队列

阶段 3 的准入规则有一个没处理的边界情况:如果某个请求一开始就声明"我需要比整个池子总容量还多的 slot",它永远不可能被放行。继续用阶段 3 那条"只看队首"的规则去处理这种请求,会发生什么?先用一种**有步数上限、绝对不会真的死等**的方式安全探测一下——`max_steps` 给"最多愿意等多久"设了一个硬上限,这不是真的死循环,是一次有界的探测:

```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class SlotPool:
    capacity: int
    free_ids: deque = field(default_factory=deque)
    owner: dict = field(default_factory=dict)

    def __post_init__(self):
        self.free_ids = deque(range(self.capacity))

    def n_free(self):
        return len(self.free_ids)

    def try_allocate(self, request_id, n_slots):
        if n_slots > self.n_free():
            return None
        allocated = [self.free_ids.popleft() for _ in range(n_slots)]
        for sid in allocated:
            self.owner[sid] = request_id
        return allocated

    def release(self, slot_ids):
        for sid in slot_ids:
            del self.owner[sid]
            self.free_ids.append(sid)


@dataclass
class SimRequest:
    rid: int
    arrival_step: int
    needed_slots: int
    remaining_steps: int
    admitted_step: int = None
    finish_step: int = None


def naive_continuous_scheduler(specs, capacity, max_steps):
    """和阶段 3 的 continuous batching 用同一条准入规则，唯独少了一件事：
    从不检查一个请求要的 slot 数是不是本来就超过了整个池子的容量。
    每个提交的 spec 都直接进 pending，不管它要多少 slot。"""
    reqs = sorted(
        (SimRequest(rid=rid, arrival_step=arr, needed_slots=needed, remaining_steps=dur)
         for rid, arr, needed, dur in specs),
        key=lambda r: r.arrival_step,
    )
    pool = SlotPool(capacity)
    pending, running, finished = deque(), [], []
    next_idx, step = 0, 0
    while (next_idx < len(reqs) or pending or running) and step < max_steps:
        while next_idx < len(reqs) and reqs[next_idx].arrival_step <= step:
            pending.append(reqs[next_idx])
            next_idx += 1

        # continuous batching 的准入规则：永远只看队首
        while pending and pending[0].needed_slots <= pool.n_free():
            r = pending.popleft()
            slots = pool.try_allocate(r.rid, r.needed_slots)
            r.admitted_step = step
            running.append([r, slots])

        still_running = []
        for r, slots in running:
            r.remaining_steps -= 1
            if r.remaining_steps <= 0:
                pool.release(slots)
                r.finish_step = step + 1
                finished.append(r)
            else:
                still_running.append([r, slots])
        running = still_running
        step += 1

    # 到达步数预算却还没跑完——这是一次有界、安全的探测，不是真的死循环：
    # max_steps 给"最多愿意等多久"设了一个硬上限
    hung = step >= max_steps
    return finished, hung, step


# 请求 1 要 99 个 slot，但池子总共只有 4 个——它永远不可能被放行
capacity = 4
poison_specs = [(0, 0, 2, 3), (1, 0, 99, 3), (2, 0, 1, 3)]
finished, hung, ran_steps = naive_continuous_scheduler(poison_specs, capacity, max_steps=300)

print("finished:", sorted(r.rid for r in finished), "hung:", hung, "steps used:", ran_steps)
assert hung is True
# 因为准入规则永远只看队首，请求 1 会永久挡住排在它后面的请求 2；
# 请求 0（先到、立刻能塞进去）是唯一跑完的
assert {r.rid for r in finished} == {0}
print("hang reproduced (bounded): an oversized request wedges every request queued behind it")
```

**实测(`.venv` 真跑):** `capacity=4`,请求 1 声明需要 99 个 slot。因为准入规则永远只检查 pending 队首,请求 1 卡在队首之后,连带把排在它后面、原本只需要 1 个 slot、本该秒过的请求 2 也一起拖死——300 步的探测预算耗尽,`hung=True`,最终只有先到、立刻能满足的请求 0 跑完。这是 head-of-line blocking 的一种,和 01 号文件知识点 6 Chunked Prefill 要解决的"一个大任务卡住所有小任务"属于同一类问题的另一种触发方式(那边是"内容合理但很大的请求"拖慢别人,这里是"内容本身就不可能被满足的请求"直接卡死别人)。

**修复思路:** 不能指望调度循环在运行时才发现"这个请求不可能被满足"——要在请求**提交**的那一刻就检查它要的 slot 数是否超过池子总容量,超过就直接拒绝,连 pending 队列都不让它进,从根上掐断"卡在队首拖死后面所有人"的可能性。把这个检查和阶段 1-3 的 `SlotPool`/准入循环组装成最终的调度器类:

```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class SlotPool:
    capacity: int
    free_ids: deque = field(default_factory=deque)
    owner: dict = field(default_factory=dict)

    def __post_init__(self):
        self.free_ids = deque(range(self.capacity))

    def n_free(self):
        return len(self.free_ids)

    def try_allocate(self, request_id, n_slots):
        if n_slots > self.n_free():
            return None
        allocated = [self.free_ids.popleft() for _ in range(n_slots)]
        for sid in allocated:
            self.owner[sid] = request_id
        return allocated

    def release(self, slot_ids):
        for sid in slot_ids:
            del self.owner[sid]
            self.free_ids.append(sid)


@dataclass
class SimRequest:
    rid: int
    arrival_step: int
    needed_slots: int
    remaining_steps: int
    admitted_step: int = None
    finish_step: int = None


class MiniKvCacheScheduler:
    """把阶段 1-3 拼起来：固定容量的 SlotPool + continuous batching 准入
    （只看 pending 队首），再加上刚才复现的那个 hang 所需要的修复——在
    submit() 这一步就拒绝任何一辈子都不可能被满足的请求，而不是让它混进
    pending 之后才被 step() 循环发现卡死。"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.pool = SlotPool(capacity)
        self.pending = deque()
        self.running = []
        self.finished = []
        self.rejected = []
        self.step_i = 0

    def submit(self, spec):
        rid, arrival, needed, dur = spec
        if needed > self.capacity:
            # 放进 pending 只会永远卡在队首、拖死后面所有请求——现在就拒绝
            self.rejected.append(spec)
            return False
        self.pending.append(SimRequest(rid=rid, arrival_step=arrival, needed_slots=needed, remaining_steps=dur))
        return True

    def step(self):
        while self.pending and self.pending[0].needed_slots <= self.pool.n_free():
            r = self.pending.popleft()
            slots = self.pool.try_allocate(r.rid, r.needed_slots)
            r.admitted_step = self.step_i
            self.running.append([r, slots])

        still_running = []
        for r, slots in self.running:
            r.remaining_steps -= 1
            if r.remaining_steps <= 0:
                self.pool.release(slots)
                r.finish_step = self.step_i + 1
                self.finished.append(r)
            else:
                still_running.append([r, slots])
        self.running = still_running
        self.step_i += 1

    def run(self, specs, max_steps=10_000):
        sorted_specs = sorted(specs, key=lambda s: s[1])
        idx = 0
        while (idx < len(sorted_specs) or self.pending or self.running) and self.step_i < max_steps:
            while idx < len(sorted_specs) and sorted_specs[idx][1] <= self.step_i:
                self.submit(sorted_specs[idx])
                idx += 1
            self.step()
        assert self.step_i < max_steps, "scheduler did not drain -- likely a real bug, not just a slow workload"
        return self.finished, self.rejected


# --- 重跑上面一模一样的 poison workload，这次修复已经就位 ---
capacity = 4
poison_specs = [(0, 0, 2, 3), (1, 0, 99, 3), (2, 0, 1, 3)]
sched = MiniKvCacheScheduler(capacity)
finished, rejected = sched.run(poison_specs)

assert [spec[0] for spec in rejected] == [1]      # 恰好只有请求 1（要 99 > 容量 4）被拒绝
assert {r.rid for r in finished} == {0, 2}        # 0 和 2 不再被卡住，正常跑完
print("rejected:", rejected)
print("finished rids:", sorted(r.rid for r in finished))
print("fix confirmed: the poison request no longer blocks its neighbors")

# --- 任务里容易和"拒绝"混淆的另一种情况：总需求超过容量是正常情况，应该排队，
# 不应该被拒绝——只有单个请求本身就超过整个池子容量时才拒绝 ---
capacity2 = 4
overload_specs = [(i, 0, 2, 5) for i in range(10)]   # 10 个请求各要 2 个 slot，总共 20，池子只有 4
sched2 = MiniKvCacheScheduler(capacity2)
finished2, rejected2 = sched2.run(overload_specs)
assert rejected2 == []                 # 没有任何一个请求单独超过 capacity2，没有人被拒绝
assert len(finished2) == 10            # 全部 10 个最终都跑完了
waits2 = [r.admitted_step - r.arrival_step for r in finished2]
assert max(waits2) > 0                 # 确实有人真的排过队
print("overload (feasible per-request): all 10 finished, 0 rejected, max wait", max(waits2))

# --- 一个普普通通的工作负载，端到端跑一遍，确认组装后的类没有破坏正常情况 ---
capacity3 = 6
ordinary = [(0, 0, 2, 3), (1, 0, 2, 3), (2, 1, 2, 3), (3, 2, 1, 2), (4, 2, 3, 4)]
sched3 = MiniKvCacheScheduler(capacity3)
finished3, rejected3 = sched3.run(ordinary)
assert rejected3 == []
assert len(finished3) == 5
print("ordinary workload: all 5 requests finished, 0 rejected")
```

**实测(`.venv` 真跑):** 同样的 poison workload,修复后请求 1 在 `submit()` 阶段就被精确拒绝(`rejected == [(1, 0, 99, 3)]`),请求 0 和请求 2 不再被拖累,正常跑完。另外验证了容易和"拒绝"混淆的另一种情况——**总需求超过容量,但每个请求单独的需求都不超过容量**,这是完全正常的场景,应该排队而不是被拒绝:10 个请求各要 2 个 slot、池子只有 4 个,全部 10 个最终都跑完、0 个被拒绝,只是要排队(最长等了 20 步)。一个普通工作负载(5 个请求,互不刁难)端到端跑一遍,确认组装后的类没有破坏正常情况。

---

## 可以怎么继续扩展(只指方向,不实现)

- **从"整块声明"到真正的 PagedAttention 分页**:这篇教程的 slot 数是请求提交时一次性声明的固定整数,真实 [01 号文件](01-inference-engine-core.md)知识点 3 `PagedKvPool`/`BlockTable` 是逐 token 按需申请物理块、用 block table 做间接寻址——把这篇的 `SlotPool` 换成真正的按需分配,是更贴近生产实现的方向。
- **调度策略从"只有队首"换成 FCFS/SJF/priority**:[01 号文件](01-inference-engine-core.md)知识点 8 的三种 picker(先到先得/预期最短优先/业务优先级)可以直接替换阶段 4 `MiniKvCacheScheduler.step()` 里"谁在 pending 队首"这个隐含的 FCFS 假设,但要小心:一旦允许"插队",阶段 4 修复的"单个超大请求卡死后面所有人"这类问题会以新的形态出现(比如一个持续被插队的普通请求会不会被饿死),需要重新设计准入检查,不能想当然认为换个 picker 就万事大吉。
- **前缀共享**:这篇的每个请求各自独占自己的 slot,互不共享;[02 号文件](02-sglang-radixattention.md)的 RadixAttention 用一棵 radix tree 让多个请求共享公共前缀对应的物理块,只在分叉点才需要各自新增——把"谁独占多少 slot"换成"哪些 slot 被多少个请求共享引用",是完全不同量级的复杂度,这里不展开。
- **真正的抢占**:目前请求一旦被准入就会跑到底,没有任何机制能在中途把一个低优先级请求换出去让给更紧急的请求——这正是 01 号文件知识点 8 提到的、这份系列源码里从未真正实现过的能力,真实 vLLM 的 recompute/swap-to-CPU 机制就是解决这个问题的方向。

## 这篇教程展示的方法论

和 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 一样:挑几个关联的知识点 → 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实效果,而不是一次性甩出完整代码。这篇额外验证了一件事:教程体格式不只适合"从零拼出一个新工具",也适合"复现并修复一个真实 bug"——阶段 4 的 hang 不是为了教学效果编出来的场景,是撰写这篇教程之前的探索阶段里真实触发的边界条件,修复方式(提交时校验,而不是等运行时才发现)也是从这个真实教训里提炼出来的,不是先设计好场景再补一个"看起来像 bug"的演示。全篇纯 Python 模拟,不接真实模型/GPU/网络请求,和本系列 01-08 号文件"诚实标注纯 Python/torch 算法复现"的风格一致——这里甚至不需要 torch,调度器管理的是抽象的整数 slot 计数,不是真实张量。

---

*创建:2026-07-24*
