# 05 · 进阶深度追加:5 个真实二面级别的多级追问链案例

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是知识点,不计入统计——它和 [dsa-deep-dive/20-advanced-interview-depth.md](../dsa-deep-dive/20-advanced-interview-depth.md)、[torch-deep-dive/12-advanced-interview-depth.md](../torch-deep-dive/12-advanced-interview-depth.md) 是同一挂:方法论 + 案例,不是知识点列表。

## 为什么需要这篇追加内容

`01-04` 全部完成并自查通过之后,用户转达了一位有经验从业者的反馈:现有材料没有达到 **2026 年大厂技术二面** 的深度。[dsa-deep-dive/20-advanced-interview-depth.md](../dsa-deep-dive/20-advanced-interview-depth.md)、[torch-deep-dive/12-advanced-interview-depth.md](../torch-deep-dive/12-advanced-interview-depth.md) 等系列已经基于一次真实调研(三路 WebSearch:检索中国大厂面经、西方大厂面经、面试官视角的元讨论,而不是凭训练数据里的印象去猜)落地验证过一套格式并沉淀进项目 memory,核心发现是:真实的追问不是"正确性 → 复杂度 → 能不能优化"这一条线性链,而是至少沿着 **5 条独立轴线** 展开,并且经常在同一道题里综合出现:

1. **规模递增轴**——数据/场景规模从小到大跳跃式增长,每次跳跃都让原方案失效,逼出下一层方案。
2. **工程约束递增轴(并发/分布式)**——单机/单线程正确,不等于并发场景下还正确。
3. **方案批判迭代轴**——面试官不深挖单一方案的复杂度,而是连续指出该方案的具体工程缺陷逼你换方案,一题换 3-4 个技术方案是真实案例。
4. **决策依据追问轴("为什么选这个不选那个")**——不纠错,只逼问选择依据。
5. **真实性验证轴**——把候选人回答里抽象的表述("应该会更快""理论上更安全")追问压向具体证据(真实数字、真实报错信息、真实跑出来的行为),鉴别是真验证过还是凭直觉猜的。

面试官视角的元规律:一个 55 分钟轮次里,候选人给出初版方案通常只占 15 分钟,剩下 40 分钟全是追问——这提示"追问链"内容的权重应该远高于普通知识点讲解。

`python-advanced` 系列(`01-04`)覆盖函数与闭包进阶、迭代器与生成器、面向对象进阶、类型注解与上下文管理/并发共 20 个知识点,几乎每一节都以"这是 PyTorch/HuggingFace 源码里的真实写法"收尾——这批案例延续同样的路线,但把深度推到"这个你已经会用的黑盒机制,底层到底靠什么实现""这个方案只跑一次没问题,跑很多轮/换一种任务规模之后还成立吗""能不能现场把这个结论的反例也跑出来,而不是只验证正面案例"这个层次。其中至少一个案例专门展开了原文里明确留下的伏笔:[02 类"常见坑"第 3 条](02-iterators-and-generators.md)写过"这也是 PyTorch `IterableDataset` 常见的写法,这里先点一下,不展开"——这次真的展开,而且用真实的 `torch.utils.data.IterableDataset` 把对应的 bug 跑了出来。

本篇收敛到 **5 个案例**,组织原则是每个案例明确挂一条主轴线(部分案例会自然带出第二条轴线,在文末表格里如实标注,不强行只挂一条):

- **案例 1**(工程约束递增轴 + 决策依据追问轴):`threading` 还是 `multiprocessing`——任务粒度决定 multiprocessing 是助力还是拖累,以及 GIL 会不会消失的前瞻追问,建立在 [04-typing-context-and-concurrency.md 第5节](04-typing-context-and-concurrency.md) 之上。
- **案例 2**(方案批判迭代轴 + 真实性验证轴):装饰器叠加顺序——审计日志被缓存悄悄吞掉,建立在 [01-functions-and-closures.md 第4/5/6.3节](01-functions-and-closures.md) 之上。
- **案例 3**(决策依据追问轴 + 方案批判迭代轴):descriptor 协议——`@property` 的底层机制,以及"一份状态被所有实例共享"的坑,建立在 [03-oop-advanced.md 第1节](03-oop-advanced.md) 之上。
- **案例 4**(真实性验证轴 + 决策依据追问轴):类型注解真的不被强制检查吗——自己实现的运行时校验器会在哪里碰壁,建立在 [04-typing-context-and-concurrency.md 第1节](04-typing-context-and-concurrency.md) 之上。
- **案例 5**(规模递增轴 + 方案批判迭代轴):`IterableDataset` 的生成器只能用一次陷阱——第二个 epoch 静默返回空批次,建立在 [02-iterators-and-generators.md 第2/3/4节](02-iterators-and-generators.md) 之上。

**范围声明:** 这是方法论范例,不是把 20 个知识点全部重写一遍。每个案例都要求读者能看到"同样的追问方式,怎么套到任何一个已经掌握的知识点上"——读完之后,应该能自己对着 01-04 里任何一个没在这里出现的知识点(比如 `functools.partial`、`abc.ABC`、`asyncio.gather`、`@dataclass`),现场把这几条轴线走一遍练习,而不是指望这篇文档穷举所有可能的追问。

---

## 案例 1:threading 还是 multiprocessing——任务粒度决定 multiprocessing 是助力还是拖累(工程约束递增轴 + 决策依据追问轴)

建立在 [04-typing-context-and-concurrency.md 第5节](04-typing-context-and-concurrency.md) `threading` vs `multiprocessing` 之上——那一节已经用真实测时数据证明了"CPU 密集型任务用 `multiprocessing`、I/O 密集型任务用 `threading`"这条判断口诀(单线程 4.96s / 多线程 5.32s / 多进程 3.14s,以及 I/O 场景单线程 1.50s / 多线程 0.50s,全部是本机实测)。但这条口诀本身还留着两个没被追问过的缺口:"CPU 密集型任务是不是无脑上 `multiprocessing` 就一定更快",以及"GIL 这个前提本身会不会变"。

**追问链条完整还原:**

- **Q(基础,04 类已覆盖):** "CPU 密集型任务和 I/O 密集型任务,分别应该选 `threading` 还是 `multiprocessing`?" —— 期望候选人复述 04 类的实测结论:CPU 密集型选 `multiprocessing`(真正用上多核),I/O 密集型选 `threading` 或 `asyncio`(等待期间 GIL 会被释放)。
- **追问 1(不满足于口诀本身,逼问是不是无条件成立):** "是不是所有 CPU 密集型任务,只要上了 `multiprocessing` 就一定比单进程快?" —— 期望候选人意识到:04 类的实测例子里,每个子任务本身要跑很久(6000 万次循环),相对于"开一个新进程、把参数和结果跨进程传递(pickle 序列化)"这份固定开销,计算本身占了绝对大头;但如果单个任务很小(比如一次乘法),固定开销就会反过来主宰总耗时,这时候上 `multiprocessing` 反而更慢。
- **追问 2(逼出真实数字,不能停在直觉上):** "能不能现场测一下?一组是几百个'极小'的任务,一组是几个'很重'的任务,分别用 `multiprocessing.Pool` 和顺序执行对比,看这两种场景下 `multiprocessing` 是帮忙还是添乱。" —— 期望候选人写代码现场测量,给出具体倍数,而不是"应该会有区别"这种模糊表述。
- **深挖追问(前瞻,GIL 这个前提本身会不会变):** "你刚才所有的结论,都建立在'CPython 有 GIL'这个前提上——Python 3.13 开始官方支持 free-threading(无 GIL)构建,如果 GIL 被真的移除了,以上结论还成立吗?" —— 期望候选人知道 PEP 703 这件事本身,但更重要的是要能诚实分辨"我现在用的这个解释器,到底是不是那种构建、GIL 到底有没有被启用",而不是不假思索地说"反正以后这些都不用考虑了"——这正是"真实性验证"的态度延伸到"对前沿特性的态度"上。

**可运行例子(1/2):任务粒度决定 multiprocessing 是助力还是拖累——两个方向都真实测出来**

```python
import time
import multiprocessing


def tiny_task(x):
    """极细粒度任务:一次乘法,计算本身几乎不耗时"""
    return x * x


def big_task(n):
    """粗粒度任务:几百万次循环,计算本身占绝对大头"""
    total = 0
    for i in range(n):
        total += i * i
    return total


if __name__ == "__main__":
    # 场景一:400 个"极细粒度"任务——进程池的创建/调度/pickle 开销远大于任务本身
    items = list(range(400))

    t0 = time.perf_counter()
    seq_result = [tiny_task(x) for x in items]
    t_seq_tiny = time.perf_counter() - t0

    t0 = time.perf_counter()
    with multiprocessing.Pool(4) as pool:
        mp_result = pool.map(tiny_task, items)
    t_mp_tiny = time.perf_counter() - t0

    assert seq_result == mp_result
    assert t_mp_tiny > t_seq_tiny * 10, (
        f"细粒度任务下,multiprocessing 应该明显慢于顺序执行,"
        f"实测 mp={t_mp_tiny:.4f}s seq={t_seq_tiny:.4f}s"
    )

    # 场景二:4 个"粗粒度"任务——计算本身占绝对大头,进程池的固定开销被摊薄
    N = 6_000_000
    t0 = time.perf_counter()
    seq_result2 = [big_task(N) for _ in range(4)]
    t_seq_big = time.perf_counter() - t0

    t0 = time.perf_counter()
    with multiprocessing.Pool(4) as pool:
        mp_result2 = pool.map(big_task, [N] * 4)
    t_mp_big = time.perf_counter() - t0

    assert seq_result2 == mp_result2
    assert t_mp_big < t_seq_big * 0.7, (
        f"粗粒度任务下,multiprocessing 应该明显快于顺序执行,"
        f"实测 mp={t_mp_big:.4f}s seq={t_seq_big:.4f}s"
    )

    print(
        f"OK: 细粒度任务(400 个,每个只是一次乘法)multiprocessing 比顺序执行慢了 "
        f"{t_mp_tiny / t_seq_tiny:.0f} 倍(mp={t_mp_tiny:.4f}s, seq={t_seq_tiny:.4f}s);"
        f"粗粒度任务(4 个,每个 {N:,} 次循环)multiprocessing 比顺序执行快了 "
        f"{t_seq_big / t_mp_big:.2f} 倍(mp={t_mp_big:.2f}s, seq={t_seq_big:.2f}s)。"
        f"同一个 multiprocessing.Pool,谁快谁慢完全取决于'单个任务的计算量'能不能盖过"
        f"'创建进程 + 跨进程传参传结果'这份固定开销——不是'CPU 密集型就无脑上 multiprocessing'。"
    )
```

**可运行例子(2/2):先验证当前环境的 GIL 前提是否成立,再验证"GIL 挡的是 CPU-CPU 并行,不挡 CPU-IO 重叠"这条更精细的结论**

```python
import sys
import time
import threading


def cpu_busy(n, out, idx):
    total = 0
    for i in range(n):
        total += i * i
    out[idx] = total


def io_wait(seconds, out, idx):
    time.sleep(seconds)
    out[idx] = "done"


# 第一步:确认当前仓库 venv 到底是不是"没有 GIL"的构建——不能凭 Python 版本号就假设
assert sys.version_info >= (3, 13), "free-threading 相关的 sys._is_gil_enabled() 从 3.13 才开始提供"
assert hasattr(sys, "_is_gil_enabled"), "3.13+ 才有这个探测接口"
gil_enabled = sys._is_gil_enabled()
assert gil_enabled is True, (
    "本仓库 .venv 是常规构建(不是 python3.13t 那种 free-threading 构建),"
    "GIL 应该是启用状态——如果这里是 False,说明运行环境换成了 free-threading 解释器,"
    "本节后面'两个 CPU 线程不会真正并行'这条结论就不再成立,需要重新测量"
)

N = 30_000_000
SLEEP_S = 0.8

t0 = time.perf_counter()
out = [None]
cpu_busy(N, out, 0)
cpu_alone = time.perf_counter() - t0

t0 = time.perf_counter()
io_wait(SLEEP_S, out, 0)
sleep_alone = time.perf_counter() - t0

# 混合场景:一个 CPU 密集线程 + 一个纯 sleep 线程同时跑
out2 = [None, None]
t0 = time.perf_counter()
t1 = threading.Thread(target=cpu_busy, args=(N, out2, 0))
t2 = threading.Thread(target=io_wait, args=(SLEEP_S, out2, 1))
t1.start()
t2.start()
t1.join()
t2.join()
mixed = time.perf_counter() - t0

assert mixed < (cpu_alone + sleep_alone) * 0.92, (
    f"如果 GIL 完全不释放,mixed 应该约等于两者相加({cpu_alone + sleep_alone:.2f}s);"
    f"实测 mixed={mixed:.2f}s,应该明显小于相加值,说明 sleep 期间 GIL 真的被让了出去"
)

print(
    f"OK: 本机 venv sys._is_gil_enabled()={gil_enabled}(常规构建,GIL 启用中)。"
    f"CPU 密集线程单独跑 {cpu_alone:.2f}s,sleep 线程单独跑 {sleep_alone:.2f}s,"
    f"两个线程同时跑总耗时只要 {mixed:.2f}s(远小于顺序相加的 {cpu_alone + sleep_alone:.2f}s)——"
    f"这证明 GIL 挡住的是'两个 CPU 密集线程互相并行',不挡'CPU 线程和 sleep 线程重叠推进'"
    f"(sleep 期间线程会主动释放 GIL);04 类的结论('CPU 密集型多线程不加速')只在'两边都是 CPU 密集'"
    f"这个前提下成立,不能过度引申成'只要有 CPU 线程在跑,别的线程就完全动不了'。"
    f"如果换成 python3.13t 这种 free-threading 构建(sys._is_gil_enabled() 会是 False,或者需要用 "
    f"-X gil=0/PYTHON_GIL=0 显式关闭),两个 CPU 密集线程之间也能真正并行——但这个仓库的 .venv "
    f"现在跑的不是那种构建,这一点本身也是现场验证出来的,不是猜的。"
)
```

**常见坑:** 把"CPU 密集型任务用 multiprocessing"这条口诀当成无条件成立,没意识到任务粒度太细时,`multiprocessing` 的固定开销(创建进程 + 序列化传参传结果)会反过来主宰总耗时,是真实的性能倒退,不是"理论上有一点开销"这种轻描淡写;把"GIL"和"多线程完全没用"划等号,只要有 CPU 密集型代码在跑,就以为别的线程完全动不了,而实际上 GIL 会在解释器内部按时间片切换、遇到 I/O 阻塞也会主动释放,只有"两个纯 CPU 密集型线程互相并行"这一种场景才是 GIL 真正挡住的地方;一提到 free-threading/GIL 移除就想当然地说"以后这些结论全部作废",却没有先确认自己当前用的解释器构建到底支不支持、有没有开启。

---

## 案例 2:装饰器叠加顺序——审计日志被缓存悄悄吞掉(方案批判迭代轴 + 真实性验证轴)

建立在 [01-functions-and-closures.md 第4/5/6.3节](01-functions-and-closures.md) 装饰器基础、装饰器工厂、`functools.lru_cache` 之上——那几节分别讲了"怎么写一个装饰器""怎么给装饰器传参数""`lru_cache` 怎么自动缓存",但从来没有讨论过"多个装饰器叠在同一个函数上,顺序会不会影响行为",01 类甚至没有出现过"两个装饰器同时用"这种写法。

**追问链条完整还原(面试官/候选人对话,方案批判迭代):**

- **面试官给约束:** "有一个查询用户权限的函数,你需要同时做两件事:①用 `lru_cache` 缓存结果,避免重复查询数据库;②每一次有人调用这个函数,不管是不是命中缓存,都要往审计日志里记一条'谁在什么时候查询了什么资源'——这是安全合规的硬性要求。你会怎么写?"
- **候选人方案 1:** "两个装饰器叠在一起用不就行了——`@functools.lru_cache(maxsize=None)` 放外层,`@audit_log` 放内层。"
- **面试官指出具体缺陷(现场构造场景,不是空泛的"顺序好像有讲究"):** "同一个用户 `alice` 连续查询 3 次同一个资源,你这份审计日志里应该有几条记录?" —— 候选人现场跑一遍代码,发现只有 1 条,而不是应该有的 3 条。"如果安全团队回头审计,发现 `alice` 只被记录访问过一次,但实际上她访问了 3 次——你的审计日志现在是不是在说谎?"
- **候选人方案 2(纠正顺序):** "那反过来,`audit_log` 放外层,`lru_cache` 放内层。" —— 验证后确实每次都有记录了。
- **面试官追问(逼问失败路径,不能只验证成功路径):** "刚才用的例子里,`alice` 三次查询都成功了。如果 `alice` 三次查询的是一个她没有权限访问的资源,函数每次都抛异常拒绝访问——这种情况下,你最开始那个'有 bug'的顺序(`lru_cache` 在外层)会不会也丢日志?" —— 期望候选人意识到:`functools.lru_cache` 不会缓存异常(函数抛异常时不会存入缓存),所以哪怕是"错误顺序",失败的调用每次都会重新走到底层函数、重新抛异常、重新被内层的 `audit_log` 记录下来——这个 bug 只对"成功的重复调用"生效,失败路径完全不受影响,这本身是更隐蔽的性质:如果测试的时候刚好只测了会失败的输入,压根发现不了这个问题。
- **深挖追问(决策依据,收束成可操作的规律):** "所以你怎么总结'多个装饰器叠加'该按什么顺序摆?" —— 期望候选人给出可操作的判断标准:必须"看见每一次调用"的关注点(审计、埋点、限流计数)要放在最外层;可以"被短路掉"的关注点(缓存)放在内层——外层装饰器能不能感知到内层某次调用,完全取决于内层会不会提前短路返回而不真的往下传播,不是"谁在语法上写在上面谁就更重要"。

**可运行例子(1/2):同一个顺序问题,连续 3 次成功调用下两种顺序的审计日志条数差异**

```python
import functools

audit_log = []


def audit(func):
    """记录每一次真正到达这里的调用——不管是不是命中缓存都应该被记录,
    这是这个装饰器存在的全部意义"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        audit_log.append((func.__name__, args, kwargs))
        return func(*args, **kwargs)
    return wrapper


@functools.lru_cache(maxsize=None)
@audit
def check_access_cache_outside(user_id, resource):
    return f"granted:{user_id}:{resource}"


@audit
@functools.lru_cache(maxsize=None)
def check_access_audit_outside(user_id, resource):
    return f"granted:{user_id}:{resource}"


# 场景:同一个用户对同一个资源连续查询 3 次(全部成功)
audit_log.clear()
for _ in range(3):
    check_access_cache_outside("alice", "dataset-A")
count_cache_outside = len(audit_log)

audit_log.clear()
for _ in range(3):
    check_access_audit_outside("alice", "dataset-A")
count_audit_outside = len(audit_log)

assert count_cache_outside == 1, (
    f"lru_cache 在外层时,3 次相同调用应该只有第 1 次真正走到 audit 装饰器,"
    f"后 2 次被缓存直接短路掉,实际记录了 {count_cache_outside} 条"
)
assert count_audit_outside == 3, (
    f"audit 在外层时,不管内层是不是命中缓存,3 次调用都应该各自留下一条审计记录,"
    f"实际只记录了 {count_audit_outside} 条"
)

print(
    f"OK: 同一个用户对同一个资源连续查询 3 次(全部成功)。"
    f"@lru_cache 叠在外层(@lru_cache / @audit):审计日志只有 {count_cache_outside} 条"
    f"(后 2 次成功的重复访问被缓存悄悄吞掉,安全团队审计时会看漏 2 次真实访问);"
    f"@audit 叠在外层(@audit / @lru_cache):审计日志有 {count_audit_outside} 条"
    f"(不管命不命中缓存,每次调用都被完整记录,这才是审计日志应有的行为)。"
)
```

**可运行例子(2/2):这个 bug 更隐蔽的一面——只吞掉成功的重复调用,失败路径完全不受影响**

```python
import functools

audit_log = []


def audit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        audit_log.append((func.__name__, args, kwargs))
        return func(*args, **kwargs)
    return wrapper


class AccessDenied(Exception):
    pass


@functools.lru_cache(maxsize=None)
@audit
def check_access_cache_outside(user_id, resource):
    if resource == "restricted-vault":
        raise AccessDenied(f"{user_id} 无权访问 {resource}")
    return f"granted:{user_id}:{resource}"


# 对照组 1:连续 3 次成功调用(相同参数)——bug 会让日志条数变少
audit_log.clear()
for _ in range(3):
    check_access_cache_outside("alice", "dataset-A")
success_log_count = len(audit_log)

# 对照组 2:连续 3 次失败调用(相同参数,每次都被拒绝)——异常不会被 lru_cache 缓存
audit_log.clear()
denied_count = 0
for _ in range(3):
    try:
        check_access_cache_outside("mallory", "restricted-vault")
    except AccessDenied:
        denied_count += 1
failure_log_count = len(audit_log)

assert success_log_count == 1, "重复的成功调用应该被缓存吞掉,只留下 1 条审计记录"
assert denied_count == 3, "3 次都应该真的被拒绝(拒绝逻辑本身没有被缓存影响)"
assert failure_log_count == 3, (
    f"functools.lru_cache 不会缓存异常,所以哪怕装饰器顺序是'有 bug'的那种,"
    f"失败路径依然每次都会重新走到底层函数,每次都应该被 audit 记录,"
    f"实际记录了 {failure_log_count} 条"
)

print(
    f"OK: 同样是'lru_cache 叠在外层'这个有 bug 的顺序——连续 3 次成功调用只留下 "
    f"{success_log_count} 条审计记录(bug 真实存在);但连续 3 次失败调用(被拒绝)留下了 "
    f"{failure_log_count} 条(bug 不影响失败路径,因为 lru_cache 从不缓存异常)。"
    f"这就是这个坑更隐蔽的地方:如果测试或者第一次上线验证时,凑巧用的是会失败/会被拒绝的输入,"
    f"审计日志看起来完全正常,只有'重复的成功访问'才会被悄悄吞掉,很容易蒙混过关。"
)
```

**常见坑:** 只用"成功路径"测试装饰器叠加顺序对不对,而这类 bug 恰好在失败路径下不发作,造成"看起来测过了、其实只测了一半"的假象;觉得装饰器顺序只是"风格问题"随手叠加,没意识到外层装饰器能不能"看见"内层某次调用,完全取决于内层会不会提前短路返回(缓存命中,也包括其他会提前 `return` 而不继续调用 `func(...)` 的装饰器,比如限流拒绝);判断准则记反——把"需要看见每一次调用"的装饰器放进了内层,"可以被短路"的装饰器放在了外层,外层不代表"更重要"或"更先执行的逻辑",外层代表的是"不管内层做了什么优化/短路,都逃不掉我的视线"。

---

## 案例 3:descriptor 协议——`@property` 的底层机制,以及"一份状态被所有实例共享"的坑(决策依据追问轴 + 方案批判迭代轴)

建立在 [03-oop-advanced.md 第1节](03-oop-advanced.md) `@property`/`@staticmethod`/`@classmethod` 之上——那一节用"温度不能低于绝对零度"这个例子讲清楚了 `@property` + `.setter` 怎么在"调用方式不变"的前提下加上校验逻辑,但没有深入到"`@property` 本身是靠什么机制生效的"这一层,也没有讨论过"如果同一种校验逻辑要在好几个字段上重复使用,`@property` 该怎么复用"这个问题。

**追问链条完整还原:**

- **Q(基础,03 类已覆盖):** "`@property` 你已经会用了,它在 Python 语言层面到底靠什么机制生效?" —— 期望候选人答出(或在提示下想到)**描述符协议(descriptor protocol)**:一个类只要定义了 `__get__`(以及可选的 `__set__`/`__delete__`),它的实例被当成另一个类的**类属性**使用时,对这个属性的访问就会被自动路由到这几个方法上——`@property` 本身就是标准库提供的一个描述符实现,不是专属魔法。
- **追问 1(逼出现场实现,不能只说名词):** "不用 `@property` 语法糖,自己写一个类,实现一样的'正数校验'逻辑,让它能被多个不同的字段复用。" —— 期望候选人写出一个 `PositiveNumber` 描述符类,内部有 `__get__`/`__set__`。
- **追问 2(方案批判,现场构造出具体缺陷):** "你这个描述符类的实例,是在 `class Account:` 定义的时候创建的一次,还是每个 `Account` 实例各自有一份?如果我创建两个 `Account` 实例,`a = Account(100)` 和 `b = Account(50)`,再回头看 `a.balance`,应该是多少?" —— 逼着候选人现场跑一遍代码验证:如果状态被朴素地存在描述符对象自己身上(`self.value = value`),会发现 `a.balance` 被 `b` 的赋值"污染"成了 50,这是一个真实的、结构性的 bug,不是理论上的边缘情况。
- **深挖追问(决策依据,逼问修复方案本身的选择):** "怎么修?用一个普通的 `dict`,以实例本身当 key,把每个实例的值分开存,可以吗?" —— 期望候选人先给出这个"看起来对"的方案,面试官继续追问:"这样的 `dict` 会不会导致内存泄漏?" —— 期望候选人推出:普通 `dict` 会对 key(也就是每个 `Account` 实例)持有一份强引用,只要这个描述符对象(通常和类本身一样长寿)还活着,它内部那个 `dict` 就会一直拽着每一个曾经被赋值过的实例不放,哪怕外部所有其它地方都已经不再需要这个实例,也无法被垃圾回收——这是一个真实的、缓慢积累的内存泄漏。正确答案是 `weakref.WeakKeyDictionary`:它对 key 只持弱引用,实例被外部正常回收时,`WeakKeyDictionary` 里对应的条目会自动消失。

**可运行例子(1/2):朴素描述符的状态污染——两个不同实例的赋值互相覆盖**

```python
class PositiveNumberBuggy:
    """朴素版描述符:校验逻辑是对的,但存储状态的地方错了——
    self.value 存在描述符对象自己身上,而这个描述符对象本身是类属性,
    在类定义时只创建了一次,会被这个类的所有实例共享"""

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.value

    def __set__(self, instance, value):
        if value <= 0:
            raise ValueError(f"{self.name} 必须为正数, 收到 {value}")
        self.value = value  # bug 就在这一行


class AccountBuggy:
    balance = PositiveNumberBuggy("balance")   # 只创建了 1 个 PositiveNumberBuggy 实例

    def __init__(self, balance):
        self.balance = balance


a = AccountBuggy(100)
b = AccountBuggy(50)

# 两个不同的 Account 实例,却共用同一个描述符对象,b 的赋值"污染"了 a 已经设置好的值
assert AccountBuggy.__dict__["balance"] is AccountBuggy.__dict__["balance"]
assert a.balance == 50, f"污染 bug 真实复现:a 明明被设成了 100,读出来却是 {a.balance}"
assert a.balance == b.balance, "两个本应独立的实例,现在读到的是同一份状态"

print(
    f"OK: AccountBuggy.balance 这个描述符对象在两个实例之间只有一份——a.balance 读出来是 "
    f"{a.balance},而不是构造时传入的 100。b = AccountBuggy(50) 的赋值把 a 的状态覆盖了。"
    f"这不是理论上的边缘情况,是两个实例、两行代码就能真实复现的结构性 bug。"
)
```

**可运行例子(2/2):用 WeakKeyDictionary 修复状态污染,并用 weakref.finalize 精确验证没有内存泄漏**

```python
import weakref
import gc


class PositiveNumberFixed:
    """修复版:用 WeakKeyDictionary 把状态分开存到每个实例名下,
    而不是存在描述符对象自己身上"""

    def __init__(self, name):
        self.name = name
        self._values = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self._values[instance]

    def __set__(self, instance, value):
        if value <= 0:
            raise ValueError(f"{self.name} 必须为正数, 收到 {value}")
        self._values[instance] = value


class AccountFixed:
    balance = PositiveNumberFixed("balance")

    def __init__(self, balance):
        self.balance = balance


a2 = AccountFixed(100)
b2 = AccountFixed(50)
assert a2.balance == 100 and b2.balance == 50, "两个实例现在应该各自独立"

# 验证没有内存泄漏:用 weakref.finalize 精确捕捉"对象真的被销毁了"这一刻,不用猜
c2 = AccountFixed(77)
finalized = []
weakref.finalize(c2, lambda: finalized.append(True))

before_count = len(AccountFixed.balance._values)
del c2
gc.collect()
after_count = len(AccountFixed.balance._values)

assert before_count == 3, f"删除前应该有 3 个实例的状态(a2/b2/c2),实际 {before_count}"
assert finalized == [True], "c2 应该被真实回收,不应该被 WeakKeyDictionary 拽住不放"
assert after_count == 2, f"c2 被回收后,WeakKeyDictionary 里对应的条目应该自动消失,实际还剩 {after_count} 条"

print(
    f"OK: WeakKeyDictionary 修复后,a2.balance={a2.balance}, b2.balance={b2.balance}(互不干扰)。"
    f"c2 被 del 并 gc.collect() 之后,weakref.finalize 回调真实触发({finalized}),"
    f"WeakKeyDictionary 里的条目数从 {before_count} 自动降到 {after_count}——如果这里用的是普通 dict,"
    f"c2 会被 dict 的 key 强引用永远拽住,finalize 回调永远不会触发,条目数会一直停在 {before_count}。"
)
```

**常见坑:** 把描述符类的实例状态直接存在描述符自己身上(`self.value = value`),没意识到这个描述符对象本身是类属性,类定义时只创建一次,会被这个类的所有实例共享——这是新手写描述符最容易踩的坑,而且不会报错,只会安静地"串数据";修复时选用普通 `dict` 而不是 `WeakKeyDictionary`,顺手解决了"共享状态"的 bug,却引入了一个更慢性、更难发现的内存泄漏,只要这个描述符所在的类还活着(通常和进程一样长寿),它内部的普通 `dict` 就会不断累积"再也没人用得到,但因为被当过 key 而无法被回收"的实例;把描述符协议和 `@property` 当成两个不相关的东西,没意识到 `@property` 就是标准库内置的一个描述符实现——理解了描述符协议本身,才能看懂 `functools.cached_property`、ORM 框架里的字段定义这类"看起来是普通属性赋值,其实背后有一整套协议在运作"的写法。

---

## 案例 4:类型注解真的不被强制检查吗——自己实现的运行时校验器会在哪里碰壁(真实性验证轴 + 决策依据追问轴)

建立在 [04-typing-context-and-concurrency.md 第1节](04-typing-context-and-concurrency.md) 类型注解与 `typing` 模块之上——那一节用真实代码证明了"Python 解释器在运行时完全不检查类型注解,注解错了也不会报错",这个结论本身是对的,但它制造了一个看起来的矛盾:很多人真实用过 FastAPI,传一个类型不对的参数进去,接口真的会报错、真的会返回详细的校验错误信息。这个案例现场把这个"矛盾"拆开。

**追问链条完整还原:**

- **Q(基础,04 类已覆盖):** "类型注解在 Python 运行时会被强制检查吗?" —— 期望候选人复述 04 类的结论:不会,类型注解只是文档 + 给 mypy/IDE 用的元数据,解释器不检查。
- **追问 1(制造表面矛盾,逼问候选人解决而不是回避):** "那为什么很多人实际用 FastAPI 定义一个接口,参数标了 `age: int`,真的传一个非法字符串进去,FastAPI 会真的返回 422 错误,里面还有具体的校验信息?这不就是运行时检查吗,和你刚才说的是不是矛盾?" —— 期望候选人意识到并不矛盾:Python 解释器本身确实什么都不查,但类型注解这份信息在运行时是**可以被读取的普通数据**(`__annotations__`/`typing.get_type_hints()`)——FastAPI/Pydantic 是在用户空间的库代码里,自己读了这份注解信息,自己拿它去做 `isinstance` 之类的校验,校验逻辑是**它们自己手写出来的**,不是 Python 语言本身在替你查。
- **追问 2(逼出现场验证,不能只停留在"应该是这样"):** "你能现场写一个简化版吗?一个装饰器,读函数的类型注解,调用前自己做校验,校验不过就抛错。" —— 期望候选人现场用 `typing.get_type_hints` + `inspect.signature` + `isinstance` 写出一个 `@validate_types` 装饰器,验证它确实能拦下类型不对的参数。
- **深挖追问(找简化版的边界,决策依据):** "你这个简化版拿一个标注成 `list[int]` 的参数试试,传一个真的 `list` 进去,会发生什么?" —— 期望候选人现场跑,发现 `isinstance(value, list[int])` 本身就会抛 `TypeError: isinstance() argument 2 cannot be a parameterized generic`——连"校验能不能进行"这件事本身都会先崩溃。逼问怎么修:候选人需要想到用 `typing.get_origin()` 把 `list[int]` 降级成裸 `list` 再做 `isinstance`。面试官继续追问:"这样修好之后,还有什么校验不到的地方?" —— 期望候选人认识到:降级成裸 `list` 之后,只能校验"这是不是一个 list",完全校验不了"list 里的元素是不是都是 int"——如果调用方传一个装满字符串的 list,这个简化校验器会放行,错误会被推迟到函数体内部某个地方才爆出来,而且报错信息完全不会指向"你传错类型了"这个真正的原因。这正是为什么 Pydantic/FastAPI 这类真正成熟的库,内部要处理的远不止一个 `isinstance` 调用这么简单。

**可运行例子(1/2):自己实现的 @validate_types 确实能拦下简单类型的错误调用**

```python
import functools
import inspect
import typing


def validate_types(func):
    """简化版运行时类型校验器:读函数的类型注解,调用前用 isinstance 挨个检查实参"""
    hints = typing.get_type_hints(func)
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            if name not in hints:
                continue
            expected = hints[name]
            if not isinstance(value, expected):
                raise TypeError(
                    f"{func.__name__}() 参数 {name!r} 期望类型 {expected}, 实际收到 {type(value)}"
                )
        return func(*args, **kwargs)
    return wrapper


@validate_types
def compute_score(name: str, weight: float) -> float:
    return len(name) * weight


ok = compute_score("alice", 2.0)
assert ok == len("alice") * 2.0   # 用当场计算出来的期望值比较,不写死一个心算的浮点数

try:
    compute_score("alice", "not-a-float")
    assert False, "传字符串给标注成 float 的参数,应该被拦下来"
except TypeError as e:
    caught_message = str(e)

assert "weight" in caught_message and "float" in caught_message

print(
    f"OK: 简化版 @validate_types 装饰器——Python 解释器本身不检查类型注解,但注解信息"
    f"(__annotations__/typing.get_type_hints())在运行时是可以被读取的普通数据。"
    f"这个装饰器自己读了这份数据,自己用 isinstance 做校验,真的拦下了"
    f"compute_score('alice', 'not-a-float') 这次调用:{caught_message!r}。"
    f"这就是 FastAPI/Pydantic'看起来在运行时强制检查类型'的真相:检查是库代码自己写的,"
    f"不是 Python 语言本身在做。"
)
```

**可运行例子(2/2):同一个校验器遇到参数化泛型直接崩溃,修复后又暴露出"只查外壳不查内容"的真实缺口**

```python
import functools
import inspect
import typing


def validate_types_naive(func):
    hints = typing.get_type_hints(func)
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            if name not in hints:
                continue
            expected = hints[name]
            if not isinstance(value, expected):
                raise TypeError(f"{name} 期望 {expected}")
        return func(*args, **kwargs)
    return wrapper


@validate_types_naive
def process_batch_naive(items: list[int]) -> int:
    return sum(items)


# 第一步:参数化泛型直接把"能不能校验"这件事本身干崩溃,不是校验失败,是校验器自己报错
try:
    process_batch_naive([1, 2, 3])
    assert False, "isinstance() 遇到 list[int] 这种参数化泛型应该直接抛 TypeError"
except TypeError as e:
    naive_crash_message = str(e)
assert "parameterized generic" in naive_crash_message


def validate_types_v2(func):
    """用 typing.get_origin() 把 list[int] 降级成裸 list 再做 isinstance,
    修好了"校验器自己崩溃"这个问题,但只能查外壳(是不是 list),查不了里面的元素类型"""
    hints = typing.get_type_hints(func)
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            if name not in hints:
                continue
            expected = hints[name]
            origin = typing.get_origin(expected)
            check_against = origin if origin is not None else expected
            if not isinstance(value, check_against):
                raise TypeError(f"{name} 期望 {expected}, 实际 {type(value)}")
        return func(*args, **kwargs)
    return wrapper


@validate_types_v2
def process_batch_v2(items: list[int]) -> int:
    return sum(items)


correct = process_batch_v2([1, 2, 3])
assert correct == sum([1, 2, 3])

# 第二步:真实的缺口——传一个 list[str],外壳校验(是不是 list)能通过,
# 但元素类型完全没被检查,错误被推迟到函数体内部,报错也完全不指向"参数类型错了"
try:
    process_batch_v2(["a", "b", "c"])
    assert False, "list 里全是字符串,求和应该在函数体内部失败"
except TypeError as e:
    downstream_message = str(e)

assert "items" not in downstream_message  # 报错完全没提到是 items 参数的问题
assert "unsupported operand" in downstream_message

print(
    f"OK: 朴素版校验器遇到 list[int] 这种参数化泛型,isinstance 本身直接崩溃:"
    f"{naive_crash_message!r}(校验器自己报错,不是校验没通过)。用 typing.get_origin() "
    f"降级成裸 list 修好之后,process_batch_v2([1,2,3]) 能正常校验通过并算出 {correct};"
    f"但传 process_batch_v2(['a','b','c'])(外壳是 list,满足'裸 list'这层校验,元素类型"
    f"完全没被检查)时,错误被推迟到函数体内部的 sum() 才爆出来:{downstream_message!r}——"
    f"这条报错完全不会提到 items 参数或者类型注解,排查起来比'校验器一开始就说清楚哪个参数"
    f"类型不对'要困难得多。这就是为什么自己写的 isinstance 版校验器只能算'能用',真正成熟的"
    f"Pydantic/FastAPI 内部要对 typing 模块的每一种泛型结构分别处理、递归校验元素,复杂度"
    f"远不是一个 isinstance 调用能打发的。"
)
```

**常见坑:** 看到 FastAPI/Pydantic"真的会报运行时类型错误",就得出"Python 类型注解在运行时其实是会被检查的"这个错误结论——没有分清"语言本身不检查"和"库代码利用语言暴露的反射信息自己实现了检查"是两件完全不同的事;自己写校验器时,对 `list[int]`/`dict[str, int]` 这类参数化泛型直接套 `isinstance()`,不知道这本身就会抛错——这个坑比"校验逻辑写错了"更底层,是"校验这个动作本身还没开始就已经失败了";用 `typing.get_origin()` 修好参数化泛型的外壳校验之后,误以为已经"支持泛型校验"了,没意识到元素类型完全没查,给使用这个装饰器的同事传达了错误的安全感。

---

## 案例 5:IterableDataset 的生成器只能用一次陷阱——第二个 epoch 静默返回空批次(规模递增轴 + 方案批判迭代轴)

建立在 [02-iterators-and-generators.md 第2/3/4节](02-iterators-and-generators.md) 之上——那一篇的常见坑第 3 条已经点出"生成器只能遍历一次",并且专门留了一句伏笔:"想要重新遍历,必须重新调用生成器函数,拿一个全新的生成器对象……这也是 PyTorch `IterableDataset` 常见的写法,这里先点一下,不展开。"这个案例现场把这句"不展开"的话展开,用真实的 `torch.utils.data.IterableDataset` + `DataLoader` 复现出一个只有跑够"第二个 epoch"才会暴露的真实 bug。

**追问链条完整还原(面试官/候选人对话,方案批判迭代):**

- **面试官给约束:** "写一个 `IterableDataset`,包一份数据,交给 `DataLoader` 训练——训练要跑好几个 epoch,每个 epoch 都要把这份数据完整过一遍。"
- **候选人方案 1(直觉上"提前把生成器准备好,效率更高"):** "我在 `__init__` 里就把生成器建好存成 `self.gen`,`__iter__` 直接 `return self.gen`,这样每次迭代都不用重新构造。" —— 候选人现场跑一遍单个 epoch,数据完整、顺序正确,看起来没问题。
- **面试官指出具体缺陷(不是空泛的"这样不太好",是现场逼出一个具体、可复现的错误结果):** "你这跑的是第 1 个 epoch。真实训练不会只跑 1 个 epoch——把这个 `DataLoader` 再完整遍历一次,模拟第 2 个 epoch,你猜会拿到什么?" —— 候选人现场跑,发现第 2 个 epoch 拿到的是空的,不是数据变少,是完全一个 batch 都没有。
- **面试官追问(逼问这个 bug 有多危险,不能轻描淡写):** "这次遍历会报错、会崩溃、会有任何警告吗?" —— 期望候选人现场确认:完全没有,`for batch in dataloader:` 循环体一次都不会执行,程序继续往下跑,没有任何异常、任何警告。如果训练循环里"这个 epoch 的 loss"只在循环体内部更新,这一个 epoch 会表现成"瞬间跑完,指标却毫无变化"——这种静默失败远比直接崩溃更难排查。
- **候选人方案 2(纠正):** "`__iter__` 应该每次被调用都建一个新的生成器,不能重复用存好的那个。" —— 两种具体写法:要么 `__iter__` 里现场写 `return (x for x in self.data)`,要么把 `__iter__` 自己写成一个带 `yield`/`yield from` 的生成器函数——后者更值得深挖,因为它和 02 类已经学过的 `yield from` 直接呼应。
- **深挖追问(决策依据,不止一种修法,要能对比代价):** "除了'每次都建一个新生成器'这个思路,还有没有别的解法?各自的代价是什么?" —— 期望候选人提出替代方案:如果数据量不是大到内存放不下,干脆别用 `IterableDataset`(它天生就要求你自己保证"可重复迭代"这件事),改用普通 `Dataset` + `__len__`/`__getitem__`(02 类第 1 节已经学过的协议),`DataLoader` 每次通过下标随机访问,天然支持重复遍历、乱序访问,不存在"只能用一次"的问题——代价是数据必须能被"随机访问"(比如已经全部加载进内存,或者是能按下标直接寻址的文件格式);对于真正的流式数据源(比如从网络实时拉取、数据大到内存放不下),就只能老老实实保证 `IterableDataset.__iter__` 每次都返回新的生成器。

**可运行例子(1/2):用真实的 torch.utils.data.IterableDataset + DataLoader 复现"第二个 epoch 静默返回空批次"**

```python
import torch
from torch.utils.data import IterableDataset, DataLoader


class BadIterableDataset(IterableDataset):
    """直觉上"提前建好生成器更高效"——但生成器只能用一次(02 类已经讲过),
    这里只创建了一次,存成 self.gen,后面每次 __iter__ 都返回同一个对象"""
    def __init__(self, data):
        self.gen = (x for x in data)

    def __iter__(self):
        return self.gen


data = list(range(6))
bad_ds = BadIterableDataset(data)
bad_loader = DataLoader(bad_ds, batch_size=2)

epoch1 = [batch.tolist() for batch in bad_loader]
epoch2 = [batch.tolist() for batch in bad_loader]   # 模拟第二个 epoch,重新完整遍历一次

assert epoch1 == [[0, 1], [2, 3], [4, 5]], f"第一个 epoch 应该拿到完整数据,实际 {epoch1}"
assert epoch2 == [], (
    f"第二个 epoch 应该静默拿到空列表(不是报错,是真的一个 batch 都没有),实际 {epoch2}"
)

print(
    f"OK: BadIterableDataset 真实复现——第一个 epoch 完整拿到 {epoch1},"
    f"第二个 epoch(对同一个 DataLoader 再完整遍历一次)拿到 {epoch2}(完全是空的)。"
    f"整个过程没有任何异常、任何警告,for batch in bad_loader 的循环体在第二个 epoch 里"
    f"一次都不会执行——这是 02 类'生成器只能用一次'那条常见坑,在真实 PyTorch 训练场景下"
    f"的真实代价:不是崩溃,是训练脚本从第 2 个 epoch 开始安静地什么都没训练。"
)
```

**可运行例子(2/2):两种修法都能正确支持多个 epoch——新建生成器表达式 / 让 __iter__ 自己就是生成器函数**

```python
import torch
from torch.utils.data import IterableDataset, DataLoader


class GoodIterableDatasetExpr(IterableDataset):
    """修法一:__iter__ 每次被调用都现场建一个新的生成器表达式"""
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return (x for x in self.data)


class GoodIterableDatasetYieldFrom(IterableDataset):
    """修法二:__iter__ 自己写成一个生成器函数(内部有 yield from)——
    每次"调用"一个生成器函数,本身就会产出一个全新的生成器对象,
    这和 02 类的 yield from 委托技巧是同一个机制,只是用在了 __iter__ 这个协议方法上"""
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        yield from self.data


data = list(range(6))
expected = [[0, 1], [2, 3], [4, 5]]

for cls in (GoodIterableDatasetExpr, GoodIterableDatasetYieldFrom):
    ds = cls(data)
    loader = DataLoader(ds, batch_size=2)
    ep1 = [batch.tolist() for batch in loader]
    ep2 = [batch.tolist() for batch in loader]
    ep3 = [batch.tolist() for batch in loader]
    assert ep1 == ep2 == ep3 == expected, (
        f"{cls.__name__}: 三个 epoch 应该每次都拿到完整数据,实际 ep1={ep1} ep2={ep2} ep3={ep3}"
    )

print(
    f"OK: 两种修法(现场新建生成器表达式 / __iter__ 自己就是带 yield from 的生成器函数),"
    f"连续跑 3 个模拟 epoch,每次都完整拿到 {expected},互不影响。"
    f"两者本质是同一件事:__iter__ 每次被'调用'这个动作本身,都必须产出一个全新的生成器对象——"
    f"要么手动 return 一个新的生成器表达式,要么干脆让 __iter__ 自己带 yield/yield from"
    f"(调用一个生成器函数永远返回新对象,这一点在 02 类第 2 节已经验证过),两种写法在这里等价。"
)
```

**常见坑:** 在 `__init__` 里"提前"把生成器建好存成实例属性,直觉上像是一种优化(避免每次 `__iter__` 都重新构造),实际上是唯一一份状态被首次遍历耗尽后,后续所有遍历都会静默返回空结果——这是"生成器只能用一次"这条已知规则,换了一个更容易被忽视的地方(类的 `__init__` vs `__iter__`)复发;只用"跑一次看结果对不对"来验证 `IterableDataset` 写得对不对,没有模拟"多个 epoch"这个真实训练场景下必然发生的"重复遍历"——不少真实项目里这类 bug 是靠"模型效果第二个 epoch 开始不再提升"这种间接现象过了很久才被排查出来的,而不是靠报错发现的;把 `IterableDataset` 当成"万能"的数据集写法,而不评估数据本身是不是真的需要"流式、不能随机访问"这个前提——如果数据量本来就能整个放进内存,用支持 `__len__`/`__getitem__` 的普通 `Dataset`(02 类第 1 节已经讲过这套协议)反而从根本上不会有"只能遍历一次"这个问题。

---

## 小结:5 个案例对应调研发现的哪些轴线

| 案例 | 规模递增轴 | 工程约束递增轴(并发) | 方案批判迭代轴 | 决策依据追问轴 | 真实性验证轴 |
|---|---|---|---|---|---|
| 1. threading/multiprocessing 任务粒度 + GIL 前瞻 | | ✅ 核心 | | ✅(粒度判断标准) | |
| 2. 装饰器叠加顺序吞审计日志 | | | ✅ 核心 | | ✅(失败路径不受影响) |
| 3. descriptor 协议状态共享 | | | ✅(朴素实现被否定) | ✅ 核心 | |
| 4. 类型注解运行时校验器 | | | | ✅(isinstance 的边界) | ✅ 核心 |
| 5. IterableDataset 静默空 epoch | ✅ 核心 | | ✅(方案被否定换写法) | | |

这 5 个案例不是要覆盖 20 个知识点里的每一个——它们演示的是**方法论本身**:拿到任何一个已经掌握的知识点,都可以自己追问"这个你已经会用的机制,底层到底靠什么实现""这个方案只跑一次没问题,跑很多轮/换一种任务规模之后还成立吗""面试官连续指出具体缺陷时,下一个更合理的方案是什么""能不能用真实跑出来的证据而不是一句'应该会更快/更安全'说服别人"。真正的二面深度,是能不能对着一个自己没准备过的知识点,现场把这几条轴线走一遍。
