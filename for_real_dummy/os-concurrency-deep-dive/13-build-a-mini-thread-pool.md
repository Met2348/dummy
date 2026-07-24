# 13 · 手把手实战:从零写一个迷你线程池

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 13 个"知识点",不计入 79 个知识点的统计——和 [12 类](12-mock-interview-capstone.md)模拟终面 capstone 是同一挂"正式知识点之外的额外产出",但风格不一样:12 号文件里,你是**旁观者**,跟着面试官和候选人的对话把一条推理链条看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立搭出一个真实能跑的迷你线程池,全程不调用标准库自带的 `concurrent.futures.ThreadPoolExecutor`。不用现成线程池,是因为这一篇的目的不是"用工具"而是"造工具"——只有真的把 [03 类](03-synchronization-primitives.md)、[04 类](04-advanced-concurrency-patterns.md)讲过的原语拼一遍,才会撞见文字描述里讲不出来的那些细节(比如下面阶段 1 会真实复现的一个坑:任务里的异常处理不好,会让整个 worker 线程"安静地死掉"而不报任何错)。

## 为什么是"线程池"

不是要发明新知识点,是把几个你已经学过的知识点串成一个真实有用的东西。[04 类知识点3](04-advanced-concurrency-patterns.md)已经讲过线程池的设计参数——核心线程数、任务队列、拒绝策略——是什么、为什么这样设计,但那一节的可运行例子直接用 `concurrent.futures.ThreadPoolExecutor` 验证"`max_workers=3` 时最大并发数不超过 3"这一个结论,并没有要求你把线程池的内部机制真正敲出来。这一篇要做的正是那件事:自己维护一个任务队列、自己管理一组 worker 线程、自己实现"提交任务后怎么才能在需要的时候拿到结果",最后自己实现"怎么优雅地关掉这一整套机制"。

| 阶段 | 让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 一个 worker 线程能不停地从队列里取任务、执行,不用为每个任务创建一个新线程 | [04 类知识点3](04-advanced-concurrency-patterns.md) 线程池设计思想、[04 类知识点5](04-advanced-concurrency-patterns.md) `queue.Queue` 并发容器 |
| 阶段 2 | 扩展到多个 worker 线程,现场验证任务确实被并发处理,而不是排着队串行执行 | [01 类知识点3](01-processes-and-threads.md) GIL 在 IO 密集场景下仍允许多线程真实加速 |
| 阶段 3 | `submit` 不再是"发射后不管",而是返回一个凭证,之后能在需要的时候拿到任务的真实返回值(或者真实抛出的异常) | [03 类知识点6](03-synchronization-primitives.md) 条件变量——等待"结果是否已经就绪"这个条件 |
| 阶段 4 | 支持优雅关闭:不再接受新任务,但已经排队的任务会被处理完才真正退出 | [03 类知识点4](03-synchronization-primitives.md) 信号量/生产者消费者"生产者停止生产、消费者排空后退出"的思想 |

每个阶段的代码都能独立运行(本文件用同目录下的 `_verify_md.py` 校验,校验方式和 dsa-deep-dive/21 一样:把每个 ` ```python ` 代码块单独拎出来起一个新的子进程执行——块与块之间**不共享任何变量**,所以后面阶段用到前面阶段的类时会重新贴一遍完整定义,不是偷懒复制,是这套校验机制要求的)。

---

## 阶段 1:单 worker 版本——先让"任务排队执行"这件事跑起来

最省事的并发方式是来一个任务就 `threading.Thread(target=fn).start()`,但[04 类知识点3](04-advanced-concurrency-patterns.md)的底层机制已经讲过:创建一个操作系统线程有真实的开销(内核对象分配、栈内存分配、调度信息初始化),任务一多、单个任务一短,这份开销会变得不可忽视。线程池的解法是反过来:线程只创建一次,反复处理不同任务——这需要一个"任务在哪里排队"的容器和一个"不停去容器里取任务执行"的循环。

容器直接用标准库的 `queue.Queue`,不自己手写。[04 类知识点5](04-advanced-concurrency-patterns.md)已经指出 `queue.Queue` 内部就是用锁和条件变量实现的:`get()` 在队列为空时会阻塞等待,`put()` 会唤醒等待者——这正是[03 类知识点6](03-synchronization-primitives.md)条件变量那一节讲的"worker 线程等待『队列非空』这个条件"在标准库里的真实实现,不需要我们自己重新发明一遍 `Condition` 的 wait/notify。这一篇要手写的是线程池本身(worker 循环、`submit`、`Future`、`shutdown`),不是重新造一个队列。

```python
import queue
import threading

class MiniThreadPoolV1:
    def __init__(self):
        self._task_queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        while True:
            fn, args, kwargs = self._task_queue.get()
            fn(*args, **kwargs)
            self._task_queue.task_done()

    def submit(self, fn, *args, **kwargs):
        self._task_queue.put((fn, args, kwargs))

    def wait_all(self):
        self._task_queue.join()

results = []
results_lock = threading.Lock()

def record(x):
    with results_lock:
        results.append(x * x)

pool = MiniThreadPoolV1()
for i in range(6):
    pool.submit(record, i)
pool.wait_all()

print('results=%s' % sorted(results))
assert sorted(results) == [0, 1, 4, 9, 16, 25]
print("STAGE1_BASIC_TEST=PASS")
```

`_worker` 用 `daemon=True` 创建:守护线程不会阻止 Python 进程退出,如果这是一段真实脚本、主线程跑完了却忘记调用任何关闭逻辑,进程依然能正常退出,不会被这个"永远 `while True` 取任务"的后台线程卡住。`wait_all()` 直接转发到 `queue.Queue.join()`——它会阻塞到"每一个 `put()` 进去的任务都被对应调用过一次 `task_done()`"为止,这是标准库提供的、不需要我们自己写计数器的"等所有任务做完"机制。

**这个版本"看起来"是正确的——直到某个任务自己抛出异常。** 先如实复现这个问题,再决定怎么修:

```python
import queue
import threading
import time

class MiniThreadPoolV1:
    def __init__(self):
        self._task_queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        while True:
            fn, args, kwargs = self._task_queue.get()
            fn(*args, **kwargs)
            self._task_queue.task_done()

    def submit(self, fn, *args, **kwargs):
        self._task_queue.put((fn, args, kwargs))

results = []
results_lock = threading.Lock()

def record(x):
    with results_lock:
        results.append(x)

def boom():
    raise ValueError("simulated task failure")

pool = MiniThreadPoolV1()
pool.submit(record, 'before')
pool.submit(boom)
pool._worker.join(timeout=2)
print('worker_alive_after_crash=%s' % pool._worker.is_alive())
assert not pool._worker.is_alive()

pool.submit(record, 'after')
time.sleep(0.3)
print('results_after_crash=%s' % results)
assert results == ['before']
print("STAGE1_BUG_REPRODUCED=PASS")
```

真实运行这段代码,stderr 会先打出一段完整的 `Traceback`(`ValueError: simulated task failure`),这不是脚本出错——这是 CPython 默认的线程行为:一个线程内部有未捕获的异常,`threading` 模块会把 traceback 打印到 stderr,然后让这个线程**安静地结束**,既不会让主线程崩溃,也不会有任何机制主动通知你"这个线程已经死了"。断言证实了这一点:`worker_alive_after_crash=False`,`_run` 里的 `while True` 循环随着那次未捕获异常永久终止了;之后再 `submit('after')`,任务确实被放进了队列,但再也没有线程去取它——`results_after_crash` 停在 `['before']`,`'after'` 永远不会出现,而且没有任何异常、没有任何提示,程序表现得"一切正常"。这正是本系列反复强调的并发 bug 的典型特征(呼应 [03 类知识点1](03-synchronization-primitives.md)"程序不会报错、只是结果不对"):一个不完善的任务池,会因为业务代码里一次普通的异常,悄悄地永久失去处理能力。

修复很直接:把任务执行包进 `try/except/finally`,异常不再向上传播炸穿 `_run` 的循环,`task_done()` 挪进 `finally` 保证不管任务成功还是失败都会被正确计数:

```python
import queue
import threading

class MiniThreadPoolV1:
    def __init__(self):
        self._task_queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        while True:
            fn, args, kwargs = self._task_queue.get()
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                print('task raised and was swallowed: %r' % (exc,))
            finally:
                self._task_queue.task_done()

    def submit(self, fn, *args, **kwargs):
        self._task_queue.put((fn, args, kwargs))

    def wait_all(self):
        self._task_queue.join()

results = []
results_lock = threading.Lock()

def record(x):
    with results_lock:
        results.append(x)

def boom():
    raise ValueError("simulated task failure")

pool = MiniThreadPoolV1()
pool.submit(record, 'before')
pool.submit(boom)
pool.submit(record, 'after')
pool.wait_all()

print('worker_alive=%s' % pool._worker.is_alive())
print('results=%s' % results)
assert pool._worker.is_alive(), "with the task body wrapped in try/except/finally, a raising task no longer kills the worker thread"
assert results == ['before', 'after'], "the task submitted after the crashing one is still processed normally once the worker survives the exception"
print("STAGE1_FIXED_TEST=PASS")
```

现在 `'after'` 正常出现在结果里,worker 线程在处理完那次异常之后继续存活、继续干活。这里的 `except Exception as exc: print(...)` 只是把异常打印出来、直接"吞掉"——调用 `submit(boom)` 的人完全不知道这个任务失败了、更拿不到失败原因,这是一个明显不够用的权宜之计,阶段 3 引入 `Future` 之后会有真正的解法(把异常带回调用方,而不是在 worker 线程里印一行字就算完)。

---

## 阶段 2:多 worker 版本——现场验证任务确实被并发处理

阶段 1 的池子自始至终只有一个 worker 线程:即使队列里堆了 100 个任务,也是被这一个线程一个接一个地拿去执行——"排队"和"并发处理"是两件不同的事,阶段 1 只做到了前者。把 worker 数量从 1 个变成可配置的多个,才能真正利用多线程。

```python
import queue
import threading
import time

class MiniThreadPoolV2:
    def __init__(self, num_workers):
        self._task_queue = queue.Queue()
        self._workers = [threading.Thread(target=self._run, daemon=True) for _ in range(num_workers)]
        for w in self._workers:
            w.start()

    def _run(self):
        while True:
            fn, args, kwargs = self._task_queue.get()
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                print('task raised and was swallowed: %r' % (exc,))
            finally:
                self._task_queue.task_done()

    def submit(self, fn, *args, **kwargs):
        self._task_queue.put((fn, args, kwargs))

    def wait_all(self):
        self._task_queue.join()

max_concurrent_seen = [0]
current_concurrent = [0]
concurrency_lock = threading.Lock()

def track_concurrency_task():
    with concurrency_lock:
        current_concurrent[0] += 1
        max_concurrent_seen[0] = max(max_concurrent_seen[0], current_concurrent[0])
    time.sleep(0.2)
    with concurrency_lock:
        current_concurrent[0] -= 1

pool = MiniThreadPoolV2(num_workers=4)
for _ in range(12):
    pool.submit(track_concurrency_task)
pool.wait_all()

print('max_concurrent_seen=%d (num_workers=4, 12 tasks)' % max_concurrent_seen[0])
assert max_concurrent_seen[0] == 4, "with 4 worker threads pulling from the same queue, peak concurrency should hit exactly 4, not stay at 1"
print("STAGE2_MAX_CONCURRENCY_TEST=PASS")
```

多个 worker 线程共享同一个 `self._task_queue`,同时调用 `get()` 也是安全的——`queue.Queue` 本来就是为"多个消费者同时取"这个场景设计的并发安全容器([04 类知识点5](04-advanced-concurrency-patterns.md))。实测 `num_workers=4`、一次性提交 12 个"占用 0.2 秒"的任务,`max_concurrent_seen` 稳定停在 4,不多不少——不多,是因为只有 4 个 worker;不少,是因为 12 个任务足够多、4 个 worker 确实能同时都有活干。

只看"同时执行的任务数"还不够直观,再从时间的角度验证一次同样的结论:8 个各睡 0.15 秒的任务,分别交给"1 个 worker"和"8 个 worker"处理,总耗时应该有数量级的差别(这正是[01 类知识点3](01-processes-and-threads.md)讲过的:`time.sleep()` 这类阻塞式系统调用会主动释放 GIL,IO 密集型任务用多线程能拿到真实的并发收益,不像 CPU 密集型任务会被 GIL 卡死)。

```python
import queue
import threading
import time

class MiniThreadPoolV2:
    def __init__(self, num_workers):
        self._task_queue = queue.Queue()
        self._workers = [threading.Thread(target=self._run, daemon=True) for _ in range(num_workers)]
        for w in self._workers:
            w.start()

    def _run(self):
        while True:
            fn, args, kwargs = self._task_queue.get()
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                print('task raised and was swallowed: %r' % (exc,))
            finally:
                self._task_queue.task_done()

    def submit(self, fn, *args, **kwargs):
        self._task_queue.put((fn, args, kwargs))

    def wait_all(self):
        self._task_queue.join()

def sleep_task():
    time.sleep(0.15)

serial_pool = MiniThreadPoolV2(num_workers=1)
t0 = time.perf_counter()
for _ in range(8):
    serial_pool.submit(sleep_task)
serial_pool.wait_all()
serial_elapsed = time.perf_counter() - t0

parallel_pool = MiniThreadPoolV2(num_workers=8)
t0 = time.perf_counter()
for _ in range(8):
    parallel_pool.submit(sleep_task)
parallel_pool.wait_all()
parallel_elapsed = time.perf_counter() - t0

print('serial_elapsed=%.3f parallel_elapsed=%.3f speedup=%.2fx' % (serial_elapsed, parallel_elapsed, serial_elapsed / parallel_elapsed))
assert parallel_elapsed < serial_elapsed / 3, "8 sleep-bound tasks funneled through 1 worker vs spread across 8 workers must show a dramatic wall-clock difference, not just a theoretical one"
print("STAGE2_WALLCLOCK_SPEEDUP_TEST=PASS")
```

验证记录:本机连续实测 3 次,`serial_elapsed` 稳定在 1.203~1.206 秒(8 个任务顺序执行,`8 * 0.15 = 1.2` 秒的理论值几乎精确复现),`parallel_elapsed` 稳定在 0.150~0.151 秒(8 个 worker 同时各自睡 0.15 秒,理论下限就是单个任务的耗时),加速比稳定在 7.98x~8.00x——非常接近 8 个 worker 理论上限的 8 倍,这组数字比大多数真实测量都"干净",原因是任务本身就是纯 `time.sleep()`、没有掺杂任何计算,GIL 在睡眠期间完全释放,8 个线程能几乎无损地同时睡觉。换成 CPU 密集型任务(比如纯 Python 循环做算术),多 worker 不会带来这种量级的加速——[01 类知识点3](01-processes-and-threads.md)已经用实测数据验证过这个对比,这里不重复。

---

## 阶段 3:`submit` 返回一个简易 `Future`——支持稍后取返回值

到目前为止,`submit` 是"发射后不管":任务在后台某个 worker 线程里执行,调用 `submit` 的那一刻,函数的返回值(或者抛出的异常)去哪了?阶段 1 的修复版本只是 `print` 一行然后彻底丢弃——如果任务是"从数据库查一条记录"或者"下载一个文件返回内容",这种丢弃是不可接受的。需要一个凭证,让调用方能在自己需要的时候拿到结果——这正是 `Future` 模式要解决的问题(也是 [04 类知识点3](04-advanced-concurrency-patterns.md)示例代码里 `executor.submit()` 返回值的真实身份:一个 `concurrent.futures.Future` 对象)。

`Future` 内部需要一种"线程 A 等着,线程 B 算完了通知线程 A"的机制——这正是[03 类知识点6](03-synchronization-primitives.md)条件变量要解决的问题。这里不直接用 `threading.Condition`,而是用标准库更简单的 `threading.Event`。`Event` 内部实际上就是包了一层的 `Condition`(`threading.py` 源码里 `Event.__init__` 就是 `self._cond = Condition(Lock())`),`set()`/`wait()` 是对"通知一个条件已经满足"这个模式的简化封装。[03 类知识点6](03-synchronization-primitives.md)强调条件变量必须用 `while` 而不是 `if` 重新检查条件,因为"被唤醒"不等于"条件一定成立"——但这里不需要担心这个问题,不是因为 `Event` 替你做了重新检查,而是因为我们的用法里 `Future` 的状态只会朝一个方向翻转一次(没结果 → 有结果,从不调用 `Event.clear()` 把它翻回去),不存在"醒来时条件又被谁改回去了"这种可能性——条件变量的 `while` 防御针对的是"任意条件、可能反复变化"这种通用场景,`Future` 这里只是这个通用场景里最简单的一种特化。

```python
import queue
import threading

class MiniFuture:
    def __init__(self):
        self._event = threading.Event()
        self._value = None
        self._exception = None

    def set_result(self, value):
        self._value = value
        self._event.set()

    def set_exception(self, exc):
        self._exception = exc
        self._event.set()

    def result(self, timeout=None):
        finished = self._event.wait(timeout)
        if not finished:
            raise TimeoutError("result not ready within %r seconds" % (timeout,))
        if self._exception is not None:
            raise self._exception
        return self._value

class MiniThreadPoolV3:
    def __init__(self, num_workers):
        self._task_queue = queue.Queue()
        self._workers = [threading.Thread(target=self._run, daemon=True) for _ in range(num_workers)]
        for w in self._workers:
            w.start()

    def _run(self):
        while True:
            fn, args, kwargs, future = self._task_queue.get()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                self._task_queue.task_done()

    def submit(self, fn, *args, **kwargs):
        future = MiniFuture()
        self._task_queue.put((fn, args, kwargs, future))
        return future

pool = MiniThreadPoolV3(num_workers=3)
f1 = pool.submit(lambda a, b: a + b, 3, 4)
f2 = pool.submit(lambda s: s.upper(), 'hello')
f3 = pool.submit(pow, 2, 10)

print('f1=%r f2=%r f3=%r' % (f1.result(timeout=2), f2.result(timeout=2), f3.result(timeout=2)))
assert f1.result(timeout=2) == 7
assert f2.result(timeout=2) == 'HELLO'
assert f3.result(timeout=2) == 1024
print("STAGE3_RESULT_TEST=PASS")
```

`submit` 现在把 `(fn, args, kwargs, future)` 一起放进队列,worker 执行完之后把结果(或异常)写回这个 `future` 对象,调用方拿着 `submit` 返回的 `Future`、在自己方便的时候调用 `.result()` 取值——这中间可以隔着任意多行其他代码,不要求调用方在 `submit` 那一刻就守在原地等结果,这是"发射后不管"和"发射后拿一个凭证、随时可以来取货"最关键的区别。

异常也要能正确穿越线程边界。任务在 worker 线程里抛出的异常,不会自动出现在调用 `submit` 的那个线程里(这是两个完全独立的 Python 调用栈)——`MiniFuture.set_exception` 把异常对象保存下来,`.result()` 在调用方所在的线程里重新 `raise` 它,验证这一点确实按预期工作:

```python
import queue
import threading

class MiniFuture:
    def __init__(self):
        self._event = threading.Event()
        self._value = None
        self._exception = None

    def set_result(self, value):
        self._value = value
        self._event.set()

    def set_exception(self, exc):
        self._exception = exc
        self._event.set()

    def result(self, timeout=None):
        finished = self._event.wait(timeout)
        if not finished:
            raise TimeoutError("result not ready within %r seconds" % (timeout,))
        if self._exception is not None:
            raise self._exception
        return self._value

class MiniThreadPoolV3:
    def __init__(self, num_workers):
        self._task_queue = queue.Queue()
        self._workers = [threading.Thread(target=self._run, daemon=True) for _ in range(num_workers)]
        for w in self._workers:
            w.start()

    def _run(self):
        while True:
            fn, args, kwargs, future = self._task_queue.get()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                self._task_queue.task_done()

    def submit(self, fn, *args, **kwargs):
        future = MiniFuture()
        self._task_queue.put((fn, args, kwargs, future))
        return future

pool = MiniThreadPoolV3(num_workers=2)

def boom():
    raise ValueError('stage3 boom')

f_err = pool.submit(boom)
try:
    f_err.result(timeout=2)
    raised = False
    err_msg = None
except ValueError as e:
    raised = True
    err_msg = str(e)

print('raised=%s err_msg=%s' % (raised, err_msg))
assert raised and err_msg == 'stage3 boom', "the exception raised inside the worker thread must surface on the CALLING thread when .result() is invoked, not vanish"
print("STAGE3_EXCEPTION_PROPAGATION_TEST=PASS")
```

比起阶段 1 那个"打印一行就丢弃"的权宜之计,这才是异常该有的归宿:谁提交的任务,谁负责处理这个任务失败的后果,而不是让错误消失在某个后台线程里。

最后验证 `timeout` 参数本身。这里涉及"任务还没执行完,`.result()` 应该在多久后放弃等待"这种精确时机的断言——不依赖 `sleep` 时长赌运气(万一系统这一刻负载高、调度慢了几十毫秒,基于 `sleep` 猜时间点的断言就会变得不稳定),而是用 [03 类](03-synchronization-primitives.md)已经用过的手法:`threading.Event` 把 worker 精确卡在一个我们完全掌控的点上。

```python
import queue
import threading

class MiniFuture:
    def __init__(self):
        self._event = threading.Event()
        self._value = None
        self._exception = None

    def set_result(self, value):
        self._value = value
        self._event.set()

    def set_exception(self, exc):
        self._exception = exc
        self._event.set()

    def result(self, timeout=None):
        finished = self._event.wait(timeout)
        if not finished:
            raise TimeoutError("result not ready within %r seconds" % (timeout,))
        if self._exception is not None:
            raise self._exception
        return self._value

class MiniThreadPoolV3:
    def __init__(self, num_workers):
        self._task_queue = queue.Queue()
        self._workers = [threading.Thread(target=self._run, daemon=True) for _ in range(num_workers)]
        for w in self._workers:
            w.start()

    def _run(self):
        while True:
            fn, args, kwargs, future = self._task_queue.get()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                self._task_queue.task_done()

    def submit(self, fn, *args, **kwargs):
        future = MiniFuture()
        self._task_queue.put((fn, args, kwargs, future))
        return future

# 用threading.Event精确卡住唯一的worker，不依赖sleep时长赌时机
hold = threading.Event()

def blocking_task():
    hold.wait()
    return 'released'

single_pool = MiniThreadPoolV3(num_workers=1)
f_block = single_pool.submit(blocking_task)   # occupies the only worker, deterministically, until hold.set()
f_next = single_pool.submit(lambda: 'quick')  # queued behind it; cannot possibly be ready yet

timed_out = False
try:
    f_next.result(timeout=0.3)
except TimeoutError:
    timed_out = True
print('timed_out=%s' % timed_out)
assert timed_out, "f_next's task cannot have run yet because the single worker is deterministically still stuck in blocking_task"

hold.set()  # release the held task now
assert f_block.result(timeout=2) == 'released'
assert f_next.result(timeout=2) == 'quick'
print("STAGE3_TIMEOUT_TEST=PASS")
```

`single_pool` 只有 1 个 worker,`f_block` 对应的任务卡在 `hold.wait()` 上不会自己返回——这不是"大概率不会这么快",是**确定性**地不会返回,因为除了这段代码主动调用 `hold.set()`,没有任何东西能让它继续。`f_next` 因此百分之百排在队列里没人处理,`f_next.result(timeout=0.3)` 必定超时,这个 `assert` 不依赖任何运气。等主动 `hold.set()` 放行之后,两个 `Future` 才先后拿到各自正确的结果。

---

## 阶段 4:`shutdown(wait=True/False)`——优雅关闭

前三个阶段的 worker 线程都是 `while True` 死循环,一旦启动就永远不退出。真实程序里,线程池往往对应一段明确的生命周期(比如一个请求处理服务关闭时,应该先停止接收新请求、把手头正在处理的请求做完,再真正退出),需要一种"关闭"语义,而且这个语义要同时回答两个问题:**还没处理的任务怎么办**(不能直接扔掉)和**新任务还能不能进来**(不能再进来)。

沿用[03 类知识点4](03-synchronization-primitives.md)生产者消费者例子里已经验证过的哨兵(sentinel)思路:再往队列里放几个特殊的"停止标记",worker 一旦取到这个标记就退出循环,不再继续 `get()`。因为 `queue.Queue` 是严格先进先出的([04 类知识点5](04-advanced-concurrency-patterns.md)),只要保证"先放完所有真实任务,再放停止标记",停止标记必然排在所有真实任务后面被取到——每个 worker 在拿到属于自己的那个停止标记之前,一定已经完整处理完了它此前拿到的所有真实任务(因为单个 worker 的循环是严格顺序的:处理完当前项、调用 `task_done()`,才会去 `get()` 下一项)。停止标记的数量精确等于 worker 数量,每个 worker 退出前只会消费恰好一个停止标记,不会有 worker 抢到两个、导致另一个 worker 永远等不到自己的那一个。

`submit` 和 `shutdown` 之间还有一个不能忽视的竞态窗口:如果 `submit` 检查"有没有关闭"和真正 `put` 任务这两步中间,`shutdown` 恰好插进来完成了关闭并放入了停止标记,这个任务就会被放到停止标记**后面**,永远没有 worker 会处理它——这正是[03 类知识点2](03-synchronization-primitives.md)"决策依据追问轴"提到的"检查-然后-操作"被拆开引入新竞态窗口的真实例子。用一把锁把"检查+放任务"和"标记关闭+放停止标记"分别包成整体、互斥执行,消除这个窗口。

```python
import queue
import threading
import time

class MiniFuture:
    def __init__(self):
        self._event = threading.Event()
        self._value = None
        self._exception = None

    def set_result(self, value):
        self._value = value
        self._event.set()

    def set_exception(self, exc):
        self._exception = exc
        self._event.set()

    def result(self, timeout=None):
        finished = self._event.wait(timeout)
        if not finished:
            raise TimeoutError("result not ready within %r seconds" % (timeout,))
        if self._exception is not None:
            raise self._exception
        return self._value

class MiniThreadPool:
    def __init__(self, num_workers):
        self._task_queue = queue.Queue()
        self._num_workers = num_workers
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._SENTINEL = object()
        self._workers = [threading.Thread(target=self._run, daemon=True) for _ in range(num_workers)]
        for w in self._workers:
            w.start()

    def _run(self):
        while True:
            item = self._task_queue.get()
            if item is self._SENTINEL:
                self._task_queue.task_done()
                break
            fn, args, kwargs, future = item
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                self._task_queue.task_done()

    def submit(self, fn, *args, **kwargs):
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("cannot submit to a pool that has already been shut down")
            future = MiniFuture()
            self._task_queue.put((fn, args, kwargs, future))
            return future

    def shutdown(self, wait=True):
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
            for _ in range(self._num_workers):
                self._task_queue.put(self._SENTINEL)
        if wait:
            for w in self._workers:
                w.join()

results = []
results_lock = threading.Lock()

def slow_record(x, dt):
    time.sleep(dt)
    with results_lock:
        results.append(x)

pool = MiniThreadPool(num_workers=3)
durations = [0.05, 0.3, 0.05, 0.05, 0.3, 0.05, 0.05, 0.05, 0.3]
for i, dt in enumerate(durations):
    pool.submit(slow_record, i, dt)

t0 = time.perf_counter()
pool.shutdown(wait=True)
shutdown_elapsed = time.perf_counter() - t0

print('shutdown_elapsed=%.3f results=%s' % (shutdown_elapsed, sorted(results)))
assert sorted(results) == list(range(len(durations))), "shutdown(wait=True) must not return until every already-queued task has fully finished"
assert shutdown_elapsed >= 0.28, "shutdown(wait=True) should really block for as long as the slowest in-flight task takes, not return early"
assert all(not w.is_alive() for w in pool._workers), "after shutdown(wait=True) returns, every worker thread must have actually exited"

try:
    pool.submit(slow_record, 'late', 0.0)
    rejected = False
except RuntimeError:
    rejected = True
print('rejected=%s' % rejected)
assert rejected, "submitting after shutdown must be rejected, not silently queued forever"
print("STAGE4_WAIT_TRUE_TEST=PASS")
```

验证记录:连续实测 5 次,`shutdown_elapsed` 稳定在 0.552~0.553 秒——这个数字本身也值得读一读:9 个任务(3 个耗时 0.3 秒、6 个耗时 0.05 秒,总工作量 `3*0.3 + 6*0.05 = 1.2` 秒)分给 3 个 worker,如果分配完全均匀,理论最优是 `1.2 / 3 = 0.4` 秒;实测 0.552 秒明显高于这个理论最优、又明显低于完全串行的 1.2 秒——因为任务在 3 个 worker 之间具体怎么分配,取决于运行时哪个 worker 恰好空闲下来去 `get()` 下一个任务,3 个耗时 0.3 秒的"重任务"不一定被均匀分到 3 个不同的 worker 上,真实调度带来的不均衡是本来就该出现的正常现象,不是 bug。`shutdown(wait=True)` 确实等到了最后一个任务完成才返回(而不是队列一空就提前返回),`submit` 在关闭之后也确实被拒绝。

再验证 `wait=False`:这次要证明的是"函数几乎立刻返回",不等任何任务做完,但已经排队的任务不会被放弃、依然会在后台执行完:

```python
import queue
import threading
import time

class MiniFuture:
    def __init__(self):
        self._event = threading.Event()
        self._value = None
        self._exception = None

    def set_result(self, value):
        self._value = value
        self._event.set()

    def set_exception(self, exc):
        self._exception = exc
        self._event.set()

    def result(self, timeout=None):
        finished = self._event.wait(timeout)
        if not finished:
            raise TimeoutError("result not ready within %r seconds" % (timeout,))
        if self._exception is not None:
            raise self._exception
        return self._value

class MiniThreadPool:
    def __init__(self, num_workers):
        self._task_queue = queue.Queue()
        self._num_workers = num_workers
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._SENTINEL = object()
        self._workers = [threading.Thread(target=self._run, daemon=True) for _ in range(num_workers)]
        for w in self._workers:
            w.start()

    def _run(self):
        while True:
            item = self._task_queue.get()
            if item is self._SENTINEL:
                self._task_queue.task_done()
                break
            fn, args, kwargs, future = item
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                self._task_queue.task_done()

    def submit(self, fn, *args, **kwargs):
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("cannot submit to a pool that has already been shut down")
            future = MiniFuture()
            self._task_queue.put((fn, args, kwargs, future))
            return future

    def shutdown(self, wait=True):
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
            for _ in range(self._num_workers):
                self._task_queue.put(self._SENTINEL)
        if wait:
            for w in self._workers:
                w.join()

results = []
results_lock = threading.Lock()

def slow_record(x, dt):
    time.sleep(dt)
    with results_lock:
        results.append(x)

pool = MiniThreadPool(num_workers=2)
for i in range(4):
    pool.submit(slow_record, i, 0.2)

t0 = time.perf_counter()
pool.shutdown(wait=False)
nonblocking_elapsed = time.perf_counter() - t0
print('nonblocking_elapsed=%.4f' % nonblocking_elapsed)
assert nonblocking_elapsed < 0.05, "shutdown(wait=False) must return near-instantly without waiting for queued work to drain"

# bounded wait for background completion (never an indefinite hang in this demo)
for w in pool._workers:
    w.join(timeout=3)
assert all(not w.is_alive() for w in pool._workers)
print('results=%s' % sorted(results))
assert sorted(results) == [0, 1, 2, 3], "even with wait=False, the queued tasks still all eventually run to completion in the background"
print("STAGE4_WAIT_FALSE_TEST=PASS")
```

验证记录:`nonblocking_elapsed` 实测稳定为 `0.0000`(四舍五入到小数点后 4 位,真实值在微秒级)——`shutdown(wait=False)` 只做了"标记关闭 + 放入停止标记"这一件事就返回,完全不等任何一个还在跑的任务;随后用 `w.join(timeout=3)`(限时等待,不是无限期等待)确认所有 worker 最终都会退出、4 个任务的结果全部正确到齐,证明"不阻塞调用方"和"任务不会被放弃"这两件事同时成立,互不矛盾。

**这里也解释一下代码里两种"等待"的区别,不要混用**:阶段 1~3 的 `wait_all()` 等的是 `queue.Queue.join()`——"队列里的任务都被 `task_done()` 过了";`shutdown(wait=True)` 等的是 `Thread.join()`——"worker 线程本身已经退出"。两者不完全等价:`task_done()` 计数够了,并不代表 worker 线程已经跑完 `_run` 方法里那次 `get()` 循环剩下的收尾代码、真正从操作系统层面终止;`shutdown` 要保证的是"线程真的不再运行了"(这样才能放心认为"这一批 worker 占用的资源已经完全释放"),所以选用更精确的 `Thread.join()`,而不是复用队列自己的 `join()`。也正因为 `shutdown` 提供了这样一种更强的等待语义,最终版本的 `MiniThreadPool` 不再保留 `wait_all()`(如果想保留完全可以,两者并不冲突,只是这里为了让类的对外接口保持聚焦,不同时暴露两套含义相近但不完全相同的"等待"方法)。

---

## 可以怎么继续扩展(只指方向,不在本文实现)

- **有界队列 + 拒绝策略**:现在的 `queue.Queue()` 不设 `maxsize`,是一个无界队列——[04 类知识点3](04-advanced-concurrency-patterns.md)已经指出过,无界队列在任务提交速度长期快于处理速度时,排队对象会持续占用内存直到耗尽。换成 `queue.Queue(maxsize=N)`,`put()` 在队列满时会阻塞,再配合"满了就拒绝/丢弃/调用方自己执行"这几种拒绝策略(04 类知识点3 讨论过的决策空间),是贴近生产可用性的下一步,这里没有实现。
- **核心线程数与最大线程数分离**:现在的 worker 数量在构造时一次性固定。真实的线程池(包括 Java 的 `ThreadPoolExecutor`)通常区分"常驻的核心线程数"和"负载高时能临时扩展到的最大线程数",空闲的临时线程超过一定时间会被回收——这需要在 `_run` 的 `get()` 上加超时,配合动态增减 `self._workers` 列表,不在本文实现。
- **更通用的 `Future`**:`MiniFuture` 只支持"提交时创建、执行完写入结果"这一种最简单的形态,不支持 `cancel()`(任务还没被 worker 取走时撤销它)、不支持多个线程同时对同一个 `Future` 调用 `.result()`(当前实现其实已经天然支持这一点,因为 `Event.wait()` 允许多个线程同时等待同一个 `Event`,但没有专门验证过)。真正的 `concurrent.futures.Future` 内部用的是 `Condition` 而不是 `Event`,能表达比"单向翻转一次"更丰富的状态机(pending/running/cancelled/finished),这是 [03 类知识点6](03-synchronization-primitives.md)条件变量比 `Event` 更通用的地方。
- **优先级任务**:把 `queue.Queue` 换成标准库的 `queue.PriorityQueue`,让紧急任务插队到普通任务前面执行——需要给每个任务定义一个可比较的优先级,这里不实现。

## 这篇教程展示的方法论

任何一条已经完成的深挖系列,都可以用同样的模式产出"教程体"内容:挑几个关联的知识点 → 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实效果,而不是一次性甩出完整代码。这是 dsa-deep-dive/21 试点过的格式第一次被推广到另一条系列,验证了它在"并发/系统"这类主题下同样成立——但主题不同,阶段的"形状"也不必强行套用同一个模板:dsa-deep-dive/21 的三个阶段(倒排索引、Trie、堆排序)是三个相对独立的组件,第四阶段才把它们组装成一个类;这一篇的四个阶段,从始至终都是同一个类在不断长出新能力(单 worker → 多 worker → 能拿到返回值 → 能优雅关闭),没有一个单独的"组装"阶段——因为线程池本来就是一个不可拆分的整体机制,不是几个独立部件拼出来的,阶段该怎么分,由要造的东西本身的结构决定,不是由格式本身决定。

---

*创建:2026-07-24*
