# L25 · 并发（可选层）—— 用同步原语控制多线程按指定顺序交替执行

> 对应代码：`src/problems/p25_concurrency.py`。
>
> **这一类不是算法类别，是"并发协调原语"的应用**：用 Lock/Semaphore/Condition
> 控制多个线程按指定顺序交替执行，是后端方向岗位偶尔会考的独立题型。如果你的
> 目标是研究岗/纯算法岗，这一类可以跳过不学，标记为可选；如果目标包含后端/系统
> 方向，这三道题是标准入门。

## 这类题在考什么

普通算法题关心的是"用什么数据结构/算法能把问题在时间和空间上解得更快"，这一类题
关心的是完全不同的维度：**多个线程并发执行时，怎么保证它们按题目要求的顺序交替
执行，而不是靠运气**。核心工具是 `threading` 模块提供的几种同步原语——`Lock`（
互斥锁）、`Event`（一次性信号）、`Semaphore`（带计数的信号量）、`Condition`（
条件变量）。这几道题的共同陷阱是：**用 `time.sleep()` 去"凑"顺序**——比如让某个
线程睡 10 毫秒指望另一个线程先跑完。这种做法在测试机器空闲时可能凑巧通过，但
本质上是不确定的，机器负载高、CI 环境慢一点就可能失败。正确的解法必须用同步原语
从根本上保证顺序，不依赖任何时间上的巧合。

## 深挖：交替打印 FooBar（LC1115，Medium）—— 两个初始值不同的信号量如何实现"接力棒"

**先想清楚"交替"这个约束的本质**：题目要求两个线程各自循环 n 次，输出效果必须
是 `foo bar foo bar ...` 严格交替，不能连续出现两个 `foo` 或两个 `bar`。这意味着
"同一时刻，只有一个线程被允许往下执行"，而且这个"许可"要在 foo 和 bar 之间来回
传递。

**为什么两个信号量、且初始值不同**：用两个 `threading.Semaphore`：

```python
self.foo_semaphore = threading.Semaphore(1)   # foo 一开始就可以执行
self.bar_semaphore = threading.Semaphore(0)   # bar 一开始不能执行，必须等 foo
```

`foo_semaphore` 初始值是 1，表示"foo 可以立刻执行第一次"；`bar_semaphore` 初始值
是 0，表示"bar 现在没有许可，必须等待"。foo() 每一轮先 `acquire` 自己的信号量
（第一轮凭初始值 1 直接拿到），执行 `print_foo()`，然后 `release` bar_semaphore
——这一步就是把"接力棒"正式交给 bar；bar() 对称地先 `acquire` bar_semaphore（初始
是 0，所以第一次必然阻塞，直到 foo 完成一轮并 release 才被唤醒），执行
`print_bar()` 后 `release` foo_semaphore，把接力棒还给 foo：

```python
def foo(self, print_foo) -> None:
    for _ in range(self.n):
        self.foo_semaphore.acquire()
        print_foo()
        self.bar_semaphore.release()

def bar(self, print_bar) -> None:
    for _ in range(self.n):
        self.bar_semaphore.acquire()
        print_bar()
        self.foo_semaphore.release()
```

**为什么这样必然严格交替**：任意时刻，两个信号量里最多只有一个的"可用许可数"是
非零的——foo 拿到许可执行完之后，立刻把许可转移给 bar（release bar_semaphore
的同时并没有重新 release 自己的 foo_semaphore），所以 foo 线程下一轮再来
`acquire` 时会阻塞，直到 bar 执行完、把许可还回来。这个"许可只会存在于其中一方"
的性质，正是严格交替的根本保证——不是"大概率交替"，而是**只要代码逻辑不变，
无论操作系统怎么调度这两个线程，交替顺序都不可能被打破**。

**容易出错的地方**：如果把两个信号量都初始化成 1，两个线程第一轮会同时抢到各自
的许可，谁先真正执行完全取决于操作系统调度，可能出现连续两次 foo 或连续两次
bar——这正是"两个初始值必须不同"这个设计的意义所在；另外 release 的时机必须
紧跟在"这一轮该做的事情做完之后"，如果提前 release，会导致对方在这一轮真正执行
完之前就被唤醒，破坏交替顺序。

## 其余 2 题点睛

- **LC1114 按序打印**：三个方法 first/second/third 分别被三个线程调用，调用/
  启动的顺序不受控制，但执行效果必须是 first 先于 second、second 先于 third。
  用两个 `threading.Event` 做"接力棒"：`first_done`、`second_done`。second()
  一进来就 `first_done.wait()` 阻塞，直到 first() 执行完设置了这个 Event 才被
  放行；third() 同理等待 `second_done`。即使三个线程的 `start()` 顺序完全打乱，
  没到自己该执行的时机，线程会阻塞在 `.wait()` 上，而不是靠猜测或者睡眠来保证
  顺序。
- **LC1195 交替打印字符串（FizzBuzz）**：四个线程分别负责"3 的倍数非 5 的倍数"、
  "5 的倍数非 3 的倍数"、"15 的倍数"、"其余数字"，"轮到谁"取决于当前数字的数学
  性质，不适合用固定的几个信号量互相传递（那是"两个角色接力"的场景），更自然的
  是用一个共享的 `threading.Condition`：维护共享变量 `current`，四个线程各自
  循环检查"当前数字是不是该由我负责"，不是就 `condition.wait()`，是就在持有锁
  的状态下打印、把 `current` 加一、`notify_all()` 唤醒所有等待者重新检查。关键
  细节是判断条件必须写成 `while` 而不是 `if`——`notify_all()` 会唤醒所有等待
  线程，但只有一个线程的条件真正满足，其余线程被唤醒后必须重新检查条件、发现
  还没轮到自己就继续等待，这是条件变量的标准使用范式。
