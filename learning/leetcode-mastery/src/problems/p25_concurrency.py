"""分类 25（可选层）：并发协调原语 —— Lock/Semaphore/Condition 控制多线程按指定顺序交替执行。

这类题不是算法类别，考的是"并发协调原语"的应用，是后端方向岗位偶尔会考的独立题型。
如果你的目标是研究岗/纯算法岗，这一类可以跳过不学，标记为可选；如果目标包含后端/
系统方向，这三道题是"用 Lock/Semaphore/Condition 实现确定性线程执行顺序"的标准入门。
本文件全部用 Python 标准库 `threading`，不依赖 sleep 计时凑合同步——所有正确性都
由同步原语本身保证，因此测试结果是确定性的，不会因为线程调度不同而偶尔失败。
"""
from __future__ import annotations

import threading


class Foo:
    """
    【题意】设计一个类，三个方法 first(printFirst)、second(printSecond)、
    third(printThird) 会被三个不同的线程分别调用一次，调用顺序（也就是线程启动的
    先后）不受控制，但要求无论线程以什么顺序启动，实际执行效果必须保证
    printFirst() 先于 printSecond()、printSecond() 先于 printThird()。
    【思路】用两个 `threading.Event` 作为"接力棒"：`first_done` 表示"first 已经
    执行完"，`second_done` 表示"second 已经执行完"。first() 不需要等任何人，直接
    执行 printFirst()，执行完把 first_done 设置为已触发；second() 一进来就先
    `first_done.wait()` 阻塞住，直到 first() 执行完才会被放行，放行后执行
    printSecond()，再设置 second_done；third() 同理先等 second_done。这样即使
    三个线程的 `start()` 调用顺序完全打乱，谁先抢到 CPU 都不影响最终效果——没到
    自己该执行的时机，线程会阻塞在 `.wait()` 上，而不是靠"猜"或者"睡一会儿"来
    保证顺序。
    【复杂度】时间/空间都是 O(1)（只有两个 Event 对象，和线程数、调用次数无关）。
    【易错点】1) 如果用一个普通的布尔变量 + 忙轮询（`while not done: pass`）来
    模拟"等待"，虽然功能上也可能凑效，但会造成 CPU 空转，而且没有内存可见性保证，
    正确的做法是用 `threading.Event` 这种阻塞式原语；2) 不能用 `time.sleep` 来
    "祈祷" first 一定比 second 先跑完——sleep 只是缩小了出错的概率，不是消除，
    在测试机器负载高的时候依然可能失败，必须用真正的同步原语（Event/Semaphore/
    Condition/Lock）从根本上保证顺序。
    """

    def __init__(self) -> None:
        self.first_done = threading.Event()
        self.second_done = threading.Event()

    def first(self, printFirst) -> None:
        printFirst()
        self.first_done.set()

    def second(self, printSecond) -> None:
        self.first_done.wait()
        printSecond()
        self.second_done.set()

    def third(self, printThird) -> None:
        self.second_done.wait()
        printThird()


class FooBar:
    """
    【题意】设计一个类 FooBar(n)，两个方法 foo(print_foo)、bar(print_bar) 分别被
    两个线程各自循环调用 n 次，要求最终打印效果严格交替：foo bar foo bar ... 一共
    n 对，foo 永远先于紧跟着的那个 bar。
    【思路】用两个初始值不同的 `threading.Semaphore`：`foo_semaphore` 初始值为 1
    （表示"foo 可以立刻执行一次"），`bar_semaphore` 初始值为 0（表示"bar 现在还不能
    执行，得等 foo 先跑一次"）。foo() 每一轮先 `acquire` 自己的信号量（第一轮能立刻
    拿到，因为初始值是 1；后续轮次能不能拿到取决于上一轮 bar 有没有 release 回来），
    拿到之后执行 print_foo()，执行完立刻 `release` bar_semaphore——这一步是把"接力棒"
    正式交给 bar；bar() 同理，先 acquire 自己的信号量（初始是 0，所以第一次一定会
    阻塞，直到 foo 完成第一轮并 release 了 bar_semaphore 才能被唤醒），执行完
    print_bar() 后 release foo_semaphore，把接力棒还给 foo。两个信号量一来一回，
    就像接力赛跑的接力棒——同一时刻永远只有一个信号量的值是"1"（可被拿到），保证了
    严格交替，不会出现连续两次 foo 或连续两次 bar。
    【复杂度】时间/空间都是 O(1) 的额外开销（只有两个信号量，和 n 无关）。
    【易错点】1) 两个信号量的初始值必须不同（一个 1 一个 0），如果都初始化成 1，
    两个线程第一轮会同时抢到各自的信号量，谁先执行完全不确定，可能出现 foo foo 或
    bar bar 连续执行；2) release 的时机必须紧跟在"这一轮该做的事情做完之后"，
    如果提前 release（比如在 print_foo() 之前就 release 了 bar_semaphore），bar
    可能在 foo 真正打印之前就被唤醒，交替顺序被破坏。
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.foo_semaphore = threading.Semaphore(1)
        self.bar_semaphore = threading.Semaphore(0)

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


class FizzBuzz:
    """
    【题意】设计一个类 FizzBuzz(n)，四个方法 fizz(printFizz)（负责是 3 的倍数但不是
    5 的倍数的数字）、buzz(printBuzz)（5 的倍数但不是 3 的倍数）、
    fizzbuzz(printFizzBuzz)（15 的倍数）、number(printNumber)（其余普通数字，回调
    接收数字本身作为参数）分别由四个线程并发调用，要求最终按 1 到 n 顺序、每个数字
    交给对应负责的那个方法打印，效果等价于单线程版本的标准 FizzBuzz 输出序列。
    【思路】和 Foo/FooBar 用"专属信号量做接力棒"不同，这里有 4 个角色、且"轮到谁"
    取决于当前数字自身的数学性质（能不能被 3/5 整除），不适合用固定的几个信号量
    互相传递，更自然的写法是用一个共享的 `threading.Condition`：维护一个共享变量
    `current`（下一个该被处理的数字），四个线程各自跑一个循环，每一轮先
    `acquire` 这个 Condition 关联的锁，判断"当前数字是不是该由我负责的类型"——不是
    就调用 `condition.wait()` 释放锁并阻塞，被其他线程 notify 后醒来重新检查；是的话
    就在持有锁的状态下执行对应的打印回调、把 `current` 加一、然后 `notify_all()`
    唤醒所有等待者重新检查条件（因为 current 变了，可能轮到别的线程了）。整个"检查
    条件 -> 打印 -> 递增 -> 通知"都在同一次持锁期间完成，天然保证了打印顺序和
    `current` 递增顺序的一致性，不会有两个线程同时认为"轮到自己"。
    【复杂度】时间 O(n)（总共处理 n 个数字，每个数字被唤醒常数次）；空间 O(1)。
    【易错点】1) 用 `while` 而不是 `if` 检查"轮到自己了吗"——`notify_all()` 会
    唤醒所有等待线程，但只有一个线程的条件真正满足，其余线程被唤醒后必须重新检查
    条件、发现还没轮到自己就继续 wait，用 `if` 只检查一次会导致条件其实不满足也
    继续往下执行（这是条件变量的标准使用范式："wait 必须包在 while 循环里"）；
    2) 四个方法判断"是否轮到自己"的条件必须互斥又覆盖所有情况（3 的倍数且非 5 的
    倍数 / 5 的倍数且非 3 的倍数 / 15 的倍数 / 都不是），少判断或者判断重叠会导致
    死锁（没人愿意处理某个数字，所有线程永远 wait）或者重复打印。
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.current = 1
        self.condition = threading.Condition()

    def fizz(self, printFizz) -> None:
        while True:
            with self.condition:
                while self.current <= self.n and not (
                    self.current % 3 == 0 and self.current % 5 != 0
                ):
                    self.condition.wait()
                if self.current > self.n:
                    return
                printFizz()
                self.current += 1
                self.condition.notify_all()

    def buzz(self, printBuzz) -> None:
        while True:
            with self.condition:
                while self.current <= self.n and not (
                    self.current % 5 == 0 and self.current % 3 != 0
                ):
                    self.condition.wait()
                if self.current > self.n:
                    return
                printBuzz()
                self.current += 1
                self.condition.notify_all()

    def fizzbuzz(self, printFizzBuzz) -> None:
        while True:
            with self.condition:
                while self.current <= self.n and self.current % 15 != 0:
                    self.condition.wait()
                if self.current > self.n:
                    return
                printFizzBuzz()
                self.current += 1
                self.condition.notify_all()

    def number(self, printNumber) -> None:
        while True:
            with self.condition:
                while self.current <= self.n and (
                    self.current % 3 == 0 or self.current % 5 == 0
                ):
                    self.condition.wait()
                if self.current > self.n:
                    return
                printNumber(self.current)
                self.current += 1
                self.condition.notify_all()


def _test_foo() -> None:
    foo = Foo()
    order: list[int] = []
    lock = threading.Lock()

    def record(x: int):
        with lock:
            order.append(x)

    # 故意乱序启动线程，验证同步原语能纠正实际执行顺序，而不是靠启动顺序侥幸对拍
    t3 = threading.Thread(target=foo.third, args=(lambda: record(3),))
    t2 = threading.Thread(target=foo.second, args=(lambda: record(2),))
    t1 = threading.Thread(target=foo.first, args=(lambda: record(1),))
    t3.start()
    t2.start()
    t1.start()
    t3.join()
    t2.join()
    t1.join()
    assert order == [1, 2, 3]


def _test_foobar() -> None:
    fb = FooBar(2)
    output: list[str] = []
    lock = threading.Lock()

    def print_foo():
        with lock:
            output.append("foo")

    def print_bar():
        with lock:
            output.append("bar")

    t_bar = threading.Thread(target=fb.bar, args=(print_bar,))
    t_foo = threading.Thread(target=fb.foo, args=(print_foo,))
    t_bar.start()
    t_foo.start()
    t_bar.join()
    t_foo.join()
    assert output == ["foo", "bar", "foo", "bar"]


def _test_fizzbuzz() -> None:
    fb = FizzBuzz(15)
    output: list[str] = []
    lock = threading.Lock()

    def print_fizz():
        with lock:
            output.append("fizz")

    def print_buzz():
        with lock:
            output.append("buzz")

    def print_fizzbuzz():
        with lock:
            output.append("fizzbuzz")

    def print_number(x: int):
        with lock:
            output.append(str(x))

    threads = [
        threading.Thread(target=fb.number, args=(print_number,)),
        threading.Thread(target=fb.fizzbuzz, args=(print_fizzbuzz,)),
        threading.Thread(target=fb.fizz, args=(print_fizz,)),
        threading.Thread(target=fb.buzz, args=(print_buzz,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = [
        "1", "2", "fizz", "4", "buzz", "fizz", "7", "8", "fizz", "buzz",
        "11", "fizz", "13", "14", "fizzbuzz",
    ]
    assert output == expected


def _self_test() -> None:
    # 并发正确性由 Event/Semaphore/Condition 保证是确定性的，重复跑几轮进一步
    # 确认没有偶发的时序漏洞（不是靠 sleep 凑合，而是重复验证同步原语本身的可靠性）。
    for _ in range(20):
        _test_foo()
        _test_foobar()
        _test_fizzbuzz()

    print(
        "[PASS] p25_concurrency: 3/3 题通过（20 轮重复验证无时序漏洞） "
        "(按序打印/交替打印FooBar/交替打印字符串FizzBuzz)"
    )


if __name__ == "__main__":
    _self_test()
