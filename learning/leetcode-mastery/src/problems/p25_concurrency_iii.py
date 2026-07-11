"""分类 25（可选层，Phase 3 竞赛级补充）：并发进阶 —— 三线程协调打印、经典的
"哲学家进餐"死锁避免问题、按比例分组的屏障同步。和基础篇一样全部用 Python
标准库 `threading` 的真同步原语（Semaphore/Lock/Barrier）实现，不依赖 sleep
计时凑合，正确性由同步原语本身保证，测试结果是确定性的。"""
from __future__ import annotations

import threading


class ZeroEvenOdd:
    """
    【题意】设计一个类 ZeroEvenOdd(n)，三个方法 zero(printNumber)、
    even(printNumber)、odd(printNumber) 分别被三个不同的线程调用：线程 A 调用
    zero()，只应该输出 n 次 0；线程 B 调用 even()，只应该按顺序输出偶数；线程 C
    调用 odd()，只应该按顺序输出奇数。三个线程配合输出的最终序列必须是
    "010203...0n"（长度为 2n）。
    【思路】和"交替打印 FooBar"是同一类"接力棒"模式，只是这里有三个角色、且
    "该轮到谁"取决于当前数字的奇偶性，不是简单的两两交替。用三个初始值不同的
    `threading.Semaphore`：`zero_semaphore` 初始值为 1（zero 第一轮可以立刻执行），
    `even_semaphore` 和 `odd_semaphore` 初始值都是 0（谁都不能抢在 zero 前面执行）。
    zero() 循环 n 次，每轮先 `acquire` 自己的信号量，打印 0，然后根据"这是第几轮"
    的奇偶性，把接力棒 release 给 odd_semaphore（奇数轮）或者 even_semaphore
    （偶数轮）；odd()/even() 各自只需要 acquire 自己的信号量、打印对应的数字、
    再把接力棒 release 回 zero_semaphore，交回控制权。这样"零 -> 奇/偶 -> 零 ->
    奇/偶 ..."的顺序完全由信号量的持有权决定，不依赖线程调度的巧合。
    【复杂度】时间/空间都是 O(1) 的额外开销（三个信号量，和 n 无关，n 只影响
    各自循环的次数）。
    【易错点】1) 三个信号量的初始值必须严格是"zero=1，even=odd=0"，如果 even
    或 odd 的初始值也设成非 0，会导致它们在 zero 还没打印第一个 0 之前就抢先
    执行；2) zero() 内部判断"这一轮该交给谁"必须用当前是第几轮（比如从 1 开始
    计数的 `i`）的奇偶性，而不是用即将打印的数字本身的奇偶性搞反顺序（第 1 轮
    对应要交给 odd 去打印数字 1，第 2 轮对应交给 even 去打印数字 2，轮次的奇偶性
    和"该交给谁"是直接对应的）；3) even()/odd() 内部循环的步进和范围要写对
    （`range(2, n+1, 2)` 和 `range(1, n+1, 2)`），范围写错会导致某一方多打印
    或少打印。
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.zero_semaphore = threading.Semaphore(1)
        self.even_semaphore = threading.Semaphore(0)
        self.odd_semaphore = threading.Semaphore(0)

    def zero(self, printNumber) -> None:
        for i in range(1, self.n + 1):
            self.zero_semaphore.acquire()
            printNumber(0)
            if i % 2 == 1:
                self.odd_semaphore.release()
            else:
                self.even_semaphore.release()

    def even(self, printNumber) -> None:
        for i in range(2, self.n + 1, 2):
            self.even_semaphore.acquire()
            printNumber(i)
            self.zero_semaphore.release()

    def odd(self, printNumber) -> None:
        for i in range(1, self.n + 1, 2):
            self.odd_semaphore.acquire()
            printNumber(i)
            self.zero_semaphore.release()


class DiningPhilosophers:
    """
    【题意】5 个哲学家围坐圆桌，每两个相邻的哲学家之间放一把叉子（5 人 5 把叉子）。
    每个哲学家只在"思考"和"进餐"之间交替，只有同时拿到左边和右边的叉子才能进餐，
    同一把叉子同一时刻只能被一个人使用，吃完要把两把叉子放回桌面。设计一个方案，
    实现 `wantsToEat(philosopher, pickLeftFork, pickRightFork, eat, putLeftFork,
    putRightFork)`，使得 5 个哲学家用 5 个线程并发调用这个方法时，任何一个人都
    不会永远吃不上饭（不出现死锁，也不出现某个人被无限期饿死）。
    【思路】朴素的做法——"每个人都先拿左边的叉子，再拿右边的叉子"——存在经典的
    死锁风险：如果 5 个人同时都拿到了自己左边的叉子，每个人都在等待右边那把叉子
    （被自己右边的邻居攥在手里），就形成了一个首尾相接的循环等待，谁都拿不到
    第二把叉子，永远僵持。用一把互斥锁代表一把叉子（`threading.Lock`），要打破
    这个循环等待，经典解法是让其中一个人的取叉顺序和其他人相反——让编号最大的
    那个人（4 号）先拿右边的叉子、再拿左边的叉子，其余 4 个人仍然按"先左后右"
    的顺序。这样一来，4 号哲学家和他左边的 3 号哲学家会竞争同一把叉子（4 号的
    左边叉子也是 3 号的右边叉子）时，4 号并不会先占住这把叉子干等着，而是先去
    抢 0 号和 4 号之间的那把叉子——这就打破了"每个人都攥着一把、等待下一把"这个
    环形依赖，使得死锁四个必要条件之一（循环等待）不再成立。
    【复杂度】时间：每次进餐固定获取、释放 2 把锁，均摊 O(1)；空间 O(1)（5 把锁，
    数量固定，不随调用次数变化）。
    【易错点】1) 只用了互斥锁保证"同一把叉子同时只能被一人持有"是不够的——如果
    所有人都用相同的"先左后右"顺序，即使每把叉子本身是互斥的，仍然可能全体
    陷入循环等待；必须有至少一个人的获取顺序和别人相反，才能打破环形依赖；
    2) 用嵌套的 `with lock1: with lock2:` 时，加锁顺序必须和"该由谁先拿哪把叉子"
    的设计完全一致——如果这个哲学家该"先右后左"，却写成先加左边那把锁的
    `with`，等于没有真正改变获取顺序，白白实现了打破死锁的逻辑但没生效；3) 忘记
    在获取到叉子之后要真正调用 pickLeftFork/pickRightFork/eat/putLeftFork/
    putRightFork 这几个回调（它们是 LeetCode 判题系统用来验证正确性和检测数据
    竞争的探针，遗漏调用会导致判题判定失败，即使锁本身的逻辑是对的）。
    """

    def __init__(self) -> None:
        self.forks = [threading.Lock() for _ in range(5)]

    def wantsToEat(
        self,
        philosopher: int,
        pickLeftFork,
        pickRightFork,
        eat,
        putLeftFork,
        putRightFork,
    ) -> None:
        left = philosopher
        right = (philosopher + 1) % 5
        if philosopher == 4:
            # 打破循环等待：编号最大的哲学家（4 号）先锁右边的叉子，再锁左边的叉子
            with self.forks[right]:
                with self.forks[left]:
                    pickRightFork()
                    pickLeftFork()
                    eat()
                    putRightFork()
                    putLeftFork()
        else:
            with self.forks[left]:
                with self.forks[right]:
                    pickLeftFork()
                    pickRightFork()
                    eat()
                    putLeftFork()
                    putRightFork()


class H2O:
    """
    【题意】有若干氢线程和氧线程，目标是组织它们生成水分子：每个水分子需要恰好
    2 个氢原子和 1 个氧原子"结合"，且必须是完整的一组（2H+1O）先结合完毕，下一组
    的线程才能开始结合，不能出现不同分子的线程交叉混在一起结合。氢线程调用
    releaseHydrogen()、氧线程调用 releaseOxygen() 来"通过屏障"。
    【思路】需要同时保证两件事：1) 同一时刻最多只有 2 个氢线程、1 个氧线程能够
    通过屏障准备结合；2) 一组 3 个线程（2H+1O）必须**全部**通过屏障之后，才允许
    下一组的线程开始通过。第 1 点用两个 `threading.Semaphore` 控制准入人数——
    `hydrogen_semaphore` 初始值 2，`oxygen_semaphore` 初始值 1，天然限制了"同时
    最多有 2H+1O 在尝试结合"；第 2 点用一个 `threading.Barrier(3)`：每个线程
    acquire 到许可、执行完自己的 release 回调之后，都要在这个 Barrier 上
    `wait()`，Barrier 会阻塞直到凑够 3 个线程一起到达才整体放行——这就保证了
    "同一组的 3 个线程必须互相等待，全部到齐才能继续"。放行之后，每个线程再把
    自己占用的信号量 release 回去，允许下一组对应角色的线程进来。Python 的
    `threading.Barrier` 在每一批 3 个线程通过之后会自动重置，可以被下一批
    继续复用，不需要每次手动创建新的 Barrier。
    【复杂度】时间/空间都是 O(1) 的额外开销（两个信号量 + 一个 Barrier，数量
    和线程总数无关）。
    【易错点】1) 只用信号量限制"同时最多 2H+1O 能进来"是不够的——信号量本身
    不能保证"这一组必须凑齐 3 个才能一起继续"，如果没有 Barrier，可能出现
    1 个氢线程执行完就提前继续下一轮、而另一个氢线程还没跟上的情况，破坏"一组
    一组严格结合"的要求；2) release 信号量的时机必须放在 `barrier.wait()`
    **之后**，而不是打印完 releaseHydrogen/releaseOxygen 立刻就 release——如果
    提前 release，会让下一组的线程提前抢到许可、和当前组的线程交叉执行；3) 每次
    结合出的分子里 H 和 O 的相对打印顺序（"HHO"/"HOH"/"OHH"）都算合法，不需要
    额外加逻辑强制某种固定顺序，题目本身允许这种不确定性，只要保证"每 3 个一组、
    每组恰好 2H+1O"即可。
    """

    def __init__(self) -> None:
        self.hydrogen_semaphore = threading.Semaphore(2)
        self.oxygen_semaphore = threading.Semaphore(1)
        self.barrier = threading.Barrier(3)

    def hydrogen(self, releaseHydrogen) -> None:
        self.hydrogen_semaphore.acquire()
        releaseHydrogen()
        self.barrier.wait()
        self.hydrogen_semaphore.release()

    def oxygen(self, releaseOxygen) -> None:
        self.oxygen_semaphore.acquire()
        releaseOxygen()
        self.barrier.wait()
        self.oxygen_semaphore.release()


def _test_zero_even_odd() -> None:
    n = 5
    zeo = ZeroEvenOdd(n)
    output: list[str] = []
    lock = threading.Lock()

    def print_number(x: int):
        with lock:
            output.append(str(x))

    t_zero = threading.Thread(target=zeo.zero, args=(print_number,))
    t_even = threading.Thread(target=zeo.even, args=(print_number,))
    t_odd = threading.Thread(target=zeo.odd, args=(print_number,))
    # 故意乱序启动，验证顺序由信号量保证而不是启动顺序侥幸对拍
    t_even.start()
    t_odd.start()
    t_zero.start()
    t_even.join(timeout=5)
    t_odd.join(timeout=5)
    t_zero.join(timeout=5)
    assert not (t_even.is_alive() or t_odd.is_alive() or t_zero.is_alive()), "出现死锁"
    assert "".join(output) == "0102030405"


def _test_dining_philosophers() -> None:
    dp = DiningPhilosophers()
    repetitions = 4
    eat_count = [0] * 5
    lock = threading.Lock()

    def make_eat(idx: int):
        def _eat():
            with lock:
                eat_count[idx] += 1

        return _eat

    def run(idx: int):
        noop = lambda: None
        eat_fn = make_eat(idx)
        for _ in range(repetitions):
            dp.wantsToEat(idx, noop, noop, eat_fn, noop, noop)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive(), "哲学家进餐出现死锁，线程未能在超时时间内结束"

    assert eat_count == [repetitions] * 5


def _test_h2o() -> None:
    for water in ("HOH", "OOHHHH", "HHHHOO"):
        h2o = H2O()
        output: list[str] = []
        lock = threading.Lock()

        def release_hydrogen():
            with lock:
                output.append("H")

        def release_oxygen():
            with lock:
                output.append("O")

        threads = []
        for ch in water:
            if ch == "H":
                threads.append(threading.Thread(target=h2o.hydrogen, args=(release_hydrogen,)))
            else:
                threads.append(threading.Thread(target=h2o.oxygen, args=(release_oxygen,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive(), "H2O 生成出现死锁"

        h_count = water.count("H")
        o_count = water.count("O")
        assert len(output) == h_count + o_count
        assert output.count("H") == h_count
        assert output.count("O") == o_count
        # 每 3 个字符一组，必须恰好是 2 个 H + 1 个 O（组内顺序不限，HHO/HOH/OHH 皆合法）
        for i in range(0, len(output), 3):
            chunk = output[i : i + 3]
            assert chunk.count("H") == 2 and chunk.count("O") == 1


def _self_test() -> None:
    # 正确性由 Semaphore/Lock/Barrier 保证是确定性的，重复跑几轮进一步确认没有
    # 偶发的时序漏洞（不是靠 sleep 凑合，而是重复验证同步原语本身的可靠性）。
    for _ in range(10):
        _test_zero_even_odd()
        _test_dining_philosophers()
        _test_h2o()

    print(
        "[PASS] p25_concurrency_iii: 3/3 题通过（10 轮重复验证无死锁/时序漏洞） "
        "(打印零与奇偶数/哲学家进餐/H2O生成)"
    )


if __name__ == "__main__":
    _self_test()
