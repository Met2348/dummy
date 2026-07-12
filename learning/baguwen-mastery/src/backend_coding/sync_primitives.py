"""生产者-消费者模型：手写 Lock + Condition 实现有界缓冲区（不直接用 queue.Queue）。

多个生产者线程把 0..N-1 的整数放入一个有界共享缓冲区，多个消费者线程从缓冲区取出并
处理。配合 backend_qa/qbank_os.py 里关于进程/线程同步、协程调度的问答，用能跑的代码
验证对 Lock/Condition 这两个同步原语的理解——"满则生产者等待、空则消费者等待、条件
变量被正确 notify" 这几件事有没有真的搞对，而不是背一句"用条件变量做同步"就了事。

停止方式：主线程等所有生产者用 join() 彻底结束（确认"不会再有新数据"）之后，才往
缓冲区里放入与消费者数量相等的哨兵（sentinel），每个消费者收到一个哨兵就退出——不靠
sleep 时长去猜测生产/消费是否完成，所有断言都建立在 join() 之后。
"""
from __future__ import annotations

import threading
from collections import deque

_SENTINEL = None  # 放入缓冲区表示"没有更多数据了"，消费者收到就退出


class BoundedBuffer:
    """手写的有界阻塞缓冲区：一把 Lock + 两个共用该 Lock 的 Condition。"""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buf: deque = deque()
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

    def put(self, item) -> None:
        with self._not_full:
            while len(self._buf) >= self.capacity:
                self._not_full.wait()
            self._buf.append(item)
            self._not_empty.notify()

    def get(self):
        with self._not_empty:
            while not self._buf:
                self._not_empty.wait()
            item = self._buf.popleft()
            self._not_full.notify()
            return item

    def qsize(self) -> int:
        with self._lock:
            return len(self._buf)


class _Counter:
    """生产者共享的"下一个待生产整数"计数器，自带锁保证多生产者互斥递增，
    从而保证 0..total-1 里每个值恰好被某一个生产者取走一次，不重不漏。"""

    def __init__(self, total: int) -> None:
        self.total = total
        self.next_value = 0
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            if self.next_value >= self.total:
                return None
            v = self.next_value
            self.next_value += 1
            return v


def _producer(buf: BoundedBuffer, counter: _Counter) -> None:
    while True:
        v = counter.take()
        if v is None:
            return
        buf.put(v)


def _consumer(buf: BoundedBuffer, consumed: list, consumed_lock: threading.Lock) -> None:
    while True:
        item = buf.get()
        if item is _SENTINEL:
            return
        with consumed_lock:
            consumed.append(item)


def run_producer_consumer(
    n_items: int = 500, n_producers: int = 4, n_consumers: int = 5, capacity: int = 16
) -> tuple[list[int], int]:
    """跑一轮生产者-消费者：生产 0..n_items-1。

    返回 (消费者实际收集到的完整列表, 全部线程结束后缓冲区里剩余的元素个数)。
    """
    buf = BoundedBuffer(capacity)
    counter = _Counter(n_items)
    consumed: list[int] = []
    consumed_lock = threading.Lock()

    producers = [
        threading.Thread(target=_producer, args=(buf, counter)) for _ in range(n_producers)
    ]
    consumers = [
        threading.Thread(target=_consumer, args=(buf, consumed, consumed_lock))
        for _ in range(n_consumers)
    ]

    for t in producers:
        t.start()
    for t in consumers:
        t.start()

    # 所有生产者必须先彻底结束（join），才能确定"不会再有新数据"，这时才能安全地
    # 放入哨兵通知消费者退出——不能靠 sleep 猜测生产是否完成。
    for t in producers:
        t.join()

    for _ in consumers:
        buf.put(_SENTINEL)

    # 同样必须等所有消费者 join() 结束，才能对最终状态做断言。
    for t in consumers:
        t.join()

    return consumed, buf.qsize()


def _self_test() -> None:
    n_items = 500
    consumed, remaining = run_producer_consumer(
        n_items=n_items, n_producers=4, n_consumers=5, capacity=16
    )

    # 生产总数量 == 消费总数量。
    assert len(consumed) == n_items, (len(consumed), n_items)
    # 消费到的数据集合完整覆盖生产的数据集合：不丢、不重复。
    assert sorted(consumed) == list(range(n_items))
    assert len(set(consumed)) == n_items
    # 全部线程 join() 结束后，共享缓冲区必须为空。
    assert remaining == 0, remaining

    # 再用一组不同的生产者/消费者/容量组合（极端到容量=1，逼出大量等待/唤醒）
    # 交叉验证正确性不是巧合。
    n_items_2 = 777
    consumed_2, remaining_2 = run_producer_consumer(
        n_items=n_items_2, n_producers=7, n_consumers=3, capacity=1
    )
    assert sorted(consumed_2) == list(range(n_items_2))
    assert len(set(consumed_2)) == n_items_2
    assert remaining_2 == 0, remaining_2

    print(
        f"[PASS] sync_primitives: 生产{n_items}/{n_items_2} == 消费"
        f"{len(consumed)}/{len(consumed_2)}，无丢失无重复，缓冲区正确清空"
    )


if __name__ == "__main__":
    _self_test()
