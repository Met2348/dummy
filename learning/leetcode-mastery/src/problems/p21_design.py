"""分类 21：设计题 —— 给一组接口复杂度要求，考验能否选对底层数据结构组合。"""
from __future__ import annotations

import bisect
import heapq
from collections import OrderedDict, defaultdict


class LRUCache:
    """
    【题意】设计一个容量固定为 capacity 的缓存：get(key) 命中返回 value、未命中返回 -1，
    并把该 key 标记为"最近使用"；put(key, value) 插入或更新，超出容量时淘汰"最久未使用"
    的那个 key。要求 get/put 都是 O(1)。
    【思路】"最近使用"这个概念天然是一个双向链表能表达的顺序（每次访问就把节点挪到
    链表一端），但单独用双向链表无法 O(1) 定位某个 key 在链表里的位置；单独用哈希表
    能 O(1) 定位，但哈希表本身不记录"访问的先后顺序"。两者缺一不可，"哈希表 + 双向
    链表"的组合缺一不可——哈希表负责"给定 key，O(1) 找到对应节点"，双向链表负责
    "O(1) 把这个节点挪到最近使用的一端、以及 O(1) 删除链表另一端最久未使用的节点"。
    Python 的 `collections.OrderedDict` 已经用双向链表 + 哈希表实现了这个组合，
    `move_to_end` 就是"挪到最近使用一端"，`popitem(last=False)` 就是"淘汰最久未使用
    的一端"，不需要手写链表节点。
    【复杂度】get/put 均为 O(1)（OrderedDict 的哈希查找 + 双向链表调整都是 O(1)）；
    空间 O(capacity)。
    【易错点】1) get 命中之后忘记 move_to_end，会导致这个 key 明明刚被访问过，却在
    下一次淘汰时被误判成"最久未使用"；2) put 已存在的 key 时，必须先更新值再
    move_to_end（或者反过来都行，但两步都不能漏），同时不需要，也不应该在这种情况下
    触发淘汰逻辑（容量没有真正增加）；3) 淘汰用 `popitem(last=False)` 而不是默认的
    `popitem()`——默认参数弹出的是最近插入的一端，正好和需求相反。
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class LFUCache:
    """
    【题意】设计一个容量固定为 capacity 的缓存，get/put 语义和 LRU 类似，但淘汰策略是
    "淘汰访问频率最低的 key"；如果多个 key 频率并列最低，淘汰其中最久未使用的那个。
    要求 get/put 都是 O(1)。
    【思路】和 LRU 相比，LFU 多了一维信息——"频率"，所以需要在 LRU 那套"哈希表 +
    双向链表"的基础上，再多一层"频率 -> 该频率下的 LRU 队列"的结构：`freq_to_keys`
    是一个 `频率 -> OrderedDict(该频率下的 key 们，按最近使用顺序排列)` 的映射；
    `key_to_val`、`key_to_freq` 分别记录每个 key 当前的值和当前的频率；额外维护
    `min_freq` 表示"当前所有 key 里最低的频率是多少"，淘汰时直接去
    `freq_to_keys[min_freq]` 这个 OrderedDict 里弹出最久未使用的一个（`popitem
    (last=False)`）——因为它是这个最低频率桶里最久未被访问的，同时也是全局唯一满足
    "频率最低 + 频率并列时最久未使用"这个双重淘汰条件的 key。每次访问（get 命中或
    put 命中已有 key）都要把该 key 的频率 +1，从旧频率桶挪到新频率桶；如果挪走之后
    旧频率桶（恰好是 min_freq 桶）空了，min_freq 要 +1。
    【复杂度】get/put 均为 O(1)（哈希查找 + OrderedDict 的 O(1) 挪动）；空间
    O(capacity)。
    【易错点】1) 忘记维护 min_freq，或者维护错了时机——只有当"挪走的 key 恰好来自
    min_freq 桶，且挪走后该桶变空"时，min_freq 才需要 +1，不能每次访问都无脑加一；
    2) 新插入一个 key 时，它的初始频率是 1，min_freq 也要重置为 1（因为新 key 一定是
    当前频率最低的）；3) capacity 为 0 时（LFUCache(0)）必须任何 put 都不生效，
    否则会在空的 freq_to_keys 结构上出错。
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.min_freq = 0
        self.key_to_val: dict[int, int] = {}
        self.key_to_freq: dict[int, int] = {}
        self.freq_to_keys: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)

    def _bump_freq(self, key: int) -> None:
        freq = self.key_to_freq[key]
        del self.freq_to_keys[freq][key]
        if not self.freq_to_keys[freq] and self.min_freq == freq:
            self.min_freq += 1
        self.key_to_freq[key] = freq + 1
        self.freq_to_keys[freq + 1][key] = None

    def get(self, key: int) -> int:
        if key not in self.key_to_val:
            return -1
        self._bump_freq(key)
        return self.key_to_val[key]

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        if key in self.key_to_val:
            self.key_to_val[key] = value
            self._bump_freq(key)
            return
        if len(self.key_to_val) >= self.capacity:
            evict_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
            del self.key_to_val[evict_key]
            del self.key_to_freq[evict_key]
        self.key_to_val[key] = value
        self.key_to_freq[key] = 1
        self.freq_to_keys[1][key] = None
        self.min_freq = 1


class Twitter:
    """
    【题意】设计一个简化版推特：post_tweet(user_id, tweet_id) 发推；
    get_news_feed(user_id) 返回该用户自己 + 所关注的人发的推文里，最近 10 条的
    tweet_id（按时间从新到旧）；follow/unfollow 维护关注关系。
    【思路】每条推文打上一个全局递增的时间戳（用一个自增计数器模拟，不需要真实时间），
    每个用户维护自己发过的 (时间戳, tweet_id) 列表。查询 news feed 时，把"自己 + 所有
    关注的人"这些人各自的推文列表合并到一起，取时间戳最大的前 10 个——这是标准的
    "从多个（各自内部有序的）列表里找整体 Top-K"场景，`heapq.nlargest` 内部就是用堆
    实现的（维护一个大小为 10 的最小堆，见 15 章"堆/Top-K"的技巧），比把所有候选推文
    拼起来整体排序更省：只需要 O(候选数 * log 10)，不需要 O(候选数 * log 候选数)。
    【复杂度】post_tweet: O(1)；get_news_feed: O(F log 10)（F 为候选推文总数，即
    自己和关注的人发过的推文数之和）；follow/unfollow: O(1)；空间 O(所有推文数 +
    关注关系数)。
    【易错点】1) 自己发的推文本身也要出现在自己的 news feed 里，容易只收集"关注的人"
    的推文而漏掉自己；2) follow 一个已经关注的人、或者 unfollow 一个没关注的人，都
    应该静默处理（不报错），用 set 天然满足这个要求；3) 不能 follow 自己（题目通常
    保证不会出现，但用 set 存关注关系、且 news feed 单独把"自己"加进候选而不依赖
    follow 自己，逻辑上更稳妥）。
    """

    def __init__(self) -> None:
        self.timestamp = 0
        self.tweets: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.following: dict[int, set[int]] = defaultdict(set)

    def post_tweet(self, user_id: int, tweet_id: int) -> None:
        self.tweets[user_id].append((self.timestamp, tweet_id))
        self.timestamp += 1

    def get_news_feed(self, user_id: int) -> list[int]:
        candidates = list(self.tweets[user_id])
        for followee_id in self.following[user_id]:
            candidates.extend(self.tweets[followee_id])
        top = heapq.nlargest(10, candidates, key=lambda item: item[0])
        return [tweet_id for _ts, tweet_id in top]

    def follow(self, follower_id: int, followee_id: int) -> None:
        if follower_id != followee_id:
            self.following[follower_id].add(followee_id)

    def unfollow(self, follower_id: int, followee_id: int) -> None:
        self.following[follower_id].discard(followee_id)


class SnapshotArray:
    """
    【题意】设计一个长度固定为 length、初始值全为 0 的数组：set(index, val) 修改某个
    下标的值；snap() 给当前整个数组的状态拍一张"快照"，返回快照编号（从 0 开始递增）；
    get(index, snap_id) 查询某个下标在某次快照那一刻的值。
    【思路】如果每次 snap() 都真的把整个数组拷贝一份，空间和时间都是 O(length)，当
    snap 很频繁时代价很高。核心 insight 是：一个下标在两次快照之间，如果没被 set 过，
    它的值就压根没变过——没必要在每次快照时都重新记录一遍。于是给**每个下标**单独维护
    一条"修改历史"：`[(发生修改时的快照编号, 修改后的值), ...]`，只有真正调用 set 时
    才追加一条记录（而且记录的是"当前还没被拍过的快照编号"，也就是 self.snap_id）。
    查询 get(index, snap_id) 时，要找的是"这条历史记录里，快照编号 <= snap_id 的最后
    一次修改"——因为历史记录本身按快照编号严格递增排列，这是一个标准的"在有序序列里
    找最后一个 <= 目标值的位置"，用 `bisect_right` 二分即可，不需要线性扫描整条历史。
    【复杂度】set: 均摊 O(1)（同一快照内多次 set 同一下标会直接覆盖，不会无限增长）；
    snap: O(1)；get: O(log h)（h 为该下标历史修改次数）；空间 O(总的 set 调用次数)。
    【易错点】1) 同一个快照编号内对同一下标多次 set，如果每次都往历史里追加一条新
    记录，会让同一个快照编号出现两条历史（后一条覆盖前一条的语义），既浪费空间又让
    二分逻辑变复杂——应该判断"历史最后一条记录的快照编号是否等于当前快照编号"，是就
    原地覆盖，不是才追加；2) 二分要用 `bisect_right` 而不是 `bisect_left`，因为要找
    的是"最后一个 <= snap_id"的位置，`bisect_right` 定位到的插入点减一才是正确下标；
    3) 某个下标从未被 set 过、或者查询的 snap_id 早于它第一次被 set 的时刻，都应该
    返回默认值 0，而不是抛出下标越界异常。
    """

    def __init__(self, length: int) -> None:
        self.snap_id = 0
        self.history: list[list[tuple[int, int]]] = [[] for _ in range(length)]

    def set(self, index: int, val: int) -> None:
        h = self.history[index]
        if h and h[-1][0] == self.snap_id:
            h[-1] = (self.snap_id, val)
        else:
            h.append((self.snap_id, val))

    def snap(self) -> int:
        sid = self.snap_id
        self.snap_id += 1
        return sid

    def get(self, index: int, snap_id: int) -> int:
        h = self.history[index]
        i = bisect.bisect_right(h, (snap_id, float("inf"))) - 1
        return h[i][1] if i >= 0 else 0


class MyCircularQueue:
    """
    【题意】设计一个固定容量为 k 的循环队列：en_queue(value) 入队（满了返回 False）、
    de_queue() 出队（空了返回 False）、front()/rear() 查看队首/队尾（空则返回 -1）、
    is_empty()/is_full() 判断空/满状态。
    【思路】"循环"的含义是：底层用一个固定大小 k 的数组，队首/队尾指针在数组末尾之后
    会绕回到数组开头，而不是像普通队列那样让数组无限往后增长（这样空间恒定为 O(k)，
    不会随着反复入队出队而浪费）。用 `head` 记录队首下标、`count` 记录当前元素个数
    （队尾下标可以直接用 `(head + count) % k` 算出来，不需要单独维护，减少一个容易
    弄错同步关系的状态变量）；入队时把新值写到 `(head + count) % k` 这个位置、count
    加一；出队时把 head 往后移一位（对 k 取模实现"绕回"）、count 减一。
    【复杂度】全部操作均为 O(1)；空间 O(k)。
    【易错点】1) 判空/判满都必须通过 count 和 0/k 比较来做，不能用 `head == tail`
    这种朴素写法——队列全空和队列全满时 head 和"下一个可写位置"经常会重合，单纯比较
    两个指针无法区分这两种状态；2) 取模时下标运算一定要对 `self.capacity` 取模，
    写成对 `len(self.data)` 之外的其他数字取模会导致越界或者"绕不回去"；3) rear()
    对应的下标是 `(head + count - 1) % k`，容易漏掉这个 -1 写成 `(head + count) % k`
    （那其实是"下一个即将写入的位置"，不是"当前队尾"）。
    """

    def __init__(self, k: int) -> None:
        self.capacity = k
        self.data: list[int] = [0] * k
        self.head = 0
        self.count = 0

    def en_queue(self, value: int) -> bool:
        if self.is_full():
            return False
        tail = (self.head + self.count) % self.capacity
        self.data[tail] = value
        self.count += 1
        return True

    def de_queue(self) -> bool:
        if self.is_empty():
            return False
        self.head = (self.head + 1) % self.capacity
        self.count -= 1
        return True

    def front(self) -> int:
        return -1 if self.is_empty() else self.data[self.head]

    def rear(self) -> int:
        if self.is_empty():
            return -1
        tail = (self.head + self.count - 1) % self.capacity
        return self.data[tail]

    def is_empty(self) -> bool:
        return self.count == 0

    def is_full(self) -> bool:
        return self.count == self.capacity


class MyHashSet:
    """
    【题意】不借助 Python 内置 set，自己实现一个哈希集合：add(key)、remove(key)、
    contains(key) -> bool。
    【思路】这道题的教学目的是"理解哈希表本身是怎么实现的"，而不是拿现成的 set 走
    捷径。用固定数量（选一个不太小的质数，减少同一个桶里冲突扎堆的概率）的"桶"数组，
    每个桶是一个普通列表，处理哈希冲突用最朴素的"链地址法"：一个 key 先通过
    `key % 桶数` 算出该进哪个桶，桶内如果已有这个 key 就不重复添加、删除时线性扫描
    桶内列表找到并移除。这正是教科书里哈希表最基础的实现方式：哈希函数负责"缩小
    查找范围"，链表/列表负责"在缩小后的范围内兜底处理冲突"。
    【复杂度】桶数固定、且假设哈希分布均匀时，add/remove/contains 均摊 O(1)（退化到
    最坏情况——所有 key 哈希到同一个桶——是 O(n)，但足够多的桶能让这种情况在实践中
    很少发生）；空间 O(不同 key 的个数 + 桶数)。
    【易错点】1) 直接把 Python 内置 `set()` 包一层当作实现，虽然能通过测试，但完全
    没有练到"哈希表内部原理"这个考点，属于"技术上合规但没达到出题目的"；2) add 一个
    已存在的 key 时要判重（不能让同一个桶里出现重复的 key）；3) 桶数选得太小（比如
    直接选 1 或者选一个和测试数据规律吻合的数），容易让链表退化成近似 O(n) 的线性
    列表，掩盖了哈希表本该有的效率。
    """

    def __init__(self) -> None:
        self._num_buckets = 1009
        self._buckets: list[list[int]] = [[] for _ in range(self._num_buckets)]

    def _hash(self, key: int) -> int:
        return key % self._num_buckets

    def add(self, key: int) -> None:
        bucket = self._buckets[self._hash(key)]
        if key not in bucket:
            bucket.append(key)

    def remove(self, key: int) -> None:
        bucket = self._buckets[self._hash(key)]
        if key in bucket:
            bucket.remove(key)

    def contains(self, key: int) -> bool:
        return key in self._buckets[self._hash(key)]


def _self_test() -> None:
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)
    assert cache.get(2) == -1
    cache.put(4, 4)
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4

    lfu = LFUCache(2)
    lfu.put(1, 1)
    lfu.put(2, 2)
    assert lfu.get(1) == 1
    lfu.put(3, 3)
    assert lfu.get(2) == -1
    assert lfu.get(3) == 3
    lfu.put(4, 4)
    assert lfu.get(1) == -1
    assert lfu.get(3) == 3
    assert lfu.get(4) == 4

    tw = Twitter()
    tw.post_tweet(1, 5)
    assert tw.get_news_feed(1) == [5]
    tw.follow(1, 2)
    tw.post_tweet(2, 6)
    assert tw.get_news_feed(1) == [6, 5]
    tw.unfollow(1, 2)
    assert tw.get_news_feed(1) == [5]

    arr = SnapshotArray(3)
    arr.set(0, 5)
    assert arr.snap() == 0
    arr.set(0, 6)
    assert arr.get(0, 0) == 5

    cq = MyCircularQueue(3)
    assert cq.en_queue(1) is True
    assert cq.en_queue(2) is True
    assert cq.en_queue(3) is True
    assert cq.en_queue(4) is False
    assert cq.rear() == 3
    assert cq.is_full() is True
    assert cq.de_queue() is True
    assert cq.en_queue(4) is True
    assert cq.rear() == 4

    hs = MyHashSet()
    hs.add(1)
    hs.add(2)
    assert hs.contains(1) is True
    assert hs.contains(3) is False
    hs.add(2)
    assert hs.contains(2) is True
    hs.remove(2)
    assert hs.contains(2) is False

    print(
        "[PASS] p21_design: 6/6 题通过 "
        "(LRU缓存/LFU缓存/设计推特/快照数组/设计循环队列/设计哈希集合)"
    )


if __name__ == "__main__":
    _self_test()
