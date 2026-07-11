"""哈希表：用 O(n) 额外空间换 O(1) 查找/去重/计数的 5 道代表题。"""
from __future__ import annotations

import random
from collections import defaultdict


def contains_duplicate(nums: list[int]) -> bool:
    """
    【题意】给定整数数组，判断是否存在任意一个值出现至少两次。
    【思路】暴力解两层循环比较所有元素对，O(n^2)；或者排序后比较相邻元素，O(n log n)。但
    "这个值之前是否出现过"本质是一个查找问题——哈希 set 天生就是为这类问题设计的：一边遍历
    一边往 set 里塞，塞之前先看这个值在不在 set 里，在就说明重复，不在才继续。
    【复杂度】时间 O(n)，空间 O(n)。
    【易错点】1) 容易先把整个数组塞进 set 再比较 len(set) != len(nums)（这个写法同样正确
    且更简洁，可以互相印证，但没法在发现重复的第一时间提前退出）；2) 别把"判断是否重复"误写
    成"先计数再判断计数是否 > 1"，多做了无谓的工作。
    """
    seen: set[int] = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False


def is_anagram(s: str, t: str) -> bool:
    """
    【题意】判断字符串 t 是否是 s 的字母异位词（由 s 的字符重新排列得到，每个字符出现次数
    必须完全一致）。
    【思路】"字符出现次数完全一致"就是一个频次比较问题——用哈希表分别统计两个字符串每个字符
    的出现次数，比较两张频次表是否相等；这比"排序后比较"（O(n log n)）更能体现哈希表"计数"
    这个典型用法，且是线性时间。
    【复杂度】时间 O(n)，空间 O(字符集大小)。
    【易错点】1) 长度不等时可以直接提前返回 False，省去构造计数表的开销（非必须但是常见优化）；
    2) 用同一张表对 s 做 +1、对 t 做 -1，最后判断所有计数是否都归零，比维护两张表再比较更省一次
    遍历，但两种写法都对；3) 别忘了大小写、非字母字符等边界（本题只考虑给定的字符集合本身）。
    """
    if len(s) != len(t):
        return False
    count: dict[str, int] = defaultdict(int)
    for ch in s:
        count[ch] += 1
    for ch in t:
        count[ch] -= 1
    return all(v == 0 for v in count.values())


def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    【题意】给定字符串数组，把互为字母异位词的字符串分到同一组，返回所有组（组和组内顺序
    都不作要求）。
    【思路】"互为异位词"是一种等价关系，这题的关键 insight 是找一个"规范形式"当哈希表的
    key，使得两个字符串互为异位词当且仅当它们的规范形式相同——把每个字符串排序后得到的字符串
    就是这样一个 key（"eat" 和 "tea" 排序后都是 "aet"）。有了这个 key，分组就退化成了
    "按 key 分桶"，用哈希表 key→list 一次遍历即可。
    【复杂度】时间 O(n·k log k)（n 个字符串，每个长度 k，排序耗时 k log k），空间 O(n·k)。
    【易错点】1) key 不一定非要用排序字符串，也可以用"每个字母出现次数"的元组（能把单次排序的
    O(k log k) 降到 O(k)），排序法更直观、代码更短，适合新手先掌握；2) 忘记最终要返回
    list(分组.values())，而不是字典本身；3) 测试比较结果时组和组间顺序都不固定，要先对组内
    排序、再对组间排序才能比较是否相等。
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for w in strs:
        key = "".join(sorted(w))
        groups[key].append(w)
    return list(groups.values())


def longest_consecutive(nums: list[int]) -> int:
    """
    【题意】给定未排序的整数数组，找出数字连续的最长序列（如 [100,4,200,1,3,2] 中
    1,2,3,4 是长度为 4 的连续序列）的长度，要求整体时间复杂度为 O(n)。
    【思路】详见讲义里的深挖：暴力解先排序再线性扫描相邻差是否为 1，卡在排序这一步的
    O(n log n)。把数组全部丢进 set 之后，只对"没有前驱"（即 x - 1 不在 set 里）的数字才
    当作一个候选序列的起点，向后数 x+1, x+2, ...；因为每个数字只会在"自己就是某个序列起点"
    时被内层 while 循环访问，其余情况都被"有前驱"这个门槛挡掉了，所以总的内层循环次数摊销下来
    是 O(n)，而不是每个数字都各自向后数一遍的 O(n^2)。
    【复杂度】时间 O(n)，空间 O(n)。
    【易错点】1) 不加"是否是起点"这个门槛，直接对每个数字都向后数，会退化成 O(n^2)——这是
    新手最容易掉进的坑，一定要有 x - 1 not in num_set 这句判断；2) 数组可能有重复值，转成 set
    天然去重，若不转 set 直接在 list 上做 in 判断，每次 in 都是 O(n)，同样会退化；3) 空数组
    要能正确返回 0。
    """
    num_set = set(nums)
    best = 0
    for x in num_set:
        if x - 1 not in num_set:
            length = 1
            y = x
            while y + 1 in num_set:
                y += 1
                length += 1
            best = max(best, length)
    return best


class RandomizedSet:
    """
    【题意】设计一个数据结构，支持平均 O(1) 时间的 insert(val)（不存在则插入并返回 True，
    否则返回 False）、remove(val)（存在则删除并返回 True，否则返回 False）、get_random()
    （等概率返回集合中的一个元素）。
    【思路】哈希表能 O(1) 查找/插入/删除，但不支持"等概率随机取一个元素"（无法 O(1) 随机访问
    下标）；数组能 O(1) 随机访问（下标随机取值），但数组删除中间元素是 O(n)（要搬移后面所有
    元素）。组合两者：用 list 存元素本身（支持随机下标访问），用 dict 存"值→它在 list 中的
    下标"（支持 O(1) 定位）；删除时的关键技巧是不做"搬移"，而是把待删元素和 list 末尾元素
    交换位置后再 pop 末尾——交换只需要 O(1) 更新两个下标，pop 末尾也是 O(1)。
    【复杂度】insert/remove/get_random 均为平均 O(1) 时间，空间 O(n)。
    【易错点】1) 删除时如果待删元素恰好就是末尾元素，交换会把它和自己交换，index 字典的更新
    要保证在这种情况下依然自洽（先取出 last 再统一走同一套赋值逻辑，不用单独分支也能正确）；
    2) 交换后必须同步更新被换到前面那个元素在 index 字典里的下标，否则字典和 list 会不一致；
    3) get_random 必须通过"随机下标取 list 元素"实现（如 random.choice(list)），不能对 dict
    做随机——dict 不支持 O(1) 的随机下标访问。
    """

    def __init__(self) -> None:
        self.values: list[int] = []
        self.index: dict[int, int] = {}

    def insert(self, val: int) -> bool:
        if val in self.index:
            return False
        self.index[val] = len(self.values)
        self.values.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.index:
            return False
        i = self.index[val]
        last = self.values[-1]
        self.values[i] = last
        self.index[last] = i
        self.values.pop()
        del self.index[val]
        return True

    def get_random(self) -> int:
        return random.choice(self.values)


def _self_test() -> None:
    assert contains_duplicate([1, 2, 3, 1]) is True
    assert contains_duplicate([1, 2, 3, 4]) is False

    assert is_anagram("anagram", "nagaram") is True
    assert is_anagram("rat", "car") is False

    got = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    normalized = sorted(sorted(g) for g in got)
    expected = sorted(sorted(g) for g in [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]])
    assert normalized == expected

    assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
    assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9

    rs = RandomizedSet()
    assert rs.insert(1) is True
    assert rs.remove(2) is False
    assert rs.insert(2) is True
    assert rs.get_random() in (1, 2)
    assert rs.remove(1) is True
    assert rs.insert(2) is False
    assert rs.get_random() == 2

    print("[PASS] p03_hashing: 5 题（存在重复元素/有效字母异位词/字母异位词分组/最长连续序列/O(1)随机集合）全部通过")


if __name__ == "__main__":
    _self_test()
