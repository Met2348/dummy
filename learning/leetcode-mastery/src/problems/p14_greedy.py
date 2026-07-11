"""14 贪心：五道题的规范实现，每题都在 docstring 里给出"为什么这个贪心是对的"的论证。"""
from __future__ import annotations


# ── LC55 跳跃游戏(Medium) ────────────────────────────────────────────────
def can_jump(nums: list[int]) -> bool:
    """
    【题意】给定数组 nums，nums[i] 表示在下标 i 处最多能向前跳的步数，判断从下标 0
    出发能否到达最后一个下标。
    【思路】不需要枚举"每一步具体跳几格"这种指数级的路径空间，只需要维护一个变量
    reach = 目前为止（从下标 0 到当前已经能确认可达的所有位置里）能到达的最远下标。
    从左到右扫描，如果扫到某个下标 i 时 i > reach，说明 i 这个位置根本走不到，后面
    再远也没用，直接判 False；否则用 reach = max(reach, i + nums[i]) 持续扩展"能到达
    的边界"。这个贪心之所以对：能不能跳得更远只取决于"当前能到达的最远点"，不需要记录
    "具体是通过哪条路径到达的"，所以局部维护一个最远值就够了。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】容易写成"只看 nums[i] 本身够不够跳到下一格"，而漏掉"当前位置本身是否
    可达"这一前提判断（i > reach 就应该提前退出，即使后面某个 nums[j] 很大也没用）。
    """
    reach = 0
    for i, step in enumerate(nums):
        if i > reach:
            return False
        reach = max(reach, i + step)
    return True


# ── LC45 跳跃游戏 II(Medium) ─────────────────────────────────────────────
def jump(nums: list[int]) -> int:
    """
    【题意】同样给定 nums（一定能到达终点），求到达最后一个下标所需的最少跳跃次数。
    【思路】BFS 的"层"思想用贪心实现：把"跳跃次数"看成层数，cur_end 表示"当前这一跳
    最远能覆盖到哪个下标"（这一层的边界），farthest 表示"如果再多跳一步，下一层最远
    能到哪"。从左到右扫描，边走边更新 farthest = max(farthest, i + nums[i])；一旦扫到
    i == cur_end，说明当前这一层已经扫描完毕，必须再跳一次才能进入下一层，于是
    jumps += 1，并把 cur_end 更新为 farthest。之所以贪心有效：每一层内只关心"这一层
    内所有点能覆盖到的最远范围"，不需要关心具体从层内哪个点跳的，因为目标只是"最少
    层数"，不是"具体路径"。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】① 循环要在 range(n-1) 内结束（到达最后一个下标那一刻不需要再"进入下一层"），
    如果遍历到 n-1 会多算一次跳跃；② cur_end 和 farthest 两个变量容易搞混，cur_end 是
    "已确定"的这一跳边界，farthest 是"还在探索"的下一跳边界，更新 jumps 时用的是
    i == cur_end 而不是 i == farthest。
    """
    n = len(nums)
    jumps = 0
    cur_end = 0
    farthest = 0
    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])
        if i == cur_end:
            jumps += 1
            cur_end = farthest
    return jumps


# ── LC134 加油站(Medium) ─────────────────────────────────────────────────
def can_complete_circuit(gas: list[int], cost: list[int]) -> int:
    """
    【题意】环形路上有 n 个加油站，第 i 个站能加 gas[i] 油，从站 i 开到站 i+1 消耗
    cost[i] 油；油箱容量无限但不能为负。判断是否存在一个出发站能绕环一圈回到起点，
    若存在返回该站下标（题目保证解唯一），否则返回 -1。
    【思路】核心结论一："如果 sum(gas) >= sum(cost)，则一定存在解"——把整个环看成
    一条首尾相连的差价序列 diff[i] = gas[i]-cost[i]，总差价非负意味着"总账"是够付的，
    唯一的问题只是"从哪里开始记账，账户余额才不会中途为负"。核心结论二："如果从站 s
    出发，走到某个站 i 时油箱变负，那么 [s, i] 之间的任何一个站作为起点都不可能成功"
    ——因为从 s 到这些中间站积累的余额此刻都还是非负的（否则会在更早的地方就失败），
    把这段非负的"垫底"去掉，剩下的从中间某点出发只会更早地耗尽，所以可以直接把失败点
    的下一个站 i+1 当作新的候选起点，不需要回头重新从 s+1 逐一试——这正是贪心比暴力
    枚举每个起点快的关键（暴力是 O(n^2)，贪心一次扫描 O(n)）。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】容易忘记同时维护"全局总差价 total"和"当前段差价 tank"两个变量：tank
    只负责判断"从当前候选起点出发是否中途破产"，total 才是"存不存在解"的最终判据；
    只看 tank 不看 total 会漏判无解的情况。
    """
    total = 0
    tank = 0
    start = 0
    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total += diff
        tank += diff
        if tank < 0:
            start = i + 1
            tank = 0
    return start if total >= 0 else -1


# ── LC435 无重叠区间(Medium) ─────────────────────────────────────────────
def erase_overlap_intervals(intervals: list[list[int]]) -> int:
    """
    【题意】给定若干区间，求最少需要移除多少个区间，才能让剩下的区间两两不重叠
    （区间端点相接，如 [1,2] 和 [2,3]，不算重叠）。
    【思路】等价于经典的"活动选择问题"：按**结束时间**从小到大排序，贪心地尽量保留
    结束最早的区间——因为结束越早，留给后面区间的空间就越大，不会比"保留结束更晚的
    区间"更差。扫描时维护 prev_end = 上一个被保留区间的结束时间：如果当前区间的起点
    < prev_end，说明和上一个保留的区间重叠，必须移除当前这个（计数 +1，且不更新
    prev_end，因为已保留的那个结束更早、更优）；否则保留当前区间，更新 prev_end。
    【复杂度】时间 O(n log n)（排序主导），空间 O(1)（不计排序开销）。
    【易错点】① 容易按起点排序而不是按结束时间排序——按结束时间排序才能保证"贪心保留
    的区间结束时间最早"这个性质成立；② 判断重叠时要用严格小于 `start < prev_end`，
    端点相等（如 [1,2] 后接 [2,3]）不算重叠，写成 `<=` 会多删。
    """
    if not intervals:
        return 0
    intervals = sorted(intervals, key=lambda iv: iv[1])
    removed = 0
    prev_end = intervals[0][1]
    for start, end in intervals[1:]:
        if start < prev_end:
            removed += 1
        else:
            prev_end = end
    return removed


# ── LC763 划分字母区间(Medium) ───────────────────────────────────────────
def partition_labels(s: str) -> list[int]:
    """
    【题意】把字符串 s 划分成尽量多的片段，使同一个字母只会出现在一个片段里，
    返回每个片段的长度列表（各片段顺序拼接起来等于原串）。
    【思路】关键 insight："每个字符最后一次出现的位置"决定了包含这个字符的片段最早
    能在哪里结束——如果某片段里出现了字符 c，那么这个片段的右边界至少要延伸到 c 在
    整个字符串里最后一次出现的位置，否则后面再出现的 c 会和当前片段"撞车"，破坏
    "同一字母只在一个片段"的约束。于是先用一次遍历记录 last[c] = c 最后出现的下标；
    再从左到右扫描，用 end = max(end, last[当前字符]) 不断扩展当前片段的右边界；
    一旦扫描下标 i 追上了 end（说明当前片段里所有字符的"最后出现位置"都已经被覆盖，
    不会再有字符跑出这个片段），就说明可以在这里切一刀，记录片段长度并开启下一段。
    【复杂度】时间 O(n)（两次线性扫描），空间 O(1)（字符集大小是常数，如小写字母 26）。
    【易错点】容易只扫一遍就想切割，忘记"先算好每个字符的最后出现位置"这个预处理步骤
    ——没有这个全局信息，扫描到某个位置时根本不知道当前片段是否已经安全（后面还会不会
    再冒出同一个字符）。
    """
    last = {ch: i for i, ch in enumerate(s)}
    result: list[int] = []
    start = end = 0
    for i, ch in enumerate(s):
        end = max(end, last[ch])
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    return result


def _self_test() -> None:
    assert can_jump([2, 3, 1, 1, 4]) is True
    assert can_jump([3, 2, 1, 0, 4]) is False

    assert jump([2, 3, 1, 1, 4]) == 2
    assert jump([2, 3, 0, 1, 4]) == 2

    assert can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]) == 3
    assert can_complete_circuit([2, 3, 4], [3, 4, 3]) == -1

    assert erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
    assert erase_overlap_intervals([[1, 2], [1, 2], [1, 2]]) == 2
    assert erase_overlap_intervals([[1, 2], [2, 3]]) == 0

    assert partition_labels("ababcbacadefegdehijhklij") == [9, 7, 8]

    print("[PASS] p14_greedy: 5 题(跳跃游戏x2/加油站/无重叠区间/划分字母区间)全部通过")


if __name__ == "__main__":
    _self_test()
