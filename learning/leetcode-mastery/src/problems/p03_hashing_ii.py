"""哈希表 · 进阶补充（Part II）：不重讲框架，扩大"哈希表当计数器/去重判断"应用场景的 7 道题。"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


def is_isomorphic(s: str, t: str) -> bool:
    """
    【题意】给定两个等长字符串 s、t，判断 s 中的字符是否可以按某种一一映射规则替换成 t 中对应
    字符，使替换后 s 变成 t（相同字符必须映射到相同字符，不同字符不能映射到同一个字符）。
    【思路】"一一映射"是双射，只检查"s 的字符 -> t 的字符"这一个方向的哈希表不够——比如
    s="ab"、t="aa"，从 s 到 t 的映射 a->a、b->a 看起来每个 s 字符都映射到了唯一的 t 字符，但
    两个不同的 s 字符（a 和 b）映射到了同一个 t 字符，违反双射。必须同时维护两张哈希表（s->t
    和 t->s），一边遍历一边检查两个方向是否都自洽。
    【复杂度】时间 O(n)，空间 O(字符集大小)。
    【易错点】1) 只查一个方向的映射表是这题最典型的漏洞，一定要同时验证反向映射；2) 判断
    "冲突"时要区分"从未见过这个字符"（可以放心建立新映射）和"见过但映射到了别的字符"（应该
    返回 False），不能把两种情况混为一谈；3) 长度不同直接返回 False，省去后续遍历。
    """
    if len(s) != len(t):
        return False
    map_st: dict[str, str] = {}
    map_ts: dict[str, str] = {}
    for a, b in zip(s, t):
        if a in map_st and map_st[a] != b:
            return False
        if b in map_ts and map_ts[b] != a:
            return False
        map_st[a] = b
        map_ts[b] = a
    return True


def word_pattern(pattern: str, s: str) -> bool:
    """
    【题意】给定模式串 pattern（每个字符代表一类）和以空格分隔的字符串 s，判断 s 里的单词序列
    是否符合 pattern 描述的双射规律（pattern 里相同字符对应相同单词，不同字符对应不同单词）。
    【思路】和"同构字符串"是同一个双射校验模型，只是把"字符↔字符"换成了"字符↔单词"：同样需要
    两张哈希表（pattern 字符 -> 单词、单词 -> pattern 字符）互相校验，任何一个方向出现"同一个
    key 映射到不同 value"就说明不满足双射。
    【复杂度】时间 O(n)（n 是单词总数），空间 O(不同字符数 + 不同单词数)。
    【易错点】1) 必须先把 s 按空格切分成单词列表，再和 pattern 的字符一一对应，直接拿字符串
    整体比较是错的；2) pattern 长度和单词个数不相等时要直接返回 False（比如 "aaaa" 对
    4 个单词但其中有单词重复使用导致矛盾，这属于双射校验会自然抓到的情况，而"长度都不相等"是
    另一种更基础的不合法输入，需要单独判断）；3) 两个方向的哈希表缺一不可，只查"pattern ->
    单词"会漏掉"两个不同字符映射到同一个单词"这种情况。
    """
    words = s.split()
    if len(pattern) != len(words):
        return False
    map_pw: dict[str, str] = {}
    map_wp: dict[str, str] = {}
    for p, w in zip(pattern, words):
        if p in map_pw and map_pw[p] != w:
            return False
        if w in map_wp and map_wp[w] != p:
            return False
        map_pw[p] = w
        map_wp[w] = p
    return True


def intersect(nums1: list[int], nums2: list[int]) -> list[int]:
    """
    【题意】给定两个整数数组，返回它们的交集，重复元素要按两个数组里都出现的次数取较小值
    （比如 nums1 里某值出现 3 次、nums2 里出现 2 次，结果里这个值出现 2 次），结果顺序不作要求。
    【思路】这是哈希表"计数"用法的直接应用：先用 Counter 统计其中一个数组的词频，再遍历另一个
    数组，每命中一个"计数仍大于 0"的值就收集下来并把计数减一——用"减一"这个操作天然保证了
    "取较小的重复次数"这个约束，不需要额外比较两边的计数谁更小。
    【复杂度】时间 O(n + m)，空间 O(min(n, m))（对较短的数组建计数表更省空间，但两种写法都
    正确）。
    【易错点】1) 如果用 set 求交集（`set(nums1) & set(nums2)`）会丢失"重复次数"这个要求，
    比如 nums1=[1,2,2,1]、nums2=[2,2] 用 set 求交集只会得到 {2}，丢了应该出现两次的信息；
    2) 命中后要把计数减一，忘记这一步会导致重复次数没有上限地重复收集；3) 结果顺序天然跟随
    第二个数组遍历的顺序，如果题目/测试要求顺序不敏感，比较时要用 sorted() 而不是直接比较
    list。
    """
    count = Counter(nums1)
    res: list[int] = []
    for x in nums2:
        if count[x] > 0:
            res.append(x)
            count[x] -= 1
    return res


def max_points(points: list[list[int]]) -> int:
    """
    【题意】给定平面上若干个点，找出在同一条直线上的最多点数。
    【思路】枚举每个点作为"参考点"，对其余每个点计算它相对参考点的斜率，用哈希表把"斜率
    相同的点"分到同一组——同一组里的点数（再加上参考点自己）就是经过参考点、沿这个斜率方向的
    共线点数，取所有参考点、所有斜率里的最大值就是答案。核心难点全部在"怎么把斜率当哈希表的
    key"（见下面深挖）。
    【复杂度】时间 O(n^2)（外层枚举参考点 O(n)，内层对其余点算斜率并查表 O(n)），空间
    O(n)（哈希表最多存 n-1 个斜率）。
    【易错点】1) 直接用浮点数除法算斜率（dy/dx）在数据精度要求高时会因为浮点误差把同一条直线
    上的点误判成不同斜率，必须改用"最简分数"表示（见深挖）；2) 垂直线（dx=0）和水平线（dy=0）
    是斜率表示法的两个边界情况，必须能被同一套归一化逻辑正确处理，不能只针对普通斜率写代码而
    让这两种情况报错或者算错；3) 归一化最简分数时要统一符号（比如强制 dx 为正，dx=0 时强制
    dy 为正），否则 (1,-2) 和 (-1,2) 这两个本该是同一条直线的斜率会被误判成不同的 key。
    """
    n = len(points)
    if n <= 2:
        return n
    best = 1
    for i in range(n):
        slopes: dict[tuple[int, int], int] = defaultdict(int)
        for j in range(n):
            if i == j:
                continue
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            g = math.gcd(dx, dy)
            if g != 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            slopes[(dx, dy)] += 1
        if slopes:
            best = max(best, max(slopes.values()) + 1)
    return best


def is_happy(n: int) -> bool:
    """
    【题意】给定正整数 n，反复将它替换为"各位数字的平方和"，如果最终能变成 1 则是快乐数，
    如果陷入不含 1 的循环则不是；判断 n 是不是快乐数。
    【思路】"是否会陷入循环"本质是一个"判断链表/序列是否有环"的问题，最直接的实现是用哈希
    set 记录所有出现过的中间结果——一旦某个值重复出现（在 set 里已经存在），说明进入了循环
    且这个循环不含 1（因为一旦出现 1，循环条件 n != 1 会直接终止），可以立即判定不是快乐数；
    这是哈希表"记录见过的值以检测重复"这个最基础能力在"检测循环"场景下的应用（也可以像
    LC287 寻找重复数那样用快慢指针做到 O(1) 空间，但哈希 set 的写法更直观）。
    【复杂度】时间 O(log n) 级别的每一步计算 * 循环检测前的步数（可证明有限），空间 O(步数)
    （存储所有出现过的中间结果）。
    【易错点】1) 循环条件必须同时检查 `n != 1` 和 `n not in seen`，只检查其中一个会漏判（比如
    只检查 not in seen 而没有 n != 1 的提前退出，虽然结果一样正确但更绕）；2) 计算"各位数字
    平方和"时容易写成对字符串下标操作出错，用 `int(d) ** 2 for d in str(n)` 更不容易出错；
    3) 别忘了 set 要"先判断是否已存在，再添加当前值"的顺序（这里写成先添加也可以，因为判断
    条件在添加之前已经检查过 while 循环条件，不会误判自身）。
    """
    seen: set[int] = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d) ** 2 for d in str(n))
    return n == 1


def first_uniq_char(s: str) -> int:
    """
    【题意】给定字符串 s，返回第一个不重复字符的下标；不存在则返回 -1。
    【思路】哈希表"计数"用法的最简单直接应用：先用 Counter 统计每个字符出现的总次数，再按
    原字符串顺序遍历一遍，第一个"总次数恰好为 1"的字符下标就是答案——两次遍历都是线性的，
    不需要对每个字符都重新扫描一遍字符串去数它出现了几次（那样是 O(n^2)）。
    【复杂度】时间 O(n)，空间 O(字符集大小)。
    【易错点】1) 容易只做一次遍历、边统计边判断，但这样在统计的当下还不知道这个字符后面会不会
    再出现，必须先统计完整个字符串的计数表，再单独做第二次遍历按顺序找答案；2) "第一个不重复"
    强调的是原字符串里出现的顺序，不是计数表本身的顺序（Python 3.7+ 的 dict/Counter 插入顺序
    恰好和遍历顺序一致，但依赖这个隐式顺序不如显式按原字符串再遍历一遍清晰可靠）；3) 没有找到
    时要返回 -1 而不是 None 或抛异常。
    """
    count = Counter(s)
    for i, ch in enumerate(s):
        if count[ch] == 1:
            return i
    return -1


def four_sum_count(
    nums1: list[int], nums2: list[int], nums3: list[int], nums4: list[int]
) -> int:
    """
    【题意】给定四个长度相同的整数数组，统计有多少组下标 (i, j, k, l)，使得
    nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0（四个下标各自独立选自各自的数组，不要求
    不同数组间下标有任何关系）。
    【思路】暴力解四层循环枚举所有组合是 O(n^4)。关键 insight 是"分组配对"：既然要求的是四数
    之和为 0，可以先把问题拆成两半——用哈希表统计 nums1、nums2 所有两两组合的和出现的次数
    （key 是和的值，value 是出现次数），再遍历 nums3、nums4 的两两组合，对每个和 s，去查表里
    "-s 出现了多少次"并累加——把四层嵌套枚举变成了"预处理一半 O(n^2) + 查表另一半 O(n^2)"，
    彻底避免了真正的四重循环。
    【复杂度】时间 O(n^2)，空间 O(n^2)（前两个数组两两组合的和最多有 n^2 种）。
    【易错点】1) 分组方式是"前两个数组"和"后两个数组"各自内部两两组合，而不是"每个数组各自
    单独计数"，见到这题容易想复杂去对四个数组分别建哈希表，其实只需要建一张"两数之和"的表；
    2) 查表时查的是 `-(c + d)` 而不是 `c + d` 本身，这是"两半的和要抵消为 0"这个约束的直接
    体现，容易漏掉这个负号；3) 这题只要求"个数"，不要求返回具体的下标组合，不需要在哈希表里
    存下标，只存"和 -> 出现次数"即可，存下标反而是不必要的额外开销。
    """
    sum_ab: dict[int, int] = defaultdict(int)
    for a in nums1:
        for b in nums2:
            sum_ab[a + b] += 1
    count = 0
    for c in nums3:
        for d in nums4:
            count += sum_ab.get(-(c + d), 0)
    return count


def _self_test() -> None:
    assert is_isomorphic("egg", "add") is True
    assert is_isomorphic("foo", "bar") is False
    assert is_isomorphic("paper", "title") is True

    assert word_pattern("abba", "dog cat cat dog") is True
    assert word_pattern("abba", "dog cat cat fish") is False
    assert word_pattern("aaaa", "dog cat cat dog") is False

    assert sorted(intersect([1, 2, 2, 1], [2, 2])) == [2, 2]
    assert sorted(intersect([4, 9, 5], [9, 4, 9, 8, 4])) == [4, 9]

    assert max_points([[1, 1], [2, 2], [3, 3]]) == 3
    assert max_points([[1, 1], [3, 2], [5, 3], [4, 1], [2, 3], [1, 4]]) == 4

    assert is_happy(19) is True
    assert is_happy(2) is False

    assert first_uniq_char("leetcode") == 0
    assert first_uniq_char("loveleetcode") == 2
    assert first_uniq_char("aabb") == -1

    assert four_sum_count([1, 2], [-2, -1], [-1, 2], [0, 2]) == 2

    print(
        "[PASS] p03_hashing_ii: 7 题"
        "（同构字符串/单词规律/两个数组的交集II/直线上最多的点数/"
        "快乐数/字符串中第一个唯一字符/四数相加II）全部通过"
    )


if __name__ == "__main__":
    _self_test()
