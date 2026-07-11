"""数组与双指针 · Phase 3 竞赛级补充：8 道偏 Hard / 易被想复杂的双指针与原地处理变体，不重讲基础框架。"""
from __future__ import annotations


def check_possibility(nums: list[int]) -> bool:
    """
    【题意】LC665.非递减数列。给定数组 nums，判断最多修改 1 个元素能否使其变为非递减
    （nums[i] <= nums[i+1] 对所有 i 成立）。
    【思路】一次线性扫描找"逆序对"（nums[i] > nums[i+1]）。核心难点不在"发现逆序"，而在
    "发现逆序后该改左边那个还是右边那个"：如果 i==0（没有更左的元素约束）或者
    nums[i-1] <= nums[i+1]（把 nums[i] 降到 nums[i+1] 不会破坏和更左元素的关系），就降低
    nums[i]；否则说明降低 nums[i] 会导致 nums[i-1] > nums[i]，只能反过来把 nums[i+1] 抬高到
    nums[i]。全程最多允许出现一次逆序，出现第二次直接判否。
    【复杂度】时间 O(n)，空间 O(1)（原地判断，未必真的需要修改数组，但为了让后续比较用
    修改后的值，实现里仍会写回 nums）。
    【易错点】1) 判断"该降左边还是抬右边"时忘记特判 i==0，会在数组开头就越界访问
    nums[i-1]；2) 抬高/降低操作必须真的写回 nums[i]/nums[i+1]，否则后续的比较仍然用旧值，
    会误判后面本不冲突的位置；3) 恰好只允许 1 次修改，第二次出现逆序要立刻返回 False，
    不能"多次修改也不超过 2 次就放行"。
    """
    n = len(nums)
    modified = 0
    for i in range(n - 1):
        if nums[i] <= nums[i + 1]:
            continue
        modified += 1
        if modified > 1:
            return False
        if i == 0 or nums[i - 1] <= nums[i + 1]:
            nums[i] = nums[i + 1]
        else:
            nums[i + 1] = nums[i]
    return True


def find_unsorted_subarray(nums: list[int]) -> int:
    """
    【题意】LC581.最短无序连续子数组。找出最短的连续子数组，只要把这一段排序，整个数组就
    变成非递减；若数组本身已经有序，返回 0。
    【思路】排序整个数组再逐位比较能找到左右边界，但那是 O(n log n)。O(n) 做法用两次
    单向扫描抓住"边界的本质"：从左到右维护当前最大值 max_seen，只要 nums[i] < max_seen
    （出现了"比前面某个数还小"的值），就说明位置 i 一定在无序区间内，不断更新右边界 right
    为最新命中的 i；同理从右到左维护当前最小值 min_seen，只要 nums[i] > min_seen，就更新
    左边界 left。最终 [left, right] 就是最短无序区间。
    【复杂度】时间 O(n)（两次线性扫描），空间 O(1)。
    【易错点】1) right 一定要取"最后一次命中"的下标而不是"第一次命中"，因为无序区间的右边界
    可能被后面更远的乱序元素继续往右推；left 同理要取从右往左扫时"最后一次命中"（也就是
    整体最靠左的命中位置）；2) 数组已经有序时 right 会一直保持初始值 -1，必须用它作为"是否
    存在无序区间"的哨兵直接返回 0，而不是用 left/right 相减可能得到负数或错误值；3) 两次扫描
    的比较对象不同——一个用"迄今最大值"判断"是否变小了"，一个用"迄今最小值"判断"是否变大
    了"，两者不能搞反方向。
    """
    n = len(nums)
    max_seen, right = float("-inf"), -1
    min_seen, left = float("inf"), -1
    for i in range(n):
        max_seen = max(max_seen, nums[i])
        if nums[i] < max_seen:
            right = i
    for i in range(n - 1, -1, -1):
        min_seen = min(min_seen, nums[i])
        if nums[i] > min_seen:
            left = i
    return right - left + 1 if right != -1 else 0


def find_disappeared_numbers(nums: list[int]) -> list[int]:
    """
    【题意】LC448.找到所有数组中消失的数字。nums 长度为 n，每个元素在 [1, n] 内（可能重复），
    要求 O(n) 时间、O(1) 额外空间（不算返回值）找出 [1, n] 中没有出现过的所有整数。
    【思路】既然值域和下标范围同阶（都是 1..n），可以把数组自己当哈希表用：对每个值 x，把
    下标 x-1 处的值取反（标记"值 x 出现过"），取反前先判断是否已经是负数，避免重复取反抵消
    标记。走完一遍之后，凡是仍然保持正数的下标 i，说明 i+1 这个值从未被标记过，也就是消失的
    数字。
    【复杂度】时间 O(n)，空间 O(1)（返回值不计入，原地复用输入数组）。
    【易错点】1) 遍历时读到的 nums[i] 可能已经被前面的操作取反过，必须用 abs(x) 还原出真实的
    值再算下标，直接用负数当下标会访问到错误位置；2) 这个算法会真正修改输入数组（把部分值变
    成负数），如果调用方后续还需要用原数组，要额外做一次取绝对值恢复，或提前拷贝；3) 收集结果
    时判断条件是 `nums[i] > 0`（该位置从未被标记为"出现过"），返回的是 `i + 1` 而不是 `i`，
    这里有一个下标到数值的偏移，容易漏掉 +1。
    """
    n = len(nums)
    for x in nums:
        idx = abs(x) - 1
        if nums[idx] > 0:
            nums[idx] = -nums[idx]
    return [i + 1 for i in range(n) if nums[i] > 0]


def max_chunks_to_sorted(arr: list[int]) -> int:
    """
    【题意】LC769.最多能完成排序的块。arr 是 [0, n-1] 的一个排列，把它切成若干连续块，每块
    内部排序后拼接起来要得到完全升序的数组，求最多能切成几块。
    【思路】因为 arr 是 0..n-1 的排列，"排序后第 i 个位置应该是数值 i"这件事天然成立。维护
    从左到右扫描时"迄今为止见过的最大值" max_seen：只要 max_seen == i（当前下标），就说明
    [0, i] 这一段包含的数值恰好就是 {0, 1, ..., i} 这个集合（不多不少），因为如果这一段里有
    任何大于 i 的数混进来，max_seen 就会超过 i；这意味着这一段可以自成一块、排序后正好落在
    正确位置，不会影响后面的块。每次命中 max_seen == i 就切一刀。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】1) 这个贪心解法只对"无重复的排列"成立（值域恰好是 0..n-1 且不重复），如果数组
    有重复值就必须换成 LC768 的单调栈解法，直接套用本题解法会得到偏大的错误答案；2) 判断条件
    是 `max_seen == i`（下标从 0 开始），而不是 `max_seen == i + 1` 或"计数是否等于 i+1"，
    下标和"已经见过的元素个数"这两个概念在这题里恰好数值相等，容易搞混但不能省略这层验证；
    3) 不需要真的执行排序或切割操作，只需要计数满足条件的次数即可。
    """
    chunks = 0
    max_seen = 0
    for i, v in enumerate(arr):
        max_seen = max(max_seen, v)
        if max_seen == i:
            chunks += 1
    return chunks


def max_chunks_to_sorted_ii(arr: list[int]) -> int:
    """
    【题意】LC768.最多能完成排序的块 II。是 LC769 的加强版：arr 不再保证是 0..n-1 的排列，
    可以是任意整数（含重复、含较大值域），其余规则不变，求最多能切成几块。
    【思路】LC769 的"max_seen == 下标"技巧在这里失效（值域和下标不再一一对应）。改用单调
    不减的单调栈，栈里保存"每个已切分块的最大值"：遍历到新元素 v 时，如果 v 大于等于栈顶
    （当前最后一块的最大值），说明 v 可以独立成为新的一块（栈顶 push v）；如果 v 小于栈顶，
    说明 v 必须被"吸收"进前面某一块——不断弹出所有比 v 大的块最大值（因为这些块必须和 v 合并
    成一块，否则块间最大值不满足"前一块最大值 <= 后一块最小值"这个排序后仍能保持有序的条件），
    合并后把"这些块的最大值"（也就是弹出前的栈顶，因为它是这些块里最大的）重新 push 回去代表
    合并后的新块。最终栈的长度就是能切分的最大块数。
    【复杂度】时间 O(n)（每个元素最多入栈出栈一次），空间 O(n)（最坏情况栈里存 n 个块）。
    【易错点】1) 弹出循环的条件是"当前值 v 小于栈顶"，弹出的是所有比 v 大的历史块最大值，
    合并后重新入栈的值是"弹出前的第一个栈顶"（也就是这些被合并块里最大的那个），而不是 v 本身
    ——因为合并后这一块的最大值仍然是原来那些块的最大值，不会因为并入了更小的 v 而变小；
    2) 判断条件是"严格小于"（v < 栈顶）还是"小于等于"要看清楚：这里允许 v 等于栈顶时独立成块
    （因为相等值排序后顺序不影响"整体有序"），只有严格小于才需要合并；3) 这题的输入不再要求
    是排列，元素范围可以很大（题目给到 1e8），不能像 LC769 那样用下标做启发式判断。
    """
    stack: list[int] = []
    for v in arr:
        if stack and v < stack[-1]:
            cur_max = stack.pop()
            while stack and v < stack[-1]:
                stack.pop()
            stack.append(cur_max)
        else:
            stack.append(v)
    return len(stack)


def max_profit_assignment(difficulty: list[int], profit: list[int], worker: list[int]) -> int:
    """
    【题意】LC826.安排工作以达到最大收益。difficulty[i]、profit[i] 分别是第 i 份工作的难度和
    报酬，worker[j] 是第 j 个工人的能力值（只能胜任难度 <= worker[j] 的工作）。每个工人最多
    做一份工作（同一份工作可以被多个工人重复完成），工人若无法胜任任何工作则收益为 0，求所有
    工人能获得的总收益最大值。
    【思路】把 (difficulty, profit) 打包成 job 对并按难度排序，把 worker 也排序。排序之后，
    工人的能力值单调递增地扫描：对每个工人 w，他能做的工作集合是"难度 <= w 的所有工作"，
    这个集合随着 w 增大只会变大（新纳入更多工作），所以维护一个指针 i 在 job 列表上单调前移
    （从不回退），每前移一步就用一个变量 best 记录"迄今能做的工作里报酬最大值"；工人 w 的最优
    选择就是 best（选报酬最高的那个能做的工作，即使他能力有富余也不会做更低报酬的工作）。这是
    双指针"排序后单调扫描代替嵌套枚举"的模式，把 O(n*m) 的暴力枚举降到 O(n log n + m log m)。
    【复杂度】时间 O(n log n + m log m)（排序为主，n 是工作数，m 是工人数），空间 O(n)（打包
    排序用的 job 列表）。
    【易错点】1) 内层 while 推进 i 的条件是 job 的难度 <= 当前工人能力，而不是 < ，工人能力
    恰好等于某工作难度时应该算作"能胜任"；2) best 只会单调不减，不能在每个工人处重新从 0 算
    起——如果重新计算就退化成了每个工人都要重新扫一遍所有工作的暴力解；3) worker 数组必须先
    排序才能保证指针 i 单调前移不丢解，如果不排序、按原始顺序处理工人，指针就不能保证"之前
    见过的工作对当前工人也都合法"这个单调性。
    """
    jobs = sorted(zip(difficulty, profit))
    worker = sorted(worker)
    total = 0
    best = 0
    i = 0
    n = len(jobs)
    for w in worker:
        while i < n and jobs[i][0] <= w:
            best = max(best, jobs[i][1])
            i += 1
        total += best
    return total


def interval_intersection(
    first_list: list[list[int]], second_list: list[list[int]]
) -> list[list[int]]:
    """
    【题意】LC986.区间列表的交集。给定两个已按起点排序、内部互不重叠的闭区间列表
    firstList、secondList，返回它们所有的交集区间（每个区间 [a, b] 表示 a<=x<=b 的实数集合）。
    【思路】两个列表各自有序且无重叠，天然适合双指针从头各自向后推进：用 i、j 分别指向两个
    列表当前考察的区间，任意时刻交集（如果存在）的左端点是 `max(两区间起点)`，右端点是
    `min(两区间终点)`——若左端点 <= 右端点，说明这两个区间确实有重叠，记录这段交集。判断完
    之后，**结束更早的那个区间不可能再和对方列表里后面的区间产生交集**（它已经耗尽），所以让
    结束更早的那个指针前进；另一个指针留在原地继续和下一个区间比较。
    【复杂度】时间 O(n + m)（n、m 分别是两个列表长度，每步至少推进一个指针），空间 O(1)
    （不计返回值）。
    【易错点】1) 推进指针的依据是"哪个区间的**终点**更小"，而不是"哪个区间的起点更小"——
    起点更小的区间未必先结束（可能范围更长包住了对方好几个区间）；2) 交集判断条件是
    `lo <= hi`（注意允许相等，两个区间可能只在一个点相交，比如 [5,10] 和 [8,5] 这种退化情况
    题目里不会出现，但 [5,10]、[10,12] 会在 10 这一点相交，要用 <= 而不是 <）；3) 循环条件
    是 `i < len(first_list) and j < len(second_list)`，任意一个列表耗尽就应该停止，不能继续
    访问越界下标。
    """
    res: list[list[int]] = []
    i = j = 0
    while i < len(first_list) and j < len(second_list):
        lo = max(first_list[i][0], second_list[j][0])
        hi = min(first_list[i][1], second_list[j][1])
        if lo <= hi:
            res.append([lo, hi])
        if first_list[i][1] < second_list[j][1]:
            i += 1
        else:
            j += 1
    return res


def is_long_pressed_name(name: str, typed: str) -> bool:
    """
    【题意】LC925.长按键入。输入 name 时某些字符可能因为长按被连续重复输入多次，判断 typed
    是否有可能是 name 长按之后打出来的结果。
    【思路】双指针 i、j 分别扫描 name、typed：若当前字符相等，两个指针一起前进（正常敲对了
    一个字符）；若不相等，检查是不是"长按重复"——即 typed[j] 和它的前一个字符 typed[j-1]
    相同（说明这是上一个字符被按长了、多敲出来的），是的话只让 j 前进（消耗掉这个多余的重复
    字符，i 不动）；两种情况都不满足，说明 typed 在这个位置既不匹配 name 也不是合法的重复，
    直接判否。循环结束后还要检查 i 是否恰好走到了 name 末尾——如果 name 还有剩余字符没被
    匹配完，说明 typed 打少了，同样不合法。
    【复杂度】时间 O(n + m)（n、m 分别是两字符串长度），空间 O(1)。
    【易错点】1) "长按重复"的判断必须依赖"typed 内部前后字符相同"（`typed[j]==typed[j-1]`），
    而不是"typed[j] 等于 name 中已匹配的最后一个字符"——用后者在某些边界下会误判；2) 判断
    `typed[j] == typed[j-1]` 之前要先保证 `j > 0`，否则第一个字符不匹配时会误访问负数下标；
    3) 循环结束的最终返回值是 `i == len(name)`，如果只判断"typed 有没有走完"而忘记检查
    name 是否也走完，会把"name 比 typed 长"的非法情况误判为 True。
    """
    i = j = 0
    n, m = len(name), len(typed)
    while j < m:
        if i < n and name[i] == typed[j]:
            i += 1
            j += 1
        elif j > 0 and typed[j] == typed[j - 1]:
            j += 1
        else:
            return False
    return i == n


def _self_test() -> None:
    nums_665a = [4, 2, 3]
    assert check_possibility(nums_665a) is True
    nums_665b = [4, 2, 1]
    assert check_possibility(nums_665b) is False

    assert find_unsorted_subarray([2, 6, 4, 8, 10, 9, 15]) == 5
    assert find_unsorted_subarray([1, 2, 3, 4]) == 0
    assert find_unsorted_subarray([1]) == 0

    assert find_disappeared_numbers([4, 3, 2, 7, 8, 2, 3, 1]) == [5, 6]
    assert find_disappeared_numbers([1, 1]) == [2]

    assert max_chunks_to_sorted([4, 3, 2, 1, 0]) == 1
    assert max_chunks_to_sorted([1, 0, 2, 3, 4]) == 4

    assert max_chunks_to_sorted_ii([5, 4, 3, 2, 1]) == 1
    assert max_chunks_to_sorted_ii([2, 1, 3, 4, 4]) == 4

    assert (
        max_profit_assignment([2, 4, 6, 8, 10], [10, 20, 30, 40, 50], [4, 5, 6, 7]) == 100
    )
    assert max_profit_assignment([1], [1], [0]) == 0

    assert interval_intersection(
        [[0, 2], [5, 10], [13, 23], [24, 25]], [[1, 5], [8, 12], [15, 24], [25, 26]]
    ) == [[1, 2], [5, 5], [8, 10], [15, 23], [24, 24], [25, 25]]
    assert interval_intersection([], [[1, 5]]) == []

    assert is_long_pressed_name("alex", "aaleex") is True
    assert is_long_pressed_name("saeed", "ssaaedd") is False
    assert is_long_pressed_name("leelee", "lleeelee") is True
    assert is_long_pressed_name("laiden", "laiden") is True

    print(
        "[PASS] p01_arrays_two_pointers_iii: 8 题"
        "（非递减数列/最短无序连续子数组/找到所有数组中消失的数字/"
        "最多能完成排序的块/最多能完成排序的块II/安排工作以达到最大收益/"
        "区间列表的交集/长按键入）全部通过"
    )


if __name__ == "__main__":
    _self_test()
