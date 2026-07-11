"""14 贪心 · 竞赛级补充(Part III)：Frontier Lab 面试常考的更高难度贪心变体——
"双指针交换厮杀"(田忌赛马)、"正负号翻转"、"贪心+堆的反悔机制"，不重复 Part I/II
已经讲过的区间贪心、树形贪心框架，只挑竞赛级题目。"""
from __future__ import annotations

import heapq


# ── LC881 救生艇(Medium) ─────────────────────────────────────────────────
def num_rescue_boats(people: list[int], limit: int) -> int:
    """
    【题意】给定每个人的体重 people 和每艘船的载重上限 limit，每艘船最多同时载
    两人（且两人体重之和不超过 limit），求运送所有人最少需要多少艘船。
    【思路】排序后用双指针贪心配对："能不能让最重的人和最轻的人拼一条船"——
    先排序，left 指向最轻的人，right 指向最重的人：如果 people[left] +
    people[right] <= limit，说明最轻的人可以顺路搭最重的人的船，两人一起上船，
    left、right 同时收缩；否则最重的人只能独自坐一条船（他 + 除自己外体重最小
    的人都装不下，那他和任何人都装不下），right 单独收缩。这样贪心成立的原因：
    最重的人无论如何都得坐船，如果他能带走一个人，"带走最轻的那个"不会让整体
    船数变差——如果最重的人不能和最轻的人拼船，那他和任何人都拼不了船，必须
    独占一条。
    【复杂度】时间 O(n log n)（排序主导），空间 O(1)（不计排序开销）。
    【易错点】容易在 people[left] + people[right] <= limit 时忘记同时移动
    left 和 right（这一船同时载了两人，两个指针都要收缩一步）；如果只移动
    right，会把已经上船的 left 重复计算进后续配对。
    """
    people = sorted(people)
    left, right = 0, len(people) - 1
    boats = 0
    while left <= right:
        if left < right and people[left] + people[right] <= limit:
            left += 1
        right -= 1
        boats += 1
    return boats


# ── LC948 令牌放置(Medium) ────────────────────────────────────────────────
def bag_of_tokens_score(tokens: list[int], power: int) -> int:
    """
    【题意】给定令牌数组 tokens 和初始体力 power，初始分数为 0。每个令牌可以
    「正面打出」（消耗体力 tokens[i]、获得 1 分，要求当前体力 >= tokens[i]）或者
    「反面打出」（消耗 1 分、获得体力 tokens[i]，要求当前分数 >= 1），每个令牌
    最多使用一次，可以不用完所有令牌，求能获得的最大分数。
    【思路】排序后双指针贪心："花小钱多办事"——用最小的体力代价换分数，用最大
    的令牌反面套现体力。先排序，lo 指向最小令牌，hi 指向最大令牌：只要当前体力
    够买最小令牌（正面打出 tokens[lo]），就买（花最少的体力获得 1 分最划算）；
    买不起了但已经有分数，就反面打出当前最大的令牌 tokens[hi] 换体力（损失的
    分数固定是 1，但换回的体力最多，最划算）。不断重复直到两指针相遇，或者
    「既买不起最小令牌、又没有分数可以换体力」为止。过程中每次正面打出后都要
    记录分数的历史最大值（反面操作会让分数暂时下降，最终停止时刻的分数不一定
    是历史最高点）。
    【复杂度】时间 O(n log n)（排序主导），空间 O(1)（不计排序开销）。
    【易错点】① 反面操作会让分数变小，必须在每一步「正面买入」之后就更新
    best，不能只在循环结束时看最终分数；② 循环终止条件是「体力不够买最小令牌
    且没有分数可以换体力」，忘记第二个条件会在分数为 0 时误判死循环或提前
    退出。
    """
    tokens = sorted(tokens)
    lo, hi = 0, len(tokens) - 1
    score = 0
    best = 0
    while lo <= hi:
        if power >= tokens[lo]:
            power -= tokens[lo]
            score += 1
            lo += 1
            best = max(best, score)
        elif score > 0:
            power += tokens[hi]
            score -= 1
            hi -= 1
        else:
            break
    return best


# ── LC1005 K次取反后最大化的数组和(Easy) ──────────────────────────────────
def largest_sum_after_k_negations(nums: list[int], k: int) -> int:
    """
    【题意】给定整数数组 nums 和整数 k，每次操作选一个下标 i，把 nums[i] 变成
    -nums[i]，恰好执行 k 次操作（同一下标可以选多次），求操作后数组元素和的
    最大值。
    【思路】排序后贪心：想让和最大，应该优先把「最负」的数变正——负数变正数
    带来的收益是 `2 * abs(值)`，值越负收益越大，所以从最小值开始，只要它还是
    负数且 k 还有剩余，就翻转它。这一步做完后可能出现两种情况：① k 恰好用完；
    ② 数组里已经没有负数了但 k 还有剩余——这时剩下的翻转次数只能反复作用在
    某个数上，而「同一个数翻两次等于没翻」，所以剩余 k 是偶数就直接抵消不用管；
    是奇数则等价于「必须真的翻一次」，这时应该选**绝对值最小**的那个数来翻
    （这样损失最小，因为它对总和的负面贡献 `2 * abs(值)` 最小）。
    【复杂度】时间 O(n log n)（排序主导），空间 O(1)（不计排序开销，原地翻转）。
    【易错点】① 忘记处理「负数提前翻完但 k 仍有剩余」的分支，直接把剩余 k
    次全部作用在最后一个元素上是错的（必须找全局绝对值最小的元素）；② 翻转
    过程中可能把某个负数翻成一个比后面正数还小的正数，绝对值最小的元素不一定
    还停留在原来的位置，稳妥做法是翻转结束后重新扫描整个数组找最小绝对值
    （即 `min(nums)`，此时数组里已无负数，min 就是绝对值最小的元素）。
    """
    nums = sorted(nums)
    i = 0
    n = len(nums)
    while i < n and k > 0 and nums[i] < 0:
        nums[i] = -nums[i]
        k -= 1
        i += 1
    total = sum(nums)
    if k % 2 == 1:
        total -= 2 * min(nums)
    return total


# ── LC870 优势洗牌(Medium) ────────────────────────────────────────────────
def advantage_count(nums1: list[int], nums2: list[int]) -> list[int]:
    """
    【题意】给定两个等长数组 nums1、nums2，nums1 相对 nums2 的「优势」定义为
    满足 nums1[i] > nums2[i] 的下标数量。返回 nums1 的任意一种排列，使得这个
    优势最大化。
    【思路】这是"田忌赛马"的标准变体：把 nums1 排序，把 nums2 的每个值连同
    原始下标一起按值从大到小排序，然后用双指针贪心决策——对 nums2 里当前最大
    的那个值，检查 nums1 里当前最大的候选能不能赢过它：能赢就派它上场（消耗掉
    nums1 里最大的一个，这一局赢了，其它更小的 nums1 值留着以后可能赢更小的
    对手）；不能赢（nums1 最大的都打不过 nums2 当前这个最大值），说明这个
    nums1 最大值这一局注定是"垫底送人头"，与其送一个可能有用的中等值，不如送
    nums1 里最小的那个去无意义地消耗（反正打不过，送小的不浪费大的去赢别的
    对手的机会）。用两个指针 lo、hi 维护 nums1 里还没分配的值。
    【复杂度】时间 O(n log n)（两次排序），空间 O(n)（存结果 + 排序 nums2 的
    下标）。
    【易错点】① nums2 必须连同原始下标一起排序，因为最终答案要按 nums2 的
    「原始位置」摆放对应的 nums1 值，直接对 nums2 排序后旧下标信息就丢失了；
    ② 从大到小处理 nums2 时，"打不过就送最小值"这一步容易写成"送次大值"，
    次大值应该被留下来去尝试赢比它小的对手。
    """
    n = len(nums1)
    sorted1 = sorted(nums1)
    order2 = sorted(range(n), key=lambda i: nums2[i], reverse=True)
    result = [0] * n
    lo, hi = 0, n - 1
    for idx in order2:
        if sorted1[hi] > nums2[idx]:
            result[idx] = sorted1[hi]
            hi -= 1
        else:
            result[idx] = sorted1[lo]
            lo += 1
    return result


# ── LC1029 两地调度(Medium) ───────────────────────────────────────────────
def two_city_sched_cost(costs: list[list[int]]) -> int:
    """
    【题意】2n 个人参加面试，costs[i] = [costA, costB] 表示第 i 个人去 A 城市
    /B 城市面试的花费，要求恰好 n 人去 A、n 人去 B，求总花费最小值。
    【思路】如果不考虑「各去一半」这个约束，每个人都应该去花费更低的城市——
    但这会破坏人数平衡。所以贪心的角度换成"改变主意的代价"：假设所有人先都
    被指派去 A，再选择 n 个人「反悔」改去 B。某人改去 B 之后总花费的变化量是
    `costB - costA`，这个值越小（甚至是负数），反悔的性价比越高。因此按
    `costA - costB` 升序排序（等价于按"改去 B 的净收益"从大到小排序），让净
    收益最大的前 n 个人改去 B，其余 n 人留在 A，就能取得全局最小总花费。
    【复杂度】时间 O(n log n)（排序主导，n 为总人数），空间 O(1)（不计排序
    开销）。
    【易错点】容易直接按 costA 或 costB 单独排序（贪心地"谁便宜去哪"），这样
    完全忽略了"必须各去一半"的约束，得到的分配可能一边超员一边不够；正确做法
    必须按"差值"排序，这才是同时兼顾两个方向开销的关键。
    """
    ordered = sorted(costs, key=lambda c: c[0] - c[1])
    n = len(costs) // 2
    total = 0
    for i, (cost_a, cost_b) in enumerate(ordered):
        total += cost_a if i < n else cost_b
    return total


# ── LC1996 游戏中弱角色的数量(Medium) ─────────────────────────────────────
def number_of_weak_characters(properties: list[list[int]]) -> int:
    """
    【题意】每个角色有 [攻击力, 防御力] 两个属性。如果存在另一个角色的攻击力和
    防御力都**严格大于**当前角色，当前角色就是"弱角色"。求弱角色的数量。
    【思路】按攻击力**降序**排序，攻击力相同时按防御力**升序**排序（这个次序
    很关键，原因见易错点）。排序之后从左到右扫描，维护"目前扫描过的角色里出现
    过的最大防御力" max_defense：当前角色如果防御力 < max_defense，说明前面
    一定存在某个角色攻击力 >= 当前角色（排序保证）且防御力严格更大，是弱角色；
    否则更新 max_defense。
    【复杂度】时间 O(n log n)（排序主导），空间 O(1)（不计排序开销）。
    【易错点】攻击力相同的一组角色，必须按防御力**升序**排列，而不是降序——
    否则同一攻击力批次里防御力大的会先被处理并更新 max_defense，导致同一批次
    里防御力小的角色被误判为"弱角色"（但它们攻击力相同，不满足"严格大于"的
    定义，不应该被判定为弱）。
    """
    ordered = sorted(properties, key=lambda p: (-p[0], p[1]))
    weak = 0
    max_defense = 0
    for _, defense in ordered:
        if defense < max_defense:
            weak += 1
        else:
            max_defense = defense
    return weak


# ── LC630 课程表III(Hard) ────────────────────────────────────────────────
def schedule_course(courses: list[list[int]]) -> int:
    """
    【题意】给定课程列表 courses，courses[i] = [duration, lastDay] 表示第 i
    门课需要连续学习 duration 天，且必须在第 lastDay 天（含）之前学完。从第 1
    天开始，同一时间只能学一门课，求最多能学完多少门课。
    【思路】"贪心 + 堆的反悔机制"：按 lastDay 从小到大排序（截止日期早的课要
    优先考虑安排），维护变量 total_time 表示"已选中课程按顺序连续学完总共用掉
    的天数"，以及一个**最大堆**存放已选中课程各自的 duration。对每门新课，先
    「假设」把它也选进来：total_time += duration，把它的 duration 压入堆。
    如果这时候 total_time 超过了这门新课的 lastDay（说明按目前的顺序学不完
    这门课或者更早的某门课），就必须放弃已选课程里耗时**最长**的那一门——把
    堆顶（最大 duration）弹出，total_time 减去它的值。反悔弹出的一定是"最长
    的"，因为耗时最长的课程对总时间的拖累最大，扔掉它换来的时间盈余也最多，
    最有希望让排期重新变得可行；即使被扔掉的恰好是刚放进去的这门新课本身，
    也没关系（相当于试探性纳入后发现不划算就撤销）。全部课程处理完后，堆的
    大小就是最终选中的课程数。
    【复杂度】时间 O(n log n)（排序 + 每门课最多入堆出堆各一次），空间 O(n)。
    【易错点】① 反悔的对象必须是"目前已选课程里耗时最长的"，而不是必须撤销
    刚放进去的这门新课——很多人直接判断"这门新课能不能塞进去"，忽略了"刚学的
    某门早期课程如果特别长，换掉它可能比放弃新课更划算"这一更优选择；② 判断
    条件是 `total_time > lastDay`（严格大于）才触发反悔，等于时说明恰好能在
    截止日当天学完，不需要放弃任何课程。
    """
    heap: list[int] = []
    total_time = 0
    for duration, last_day in sorted(courses, key=lambda c: c[1]):
        total_time += duration
        heapq.heappush(heap, -duration)
        if total_time > last_day:
            total_time += heapq.heappop(heap)  # 堆里存负数，加负数 = 减去最大值
    return len(heap)


def _self_test() -> None:
    assert num_rescue_boats([1, 2], 3) == 1
    assert num_rescue_boats([3, 2, 2, 1], 3) == 3
    assert num_rescue_boats([3, 5, 3, 4], 5) == 4

    assert bag_of_tokens_score([100], 50) == 0
    assert bag_of_tokens_score([100, 200], 150) == 1
    assert bag_of_tokens_score([100, 200, 300, 400], 200) == 2

    assert largest_sum_after_k_negations([4, 2, 3], 1) == 5
    assert largest_sum_after_k_negations([3, -1, 0, 2], 3) == 6
    assert largest_sum_after_k_negations([2, -3, -1, 5, -4], 2) == 13

    assert advantage_count([2, 7, 11, 15], [1, 10, 4, 11]) == [2, 11, 7, 15]
    assert advantage_count([12, 24, 8, 32], [13, 25, 32, 11]) == [24, 32, 8, 12]

    assert two_city_sched_cost([[10, 20], [30, 200], [400, 50], [30, 20]]) == 110
    assert (
        two_city_sched_cost(
            [[259, 770], [448, 54], [926, 667], [184, 139], [840, 118], [577, 469]]
        )
        == 1859
    )

    assert number_of_weak_characters([[5, 5], [6, 3], [3, 6]]) == 0
    assert number_of_weak_characters([[2, 2], [3, 3]]) == 1
    assert number_of_weak_characters([[1, 5], [10, 4], [4, 3]]) == 1

    assert (
        schedule_course([[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]) == 3
    )

    print(
        "[PASS] p14_greedy_iii: 7 题(救生艇/令牌放置/K次取反后最大化数组和/"
        "优势洗牌/两地调度/游戏中弱角色的数量/课程表III)全部通过"
    )


if __name__ == "__main__":
    _self_test()
