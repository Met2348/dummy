"""16 单调栈 · 进阶补充(Part II)：把单调栈扩展到循环数组、二维矩阵、以及
带"保留几个"约束的贪心选择场景，不重复 Part I 的核心动作，只扩变体。"""
from __future__ import annotations


# ── LC503 下一个更大元素II(Medium) ───────────────────────────────────────
def next_greater_elements(nums: list[int]) -> list[int]:
    """
    【题意】给定一个**循环数组**（最后一个元素的下一个元素是第一个元素），对每个
    位置求"沿着循环方向往右看，第一个比它大的元素"；找不到则记 -1。
    【思路】和 Part I 的"下一个更大元素 I"核心动作一致（维护一个数值从底到顶
    递减的单调栈），唯一的区别是数组是循环的。处理循环的标准技巧是**遍历两倍
    长度**：用 `i % n` 把下标映射回原数组，相当于把数组在逻辑上"接上自己的头"
    再扫一遍。第二轮扫描（i 从 n 到 2n-1）只用来"消费"栈里还没找到答案的元素，
    不需要再把这些下标重新压回栈——因为一个位置的"下一个更大元素"最多在绕一整
    圈的范围内出现，绕完一整圈还没找到就真的没有了。
    【复杂度】时间 O(n)（每个下标最多入栈一次，两轮扫描加起来仍是 O(n) 次
    出栈操作），空间 O(n)。
    【易错点】① 第二轮扫描（i >= n）时不能再把下标压入栈——否则栈会无限增长，
    且会把同一个下标的"下一个更大元素"更新成"绕了不止一圈"之后才遇到的元素，
    错误地扩大了搜索范围；② 忘记用取模 `i % n` 访问数值，直接用 i 索引会越界。
    """
    n = len(nums)
    result = [-1] * n
    stack: list[int] = []  # 存下标，对应数值从底到顶递减
    for i in range(2 * n):
        idx = i % n
        while stack and nums[stack[-1]] < nums[idx]:
            result[stack.pop()] = nums[idx]
        if i < n:
            stack.append(idx)
    return result


# ── LC901 股票价格跨度(Medium) ───────────────────────────────────────────
class StockSpanner:
    """
    【题意】设计一个类，每天调用一次 next(price) 输入当天股价，返回"股票跨度"
    ——从今天开始往前数，连续多少天（包含今天）的股价都 <= 今天的股价。
    【思路】暴力做法是每天都往前扫描直到遇到比今天高的价格为止，最坏 O(n) 每天、
    整体 O(n^2)。观察到：如果昨天的价格 <= 今天，那么"昨天的跨度"能覆盖的那些
    更早的日子，今天必然也能覆盖（因为它们都 <= 昨天的价格 <= 今天的价格）——
    可以直接"继承"昨天算好的跨度，不需要重新逐天扫描。用一个栈存 (价格, 跨度)
    二元组，维护栈内价格从底到顶递减：每来一个新价格，只要栈顶价格 <= 当前价格，
    就弹出栈顶并把它的跨度累加进当前跨度（因为栈顶那一段"历史"已经被今天覆盖
    掉了），重复直到栈顶价格更高或栈空，再把 (今天价格, 累加后的跨度) 压入栈。
    【复杂度】时间 O(1) 均摊每次调用（每天最多入栈一次、出栈一次），空间 O(n)。
    【易错点】① 弹栈条件要用 `<=`（非严格）——如果今天价格和栈顶价格相等，
    栈顶那一段跨度也应该被"继承"进来，写成严格 `<` 会漏算；② 容易忘记"跨度"
    要累加而不是替换——弹出的每一段跨度都要加到当前跨度上，因为它们都是今天
    价格能覆盖到的历史天数。
    """

    def __init__(self) -> None:
        self.stack: list[tuple[int, int]] = []  # (价格, 跨度)，价格从底到顶递减

    def next(self, price: int) -> int:
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        self.stack.append((price, span))
        return span


# ── LC456 132模式(Medium) ────────────────────────────────────────────────
def find132pattern(nums: list[int]) -> bool:
    """
    【题意】给定整数数组 nums，判断是否存在下标 i < j < k，使得
    nums[i] < nums[k] < nums[j]（即"1-3-2"模式：第一个数最小，第二个数最大，
    第三个数居中）。
    【思路】关键是把"寻找中间大小的 nums[k]"和"寻找更早出现的更小 nums[i]"这两
    件事解耦。从**右往左**扫描，维护一个数值从底到顶递减的单调栈（栈里存的是
    "潜在的 3"，也就是模式里的 nums[j] 候选），以及一个变量 candidate_k 表示
    "目前为止能确定的、最适合当模式里 nums[k] 的最大候选值"。每来一个新数 num
    （下标比之前处理过的都靠左）：如果 num < candidate_k，说明我们已经找到了
    合法的 (i, j, k) 三元组——num 就是模式里的 nums[i]（它更靠左、比 candidate_k
    小），直接返回 True。否则，只要栈顶比 num 小，就说明栈顶这个数不可能再作为
    "更大的 nums[j]"候选了（num 比它大，num 才更有资格），于是把它弹出，
    并更新 candidate_k 为这个被弹出的值——因为它虽然当不了 nums[j]，但可以当
    比当前 num 小的"nums[k]"候选（它出现在 num 右边，比 num 小，恰好符合
    nums[j] > nums[k] 的关系,栈顶即num右侧第一个比num大的数nums[j]的候选，
    弹出的是nums[j]和num之间的nums[k]候选，取弹出过程中最大的一个作为
    candidate_k，最有可能被后面更左边的更小值命中）。最后把 num 压入栈。
    【复杂度】时间 O(n)（一次线性扫描，每个元素最多入栈出栈各一次），空间 O(n)。
    【易错点】① candidate_k 的更新必须取"弹出过程中的最大值"，而不是任意一个
    弹出值——因为 candidate_k 越大，后面能命中 `num < candidate_k` 的概率越高，
    这也是为什么可以直接用"最后一次弹出的值"（单调栈弹出的过程本身就是递增的，
    最后弹出的天然是最大的）；② 容易在栈里存下标当值来用，其实这题只需要值
    本身参与比较，不需要下标信息。
    """
    stack: list[int] = []  # 数值从底到顶递减，潜在的"3"
    candidate_k = float("-inf")
    for num in reversed(nums):
        if num < candidate_k:
            return True
        while stack and stack[-1] < num:
            candidate_k = stack.pop()
        stack.append(num)
    return False


# ── LC85 最大矩形(Hard) ──────────────────────────────────────────────────
def _largest_rectangle_area(heights: list[int]) -> int:
    """柱状图中最大矩形面积（84题的标准单调栈解法，供 maximal_rectangle 逐行复用）。"""
    stack: list[int] = []  # 存下标，对应高度从底到顶递增
    best = 0
    for i, h in enumerate(heights + [0]):  # 末尾补哨兵，保证栈最终清空结算
        while stack and heights[stack[-1]] >= h:
            top = stack.pop()
            height = heights[top]
            width = i if not stack else i - stack[-1] - 1
            best = max(best, height * width)
        stack.append(i)
    return best


def maximal_rectangle(matrix: list[list[str]]) -> int:
    """
    【题意】给定一个由字符 '0'/'1' 组成的二维矩阵，求只由 '1' 构成的最大矩形的
    面积。
    【思路】这是"二维问题降维成多个一维子问题"的典型手法：把矩阵的**每一行**都
    看作一张以该行为"地面"的柱状图——柱子 j 的高度是"从当前行往上数，连续有
    多少个 '1'"（如果当前行第 j 列是 '0'，高度直接归零，因为地面出现了缺口，
    往上的 '1' 不可能再和地面相连形成矩形）。逐行累积维护这个高度数组：每处理
    完一行，高度数组就代表"以这一行为底"的柱状图，直接复用 84 题"柱状图中
    最大的矩形"的单调栈解法算一次最大矩形面积。所有行各自算出的最大矩形面积
    取最大值，就是整个二维矩阵里的答案——因为原矩阵里任何一个"全 1 矩形"，
    一定以某一行为它的底边，也就必然会在处理到那一行时，被该行对应的柱状图
    最大矩形算法捕捉到。
    【复杂度】时间 O(rows * cols)（每一行的柱状图求解都是 O(cols)，一共
    rows 行），空间 O(cols)（高度数组 + 单调栈）。
    【易错点】① 高度数组的更新规则容易写反——遇到 '1' 应该是"在上一行高度基础
    上 +1"（累积连续 1 的个数），遇到 '0' 必须**归零**而不是保持不变，因为
    '0' 打断了这一列往上连续为 1 的链条；② 容易漏掉"每一行都要重新跑一次
    最大矩形算法"，而不是只在最后一行跑一次——最大的全 1 矩形不一定是以最后
    一行为底。
    """
    if not matrix or not matrix[0]:
        return 0
    cols = len(matrix[0])
    heights = [0] * cols
    best = 0
    for row in matrix:
        for j in range(cols):
            heights[j] = heights[j] + 1 if row[j] == "1" else 0
        best = max(best, _largest_rectangle_area(heights))
    return best


# ── LC1856 子数组最小乘积的最大值(Medium) ────────────────────────────────
def max_sum_min_product(nums: list[int]) -> int:
    """
    【题意】定义一个数组的"最小乘积"为：数组中的最小值 乘以 数组所有元素之和。
    给定 nums，求它所有**连续子数组**中，最小乘积的最大值（结果对 1e9+7 取模）。
    【思路】枚举每个元素作为"这个子数组的最小值"：如果 nums[i] 是某个子数组的
    最小值，那么这个子数组能扩展到的范围，左边界是"左边第一个比 nums[i] 小的
    元素"的右侧一格，右边界是"右边第一个比 nums[i] 小的元素"的左侧一格——超出
    这个范围，nums[i] 就不再是最小值了。而"以 nums[i] 为最小值能取到的最大乘积"
    一定是让子数组尽量扩展到这个最大范围（子数组和只会更大，乘积也只会更大，
    不会因为扩展了范围而让最小值变化，因为范围边界正是由"最小值改变"这个条件
    定义的）。于是用两次单调栈分别预处理每个位置的 left[i]（左边第一个更小元素
    的下标，没有则 -1）和 right[i]（右边第一个更小元素的下标，没有则 n），
    再配合前缀和数组，就能 O(1) 算出"以 nums[i] 为最小值的最大子数组和"，
    乘以 nums[i] 就是候选答案，对所有 i 取最大值。
    【复杂度】时间 O(n)（两次单调栈预处理 + 一次前缀和 + 一次遍历取最大值），
    空间 O(n)。
    【易错点】① 前缀和下标容易偏移出错——子数组 [left[i]+1, right[i]-1] 的和
    应该是 `prefix[right[i]] - prefix[left[i]+1]`（前缀和数组比原数组多一位，
    prefix[x] 表示 nums[0:x] 的和）；② 忘记结果需要对 1e9+7 取模（虽然本文件
    给的测试样例数值都不大，取不取模不影响这几个样例的断言，但完整题目要求
    取模，这里按标准写法保留）。
    """
    mod = 10**9 + 7
    n = len(nums)
    prefix = [0] * (n + 1)
    for i, x in enumerate(nums):
        prefix[i + 1] = prefix[i] + x

    left = [-1] * n
    stack: list[int] = []
    for i in range(n):
        while stack and nums[stack[-1]] >= nums[i]:
            stack.pop()
        left[i] = stack[-1] if stack else -1
        stack.append(i)

    right = [n] * n
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and nums[stack[-1]] >= nums[i]:
            stack.pop()
        right[i] = stack[-1] if stack else n
        stack.append(i)

    best = 0
    for i in range(n):
        total = prefix[right[i]] - prefix[left[i] + 1]
        best = max(best, total * nums[i])
    return best % mod


# ── LC1475 商品折扣后的最终价格(Easy) ────────────────────────────────────
def final_prices(prices: list[int]) -> list[int]:
    """
    【题意】给定商品价格数组，对第 i 件商品：找到右边第一个价格 <= prices[i]
    的商品 j，第 i 件商品最终价格 = prices[i] - prices[j]（找不到则不打折）。
    返回所有商品的最终价格。
    【思路】"右边第一个 <= 当前值的元素"正是单调栈的标准母题——维护一个数值从
    底到顶**递增**的下标栈：新价格 p 如果 <= 栈顶对应的价格，说明栈顶这件商品
    找到了它的折扣来源（就是 p），弹出栈顶并直接结算（栈顶最终价格 -= p），
    重复直到栈顶价格更高或栈空，再把当前下标压入栈。栈里剩下的、从未被弹出的
    下标，说明它们右边没有更低的价格，维持原价。
    【复杂度】时间 O(n)（每个下标最多入栈出栈各一次），空间 O(n)。
    【易错点】① 弹栈条件要用 `<=`（非严格）——价格相等也算"找到折扣"，写成
    严格 `<` 会让相等的情况被错误地跳过；② 结算时容易写反方向，应该是
    "被弹出的商品价格 -= 触发弹出的新价格"，不是反过来。
    """
    result = list(prices)
    stack: list[int] = []  # 存下标，对应价格从底到顶递增
    for i, p in enumerate(prices):
        while stack and prices[stack[-1]] >= p:
            j = stack.pop()
            result[j] = prices[j] - p
        stack.append(i)
    return result


# ── LC1673 找出最具竞争力的子序列(Medium) ────────────────────────────────
def most_competitive(nums: list[int], k: int) -> list[int]:
    """
    【题意】从 nums 中挑出一个长度为 k 的子序列（保持原相对顺序），使得这个
    子序列在字典序上"最具竞争力"（即字典序最小）。
    【思路】单调栈贪心："只要栈顶元素比当前元素大，且删掉栈顶之后剩下的元素
    加上还没扫到的元素仍然凑得够 k 个"，就应该弹出栈顶——因为把一个更大的数
    换成后面出现的更小的数，子序列的字典序只会变小（更靠前的位置数值更小，
    字典序天然更优），这个操作只要"够本"（删掉之后还有足够多元素补齐长度 k）
    就一定不亏。判断"是否够本"的条件是 `len(stack) - 1 + (n - i) >= k`
    （弹出之后栈的剩余长度，加上从当前元素 i 到末尾还没扫描的元素个数，
    必须仍然能凑够 k 个）。扫描结束后，栈里从底到顶就是答案（如果栈内元素数
    小于 k，说明中途没有弹出这么多次，栈会随着"len(stack) < k 就压入"这个
    条件自然被压满到 k）。
    【复杂度】时间 O(n)（每个下标最多入栈出栈各一次），空间 O(k)。
    【易错点】① 弹栈条件必须同时满足"栈顶更大"和"删除后仍凑得够 k 个"两个
    条件，只看第一个会导致删过头，剩下的元素不够填满长度 k；② 只有在
    `len(stack) < k` 时才允许压入新元素——即使新元素本该继续尝试弹栈，一旦
    栈已经满了 k 个，就不能再压入更多（但仍然可能继续弹出更差的旧元素后再压入）。
    """
    stack: list[int] = []
    n = len(nums)
    for i, x in enumerate(nums):
        while stack and stack[-1] > x and len(stack) - 1 + (n - i) >= k:
            stack.pop()
        if len(stack) < k:
            stack.append(x)
    return stack


def _self_test() -> None:
    assert next_greater_elements([1, 2, 1]) == [2, -1, 2]
    assert next_greater_elements([1, 2, 3, 4, 3]) == [2, 3, 4, -1, 4]

    spanner = StockSpanner()
    assert spanner.next(100) == 1
    assert spanner.next(80) == 1
    assert spanner.next(60) == 1
    assert spanner.next(70) == 2
    assert spanner.next(60) == 1
    assert spanner.next(75) == 4
    assert spanner.next(85) == 6

    assert find132pattern([1, 2, 3, 4]) is False
    assert find132pattern([3, 1, 4, 2]) is True
    assert find132pattern([-1, 3, 2, 0]) is True

    assert (
        maximal_rectangle(
            [
                ["1", "0", "1", "0", "0"],
                ["1", "0", "1", "1", "1"],
                ["1", "1", "1", "1", "1"],
                ["1", "0", "0", "1", "0"],
            ]
        )
        == 6
    )

    assert max_sum_min_product([1, 2, 3, 2]) == 14
    assert max_sum_min_product([2, 3, 3, 1, 2]) == 18
    assert max_sum_min_product([3, 1, 5, 6, 4, 2]) == 60

    assert final_prices([8, 4, 6, 2, 3]) == [4, 2, 4, 2, 3]
    assert final_prices([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
    assert final_prices([10, 1, 1, 6]) == [9, 0, 1, 6]

    assert most_competitive([3, 5, 2, 6], 2) == [2, 6]
    assert most_competitive([2, 4, 3, 3, 5, 4, 9, 6], 4) == [2, 3, 3, 4]

    print(
        "[PASS] p16_monotonic_stack_ii: 7 题(下一个更大元素II/股票价格跨度/"
        "132模式/最大矩形/子数组最小乘积最大值/折扣后最终价格/最具竞争力子序列)"
        "全部通过"
    )


if __name__ == "__main__":
    _self_test()
