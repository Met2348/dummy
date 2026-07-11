"""滑动窗口：右扩找可行解、左缩找最优解的 5 道代表题。"""
from __future__ import annotations

from collections import Counter


def length_of_longest_substring(s: str) -> int:
    """
    【题意】给定字符串 s，返回其中不含重复字符的最长子串的长度。
    【思路】暴力解枚举所有子串再逐一检查是否有重复字符，O(n^3)（用 set 检查也要 O(n^2)）。
    关键 insight：合法子串的判断条件"窗口内无重复字符"具有单调性——如果 [left, right] 已经
    无重复，加入 s[right+1] 只可能因为这一个新字符造成重复，完全不需要重新扫描整个窗口；
    用一个哈希表记录"字符→最近一次出现的下标"，右指针每扩张一步，只需要把 left 跳到
    "重复字符上次出现位置 + 1"，全程 left 只增不减，均摊 O(n)。
    【复杂度】时间 O(n)，空间 O(min(n, 字符集大小))。
    【易错点】1) 判断重复时要检查"上次出现的下标是否 >= left"，否则会把已经滑出窗口的旧记录
    也算进来，错误地收缩 left（例如 "abba"：第二个 a 出现时若不检查范围，会被 b 的旧记录误导）；
    2) left 只能前进不能后退，容易写成 left = seen[ch] + 1 却忘了和当前 left 取 max；
    3) 空字符串要能正确返回 0（本写法自然处理，但写测试时容易漏掉这个边界）。
    """
    seen: dict[str, int] = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        best = max(best, right - left + 1)
    return best


def min_sub_array_len(target: int, nums: list[int]) -> int:
    """
    【题意】给定正整数 target 和全为正整数的数组 nums，找出和 >= target 的最短连续子数组的
    长度；不存在则返回 0。
    【思路】因为数组元素全为正数，"窗口和"随右端点扩张单调不减、随左端点收缩单调不增——这个
    单调性正是滑动窗口能用在这里的前提（若有负数，收缩左边界不一定能让和变小，就不能这样贪心
    收缩）。于是右指针不断扩张窗口、累加和；只要当前窗口和已经 >= target（找到一个可行解），
    就不断收缩左边界寻找这个右端点下的最短窗口（最优解），直到窗口和 < target 为止，收缩过程
    中不断更新全局最短长度。
    【复杂度】时间 O(n)（left、right 都只增不减，均摊线性），空间 O(1)。
    【易错点】1) 必须在"和仍然 >= target"时才继续收缩左边界，一旦不满足就要停止，而不是收缩
    固定次数；2) 初始最短长度要设成一个不可能达到的大值（如 n+1），最后用它是否被更新过来决定
    返回 0 还是具体长度；3) 这个算法依赖"元素全为正"，若数组里有 0 或负数就不再成立。
    """
    n = len(nums)
    left = 0
    total = 0
    best = n + 1
    for right in range(n):
        total += nums[right]
        while total >= target:
            best = min(best, right - left + 1)
            total -= nums[left]
            left += 1
    return best if best <= n else 0


def find_anagrams(s: str, p: str) -> list[int]:
    """
    【题意】给定字符串 s 和 p，找出 s 中所有恰好是 p 的字母异位词（同样的字符、任意顺序）的
    子串起始下标，按升序返回。
    【思路】p 的异位词一定和 p 等长，所以窗口大小固定为 len(p)——不需要像前两题那样动态伸缩，
    而是"定长窗口平移"：先统计 p 的字符频次和窗口初始 len(p) 个字符的频次，之后每平移一步，
    只需要把移出窗口的字符计数减一、移入窗口的字符计数加一（O(1) 更新），再比较两个频次表
    是否相等，而不必每次都重新统计整个窗口。
    【复杂度】时间 O(n)（n = len(s)，每步窗口更新和比较均摊 O(字符集大小)），空间 O(字符集大小)。
    【易错点】1) 窗口长度必须严格等于 len(p)，s 比 p 短时要直接返回空列表；2) 用 Counter 手写
    增减计数时，计数变成 0 后如果不从字典里删掉，比较两个 Counter 是否相等依然正确（Counter 的
    __eq__ 会忽略值为 0 的键），但如果换成手写 dict 比较就必须记得清理零值，否则会误判不相等；
    3) 平移窗口"先移出旧字符、再移入新字符"的顺序要保持一致，避免窗口大小算错。
    """
    n, m = len(s), len(p)
    if n < m:
        return []
    need = Counter(p)
    window = Counter(s[:m])
    res: list[int] = []
    if window == need:
        res.append(0)
    for right in range(m, n):
        left_char = s[right - m]
        window[left_char] -= 1
        if window[left_char] == 0:
            del window[left_char]
        right_char = s[right]
        window[right_char] += 1
        if window == need:
            res.append(right - m + 1)
    return res


def check_inclusion(s1: str, s2: str) -> bool:
    """
    【题意】判断 s2 中是否存在一个连续子串，恰好是 s1 的某个排列（字符集合和每个字符的出现
    次数都相同）。
    【思路】和"找字母异位词"是同一个模型——排列不改变字符频次，只需要在 s2 上用长度固定为
    len(s1) 的窗口平移，比较窗口内字符频次是否等于 s1 的字符频次；这里只需要"是否存在"，
    找到第一个满足的窗口就能立刻返回 True，不需要跑完整个 s2。
    【复杂度】时间 O(n)（n = len(s2)），空间 O(字符集大小)。
    【易错点】1) 容易忘记先判断 len(s1) > len(s2) 时直接返回 False；2) 这题只要"存在一个"就
    返回，若把上一题的"收集所有下标"整个复用再判断列表非空，正确但多做了不必要的工作；
    3) 复用滑动窗口模板时注意变量名对应关系不要接反——谁是"要匹配的模板"（s1），谁是
    "被搜索的文本"（s2）。
    """
    n1, n2 = len(s1), len(s2)
    if n1 > n2:
        return False
    need = Counter(s1)
    window = Counter(s2[:n1])
    if window == need:
        return True
    for right in range(n1, n2):
        left_char = s2[right - n1]
        window[left_char] -= 1
        if window[left_char] == 0:
            del window[left_char]
        right_char = s2[right]
        window[right_char] += 1
        if window == need:
            return True
    return False


def min_window(s: str, t: str) -> str:
    """
    【题意】给定字符串 s 和 t，找出 s 中涵盖 t 所有字符（含重复次数）的最短连续子串；不存在
    则返回空字符串。
    【思路】和"无重复字符的最长子串"结构相同——右扩找"可行解"、左缩找"最优解"——但收缩条件
    完全不同：那一题的收缩条件是"窗口内出现了重复字符就要缩"，这一题的收缩条件是"窗口已经
    覆盖了 t 的所有字符，才尝试继续缩小看能不能更短"。用一个计数器记录 t 里每个字符还缺多少个，
    用 missing 变量记录"还有多少个字符的需求没被满足"；每次右扩满足了一个字符的需求就
    missing -= 1，missing == 0 时说明窗口已经"可行"，此时才开始收缩左边界，直到收缩会让
    missing 重新变为正数为止，期间不断更新最短窗口的起止位置。
    【复杂度】时间 O(n + m)（n=len(s), m=len(t)，每个字符最多进出窗口一次），空间 O(字符集大小)。
    【易错点】1) t 中可能有重复字符（如 "aa"），必须按次数覆盖而不是"字符集合覆盖"，要用计数
    而不是 set；2) 收缩左边界的时机极易写反：必须先确认 missing == 0（已可行）才开始收缩，
    且一旦收缩会破坏可行性就要停止，而不是无条件收缩到窗口最小；3) 最终答案要记录"起止下标"
    再一次性切片，而不是在过程中拼接字符串；没找到时要返回空字符串而不是 None。
    """
    if not t or not s:
        return ""
    need = Counter(t)
    missing = len(t)
    left = 0
    best_len = len(s) + 1
    best_start = 0
    for right, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        if missing == 0:
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            if right - left + 1 < best_len:
                best_len = right - left + 1
                best_start = left
            need[s[left]] += 1
            missing += 1
            left += 1
    return "" if best_len > len(s) else s[best_start:best_start + best_len]


def _self_test() -> None:
    assert length_of_longest_substring("abcabcbb") == 3
    assert length_of_longest_substring("bbbbb") == 1
    assert length_of_longest_substring("pwwkew") == 3

    assert min_sub_array_len(7, [2, 3, 1, 2, 4, 3]) == 2
    assert min_sub_array_len(4, [1, 4, 4]) == 1
    assert min_sub_array_len(11, [1, 1, 1, 1, 1, 1, 1, 1]) == 0

    assert find_anagrams("cbaebabacd", "abc") == [0, 6]
    assert find_anagrams("abab", "ab") == [0, 1, 2]

    assert check_inclusion("ab", "eidbaooo") is True
    assert check_inclusion("ab", "eidboaoo") is False

    assert min_window("ADOBECODEBANC", "ABC") == "BANC"
    assert min_window("a", "a") == "a"
    assert min_window("a", "aa") == ""

    print("[PASS] p02_sliding_window: 5 题（最长无重复子串/最短子数组/异位词起始下标/排列包含/最小覆盖子串）全部通过")


if __name__ == "__main__":
    _self_test()
