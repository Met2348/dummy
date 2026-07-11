"""LeetCode 分类 04·字符串：双指针原地处理 + 中心扩展/边界扫描类技巧的 6 道范例。"""
from __future__ import annotations


def reverse_string(s: list[str]) -> None:
    """
    【题意】给定字符列表 s，要求原地反转（不能返回新列表，也不能额外开一个 O(n) 数组）。
    【思路】首尾两个指针相向而行，每一步交换 s[left]/s[right]，再各自向中间靠拢一步，
    直到相遇为止——这是"原地反转"类问题的通用骨架：只要能用下标同时访问首尾并交换，
    就不需要额外空间建新结构。这个骨架之后在链表、数组的原地题里会反复出现。
    【复杂度】时间 O(n)，每个字符只交换一次；空间 O(1)，原地操作不开辟新数组。
    【易错点】1) 循环条件写成 left <= right 会多做一次无意义的自我交换（不算错但多余，
    正确写法是 left < right，相遇或错开就停）；2) 忘记题目要求"无返回值、原地修改"，
    误写成 return s[::-1] 返回新列表，不满足"in-place"的题意（判题会检查原列表是否被修改）。
    """
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1


def longest_common_prefix(strs: list[str]) -> str:
    """
    【题意】给定字符串数组 strs，返回它们共同的最长前缀；不存在公共前缀则返回空字符串 ""。
    【思路】换个角度做"纵向扫描"：把所有字符串按位置对齐（zip(*strs) 相当于把字符串数组
    转置成一列一列），逐列检查这一列上所有字符串的字符是否完全相同；一旦某一列出现不一致，
    前面已收集到的字符就是答案，因为公共前缀只能到这里为止。zip 天然处理了"字符串长度
    不一"的情况——最短的字符串耗尽后 zip 自动停止迭代，不会越界。
    【复杂度】时间 O(S)，S 为所有字符串长度之和；空间 O(1)（不计返回值）。
    【易错点】1) 判断"这一列字符是否全相同"不能只跟第一个字符串的字符比，要看这一列
    所有字符的集合大小是否为 1（set(chars) 的大小）；2) strs 为空列表时 zip(*strs) 等价于
    zip()（没有任何可迭代对象参与），会直接得到空迭代器，循环一次都不执行，自然返回 ""——
    想清楚这一点，就不需要为空列表单独写 if 分支。
    """
    result: list[str] = []
    for chars in zip(*strs):
        if len(set(chars)) > 1:
            break
        result.append(chars[0])
    return "".join(result)


def longest_palindrome(s: str) -> str:
    """
    【题意】给定字符串 s，返回其中最长的回文子串（只要求返回一个满足条件的答案，不要求唯一）。
    【思路】回文串的关键性质是"以某个中心对称"。枚举每一个可能的中心，向两边扩展，
    只要左右字符相等就继续扩，直到不等或越界为止——这就是"中心扩展法"。难点在于回文的
    中心有两种：奇数长度回文的中心是单个字符（如 "aba" 的中心是 'b'），偶数长度回文的
    中心在两个字符之间的空隙（如 "abba" 的中心在两个 'b' 中间，不对应任何单个下标）。
    所以每个下标 i 都要分别以 (i, i)（奇中心）和 (i, i+1)（偶中心）为起点各扩展一次，
    取两者中更长的那个，缺一种中心就会漏掉对应奇偶性的回文答案。
    【复杂度】时间 O(n^2)（n 个中心，每个中心最坏扩展 O(n)）；空间 O(1)（不计返回值）。
    【易错点】1) 只做奇数中心扩展，漏掉偶数长度的回文（比如 "cbbd" 的正确答案 "bb" 是
    偶中心扩展出来的，纯奇中心扩展会漏掉）；2) 扩展函数里 while 循环退出时 left/right
    已经"多走一步"（指向不满足条件或越界的位置），所以子串应该是 s[left+1:right]，
    写成 s[left:right] 会少一个字符或者在负下标处出错。
    """
    if not s:
        return ""

    def expand(left: int, right: int) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    best = ""
    for i in range(len(s)):
        for cand in (expand(i, i), expand(i, i + 1)):
            if len(cand) > len(best):
                best = cand
    return best


def reverse_words(s: str) -> str:
    """
    【题意】给定字符串 s（可能有前导/尾随空格，单词间可能有多个连续空格），按单词为单位
    整体反转顺序输出，且输出里单词间只保留单个空格、首尾不留空格。
    【思路】"先拆词、再倒序、再拼接"：Python 的 s.split()（不传参数）会自动按任意长度的
    连续空白分割，并且自动丢弃首尾空白和中间产生的空字符串——这一个内建行为直接消掉了
    本题一半的边界处理，不需要自己手写"跳过连续空格"的状态机。拿到干净的单词列表后，
    倒序拼接即可得到答案。
    【复杂度】时间 O(n)，空间 O(n)（存中间的单词列表）。
    【易错点】1) 如果手写 s.split(" ")（传入单个空格作为分隔符），连续空格处会产生空
    字符串 ""，必须显式过滤掉，否则 "a good   example" 这种输入里的空字符串会污染结果；
    用无参 split() 可以直接规避这个坑；2) 拼接时要用单个空格 " ".join(...)，而不是原样
    保留输入里的多个空格。
    """
    words = s.split()
    return " ".join(reversed(words))


def convert(s: str, num_rows: int) -> str:
    """
    【题意】把字符串 s 按 "Z 字形"排布在 num_rows 行上（先竖直向下走满一列，再斜向右上
    走到下一列的顶部，如此反复），然后按行从上到下、每行从左到右读出拼成新字符串。
    【思路】不需要真的建一个二维网格来模拟排布，只需要维护"当前正处在第几行"这一个状态：
    用一个方向变量 direction 控制行号在 0 和 num_rows-1 之间来回折返（走到最顶或最底就
    反向），每读一个字符就把它追加到对应行号的桶（列表）里，最后把所有行的桶依次拼接起来
    就是答案。这是"用一个额外的状态变量代替显式二维结构"的典型技巧。
    【复杂度】时间 O(n)，空间 O(n)（n 个桶存下所有字符，加上输出）。
    【易错点】1) num_rows == 1 时不存在"折返"这回事（只有一行），如果不特判会导致
    direction 逻辑在只有一行时反复触发、产生错误的行为，必须直接特判返回 s 本身；
    2) 折返方向的判断容易写反：应该是"行号走到 0 就把方向切为向下（+1），行号走到
    num_rows-1 就把方向切为向上（-1）"，而不是"行号等于 0 就一直向下、等于末尾就一直向上"
    这种描述本身没错但代码里判断的时机（在追加字符之后、还是之前）容易搞混导致行号越界。
    """
    if num_rows == 1 or num_rows >= len(s):
        return s

    rows: list[list[str]] = [[] for _ in range(num_rows)]
    row, direction = 0, 1
    for ch in s:
        rows[row].append(ch)
        if row == 0:
            direction = 1
        elif row == num_rows - 1:
            direction = -1
        row += direction
    return "".join("".join(r) for r in rows)


def my_atoi(s: str) -> int:
    """
    【题意】实现字符串转 32 位有符号整数：跳过前导空白 → 读取可选的一个 + 或 - 号 →
    读取尽可能多的连续数字字符；只要遇到第一个非数字字符（或字符串结束）就停止转换；
    转换结果如果超出 [-2^31, 2^31-1]，就截断到对应的边界值；如果没有任何数字可转换
    （包括空串、纯空格、只有符号没有数字），返回 0。
    【思路】这题算法本身没有难度（就是逐字符顺序扫描），真正的难点是**把题目描述里隐藏的
    每一条边界规则都翻译成一个显式的 if**——这正是工业界代码审查最容易挑出漏洞的一类题：
    "看起来是道简单题，写的人往往漏掉一半的隐藏条件"。按题目描述的顺序，依次消耗字符串
    就不会漏：先吃掉空格 → 再吃掉一个符号（如果有）→ 再吃数字直到遇到非数字为止（这一步
    里要顺带做溢出检测）→ 最后应用符号并夹紧到 32 位范围。
    【复杂度】时间 O(n)，空间 O(1)。
    【易错点】（本题的"坑"本身就是考点，逐条列出）
    1) 空字符串或纯空格：跳过空格后已经到达字符串末尾，此时要直接返回 0，而不能继续
    对越界下标取值；
    2) 符号位：只允许紧跟在前导空格之后出现一次 + 或 -，出现在数字中间的符号不合法
    （比如 "1-1" 应该只取到 "1" 就停止，因为第二个 '-' 出现在数字扫描阶段的非数字字符）；
    3) 非数字字符要立即截断转换：一旦遇到数字以外的字符（包括小数点、字母），立刻停止
    读数字，不是报错也不是跳过继续找下一个数字段；
    4) 溢出截断：Python 的整数没有固定位宽，不会像其他语言那样自动溢出，所以必须在
    每读入一位数字之后就手动检查当前值是否已经超出 [-2^31, 2^31-1]，一旦超出立刻返回
    对应的边界值，而不是等读完整个数字串再检查（数字串本身可能远超 32 位，检查太晚
    没有意义，但由于 Python 大整数不会报错，这一步很容易被遗漏）；
    5) 只有符号没有数字（如 "+"、"   -"、"   "）：视为没有任何有效数字转换，返回 0。
    """
    i, n = 0, len(s)
    INT_MIN, INT_MAX = -2**31, 2**31 - 1

    while i < n and s[i] == " ":
        i += 1

    if i == n:
        return 0

    sign = 1
    if s[i] in "+-":
        if s[i] == "-":
            sign = -1
        i += 1

    num = 0
    has_digit = False
    while i < n and s[i].isdigit():
        has_digit = True
        num = num * 10 + int(s[i])
        i += 1
        if num > 2**31:            # 早停：数字位数很多时没必要继续无限增长下去，
            break                   # 因为不论正负号，之后都会被下面的夹紧逻辑截断

    if not has_digit:
        return 0

    num = sign * num
    if num < INT_MIN:
        return INT_MIN
    if num > INT_MAX:
        return INT_MAX
    return num


def _self_test() -> None:
    s = ["h", "e", "l", "l", "o"]
    reverse_string(s)
    assert s == ["o", "l", "l", "e", "h"]

    assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
    assert longest_common_prefix(["dog", "racecar", "car"]) == ""

    assert longest_palindrome("babad") in ("bab", "aba")
    assert longest_palindrome("cbbd") == "bb"

    assert reverse_words("the sky is blue") == "blue is sky the"
    assert reverse_words("  hello world  ") == "world hello"
    assert reverse_words("a good   example") == "example good a"

    assert convert("PAYPALISHIRING", 3) == "PAHNAPLSIIGYIR"
    assert convert("PAYPALISHIRING", 4) == "PINALSIGYAHRPI"
    assert convert("A", 1) == "A"

    assert my_atoi("42") == 42
    assert my_atoi("   -42") == -42
    assert my_atoi("4193 with words") == 4193
    assert my_atoi("words and 987") == 0

    print(
        "[PASS] p04_strings: 6/6 题通过 "
        "(反转字符串/最长公共前缀/最长回文子串/反转单词/Z字形变换/atoi)"
    )


if __name__ == "__main__":
    _self_test()
