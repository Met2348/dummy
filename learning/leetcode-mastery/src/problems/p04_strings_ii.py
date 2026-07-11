"""LeetCode 分类 04·字符串 进阶补充（Part II）：竖式模拟运算 + 贪心两端对齐 +
单调栈去重 + 数字↔罗马数字互转 + 双指针力传导，共 8 道范例，扩大字符串处理的变体覆盖面。"""
from __future__ import annotations


def multiply(num1: str, num2: str) -> str:
    """
    【题意】给定两个以字符串形式表示的非负整数 num1、num2（可能很长，不能假设能安全地
    直接转成内置整数类型来做乘法——这是本题被归为"字符串模拟"而非"数学题"的原因），
    返回它们相乘的结果，同样以字符串表示，且结果不能有前导 0（除非结果本身就是 "0"）。
    【思路】模拟小学竖式乘法："每一位分别相乘、按位对齐、统一处理进位"。用一个长度为
    len(num1)+len(num2) 的数组 result 存放结果的每一位（这个长度是两数相乘结果位数的
    上界：m 位数乘 n 位数，结果最多 m+n 位）。原始字符串下标 i、j（从左往右数）对应的
    两个数字相乘，其结果的个位必然落在 result 的下标 i+j+1 处，可能产生的进位落在
    下标 i+j 处——这是竖式乘法"对位"的核心规律。把所有 (i, j) 组合的乘积都按这个规则
    累加进 result 数组（累加而不是覆盖，因为多组 (i, j) 可能贡献到同一个下标），最后
    每一位只需要对 10 取模、进位加到左边一位即可得到最终数字数组。
    【复杂度】时间 O(m·n)，m、n 分别为 num1、num2 的长度；空间 O(m+n)（结果数组）。
    【易错点】
    1) 下标对应关系容易搞反：这里的 i、j 是从左往右数的原始下标，对应结果数组下标是
    i+j（进位位）和 i+j+1（当前位），而不是凭直觉去对应"个位、十位……"这种从右往左的
    位权，一旦搞反整道题的进位全部错乱；
    2) 结果数组可能带前导 0（比如两个 1 位数相乘，数组长度是 2 但结果只占 1 位），拼接
    成字符串前必须跳过前导 0，同时要保证跳过之后至少保留最后一位（避免把合法的 "0"
    结果跳成空字符串）；
    3) 不能图省事直接 str(int(num1) * int(num2))——虽然 Python 大整数不会溢出，但这样
    完全没有练到"用数组模拟竖式"这个考点本身，遇到"不能使用内置大数运算"的追问变体时
    这条捷径会直接失效。
    """
    if num1 == "0" or num2 == "0":
        return "0"

    m, n = len(num1), len(num2)
    result = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            mul = int(num1[i]) * int(num2[j])
            p_carry, p_cur = i + j, i + j + 1
            total = mul + result[p_cur]
            result[p_cur] = total % 10
            result[p_carry] += total // 10

    start = 0
    while start < len(result) - 1 and result[start] == 0:
        start += 1
    return "".join(map(str, result[start:]))


def full_justify(words: list[str], max_width: int) -> list[str]:
    """
    【题意】给定单词列表 words 和每行最大宽度 max_width，把单词重新排布成两端对齐的
    文本：每行尽量多塞单词，非最后一行的单词间距要用空格补满整行、且空格尽量平均分配
    （左边的间隙比右边多分配 1 个，当不能整除时），只有一个单词的行整体左对齐；最后一行
    只用单个空格分隔单词、整体左对齐、右侧补空格到 max_width。
    【思路】贪心分两步走：第一步"贪心装行"——按顺序把单词往当前行里塞，只要"塞进这个
    单词后，按最少 1 个空格分隔计算出的最小行宽"（已有字符数 + 已有单词数（当作间隙数）
    + 新单词长度）不超过 max_width，就继续塞；一旦超了，当前行就此打住，交给第二步处理。
    第二步"两端对齐"：非最后一行，总空格数 = max_width - 单词字符总长，平均分给
    (单词数-1) 个间隙，除不尽的余数依次分给最左边的几个间隙（这样最左边的间隙最多比
    最右边多 1 个空格，符合"左边空格更多"的规则）；只有一个单词的行和最后一行都统一按
    "单词间单个空格 + 整体左对齐 + 末尾补空格" 处理。
    【复杂度】时间 O(n)，n 为所有字符总数（每个字符最多被访问常数次）；空间 O(n)（存储
    行结构与输出）。
    【易错点】
    1) 判断"当前单词放不放得下"要用"加入这个单词后的最小行宽"提前比较，而不是先加进
    当前行再判断超没超再撤销——用 `已有字符数 + 已有单词数 + 新单词长度 > max_width`
    这个不等式就能在加入之前就做出正确判断；
    2) 只有一个单词的行必须整体左对齐、右侧补空格，不能按"总空格数 ÷ (单词数-1)" 去除，
    分母为 0 会直接报错，必须单独识别这种情况（或者和"最后一行"共用同一段左对齐逻辑）；
    3) 空格分配余数必须给最左边的间隙，越靠左的间隙空格越多——这是最容易写反方向的一步；
    4) 最后一行的规则和其它行完全不同：只能是"单词间单个空格 + 整体左对齐 + 末尾补空格
    到 max_width"，绝不能对最后一行做两端对齐处理，即使它凑巧能塞满整行。
    """
    def justify(line_words: list[str], total_chars: int, is_last: bool) -> str:
        if is_last or len(line_words) == 1:
            return " ".join(line_words).ljust(max_width)
        spaces = max_width - total_chars
        gaps = len(line_words) - 1
        base, extra = divmod(spaces, gaps)
        parts: list[str] = []
        for idx, w in enumerate(line_words[:-1]):
            parts.append(w)
            parts.append(" " * (base + (1 if idx < extra else 0)))
        parts.append(line_words[-1])
        return "".join(parts)

    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        if cur and cur_len + len(cur) + len(w) > max_width:
            lines.append(justify(cur, cur_len, False))
            cur, cur_len = [], 0
        cur.append(w)
        cur_len += len(w)
    lines.append(justify(cur, cur_len, True))
    return lines


def simplify_path(path: str) -> str:
    """
    【题意】给定一个 Unix 风格的绝对路径 path（以 "/" 开头），把它化简成标准形式：
    去掉多余的 "/"、"."（当前目录，忽略）、".."（返回上一级目录），标准路径必须以单个
    "/" 开头，目录之间用单个 "/" 分隔，且不以 "/" 结尾（除非结果就是根目录 "/"）。
    【思路】按 "/" 把路径整体切开，得到的片段里会混杂空字符串（连续 "/" 或首尾 "/"
    产生）、"."、".." 和真正的目录名。用一个栈模拟"当前所在的目录层级"：遇到空字符串
    或 "." 直接跳过（不影响当前层级）；遇到 ".." 就弹出栈顶（返回上一级），但只有栈非空
    才弹（已经在根目录时再出现 ".." 应该原地不动，不能弹出不存在的层级）；其余情况视为
    合法目录名，压入栈。扫描完成后，栈里从栈底到栈顶的顺序就是化简后的路径层级，用
    "/" + "/".join(stack) 拼接即可（栈为空时自然得到根目录 "/"）。
    【复杂度】时间 O(n)，空间 O(n)（栈 + split 产生的中间列表）。
    【易错点】
    1) split("/") 会在连续 "/" 或首尾 "/" 处产生空字符串，必须显式跳过，不能把它们当成
    合法目录名压栈；
    2) ".." 只有在栈非空时才弹出，栈为空时的 ".." 必须原地忽略（保持在根目录），否则会
    抛异常或者产生"负深度"的错误状态；
    3) 判断顺序上 "." 和 ".." 是两种完全不同的处理（前者忽略、后者弹栈），字符串比较时
    容易因为 ".." 里包含 "." 而弄混两者的判断顺序，需要用精确相等比较（`part == "."`
    和 `part == ".."`）而不是用 in 之类的模糊包含判断；
    4) 最终拼接必须用 "/" + "/".join(stack) 这种统一形式，栈为空时它能自动退化为 "/"，
    如果改成对每个目录手动拼接前缀 "/" 再首尾特殊处理，边界情况容易漏掉或多算。
    """
    stack: list[str] = []
    for part in path.split("/"):
        if part == "" or part == ".":
            continue
        if part == "..":
            if stack:
                stack.pop()
        else:
            stack.append(part)
    return "/" + "/".join(stack)


def compare_version(version1: str, version2: str) -> int:
    """
    【题意】给定两个版本号字符串 version1、version2（形如 "1.01.3" ，由若干个用 "."
    分隔的数字段组成，每段可能有前导 0），比较两个版本号的大小：version1 < version2
    返回 -1，相等返回 0，version1 > version2 返回 1。
    【思路】按 "." 把两个版本号分别切成数字段列表，逐段转成整数比较（转成 int 会自动
    去掉前导 0，"01" 和 "1" 转成整数后都是 1，天然解决了"前导 0 不影响数值大小"这个
    要求）。两个版本号的段数可能不同（比如 "1.0" 只有 2 段、"1.0.0" 有 3 段），遍历应该
    以两者中**较长**的段数为准，缺失的段按数值 0 处理——这对应"版本号省略的后续段视为
    0"这一约定（1.0 等价于 1.0.0.0.0...）。逐段比较，一旦某一段不相等就能立刻下结论
    并返回，只有全部段都相等（含补 0 之后）才返回 0。
    【复杂度】时间 O(max(m,n))，m、n 为两个版本号分别的段数；空间 O(m+n)（split 产生的
    中间列表）。
    【易错点】
    1) 不能按字符串直接比较每一段，必须先转成 int 再比较——"01" 和 "1" 字符串不相等，
    但数值相等，直接字符串比较会得出错误的"不相等"结论；
    2) 遍历段数必须取两者中较长的一方（max(len(parts1), len(parts2))），如果只按较短
    的一方遍历，会漏掉较长版本号后面多出来的非零段，比如 "1.1" 和 "1.1.0.1" 只比较到
    第 2 段就会误判为相等，而实际上第 4 段的 1 应该让后者更大；
    3) 缺失的段要按 0 处理，而不是把"段数更少"直接等同于"版本号更小"——"1.0.0" 和
    "1.0" 是相等的，不能因为前者段数更多就武断地判定它更大。
    """
    parts1 = version1.split(".")
    parts2 = version2.split(".")
    n = max(len(parts1), len(parts2))
    for i in range(n):
        v1 = int(parts1[i]) if i < len(parts1) else 0
        v2 = int(parts2[i]) if i < len(parts2) else 0
        if v1 != v2:
            return -1 if v1 < v2 else 1
    return 0


def remove_duplicate_letters(s: str) -> str:
    """
    【题意】给定字符串 s，去除其中重复出现的字母，使每个字母只保留一次；在所有满足
    "每个字母只出现一次、且相对顺序不能颠倒各字母首次引入的相对次序约束" 的结果里，
    返回字典序最小的那一个。
    【思路】单调栈 + "这个字符以后还会不会出现"的贪心判断，是"贪心+单调栈"的经典范式。
    预处理 last_index 记录每个字符最后一次出现的下标；维护一个栈 stack 存当前结果，
    再用 in_stack 记录哪些字符已经在栈里（防止同一个字符被压入多次）。扫描到字符 ch
    时：如果 ch 已经在栈里，直接跳过——它已经占好位置了，题目要求每个字母只留一份；
    否则，只要栈顶字符比 ch 大、并且栈顶字符在后面（下标大于当前 i 的位置）还会再出现，
    就可以放心弹出栈顶——"放心"是因为它以后还有机会重新压回来，现在弹出它能让字典序
    更小的 ch 提前出现在更靠前的位置，让整体字典序变小；反过来，如果栈顶字符在后面
    不会再出现了（这是它最后一次出现的机会），即使它比 ch 大也绝不能弹出，弹出就永久
    丢失了这个字符，导致最终结果里少了一个必须出现的字母。
    【复杂度】时间 O(n)，每个字符最多入栈、出栈各一次；空间 O(Σ)，Σ 为字符集大小（栈和
    两个哈希表的大小都不超过字符集大小，比如全小写字母最多 26）。
    【易错点】
    1) 弹栈条件必须同时满足"栈顶字符更大"和"栈顶字符后面还会出现"两者，只判断其中
    一个都会出错：只看"更大"会把某个字符最后一次出现的机会白白弹掉；只看"后面还会
    出现"而不比较大小，会破坏"结果字典序最小"这个优化目标；
    2) 已经在栈里的字符（in_stack 命中）必须直接跳过整个弹栈判断和入栈操作，不能重复
    压入，否则同一个字母会在结果里出现多次；
    3) last_index 必须在扫描正式开始前，用一次完整遍历预先算好每个字符最后一次出现的
    位置，不能在扫描过程中现算"剩余字符串里是否还有这个字符"（那样每一步都要重新扫一遍
    剩余部分，整体退化成 O(n^2)）。
    """
    last_index = {ch: i for i, ch in enumerate(s)}
    stack: list[str] = []
    in_stack: set[str] = set()

    for i, ch in enumerate(s):
        if ch in in_stack:
            continue
        while stack and stack[-1] > ch and last_index[stack[-1]] > i:
            in_stack.remove(stack.pop())
        stack.append(ch)
        in_stack.add(ch)

    return "".join(stack)


def int_to_roman(num: int) -> str:
    """
    【题意】给定一个 1 到 3999 之间的整数 num，把它转换成对应的罗马数字字符串。
    【思路】贪心：准备一张按数值从大到小排列的"符号-数值"对照表，从表头开始，能减多少
    次就减多少次、每减一次就追加对应的符号，减不动了就换下一档更小的符号继续——这正是
    平时手写罗马数字的直觉过程。关键技巧是把 900(CM)、400(CD)、90(XC)、40(XL)、9(IX)、
    4(IV) 这 6 个"减法"特例，也当作和 1000/500/100/50/10/5/1 同等地位的独立表项放进
    同一张表（一共 13 项），而不是只放 7 个基本符号再另写一套"要不要用减法"的判断逻辑
    ——这样代码里不需要任何 if 特判，从大到小挨个用 divmod 试除即可。
    【复杂度】时间 O(1)（数值范围固定在 3999 以内，符号表固定 13 项，最多做 13 次
    divmod）；空间 O(1)（不计返回值）。
    【易错点】
    1) 符号表必须严格按数值从大到小排列，一旦顺序错乱（比如把 CM 放到 D 后面），贪心
    每一步"能减就减"的前提就不成立，会得到不合法的罗马数字；
    2) 减法特例（CM/CD/XC/XL/IX/IV）必须作为表里独立的一项参与贪心，而不是等基本符号
    处理完之后再"打补丁"式地修正——比如 900 如果不作为独立项，会先被拆成 500(D) +
    100+100+100+100(CCCC)，得到不合法的 "DCCCC" 而不是 "CM"；
    3) 用 divmod(num, v) 同时拿到"这一档符号用几次"和"用完之后剩多少"，逐项处理完
    整张表，不能中途因为看到 num 变小就手动判断该跳到哪一档——让表本身的顺序和步长去
    自然覆盖所有情况。
    """
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    parts: list[str] = []
    for v, sym in zip(values, symbols):
        count, num = divmod(num, v)
        if count:
            parts.append(sym * count)
    return "".join(parts)


def roman_to_int(s: str) -> int:
    """
    【题意】给定一个合法的罗马数字字符串 s，把它转换成对应的整数。
    【思路】从左到右扫描每一个字符，把它换算成对应数值；核心规则是"当前字符的数值如果
    比紧邻的下一个字符小，说明这是一个减法组合"（比如 "IV" 里的 'I' 比 'V' 小，实际
    贡献的是 -1 而不是 +1），此时要用减法；否则按普通加法累加。这条规则天然覆盖了
    所有罗马数字的减法特例（IV/IX/XL/XC/CD/CM），不需要单独识别这 6 种特例字符串。
    【复杂度】时间 O(n)，空间 O(1)（符号-数值表大小固定为 7）。
    【易错点】
    1) 判断"是否减法"只需要看当前字符和紧邻的下一个字符，罗马数字的减法规则只允许
    相邻两个字符构成减法组合，不存在跨字符的减法关系，不能也不需要往后看更远；
    2) 字符串最后一个字符没有"下一个字符"可比较，必须先判断 i+1 < len(s) 再取
    s[i+1]，否则会越界抛异常；
    3) 用 `for i, ch in enumerate(s)` 搭配 `i+1 < len(s)` 的短路判断，最后一个字符
    因为没有下一个可比较，自然落入"加法"分支，不需要在循环外单独补处理最后一个字符
    这一步逻辑。
    """
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        v = values[ch]
        if i + 1 < len(s) and v < values[s[i + 1]]:
            total -= v
        else:
            total += v
    return total


def push_dominoes(dominoes: str) -> str:
    """
    【题意】给定一排多米诺骨牌的初始状态字符串 dominoes（'L' 表示被向左推倒，'R' 表示
    被向右推倒，'.' 表示直立），骨牌之间的推力会持续传导直到全部稳定。规则：某张直立的
    骨牌如果同时受到左右两侧方向相反的力（一侧传来的力要倒向左、另一侧传来的力要倒向
    右），两个力相互抵消，这张牌保持直立；否则跟随唯一受到的那个方向倒下。返回最终
    每张骨牌的状态。
    【思路】把问题看成"力"的传导，而不是逐轮模拟。在字符串两端各补一个虚拟哨兵字符
    （左边补 'L'、右边补 'R'，代表"没有力"的边界），这样任意一个真实字符（施力点）的
    左右两侧都必然存在一个"上一个/下一个施力点"可以比较，不需要对字符串开头/结尾就是
    连续 '.' 的情况单独写特判。用一次遍历，记录"上一个施力点"的下标和方向，每碰到
    下一个施力点，就处理它们中间夹的这一段 '.'：
    - 方向相同（同为 L 或同为 R）：中间这一段整体倒向这个方向（骨牌像多米诺一样依次
      传导下去）；
    - "L 在左、R 在右"（背对背）：两个力分别往两侧推开，中间这一段完全不受力，保持
      直立；
    - "R 在左、L 在右"（面对面）：两侧的力相向传导、在中间相遇，用双指针从两端向中间
      对称地填 R、L；如果中间字符数是奇数，正中间那一个位置的力恰好被两侧抵消，保持
      直立。
    【复杂度】时间 O(n)，空间 O(n)（补哨兵后的临时字符串 + 结果数组）。
    【易错点】
    1) 必须在两端都补上虚拟哨兵字符，字符串开头/结尾本身是一段连续 '.' 且这一侧没有
    真实施力点时，才能用统一的"比较相邻两个施力点方向"逻辑正确处理，不然需要对这两种
    边界情况单独写 if；
    2) "L 在左 R 在右"和"R 在左 L 在右"这两种组合的处理结果完全相反，极易搞混：前者是
    背对背推开、中间不受力，后者才是面对面挤压、需要双指针从两端向中间填充；
    3) 双指针相向填充时，如果中间字符数是奇数，两个指针会在正中间的前一位交错相遇，
    正中间那一个位置必须保持不被填充（维持 '.'）；如果循环条件写成 `left <= right`
    强行填到底，会把这个本该保持直立的骨牌错误地推倒。
    """
    s = "L" + dominoes + "R"
    result = list(dominoes)
    prev_index = 0
    prev_force = "L"

    for i in range(1, len(s)):
        force = s[i]
        if force == ".":
            continue
        if prev_force == force:
            for j in range(prev_index + 1, i):
                result[j - 1] = force
        elif prev_force == "R" and force == "L":
            left, right = prev_index + 1, i - 1
            while left < right:
                result[left - 1] = "R"
                result[right - 1] = "L"
                left += 1
                right -= 1
        # prev_force == "L" and force == "R"：背对背推开，中间保持 '.'，无需处理
        prev_index = i
        prev_force = force

    return "".join(result)


def _self_test() -> None:
    assert multiply("2", "3") == "6"
    assert multiply("123", "456") == "56088"
    assert multiply("0", "52") == "0"

    assert full_justify(
        ["This", "is", "an", "example", "of", "text", "justification."], 16
    ) == ["This    is    an", "example  of text", "justification.  "]

    assert simplify_path("/home/") == "/home"
    assert simplify_path("/../") == "/"
    assert simplify_path("/home//foo/") == "/home/foo"
    assert simplify_path("/a/./b/../../c/") == "/c"

    assert compare_version("1.01", "1.001") == 0
    assert compare_version("1.0", "1.0.0") == 0
    assert compare_version("0.1", "1.1") == -1

    assert remove_duplicate_letters("bcabc") == "abc"
    assert remove_duplicate_letters("cbacdcbc") == "acdb"

    assert int_to_roman(3749) == "MMMDCCXLIX"
    assert int_to_roman(58) == "LVIII"
    assert int_to_roman(1994) == "MCMXCIV"

    assert roman_to_int("III") == 3
    assert roman_to_int("LVIII") == 58
    assert roman_to_int("MCMXCIV") == 1994

    assert push_dominoes("RR.L") == "RR.L"
    assert push_dominoes(".L.R...LR..L..") == "LL.RR.LLRRLL.."

    print(
        "[PASS] p04_strings_ii: 8/8 题通过 "
        "(字符串相乘/文本左右对齐/简化路径/比较版本号/去除重复字母/"
        "整数转罗马数字/罗马数字转整数/推多米诺)"
    )


if __name__ == "__main__":
    _self_test()
