"""分类 17：并查集 —— 省份数量 / 冗余连接 / 账户合并。

并查集（Union-Find / Disjoint Set）解决的是"动态维护连通性"：一边不断地把两个元素
合并到同一个集合里，一边随时能回答"这两个元素现在是否在同一个集合（连通分量）"。
它比"每次查询都重新 BFS/DFS 整张图"快得多——核心是两个优化：路径压缩（find 时把
路径上所有节点直接挂到根节点下）和按秩合并（union 时让矮树挂到高树下面），合起来
让每次 find/union 的均摊复杂度逼近 O(1)（严格说是 O(α(n))，反阿克曼函数，可以当
常数看待）。
"""
from __future__ import annotations


class UnionFind:
    """通用并查集：路径压缩 + 按秩合并。三道题共用同一份实现。"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # 路径压缩
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.count -= 1
        return True


def find_circle_num(is_connected: list[list[int]]) -> int:
    """
    【题意】给 n 个城市的邻接矩阵 is_connected（`is_connected[i][j]==1` 表示 i、j
    直接相连），"省份"是若干城市通过直接或间接相连组成的连通分量，求省份总数。

    【思路】这本质是"数无向图里有多少个连通分量"。可以用 BFS/DFS 对每个还没访问过的
    城市跑一遍遍历、每跑一次遍历计数加一，但既然图已经完整给出（邻接矩阵），更省事的
    写法是并查集：初始时每个城市各自是一个集合（count=n），扫描邻接矩阵，凡是
    `is_connected[i][j]==1` 就把 i、j union 到同一个集合（第一次真正合并两个不同集合时
    count 会减一）。矩阵是对称的，所以只需要扫上三角（`j` 从 `i+1` 开始）就够了，
    不会重复处理同一条边。扫完之后 `uf.count` 就是剩下的集合数，即省份数量。

    【复杂度】时间 O(n^2)（矩阵大小决定了下限，扫矩阵本身就是 O(n^2)，并查集操作
    均摊近似 O(1)）；空间 O(n)。

    【易错点】
    - 忘记矩阵是对称的，把下三角也扫一遍——虽然不会算错（union 两次同一对集合，
      第二次直接返回 False），但是多余的重复工作。
    - 对角线 `is_connected[i][i]` 恒为 1（自己和自己相连），如果从 `j=0` 开始扫容易
      多做一次没有意义的 `union(i, i)`（虽然无害，因为 find(i)==find(i) 会被识别为
      同一集合，但徒增调用次数），从 `j=i+1` 开始更干净。
    - 返回的是 `uf.count`（剩余集合数），不是 union 成功的次数或矩阵里 1 的个数。
    """
    n = len(is_connected)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if is_connected[i][j] == 1:
                uf.union(i, j)
    return uf.count


def find_redundant_connection(edges: list[list[int]]) -> list[int]:
    """
    【题意】一棵有 n 个节点的树本该只有 n-1 条边，现在多加了一条边，形成了一个环。
    节点编号从 1 到 n，边按输入顺序给出。找出这条"多余的边"——如果有多条边都可能是
    冗余边，返回在输入中最后出现的那一条。

    【思路】一棵树的性质是"任意两点之间只有一条路径"，也就是说，如果一条边连接的
    两个端点在加入这条边之前就已经连通了，这条边就是那条导致成环的多余边。用并查集
    逐条按输入顺序处理边：对边 (a, b)，如果 `find(a) == find(b)`（已经连通），说明
    这条边就是答案，直接返回；否则正常 union 这两个节点。因为题目保证"恰好一条边"
    是多余的，按输入顺序第一次遇到"两端已连通"的边，就一定是要找的那条（且它必然是
    输入里最后出现的合法答案，因为只有一条环边）。节点编号从 1 开始，所以并查集要开
    n+1 大小（下标 0 空置不用，避免每次都做 -1 的编号转换,更不容易出错）。

    【复杂度】时间 O(n·α(n))（n 条边，每条边一次 find + 可能一次 union）；空间 O(n)。

    【易错点】
    - 节点编号从 1 开始而不是 0——`UnionFind(n)` 要开 `n+1` 的大小，否则编号为 n
      的节点会越界；也可以每次都 -1 转换，但更容易在某处漏掉，不如直接开大小 n+1。
    - 要按输入顺序处理边，且遇到第一条"两端已连通"的边就立刻返回——不能扫完全部
      边再挑，也不能提前排序打乱了原始顺序（题目要"输入中最后出现"的那条）。
    - 判断"是否已连通"要用 `find(a) == find(b)`，不能直接调用 `union` 看返回值之后
      又忘记提前 return，导致多余边被错误地也 union 掉了。
    """
    n = len(edges)
    uf = UnionFind(n + 1)  # 节点编号 1..n，下标 0 空置
    for a, b in edges:
        if uf.find(a) == uf.find(b):
            return [a, b]
        uf.union(a, b)
    return []  # 题目保证一定存在冗余边，不会走到这里


def accounts_merge(accounts: list[list[str]]) -> list[list[str]]:
    """
    【题意】每个 account 是 `[name, email1, email2, ...]`，同一个人可能注册了多个
    account（用户名相同、但邮箱有重叠即可判定是同一人；用户名不同的人不会共享邮箱）。
    把属于同一个人的所有 account 合并成一个 `[name, 排序后的邮箱...]`，返回结果列表
    （多个合并结果之间顺序、以及邮箱重复的账户合并顺序不要求固定）。

    【思路】这题最容易卡住的地方是：并查集要合并的对象是"邮箱"，而不是"账户下标"。
    直觉上可能想"把账户两两比较、有公共邮箱就合并账户"，但账户数量、每个账户邮箱数
    都不固定，两两比较账户效率低还容易漏判。更干净的做法是——把每个不同的邮箱地址
    当成并查集里的一个节点：遍历每个账户，把该账户内的所有邮箱两两 union 到一起
    （比如都 union 到该账户第一个邮箱上），这样"同一个账户内的邮箱"必然被分到同一个
    集合；如果两个账户共享了一个邮箱，那个共享邮箱在两次不同账户的 union 过程中，
    会把这两个账户的所有邮箱都拉进同一个集合——不需要显式比较账户，公共邮箱自然把
    它们"传递地"合并了。同时用一个 `email_to_name` 字典记住每个邮箱属于谁（同一
    集合内的邮箱对应的 name 必然相同，随便记一次就够）。
    合并完，第二步是"重新分组"：对每个邮箱，找到它的根节点，把邮箱塞进
    `root -> [emails]` 的分组字典里；最后对每个分组，取出该组任意一个邮箱对应的
    name，把邮箱排序，拼成 `[name, email1, email2, ...]` 即为一个合并后的账户。

    【复杂度】时间 O(E log E)（E 为邮箱总数，主要开销在给每组邮箱排序；并查集操作
    本身均摊接近 O(1)）；空间 O(E)（邮箱到编号的映射、并查集数组、分组字典）。

    【易错点】
    - 把并查集建在"账户下标"上而不是"邮箱"上——这是本题最典型的思路误区，账户
      下标之间没有直接可比较的东西，真正承载"同一个人"信息的是邮箱本身。
    - 给邮箱分配并查集编号时，要先扫一遍所有账户建立 `email -> 编号` 的映射（因为
      并查集需要连续整数下标），漏掉任何一个邮箱都会导致 KeyError 或分组遗漏。
    - 重新分组时要按"根节点"分组，而不是按"账户下标"或"第一次出现的邮箱"分组，
      否则共享邮箱的多个账户不会被真正合并到同一条输出记录里。
    - 输出的邮箱列表题目要求排序（方便去重比较也方便测试），容易忘记 `sorted(...)`。
    """
    email_to_name: dict[str, str] = {}
    email_to_id: dict[str, int] = {}
    next_id = 0
    for account in accounts:
        name = account[0]
        for email in account[1:]:
            if email not in email_to_id:
                email_to_id[email] = next_id
                next_id += 1
            email_to_name[email] = name

    uf = UnionFind(next_id)
    for account in accounts:
        emails = account[1:]
        if not emails:
            continue
        first_id = email_to_id[emails[0]]
        for email in emails[1:]:
            uf.union(first_id, email_to_id[email])

    groups: dict[int, list[str]] = {}
    for email, eid in email_to_id.items():
        root = uf.find(eid)
        groups.setdefault(root, []).append(email)

    result: list[list[str]] = []
    for emails in groups.values():
        name = email_to_name[emails[0]]
        result.append([name] + sorted(emails))
    return result


def _self_test() -> None:
    assert find_circle_num([[1, 1, 0], [1, 1, 0], [0, 0, 1]]) == 2
    assert find_circle_num([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 3

    assert find_redundant_connection([[1, 2], [1, 3], [2, 3]]) == [2, 3]
    assert find_redundant_connection([[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]]) == [1, 4]

    accounts = [
        ["John", "johnsmith@mail.com", "john_newyork@mail.com"],
        ["John", "johnsmith@mail.com", "john00@mail.com"],
        ["Mary", "mary@mail.com"],
        ["John", "johnnybravo@mail.com"],
    ]
    result = accounts_merge(accounts)
    expected = {
        ("John", frozenset({"johnsmith@mail.com", "john00@mail.com", "john_newyork@mail.com"})),
        ("Mary", frozenset({"mary@mail.com"})),
        ("John", frozenset({"johnnybravo@mail.com"})),
    }
    actual = {(acc[0], frozenset(acc[1:])) for acc in result}
    assert actual == expected

    print("[PASS] p17_union_find: 省份数量 / 冗余连接 / 账户合并 全部正确")


if __name__ == "__main__":
    _self_test()
