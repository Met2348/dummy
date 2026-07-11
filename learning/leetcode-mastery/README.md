# LeetCode Mastery —— 100 题体系化教学（零 OJ 经验起步）

> 定位：这不是 `interview-prep/src/leetcode/patterns.py` 那种"已经会刷题、只是复习"的 15-pattern 速查表。这是**从零开始**的完整教学体系——假设你**一道 OJ 题都没提交过、算法只停留在理论**，把"知道概念"和"能在编辑器里独立写出能跑的代码"之间的那道手续，用固定流程练出来。

## 先读这篇，别跳过

**[`lectures/00-how-to-solve.md`](lectures/00-how-to-solve.md)** —— 通用解题四步法（排歧义→暴力解→找规律优化→编码对拍）、没有自动补全的"白板"写法建议、新手高频踩坑清单、复杂度速查表。所有后面 19 类的教学都建立在这篇之上。

## 100 题总览（由易到难分类排序）

| # | 分类 | 题数 | Lecture |
|---|------|:--:|------|
| 01 | 数组与双指针 | 6 | [01](lectures/01-arrays-two-pointers.md) |
| 02 | 滑动窗口 | 5 | [02](lectures/02-sliding-window.md) |
| 03 | 哈希表 | 5 | [03](lectures/03-hashing.md) |
| 04 | 字符串 | 6 | [04](lectures/04-strings.md) |
| 05 | 链表 | 6 | [05](lectures/05-linked-list.md) |
| 06 | 栈与队列 | 5 | [06](lectures/06-stack-queue.md) |
| 07 | 二分查找 | 5 | [07](lectures/07-binary-search.md) |
| 08 | 二叉树遍历与构造 | 7 | [08](lectures/08-binary-tree.md) |
| 09 | 二叉搜索树 | 4 | [09](lectures/09-bst.md) |
| 10 | 回溯 | 6 | [10](lectures/10-backtracking.md) |
| 11 | 图 BFS/DFS | 6 | [11](lectures/11-graph-bfs-dfs.md) |
| 12 | DP 基础/序列 | 7 | [12](lectures/12-dp-basics.md) |
| 13 | DP 进阶（背包/区间/编辑距离） | 6 | [13](lectures/13-dp-advanced.md) |
| 14 | 贪心 | 5 | [14](lectures/14-greedy.md) |
| 15 | 堆 / Top-K | 5 | [15](lectures/15-heap-topk.md) |
| 16 | 单调栈 / 单调队列 | 4 | [16](lectures/16-monotonic-stack.md) |
| 17 | 并查集 | 3 | [17](lectures/17-union-find.md) |
| 18 | 前缀和与差分 | 4 | [18](lectures/18-prefix-sum.md) |
| 19 | 位运算与数学 | 5 | [19](lectures/19-bit-math.md) |

**合计 100 题**，难度分布 25 易 / 62 中 / 13 难（见 `src/catalog.py` 的 `stats()`）——以中等题为主、少量热身、少量拉伸，对齐真实面试的题目分布。完整元数据（LeetCode 编号/名称/分类/难度）见 [`src/catalog.py`](src/catalog.py)。

## 每道题的教学模板

每个 `src/problems/pNN_*.py` 文件里，每道题一个函数（或类），docstring 固定四段：

```
【题意】一句话复述（不抄原题，用自己的话讲清输入输出约束）
【思路】为什么这个 pattern 能用在这——核心 insight，不是"怎么写"而是"为什么这么想"
【复杂度】时间 / 空间
【易错点】新手最容易栽的坑
```

每类还有 1-2 道代表题在对应 `lectures/NN-*.md` 里做**"思考过程全展开"深挖**：暴力解怎么想 → 哪里能优化 → 为什么换成这个数据结构/技巧 → 一步步编码。其余题目给"一句话点睛 + 和深挖题的联系"。`_self_test()` 用的都是**真实 LeetCode 官方样例**，不是编造的测试数据。

## 怎么用

1. 先读 `lectures/00-how-to-solve.md`。
2. 按 `01` → `19` 分类顺序推进（已按由易到难排好）。
3. 每类先读 lecture 里的深挖例题，理解完整思考过程。
4. **合上参考代码**，自己在编辑器里重写整类题目（哪怕比参考解慢，先保证跑通）。
5. 跑对应 `pNN_*.py` 的 `_self_test()` 对拍，红了就对照参考实现看哪一步想岔了。
6. 用 `src/tracker.py` 把做过的题排进间隔复习节奏（见 [`lectures/20-review-strategy.md`](lectures/20-review-strategy.md)），而不是做完就忘。

## 环境（纯 stdlib，无需 torch/numpy —— 这是和 `interview-prep` 的关键区别）

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/leetcode-mastery/environment/verify_env.py
```

## 跑测试

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/leetcode-mastery/src/tests/test_all.py     # 预期 19/19 modules passed
```

单文件也可直接跑（每个都有 `__main__`）：

```powershell
python learning/leetcode-mastery/src/problems/p01_arrays_two_pointers.py
python learning/leetcode-mastery/src/catalog.py
python learning/leetcode-mastery/src/tracker.py
```

## 与 `interview-prep/src/leetcode` 的边界

- `interview-prep/src/leetcode/patterns.py`：15 个 pattern 各一道范例解，是给**已经会刷题的人**用的"面试前地板速查"，追求精简。
- 本 track：100 题、每题四段式教学 + 分类深挖讲义，是给**零 OJ 经验的人**用的完整体系，追求"从理论到能独立写出代码"的手把手过渡。
- 两者的 `ReviewTracker`（SM-2 间隔复习）算法一致，但各自独立成册，不做跨 track import。

## 一句话

> 你缺的不是算法知识，是"把知识翻译成能跑的代码"这道手续——这道手续是可以练出来的固定流程，不是天赋，本 track 就是把这道手续拆开、一步步教。
