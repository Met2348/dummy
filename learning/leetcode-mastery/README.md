# LeetCode Mastery —— 544 题体系化教学（零 OJ 经验起步 → 大厂 SWE → 竞赛级/Frontier Lab）

> 定位：这不是 `interview-prep/src/leetcode/patterns.py` 那种"已经会刷题、只是复习"的 15-pattern 速查表。这是**从零开始、一路推到竞赛级**的完整教学体系——假设你**一道 OJ 题都没提交过、算法只停留在理论**，把"知道概念"和"能在编辑器里独立写出能跑的代码"之间的那道手续，用固定流程练出来，并且量级和难度持续对齐真实面试反馈。

## 先读这篇，别跳过

**[`lectures/00-how-to-solve.md`](lectures/00-how-to-solve.md)** —— 通用解题四步法（排歧义→暴力解→找规律优化→编码对拍）、没有自动补全的"白板"写法建议、新手高频踩坑清单、复杂度速查表。所有后面 33 类的教学都建立在这篇之上。

## 三个 Phase：起步 100 题 + 进阶 198 题 + 竞赛级 246 题 = 544 题

| Phase | 题量 | tier | 定位 |
|---|:--:|---|---|
| Phase 1 | 100 | `core` | 零经验起步，19 类，每类 4-7 道代表题打地基 |
| Phase 2 | 198 | `advanced` | 冲大厂 SWE 量级，19 类各追加"进阶补充"（`_ii`）+ 6 个新高频类别（Trie/设计题/高级图论/矩阵模拟/高级字符串算法/并发） |
| Phase 3 | 246 | `frontier` | 竞赛级/Frontier Lab 难度，25 类各追加"竞赛级补充"（`_iii`，Hard 为主）+ 8 个全新竞赛分类（数论/线段树树状数组/字符串匹配进阶/计算几何/高级图论II/状压DP/博弈论组合数学/高级数据结构），其中 31 题是不对应具体 LeetCode 编号的**自实现算法组件**（如 Fenwick 树、Tarjan 强连通分量、凸包、Aho-Corasick），用暴力/朴素实现交叉验证而非编造官方样例 |

`src/catalog.py` 里每条 `Problem` 有 `tier` 字段（`"core"`/`"advanced"`/`"frontier"`），可按 tier 筛选练习顺序；自实现算法的 `leetcode_no` 用负数占位（保证 catalog 内编号唯一性校验仍然成立），不是真实 LeetCode 编号。

## 关于"竞赛级"的诚实说明

Phase 3 是应用户请一位 OJ 老手看过后的反馈而建——补齐了"缺一个真正的地基年级：Trie/设计题/高级图论/矩阵模拟/高级字符串算法"，并把 25 个基础类别都推到了 Hard 难度为主的竞赛级题目。但如果你的目标是**纯研究岗（RS/RE）**面试，务必对照 `learning/interview-prep/` 那个 track——多数前沿实验室的研究岗面试并不考纯竞赛式算法（segment tree beats、后缀自动机这类），更看重真实 PyTorch 手写能力和研究判断力。Phase 3 对**大厂 SWE/MLE 正式 coding 面试**和"以防万一"的地板保险都有实打实的价值，但不要误以为"刷完这 544 题 = 搞定研究岗面试"——两者是不同的准备维度，详见 `learning/interview-prep/README.md`。

## 33 类总览

| # | 分类 | Phase1 | Phase2 | Phase3 | 合计 | Lecture |
|---|------|:--:|:--:|:--:|:--:|------|
| 01 | 数组与双指针 | 6 | 9 | 8 | 23 | [01](lectures/01-arrays-two-pointers.md) |
| 02 | 滑动窗口 | 5 | 7 | 6 | 18 | [02](lectures/02-sliding-window.md) |
| 03 | 哈希表 | 5 | 7 | 7 | 19 | [03](lectures/03-hashing.md) |
| 04 | 字符串 | 6 | 8 | 7 | 21 | [04](lectures/04-strings.md) |
| 05 | 链表 | 6 | 8 | 7 | 21 | [05](lectures/05-linked-list.md) |
| 06 | 栈与队列 | 5 | 8 | 7 | 20 | [06](lectures/06-stack-queue.md) |
| 07 | 二分查找 | 5 | 8 | 8 | 21 | [07](lectures/07-binary-search.md) |
| 08 | 二叉树遍历与构造 | 7 | 12 | 9 | 28 | [08](lectures/08-binary-tree.md) |
| 09 | 二叉搜索树 | 4 | 7 | 7 | 18 | [09](lectures/09-bst.md) |
| 10 | 回溯 | 6 | 12 | 8 | 26 | [10](lectures/10-backtracking.md) |
| 11 | 图 BFS/DFS | 6 | 12 | 9 | 27 | [11](lectures/11-graph-bfs-dfs.md) |
| 12 | DP 基础/序列 | 7 | 13 | 11 | 31 | [12](lectures/12-dp-basics.md) |
| 13 | DP 进阶 | 6 | 13 | 10 | 29 | [13](lectures/13-dp-advanced.md) |
| 14 | 贪心 | 5 | 8 | 7 | 20 | [14](lectures/14-greedy.md) |
| 15 | 堆 / Top-K | 5 | 8 | 7 | 20 | [15](lectures/15-heap-topk.md) |
| 16 | 单调栈 / 单调队列 | 4 | 7 | 5 | 16 | [16](lectures/16-monotonic-stack.md) |
| 17 | 并查集 | 3 | 6 | 5 | 14 | [17](lectures/17-union-find.md) |
| 18 | 前缀和与差分 | 4 | 7 | 5 | 16 | [18](lectures/18-prefix-sum.md) |
| 19 | 位运算与数学 | 5 | 7 | 5 | 17 | [19](lectures/19-bit-math.md) |
| 20 | Trie 前缀树 | — | 5 | 5 | 10 | [20](lectures/20-trie.md) |
| 21 | 设计题 | — | 6 | 4 | 10 | [21](lectures/21-design.md) |
| 22 | 高级图论 | — | 7 | 6 | 13 | [22](lectures/22-advanced-graph.md) |
| 23 | 矩阵与模拟 | — | 6 | 5 | 11 | [23](lectures/23-matrix-simulation.md) |
| 24 | 高级字符串算法 | — | 4 | 4 | 8 | [24](lectures/24-advanced-strings.md) |
| 25 | 并发（可选层） | — | 3 | 3 | 6 | [25](lectures/25-concurrency.md) |
| 26 | 数论与数学进阶（新） | — | — | 14 | 14 | [26](lectures/26-number-theory.md) |
| 27 | 线段树与树状数组进阶（新） | — | — | 10 | 10 | [27](lectures/27-segment-tree-bit.md) |
| 28 | 字符串匹配进阶（新） | — | — | 9 | 9 | [28](lectures/28-string-matching-advanced.md) |
| 29 | 计算几何（新） | — | — | 9 | 9 | [29](lectures/29-computational-geometry.md) |
| 30 | 高级图论II（新） | — | — | 12 | 12 | [30](lectures/30-advanced-graph-ii.md) |
| 31 | 位运算与状态压缩DP（新） | — | — | 10 | 10 | [31](lectures/31-bitmask-dp.md) |
| 32 | 博弈论与组合数学（新） | — | — | 9 | 9 | [32](lectures/32-game-theory-combinatorics.md) |
| 33 | 高级数据结构（新） | — | — | 8 | 8 | [33](lectures/33-advanced-data-structures.md) |

**合计 544 题**（100 core + 198 advanced + 246 frontier），难度分布 78 易 / 333 中 / 133 难（见 `src/catalog.py` 的 `stats()`）。完整元数据（LeetCode 编号/名称/分类/难度/tier）见 [`src/catalog.py`](src/catalog.py)。

## 每道题的教学模板

每个 `src/problems/pNN_*.py`（Phase 1）/ `pNN_*_ii.py`（Phase 2 进阶）/ `pNN_*_iii.py`（Phase 3 竞赛级）/ `p2N_*.py`、`p3N_*.py`（Phase 2/3 全新分类）文件里，每道题一个函数（或类），docstring 固定四段：

```
【题意】一句话复述（不抄原题，用自己的话讲清输入输出约束）
【思路】为什么这个 pattern 能用在这——核心 insight，不是"怎么写"而是"为什么这么想"
【复杂度】时间 / 空间
【易错点】新手最容易栽的坑
```

Phase 1 每类有 1-2 道代表题在 `lectures/NN-*.md` 里做**"思考过程全展开"深挖**；Phase 2/3 的补充以追加到同一篇 lecture 末尾的 `## 进阶补充（Part II）`/`## 竞赛级补充（Part III）` 呈现（不重新教框架，专注扩大变体覆盖面 / 拉高难度）。`_self_test()` 用的都是**真实 LeetCode 官方样例**（自实现算法用暴力/朴素实现交叉验证），不是编造的测试数据。

## 怎么用

1. 先读 `lectures/00-how-to-solve.md`。
2. 按 `01` → `33` 分类顺序推进（由易到难，Phase 2/3 新分类排在最后）。
3. 每类先刷完 Phase 1（`core`），再刷 Phase 2 进阶（`advanced`），最后按需刷 Phase 3 竞赛级（`frontier`，可以只挑目标公司/岗位真正会考的类别深入，不必每类都刷满）。
4. 每类先读 lecture 里的深挖例题，理解完整思考过程。
5. **合上参考代码**，自己在编辑器里重写整类题目（哪怕比参考解慢，先保证跑通）。
6. 跑对应文件的 `_self_test()` 对拍，红了就对照参考实现看哪一步想岔了。
7. 用 `src/tracker.py` 把做过的题排进间隔复习节奏（见 [`lectures/34-review-strategy.md`](lectures/34-review-strategy.md)），而不是做完就忘。

## 环境（纯 stdlib，无需 torch/numpy —— 这是和 `interview-prep` 的关键区别）

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/leetcode-mastery/environment/verify_env.py
```

## 跑测试

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/leetcode-mastery/src/tests/test_all.py     # 预期 77/77 modules passed
```

单文件也可直接跑（每个都有 `__main__`）：

```powershell
python learning/leetcode-mastery/src/problems/p01_arrays_two_pointers.py
python learning/leetcode-mastery/src/problems/p01_arrays_two_pointers_iii.py
python learning/leetcode-mastery/src/problems/p27_segment_tree_bit.py
python learning/leetcode-mastery/src/catalog.py
python learning/leetcode-mastery/src/tracker.py
```

## 与 `interview-prep/src/leetcode` 的边界

- `interview-prep/src/leetcode/patterns.py`：15 个 pattern 各一道范例解，是给**已经会刷题的人**用的"面试前地板速查"，追求精简。
- 本 track：544 题、每题四段式教学 + 分类深挖讲义，是给**零 OJ 经验起步、要一路冲到竞赛级/大厂 SWE 面试量级**的人用的完整体系。
- 两者的 `ReviewTracker`（SM-2 间隔复习）算法一致，但各自独立成册，不做跨 track import。

## 一句话

> 你缺的不是算法知识，是"把知识翻译成能跑的代码"这道手续——这道手续是可以练出来的固定流程，不是天赋，本 track 把这道手续从零拆到竞赛级，但记得对照你真正要面的岗位，按需取用，而不是盲目求全。
