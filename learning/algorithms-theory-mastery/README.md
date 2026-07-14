# Algorithms Theory Mastery —— 算法设计与分析(证明向)专题(浅→深→社招级别)

> 定位:老师在CS基础五件套完成后要求"仔细思考和科班CS本科相比还差哪些能力",梳理出六个理论必修课空白,这是第三个。这个专题和仓库里已有的 `leetcode-mastery`(544题刷题练习)性质完全不同——leetcode-mastery练的是"写出能跑的代码解决具体题目",这个专题练的是"证明为什么一个算法是对的"这个理论层面,建立在[[discrete-math-mastery]]的证明方法(归纳法/不变量)和[[theory-of-computation-mastery]]的复杂性理论基础上。

## 和其他专题的关系

第一章"渐进分析与主定理"的严格数学定义直接用到离散数学的极限/存在量词语言;第三、四章"动态规划正确性证明"和"图算法正确性证明"反复用到离散数学的归纳法和不变量证明技巧;网络流一章的最大流最小割定理和计算理论专题里P/NP的关系构成对比(网络流问题虽然看起来复杂但有多项式算法,不是NP难)。

## 组织轴:难度分层

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | 渐进分析与主定理(大O/大Omega/大Theta严格定义/递归式求解/主定理/均摊分析)、分治与贪心算法设计基础(分治三步骤/贪心选择性质/交换论证雏形) | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | 动态规划正确性证明深水(最优子结构剪切-粘贴证明/常见DP证明陷阱)、图算法正确性深水(Dijkstra/Bellman-Ford/Prim/Kruskal正确性证明)、网络流与匹配深水(Ford-Fulkerson/最大流最小割定理/Hall定理) | 45 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 算法设计范式选择判断(给定实际问题判断该用贪心/DP/分治)、复杂度权衡与工程实现判断(理论最优vs工程实用性) | 28 | ScenarioPoint |

**合计108点(80个DeepPoint + 28个ScenarioPoint)**,建议先完成discrete-math-mastery(归纳法/不变量证明技巧)再学这个专题。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_alg_asymptotic_analysis_master_theorem.py` | 大O/大Omega/大Theta严格定义、递归式求解(代入法/递归树法)、主定理三种情形、均摊分析(聚合/记账/势能法) | 18 |
| `tier1_shallow/dp_alg_divide_conquer_greedy_basics.py` | 分治三步骤、快速排序最坏情况分析、贪心选择性质、贪心与DP的关系 | 17 |
| `tier2_deep/dp_alg_dynamic_programming_correctness_deep.py` | 最优子结构剪切-粘贴证明、状态定义设计原则、常见DP证明陷阱(状态遗漏维度/转移顺序错误) | 15 |
| `tier2_deep/dp_alg_graph_algorithms_correctness_deep.py` | Dijkstra/Bellman-Ford/Prim/Kruskal正确性证明、cut property、拓扑排序正确性 | 15 |
| `tier2_deep/dp_alg_network_flow_matching_deep.py` | Ford-Fulkerson、最大流最小割定理、Edmonds-Karp、二分图匹配、Hall定理 | 15 |
| `tier3_social_hire/sc_alg_paradigm_selection_judgment.py` | 给定实际问题判断该用贪心/DP/分治设计算法,如何验证选择正确性 | 14 |
| `tier3_social_hire/sc_alg_complexity_engineering_tradeoff_judgment.py` | 理论最优算法是否是最佳工程选择(常数因子/缓存友好性/实现复杂度权衡) | 14 |

## 新增:本科课件式完整教材

沿用CS基础五件套的做法,新增 `lectures/01-textbook.md`——七章连贯教科书叙述,按渐进分析主定理→分治贪心设计基础→动态规划正确性证明→图算法正确性→网络流与匹配→算法范式选择判断→复杂度权衡工程判断的顺序系统教学。**建议先读完 `01-textbook.md` 再做下面的追问链/场景判断自测。**

## 数据结构:DeepPoint + ScenarioPoint(含 explain 字段)

```python
import sys
sys.path.insert(0, "learning/algorithms-theory-mastery/src")
from algorithms_theory_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/algorithms-theory-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部是经典算法教材共识(CLRS《算法导论》/Kleinberg-Tardos《算法设计》体系里的标准证明),不需要追赶前沿论文,但仍要求核实具体定理名称和证明思路的准确性。`real_world_link` 字段全部留空。

至此,老师要求的六个CS科班理论专题第三个完成,后续三个(计算机体系结构/编译原理/安全密码学基础)陆续推进。
