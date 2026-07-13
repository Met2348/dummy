# Discrete Math Mastery —— 离散数学专题(浅→深→社招级别)

> 定位:老师在CS基础五件套(软件工程/设计模式/数据库/网络/OS)完成后,进一步指出"仔细思考和科班CS本科相比还差哪些能力"。梳理后确认还有六个科班CS本科必修课程的空白——离散数学、计算理论、算法设计与分析(证明向)、计算机体系结构、编译原理、安全与密码学基础。这是这六个新专题的第一个。

## 和已完成的CS基础五件套的关系

沿用完全相同的数据结构和组织方式:`DeepPoint`/`ScenarioPoint` 都有 `explain: str` 字段(≥150字的系统性讲解:是什么/为什么/怎么用/常见误区),再接三层追问链或场景判断rubric,兼顾"系统学会"和"面试接得住"。不同的是这六个新专题更偏"科班理论素养"而不是"生产环境八股",尤其是离散数学这类课程,面试直接问到的概率低于OS/DB,但它是后续"计算理论"和"算法理论证明向"两个专题的地基——图论、组合计数、证明方法这三块内容会被反复引用。

## 组织轴:难度分层

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | 逻辑与集合函数(命题逻辑/谓词逻辑/集合运算/函数性质)、关系-偏序与计数基础(等价关系/偏序/哈斯图/基本计数原理/排列组合/鸽笼原理) | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | 组合数学深水(容斥原理/生成函数/递推关系/Catalan数)、图论深水(树/欧拉路径/哈密顿路径/图着色/二分图/平面图)、证明方法深水(直接证明/反证法/归纳法/结构归纳法/不变量) | 45 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 算法建模判断(给定实际问题该用什么离散数学工具建模)、证明策略与常见错误判断(识别证明里的逻辑漏洞) | 28 | ScenarioPoint |

**合计108点(80个DeepPoint + 28个ScenarioPoint)**,练法建议按 Tier 1→2→3 顺序推进——先读 `explain` 建立认知,再练 `chain`/`rubric` 检验是否真的能被追问住。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_dm_logic_sets_functions.py` | 命题逻辑(连接词/真值表/重言式)、谓词逻辑(量词/嵌套量词顺序)、德摩根律、集合运算、函数(单射满射双射) | 18 |
| `tier1_shallow/dp_dm_relations_orders_counting_basics.py` | 等价关系与等价类、偏序与哈斯图、良序、加法乘法原理、排列组合、鸽笼原理、二项式定理 | 17 |
| `tier2_deep/dp_dm_combinatorics_deep.py` | 容斥原理深水、生成函数、递推关系特征根法、Catalan数、组合恒等式证明技巧 | 15 |
| `tier2_deep/dp_dm_graph_theory_deep.py` | 图的表示与握手定理、树的性质、欧拉路径/回路、哈密顿路径NP难前瞻、图着色、二分图、平面图欧拉公式 | 15 |
| `tier2_deep/dp_dm_proof_techniques_deep.py` | 直接证明/反证法/逆否证明、弱归纳法与经典谬误、强归纳法、结构归纳法、不变量、良序原理 | 15 |
| `tier3_social_hire/sc_dm_algorithm_modeling_judgment.py` | 给定实际问题判断该用图论/组合计数/关系偏序/生成函数中哪种工具建模 | 14 |
| `tier3_social_hire/sc_dm_proof_strategy_pitfalls_judgment.py` | 识别证明里的逻辑漏洞(循环论证/归纳基础步骤缺失/隐藏假设/以偏概全) | 14 |

## 新增:本科课件式完整教材

沿用CS基础五件套的做法,新增 `lectures/01-textbook.md`——一份七章的完整教材式讲义,按逻辑与集合函数→关系偏序与计数基础→组合数学深水→图论深水→证明方法深水→算法建模判断→证明策略与常见错误判断的顺序,用连贯的教科书叙述(引言/核心概念展开/场景走查/常见误区/小结)系统教会每个主题,不是知识点罗列或追问链。**建议学习顺序:先读完 `01-textbook.md` 建立系统认知,再回到下面的 DeepPoint/ScenarioPoint 用追问链和场景判断检验"是否真的能被追问住"。**

## 数据结构:DeepPoint + ScenarioPoint(含 explain 字段)

```python
import sys
sys.path.insert(0, "learning/discrete-math-mastery/src")
from discrete_math_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/discrete-math-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部是经典离散数学教材共识(Kenneth Rosen《离散数学及其应用》/图论标准教材如West《Introduction to Graph Theory》体系里的标准表述),不需要追赶前沿论文,但仍要求核实具体定理名称和表述的准确性。`real_world_link` 字段全部留空——这批内容没有可验证的本地文件或需要引用的具体外部产品案例,不编造路径。

至此,老师要求的六个CS科班理论专题(离散数学/计算理论/算法理论证明向/计算机体系结构/编译原理/安全密码学基础)第一个完成,后续五个陆续推进。
