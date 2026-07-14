# Theory of Computation Mastery —— 计算理论专题(浅→深→社招级别)

> 定位:老师在CS基础五件套完成后要求"仔细思考和科班CS本科相比还差哪些能力",梳理出六个理论必修课空白(离散数学/计算理论/算法理论证明向/计算机体系结构/编译原理/安全密码学基础),这是第二个。计算理论(自动机/图灵机/可计算性/复杂性理论)是CS理论里最"纯粹"的一支——它回答"计算的边界在哪里"这个根本问题,建立在[[discrete-math-mastery]]的图论、组合、证明方法基础上。

## 和离散数学专题的关系

第一章"自动机与正则语言"和第二章"文法与下推自动机"直接建立在离散数学的证明方法之上(泵引理的证明本质是反证法);第四、五章的"图灵机不可判定性"和"NP完全性"呼应离散数学图论章节里"哈密顿路径判定是NP难"这个前瞻性的点。沿用完全相同的数据结构:`DeepPoint`/`ScenarioPoint` 都有 `explain: str` 字段(≥150字系统性讲解),再接三层追问链或场景判断rubric。

## 组织轴:难度分层

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | 自动机与正则语言(DFA/NFA/子集构造法/正则表达式/泵引理)、上下文无关文法与下推自动机(CFG/二义性/乔姆斯基范式/PDA) | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | 图灵机与可计算性深水(邱奇-图灵论题/停机问题/Rice定理)、计算复杂性类深水(P/NP/Cook-Levin定理)、归约与NP完全性证明深水(归约套路/近似算法/参数化复杂度) | 45 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 算法可行性判断(给定实际问题判断NP难/不可判定及应对策略)、形式化建模判断(该用什么层级的计算模型描述问题) | 28 | ScenarioPoint |

**合计108点(80个DeepPoint + 28个ScenarioPoint)**,建议先完成 discrete-math-mastery 再学这个专题(图论/证明方法是地基)。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_toc_automata_regular_languages.py` | DFA/NFA形式化定义、子集构造法、正则表达式与Kleene定理、正则语言封闭性质、泵引理 | 18 |
| `tier1_shallow/dp_toc_grammars_pushdown_automata.py` | CFG形式化定义、文法二义性、乔姆斯基范式、PDA、CFG-PDA等价性、CFL泵引理 | 17 |
| `tier2_deep/dp_toc_turing_machines_computability_deep.py` | 图灵机形式化定义、邱奇-图灵论题、通用图灵机、停机问题不可判定性、Rice定理 | 15 |
| `tier2_deep/dp_toc_complexity_classes_deep.py` | P/NP/co-NP、多项式时间归约、Cook-Levin定理、经典NP完全问题 | 15 |
| `tier2_deep/dp_toc_reductions_np_completeness_proofs_deep.py` | 归约证明标准套路、近似算法、参数化复杂度、工程应对策略 | 15 |
| `tier3_social_hire/sc_toc_computational_feasibility_judgment.py` | 给定实际问题判断是否NP难/不可判定及应对策略 | 14 |
| `tier3_social_hire/sc_toc_formal_modeling_judgment.py` | 给定实际问题判断该用什么层级的计算模型(FSM/CFG/图灵完备)描述 | 14 |

## 新增:本科课件式完整教材

沿用CS基础五件套和discrete-math-mastery的做法,新增 `lectures/01-textbook.md`——七章连贯教科书叙述,按自动机与正则语言→文法与下推自动机→图灵机与可计算性→计算复杂性类→归约与NP完全性证明→算法可行性判断→形式化建模判断的顺序系统教学。**建议先读完 `01-textbook.md` 再做下面的追问链/场景判断自测。**

## 数据结构:DeepPoint + ScenarioPoint(含 explain 字段)

```python
import sys
sys.path.insert(0, "learning/theory-of-computation-mastery/src")
from theory_of_computation_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/theory-of-computation-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部是经典计算理论教材共识(Michael Sipser《计算理论导论》/Hopcroft-Ullman《自动机理论、语言和计算导论》/Garey-Johnson《计算机与难解性》体系里的标准表述),不需要追赶前沿论文,但仍要求核实具体定理名称和表述的准确性。`real_world_link` 字段全部留空。

至此,老师要求的六个CS科班理论专题第二个完成,后续四个(算法理论证明向/计算机体系结构/编译原理/安全密码学基础)陆续推进。

## 补充:直接问答自测

`qa-practice.md`——一份轻量级补充练习(24题,不含追问链评分逻辑),每题一个问题配一个完整参考答案,和上面108点的3层追问链形式不同,提问角度也刻意避开了源码里的trigger原文,适合刷完追问链之后做最后一遍考前快速过关。
