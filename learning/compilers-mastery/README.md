# Compilers Mastery —— 编译原理专题(浅→深→社招级别)

> 定位:老师在CS基础五件套完成后要求"仔细思考和科班CS本科相比还差哪些能力",梳理出六个理论必修课空白,这是第五个。编译原理是[[theory-of-computation-mastery]](自动机/CFG)理论在真实工程系统里的落地——词法分析用到有限自动机,语法分析用到上下文无关文法和下推自动机,寄存器分配用到图着色问题(呼应[[algorithms-theory-mastery]]的图论/NP难背景)。

## 和其他专题的关系

第一章词法分析直接复用theory-of-computation-mastery专题的正则表达式-DFA等价性(Thompson构造/子集构造法);第二章语法分析的LL(1)/消除左递归、第四章LR分析都建立在CFG理论之上;第五章寄存器分配的图着色问题呼应algorithms-theory-mastery专题里"图着色是NP难问题"这个知识点,展示理论结果如何指导实际工程算法设计(用启发式而非精确算法做寄存器分配)。

## 组织轴:难度分层

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | 词法分析(token/正则表达式到DFA/最长匹配)、语法分析基础(递归下降/LL(1)/FIRST-FOLLOW/消除左递归) | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | LR分析深水(LR(0)/SLR/LALR/移进规约冲突)、语义分析与类型系统深水(符号表/类型检查/Hindley-Milner类型推断)、中间表示与代码生成优化深水(三地址码/SSA/常量折叠死代码消除/图着色寄存器分配) | 45 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 语言设计与实现权衡判断(解析策略选择/DSL设计判断)、编译器优化与调试判断(优化引入的bug定位/JIT vs AOT判断) | 28 | ScenarioPoint |

**合计108点(80个DeepPoint + 28个ScenarioPoint)**,建议先完成theory-of-computation-mastery(自动机/CFG理论)再学这个专题。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_comp_lexical_analysis_basics.py` | token/词素、正则表达式到DFA、最长匹配原则、手写vs生成器 | 18 |
| `tier1_shallow/dp_comp_parsing_basics.py` | 递归下降、LL(1)、FIRST/FOLLOW集合、消除左递归、悬空else | 17 |
| `tier2_deep/dp_comp_lr_parsing_deep.py` | LR(0)/SLR/LALR分析、分析表构造、移进规约冲突 | 15 |
| `tier2_deep/dp_comp_semantic_analysis_type_systems_deep.py` | 符号表作用域、类型检查、Hindley-Milner类型推断 | 15 |
| `tier2_deep/dp_comp_ir_codegen_optimization_deep.py` | 三地址码、SSA、CFG、常量折叠死代码消除、图着色寄存器分配 | 15 |
| `tier3_social_hire/sc_comp_language_design_tradeoff_judgment.py` | 语言设计与实现权衡判断(解析策略/DSL设计) | 14 |
| `tier3_social_hire/sc_comp_optimization_debugging_judgment.py` | 编译优化引入的bug定位、JIT vs AOT判断 | 14 |

## 新增:本科课件式完整教材

沿用CS基础五件套的做法,新增 `lectures/01-textbook.md`——七章连贯教科书叙述,按词法分析→语法分析基础→LR分析→语义分析类型系统→中间表示代码生成优化→语言设计权衡判断→编译优化调试判断的顺序系统教学。**建议先读完 `01-textbook.md` 再做下面的追问链/场景判断自测。**

## 数据结构:DeepPoint + ScenarioPoint(含 explain 字段)

```python
import sys
sys.path.insert(0, "learning/compilers-mastery/src")
from compilers_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/compilers-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部是经典编译原理教材共识(Aho-Lam-Sethi-Ullman《编译原理》"龙书"/Benjamin Pierce《Types and Programming Languages》体系里的标准表述),不需要追赶前沿论文,但仍要求核实具体术语和算法描述的准确性。`real_world_link` 字段全部留空。

至此,老师要求的六个CS科班理论专题第五个完成,最后一个(安全密码学基础)陆续推进。

## 补充:直接问答自测

`qa-practice.md`——一份轻量级补充练习(24题,不含追问链评分逻辑),每题一个问题配一个完整参考答案,和上面108点的3层追问链形式不同,提问角度也刻意避开了源码里的trigger原文,适合刷完追问链之后做最后一遍考前快速过关。
