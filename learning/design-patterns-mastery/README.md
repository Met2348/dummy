# Design Patterns Mastery —— 设计模式专题(浅→深→社招级别)

> 定位:老手要求的五个 CS 基础专题(软件工程/设计模式/数据库/网络/OS)队列的第二个,组织方式与 `software-engineering-mastery` 一致——`DeepPoint`/`ScenarioPoint` 都带 `explain` 系统性讲解字段,因为用户完全没系统学过设计模式。

## 组织轴:难度分层

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | GoF创建型模式(5种)、结构型模式(7种)基础认知 | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | 行为型模式(一)(二)(共11种)、现代模式与反模式深水 | 45 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 架构选型判断、过度设计边界判断 | 28 | ScenarioPoint |

**合计108点(80个DeepPoint + 28个ScenarioPoint)**,覆盖GoF全部23种经典模式,外加现代工程实践里的模式演化(DI容器/Repository/CQRS/函数式替代/反模式)。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_pat_creational_patterns.py` | Singleton、Factory Method、Abstract Factory、Builder、Prototype | 18 |
| `tier1_shallow/dp_pat_structural_patterns.py` | Adapter、Decorator、Proxy、Facade、Composite、Bridge、Flyweight | 17 |
| `tier2_deep/dp_pat_behavioral_patterns_one.py` | Strategy、Observer、Command、Template Method、Iterator | 15 |
| `tier2_deep/dp_pat_behavioral_patterns_two.py` | State、Chain of Responsibility、Mediator、Memento、Visitor、Interpreter | 15 |
| `tier2_deep/dp_pat_modern_patterns_antipatterns.py` | DI容器、Repository/Unit of Work、CQRS、函数式替代GoF、YAGNI/God Object反模式 | 15 |
| `tier3_social_hire/sc_pat_architecture_selection_judgment.py` | 场景化模式选型、模式组合判断、隐性模式识别、团队沟通成本判断 | 14 |
| `tier3_social_hire/sc_pat_overengineering_boundary_judgment.py` | YAGNI边界、拒绝引入模式的判断、rule of three重构时机 | 14 |

## 新增:本科课件式完整教材

老手进一步反馈:即便加了 `explain` 字段,108个知识点的追问链结构本质上仍然是"直接上面试",对完全没系统学过的人不够。因此新增 `lectures/01-textbook.md`——一份约21万字符、七章的完整教材式讲义,仿照GoF《设计模式》原书写法,按创建型模式→结构型模式→行为型模式(一)→行为型模式(二)→现代模式与反模式深水→架构选型判断→过度设计边界判断的顺序,用连贯的教科书叙述系统教会每个模式(解决什么问题/结构/伪代码示意/与相邻模式区别),不是知识点罗列或追问链。**建议学习顺序:先读完 `01-textbook.md` 建立系统认知,再回到下面的 DeepPoint/ScenarioPoint 用追问链和场景判断检验"是否真的能被追问住"。**

## 数据结构

```python
import sys
sys.path.insert(0, "learning/design-patterns-mastery/src")
from design_patterns_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/design-patterns-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部基于GoF《设计模式:可复用面向对象软件的基础》原书23种经典模式定义,以及现代工程实践共识(依赖注入/Repository模式/CQRS/函数式编程对GoF模式的替代关系)。`real_world_link` 字段全部留空——这批内容没有需要引用的具体本地文件或外部产品案例,不编造路径。

至此,五个CS基础专题队列(软件工程✓/设计模式✓/数据库/网络/OS)完成两个,后续三个陆续推进。
