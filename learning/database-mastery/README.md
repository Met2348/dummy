# Database Mastery —— 数据库专题(浅→深→社招级别)

> 定位:老手要求的五个 CS 基础专题(软件工程/设计模式/数据库/网络/OS)队列的第三个,组织方式与前两个专题一致——`DeepPoint`/`ScenarioPoint` 都带 `explain` 系统性讲解字段,因为用户完全没系统学过数据库。

## 组织轴:难度分层

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | 关系模型与SQL基础、事务与并发控制基础 | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | 索引与查询优化深水、存储引擎与并发控制深水、分布式数据库深水 | 45 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 选型与容量判断、生产故障判断 | 28 | ScenarioPoint |

**合计108点(80个DeepPoint + 28个ScenarioPoint)**。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_db_relational_model_sql_basics.py` | 关系代数、范式化1NF-BCNF、join类型与底层算法、NULL三值逻辑、子查询vs JOIN | 18 |
| `tier1_shallow/dp_db_transactions_isolation_basics.py` | ACID、四种隔离级别、脏读/不可重复读/幻读、悲观锁vs乐观锁 | 17 |
| `tier2_deep/dp_db_indexing_query_optimization.py` | B+树选型、聚簇/非聚簇索引、覆盖索引、最左前缀、索引失效场景、EXPLAIN解读 | 15 |
| `tier2_deep/dp_db_storage_engine_concurrency.py` | MVCC、InnoDB内部机制、WAL、redo/undo log、两阶段锁、死锁检测 | 15 |
| `tier2_deep/dp_db_distributed_database.py` | 分库分表、CAP定理、BASE理论、Paxos/Raft、2PC/TCC/Saga、NoSQL数据模型对比 | 15 |
| `tier3_social_hire/sc_db_selection_capacity_judgment.py` | SQL vs NoSQL选型、分片时机判断、读写分离一致性、缓存一致性判断 | 14 |
| `tier3_social_hire/sc_db_production_incident_judgment.py` | 慢查询定位、死锁排查、主从延迟处理、连接池耗尽排查、误删数据应急 | 14 |

## 数据结构

```python
import sys
sys.path.insert(0, "learning/database-mastery/src")
from database_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/database-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部基于经典数据库教材共识(《数据库系统概念》关于范式化/事务/并发控制的标准定义、MySQL InnoDB官方文档关于MVCC/锁机制的实现细节、CAP定理原始论文与后续澄清、Raft论文关于leader选举与日志复制的设计)。`real_world_link` 字段全部留空——不编造本地文件路径或不确定的公司案例。

至此,五个CS基础专题队列(软件工程✓/设计模式✓/数据库✓/网络/OS)完成三个,后续两个陆续推进。
