"""Database Mastery 总聚合：浅→深→社招级别三层，合计7个文件108点。

老手要求的五个 CS 基础专题(软件工程/设计模式/数据库/网络/OS)队列的第三个，组织方式沿用
software-engineering-mastery/design-patterns-mastery 已验证的"讲解+追问链/场景判断"混合格式：

- tier1(浅)：关系模型与SQL基础(范式化/join类型/NULL三值逻辑)、事务与并发控制基础(ACID/四种
  隔离级别/脏读不可重复读幻读)
- tier2(深)：索引与查询优化深水(B+树/聚簇索引/覆盖索引/执行计划)、存储引擎与并发控制深水
  (MVCC/InnoDB内部/WAL/两阶段锁/死锁检测)、分布式数据库深水(分库分表/CAP定理/Paxos-Raft/
  2PC-TCC-Saga/NoSQL数据模型对比)
- tier3(社招级别)：数据库选型与容量判断(SQL vs NoSQL/分片时机/读写分离一致性)、生产故障判断
  (慢查询定位/死锁排查/主从延迟/误删数据应急)——无标准答案的资深判断
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_db_relational_model_sql_basics import BANK as _t1_sql  # noqa: E402
from tier1_shallow.dp_db_transactions_isolation_basics import BANK as _t1_txn  # noqa: E402
from tier2_deep.dp_db_indexing_query_optimization import BANK as _t2_idx  # noqa: E402
from tier2_deep.dp_db_storage_engine_concurrency import BANK as _t2_eng  # noqa: E402
from tier2_deep.dp_db_distributed_database import BANK as _t2_dist  # noqa: E402
from tier3_social_hire.sc_db_selection_capacity_judgment import BANK as _t3_sel  # noqa: E402
from tier3_social_hire.sc_db_production_incident_judgment import BANK as _t3_ops  # noqa: E402

ALL_DP = list(_t1_sql) + list(_t1_txn) + list(_t2_idx) + list(_t2_eng) + list(_t2_dist)
ALL_SP = list(_t3_sel) + list(_t3_ops)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:关系模型/SQL与事务并发基础", "kind": "dp",
     "modules": ({"name": "db_relational_model_sql_basics", "cat": "数据库基础一:关系模型与SQL基础", "bank": _t1_sql},
                 {"name": "db_transactions_isolation_basics", "cat": "数据库基础二:事务与并发基础", "bank": _t1_txn})},
    {"n": 2, "name": "deep", "label": "深:索引/存储引擎/分布式数据库深水", "kind": "dp",
     "modules": ({"name": "db_indexing_query_optimization", "cat": "数据库深水一:索引与查询优化深水", "bank": _t2_idx},
                 {"name": "db_storage_engine_concurrency", "cat": "数据库深水二:存储引擎与并发控制深水", "bank": _t2_eng},
                 {"name": "db_distributed_database", "cat": "数据库深水三:分布式数据库深水", "bank": _t2_dist})},
    {"n": 3, "name": "social_hire", "label": "社招级别:选型容量与生产故障判断", "kind": "sp",
     "modules": ({"name": "db_selection_capacity_judgment", "cat": "数据库社招级别一:选型与容量判断", "bank": _t3_sel},
                 {"name": "db_production_incident_judgment", "cat": "数据库社招级别二:生产故障判断", "bank": _t3_ops})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "database-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "database-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-db-") for i in dp_ids), "存在不以dp-db-开头的DeepPoint id"
    assert all(i.startswith("sc-db-") for i in sp_ids), "存在不以sc-db-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "database-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "database-mastery ALL_SP 内存在重复 trigger"

    assert all(len(dp.explain) >= 100 for dp in ALL_DP), "存在explain过短的DeepPoint"
    assert all(len(sp.explain) >= 100 for sp in ALL_SP), "存在explain过短的ScenarioPoint"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] database_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
