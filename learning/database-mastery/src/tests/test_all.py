"""跑全部 Database Mastery 模块的 _self_test():
deep_common 自检 + 7 个 tier 模块自检 + 总聚合校验。
"""
from __future__ import annotations

import importlib
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

MODULES = [
    "deep_common",
    # tier1 浅:关系模型/SQL基础与事务并发基础
    "tier1_shallow.dp_db_relational_model_sql_basics",
    "tier1_shallow.dp_db_transactions_isolation_basics",
    # tier2 深:索引/存储引擎/分布式数据库深水
    "tier2_deep.dp_db_indexing_query_optimization",
    "tier2_deep.dp_db_storage_engine_concurrency",
    "tier2_deep.dp_db_distributed_database",
    # tier3 社招级别:选型容量与生产故障判断
    "tier3_social_hire.sc_db_selection_capacity_judgment",
    "tier3_social_hire.sc_db_production_incident_judgment",
    # 总聚合校验
    "database_mastery",
]


def main() -> int:
    passed = 0
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            mod._self_test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"=== {passed}/{len(MODULES)} modules passed ===")
    return 0 if passed == len(MODULES) else 1


if __name__ == "__main__":
    sys.exit(main())
