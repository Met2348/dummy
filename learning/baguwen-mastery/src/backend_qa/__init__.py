"""后端通用八股问答库聚合（8 类，128 题）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import categories, grade, quiz  # noqa: E402

from .qbank_os import BANK as _os
from .qbank_network import BANK as _network
from .qbank_database import BANK as _database
from .qbank_jvm_concurrency import BANK as _jvm_concurrency
from .qbank_distributed_systems import BANK as _distributed_systems
from .qbank_cache_storage import BANK as _cache_storage
from .qbank_design_patterns import BANK as _design_patterns
from .qbank_linux_ops import BANK as _linux_ops

ALL_QA = (
    _os + _network + _database + _jvm_concurrency + _distributed_systems
    + _cache_storage + _design_patterns + _linux_ops
)


def _self_test() -> None:
    assert len(ALL_QA) == 128, len(ALL_QA)
    assert len(categories(ALL_QA)) == 8, len(categories(ALL_QA))
    ids = [qa.id for qa in ALL_QA]
    assert len(ids) == len(set(ids)), "backend_qa 内存在重复 id"
    assert all(i.startswith("be-") for i in ids)
    qs = [qa.q for qa in ALL_QA]
    assert len(qs) == len(set(qs)), "backend_qa 内存在重复题目文本"
    print(f"[PASS] backend_qa: {len(ALL_QA)}题 / {len(categories(ALL_QA))}类 汇总完整")


if __name__ == "__main__":
    _self_test()
