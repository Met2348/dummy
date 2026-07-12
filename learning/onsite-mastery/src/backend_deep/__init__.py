"""后端深水区聚合（5 类，约 64 点）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import categories  # noqa: E402

from .dp_os_internals import BANK as _os_internals
from .dp_network_internals import BANK as _network_internals
from .dp_database_internals import BANK as _database_internals
from .dp_distributed_systems import BANK as _distributed_systems
from .dp_cache_concurrency import BANK as _cache_concurrency

ALL_DP = (
    _os_internals + _network_internals + _database_internals
    + _distributed_systems + _cache_concurrency
)


def _self_test() -> None:
    assert len(ALL_DP) >= 55, len(ALL_DP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    ids = [dp.id for dp in ALL_DP]
    assert len(ids) == len(set(ids)), "backend_deep 内存在重复 id"
    assert all(i.startswith("dp-be-") for i in ids)
    triggers = [dp.trigger for dp in ALL_DP]
    assert len(triggers) == len(set(triggers)), "backend_deep 内存在重复 trigger"
    print(f"[PASS] backend_deep: {len(ALL_DP)}点 / {len(categories(ALL_DP))}类 汇总完整")


if __name__ == "__main__":
    _self_test()
