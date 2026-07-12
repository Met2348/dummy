"""2026 前沿补强聚合（9 类 DeepPoint + 1 类 ScenarioPoint，目标 >=200 点）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import categories  # noqa: E402

from .dp_reasoning_testtime import BANK as _reasoning_testtime
from .dp_agentic_production import BANK as _agentic_production
from .dp_alignment_oversight import BANK as _alignment_oversight
from .dp_interpretability_2026 import BANK as _interpretability_2026
from .dp_multimodal_vla import BANK as _multimodal_vla
from .dp_frontier_oss_models import BANK as _frontier_oss_models
from .dp_llm_infra_2026 import BANK as _llm_infra_2026
from .dp_data_scaling_2026 import BANK as _data_scaling_2026
from .dp_rag_tooling_2026 import BANK as _rag_tooling_2026
from .sc_engineering_judgment import BANK as _engineering_judgment

ALL_DP = (
    _reasoning_testtime + _agentic_production + _alignment_oversight
    + _interpretability_2026 + _multimodal_vla + _frontier_oss_models
    + _llm_infra_2026 + _data_scaling_2026 + _rag_tooling_2026
)
ALL_SP = list(_engineering_judgment)


def _self_test() -> None:
    assert len(ALL_DP) >= 160, len(ALL_DP)
    assert len(ALL_SP) >= 18, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 200, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 9, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 1, len(categories(ALL_SP))
    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "frontier_deep ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "frontier_deep ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-fr-") for i in dp_ids)
    assert all(i.startswith("sc-fr-") for i in sp_ids)
    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "frontier_deep ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "frontier_deep ALL_SP 内存在重复 trigger"
    print(
        f"[PASS] frontier_deep: {len(ALL_DP)}个DeepPoint(9类) + "
        f"{len(ALL_SP)}个ScenarioPoint(1类) 汇总完整，合计{len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
