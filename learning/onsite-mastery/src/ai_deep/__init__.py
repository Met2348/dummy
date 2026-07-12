"""AI 深水区聚合（12 类，约 144 点）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import categories  # noqa: E402

from .dp_transformer_attention import BANK as _transformer_attention
from .dp_rlhf_alignment import BANK as _rlhf_alignment
from .dp_moe_systems import BANK as _moe_systems
from .dp_distributed_training import BANK as _distributed_training
from .dp_inference_serving import BANK as _inference_serving
from .dp_interpretability import BANK as _interpretability
from .dp_rag_agent import BANK as _rag_agent
from .dp_pretraining_data import BANK as _pretraining_data
from .dp_eval_safety import BANK as _eval_safety
from .dp_classic_ml_systemdesign import BANK as _classic_ml_systemdesign
from .dp_scaling_dynamics import BANK as _scaling_dynamics
from .dp_agent_harness import BANK as _agent_harness

ALL_DP = (
    _transformer_attention + _rlhf_alignment + _moe_systems
    + _distributed_training + _inference_serving + _interpretability
    + _rag_agent + _pretraining_data + _eval_safety
    + _classic_ml_systemdesign + _scaling_dynamics + _agent_harness
)


def _self_test() -> None:
    assert len(ALL_DP) >= 130, len(ALL_DP)
    assert len(categories(ALL_DP)) == 12, len(categories(ALL_DP))
    ids = [dp.id for dp in ALL_DP]
    assert len(ids) == len(set(ids)), "ai_deep 内存在重复 id"
    assert all(i.startswith("dp-ai-") for i in ids)
    triggers = [dp.trigger for dp in ALL_DP]
    assert len(triggers) == len(set(triggers)), "ai_deep 内存在重复 trigger"
    print(f"[PASS] ai_deep: {len(ALL_DP)}点 / {len(categories(ALL_DP))}类 汇总完整")


if __name__ == "__main__":
    _self_test()
