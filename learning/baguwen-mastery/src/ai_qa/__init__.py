"""AI/算法岗八股问答库聚合（15 类，178 题）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import categories, grade, quiz  # noqa: E402

from .qbank_optimization import BANK as _optimization
from .qbank_regularization import BANK as _regularization
from .qbank_normalization import BANK as _normalization
from .qbank_metrics import BANK as _metrics
from .qbank_transformer import BANK as _transformer
from .qbank_tokenizer_data import BANK as _tokenizer_data
from .qbank_peft import BANK as _peft
from .qbank_rlhf import BANK as _rlhf
from .qbank_moe import BANK as _moe
from .qbank_distributed_training import BANK as _distributed_training
from .qbank_inference_serving import BANK as _inference_serving
from .qbank_agent_rag import BANK as _agent_rag
from .qbank_interpretability import BANK as _interpretability
from .qbank_classic_ml import BANK as _classic_ml
from .qbank_system_design import BANK as _system_design

ALL_QA = (
    _optimization + _regularization + _normalization + _metrics + _transformer
    + _tokenizer_data + _peft + _rlhf + _moe + _distributed_training
    + _inference_serving + _agent_rag + _interpretability + _classic_ml
    + _system_design
)


def _self_test() -> None:
    assert len(ALL_QA) == 178, len(ALL_QA)
    assert len(categories(ALL_QA)) == 15, len(categories(ALL_QA))
    ids = [qa.id for qa in ALL_QA]
    assert len(ids) == len(set(ids)), "ai_qa 内存在重复 id"
    assert all(i.startswith("ai-") for i in ids)
    qs = [qa.q for qa in ALL_QA]
    assert len(qs) == len(set(qs)), "ai_qa 内存在重复题目文本"
    print(f"[PASS] ai_qa: {len(ALL_QA)}题 / {len(categories(ALL_QA))}类 汇总完整")


if __name__ == "__main__":
    _self_test()
