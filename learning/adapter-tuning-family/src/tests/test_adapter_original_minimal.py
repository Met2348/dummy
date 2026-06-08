from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from adapter_original_minimal import (  # noqa: E402
    AdapterConfig,
    adapter_total_params,
    full_finetune_total_params,
    houlsby_adapter_parameters,
    near_identity_demo,
    single_adapter_parameters,
    summarize,
)


def test_single_adapter_parameter_count() -> None:
    assert single_adapter_parameters(hidden_dim=768, adapter_dim=16) == 25_360


def test_houlsby_two_adapters_per_layer() -> None:
    cfg = AdapterConfig(layers=12, hidden_dim=768, adapter_dim=16)
    assert houlsby_adapter_parameters(cfg) == 608_640


def test_near_identity_initialization() -> None:
    x, y = near_identity_demo()
    assert x == y


def test_adapter_storage_scales_better_than_full_copies() -> None:
    cfg = AdapterConfig(layers=24, hidden_dim=1024, adapter_dim=64)
    full = full_finetune_total_params(backbone_params=330_000_000, num_tasks=9)
    adapted = adapter_total_params(
        backbone_params=330_000_000,
        cfg=cfg,
        num_tasks=9,
    )
    assert adapted < full


def test_summary_contract() -> None:
    result = summarize()
    assert result["single_adapter_gpt2_r16"] == 25_360
    assert result["houlsby_gpt2_r16"] == 608_640


if __name__ == "__main__":
    test_single_adapter_parameter_count()
    test_houlsby_two_adapters_per_layer()
    test_near_identity_initialization()
    test_adapter_storage_scales_better_than_full_copies()
    test_summary_contract()
    print("[PASS] adapter original minimal tests")
