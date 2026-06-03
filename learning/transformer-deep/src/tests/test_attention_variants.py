"""4 种 attention 同 input shape / 数值合法性测试."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mha import MHA       # noqa: E402
from mqa import MQA       # noqa: E402
from gqa import GQA       # noqa: E402
from mla import MLA       # noqa: E402


def test_all_variants_output_shape():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 64)
    for cls, kwargs in [
        (MHA, {"d_model": 64, "n_head": 8}),
        (MQA, {"d_model": 64, "n_head": 8}),
        (GQA, {"d_model": 64, "n_head": 8, "n_kv_head": 2}),
    ]:
        model = cls(**kwargs)
        out = model(x)
        assert out.shape == (2, 8, 64), f"{cls.__name__}: {out.shape}"
        assert torch.isfinite(out).all(), f"{cls.__name__} NaN/Inf"

    mla = MLA(d_model=64, n_head=8, d_low=16)
    out, c = mla(x)
    assert out.shape == (2, 8, 64), f"MLA: {out.shape}"
    assert c.shape == (2, 8, 16), f"MLA c: {c.shape}"


def test_param_count_decreases():
    """KV cache 大小:  MHA > GQA > MQA → MLA(d_low) 取决."""
    d = 256
    h = 16
    mha = MHA(d_model=d, n_head=h)
    mqa = MQA(d_model=d, n_head=h)
    gqa = GQA(d_model=d, n_head=h, n_kv_head=4)
    mla = MLA(d_model=d, n_head=h, d_low=32)
    # 参数：MHA = 4 d²; MQA = ~2.25 d²; GQA = ~2.5 d²
    n_mha = sum(p.numel() for p in mha.parameters())
    n_mqa = sum(p.numel() for p in mqa.parameters())
    n_gqa = sum(p.numel() for p in gqa.parameters())
    n_mla = sum(p.numel() for p in mla.parameters())
    print(f"params: MHA={n_mha}  GQA={n_gqa}  MQA={n_mqa}  MLA={n_mla}")
    assert n_mqa < n_gqa < n_mha


def test_causal_mask_works():
    from common import causal_mask    # noqa: E402
    x = torch.randn(1, 4, 64)
    mask = causal_mask(4)
    m = MHA(d_model=64, n_head=8)
    out_masked = m(x, mask=mask)
    out_unmasked = m(x)
    assert not torch.allclose(out_masked, out_unmasked), "mask should change output"
