"""GPT-mini KV cache 增量解码一致性测试."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpt_mini import GPTMini, GPTMiniConfig


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = GPTMiniConfig(vocab_size=64, n_layer=2, n_head=4, n_kv=2,
                        d_model=64, d_ff=128, max_seq=32)
    model = GPTMini(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        # 全 forward
        out_full = model(x)

        # 增量 forward (用 KV cache)
        cache = [None] * cfg.n_layer
        all_logits = []
        for t in range(x.shape[1]):
            logits, cache = model(x[:, t:t+1], cache=cache)
            all_logits.append(logits)
        out_cached = torch.cat(all_logits, dim=1)

    diff = (out_full - out_cached).abs().max().item()
    assert diff < 1e-3, f"KV cache 不一致: diff={diff}"


def test_generate_runs():
    cfg = GPTMiniConfig(vocab_size=64, n_layer=2, n_head=4, n_kv=2,
                        d_model=64, d_ff=128, max_seq=32)
    model = GPTMini(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(x, max_new=8)
    assert out.shape == (1, 12)
    assert out[:, :4].equal(x)
