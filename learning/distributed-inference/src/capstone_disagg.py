"""Capstone - 3-config disaggregated benchmark."""
from __future__ import annotations

from typing import List, Dict
import json

from disaggregated_mock import WorkloadConfig, HardwareConfig, colocate, disagg


def run_all(prompt_len: int = 1024, out_len: int = 128, n_reqs: int = 32) -> List[Dict]:
    w = WorkloadConfig(n_reqs=n_reqs, prompt_len=prompt_len, out_len=out_len)
    hw = HardwareConfig()
    return [colocate(w, hw), disagg(w, hw, cross_node=False), disagg(w, hw, cross_node=True)]


def to_md(rows: List[Dict]) -> str:
    head = "| config | TTFT ms | TPOT ms | tok/s | wall s |\n|---|---|---|---|---|"
    body = "\n".join(
        f"| {r['config']} | {r['ttft_ms']} | {r['tpot_ms']} | {r['tok_per_s']} | {r['wall_s']} |"
        for r in rows
    )
    return head + "\n" + body


if __name__ == "__main__":
    rows = run_all()
    print(to_md(rows))
    print("\nJSON:")
    print(json.dumps(rows, indent=2))
