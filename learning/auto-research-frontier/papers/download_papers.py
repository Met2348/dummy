#!/usr/bin/env python3
"""下载 auto-research 教学系列的核心文献 PDF（arXiv）。

- 可复现：clone 后 `python download_papers.py` 一键重下（幂等，已存在则跳过）。
- 对 arXiv 友好：浏览器 UA + 每篇之间 sleep（默认 3s）+ 429 退避重试。
- 校验：必须以 %PDF 开头且 > 10KB 才算成功，否则删档并标 FAIL。
- PDF 本体被 .gitignore（二进制大，不进库）；本脚本 + INDEX.md 进库即可复现。

用法：
    python download_papers.py            # 下载全部
    python download_papers.py --list     # 只打印清单不下载
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# (主题, arXiv id, 文件名 slug) —— 主题字母对应 INDEX.md / CURRICULUM.md 的分组
PAPERS: list[tuple[str, str, str]] = [
    # A. 综述 / 全景
    ("A", "2505.13259", "survey-automation-to-autonomy"),
    ("A", "2503.24047", "survey-scientific-intelligence-agents"),
    ("A", "2506.18096", "survey-deep-research-agents-roadmap"),
    ("A", "2507.01903", "survey-ai4research"),
    ("A", "2508.12752", "survey-deep-research-autonomous"),
    ("A", "2512.02038", "survey-deep-research-systematic"),
    # B. 端到端 AI Scientist
    ("B", "2408.06292", "ai-scientist-v1"),
    ("B", "2504.08066", "ai-scientist-v2-tree-search"),
    ("B", "2502.18864", "google-ai-co-scientist"),
    ("B", "2501.04227", "agent-laboratory"),
    ("B", "2505.18705", "ai-researcher-hkuds"),
    ("B", "2505.16938", "novelseek-internagent"),
    ("B", "2503.18102", "agentrxiv-collaborative"),
    ("B", "2411.11910", "aigs-automated-falsification"),
    # C. 创意 / 假设生成
    ("C", "2409.04109", "can-llms-generate-novel-ideas"),
    ("C", "2506.20803", "ideation-execution-gap"),
    ("C", "2404.07738", "researchagent-iterative-ideation"),
    ("C", "2412.14141", "llm-combinatorial-creativity"),
    ("C", "2511.02238", "deep-ideation-concept-network"),
    # D. Deep Research / 文献综述合成
    ("D", "2402.14207", "storm-wikipedia-from-scratch"),
    ("D", "2408.15232", "co-storm-unknown-unknowns"),
    ("D", "2411.14199", "openscholar-ai2"),
    ("D", "2508.20033", "deepscholar-bench"),
    # E. 评测 / benchmark
    ("E", "2504.01848", "paperbench-openai"),
    ("E", "2410.07095", "mle-bench-openai"),
    ("E", "2411.15114", "re-bench-metr"),
    ("E", "2410.05080", "scienceagentbench"),
    ("E", "2502.14499", "mlgym-meta"),
    ("E", "2505.19955", "mlr-bench"),
    ("E", "2310.03302", "mlagentbench"),
    ("E", "2510.21652", "astabench-ai2"),
    ("E", "2409.11363", "core-bench-reproducibility"),
    ("E", "2407.13168", "scicode-benchmark"),
    # F. 自我改进 / 自动算法发现
    ("F", "2505.22954", "darwin-godel-machine"),
    ("F", "2408.08435", "adas-agentic-system-design"),
    # G. 批判 / 陷阱
    ("G", "2502.14297", "critique-wishful-thinking-ari"),
    ("G", "2509.08713", "critique-hidden-pitfalls"),
    ("G", "2506.01372", "critique-fail-without-implementation"),
    ("G", "2601.03315", "critique-why-not-scientists-yet"),
    # 前沿追踪（2026）
    ("H", "2511.16931", "omniscientist-coevolving"),
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HERE = Path(__file__).resolve().parent


def fetch(arxiv_id: str, dest: Path, *, retries: int = 3) -> tuple[bool, str]:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                return False, f"not a PDF (got {data[:16]!r})"
            if len(data) < 10_000:
                return False, f"too small ({len(data)} bytes)"
            dest.write_bytes(data)
            return True, f"{len(data) // 1024} KB"
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(10 * attempt)  # 退避
                continue
            return False, f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            if attempt < retries:
                time.sleep(5 * attempt)
                continue
            return False, f"{type(e).__name__}: {e}"
    return False, "exhausted retries"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只打印清单不下载")
    ap.add_argument("--delay", type=float, default=3.0, help="每篇间隔秒（arXiv 礼貌）")
    args = ap.parse_args()

    if args.list:
        for theme, aid, slug in PAPERS:
            print(f"[{theme}] {aid}  {slug}")
        print(f"\n共 {len(PAPERS)} 篇")
        return 0

    ok, fail = [], []
    for i, (theme, aid, slug) in enumerate(PAPERS, 1):
        dest = HERE / f"{aid}-{slug}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[{i:2}/{len(PAPERS)}] SKIP {aid} {slug} (已存在)")
            ok.append(aid)
            continue
        good, msg = fetch(aid, dest)
        tag = "OK  " if good else "FAIL"
        print(f"[{i:2}/{len(PAPERS)}] {tag} {aid} {slug} ({msg})")
        (ok if good else fail).append(aid)
        if i < len(PAPERS):
            time.sleep(args.delay)

    print(f"\n下载完成：{len(ok)}/{len(PAPERS)} 成功。")
    if fail:
        print(f"失败 {len(fail)} 篇（需核对 ID）：{', '.join(fail)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
