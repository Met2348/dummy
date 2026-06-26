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

# (主题, arXiv id, 文件名 slug[, 可选完整 URL]) —— 主题字母对应 INDEX.md / CURRICULUM.md 的分组
# 第 4 个元素可给完整 PDF URL（如 bioRxiv），不给则默认 https://arxiv.org/pdf/<id>
PAPERS: list[tuple] = [
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

    # ===== 第二批扩充（2026-06-26，老师要求 ≥70 篇：更新、更权威、侧重 why + inspires us）=====
    # A. 综述 / 全景（补两篇最权威新综述）
    ("A", "2508.21148", "survey-scientific-llms-data-to-agent"),
    ("A", "2501.04306", "survey-llm4sr"),
    # B. 端到端 / 多智能体系统（补 5 篇）
    ("B", "2411.00816", "cycleresearcher-automated-review"),
    ("B", "2505.13400", "robin-futurehouse-discovery"),
    ("B", "2506.22653", "ursa-universal-research-agent"),
    ("B", "2509.26603", "deepscientist-frontier-findings"),
    ("B", "2506.15692", "mle-star-google-ml-engineering"),
    # C. 创意 / 假设生成（补 5 篇）
    ("C", "2409.05556", "sciagents-graph-reasoning"),
    ("C", "2412.17767", "researchtown-community-simulator"),
    ("C", "2404.04326", "hypogenic-hypothesis-generation"),
    ("C", "2410.09403", "many-heads-multi-agent-ideation"),
    ("C", "2506.08140", "autosdt-data-driven-discovery"),
    # D. Deep Research / 检索综合（补 4 篇）
    ("D", "2409.13740", "paperqa2-superhuman-synthesis"),
    ("D", "2504.03160", "deepresearcher-rl-realworld"),
    ("D", "2504.21776", "webthinker-deep-research"),
    ("D", "2507.02592", "websailor-web-agent"),
    # E. 评测 / Benchmark（补 7 篇）
    ("E", "2407.01725", "discoverybench-data-driven"),
    ("E", "2409.07440", "super-research-repositories"),
    ("E", "2505.24785", "exp-bench-ai-experiments"),
    ("E", "2506.02314", "researchcodebench-novel-code"),
    ("E", "2503.00096", "bixbench-computational-biology"),
    ("E", "2510.27598", "innovatorbench-llm-research"),
    ("E", "2504.11524", "hypobench-hypothesis-benchmark"),
    # F. 自我改进 / 自动算法发现（补 6 篇）
    ("F", "2506.13131", "alphaevolve-deepmind"),
    ("F", "2506.10943", "seal-self-adapting-lms"),
    ("F", "2410.04444", "godel-agent-self-referential"),
    ("F", "2404.18400", "llm-sr-equation-discovery"),
    ("F", "2505.22451", "ai-mathematician-frontier-math"),
    ("F", "2510.14150", "codeevolve-open-evolutionary"),
    # G. 批判 / 立场（补 1 篇）
    ("G", "2508.16613", "critique-biomedical-acceleration-limits"),
    # H. 前沿生态（补 1 篇）
    ("H", "2508.15126", "aixiv-ai-scientist-ecosystem"),
    # I. 域内落地发现（新增组：把"会做科研"落到化学/生物/材料真实发现）
    ("I", "2304.05376", "chemcrow-chemistry-tools"),
    ("I", "2508.02956", "sparksmatter-materials-discovery"),
    ("I", "2509.06917", "paper2agent-reproducible-agents"),
    # 注：Virtual Lab（Swanson/Zou, Nature 2025）仅在 bioRxiv，反爬返回 HTML 非 PDF，未纳入精读集。
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HERE = Path(__file__).resolve().parent


def fetch(url: str, dest: Path, *, retries: int = 3) -> tuple[bool, str]:
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
        for entry in PAPERS:
            theme, aid, slug = entry[:3]
            print(f"[{theme}] {aid}  {slug}")
        print(f"\n共 {len(PAPERS)} 篇")
        return 0

    ok, fail = [], []
    for i, entry in enumerate(PAPERS, 1):
        theme, aid, slug = entry[:3]
        url = entry[3] if len(entry) > 3 else f"https://arxiv.org/pdf/{aid}"
        dest = HERE / f"{aid}-{slug}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[{i:2}/{len(PAPERS)}] SKIP {aid} {slug} (已存在)")
            ok.append(aid)
            continue
        good, msg = fetch(url, dest)
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
