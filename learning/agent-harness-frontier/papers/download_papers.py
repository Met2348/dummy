#!/usr/bin/env python3
"""下载 agent-harness 教学系列的核心文献 PDF（arXiv）。

本库主题 = **Agent Harness / 脚手架**：把 LLM 变成 agent 的软件层
（控制循环、工具接口/ACI、上下文工程/记忆、编码/Web agent、harness 评测、可靠性）。
贯穿全库的论点：**Agent = Model + Harness**——能力/可信度有一大半压在 harness 上。

- 可复现：clone 后 `python download_papers.py` 一键重下（幂等，已存在则跳过）。
- 对 arXiv 友好：浏览器 UA + 每篇之间 sleep（默认 3s）+ 429 退避重试。
- 校验：必须以 %PDF 开头且 > 10KB 才算成功，否则删档并标 FAIL（天然过滤臆造/错误 ID）。
- PDF 本体被 .gitignore（二进制大，不进库）；本脚本 + INDEX.md 进库即可复现。
- 与 ../../auto-research-frontier 的 74 篇**零重叠**（建库前已用 74 个 ID 做 diff 核验）。
- 配比说明：约 30 篇 2022–2024 基石（ReAct/Reflexion/MemGPT/SWE-agent/Toolformer/WebArena…，
  harness 的"脊柱"，不可省）+ 约 44 篇 2025–2026 前沿。

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

# (主题, arXiv id, 文件名 slug[, 可选完整 URL]) —— 主题字母对应 INDEX.md / PROGRESS.md 的分组
# 第 4 个元素可给完整 PDF URL，不给则默认 https://arxiv.org/pdf/<id>
PAPERS: list[tuple] = [
    # ===== A. 综述 / 框架与定义（8）=====
    ("A", "2603.25723", "natural-language-agent-harnesses"),
    ("A", "2601.11100", "recreate-experience-driven-domain-agents"),
    ("A", "2604.03515", "inside-the-scaffold-coding-agent-taxonomy"),
    ("A", "2601.01743", "ai-agent-systems-architectures-applications-evaluation"),
    ("A", "2507.13334", "survey-of-context-engineering-for-llms"),
    ("A", "2512.13564", "memory-in-the-age-of-ai-agents"),
    ("A", "2308.11432", "survey-on-llm-based-autonomous-agents"),
    ("A", "2604.08224", "externalization-in-llm-agents-review"),
    # ===== B. 控制循环 / 推理-行动范式（10，含 7 篇 canon 脊柱）=====
    ("B", "2210.03629", "react-reasoning-and-acting"),
    ("B", "2303.11366", "reflexion-verbal-reinforcement-learning"),
    ("B", "2305.10601", "tree-of-thoughts"),
    ("B", "2310.04406", "language-agent-tree-search-lats"),
    ("B", "2305.04091", "plan-and-solve-prompting"),
    ("B", "2207.05608", "inner-monologue-embodied-planning"),
    ("B", "2303.17651", "self-refine-iterative-self-feedback"),
    ("B", "2507.11633", "general-modular-harness-gaming-agents"),
    ("B", "2604.11378", "from-agent-loops-to-structured-graphs"),
    ("B", "2602.01664", "flowsteer-reinforced-workflow-orchestration"),
    # ===== C. 工具接口 / Agent-Computer Interface（8）=====
    ("C", "2405.15793", "swe-agent-agent-computer-interface"),
    ("C", "2302.04761", "toolformer-self-taught-tool-use"),
    ("C", "2307.16789", "toolllm-toolbench-16000-apis"),
    ("C", "2305.15334", "gorilla-llm-connected-massive-apis"),
    ("C", "2409.00920", "toolace-winning-function-calling"),
    ("C", "2411.15399", "less-is-more-function-calling-edge"),
    ("C", "2507.21428", "memtool-short-term-memory-tool-calling"),
    ("C", "2509.26553", "funcbenchgen-contamination-free-eval"),
    # ===== D. 上下文工程 / 记忆（16，最大组）=====
    ("D", "2310.08560", "memgpt-llms-as-operating-systems"),
    ("D", "2305.10250", "memorybank-long-term-memory"),
    ("D", "2504.19413", "mem0-production-long-term-memory"),
    ("D", "2506.15841", "mem1-synergize-memory-reasoning"),
    ("D", "2509.25911", "mem-alpha-rl-memory-construction"),
    ("D", "2502.12110", "a-mem-agentic-memory-zettelkasten"),
    ("D", "2507.02259", "memagent-multiconv-rl-memory"),
    ("D", "2511.07327", "iterresearch-interaction-scaling"),
    ("D", "2510.24699", "agentfold-proactive-context-folding"),
    ("D", "2510.00615", "acon-context-compression-agents"),
    ("D", "2510.12635", "memory-as-action-context-curation"),
    ("D", "2511.02805", "memsearcher-reason-search-manage-memory"),
    ("D", "2601.04786", "agentocr-optical-self-compression"),
    ("D", "2601.01885", "agentic-memory-unified-ltm-stm"),
    ("D", "2602.02486", "re-trac-recursive-trajectory-compression"),
    ("D", "2606.10209", "less-context-better-agents"),
    # ===== E. 编码 / SWE Agent 集成系统（10；SWE-agent 归 C 锚点）=====
    ("E", "2511.03690", "openhands-software-agent-sdk"),
    ("E", "2512.10398", "confucius-code-agent"),
    ("E", "2510.18779", "kat-coder-technical-report"),
    ("E", "2506.19290", "skywork-swe"),
    ("E", "2402.01030", "codeact-executable-code-actions"),
    ("E", "2407.01489", "agentless"),
    ("E", "2501.05040", "swe-fixer"),
    ("E", "2406.11638", "masai-modular-architecture-swe"),
    ("E", "2503.14269", "dars-dynamic-action-resampling"),
    ("E", "2404.05427", "autocoderover"),
    # ===== F. Web / 计算机使用 / GUI Agent（7）=====
    ("F", "2307.13854", "webarena"),
    ("F", "2401.13649", "visualwebarena"),
    ("F", "2404.07972", "osworld"),
    ("F", "2501.12326", "ui-tars"),
    ("F", "2401.13919", "webvoyager"),
    ("F", "2306.06070", "mind2web"),
    ("F", "2603.27490", "agentswing-parallel-context-routing"),
    # ===== G. Harness 评测 / scaffold-aware eval（9）=====
    ("G", "2605.27922", "harness-bench-measuring-harness-effects"),
    ("G", "2310.06770", "swe-bench-resolve-github-issues"),
    ("G", "2601.11868", "terminal-bench-cli-agents"),
    ("G", "2308.03688", "agentbench-evaluating-llms-as-agents"),
    ("G", "2406.12045", "tau-bench-tool-agent-user"),
    ("G", "2311.12983", "gaia-general-ai-assistants"),
    ("G", "2507.00014", "swe-bench-cl-continual-learning"),
    ("G", "2512.18470", "swe-evo-long-horizon-software-evolution"),
    ("G", "2601.11044", "agencybench-1m-token-autonomous-agents"),
    # ===== H. 可靠性 / 安全 / 可观测 / 沙箱（6）=====
    ("H", "2505.03574", "llamafirewall-guardrail-system"),
    ("H", "2406.13352", "agentdojo-prompt-injection-eval"),
    ("H", "2512.12806", "fault-tolerant-sandboxing-coding-agents"),
    ("H", "2512.01295", "systems-security-foundations-agentic-computing"),
    ("H", "2508.11027", "hell-or-high-water-agentic-recovery"),
    ("H", "2509.03312", "agentracer-failure-attribution"),
    # 备选/缓冲（已核验，若上面有下载失败可顶上）：
    #   D 2601.02553 simplemem · D 2509.24704 memgen · F 2604.01664 contextbudget · G 2602.22769 ama-bench
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
