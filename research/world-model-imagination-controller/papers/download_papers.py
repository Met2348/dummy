#!/usr/bin/env python3
"""下载 world-model-imagination-controller 研究项目的文献 PDF（arXiv）。

本库主题 = **测试时想象预算的自适应分配**：world model 在行动前用视频生成
"想象"未来画面辅助决策，但"该不该想象/该不该采纳/什么时候停"目前分配得很
粗糙。围绕这个问题从 5 个角度调研：

- A. World Models / 基于模型的规划与决策（想象预算目前怎么被固定死的）
- B. 视频生成用于决策，及其计算成本（想象为什么贵）
- C. 自适应测试时计算分配的技术工具箱（别的领域怎么解决同构问题）
- D. 不确定性量化（决策收益怎么估计）
- E. 元推理 / 信息价值 / 最优停止理论（问题的数学锚点）
- F. 2026-07 二次深挖：直接竞争格局扫描 + VOC/最优停止理论补强（详见
  ../02-deep-gap-analysis.md，这一组是"找对研究问题"专项调研的产物）

- 可复现：clone 后 `python download_papers.py` 一键重下（幂等，已存在则跳过）。
- 对 arXiv 友好：浏览器 UA + 每篇之间 sleep（默认 3s）+ 429 退避重试。
- 校验：必须以 %PDF 开头且 > 10KB 才算成功，否则删档并标 FAIL（天然过滤臆造/错误 ID）。
- PDF 本体不进 git（只 add 本脚本 + INDEX.md，commit 时不用 git add -A，PDF 自然不入库）。
- 部分经典/理论文献无 arXiv 版本（如 Russell & Wefald 1991），不在本脚本下载列表里，
  引用信息见 INDEX.md。

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

# (track, arXiv id, 文件名 slug) —— track 字母对应 INDEX.md 的分组
PAPERS: list[tuple] = [
    # ===== A. World Models / 基于模型的规划与决策（16）=====
    ("A", "1803.10122", "world-models-ha-schmidhuber"),
    ("A", "1811.04551", "planet-latent-dynamics-planning"),
    ("A", "1912.01603", "dreamer-latent-imagination"),
    ("A", "2010.02193", "dreamerv2-discrete-world-models"),
    ("A", "2301.04104", "dreamerv3-diverse-domains"),
    ("A", "1911.08265", "muzero-planning-learned-model"),
    ("A", "2111.00210", "efficientzero-limited-data"),
    ("A", "2203.04955", "td-mpc"),
    ("A", "2310.16828", "td-mpc2-scalable-robust"),
    ("A", "2402.15391", "genie-generative-interactive-environments"),
    ("A", "2408.14837", "gamengen-diffusion-real-time-game-engine"),
    ("A", "2209.00588", "iris-transformers-sample-efficient"),
    ("A", "1906.08253", "mbpo-when-to-trust-your-model"),
    ("A", "2405.19014", "macura-uncertainty-aware-rollout"),
    ("A", "2602.08236", "avic-when-how-much-to-imagine"),
    ("A", "2606.02486", "ahead-latent-space-predictive-vla"),
    # ===== B. 视频生成用于决策，及其计算成本（11）=====
    ("B", "2302.00111", "unipi-universal-policies-video-generation"),
    ("B", "2310.10625", "video-language-planning"),
    ("B", "2310.08576", "avdc-actionless-videos-dense-correspondence"),
    ("B", "2310.06114", "unisim-interactive-real-world-simulators"),
    ("B", "2405.12399", "diamond-diffusion-world-modeling-atari"),
    ("B", "2405.17398", "vista-generalizable-driving-world-model"),
    ("B", "2501.03575", "cosmos-world-foundation-model-platform"),
    ("B", "2407.01392", "diffusion-forcing"),
    ("B", "2405.15223", "ivideogpt-interactive-scalable-world-models"),
    ("B", "2503.18942", "video-t1-test-time-scaling-video-generation"),
    ("B", "2505.17618", "evosearch-test-time-evolutionary-search"),
    # ===== C. 自适应测试时计算分配的技术工具箱（15）=====
    ("C", "2406.16858", "eagle2-dynamic-draft-trees"),
    ("C", "2404.16710", "layerskip-early-exit-self-speculative"),
    ("C", "2207.07061", "calm-confident-adaptive-language-modeling"),
    ("C", "1603.08983", "act-adaptive-computation-time"),
    ("C", "2107.05407", "pondernet"),
    ("C", "2110.14168", "training-verifiers-math-word-problems"),
    ("C", "2305.20050", "lets-verify-step-by-step"),
    ("C", "2408.03314", "scaling-llm-test-time-compute-optimally"),
    ("C", "2501.19393", "s1-simple-test-time-scaling"),
    ("C", "2407.21787", "large-language-monkeys-repeated-sampling"),
    ("C", "2305.10601", "tree-of-thoughts"),
    ("C", "2305.14992", "rap-reasoning-as-planning"),
    ("C", "2206.02380", "adaptive-rollout-length-model-based-rl"),
    ("C", "2501.16918", "infoprop-rollouts-model-based-rl"),
    ("C", "2510.10103", "refrain-adaptive-early-stopping-cot"),
    # ===== D. 不确定性量化（12）=====
    ("D", "1703.04977", "bayesian-dl-uncertainties-cv"),
    ("D", "1506.02142", "mc-dropout-bayesian-approximation"),
    ("D", "1612.01474", "deep-ensembles-predictive-uncertainty"),
    ("D", "2002.07650", "uncertainty-autoregressive-structured-prediction"),
    ("D", "2107.07511", "gentle-intro-conformal-prediction"),
    ("D", "2306.10193", "conformal-language-modeling"),
    ("D", "2402.03478", "hyperdm-single-model-epistemic-aleatoric"),
    ("D", "1805.12114", "pets-probabilistic-dynamics-models"),
    ("D", "2005.05960", "plan2explore-self-supervised-world-models"),
    ("D", "2512.05927", "c3-world-models-calibrated-uncertainty"),
    ("D", "2605.06222", "ffdc-when-to-trust-imagination"),
    ("D", "2604.25416", "biased-dreams-epistemic-uncertainty-limits"),
    # ===== E. 元推理 / 信息价值 / 最优停止理论（7）=====
    ("E", "1207.5879", "selecting-computations-theory-applications"),
    ("E", "2604.01434", "voimcp-value-of-information-pomdp"),
    ("E", "1804.05394", "deep-optimal-stopping"),
    ("E", "1112.5745", "bald-bayesian-active-learning"),
    ("E", "1003.3967", "adaptive-submodularity"),
    ("E", "2410.05563", "rational-metareasoning-for-llms"),
    ("E", "2412.20993", "certaindex-efficiently-scaling-llm-reasoning"),
    # 无 arXiv 版本，不下载，引用信息见 INDEX.md：
    #   A Genie 2 (DeepMind blog 2024) · B Sora technical report (OpenAI 2024)
    #   D PlanCP (NeurIPS 2023 proceedings) · E Russell & Wefald 1991 (AIJ) ·
    #   E Howard 1966 (IEEE Trans SSC) · E Chow/Robbins/Siegmund 1971 (book) ·
    #   E Zilberstein 1996 (AI Magazine)

    # ===== F. 2026-07 二次深挖：直接竞争格局 + VOC/最优停止理论补强（20）=====
    # 老师要求"找对研究问题"专项调研新增，全部经 WebFetch 逐篇核验（非二手转述）。
    # F1 组：2026年直接竞争/邻近论文（世界模型想象门控赛道，越来越拥挤）
    ("F", "2606.06476", "astra-thinking-with-imagination"),
    ("F", "2601.08955", "imagine-then-plan-adaptive-lookahead"),
    ("F", "2606.31132", "elastic-adaptive-test-time-compute-gcp"),
    ("F", "2606.26463", "finding-time-to-think-planning-budgets"),
    ("F", "2601.03822", "roi-reasoning-knapsack-inference"),
    ("F", "2510.18135", "world-in-world-closed-loop"),
    ("F", "2509.24387", "adanav-uncertainty-vln"),
    ("F", "2603.16673", "rarrl-when-should-robot-think"),
    ("F", "2601.03905", "agents-fail-leverage-world-model-foresight"),
    ("F", "2606.22813", "active-inference-test-time-scaling-law"),
    # F2 组：VOC / 最优停止 / model-based 规划理论补强（本次挖出的"理论盲区"证据）
    ("F", "2603.30031", "cognitive-friction-bounded-deliberation"),
    ("F", "2002.04335", "sezener-dayan-voc-in-mcts"),
    ("F", "2011.03506", "value-equivalence-principle-model-based-rl"),
    ("F", "1802.03654", "beyond-one-step-greedy-rl"),
    ("F", "1805.07956", "multiple-step-greedy-policies-rl"),
    ("F", "2011.04021", "hamrick-role-of-planning-model-based-drl"),
    ("F", "2506.17124", "when-can-model-free-rl-be-enough-for-thinking"),
    ("F", "1705.02670", "metacontrol-adaptive-imagination-based-optimization"),
    ("F", "2307.14993", "thinker-learning-to-plan-and-act"),
    ("F", "2310.01798", "llms-cannot-self-correct-reasoning-yet"),
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
