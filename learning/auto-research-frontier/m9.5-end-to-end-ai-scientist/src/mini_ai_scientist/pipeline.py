"""把五阶段串成端到端闭环：ideation → experiment → analysis → writeup → review。

run_pipeline 跑一个 idea；run_all 跑一批，输出一张**诚实光谱表**——
有的 idea 真涨点(supported)、有的反而更差(refuted)、有的无定论(inconclusive)。
把每个 idea 的"事前自评 novelty"和"事后真实 verdict"并排，就是 ideation-execution gap 的现场。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .experiment import ExperimentConfig, run_repeated
from .ideation import Idea, generate_ideas, get_idea
from . import analysis, writeup, review


def _cfg(idea_overrides: Dict, epochs: int, n_samples: int, device: str) -> ExperimentConfig:
    return ExperimentConfig(epochs=epochs, n_samples=n_samples, device=device, **idea_overrides)


def run_pipeline(
    idea: Idea,
    out_dir,
    n_seeds: int = 5,
    epochs: int = 200,
    n_samples: int = 600,
    device: str = "cpu",
) -> Dict:
    seeds = list(range(n_seeds))
    base = run_repeated(_cfg(idea.baseline, epochs, n_samples, device), seeds)
    treat = run_repeated(_cfg(idea.treatment, epochs, n_samples, device), seeds)
    cmp = analysis.compare(base, treat)
    v = analysis.verdict(cmp)

    out = Path(out_dir)
    fig_name = f"fig-{idea.id}.png"
    has_fig = analysis.make_figure(f"idea: {idea.id}", cmp, str(out / fig_name))  # ASCII 标题免 CJK 字体警告
    report_path = writeup.write_report(
        idea, base, treat, cmp, v, fig_name if has_fig else None, out
    )
    rev = review.review(report_path.read_text(encoding="utf-8"), cmp)

    return {
        "idea_id": idea.id,
        "title": idea.title,
        "self_novelty": idea.self_novelty,   # 事前"想"
        "verdict": v,                        # 事后"做出来"
        "baseline_mean": cmp["baseline_mean"],
        "treatment_mean": cmp["treatment_mean"],
        "delta": cmp["delta"],
        "review_overall": rev["overall"],
        "report": str(report_path),
        "figure": str(out / fig_name) if has_fig else None,
    }


def run_all(
    out_dir,
    n_seeds: int = 5,
    epochs: int = 200,
    n_samples: int = 600,
    device: str = "cpu",
    idea_ids: Optional[List[str]] = None,
) -> List[Dict]:
    ideas = [get_idea(i) for i in idea_ids] if idea_ids else generate_ideas()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return [
        run_pipeline(idea, out_dir, n_seeds=n_seeds, epochs=epochs,
                     n_samples=n_samples, device=device)
        for idea in ideas
    ]
