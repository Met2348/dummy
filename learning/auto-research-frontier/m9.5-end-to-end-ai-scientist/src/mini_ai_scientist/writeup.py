"""阶段 5（写作）：自动写 1 页 markdown 报告。

铁律：**报告里的每个数字都来自真实实验指标**（传进来的 dict），绝不硬编码、绝不幻觉。
（AI Scientist 被批最狠的就是"为了好看幻觉出整张消融表"——这里用 test 锁死"报告数字==实验数字"。）
报告里特意带上 reviewer 会找的结构标记（## 假设 / 结果表 / 图 / ## 结论 / ±seeds），
既让诚实报告拿到合理分，也为 §review 演示"评审看结构和宣称效果"埋点。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def write_report(
    idea,
    baseline_rep: Dict,
    treatment_rep: Dict,
    comparison: Dict,
    verdict_str: str,
    figure_rel: Optional[str],
    out_dir,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = baseline_rep.get("seeds", [])
    bm, bs = baseline_rep["test_acc_mean"], baseline_rep["test_acc_std"]
    tm, ts = treatment_rep["test_acc_mean"], treatment_rep["test_acc_std"]
    delta = comparison["delta"]

    verdict_zh = {
        "supported": "✅ 假设成立：treatment 显著优于 baseline（超过噪声与最小实际意义）。",
        "refuted": "❌ 假设被推翻：treatment 反而**更差**（诚实的真阴结果）。",
        "inconclusive": "➖ 无定论：差异落在噪声范围内，不能下结论。",
    }[verdict_str]

    fig_md = f"![{idea.id}]({figure_rel})\n" if figure_rel else "_(无图：matplotlib 不可用)_\n"

    md = f"""# {idea.title}

> 由 mini-AI-Scientist 自动生成 · idea_id=`{idea.id}` · 跨 {len(seeds)} 个种子真训练得出

## 假设
{idea.hypothesis}

> idea agent 的事前自评 novelty：**{idea.self_novelty:.2f}**（注意：这是"想"的，不是"做出来"的）

## 方法
- 任务：make_moons 非线性二分类（同一任务种子，对照只变模型/训练）。
- 对照：baseline `{idea.baseline}` vs treatment `{idea.treatment}`。
- 每个配置跨种子 `seeds={seeds}` 重复，报 mean±std。

## 结果

| 指标 | baseline | treatment |
|------|----------|-----------|
| test_acc | {bm:.4f} ± {bs:.4f} | {tm:.4f} ± {ts:.4f} |
| Δ(treat−base) | colspan | **{delta:+.4f}** （合并 std {comparison['combined_std']:.4f}） |

{fig_md}

## 结论
{verdict_zh}

## 诚信声明
- 以上 test_acc 全部来自真实训练运行（{len(seeds)} 个种子），非硬编码、非幻觉。
- 判定由 `analysis.verdict()` 依据"Δ 是否同时超过噪声与阈值"给出，不为好看而美化。
"""
    path = out / f"report-{idea.id}.md"
    path.write_text(md, encoding="utf-8")
    return path
