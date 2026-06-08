"""Module 3 final graduation capstone - 5 ckpt 全流程 + 报告生成.

dry-run 默认. --train 全跑 (24h+).
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(__file__))

from train_variant import VARIANTS, train
from bench_matrix import EXPECTED, METRICS, write_csv, ablation_breakdown, make_report_md
from visualize import loss_curve_text, spider_chart_text, make_loss_curve_data
from generations_compare import write_generations_md, MOCK_OUTPUTS, PROMPTS


def run_full_graduation(report_dir: str = "report",
                         train_real: bool = False):
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    print("=== Module 3 GRADUATION CAPSTONE ===\n")

    print("Step 1/5: Training 5 ckpt variants")
    for v in VARIANTS:
        train(VARIANTS[v], dry=not train_real)

    print("\nStep 2/5: Running benchmark matrix (using EXPECTED for dry-run)")
    results = EXPECTED
    csv_path = os.path.join(report_dir, "benchmarks.csv")
    write_csv(results, csv_path)
    print(f"  -> {csv_path}")

    print("\nStep 3/5: Generating ablation report")
    ablation = ablation_breakdown(results)
    abl_path = os.path.join(report_dir, "ablation.json")
    with open(abl_path, "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"  -> {abl_path}")

    print("\nStep 4/5: Visualization (text mode)")
    print(loss_curve_text(**make_loss_curve_data()))
    print()
    print(spider_chart_text(results, METRICS))

    print("\nStep 5/5: Markdown reports")
    report_md = os.path.join(report_dir, "report.md")
    make_report_md(results, report_md)
    print(f"  -> {report_md}")

    gen_md = os.path.join(report_dir, "generations.md")
    write_generations_md(gen_md)
    print(f"  -> {gen_md}")

    print("\n=== GRADUATION COMPLETE ===")
    print(f"Reports in: {report_dir}/")
    print("Next: git add report/ && git tag 造改-graduation")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report_dir", default="report")
    p.add_argument("--train", action="store_true")
    args = p.parse_args()
    run_full_graduation(args.report_dir, train_real=args.train)


if __name__ == "__main__":
    main()
