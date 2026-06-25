"""V2 测试：锁死本模块的"诚实性"——真 exec、真阳/真阴、不幻觉数字、评审可被刷。

这些 test 本身就是教学：它们断言的不是"代码不崩"，而是"这个 mini-AI-Scientist 不自欺"。
"""
from __future__ import annotations

import pathlib
import sys

# 让 pytest / 直跑都能 import 包（tests 在 src/tests/ 下 → parent.parent = src/）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mini_ai_scientist.experiment import ExperimentConfig, run_experiment, run_repeated
from mini_ai_scientist.ideation import get_idea
from mini_ai_scientist.pipeline import run_pipeline, run_all
from mini_ai_scientist.review import demonstrate_gaming

CPU = dict(device="cpu", n_samples=240, epochs=40)


def test_experiment_deterministic():
    """同 config 同种子 → 完全一致（指标是算出来的，不是随机/假的）。"""
    a = run_experiment(ExperimentConfig(depth=1, seed=7, **CPU))
    b = run_experiment(ExperimentConfig(depth=1, seed=7, **CPU))
    assert a == b


def test_experiment_is_real_not_constant():
    """depth=1 在 moons 上应显著强于 depth=0（线性）——证明指标真随 config 变，非硬编码。"""
    seeds = [0, 1, 2]
    lin = run_repeated(ExperimentConfig(depth=0, **CPU), seeds)
    mlp = run_repeated(ExperimentConfig(depth=1, **CPU), seeds)
    assert mlp["test_acc_mean"] > lin["test_acc_mean"] + 0.05


def test_report_numbers_match_experiment(tmp_path):
    """报告里的 treatment 数字必须等于实验真实返回值——杜绝幻觉数字。"""
    r = run_pipeline(get_idea("add-depth"), tmp_path, n_seeds=3, **CPU)
    report = pathlib.Path(r["report"]).read_text(encoding="utf-8")
    assert f"{r['treatment_mean']:.4f}" in report
    assert f"{r['baseline_mean']:.4f}" in report


def test_add_depth_is_supported(tmp_path):
    """真阳：加隐藏层在 moons 上应判 supported。"""
    r = run_pipeline(get_idea("add-depth"), tmp_path, n_seeds=3, **CPU)
    assert r["verdict"] == "supported"
    assert r["delta"] > 0


def test_honest_spectrum_not_all_supported(tmp_path):
    """诚实光谱：跑全部 idea，不能全是 supported（否则就是自欺）。

    具体地，crank-lr（lr=2.0）不该被判 supported——这是 ideation-execution gap 的现场。
    """
    results = run_all(tmp_path, n_seeds=3, **CPU)
    verdicts = {r["idea_id"]: r["verdict"] for r in results}
    assert any(v != "supported" for v in verdicts.values())
    assert verdicts["crank-lr"] != "supported"


def test_reviewer_is_gameable():
    """grading-its-own-homework：注水大假效果 比 诚实小真效果 评分更高。"""
    g = demonstrate_gaming()
    assert g["gamed"] is True
    assert g["rigged_big_fake"]["overall"] > g["honest_small_real"]["overall"]


def test_pipeline_end_to_end(tmp_path):
    """端到端冒烟：产出报告文件 + 图（matplotlib 在则有图）。"""
    r = run_pipeline(get_idea("widen"), tmp_path, n_seeds=2, **CPU)
    assert pathlib.Path(r["report"]).exists()
    assert r["verdict"] in {"supported", "refuted", "inconclusive"}


if __name__ == "__main__":   # 直跑兜底（与 harness 的脚本回退一致）
    import traceback
    import tempfile
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                # 给需要 tmp_path 的测试塞一个临时目录
                if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                    fn(pathlib.Path(tempfile.mkdtemp()))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{'OK' if fails == 0 else f'{fails} FAILED'}")
    raise SystemExit(1 if fails else 0)
