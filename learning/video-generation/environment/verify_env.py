"""video-generation 专题环境自检. 运行: python environment/verify_env.py"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REQUIRED = ["nbformat", "numpy", "matplotlib"]
OPTIONAL = ["torch"]


def check_imports() -> list[str]:
    missing = []
    for pkg in REQUIRED:
        ok = importlib.util.find_spec(pkg) is not None
        print(f"  [{'OK ' if ok else 'MISS'}] {pkg}")
        if not ok:
            missing.append(pkg)
    for pkg in OPTIONAL:
        ok = importlib.util.find_spec(pkg) is not None
        print(f"  [{'OK ' if ok else 'opt '}] {pkg} (可选)")
    return missing


def check_nbformat_roundtrip() -> bool:
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell
    nb = new_notebook(cells=[new_code_cell("print('hi')")])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "smoke.ipynb"
        nbformat.write(nb, p)
        back = nbformat.read(p, as_version=4)
    ok = back.cells[0].source == "print('hi')"
    print(f"  [{'OK ' if ok else 'FAIL'}] nbformat 读写往返")
    return ok


def check_src_tools() -> bool:
    src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src))
    import video_diffusion as vd

    data = vd.make_trajectories(n=60, seed=1)
    data_ok = data.shape == (60, vd.T_FRAMES, 2)
    print(f"  [{'OK ' if data_ok else 'FAIL'}] make_trajectories {data.shape}")
    real_coh = vd.temporal_coherence(data)
    coh_ok = real_coh > 0
    print(f"  [{'OK ' if coh_ok else 'FAIL'}] temporal_coherence (真实基准 {real_coh:.3f})")

    ok = data_ok and coh_ok
    if importlib.util.find_spec("torch") is not None:
        # 时空模型连贯性应优于逐帧 (核心论点)
        cohs = {}
        for joint in (True, False):
            m = vd.build_video_denoiser(joint=joint, seed=0)
            _, sched = vd.train_video(m, data, epochs=200, seed=0)
            cohs[joint] = vd.temporal_coherence(vd.sample_videos(m, sched, n=40, seed=2))
        joint_better = cohs[True] < cohs[False]
        print(f"  [{'OK ' if joint_better else 'FAIL'}] 时空连贯优于逐帧 "
              f"(joint {cohs[True]:.3f} < per-frame {cohs[False]:.3f})")
        ok = ok and joint_better
    return ok


def main() -> int:
    print("== Part A: 依赖 ==")
    missing = check_imports()
    print("== Part B: nbformat ==")
    nb_ok = check_nbformat_roundtrip()
    print("== Part C: src 工具 ==")
    src_ok = check_src_tools()
    print("\n== 结论 ==")
    if missing:
        print(f"缺失依赖: {missing}\n  → pip install -r environment/requirements.txt")
        return 1
    if not (nb_ok and src_ok):
        print("检查未通过.")
        return 1
    print("全部通过 ✅  两个 notebook 可端到端运行.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
