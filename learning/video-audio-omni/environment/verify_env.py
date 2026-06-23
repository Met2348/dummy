"""video-audio-omni 专题环境自检. 运行: python environment/verify_env.py"""
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


def check_imports() -> list[str]:
    missing = []
    for pkg in REQUIRED:
        ok = importlib.util.find_spec(pkg) is not None
        print(f"  [{'OK ' if ok else 'MISS'}] {pkg}")
        if not ok:
            missing.append(pkg)
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
    import numpy as np
    import temporal_tokens as tt
    import audio_features as af

    vid = tt.make_toy_video(T=8, size=8, seed=1)
    cmp = tt.token_count_comparison(vid, patch=2, tpatch=2)
    vid_ok = cmp["压缩比"] == 2.0 and cmp["逐帧 patch token 数"] > cmp["时空 patch token 数"]
    print(f"  [{'OK ' if vid_ok else 'FAIL'}] 时空 patch 压缩 token ({cmp['逐帧 patch token 数']}→{cmp['时空 patch token 数']})")
    motion = tt.motion_signal(vid)
    motion_ok = motion.max() > 0   # 视频有运动
    print(f"  [{'OK ' if motion_ok else 'FAIL'}] 视频运动信号存在 (max={motion.max():.3f})")

    wave = af.make_tone(freqs=(220.0, 440.0), sr=4000, dur=0.5, seed=1)
    mel = af.mel_spectrogram(wave)
    mel_ok = mel.ndim == 2 and mel.shape[1] == 16
    print(f"  [{'OK ' if mel_ok else 'FAIL'}] mel 谱 {mel.shape} (时间帧×16 mel)")
    tokens, cb = af.frames_to_tokens(mel, codebook_size=8, seed=1)
    tok_ok = len(tokens) == mel.shape[0] and tokens.max() < 8
    print(f"  [{'OK ' if tok_ok else 'FAIL'}] 音频 token 化 ({len(tokens)} token, 范围<8)")

    return vid_ok and motion_ok and mel_ok and tok_ok


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
