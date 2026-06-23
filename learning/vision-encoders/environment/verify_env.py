"""
vision-encoders 专题环境自检.
运行: python environment/verify_env.py
"""
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
OPTIONAL = ["torch"]  # 无 torch 也能跑 numpy 部分 (patchify/对比损失)


def check_imports() -> list[str]:
    missing = []
    for pkg in REQUIRED:
        ok = importlib.util.find_spec(pkg) is not None
        print(f"  [{'OK ' if ok else 'MISS'}] {pkg}")
        if not ok:
            missing.append(pkg)
    for pkg in OPTIONAL:
        ok = importlib.util.find_spec(pkg) is not None
        print(f"  [{'OK ' if ok else 'opt '}] {pkg} (可选; 无则 ViT 部分跳过)")
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
    import tiny_vit as tv
    import contrastive as ct

    img = tv.make_synthetic_image("blocks", size=16, seed=1)
    patches = tv.patchify(img, patch=4)
    p_ok = patches.shape == (16, 48)
    print(f"  [{'OK ' if p_ok else 'FAIL'}] tiny_vit.patchify ({patches.shape})")

    img_e, txt_e = ct.make_paired_embeddings(n=6, noise=0.3, seed=1)
    S = ct.similarity_matrix(img_e, txt_e)
    diag_ok = S.diagonal().mean() > (S.sum() - S.trace()) / (S.size - len(S))
    print(f"  [{'OK ' if diag_ok else 'FAIL'}] contrastive 相似度对角线最大")
    # 噪声大 → 损失高
    img2, txt2 = ct.make_paired_embeddings(n=6, noise=1.5, seed=1)
    loss_mono = ct.info_nce_loss(img2, txt2) > ct.info_nce_loss(img_e, txt_e)
    print(f"  [{'OK ' if loss_mono else 'FAIL'}] InfoNCE 损失随对齐变差而升高")

    vit_ok = True
    if importlib.util.find_spec("torch") is not None:
        import torch
        m = tv.build_tiny_vit(patch_dim=48, n_patches=16)
        out = m(torch.tensor(patches[None], dtype=torch.float32))
        vit_ok = tuple(out.shape) == (1, 17, 32)
        print(f"  [{'OK ' if vit_ok else 'FAIL'}] tiny_vit ViT 前向 ({tuple(out.shape)})")

    return p_ok and diag_ok and loss_mono and vit_ok


def main() -> int:
    print("== Part A: 依赖 import ==")
    missing = check_imports()
    print("== Part B: nbformat 往返 ==")
    nb_ok = check_nbformat_roundtrip()
    print("== Part C: src 工具冒烟 ==")
    src_ok = check_src_tools()

    print("\n== 结论 ==")
    if missing:
        print(f"缺失依赖: {missing}\n  → pip install -r environment/requirements.txt")
        return 1
    if not (nb_ok and src_ok):
        print("nbformat 或 src 工具检查未通过.")
        return 1
    print("全部通过 ✅  两个 notebook 可端到端运行.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
