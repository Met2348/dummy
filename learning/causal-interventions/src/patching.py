"""
patching.py — activation patching + ablation (M12.3). mech interp 最核心工具.

为什么需要它 (M12.3): 探针/logit lens (M12.2) 只证**相关**。要证**因果** (「这个组件真的负责这个行为」),
必须**干预**: activation patching = 把「干净运行」的某个激活, 贴到「污染运行」里, 看行为是否被恢复。
恢复了 → 那个激活因果地携带了行为所需信息。这是 mech interp 区别于「看图讲故事」的命门。

做法 (clean/corrupt 对照):
  - clean 输入 → 预测 A; corrupt 输入 (改一处) → 预测 B
  - 把 clean 的某 (层, 位置) 激活 patch 进 corrupt 运行 → 看预测多大程度回到 A
  - 对每个 (层, 位置) 做 → 得因果定位热图: 哪里 patch 能恢复 = 那里因果负责

用 forward hook 覆盖 Block 输出实现 patch (不改 12.1 tiny_transformer)。纯 torch tiny CPU 确定性。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_M121 = Path(__file__).resolve().parents[2] / "interp-foundations" / "src"
if str(_M121) not in sys.path:
    sys.path.insert(0, str(_M121))
import tiny_transformer as tt  # noqa: E402


def make_clean_corrupt(seed: int = 0):
    """构造一对 (clean, corrupt) 序列: 同一 increment 序列, 但 corrupt 改了最后一个 token (→改了答案)。
    返回 (clean_tokens (1,SEQ), corrupt_tokens (1,SEQ), clean_answer, corrupt_answer)。"""
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, tt.V))
    clean = ((start + np.arange(tt.SEQ)) % tt.V).astype(np.int64)
    corrupt = clean.copy()
    new_last = (int(clean[-1]) + int(rng.integers(1, tt.V))) % tt.V   # 改最后一位
    corrupt[-1] = new_last
    clean_ans = int((clean[-1] + 1) % tt.V)
    corrupt_ans = int((corrupt[-1] + 1) % tt.V)
    return clean[None], corrupt[None], clean_ans, corrupt_ans


def _logits_last(model, tokens):
    import torch
    with torch.no_grad():
        return model(torch.tensor(tokens))[0, -1]            # (V,)


def get_clean_resid(model, clean_tokens, layer_idx: int):
    """取 clean 运行在某层 (block 输出 = resid_post_layer) 的激活。返回 (1, SEQ, d_model)。"""
    import torch
    _, cache = model.run_with_cache(torch.tensor(clean_tokens))
    return cache[f"resid_post_{layer_idx}"]


def run_with_patch(model, corrupt_tokens, layer_idx: int, pos: int, clean_resid):
    """在 corrupt 运行里, 把 clean 的 (layer_idx, pos) residual 贴进去 (forward hook 覆盖 Block 输出)。
    返回最后位置 logits。"""
    import torch
    value = clean_resid[:, pos, :]                            # (1, d_model)

    def hook(m, inp, out):
        out = out.clone()
        out[:, pos, :] = value
        return out

    h = model.blocks[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(torch.tensor(corrupt_tokens))[0, -1]
    finally:
        h.remove()
    return logits


def patch_recovery(model, seed: int = 0):
    """对每个 (layer, position) 做 activation patching, 测「恢复率」。
    恢复率 = patch 后 clean答案 logit 相对 (corrupt→clean) 的恢复比例。返回 (n_layers, SEQ) 矩阵。"""
    import torch
    clean, corrupt, clean_ans, corrupt_ans = make_clean_corrupt(seed)
    clean_logits = _logits_last(model, clean)
    corrupt_logits = _logits_last(model, corrupt)
    # 度量: clean答案 vs corrupt答案 的 logit 差 (越接近 clean 越恢复)
    def metric(logits):
        return float(logits[clean_ans] - logits[corrupt_ans])
    base_clean = metric(clean_logits)        # 应 > 0
    base_corrupt = metric(corrupt_logits)    # 应 < 0
    grid = np.zeros((tt.N_LAYERS, tt.SEQ))
    for L in range(tt.N_LAYERS):
        cr = get_clean_resid(model, clean, L)
        for p in range(tt.SEQ):
            patched = run_with_patch(model, corrupt, L, p, cr)
            m = metric(patched)
            grid[L, p] = (m - base_corrupt) / (base_clean - base_corrupt + 1e-8)  # 0=没恢复, 1=完全恢复
    return grid, (clean[0], corrupt[0], clean_ans, corrupt_ans)


def ablate_effect(model, seed: int = 0):
    """ablation: 把某 (layer, pos) 激活置零, 测对 clean 答案 logit 的损害 (越大越重要)。返回 (n_layers, SEQ)。"""
    import torch
    clean, _, clean_ans, _ = make_clean_corrupt(seed)
    base = float(_logits_last(model, clean)[clean_ans])
    grid = np.zeros((tt.N_LAYERS, tt.SEQ))
    for L in range(tt.N_LAYERS):
        for p in range(tt.SEQ):
            def hook(m, inp, out, pp=p):
                out = out.clone(); out[:, pp, :] = 0.0
                return out
            h = model.blocks[L].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    logit = float(model(torch.tensor(clean))[0, -1, clean_ans])
            finally:
                h.remove()
            grid[L, p] = base - logit          # 损害 = 置零后 clean 答案 logit 掉多少
    return grid


if __name__ == "__main__":
    import torch
    Xi, Yi = tt.make_data(2000, seed=0)
    model = tt.build_model(); tt.train(model, Xi, Yi, epochs=800)
    grid, (clean, corrupt, ca, cora) = patch_recovery(model, seed=3)
    print(f"clean {clean} → 答案 {ca}; corrupt {corrupt} → 答案 {cora}")
    print("activation patching 恢复率 (行=层, 列=位置; 1=patch此处完全恢复 clean 行为):")
    for L in range(tt.N_LAYERS):
        print(f"  层{L}: " + " ".join(f"{grid[L,p]:+.2f}" for p in range(tt.SEQ)))
    best = np.unravel_index(np.argmax(grid), grid.shape)
    print(f"→ 最能恢复的位置 = 层{best[0]} 位置{best[1]} (= 因果携带答案信息的地方; 应是最后位置)。")
