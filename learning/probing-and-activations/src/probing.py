"""
probing.py — 线性探针 + logit lens (M12.2). 复用 12.1 tiny_transformer; 也能用在真实 gpt2 激活上.

为什么需要它 (M12.2): mech interp 第一件工具是「读出网络内部表示」。两招:
  - 线性探针 (linear probe): 训一个线性分类器, 看某概念是否被**线性编码**在激活里 (能读出=被编码)。
  - logit lens: 把中间层 residual 直接过 unembed, 看「每层在想什么」(预测如何逐层成形)。
注意 (引出 12.3): 探针能读出 ≠ 模型在用它 (相关非因果)。这是探针的根本陷阱。

linear_probe 是通用的 (输入任意激活矩阵), 既能用在玩具 transformer, 也能用在真实 gpt2 激活上。
纯 torch tiny CPU 确定性。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 复用 12.1 的 tiny_transformer
_M121 = Path(__file__).resolve().parents[2] / "interp-foundations" / "src"
if str(_M121) not in sys.path:
    sys.path.insert(0, str(_M121))
import tiny_transformer as tt  # noqa: E402


def linear_probe(X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int = 300,
                 lr: float = 0.05, test_frac: float = 0.3, seed: int = 0):
    """通用线性探针: 训一个线性分类器从激活 X 预测标签 y。返回 (train_acc, test_acc)。
    test_acc 高 = 概念被线性编码 (可读出); 低 = 没被线性编码 (或没编码)。"""
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    n = len(X); idx = rng.permutation(n); ntest = int(n * test_frac)
    te, tr = idx[:ntest], idx[ntest:]
    Xt = torch.tensor(X[tr], dtype=torch.float32); yt = torch.tensor(y[tr])
    Xe = torch.tensor(X[te], dtype=torch.float32); ye = torch.tensor(y[te])
    probe = nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        loss = nn.functional.cross_entropy(probe(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        tr_acc = (probe(Xt).argmax(-1) == yt).float().mean().item()
        te_acc = (probe(Xe).argmax(-1) == ye).float().mean().item()
    return tr_acc, te_acc


def tiny_layer_activations(model, inputs: np.ndarray, layer_key: str = "resid_post_1", pos: int = -1):
    """跑玩具 transformer, 取某层某位置的 residual 激活 + 对应"当前值"标签。"""
    import torch
    _, cache = model.run_with_cache(torch.tensor(inputs))
    acts = cache[layer_key][:, pos, :].numpy()       # (B, d_model)
    labels = inputs[:, pos]                            # 当前 token 值
    return acts, labels


def logit_lens_tiny(model, tokens: np.ndarray):
    """logit lens (玩具版): 把每层 residual 过最终 unembed, 看每层的 top 预测。
    返回 {layer_key: 该层在最后位置的 top 预测 token}。"""
    import torch
    _, cache = model.run_with_cache(torch.tensor(tokens))
    out = {}
    with torch.no_grad():
        for key in ["resid_pre", "resid_post_0", "resid_post_1"]:
            resid = cache[key][:, -1, :]               # 最后位置
            logits = model.unembed(model.ln_f(resid))  # 直接过最终 unembed (logit lens)
            out[key] = logits.argmax(-1).numpy()
    return out


if __name__ == "__main__":
    import torch
    Xi, Yi = tt.make_data(2000, seed=0)
    model = tt.build_model(); tt.train(model, Xi, Yi, epochs=800)
    print("=== 线性探针: 从各层 residual 读出'当前值' ===")
    Xtest, _ = tt.make_data(600, seed=5)
    for key in ["resid_pre", "resid_post_0", "resid_post_1"]:
        acts, labels = tiny_layer_activations(model, Xtest, layer_key=key)
        tr, te = linear_probe(acts, labels, n_classes=tt.V)
        print(f"  {key:14}: 探针测试准确率 {te:.2f} (高=该层线性编码了'当前值')")
    print("\n=== logit lens: 每层在'想'什么 ===")
    sample = Xtest[:1]
    lens = logit_lens_tiny(model, sample)
    true_next = (sample[0, -1] + 1) % tt.V
    print(f"  序列 {sample[0]}, 正确下一个 = {true_next}")
    for key, pred in lens.items():
        print(f"  {key:14}: top 预测 = {pred[0]} {'✓' if pred[0]==true_next else ''}")
    print("→ 探针读出概念 (相关); logit lens 看预测逐层成形。但探针≠因果 (引出 12.3)。")
