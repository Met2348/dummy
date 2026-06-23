"""
mini_vlm.py — 一个端到端可训练的最小 VLM, 在合成「视觉问答」任务上真的学会回答.

为什么需要它 (M10.3 的核心): M10.1 给了视觉塔、M10.2 给了连接器, 但它们都是随机权重。
本文件把它们组装成一个**可训练的 mini-VLM**, 并设计一个**确定性、可度量**的合成任务, 让你
亲眼看到「VLM 训练」: loss 下降、准确率上升。这把抽象的「训 VLM」变成手里能跑的实验。

合成任务 (确定性, 离线): 每张图属于 K 个「视觉类别」之一 (不同色块布局)。VLM 收到固定指令
「这是哪类?」, 要输出正确的类别 token。视觉塔必须把图的类别信息传给 LLM, LLM 才能答对 ——
这逼着「视觉→连接器→LLM」整条链真的对齐。

支持冻结策略对比 (M10.3-L2): freeze_vision / freeze_llm 开关。纯 torch (tiny, CPU 秒级)。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def make_vqa_dataset(n_classes: int = 4, n_per_class: int = 16, img_size: int = 8,
                     seed: int = 0):
    """生成合成视觉问答数据: (图 patches, 类别标签).
    每类有一个固定的「色块布局原型」, 同类图 = 原型 + 小噪声。确定性。"""
    rng = np.random.default_rng(seed)
    patch = 4
    n_patch = (img_size // patch) ** 2
    prototypes = rng.random((n_classes, img_size, img_size, 3))  # 每类一个原型
    imgs, labels = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            img = np.clip(prototypes[c] + 0.08 * rng.standard_normal((img_size, img_size, 3)), 0, 1)
            # patchify
            blocks = []
            for i in range(img_size // patch):
                for j in range(img_size // patch):
                    blocks.append(img[i*patch:(i+1)*patch, j*patch:(j+1)*patch].reshape(-1))
            imgs.append(np.stack(blocks))
            labels.append(c)
    X = np.stack(imgs).astype(np.float32)         # (N, n_patch, patch*patch*3)
    y = np.array(labels)
    return X, y, n_patch


def build_mini_vlm(patch_dim: int, n_patch: int, n_classes: int,
                   d_vis: int = 24, d_llm: int = 32, seed: int = 0):
    """组装 mini-VLM: 视觉塔(ViT) + 投影连接器 + tiny LLM + 分类头. 无 torch 返回 None."""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[mini_vlm] 无 torch ({exc!r})")
        return None

    torch.manual_seed(seed)

    class MiniVLM(nn.Module):
        def __init__(self):
            super().__init__()
            # 视觉塔 (M10.1): patch embed + transformer + CLS
            self.vis_proj = nn.Linear(patch_dim, d_vis)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_vis))
            self.vis_pos = nn.Parameter(torch.randn(1, n_patch + 1, d_vis) * 0.02)
            self.vis_enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_vis, 4, d_vis * 2, batch_first=True), 1)
            # 连接器 (M10.2 投影): 视觉 → LLM 空间
            self.connector = nn.Sequential(nn.Linear(d_vis, d_llm), nn.GELU(),
                                           nn.Linear(d_llm, d_llm))
            # LLM (tiny): 处理 [视觉 token | 指令 token]
            self.instr = nn.Parameter(torch.randn(1, 2, d_llm) * 0.02)  # 固定指令「这是哪类」
            self.llm = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_llm, 4, d_llm * 2, batch_first=True), 2)
            self.head = nn.Linear(d_llm, n_classes)   # 输出类别 (模拟输出类别 token)

        def encode_vision(self, patches):
            B = patches.shape[0]
            x = self.vis_proj(patches)
            x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1) + self.vis_pos
            return self.vis_enc(x)                    # (B, n_patch+1, d_vis)

        def forward(self, patches):
            B = patches.shape[0]
            vis = self.encode_vision(patches)
            vis_llm = self.connector(vis)             # → LLM 空间
            instr = self.instr.expand(B, -1, -1)
            seq = torch.cat([vis_llm, instr], dim=1)  # [视觉 token | 指令]
            out = self.llm(seq)
            return self.head(out[:, -1])              # 用最后位置 (指令后) 预测类别

    return MiniVLM()


def set_freeze(model, freeze_vision: bool = False, freeze_llm: bool = False):
    """冻结策略 (M10.3-L2): 控制视觉塔/LLM 是否训练. 连接器永远训。"""
    for n, p in model.named_parameters():
        if n.startswith(("vis_proj", "cls", "vis_pos", "vis_enc")):
            p.requires_grad = not freeze_vision
        elif n.startswith(("instr", "llm")):
            p.requires_grad = not freeze_llm
        else:  # connector / head 永远训
            p.requires_grad = True


def train_mini_vlm(model, X, y, epochs: int = 30, lr: float = 5e-3, seed: int = 0):
    """训 mini-VLM, 返回 (loss 历史, acc 历史). 确定性 (固定 seed + 全 batch)。"""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    Xt = torch.tensor(X); yt = torch.tensor(y)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.CrossEntropyLoss()
    hist_loss, hist_acc = [], []
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(Xt)
        loss = lossf(logits, yt)
        loss.backward(); opt.step()
        with torch.no_grad():
            acc = (logits.argmax(-1) == yt).float().mean().item()
        hist_loss.append(loss.item()); hist_acc.append(acc)
    return hist_loss, hist_acc


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    X, y, n_patch = make_vqa_dataset(n_classes=4, n_per_class=16, seed=1)
    print(f"合成 VQA 数据: {X.shape[0]} 图, {len(set(y))} 类, 每图 {n_patch} patch")
    model = build_mini_vlm(patch_dim=X.shape[2], n_patch=n_patch, n_classes=4)
    if model is not None:
        set_freeze(model, freeze_vision=False, freeze_llm=False)
        print(f"可训练参数: {count_trainable(model)}")
        loss, acc = train_mini_vlm(model, X, y, epochs=30)
        print(f"训练: loss {loss[0]:.3f} → {loss[-1]:.3f}; acc {acc[0]:.2f} → {acc[-1]:.2f}")
        print("→ VLM 学会了从图判断类别 (视觉信息成功经连接器传给 LLM)。")
