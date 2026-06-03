"""Compacter 最小实现（手写）。

对应论文: Karimi Mahabadi et al. 2021, "Compacter: Efficient Low-Rank Hypercomplex Adapter Layers" (NeurIPS)
对应 lecture: lectures/03-adapterdrop-compacter.md

核心数学 — PHM (Parameterized Hypercomplex Multiplication):
    把一个 (d_out, d_in) 矩阵 W 分解为 n 个 Kronecker 积之和：
        W = Σᵢ₌₁ⁿ Aᵢ ⊗ Bᵢ

    其中:
        Aᵢ ∈ ℝ^(n × n)              ← shared across layers!
        Bᵢ ∈ ℝ^(d_out/n × d_in/n)

    参数量: n × n² + n × (d_out · d_in / n²)
          = n³ + d_out · d_in / n

    vs 普通 Linear (d_out × d_in)，压缩比 ≈ n（当 d >> n）

示例 (d_out=768, d_in=16, n=4):
    普通: 768 × 16 = 12,288
    PHM:  4³ + 768·16/4 = 64 + 3072 = 3,136
    压缩比: 12,288 / 3,136 = 3.9×

关键 — 跨层共享 Aᵢ:
    所有 transformer block 共用同一组 Aᵢ
    → 进一步节省参数
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from common import freeze_base_model, print_param_summary  # noqa: E402
from houlsby_minimal import _MlpAdapterWrapper  # noqa: E402


def kronecker(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """计算 Kronecker 积 A ⊗ B。

    A: (m, n), B: (p, q) → (m*p, n*q)

    A ⊗ B = [[a_{ij} * B for j in cols] for i in rows]
    """
    m, n = A.shape
    p, q = B.shape
    # A.view(m, 1, n, 1) * B.view(1, p, 1, q) 然后 reshape
    out = A.unsqueeze(1).unsqueeze(3) * B.unsqueeze(0).unsqueeze(2)  # (m, p, n, q)
    return out.reshape(m * p, n * q)


class PHMLinear(nn.Module):
    """PHM-parameterized Linear layer。

    W = Σᵢ Aᵢ ⊗ Bᵢ, 然后 forward: y = W x + bias

    参数:
        in_features, out_features: 普通 Linear 形状
        n: 超复数维度（论文常用 n=4）
        shared_A: 可选的共享 A 矩阵（如果提供，本层不创建自己的 A）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n: int = 4,
        shared_A: nn.Parameter | None = None,
        bias: bool = True,
    ):
        super().__init__()
        assert in_features % n == 0 and out_features % n == 0, \
            f"in/out must be divisible by n; got in={in_features}, out={out_features}, n={n}"
        self.in_features = in_features
        self.out_features = out_features
        self.n = n

        # A: (n, n, n) 即 n 个 n×n 矩阵
        if shared_A is not None:
            self.A = shared_A  # 共享 across layers
            self._own_A = False
        else:
            self.A = nn.Parameter(torch.empty(n, n, n))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self._own_A = True

        # B: (n, out/n, in/n)
        self.B = nn.Parameter(torch.empty(n, out_features // n, in_features // n))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def construct_weight(self) -> torch.Tensor:
        """从 A, B 重建 W = Σᵢ Aᵢ ⊗ Bᵢ。"""
        Ws = []
        for i in range(self.n):
            W_i = kronecker(self.A[i], self.B[i])  # (out, in)
            Ws.append(W_i)
        return torch.stack(Ws, dim=0).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.construct_weight()  # (out, in)
        out = x @ W.T
        if self.bias is not None:
            out = out + self.bias
        return out


class CompacterAdapter(nn.Module):
    """Compacter Adapter：用 PHMLinear 替换 down/up 投影。

    Adapter(x) = x + up(act(down(x)))
        down: PHMLinear (d → r)
        up:   PHMLinear (r → d), 零初始化 (B=0)
    """

    def __init__(
        self,
        d: int,
        r: int = 16,
        n: int = 4,
        shared_A_down: nn.Parameter | None = None,
        shared_A_up: nn.Parameter | None = None,
    ):
        super().__init__()
        self.down = PHMLinear(d, r, n=n, shared_A=shared_A_down)
        self.up = PHMLinear(r, d, n=n, shared_A=shared_A_up)
        self.act = nn.GELU()
        # 零初始化 up.B → 初始 forward = x（adapter 透明）
        nn.init.zeros_(self.up.B)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        h = self.act(h)
        h = self.up(h)
        return x + h


class CompacterGPT2(nn.Module):
    """GPT-2 + Compacter Adapter（每 block 1 个，FFN 后；跨层共享 A 矩阵）。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 16,
        n: int = 4,
    ):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        freeze_base_model(self.lm)

        d = self.lm.config.n_embd
        self.r = r
        self.n = n

        # 创建共享的 A 矩阵（所有层共用）
        self.shared_A_down = nn.Parameter(torch.empty(n, n, n))
        self.shared_A_up = nn.Parameter(torch.empty(n, n, n))
        nn.init.kaiming_uniform_(self.shared_A_down, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_A_up, a=math.sqrt(5))

        # 每 block 加一个 adapter，复用 shared_A
        for block in self.lm.transformer.h:
            adapter = CompacterAdapter(
                d, r, n=n,
                shared_A_down=self.shared_A_down,
                shared_A_up=self.shared_A_up,
            )
            block.mlp = _MlpAdapterWrapper(block.mlp, adapter)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def main() -> None:
    torch.manual_seed(42)
    model = CompacterGPT2(r=16, n=4)
    print_param_summary(model, "Compacter (r=16, n=4)")

    print(f"\n参数布局推算 (d=768, r=16, n=4):")
    print(f"  共享 A_down: n^3 = 64")
    print(f"  共享 A_up:   n^3 = 64")
    print(f"  per layer B_down: 768*16/n^2 = {768*16//16}")
    print(f"  per layer B_up:   16*768/n^2 = {16*768//16}")
    print(f"  per layer bias (down + up): {16 + 768}")
    print(f"  12 layer 合计: 64*2 + 12 * (768 + 768 + 16 + 768) = ?")

    # 对比 Pfeiffer
    print(f"\n对比 Pfeiffer (r=16): 304,320")

    # 验证初始 forward = base
    from transformers import GPT2LMHeadModel
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    model.eval()
    enc = model.tokenizer("hello world", return_tensors="pt", padding=True)
    with torch.no_grad():
        out_c = model(enc["input_ids"], enc["attention_mask"])
        out_b = base(enc["input_ids"], attention_mask=enc["attention_mask"])
    diff = (out_c.logits - out_b.logits).abs().max().item()
    print(f"\n初始 forward vs base 误差: {diff:.4e}")
    print("  → up.B 零初始化保证 Adapter(x) = x")


if __name__ == "__main__":
    main()
