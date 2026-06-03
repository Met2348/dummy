"""LoRA 扩展：rsLoRA + LoRA+。

对应 lecture: lectures/01-lora.md 附录 A、B

rsLoRA (附录 A, arXiv:2312.03732):
  - 唯一改动: scaling = α / sqrt(r)（替换 LoRA 的 α / r）
  - 对大 r 训练稳定

LoRA+ (附录 B, arXiv:2402.12354):
  - 不改 forward，只改 optimizer：A、B 用不同学习率
  - 推荐 η_B = 16 × η_A
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent))
from common import freeze_base_model, get_parent_and_attr, target_linear_modules  # noqa: E402
from lora_minimal import LoRALinear  # noqa: E402


# ==============================
# rsLoRA
# ==============================
class RSLoRALinear(LoRALinear):
    """rsLoRA: 与 LoRA 唯一差异是 scaling = α / sqrt(r)。

    在大 r 下，rsLoRA 保持 |Δ W| 量级稳定，LoRA 则会趋向 0。
    """

    def __init__(self, base_linear: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__(base_linear, r=r, alpha=alpha, dropout=dropout)
        # 覆盖父类的 scaling（唯一差异）
        self.scaling = alpha / math.sqrt(r)


class RSLoRAGPT2(nn.Module):
    """把 GPT-2 用 RSLoRALinear 包装。"""

    def __init__(
        self,
        base_model_name: str = "gpt2",
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: tuple[str, ...] = ("c_attn",),
    ):
        super().__init__()
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.lm = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        freeze_base_model(self.lm)

        matches = target_linear_modules(self.lm, target_modules)
        for qname, old in matches:
            parent, attr = get_parent_and_attr(self.lm, qname)
            new = RSLoRALinear(old, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, new)

        self.r = r
        self.alpha = alpha

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


# ==============================
# LoRA+
# ==============================
def lora_plus_param_groups(
    model: nn.Module,
    lr_A: float = 1e-4,
    lambda_B: float = 16.0,
) -> list[dict]:
    """生成 LoRA+ 的 optimizer param groups。

    lr_B = lambda_B * lr_A

    Returns:
        [{"params": A_params, "lr": lr_A}, {"params": B_params, "lr": lr_B}]
    """
    lr_B = lambda_B * lr_A
    A_params, B_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 匹配 LoRA A、B 命名（兼容 minimal 的 ".A"/".B" 和 peft 的 "lora_A"/"lora_B"）
        if name.endswith(".A") or "lora_A" in name:
            A_params.append(p)
        elif name.endswith(".B") or "lora_B" in name:
            B_params.append(p)
        else:
            other_params.append(p)
    groups = []
    if A_params:
        groups.append({"params": A_params, "lr": lr_A, "_name": "lora_A"})
    if B_params:
        groups.append({"params": B_params, "lr": lr_B, "_name": "lora_B"})
    if other_params:
        groups.append({"params": other_params, "lr": lr_A, "_name": "other"})
    return groups


def main() -> None:
    torch.manual_seed(42)

    print("=" * 60)
    print("rsLoRA demo")
    print("=" * 60)
    for r in [4, 16, 64, 256]:
        m = RSLoRAGPT2(r=r, alpha=16)
        # 取第一层 LoRA 看 scaling
        for module in m.lm.modules():
            if isinstance(module, RSLoRALinear):
                print(f"  r={r:4d}: scaling = α/√r = 16/√{r} = {module.scaling:.4f}")
                break
        del m

    print("\n" + "=" * 60)
    print("LoRA+ demo")
    print("=" * 60)
    from lora_minimal import LoRAGPT2

    m = LoRAGPT2(r=8, alpha=16)
    groups = lora_plus_param_groups(m, lr_A=1e-4, lambda_B=16.0)
    for g in groups:
        nparam = sum(p.numel() for p in g["params"])
        print(f"  group {g['_name']:<10}: {nparam:>10,} params, lr = {g['lr']:.1e}")


if __name__ == "__main__":
    main()
