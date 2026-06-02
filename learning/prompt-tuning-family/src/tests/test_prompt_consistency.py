"""
Prompt Tuning 数值一致性测试。

策略：
  1) 用相同随机种子构造 minimal 和 peft 两版模型
  2) 把 minimal 的 P 复制到 peft 的 prompt_encoder.default.embedding.weight
  3) 同样的输入跑 forward
  4) 验证 logits 数值一致（误差 < 1e-4）

由于 Prompt Tuning 无 reparameterization、无随机 dropout，
理论上两者应该 bit 精确相等。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# 让 test 文件可以从父目录导入 minimal/peft 实现
sys.path.append(str(Path(__file__).parent.parent))
from prompt_tuning_minimal import PromptTuningGPT2  # noqa: E402
from prompt_tuning_peft import build_peft_model  # noqa: E402


def get_peft_prompt(peft_model) -> torch.nn.Parameter:
    """从 peft 模型里取出 prompt embedding 参数。

    peft 内部存储路径：peft_model.prompt_encoder.default.embedding.weight
    """
    return peft_model.prompt_encoder.default.embedding.weight


def test_logits_match() -> None:
    torch.manual_seed(42)
    P = 10

    # 1. 构造两个模型
    m_minimal = PromptTuningGPT2(prompt_length=P).eval()
    m_peft = build_peft_model(prompt_length=P).eval()

    # 2. 把 minimal 的 P 复制到 peft（保证初始化一致）
    with torch.no_grad():
        peft_prompt = get_peft_prompt(m_peft)
        assert peft_prompt.shape == m_minimal.prompt_embeddings.shape, (
            f"shape 不匹配: peft={peft_prompt.shape} vs minimal={m_minimal.prompt_embeddings.shape}"
        )
        peft_prompt.copy_(m_minimal.prompt_embeddings.data)

    # 3. 准备相同输入
    tok = m_minimal.tokenizer
    enc = tok("hello world", return_tensors="pt", padding=True)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    print(f"input_ids shape: {input_ids.shape}, attention_mask shape: {attn.shape}")

    # 4. 双方前向
    with torch.no_grad():
        out_minimal = m_minimal(input_ids=input_ids, attention_mask=attn)
        out_peft = m_peft(input_ids=input_ids, attention_mask=attn)

    # peft 输出 logits 形状可能与 minimal 不同
    # minimal: (B, p+n, V), peft 内部会自动管理 prompt，可能输出 (B, p+n, V) 或 (B, n, V)
    print(f"minimal logits shape: {out_minimal.logits.shape}")
    print(f"peft    logits shape: {out_peft.logits.shape}")

    # 对齐 shape：如果 peft 也输出 (B, p+n, V)，直接比；否则取最后 n 个位置
    if out_minimal.logits.shape == out_peft.logits.shape:
        diff = (out_minimal.logits - out_peft.logits).abs().max().item()
    else:
        # 取共同的最后 n 个位置（实际输入的 token 部分）
        n = attn.shape[1]
        diff = (out_minimal.logits[:, -n:] - out_peft.logits[:, -n:]).abs().max().item()
    print(f"\nlogits 最大绝对误差: {diff:.2e}")

    assert diff < 1e-4, f"Logits 差异过大: {diff}"
    print("[PASS] minimal 与 peft 输出一致")


if __name__ == "__main__":
    test_logits_match()
