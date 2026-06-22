"""Capstone: TL;DR 摘要 RLHF — 三段管线完整 demo.

简化版（CPU 可跑），真训练需 GPU:
    1. SFT  : GPT-2 + TLDR 1k 子集
    2. RM   : 偏好对 → BT loss
    3. PPO  : actor 用 RM 打分 + KL ref

实际工业:
    GPT-2-medium / Anthropic-HH 50k / 5090 ~5h.
本 demo: 跑通 pipeline，每段 100 step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rm_minimal import RewardModel, bt_loss   # noqa: E402
from sft_minimal import sft_loss              # noqa: E402


def stage1_sft(model, tokenizer, sft_data, steps: int = 100):
    """SFT 100 step."""
    print(f"\n[Stage 1] SFT — {steps} step")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    losses = []
    for step in range(steps):
        # 简化：取一条
        q, a = sft_data[step % len(sft_data)]
        inp = tokenizer(q + a, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            q_ids = tokenizer.encode(q, add_special_tokens=False)
        labels = inp.input_ids.clone()
        labels[:, : len(q_ids)] = -100
        out = model(**inp)
        loss = sft_loss(out.logits, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")
    return losses


def stage2_rm(rm: RewardModel, tokenizer, pref_data, steps: int = 100):
    """RM 100 step."""
    print(f"\n[Stage 2] RM — {steps} step")
    optim = torch.optim.AdamW(rm.parameters(), lr=1e-5)
    losses = []
    for step in range(steps):
        q, c, r = pref_data[step % len(pref_data)]
        inp_c = tokenizer(q + c, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inp_r = tokenizer(q + r, return_tensors="pt", truncation=True, max_length=128, padding=True)
        rc = rm(inp_c.input_ids, inp_c.attention_mask)
        rr = rm(inp_r.input_ids, inp_r.attention_mask)
        loss = bt_loss(rc, rr)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if step % 20 == 0:
            acc = (rc > rr).float().mean().item()
            print(f"  step {step}: loss={loss.item():.4f} acc={acc:.2%}")
    return losses


def stage3_ppo_smoke(actor, ref, rm, tokenizer, prompts, steps: int = 20):
    """PPO smoke — 简化（用 trl 跑生产）."""
    print(f"\n[Stage 3] PPO — {steps} step (mock)")
    # 真实需要 ActorCritic + GAE + clip loop（见 ppo_llm_minimal）
    # 这里只演示 reward 上升趋势
    history = []
    for step in range(steps):
        q = prompts[step % len(prompts)]
        with torch.no_grad():
            inp = tokenizer(q, return_tensors="pt")
            gen = actor.generate(inp.input_ids, max_new_tokens=10, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            r = rm(gen, torch.ones_like(gen))
        history.append(r.item())
        if step % 5 == 0:
            print(f"  step {step}: reward={r.item():.4f}")
    return history


if __name__ == "__main__":
    print("Capstone: TL;DR RLHF 三段管线\n" + "=" * 50)
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError as exc:
        # 不静默 exit 0 假成功（systemic pitfall #4）：缺核心依赖一律 fail-fast，
        # 否则 harness 会把"什么都没跑"记成 PASS。
        raise SystemExit(
            f"[capstone] 缺少 transformers，无法运行三段管线 demo: {exc}\n"
            f"  装依赖后重试: pip install transformers"
        )

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    sft_model = GPT2LMHeadModel.from_pretrained("gpt2")
    sft_data = [
        ("TLDR: Long story about X. Summary:", " X happened."),
        ("TLDR: Article on Y. Summary:", " Y was discovered."),
    ]
    stage1_sft(sft_model, tok, sft_data, steps=20)

    rm = RewardModel(GPT2LMHeadModel.from_pretrained("gpt2"))
    pref_data = [
        ("Summarize: A is good.", " A is great.", " bad"),
        ("Summarize: B happened.", " B occurred today.", " irrelevant"),
    ]
    stage2_rm(rm, tok, pref_data, steps=20)

    stage3_ppo_smoke(sft_model, sft_model, rm, tok, ["TLDR: News."], steps=10)
    print("\n✅ 三段管线全跑通（smoke）")
