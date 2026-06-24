"""
realmodels.py — 课程「小而真」真实模型公共工具 (本地 CPU, 离线缓存).

为什么有它: 课程多数 notebook 用 toy (numpy/tiny torch) 讲清机制。但有些概念
(真实注意力、真实困惑度、真实 CoT、真实拒答…) 用一个**真实小模型**看会更可信。
本机 HF 缓存里有两个能在 CPU 跑的真实模型:
  - gpt2 (124M): 真实 logits / attention / KV cache / 困惑度 / 采样 / 量化 / LoRA
  - TinyLlama-1.1B-Chat: 真实对话 / CoT / 拒答 / 当评委 / RAG 生成

设计:
  - 只读本地缓存 (local_files_only=True) → 离线确定性, nbconvert 不联网。
  - 模型缺失时返回 None (不崩溃) → 没缓存的机器上 notebook 仍 0 error (优雅降级)。
  - 全部 eval + no_grad + 贪心 (greedy) 默认 → 输出可复现。

旗舰 notebook 用法 (在 notebooks/ 里):
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parents[1] / "_shared"))
    import realmodels as rm
    tok, model = rm.gpt2()
    if model is not None:
        ...   # 真实例子
    else:
        print("无 gpt2 缓存, 跳过真实例子 (toy 部分不受影响)")
"""
from __future__ import annotations

import sys
from typing import Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GPT2 = "gpt2"
TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_CACHE: dict = {}


def _load(name: str, output_attentions: bool = False):
    """加载 (tok, model), 只读本地缓存; 缺失/出错返回 (None, None)。带进程内缓存。"""
    key = (name, output_attentions)
    if key in _CACHE:
        return _CACHE[key]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        kw = dict(local_files_only=True, dtype=torch.float32)
        if output_attentions:
            kw["attn_implementation"] = "eager"   # 拿 attention 权重需 eager
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
        model.eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        _CACHE[key] = (tok, model)
        return tok, model
    except Exception as exc:  # 缓存缺失 / 依赖问题 → 优雅降级
        print(f"[realmodels] 无法加载 {name} ({type(exc).__name__}: {exc}); 跳过真实例子。")
        _CACHE[key] = (None, None)
        return None, None


def gpt2(output_attentions: bool = False):
    """加载 gpt2 (124M). 返回 (tok, model) 或 (None, None)。"""
    return _load(GPT2, output_attentions)


def tinyllama():
    """加载 TinyLlama-1.1B-Chat. 返回 (tok, model) 或 (None, None)。"""
    return _load(TINYLLAMA)


def available() -> dict:
    """报告哪些真实模型本地可用 (用于 notebook 顶部提示)。"""
    out = {}
    for name in (GPT2, TINYLLAMA):
        tok, model = _load(name)
        out[name] = model is not None
    return out


def generate(tok, model, prompt: str, max_new_tokens: int = 20,
             greedy: bool = True, temperature: float = 1.0, top_p: float = 1.0,
             seed: int = 0) -> str:
    """从 prompt 续写, 返回**新增**文本 (不含 prompt)。greedy=True 时确定性。"""
    import torch
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt")
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if greedy:
        kw["do_sample"] = False
    else:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    with torch.no_grad():
        out = model.generate(**ids, **kw)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)


def chat(tok, model, user: str, system: Optional[str] = None,
         max_new_tokens: int = 48, greedy: bool = True, seed: int = 0) -> str:
    """TinyLlama 对话: 套 chat template, 返回 assistant 回复 (已去模板)。"""
    import torch
    torch.manual_seed(seed)
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    kw["do_sample"] = not greedy
    with torch.no_grad():
        out = model.generate(**ids, **kw)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


def next_token_topk(tok, model, text: str, k: int = 8):
    """返回 text 之后**下一个 token** 的 top-k (token 字符串, 概率)。真实 logits。"""
    import torch
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]      # 最后位置的下一 token 分布
    probs = torch.softmax(logits, -1)
    p, idx = probs.topk(k)
    return [(tok.decode([i]), float(pr)) for pr, i in zip(p, idx)]


def perplexity(tok, model, text: str) -> float:
    """真实困惑度 (越低=模型越觉得这段文本自然/可预测)。"""
    import torch
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids, labels=ids.input_ids)
    return float(torch.exp(out.loss))


def token_count(tok, text: str) -> int:
    return len(tok(text).input_ids)


if __name__ == "__main__":
    print("真实模型可用性:", available())
    tok, m = gpt2()
    if m is not None:
        print("gpt2 续写:", repr(generate(tok, m, "The capital of France is", 8)))
        print("gpt2 下一 token top5:", next_token_topk(tok, m, "The capital of France is", 5))
        print("困惑度 (通顺):", round(perplexity(tok, m, "The cat sat on the mat."), 1))
        print("困惑度 (打乱):", round(perplexity(tok, m, "mat the on sat cat The."), 1))
    tok2, m2 = tinyllama()
    if m2 is not None:
        print("TinyLlama 答 17+25:", repr(chat(tok2, m2, "What is 17+25? One word.", max_new_tokens=10)))
