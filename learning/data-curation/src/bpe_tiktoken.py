"""tiktoken 加载 + 对照 — cl100k / o200k / gpt-2.

教学目标：
    1. tiktoken 使用接口
    2. encode/decode roundtrip
    3. 多 tokenizer 压缩率对比

运行：
    python bpe_tiktoken.py --demo
"""
from __future__ import annotations

import argparse


SAMPLES = {
    "english": "Machine learning is a subset of artificial intelligence "
               "concerned with statistical algorithms that learn from data.",
    "chinese": "机器学习是人工智能的一个分支，涉及从数据中学习的统计算法。",
    "code":    "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n"
               "    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]",
    "math":    "\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}",
}


def compare_compression(model_names: list[str]) -> None:
    import tiktoken
    print(f"\n{'sample':<10} | {'len':>5} | " +
          " | ".join(f"{m:>11}" for m in model_names))
    print("-" * (24 + 14 * len(model_names)))
    for name, text in SAMPLES.items():
        n_chars = len(text)
        row = [f"{name:<10}", f"{n_chars:>5}"]
        for model in model_names:
            try:
                enc = tiktoken.get_encoding(model)
                n_tok = len(enc.encode(text))
                ratio = n_chars / n_tok
                row.append(f"{n_tok:>4} ({ratio:.1f})")
            except Exception as e:
                row.append(f"  err")
        print(" | ".join(row))


def run_demo() -> None:
    models = ["gpt2", "cl100k_base", "o200k_base"]
    compare_compression(models)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        run_demo()


if __name__ == "__main__":
    main()
