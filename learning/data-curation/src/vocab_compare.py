"""不同 tokenizer 在 5 种语言下压缩率对照.

教学目标：
    1. 横向比较 gpt2 / cl100k / o200k
    2. 看到多语言 / 代码 / 数学 token 数差异
    3. 体会"词表大小 -> 训练成本"

运行：
    python vocab_compare.py
"""
from __future__ import annotations


SAMPLES = {
    "english": "Machine learning is a subset of artificial intelligence "
               "concerned with statistical algorithms that learn from data, "
               "improving with experience. Neural networks are one approach.",
    "chinese": "机器学习是人工智能的一个分支，涉及从数据中学习的统计算法。"
               "神经网络是其中一种主要方法，通过反向传播进行训练。",
    "japanese": "機械学習は人工知能の一分野であり、データから学習する統計"
                "アルゴリズムを扱う。ニューラルネットワークが代表例である。",
    "code": "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n"
            "    pivot = arr[len(arr) // 2]\n"
            "    left  = [x for x in arr if x < pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quick_sort(left) + [pivot] + quick_sort(right)",
    "math": "\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}\n"
            "\\sum_{n=1}^\\infty \\frac{1}{n^2} = \\frac{\\pi^2}{6}\n"
            "\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}",
}


def main() -> None:
    import tiktoken
    models = ["gpt2", "cl100k_base", "o200k_base"]
    print(f"\n{'sample':<10} {'chars':>5} | " +
          " | ".join(f"{m:>20}" for m in models))
    print("-" * (20 + 22 * len(models)))
    for name, text in SAMPLES.items():
        n_chars = len(text)
        row = [f"{name:<10}", f"{n_chars:>5}"]
        for m in models:
            try:
                enc = tiktoken.get_encoding(m)
                n_tok = len(enc.encode(text))
                ratio = n_chars / n_tok
                row.append(f"{n_tok:>5} tok ({ratio:.2f} c/t)")
            except Exception:
                row.append("err")
        print(" | ".join(row))


if __name__ == "__main__":
    main()
