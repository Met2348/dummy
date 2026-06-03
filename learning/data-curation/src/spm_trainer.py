"""SentencePiece trainer — Unigram + BPE 教学.

教学目标：
    1. spm.SentencePieceTrainer 训练 Unigram 词表
    2. byte_fallback / split_digits / character_coverage 关键超参
    3. encode/decode + subword regularization

运行：
    python spm_trainer.py --demo
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def train_spm(text: str, vocab_size: int = 1000, model_type: str = "unigram",
              out_dir: str | None = None) -> str:
    import sentencepiece as spm
    out_dir = out_dir or tempfile.mkdtemp(prefix="spm_")
    text_path = Path(out_dir) / "corpus.txt"
    text_path.write_text(text, encoding="utf-8")
    prefix = str(Path(out_dir) / "m")
    spm.SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        byte_fallback=True,
        split_digits=True,
        num_threads=1,
    )
    return f"{prefix}.model"


def run_demo() -> None:
    import sentencepiece as spm
    text = "\n".join([
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subset of artificial intelligence",
        "renewable energy includes solar wind hydro geothermal",
        "the cat sat on the mat watching the bird outside",
        "natural language processing enables machine understanding",
        "quantum computing leverages superposition and entanglement",
    ] * 50)

    model_path = train_spm(text, vocab_size=200, model_type="unigram")
    print(f"trained → {model_path}")

    sp = spm.SentencePieceProcessor(model_file=model_path)
    sample = "the cat learns quickly"
    pieces = sp.encode(sample, out_type=str)
    ids = sp.encode(sample, out_type=int)
    decoded = sp.decode(ids)
    print(f"\nsample:  {sample!r}")
    print(f"pieces:  {pieces}")
    print(f"ids:     {ids}")
    print(f"decoded: {decoded!r}")

    # subword regularization
    print("\nSubword regularization (5 samples):")
    for _ in range(5):
        p = sp.encode(sample, out_type=str, enable_sampling=True, alpha=0.3)
        print(f"  {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        run_demo()


if __name__ == "__main__":
    main()
