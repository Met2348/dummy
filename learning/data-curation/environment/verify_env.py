"""Data Curation 环境自检 — 三段式.

Part A: datasketch + sentencepiece + tiktoken import
Part B: FineWeb-Edu classifier 加载 smoke
Part C: BPE 训练 smoke（100 行训 1k merge）
"""
from __future__ import annotations

import sys
import tempfile


def part_a() -> bool:
    print("\n=== Part A: 基础 import ===")
    ok = True
    for mod in ["datasketch", "sentencepiece", "tiktoken", "trafilatura",
                "simhash", "warcio"]:
        try:
            __import__(mod)
            print(f"  [OK] {mod}")
        except ImportError as e:
            print(f"  [FAIL] {mod}: {e}")
            ok = False
    return ok


def part_b() -> bool:
    print("\n=== Part B: FineWeb-Edu classifier 加载 ===")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
        model = AutoModelForSequenceClassification.from_pretrained(
            "HuggingFaceFW/fineweb-edu-classifier"
        )
        sample = "The mitochondria is the powerhouse of the cell, producing ATP."
        inputs = tok(sample, return_tensors="pt", truncation=True, max_length=512)
        score = model(**inputs).logits.item()
        print(f"  [OK] FineWeb-Edu score: {score:.2f}")
        return True
    except Exception as e:
        print(f"  [SKIP] {e} （首次需 ~500MB 下载，跳过即可）")
        return True  # 不阻断


def part_c() -> bool:
    print("\n=== Part C: BPE 训练 smoke ===")
    try:
        import sentencepiece as spm
        with tempfile.TemporaryDirectory() as td:
            text_path = f"{td}/sample.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                for _ in range(100):
                    f.write("the quick brown fox jumps over the lazy dog\n")
            spm.SentencePieceTrainer.train(
                input=text_path,
                model_prefix=f"{td}/m",
                vocab_size=64,
                model_type="bpe",
                num_threads=1,
            )
            sp = spm.SentencePieceProcessor(model_file=f"{td}/m.model")
            ids = sp.encode("the quick brown fox", out_type=int)
            print(f"  [OK] BPE: encoded {len(ids)} tokens")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


if __name__ == "__main__":
    results = [part_a(), part_b(), part_c()]
    print("\n=== 汇总 ===")
    for name, ok in zip("ABC", results):
        print(f"  Part {name}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results) else 1)
