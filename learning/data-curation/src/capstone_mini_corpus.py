"""Capstone — 1B token mini corpus pipeline.

串联 stages：
    1. extract (trafilatura + lang filter)
    2. dedup (MinHash)
    3. quality (heuristic + optional FineWeb-Edu)
    4. PII (Presidio / regex)
    5. tokenize (SentencePiece 32k)

模式：
    --smoke  小规模 1k mock docs 跑通流水
    --warc   从真实 WARC 跑（需自备）

运行：
    python capstone_mini_corpus.py --smoke
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cc_extract
import minhash_dedup
import quality_filter
import toxicity_pii_filter
import spm_trainer


# ---------- 通用 IO ----------
def write_jsonl_gz(items, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        n = 0
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------- mock 数据（smoke 模式） ----------
# 高质量 doc 由「主题 + 多句正文」组合而成：每个 doc 用不同主题/句子，
# 保证 (1) 彼此差异足够大、不会被 MinHash 整片去重；(2) 以句号结尾、句数足够，
# 能通过 quality_filter 启发式。否则整条流水会在 dedup/quality 阶段被清空（空语料 no-op）。
_QUALITY_TOPICS = [
    ("Mitochondria", [
        "Mitochondria are the powerhouse of the cell and generate most of its chemical energy.",
        "They produce adenosine triphosphate through oxidative phosphorylation across the inner membrane.",
        "Each mitochondrion carries its own circular DNA, inherited maternally in most animals.",
        "This evidence supports the endosymbiotic theory that they descend from ancient bacteria.",
    ]),
    ("Photosynthesis", [
        "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
        "The light reactions in the thylakoid membranes split water and release oxygen.",
        "The Calvin cycle then fixes carbon dioxide into sugars using the captured energy.",
        "Chlorophyll absorbs mainly red and blue wavelengths while reflecting green light.",
    ]),
    ("Plate tectonics", [
        "Plate tectonics describes how rigid plates move slowly over the ductile mantle below.",
        "Convergent boundaries build mountains and trigger earthquakes as plates collide.",
        "Divergent boundaries create new ocean floor as magma rises along mid-ocean ridges.",
        "Transform faults release energy through sudden slips that radiate seismic waves.",
    ]),
    ("Neural networks", [
        "Neural networks learn patterns by adjusting connection weights between artificial neurons.",
        "Backpropagation computes gradients of the loss with respect to every parameter.",
        "Deeper layers compose simple features into increasingly abstract representations.",
        "Regularization techniques such as dropout reduce overfitting on the training data.",
    ]),
    ("Roman aqueducts", [
        "Roman aqueducts carried fresh water across long distances using a gentle constant gradient.",
        "Engineers built arched bridges to keep the channel level over valleys and rivers.",
        "Gravity alone moved millions of litres into cities for baths, fountains, and homes.",
        "Many segments still stand today as monuments to ancient hydraulic engineering.",
    ]),
    ("Ocean currents", [
        "Ocean currents redistribute heat around the planet and shape regional climates.",
        "The Gulf Stream carries warm tropical water northward along the Atlantic coast.",
        "Differences in temperature and salinity drive the slow global thermohaline circulation.",
        "Surface currents are pushed largely by prevailing winds and deflected by rotation.",
    ]),
]
_QUALITY_TAILS = [
    "Researchers continue to refine these ideas as new measurements arrive.",
    "Understanding the mechanism helps explain many everyday observations.",
    "The topic connects several disciplines and rewards careful study.",
    "Textbooks often summarize it, yet the details repay a closer look.",
    "These principles underpin much of the modern scientific worldview.",
]


def _mock_docs(n: int = 1000):
    base_bad = "click here CLICK CLICK NOW NOW spam spam spam spam spam"
    base_dup = "renewable energy is clean and sustainable for future generations"
    docs = []
    for i in range(n):
        if i % 3 == 0:
            # 高质量：换主题 + 轮换句子顺序 + 不同结尾句 -> 真实多样、句号结尾
            topic, sentences = _QUALITY_TOPICS[(i // 3) % len(_QUALITY_TOPICS)]
            rot = (i // 3) % len(sentences)
            body = sentences[rot:] + sentences[:rot]
            tail = _QUALITY_TAILS[(i // 3) % len(_QUALITY_TAILS)]
            text = f"{topic}. " + " ".join(body) + " " + tail
        elif i % 3 == 1:
            text = base_dup + f" variant {i % 10}"   # 大量重复（dedup 应清掉）
        else:
            text = base_bad                          # 垃圾（quality 应清掉）
        docs.append({"url": f"http://mock/{i}", "ts": "2024-12-15",
                     "lang": "en", "text": text})
    return docs


# ---------- stages ----------
def stage_1_extract(in_path, out_path, lang_filter=("en",), min_len=200):
    t0 = time.time()
    if in_path is None:
        docs = _mock_docs(1000)
    else:
        docs = list(cc_extract.iter_warc(in_path))
    docs = [d for d in docs if d["lang"] in lang_filter
            and len(d["text"]) >= min_len]
    n = write_jsonl_gz(docs, out_path)
    return {"stage": "extract", "n": n, "sec": round(time.time() - t0, 1)}


def stage_2_dedup(in_path, out_path, num_perm=128, threshold=0.7):
    t0 = time.time()
    docs = list(read_jsonl_gz(in_path))
    pairs = [(d["url"], d["text"]) for d in docs]
    kept_ids, _ = minhash_dedup.dedup(pairs, threshold=threshold,
                                       num_perm=num_perm)
    kept = [d for d in docs if d["url"] in kept_ids]
    n = write_jsonl_gz(kept, out_path)
    return {"stage": "dedup", "n": n, "sec": round(time.time() - t0, 1)}


def stage_3_quality(in_path, out_path, use_edu=False, edu_threshold=2.5):
    t0 = time.time()
    out = []
    for doc in read_jsonl_gz(in_path):
        h = quality_filter.heuristic_score(doc["text"])
        if not h["passed"]:
            continue
        if use_edu:
            score = quality_filter.fineweb_edu_score(doc["text"][:2048])
            if score is None or score < edu_threshold:
                continue
        out.append(doc)
    n = write_jsonl_gz(out, out_path)
    return {"stage": "quality", "n": n, "sec": round(time.time() - t0, 1)}


def stage_4_pii(in_path, out_path, tox_thresh=0.5):
    t0 = time.time()
    out = []
    for doc in read_jsonl_gz(in_path):
        flagged, _ = toxicity_pii_filter.is_toxic(doc["text"], tox_thresh)
        if flagged:
            continue
        cleaned, _ = toxicity_pii_filter.anonymize(doc["text"], use_presidio=False)
        doc["text"] = cleaned
        out.append(doc)
    n = write_jsonl_gz(out, out_path)
    return {"stage": "pii", "n": n, "sec": round(time.time() - t0, 1)}


def stage_5_tokenize(in_path, out_path, model_dir, vocab_size=2000,
                     model_type="unigram"):
    """训 SP + encode 全部文档."""
    t0 = time.time()
    # 1. write txt for spm train
    txt_path = Path(model_dir) / "train.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    n_chars = 0
    with txt_path.open("w", encoding="utf-8") as f:
        for doc in read_jsonl_gz(in_path):
            f.write(doc["text"] + "\n")
            n_docs += 1
            n_chars += len(doc["text"])
    if n_docs == 0:
        return {"stage": "tokenize", "n": 0, "sec": round(time.time() - t0, 1)}

    # 2. spm train（玩具量级用小词表）
    model_path = spm_trainer.train_spm(
        txt_path.read_text(encoding="utf-8"),
        vocab_size=min(vocab_size, max(256, n_chars // 200)),
        model_type=model_type,
        out_dir=str(model_dir),
    )

    # 3. encode all
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model_path)
    out_docs = []
    n_tok_total = 0
    for doc in read_jsonl_gz(in_path):
        ids = sp.encode(doc["text"])
        doc["n_tokens"] = len(ids)
        n_tok_total += len(ids)
        out_docs.append(doc)
    n = write_jsonl_gz(out_docs, out_path)
    return {"stage": "tokenize", "n": n, "n_tokens": n_tok_total,
            "model": model_path, "sec": round(time.time() - t0, 1)}


# ---------- 主流程 ----------
def run_pipeline(out_dir: str, warc_path: str | None = None, use_edu: bool = False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "extract":  out_dir / "01_extract.jsonl.gz",
        "dedup":    out_dir / "02_dedup.jsonl.gz",
        "quality":  out_dir / "03_quality.jsonl.gz",
        "pii":      out_dir / "04_clean.jsonl.gz",
        "tokenize": out_dir / "05_final.jsonl.gz",
    }
    report = []
    report.append(stage_1_extract(warc_path, paths["extract"]))
    report.append(stage_2_dedup(paths["extract"], paths["dedup"]))
    report.append(stage_3_quality(paths["dedup"], paths["quality"],
                                   use_edu=use_edu))
    report.append(stage_4_pii(paths["quality"], paths["pii"]))
    report.append(stage_5_tokenize(paths["pii"], paths["tokenize"],
                                    model_dir=out_dir / "tokenizer"))

    # 写 report
    report_path = out_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Capstone-1 Report\n\n")
        f.write("| Stage | n_docs | sec | extra |\n")
        f.write("|-------|--------|-----|-------|\n")
        for r in report:
            extra = ""
            if "n_tokens" in r: extra = f"tokens={r['n_tokens']}"
            if "model" in r: extra += f" model={Path(r['model']).name}"
            f.write(f"| {r['stage']} | {r['n']} | {r['sec']} | {extra} |\n")
    print(f"\n=== Capstone done ===\n  report: {report_path}\n")
    for r in report:
        print(f"  {r}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1k mock docs smoke")
    ap.add_argument("--warc", type=str, default=None, help="real WARC path")
    ap.add_argument("--out", type=str, default="out/mini-corpus")
    ap.add_argument("--use-edu", action="store_true", help="启 FineWeb-Edu")
    args = ap.parse_args()

    if args.smoke or not args.warc:
        run_pipeline(args.out, warc_path=None, use_edu=args.use_edu)
    else:
        run_pipeline(args.out, warc_path=args.warc, use_edu=args.use_edu)


if __name__ == "__main__":
    main()
