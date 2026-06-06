"""CommonCrawl WARC 流式抽取 — trafilatura + 语种识别.

教学目标：
    1. warcio 流式读 WARC.gz（O(1) 内存）
    2. trafilatura 抽主内容（去 boilerplate）
    3. fasttext 语种识别
    4. jsonl.gz 输出

运行：
    python cc_extract.py --warc seg.warc.gz --out docs.jsonl.gz
    或 demo 模式（用内置 mock HTML）：
    python cc_extract.py --demo
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator


MIN_TEXT_LEN = 200    # 短于此丢弃（绝大多数是 nav / error page）


def extract_from_html(html: str | bytes, url: str = "", ts: str = "") -> dict | None:
    """单文档抽取 + 语种识别 → dict 或 None."""
    import trafilatura
    text = trafilatura.extract(html, include_comments=False, include_tables=False,
                               deduplicate=True)
    if not text or len(text) < MIN_TEXT_LEN:
        return None
    try:
        from ftlangdetect import detect
        lang = detect(text=text[:512].replace("\n", " "), low_memory=True)["lang"]
    except Exception:
        lang = "unk"
    return {"url": url, "ts": ts, "lang": lang, "text": text}


def iter_warc(warc_path: str | Path) -> Iterator[dict]:
    from warcio.archiveiterator import ArchiveIterator
    with open(warc_path, "rb") as f:
        for rec in ArchiveIterator(f):
            if rec.rec_type != "response":
                continue
            url = rec.rec_headers.get_header("WARC-Target-URI", "") or ""
            ts = rec.rec_headers.get_header("WARC-Date", "") or ""
            try:
                html = rec.content_stream().read()
            except Exception:
                continue
            doc = extract_from_html(html, url=url, ts=ts)
            if doc:
                yield doc


def write_jsonl_gz(items: Iterable[dict], out_path: str | Path) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n


DEMO_HTML = """<!DOCTYPE html><html><head><title>Test</title></head>
<body><nav>Home | About | Login</nav>
<article><h1>Why Cats Sleep So Much</h1>
<p>Cats sleep on average 12-16 hours per day. The reason is that as predators,
their bodies have evolved to conserve energy between hunting bursts. This
biological inheritance persists even in fully domesticated house cats today.</p>
<p>Kittens sleep even more — up to 20 hours. Growth hormones are primarily
released during deep sleep cycles, which is why young animals need extensive
rest periods.</p></article>
<footer>Copyright 2024 ExampleSite</footer>
</body></html>""".encode("utf-8")


def run_demo() -> None:
    print("--- demo 模式：使用内置 mock HTML ---")
    doc = extract_from_html(DEMO_HTML, url="http://demo.test/cats")
    if doc:
        print(f"url:  {doc['url']}")
        print(f"lang: {doc['lang']}")
        print(f"text ({len(doc['text'])} chars):\n{doc['text'][:300]}")
    else:
        print("[FAIL] 抽取返回空（依赖未装？）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warc", type=str, default=None, help="path/to/seg.warc.gz")
    ap.add_argument("--out", type=str, default="docs.jsonl.gz")
    ap.add_argument("--demo", action="store_true", help="跑内置 demo")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少 doc")
    args = ap.parse_args()

    if args.demo or not args.warc:
        run_demo()
        return

    def _iter():
        for i, doc in enumerate(iter_warc(args.warc)):
            if args.limit and i >= args.limit:
                break
            yield doc

    n = write_jsonl_gz(_iter(), args.out)
    print(f"[OK] wrote {n} docs → {args.out}")


if __name__ == "__main__":
    main()
