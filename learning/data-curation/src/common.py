"""共用工具 — text normalize / url strip / language detect / jsonl 流式."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator


_WHITESPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_PRINT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize_text(text: str, lower: bool = False) -> str:
    """NFKC 归一化 + 去控制字符 + 空白压缩."""
    text = unicodedata.normalize("NFKC", text)
    text = _NON_PRINT.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if lower:
        text = text.lower()
    return text


def strip_urls(text: str, repl: str = " ") -> str:
    return _URL.sub(repl, text)


def detect_lang(text: str) -> str:
    """fasttext 短样本语种识别，失败回退 'unk'."""
    try:
        from ftlangdetect import detect
        return detect(text=text[:512].replace("\n", " "), low_memory=True)["lang"]
    except Exception:
        return "unk"


def write_jsonl(path: str | Path, items: Iterable[dict]) -> int:
    n = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
