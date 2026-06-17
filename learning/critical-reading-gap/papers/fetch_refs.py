"""
fetch_refs.py — 尝试下载方法参考 PDF (Keshav, How to Read a Paper).

无网/被墙时优雅失败, 不影响专题 (要点已写进 lectures/L1)。
用法: python papers/fetch_refs.py
"""
from __future__ import annotations
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent

# Keshav "How to Read a Paper" 的若干公开镜像 (按序尝试)
SOURCES = {
    "Keshav-HowToReadAPaper.pdf": [
        "https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf",
        "https://www.eecs.harvard.edu/~michaelm/postscripts/ReadPaper.pdf",
        "https://cseweb.ucsd.edu/~wgg/CSE210/howtoread.pdf",
    ],
}


def try_download(name: str, urls: list[str]) -> bool:
    import requests
    dest = HERE / name
    if dest.exists():
        print(f"  已存在, 跳过: {name}")
        return True
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                dest.write_bytes(r.content)
                print(f"  ✓ 下载成功: {name}  ({len(r.content)//1024} KB)  ← {url}")
                return True
        except Exception as e:
            print(f"  · 尝试失败 {url}: {type(e).__name__}")
    print(f"  ✗ 全部源失败: {name} (无网/被墙均可能; 不影响专题, 见 lectures/L1)")
    return False


def main() -> int:
    print("尝试下载方法参考 PDF ...")
    ok = all(try_download(n, u) for n, u in SOURCES.items())
    print("完成。" if ok else "部分未下载 —— 可手动按 papers/README.md 获取。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
