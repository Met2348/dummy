"""
bib_to_cards.py — 把 BibTeX 文件解析成结构化条目, 并批量生成 markdown 文献卡.

为什么需要它: 你的「第二大脑」第一块地基是文献库。Zotero 负责抓取和存 PDF, 但研究者
真正复用的是**自己写的笔记**, 而不是 PDF 本身。这个工具把 Zotero 导出的 `.bib` 一键摊成
一座 markdown 文献卡库 —— 每篇论文一张卡, 卡里预留「我的批注 / 一句话贡献 / 关联」字段。
之后这些卡可以双链 (见 L2), 长成一张可检索的知识网。

纯 stdlib (内置一个够用的 BibTeX 子集解析器), 无外部依赖, Windows 直接跑。

用法 (命令行):
    python src/bib_to_cards.py refs.bib --out my_library

用法 (notebook / import):
    from bib_to_cards import parse_bibtex, make_cards
    entries = parse_bibtex(open("refs.bib", encoding="utf-8").read())
    paths = make_cards(entries, out_dir="my_library")
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 匹配一个 @type{ ... } 顶层条目. 用计数法找配平的大括号, 而非贪婪正则 (BibTeX 嵌套括号常见).
_ENTRY_HEAD = re.compile(r"@(\w+)\s*\{", re.IGNORECASE)


def _find_entries(text: str) -> list[tuple[str, str]]:
    """返回 [(entry_type, body)], body 是花括号内原始内容 (不含最外层括号)."""
    out: list[tuple[str, str]] = []
    for m in _ENTRY_HEAD.finditer(text):
        etype = m.group(1).lower()
        if etype in {"comment", "preamble", "string"}:
            continue
        # 从 '{' 之后开始数括号配平.
        i = m.end()
        depth = 1
        start = i
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append((etype, text[start : i - 1]))
    return out


def _split_fields(body: str) -> dict[str, str]:
    """body 形如 'key, title={...}, author={...}, year=2026'. 返回 {citekey, field:value...}."""
    # 第一个逗号前是 citation key.
    head, _, rest = body.partition(",")
    fields: dict[str, str] = {"citekey": head.strip()}

    i = 0
    n = len(rest)
    while i < n:
        # 读 field 名 (到 '=').
        eq = rest.find("=", i)
        if eq == -1:
            break
        name = rest[i:eq].strip().strip(",").lower()
        j = eq + 1
        # 跳过空白.
        while j < n and rest[j] in " \t\r\n":
            j += 1
        if j >= n:
            break
        if rest[j] == "{":
            depth = 1
            j += 1
            start = j
            while j < n and depth > 0:
                if rest[j] == "{":
                    depth += 1
                elif rest[j] == "}":
                    depth -= 1
                j += 1
            value = rest[start : j - 1]
        elif rest[j] == '"':
            j += 1
            start = j
            while j < n and rest[j] != '"':
                j += 1
            value = rest[start:j]
            j += 1
        else:  # 裸值 (如 year=2026)
            start = j
            while j < n and rest[j] not in ",\n":
                j += 1
            value = rest[start:j].strip()
        if name:
            fields[name] = _clean(value)
        # 跳到下一个逗号后.
        comma = rest.find(",", j)
        i = comma + 1 if comma != -1 else n
    return fields


def _clean(s: str) -> str:
    """去掉 BibTeX 残留的花括号、多余空白、换行."""
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def parse_bibtex(text: str) -> list[dict]:
    """解析 BibTeX 文本, 返回条目列表. 每个条目是 {citekey, entrytype, title, author, year, ...}."""
    entries = []
    for etype, body in _find_entries(text):
        fields = _split_fields(body)
        fields["entrytype"] = etype
        entries.append(fields)
    return entries


def _first_author_last(author: str) -> str:
    """从 'Rafailov, Rafael and Sharma, ...' 或 'Rafael Rafailov and ...' 取第一作者姓."""
    if not author:
        return "anon"
    first = author.split(" and ")[0].strip()
    if "," in first:  # 'Last, First'
        return first.split(",")[0].strip().split()[-1]
    parts = first.split()
    return parts[-1] if parts else "anon"


CARD_TEMPLATE = """<!-- 文献卡 · 由 bib_to_cards.py 自动生成, 之后手填「我的批注」区 -->
# {title}

| 字段 | 值 |
|---|---|
| citekey | `{citekey}` |
| 作者 | {author} |
| 年份 | {year} |
| 类型 | {entrytype} |
| 场所 | {venue} |

## 一句话贡献 (读完填)
> (它做了什么前人没做的? 一句话)

## 我的批注 / 为什么我存它
-

## 关联 (双链到我的其它笔记 / 专题)
- 相关论文: [[ ]]
- 相关我的复现: [[ ]]
- 触发的 gap / idea: [[ ]]

## 三遍读法状态
☐ pass1 鸟瞰  ☐ pass2 骨架  ☐ pass3 精读   (详见 9.3 critical-reading-gap)
"""


def make_cards(entries: list[dict], out_dir: str | Path = "library") -> list[Path]:
    """为每个条目写一张 markdown 文献卡, 文件名 = 第一作者姓+年份+citekey, 返回路径列表."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for e in entries:
        venue = e.get("journal") or e.get("booktitle") or e.get("publisher") or "—"
        stem = f"{_first_author_last(e.get('author',''))}{e.get('year','')}-{e.get('citekey','x')}"
        stem = re.sub(r"[^\w\-]", "_", stem)
        dest = out / f"{stem}.md"
        dest.write_text(
            CARD_TEMPLATE.format(
                title=e.get("title", "(无标题)"),
                citekey=e.get("citekey", "—"),
                author=e.get("author", "—"),
                year=e.get("year", "—"),
                entrytype=e.get("entrytype", "—"),
                venue=venue,
            ),
            encoding="utf-8",
        )
        paths.append(dest)
    return paths


# 一个内置离线样例库 (notebook 不必带 .bib 文件也能跑).
SAMPLE_BIB = """
@article{rafailov2023dpo,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Manning, Christopher and Ermon, Stefano and Finn, Chelsea},
  year={2023},
  journal={NeurIPS}
}
@article{deepseek2025r1,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={DeepSeek-AI},
  year={2025},
  journal={arXiv preprint}
}
@inproceedings{ouyang2022instructgpt,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeff and others},
  year={2022},
  booktitle={NeurIPS}
}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="BibTeX → markdown 文献卡库")
    ap.add_argument("bib", nargs="?", help=".bib 文件路径 (省略则用内置样例)")
    ap.add_argument("--out", default="library", help="文献卡落地目录")
    args = ap.parse_args()

    text = Path(args.bib).read_text(encoding="utf-8") if args.bib else SAMPLE_BIB
    entries = parse_bibtex(text)
    paths = make_cards(entries, out_dir=args.out)
    print(f"解析 {len(entries)} 条 BibTeX, 生成 {len(paths)} 张文献卡 → {args.out}/")
    for p in paths:
        print(f"  - {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
