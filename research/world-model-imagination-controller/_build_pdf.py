#!/usr/bin/env python3
"""把本目录下任意一份 md 讲义转成 PDF:先把全部 mermaid 代码块预渲染成 PNG,
再用 pandoc + xelatex(中文字体 Microsoft YaHei)生成最终 PDF。

用法: python _build_pdf.py [源文件名,默认 05-full-technical-briefing.md]
例如: python _build_pdf.py 06-world-model-primer.md
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_NAME = sys.argv[1] if len(sys.argv) > 1 else "05-full-technical-briefing.md"
SRC = HERE / SRC_NAME
STEM = SRC.stem
BUILD_DIR = HERE / "_pdf_build" / STEM
OUT_MD = BUILD_DIR / f"{STEM}_rendered.md"
OUT_PDF = HERE / f"{STEM}.pdf"

MERMAID_RE = re.compile(r"```mermaid\n(.*?)\n```", re.DOTALL)
SLASH_IN_WORD_RE = re.compile(r"(?<=[^\s/])/(?=[^\s/])")


def add_table_break_points(text: str) -> str:
    """表格行里"AAA/BBB/CCC"这类无空格长词,给 LaTeX longtable 窄列一个可断行点,
    否则整词比列宽还宽会触发 'Infinite glue shrinkage' 报错。只处理以 | 开头的表格行,
    且跳过含 http 的行(避免拆坏链接)。"""
    out_lines = []
    for line in text.split("\n"):
        if line.startswith("|") and "http" not in line:
            line = SLASH_IN_WORD_RE.sub(" / ", line)
        out_lines.append(line)
    return "\n".join(out_lines)


def render_mermaid_blocks(text: str) -> str:
    BUILD_DIR.mkdir(exist_ok=True)
    counter = 0

    def repl(m: re.Match) -> str:
        nonlocal counter
        counter += 1
        mmd_path = BUILD_DIR / f"diagram_{counter}.mmd"
        png_path = BUILD_DIR / f"diagram_{counter}.png"
        mmd_path.write_text(m.group(1), encoding="utf-8")
        mmdc = shutil.which("mmdc")
        cmd = ([mmdc] if mmdc else ["npx", "--yes", "@mermaid-js/mermaid-cli"]) + [
            "-i", str(mmd_path), "-o", str(png_path),
            "-w", "1100", "-b", "white",
        ]
        print(f"[{counter}] rendering mermaid diagram -> {png_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True, shell=(sys.platform == "win32"))
        if result.returncode != 0 or not png_path.exists():
            print(f"  FAILED: {result.stderr[-2000:]}")
            raise RuntimeError(f"mermaid render failed for diagram {counter}")
        return f"![]({png_path.name})\n"

    return MERMAID_RE.sub(repl, text)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    print(f"source: {len(text)} chars, {text.count('```mermaid')} mermaid blocks")
    text = add_table_break_points(text)
    rendered = render_mermaid_blocks(text)
    OUT_MD.write_text(rendered, encoding="utf-8")
    print(f"wrote intermediate markdown: {OUT_MD} ({len(rendered)} chars)")

    preamble_path = BUILD_DIR / "preamble.tex"
    preamble_path.write_text(
        r"\usepackage{etoolbox}" "\n"
        r"\AtBeginEnvironment{longtable}{\small}" "\n"
        r"\setlength{\tabcolsep}{4pt}" "\n"
        r"\usepackage{pdflscape}" "\n",
        encoding="utf-8",
    )
    cmd = [
        "pandoc", str(OUT_MD),
        "-o", str(OUT_PDF),
        "--pdf-engine=xelatex",
        "--resource-path", str(BUILD_DIR),
        "-V", "CJKmainfont=Microsoft YaHei",
        "-V", "geometry:margin=2.3cm",
        "-V", "fontsize=11pt",
        "-V", "linestretch=1.15",
        "--toc", "--toc-depth=2",
        "-V", "colorlinks=true",
        "--include-in-header", str(preamble_path),
    ]
    print("running pandoc ->", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    log_path = BUILD_DIR / "pandoc_log.txt"
    log_path.write_text(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n", encoding="utf-8")
    print(f"full log written to {log_path}")
    print((result.stdout or "")[-2000:])
    print((result.stderr or "")[-3000:])
    if result.returncode != 0 or not OUT_PDF.exists():
        raise RuntimeError("pandoc failed, see pandoc_log.txt")
    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"\nSUCCESS: {OUT_PDF} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
