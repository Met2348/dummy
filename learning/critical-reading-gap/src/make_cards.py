"""
make_cards.py — 从模板一键起卡 (论文解剖卡 / gap 卡 / idea 卡).

为什么需要它: 研究是高频、重复的动作 —— 每读一篇论文起一张解剖卡, 每发现一个洞起一张
gap 卡. 手动复制重命名会让人懒得做, 而"懒得记录"是新手研究流水线断掉的头号原因.
把起卡变成一行命令, 摩擦降到最低, 习惯才立得住.

用法 (命令行):
    python src/make_cards.py paper --name dpo-rafailov2023
    python src/make_cards.py gap
    python src/make_cards.py idea --name cw-dpo

用法 (notebook 里 import):
    from make_cards import make_card
    path = make_card("gap", out_dir="my_gap_library")
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Windows 控制台默认 GBK; 强制 UTF-8 输出.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 卡类型 → 模板文件名
TEMPLATES = {
    "paper": "paper-note-card.md",
    "gap": "gap-record-card.md",
    "idea": "idea-card.md",
}

# 卡类型 → 文件名前缀
PREFIX = {"paper": "PAPER", "gap": "GAP", "idea": "IDEA"}


def _templates_dir() -> Path:
    """templates/ 与 src/ 同级, 都在专题根下."""
    return Path(__file__).resolve().parent.parent / "templates"


def _next_index(out_dir: Path, stem_prefix: str) -> int:
    """同一天同类卡的自增编号, 避免覆盖."""
    existing = list(out_dir.glob(f"{stem_prefix}*.md"))
    return len(existing) + 1


def make_card(kind: str, out_dir: str | Path = ".", name: str | None = None,
              today: str | None = None) -> Path:
    """复制对应模板到 out_dir, 生成带日期/编号的新卡, 返回新文件路径.

    kind:   "paper" | "gap" | "idea"
    out_dir: 卡片落地目录 (你的个人 gap 库)
    name:   可选的人类可读后缀 (如论文关键词)
    today:  可选, 覆盖日期 (主要给测试用), 形如 "20260617"
    """
    if kind not in TEMPLATES:
        raise ValueError(f"未知卡类型 {kind!r}, 应为 {list(TEMPLATES)}")

    tpl = _templates_dir() / TEMPLATES[kind]
    if not tpl.exists():
        raise FileNotFoundError(f"模板不存在: {tpl}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp = today or date.today().strftime("%Y%m%d")
    prefix = f"{PREFIX[kind]}-{stamp}-"
    idx = _next_index(out, prefix)
    suffix = f"-{name}" if name else ""
    dest = out / f"{prefix}{idx:02d}{suffix}.md"

    dest.write_text(tpl.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="从模板一键起卡")
    ap.add_argument("kind", choices=list(TEMPLATES), help="卡类型")
    ap.add_argument("--out", default=".", help="落地目录 (默认当前目录)")
    ap.add_argument("--name", default=None, help="可选的人类可读后缀")
    args = ap.parse_args()

    path = make_card(args.kind, out_dir=args.out, name=args.name)
    print(f"已创建: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
