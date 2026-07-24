#!/usr/bin/env python3
"""check_anonymity.py -- 双盲投稿匿名化检查脚本

扫描一份 LaTeX 源文件，检测常见的"暴露身份"信号：
  1. AUTHOR_NAME              -- \\author{...} 里出现的真实姓名（不是 Anonymous 占位符）
  2. INSTITUTION               -- 机构关键词（University/Institute/Lab/Department of...）
  3. INSTITUTION_EMAIL_DOMAIN  -- 学术邮箱域名（.edu / .ac.xx / .edu.xx）
  4. EMAIL                     -- 任意邮箱地址
  5. GITHUB_LINK                -- 未匿名化的 github.com/<真实用户名或组织>/<repo> 链接
  6. ACKNOWLEDGMENTS_SECTION   -- Acknowledgments/Acknowledgements 章节标题
  7. FUNDING_PHRASE             -- 资助/致谢措辞（"supported by"/"we thank"/grant 编号等），
                                   即使不在 Acknowledgments 章节标题下也会暴露身份线索
  8. FIRST_PERSON_SELF_CITE    -- 用第一人称指代自己此前的工作（双盲要求改成第三人称转述）

这是一个基于正则/规则的启发式扫描器，不是完整的 LaTeX 解析器——设计目标是"覆盖真实投稿里
最常见的几类泄露",不是穷尽所有可能。见同目录教学文档"常见坑"一节讨论它的已知局限（假阳性/
假阴性的真实例子）。

用法：
    python check_anonymity.py path/to/paper.tex      # 扫描指定文件，非零退出码=发现问题
    python check_anonymity.py                        # 不带参数：跑内置自测（见 _run_self_test）
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass


@dataclass
class Finding:
    category: str
    line_no: int
    line_text: str
    detail: str


# ---------------------------------------------------------------------------
# 规则表
# ---------------------------------------------------------------------------

# 常见的匿名占位符"词块"，出现这些不算泄露——用词块拼接判断（而不是要求整段内容
# 逐字精确匹配某一句），这样 "Anonymous Author(s)\\Affiliation withheld for
# double-blind review" 这类多段拼接的、真实会议模板（比如 ICLR 官方模板的匿名分支）
# 里常见的写法也能被正确识别成"已匿名"，不会被误判成还没脱敏。
ANON_BOILERPLATE_PHRASES = [
    "affiliation withheld for double-blind review",
    "author names omitted for anonymous review",
    "paper under double-blind review",
    "institution withheld",
    "institution redacted",
    "affiliation withheld",
    "anonymous submission",
    "anonymous author(s)",
    "anonymous authors",
    "anonymous author",
    "anonymous",
]


def _is_anonymized_author_block(visible_text: str) -> bool:
    """把已知的匿名占位符词块逐个抠掉，如果抠完什么实质内容都不剩，就认为已经匿名。"""
    normalized = visible_text.lower()
    for phrase in ANON_BOILERPLATE_PHRASES:  # 已按长度从长到短排列，长词块优先吃掉
        normalized = normalized.replace(phrase, " ")
    leftover = re.sub(r"[\s.,;:\-]+", "", normalized)
    return leftover == ""

INSTITUTION_KEYWORDS = [
    r"university",
    r"institute of technology",
    r"\binstitute\b",
    r"\bcollege\b",
    r"laboratory",
    r"\blab\b",
    r"\bcorporation\b",
    r"\binc\.",
    r"department of",
    r"school of",
    r"academy of sciences",
]

FUNDING_PHRASES = [
    r"this work (was|is) (partially |in part )?(supported|funded)",
    r"this research (was|is) (partially |in part )?(supported|funded)",
    r"we (would like to )?thank\b",
    r"gratefully acknowledge",
    r"grant\s*(no\.?|number|#)",
    r"\bnsf\b",
    r"\bnih\b",
    r"national science foundation",
    r"under award number",
    r"funded by",
]

GITHUB_URL_RE = re.compile(
    r"https?://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)", re.IGNORECASE
)
ANON_HOST_HINTS = ("anonymous", "anon", "double-blind", "hidden", "redacted")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
EDU_DOMAIN_RE = re.compile(
    r"@[\w.\-]*\.(edu|ac\.[a-z]{2}|edu\.[a-z]{2})\b", re.IGNORECASE
)
SELF_CITE_RE = re.compile(
    r"\b(our (previous|prior|earlier) work|in our (previous|prior) (paper|work))\b",
    re.IGNORECASE,
)
ACK_HEADER_RE = re.compile(
    r"\\section\*?\{\s*acknowledge?ments?\s*\}", re.IGNORECASE
)
AUTHOR_CMD_RE = re.compile(r"\\author\s*(\[[^\]]*\])?\s*\{")
INSTITUTION_PATTERN = re.compile("|".join(INSTITUTION_KEYWORDS), re.IGNORECASE)
FUNDING_PATTERN = re.compile("|".join(FUNDING_PHRASES), re.IGNORECASE)


def strip_comments(line: str) -> str:
    """去掉 LaTeX 行内注释：找到第一个不是被反斜杠转义的 % 就截断。

    简化处理：不追求还原完整 LaTeX 语法（比如 verbatim 环境内部的 % 本不该被当注释），
    对本脚本要检测的"作者/机构/链接/致谢"这类内容已经够用。
    """
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(c)
            out.append(line[i + 1])
            i += 2
            continue
        if c == "%":
            break
        out.append(c)
        i += 1
    return "".join(out)


def find_author_block(lines: list[str]) -> list[Finding]:
    """在整份文档里定位 \\author{...}，用括号计数取出完整内容（可能跨多行）。"""
    findings = []
    text = "\n".join(lines)
    for m in AUTHOR_CMD_RE.finditer(text):
        start = m.end() - 1  # 指向左花括号 '{'
        depth = 0
        j = start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        content = text[start + 1 : j]
        line_no = text[: m.start()].count("\n") + 1
        visible = re.sub(r"\\thanks\{[^}]*\}", " ", content)
        visible = re.sub(r"\\\\|\\and\b|\\And\b", " ", visible, flags=re.IGNORECASE)
        visible = re.sub(r"\s+", " ", visible).strip()
        if visible and not _is_anonymized_author_block(visible):
            findings.append(
                Finding(
                    category="AUTHOR_NAME",
                    line_no=line_no,
                    line_text=visible[:100],
                    detail=(
                        "`\\author{}` content is not an anonymized placeholder; must be "
                        "replaced with 'Anonymous Author(s)' or similar during double-blind review"
                    ),
                )
            )
    return findings


def find_institutions(lines: list[str]) -> list[Finding]:
    findings = []
    for i, raw in enumerate(lines, start=1):
        line = strip_comments(raw)
        if INSTITUTION_PATTERN.search(line):
            findings.append(
                Finding(
                    category="INSTITUTION",
                    line_no=i,
                    line_text=line.strip()[:100],
                    detail=(
                        "Contains an institution-type keyword (University/Institute/Lab/"
                        "Department of/...); this must not appear in the body or author block during submission"
                    ),
                )
            )
        m = EDU_DOMAIN_RE.search(line)
        if m:
            findings.append(
                Finding(
                    category="INSTITUTION_EMAIL_DOMAIN",
                    line_no=i,
                    line_text=line.strip()[:100],
                    detail=f"Email domain `{m.group(0)}` is an academic institution domain and directly reveals affiliation",
                )
            )
    return findings


def find_emails(lines: list[str]) -> list[Finding]:
    findings = []
    for i, raw in enumerate(lines, start=1):
        line = strip_comments(raw)
        for m in EMAIL_RE.finditer(line):
            findings.append(
                Finding(
                    category="EMAIL",
                    line_no=i,
                    line_text=line.strip()[:100],
                    detail=f"Email address found: `{m.group(0)}`",
                )
            )
    return findings


def find_unanonymized_github_links(lines: list[str]) -> list[Finding]:
    findings = []
    for i, raw in enumerate(lines, start=1):
        line = strip_comments(raw)
        for m in GITHUB_URL_RE.finditer(line):
            owner, repo = m.group(1), m.group(2)
            owner_l = owner.lower()
            if any(hint in owner_l for hint in ANON_HOST_HINTS):
                continue  # owner 名字里带 anonymous/anon 之类，当作已经做过匿名化处理
            findings.append(
                Finding(
                    category="GITHUB_LINK",
                    line_no=i,
                    line_text=line.strip()[:100],
                    detail=(
                        f"`github.com/{owner}/{repo}` owner is not an anonymized placeholder, "
                        "likely a real username/organization; replace with an anonymized mirror (e.g. anonymous.4open.science)"
                    ),
                )
            )
    return findings


def find_acknowledgments(lines: list[str]) -> list[Finding]:
    findings = []
    for i, raw in enumerate(lines, start=1):
        line = strip_comments(raw)
        if ACK_HEADER_RE.search(line):
            findings.append(
                Finding(
                    category="ACKNOWLEDGMENTS_SECTION",
                    line_no=i,
                    line_text=line.strip()[:100],
                    detail=(
                        "Acknowledgments/Acknowledgements section header found; must be "
                        "removed entirely during double-blind review and added back at camera-ready"
                    ),
                )
            )
        if FUNDING_PATTERN.search(line):
            findings.append(
                Finding(
                    category="FUNDING_PHRASE",
                    line_no=i,
                    line_text=line.strip()[:100],
                    detail=(
                        "Contains funding/thanks language (e.g. 'supported by'/'we thank'/"
                        "grant number); can reveal identity even outside an Acknowledgments section"
                    ),
                )
            )
    return findings


def find_self_citation_first_person(lines: list[str]) -> list[Finding]:
    """检测"我们之前的工作"这类第一人称自引，双盲要求这类引用改成第三人称转述。"""
    findings = []
    for i, raw in enumerate(lines, start=1):
        line = strip_comments(raw)
        if SELF_CITE_RE.search(line):
            findings.append(
                Finding(
                    category="FIRST_PERSON_SELF_CITE",
                    line_no=i,
                    line_text=line.strip()[:100],
                    detail=(
                        "Refers to own prior work in first person (e.g. 'our previous work'); "
                        "double-blind requires third-person phrasing (e.g. 'Smith et al. (2024) previously showed...')"
                    ),
                )
            )
    return findings


CHECKS = (
    find_author_block,
    find_institutions,
    find_emails,
    find_unanonymized_github_links,
    find_acknowledgments,
    find_self_citation_first_person,
)


def run_checks(lines: list[str]) -> list[Finding]:
    all_findings: list[Finding] = []
    for check in CHECKS:
        all_findings.extend(check(lines))
    all_findings.sort(key=lambda f: f.line_no)
    return all_findings


def check_file(path: str) -> list[Finding]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return run_checks(lines)


def format_report(path: str, findings: list[Finding]) -> str:
    out = []
    if not findings:
        out.append(
            f"[OK] {path}: no anonymity-leak signals detected "
            "(heuristic only -- still do a manual pass before submitting)"
        )
        return "\n".join(out)
    out.append(f"[WARN] {path}: {len(findings)} suspected anonymity-leak signal(s) found")
    out.append("")
    for f in findings:
        out.append(f"  line {f.line_no:>4} [{f.category}] {f.line_text}")
        out.append(f"           -> {f.detail}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 内置自测：不需要任何外部文件，直接用两段内嵌的示例文本验证检测逻辑本身是对的。
# 这段自测既是"这个脚本自己的单元测试"，也是本文档 _verify_md.py 验证流程要跑的内容。
# ---------------------------------------------------------------------------

_LEAKY_TEX = r"""
\documentclass{article}
\author{
Eric Peng \\
Department of Computer Science, Riverside State University \\
epeng@riverside.edu \\
\And
Weikai Lin \\
Riverside AI Lab \\
wlin@riverside.edu
}
\begin{document}
In our previous work, we showed that naive rollout depth does not help.
Code: \url{https://github.com/epeng-riverside-lab/imagination-budget-controller}
\section*{Acknowledgments}
We thank the Riverside AI Lab for compute, and gratefully acknowledge funding
from NSF grant No. 2026-04821.
\end{document}
""".strip().splitlines()

_CLEAN_TEX = r"""
\documentclass{article}
\author{Anonymous Author(s) \\ Affiliation withheld for double-blind review}
\begin{document}
Peng \& Lin (2025) previously showed that naive rollout depth does not help.
Code: \url{https://anonymous.4open.science/r/imagination-budget-controller-4F21}
\end{document}
""".strip().splitlines()


def _run_self_test() -> None:
    leaky_findings = run_checks(_LEAKY_TEX)
    leaky_categories = {f.category for f in leaky_findings}
    expected_in_leaky = {
        "AUTHOR_NAME",
        "INSTITUTION",
        "INSTITUTION_EMAIL_DOMAIN",
        "EMAIL",
        "GITHUB_LINK",
        "ACKNOWLEDGMENTS_SECTION",
        "FUNDING_PHRASE",
        "FIRST_PERSON_SELF_CITE",
    }
    missing = expected_in_leaky - leaky_categories
    assert not missing, f"leaky example should trigger {missing}, but did not"
    assert len(leaky_findings) >= 8, (
        f"expected at least 8 findings in the leaky example, got {len(leaky_findings)}"
    )

    clean_findings = run_checks(_CLEAN_TEX)
    assert clean_findings == [], (
        f"clean example should have zero findings, got {[f.category for f in clean_findings]}"
    )

    print(format_report("<inline leaky example>", leaky_findings))
    print()
    print(format_report("<inline clean example>", clean_findings))
    print()
    print(
        f"self-test passed: {len(leaky_findings)} findings across "
        f"{len(leaky_categories)} categories on the leaky example, "
        "0 findings on the clean example"
    )


def main() -> None:
    if len(sys.argv) < 2:
        _run_self_test()
        return
    path = sys.argv[1]
    findings = check_file(path)
    print(format_report(path, findings))
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
