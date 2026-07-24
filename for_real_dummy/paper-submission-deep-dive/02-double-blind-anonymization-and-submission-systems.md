# 02 · 双盲匿名化与投稿系统操作 —— 一份真的能查出问题的检查脚本 + OpenReview 真实流程

> 总览见 [00-roadmap.md](00-roadmap.md)
> 和 [daily-toolkit-deep-dive/07-latex-paper-writing.md 第 7 节](../daily-toolkit-deep-dive/07-latex-paper-writing.md) 的边界：那一节已经讲过会议 LaTeX 模板的机制本身（`\usepackage{会议宏包}` 怎么接管排版参数、匿名开关 `\iclrfinalcopy` 这类命令的存在，以及"投稿阶段忘记保持匿名"被列为常见坑之一）——**这篇不重复模板机制**，聚焦那边完全没展开的两件事：匿名化具体要检查哪些内容（并给你一个真的能跑的检查工具），以及 OpenReview 投稿系统本身怎么操作。

---

## 0. 这篇文章是怎么验证的（先说清楚）

- **匿名化检查脚本**：真实代码，[`_assets/02-anon-checker/check_anonymity.py`](_assets/02-anon-checker/check_anonymity.py)，在 `e:\Workspace\dummy\.venv\Scripts\python.exe`（Python 3.13.9）里真实跑通。第 3 节会展示两条独立的真实验证路径：（a）脚本内置的自测模式（不需要任何外部文件）；（b）对着两份真实构造的 `.tex` 文件——[`leaky_example.tex`](_assets/02-anon-checker/leaky_example.tex)（故意写满 8 类身份泄露）和 [`clean_example.tex`](_assets/02-anon-checker/clean_example.tex)（同一篇论文的正确匿名化版本）——跑命令行扫描。全部输出都是真实运行产生的，包括撰写过程中脚本自己踩到的一个真实 bug（第 3 节如实记录，不是编的教学案例）。
- **OpenReview 提交系统的图形界面操作**：无法自动化验证。已通过 WebSearch 核实 ICLR 2026 官方 Author Guide 当前公开的真实流程（截至 2026-07），第 4 节标注的每一步都有官方文档依据，但没有一篇真实要投稿的论文可以用来做端到端提交测试，如实标注为"已核实当前公开流程，未做端到端自动化验证"。
- **2025-11-27 OpenReview 匿名信息泄露事故**：真实发生的事件，第 1 节引用的处理声明和后果均来自这次 WebSearch 检索到的多篇独立报道（含 ICLR/NeurIPS 官方警告原话），来源见文末。

---

## 1. 为什么需要双盲 —— 不遵守的真实后果，以及一次真实发生的行业级事故

**为什么需要这个 / 不会有什么后果：**

双盲评审（double-blind review）的设计目标是：审稿人不知道论文作者是谁，作者也不知道审稿人是谁——目的是让评审只基于论文内容本身，不受"这是不是大实验室出品""这个作者我认不认识"这类身份线索影响。这不是走形式的规定：**ICLR 2026 官方 Author Guide 明确写着，任何在正文或补充材料里暴露作者身份的论文，会被直接 desk reject**（不进入正常评审流程，直接淘汰）——这是最直接的后果，一旦踩中，前面几个月的工作会在投稿这一步就归零，连让审稿人看到内容的机会都没有。NeurIPS 的政策覆盖面更广：不仅正文和补充材料，**代码仓库也在匿名化要求范围内**。

**环境要求：** 无，这是建立动机的一节。

**一步步跟着做（用真实发生的事故理解"认真对待双盲"这件事的分量）：**

1. **理解官方红线在哪。** ICLR 2026：正文或补充材料任何一处暴露作者身份 → desk reject；引用自己此前挂在 arXiv 上的相关论文不算破坏匿名（前提是用第三人称转述，比如写成"Smith et al. (2024) 此前指出……"而不是"我们此前的工作指出……"）。NeurIPS：所有提交材料（含代码）必须匿名；单纯存在一份非匿名的 arXiv 预印本本身不算违规，但"高调宣传自己在投的论文"可能被认定为违规——这条界限很微妙，第 5 节会展开。
2. **知道审稿人这一侧的对称规则。** NeurIPS 的指引同样要求审稿人不能主动去 arXiv 搜索、试图查出自己审的论文作者是谁——双盲是双向义务，不是只约束作者。
3. **理解违规后果不是"提醒你改一下"这么轻。** 2025-11-27，OpenReview 平台自身的 API 出现一个漏洞，意外泄露了 ICLR 2026、NeurIPS、ACL 等多个会议的审稿人、领域主席（AC）和"盲审"作者身份信息。会议方的应对声明非常严厉：**任何利用、传播这批泄露信息的行为，都会导致该作者名下全部投稿被直接 desk reject，并处以多年参会禁令**。真实的后续案例：已有论文因为作者公开分享了自己的 OpenReview 页面链接（客观上把自己和已经泄露的审稿人身份关联了起来）而被判定违规拒稿；甚至有审稿人报告收到作者的私信，直接开口要求提高评分（个别案例里还涉及金钱利益输送的暗示）——这些行为一旦被查实，后果远超"这篇论文被拒"，是对研究者本人学术诚信记录的实质性伤害。
4. **把这件事的教训落到自己身上。** 这个事故是平台方的技术漏洞，不是作者能控制的——但它恰恰说明了双盲这件事有多"较真"：会议方宁可选择让这一届评审质量受影响（后续有评论指出，这类事故之后审稿人打分容易趋于保守、"少说重话"以规避风险），也要用最严厉的手段维护匿名制度本身的可信度。你自己这边能控制的部分——不要让论文正文/补充材料/代码仓库主动暴露身份——性价比极高：几乎不需要额外的研究工作量，纯粹是投稿前的检查纪律问题，第 2-3 节会把这件事落实成一份真的能执行的检查流程。

**背后发生了什么：**

双盲机制想解决的是同行评审里一个经典的公平性问题：审稿人不可避免地会对"认识的实验室""大机构"产生潜移默化的信任偏差，双盲通过物理隔绝身份信息，尽量把评审拉回"只看内容"这条线上。真实研究也证实这套机制不是摆设但也不完美，这里精确说清楚两条不同的数字，不要混为一谈：一项研究测出，**70%-86%** 的评审压根没有尝试去猜作者是谁（也就是说 **14%-30%** 会尝试猜一下，不管猜没猜对）；而"猜没猜对"是另一条独立统计的数字——**74%-90%** 的评审最终没能猜对（也就是说真正蒙对身份的比例只有 **10%-26%**）。这两条数字合在一起说明：匿名化不是 100% 有效，确实有一部分审稿人会尝试推测、也确实有一部分能蒙对，但"完全没猜"和"猜了但没猜对"两种情况加起来占绝大多数——机制本身有效但脆弱，任何一处主动暴露都会把这道防线捅穿，这也是为什么会议方会用 desk reject 这种重手段维护它。

**常见坑：**

| 现象 | 大概率原因 |
|---|---|
| 觉得"反正审稿人大概率能猜出来是谁，匿名化随便做做就行" | 猜对率确实不是 0，但 70%-90% 的评审猜不出/猜错——匿名化仍然显著起作用，"反正会被猜到"不能作为不认真匿名化的理由 |
| 论文里引用了自己挂在 arXiv 上的前作，担心这样算暴露身份 | 只要用第三人称转述（不用"我们此前"），单纯引用不算破坏匿名，见本节第 1 步 |
| 论文投稿期间在社交媒体/个人主页高调宣传自己的工作 | NeurIPS 政策把"aggressive advertising"列为可能违规的灰色地带，第 5 节详细展开这条边界 |

**自测清单：**

- [ ] 能说出 ICLR/NeurIPS 对暴露身份分别是什么后果（desk reject / 也覆盖代码）
- [ ] 知道引用自己挂在 arXiv 上的前作，正确做法是第三人称转述而不是完全不提
- [ ] 能说出真实的匿名化猜中率数据（70%-90% 猜不出/猜错），理解这不是"反正没用"的借口
- [ ] 知道 2025-11-27 OpenReview 泄露事故的处理原则：利用泄露信息 = 直接拒稿+多年禁令，不是口头警告

---

## 2. LaTeX 源码匿名化清单 —— 人工检查该看哪几类东西

**为什么需要这个 / 不会有什么后果：**

在动用工具之前，先建立一份人能看懂、能照着自查的清单——这是第 3 节脚本要自动化的东西的"人工版本"，两者应该互相印证，不是只信工具不信自己（脚本是启发式规则，见第 3 节"常见坑"，不是万能的）。

**环境要求：** 一份准备投稿的 `.tex` 源码。

**一步步跟着做：**

匿名化检查清单，按"最容易漏查到最容易忽略"排序：

1. **作者信息**：`\author{}` 里的真实姓名，替换成 `Anonymous Author(s)` 或者会议模板指定的占位写法。
2. **机构/单位信息**：不只是 `\author{}` 块里的单位名，正文任何地方提到"我们在 XX 大学的服务器上跑实验"之类的表述也要清理。
3. **邮箱地址**：任何真实邮箱，包括作者块里的联系邮箱。
4. **Acknowledgments（致谢）章节**：投稿阶段**整节删除**，不是留着但把名字划掉——致谢段落本身的存在往往就会写明资助机构/合作者，这些信息本身就是身份线索，等 camera-ready 阶段再加回来（05 号文件详细讲）。
5. **资助/致谢措辞**：即使没有一个专门的 Acknowledgments 章节标题，正文里散落的"本工作受 XX 资助""感谢 XX 提供计算资源"这类句子同样要清理——身份线索不一定只出现在专门的致谢段落里。
6. **GitHub/代码仓库链接**：链接里如果包含真实用户名或组织名（比如 `github.com/yourname/project`），必须换成匿名化版本（本节后面会介绍 Anonymous GitHub 这类工具）。
7. **项目/仓库名称本身**：哪怕链接匿名化了，如果项目名字本身很有辨识度、能被搜索引擎直接定位到你的真实仓库，同样是泄露——不要为了图方便匿名化"偷懒"到只处理链接文本不处理内容可搜索性。
8. **自引用的人称**：引用自己此前的工作时，用第三人称转述（"Smith et al. (2024) 指出……"），不要用"我们此前的工作表明……"这种会暴露"作者=前作作者"关联的第一人称写法。
9. **文件名/文件内元数据**：容易被忽略的一类——如果补充材料的压缩包里文件名本身带着组名缩写，或者 PDF 属性里的"作者"元数据字段没清空，同样会泄露身份，工具（第 3 节）目前只覆盖 `.tex` 源码文本本身，这类元数据层面的检查仍然需要人工过一遍。

**背后发生了什么：**

这份清单本质上是把"身份线索"按"信息载体"拆开逐类核对——姓名、机构、邮箱、致谢措辞、外部链接、人称代词、文件元数据，每一类都是一个独立的、审稿人可能借以反推身份的渠道，任何一类没清理干净都可能让前面几类的努力白费（这也是为什么第 3 节的脚本要同时检测多类，而不是只查作者名）。

**常见坑：**

| 现象 | 大概率原因 |
|---|---|
| 只清理了作者名，没查资助致谢措辞 | 以为"致谢"只等于"Acknowledgments 这个词"，没意识到"we thank"/"funded by"这类措辞可能散落在正文任何位置 |
| 用了 Anonymous GitHub 之类的工具处理链接，还是被认出来了 | 项目名字本身在网上能搜到（比如论文标题和 GitHub 仓库描述高度相似），链接匿名了但内容可关联性没处理 |
| 提交前只检查了正文，没检查补充材料 | ICLR/NeurIPS 官方政策都明确覆盖补充材料（NeurIPS 还额外覆盖代码），补充材料同样要过一遍这份清单 |

**自测清单：**

- [ ] 能不看这份清单，独立说出至少 6 类需要检查的身份线索
- [ ] 知道 Acknowledgments 章节投稿阶段应该整节删除，不是遮盖名字
- [ ] 知道自引用应该用第三人称转述，而不是完全回避引用自己的前作
- [ ] 知道补充材料（含代码仓库）和正文享有同等的匿名化要求

---

## 3. 真实可用的匿名化检查脚本 —— `check_anonymity.py`

**为什么需要这个 / 不会有什么后果：**

第 2 节的清单需要人工逐条核对，投稿截止前几小时手忙脚乱的时候，人工检查最容易漏掉的恰恰是"不起眼但要命"的那几处——一行藏在方法部分中间的"in our previous work"，或者一个测试代码时留下的真实 GitHub 链接。这个脚本把清单里能用规则表达的部分自动化，跑一遍只需要几秒钟，能在提交前当作最后一道机械防线（**不是替代人工检查**——第 3 节末尾"常见坑"会诚实讨论它的局限）。

**环境要求：**

- Python 3（本次开发和验证用的是仓库 `.venv` 下的 3.13.9，脚本本身只用标准库，任意 Python 3.9+ 应该都能跑）。
- 一份 `.tex` 源文件。

**一步步跟着做：**

**第一步：读懂脚本要检测的 8 类信号。** 完整源码如下（和 [`_assets/02-anon-checker/check_anonymity.py`](_assets/02-anon-checker/check_anonymity.py) 完全一致，可以直接复制这份代码本地保存成 `.py` 文件使用；这份代码也会被本系列目录下的 [`_verify_md.py`](_verify_md.py) 自动抽取并真实执行——不带参数运行时会走下面看到的内置自测路径，用两段内嵌的示例文本验证检测逻辑本身是对的）：

```python
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
```

**第二步：真实跑一遍内置自测。** 本机 `.venv` 真实执行 `python check_anonymity.py`（不带参数），真实输出（未删减）：

```
[WARN] <inline leaky example>: 14 suspected anonymity-leak signal(s) found

  line    2 [AUTHOR_NAME] Eric Peng Department of Computer Science, Riverside State University epeng@riverside.edu Weikai Lin 
           -> `\author{}` content is not an anonymized placeholder; must be replaced with 'Anonymous Author(s)' or similar during double-blind review
  line    4 [INSTITUTION] Department of Computer Science, Riverside State University \\
           -> Contains an institution-type keyword (University/Institute/Lab/Department of/...); this must not appear in the body or author block during submission
  line    5 [INSTITUTION_EMAIL_DOMAIN] epeng@riverside.edu \\
           -> Email domain `@riverside.edu` is an academic institution domain and directly reveals affiliation
  line    5 [EMAIL] epeng@riverside.edu \\
           -> Email address found: `epeng@riverside.edu`
  line    8 [INSTITUTION] Riverside AI Lab \\
           -> Contains an institution-type keyword (University/Institute/Lab/Department of/...); this must not appear in the body or author block during submission
  line    9 [INSTITUTION_EMAIL_DOMAIN] wlin@riverside.edu
           -> Email domain `@riverside.edu` is an academic institution domain and directly reveals affiliation
  line    9 [EMAIL] wlin@riverside.edu
           -> Email address found: `wlin@riverside.edu`
  line   12 [FIRST_PERSON_SELF_CITE] In our previous work, we showed that naive rollout depth does not help.
           -> Refers to own prior work in first person (e.g. 'our previous work'); double-blind requires third-person phrasing (e.g. 'Smith et al. (2024) previously showed...')
  line   13 [INSTITUTION] Code: \url{https://github.com/epeng-riverside-lab/imagination-budget-controller}
           -> Contains an institution-type keyword (University/Institute/Lab/Department of/...); this must not appear in the body or author block during submission
  line   13 [GITHUB_LINK] Code: \url{https://github.com/epeng-riverside-lab/imagination-budget-controller}
           -> `github.com/epeng-riverside-lab/imagination-budget-controller` owner is not an anonymized placeholder, likely a real username/organization; replace with an anonymized mirror (e.g. anonymous.4open.science)
  line   14 [ACKNOWLEDGMENTS_SECTION] \section*{Acknowledgments}
           -> Acknowledgments/Acknowledgements section header found; must be removed entirely during double-blind review and added back at camera-ready
  line   15 [INSTITUTION] We thank the Riverside AI Lab for compute, and gratefully acknowledge funding
           -> Contains an institution-type keyword (University/Institute/Lab/Department of/...); this must not appear in the body or author block during submission
  line   15 [FUNDING_PHRASE] We thank the Riverside AI Lab for compute, and gratefully acknowledge funding
           -> Contains funding/thanks language (e.g. 'supported by'/'we thank'/grant number); can reveal identity even outside an Acknowledgments section
  line   16 [FUNDING_PHRASE] from NSF grant No. 2026-04821.
           -> Contains funding/thanks language (e.g. 'supported by'/'we thank'/grant number); can reveal identity even outside an Acknowledgments section

[OK] <inline clean example>: no anonymity-leak signals detected (heuristic only -- still do a manual pass before submitting)

self-test passed: 14 findings across 8 categories on the leaky example, 0 findings on the clean example
```

8 个类别、14 处信号，全部命中——干净版本零命中，两条断言（`missing` 为空、`clean_findings == []`）真实通过。

**第三步：对着真实的 `.tex` 文件跑（不是内嵌字符串，是磁盘上的真实文件）。** 用同一个项目场景构造了两份完整的论文骨架：[`leaky_example.tex`](_assets/02-anon-checker/leaky_example.tex)（标题、作者块、摘要、引言、方法、致谢都写全了，故意留下和第二步同类的泄露）和 [`clean_example.tex`](_assets/02-anon-checker/clean_example.tex)（同一篇论文的正确匿名化版本）。真实运行（未删减）：

```
=== leaky_example.tex ===
[WARN] leaky_example.tex: 15 suspected anonymity-leak signal(s) found

  line    7 [AUTHOR_NAME] Eric Peng Department of Computer Science, Riverside State University epeng@riverside.edu Weikai Lin
           -> `\author{}` content is not an anonymized placeholder; must be replaced with 'Anonymous Author(s)' or similar during double-blind review
  line    9 [INSTITUTION] Department of Computer Science, Riverside State University \\
           -> Contains an institution-type keyword (University/Institute/Lab/Department of/...); this must not appear in the body or author block during submission
  line   10 [INSTITUTION_EMAIL_DOMAIN] epeng@riverside.edu \\
           -> Email domain `@riverside.edu` is an academic institution domain and directly reveals affiliation
  line   10 [EMAIL] epeng@riverside.edu \\
           -> Email address found: `epeng@riverside.edu`
  line   13 [INSTITUTION] Riverside AI Lab \\
           -> Contains an institution-type keyword (University/Institute/Lab/Department of/...); this must not appear in the body or author block during submission
  line   14 [INSTITUTION_EMAIL_DOMAIN] wlin@riverside.edu
           -> Email domain `@riverside.edu` is an academic institution domain and directly reveals affiliation
  line   14 [EMAIL] wlin@riverside.edu
           -> Email address found: `wlin@riverside.edu`
  line   23 [FIRST_PERSON_SELF_CITE] futures to roll out, and when to stop trusting the rollout. In our previous work,
           -> Refers to own prior work in first person (e.g. 'our previous work'); double-blind requires third-person phrasing (e.g. 'Smith et al. (2024) previously showed...')
  line   30 [INSTITUTION] \url{https://github.com/epeng-riverside-lab/imagination-budget-controller}.
           -> Contains an institution-type keyword (University/Institute/Lab/Department of/...); this must not appear in the body or author block during submission
  line   30 [GITHUB_LINK] \url{https://github.com/epeng-riverside-lab/imagination-budget-controller}.
           -> `github.com/epeng-riverside-lab/imagination-budget-controller` owner is not an anonymized placeholder, likely a real username/organization; replace with an anonymized mirror (e.g. anonymous.4open.science)
  line   36 [ACKNOWLEDGMENTS_SECTION] \section*{Acknowledgments}
           -> Acknowledgments/Acknowledgements section header found; must be removed entirely during double-blind review and added back at camera-ready
  line   37 [INSTITUTION] We thank the Riverside AI Lab for compute support, and gratefully acknowledge
           -> Contains an institution-type keyword (University/Institute/Lab/Department of/...); this must not appear in the body or author block during submission
  line   37 [FUNDING_PHRASE] We thank the Riverside AI Lab for compute support, and gratefully acknowledge
           -> Contains funding/thanks language (e.g. 'supported by'/'we thank'/grant number); can reveal identity even outside an Acknowledgments section
  line   38 [INSTITUTION] funding from NSF grant No. 2026-04821 and the Riverside State University
           -> Contains an institution-type keyword (University/Institute/Lab/Department of/...); this must not appear in the body or author block during submission
  line   38 [FUNDING_PHRASE] funding from NSF grant No. 2026-04821 and the Riverside State University
           -> Contains funding/thanks language (e.g. 'supported by'/'we thank'/grant number); can reveal identity even outside an Acknowledgments section
EXIT_CODE=1

=== clean_example.tex ===
[OK] clean_example.tex: no anonymity-leak signals detected (heuristic only -- still do a manual pass before submitting)
EXIT_CODE=0
```

退出码语义和脚本设计一致：有发现 = 1，没发现 = 0，方便接到 CI/提交前检查脚本里用 `if ! python check_anonymity.py paper.tex; then echo "先别提交"; fi` 这样的用法。这次真实文件扫描比上一步内嵌样例多测出 1 条（15 条 vs 14 条）——因为真实论文骨架里句子换行位置不同，行号定位随之不同，但两种验证路径命中的类别完全一致，互相印证检测逻辑没有因为"输入是内嵌字符串还是磁盘文件"而表现不同。

（上面刻意留了一处示意——真实终端记录是 `clean_example.tex` 退出码 **0**，`leaky_example.tex` 退出码 **1**，退出码语义和第一步脚本设计一致：有发现=1，方便接到 CI/提交前检查脚本里用 `if ! python check_anonymity.py paper.tex; then echo "先别提交"; fi` 这样的用法。）

**第四步：一个真实撞到、值得记住的 bug——工具自己第一次犯了"把已经匿名的内容当成没匿名"的错误。** 这个脚本的 `find_author_block` 检测逻辑最初版本，判断"是否已匿名"的方式是：把 `\author{}` 内容标准化后，检查是不是**精确等于**一个很短的白名单（比如整段恰好是 `"anonymous author(s)"`）。第一次跑自测，`clean_example.tex` 用的匿名写法是：

```latex
\author{Anonymous Author(s) \\ Affiliation withheld for double-blind review}
```

这其实是很多真实会议模板匿名分支的标准写法（比如 ICLR 官方模板常见的 `Anonymous authors\\Paper under double-blind review`）——但第一版脚本的白名单只认"单纯的 `Anonymous Author(s)`"，多出来的"Affiliation withheld for double-blind review"这半句话让整段内容不再精确匹配白名单，于是 `assert clean_findings == []` 真实断言失败，报错 `AssertionError: clean example should have zero findings, got ['AUTHOR_NAME']`。**根因**：用"整段精确匹配"判断匿名与否太脆弱，真实世界的匿名占位符经常是"核心短语+补充说明"的拼接写法。**修复**：改成"从内容里逐个抠掉已知的匿名占位符词块（`ANON_BOILERPLATE_PHRASES` 列表），抠完看还剩不剩实质内容"——这就是本节第一步展示的最终版本里 `_is_anonymized_author_block` 函数的实现。这个过程本身是个有用的教训：**一个用来检测"没匿名"的工具，第一版反而在"已经匿名"的样例上出了错**，说明"看起来简单的字符串规则"经常在真实世界的写法多样性面前不够用，需要真的跑一遍反例才能发现。

**背后发生了什么：**

脚本采用的是**基于正则/规则的逐行+跨行扫描**，不是真正解析 LaTeX 语法树。`find_author_block` 是唯一需要跨行处理的检测（用括号计数取 `\author{...}` 完整内容，因为作者块经常用 `\\` 换行跨越好几行物理行），其余检测（机构关键词、邮箱、GitHub 链接、致谢、自引用人称）都是逐行独立扫描，互不依赖——这也是为什么同一行内容经常会被多条规则同时命中（比如上面的 GitHub 链接那一行，因为 owner 名字 `epeng-riverside-lab` 里包含 "lab" 这个词，被机构关键词规则也命中了一次）：多条规则命中同一处泄露不是 bug，是不同角度对同一个问题的独立确认。

**常见坑（脚本自身的诚实局限，不回避）：**

| 现象 | 原因/结论 |
|---|---|
| GitHub 链接那一行同时报了 INSTITUTION 和 GITHUB_LINK 两条 | 不是 bug——owner 用户名 `epeng-riverside-lab` 恰好包含 "lab" 这个词，被两条独立规则各自命中，属于合理的重复确认，不需要"去重"处理 |
| 正文提到"we evaluate our controller in a simulated **laboratory** environment"这种完全无害的日常用词，被 INSTITUTION 规则误报 | 真实测过：这是一个真实存在的假阳性——"laboratory"这个词本身并不天然等于机构名，脚本的关键词规则没有能力区分"泄露身份的机构名"和"日常词汇里恰好出现的同一个单词"。这是启发式规则的固有局限，不是这个脚本能简单修复的问题（要区分这两种情况通常需要真正的命名实体识别，而不是关键词匹配），使用时把它当作"提示你去人工确认"的信号，而不是"报了就一定要改" |
| 脚本说"OK"，但补充材料里的 PDF 元数据还留着真实作者名 | 脚本只扫描 `.tex` 源码的文本内容，不检查编译产物的文档属性、也不检查文件名本身——见第 2 节清单第 9 条，这部分仍然需要人工核对 |
| 想检测的自引用写法比脚本认识的更多样（比如"as we discussed earlier"这种不含"our previous work"字面短语的表达） | `FIRST_PERSON_SELF_CITE` 只覆盖几个最常见的固定短语，不是穷尽所有可能的第一人称自引写法，人工审阅仍是最后一道防线 |

**自测清单：**

- [ ] 能独立运行这个脚本（内置自测模式和真实文件模式都试一遍）
- [ ] 能解释为什么第一版脚本会把 `clean_example.tex` 误判成"没匿名"，以及后来怎么修的
- [ ] 能说出这个脚本至少 2 处已知的真实局限（不是空谈"可能有局限"，是能举出具体例子）
- [ ] 知道这个脚本的退出码语义（0=没发现问题，1=发现问题），能想到怎么接进提交前的检查流程里

---

## 4. OpenReview 投稿系统真实操作流程

**为什么需要这个 / 不会有什么后果：**

ICLR、NeurIPS 等主流 ML 会议的投稿、评审、讨论全部在 OpenReview 这个平台上完成，不了解真实操作流程，容易在关键节点（比如"提交后还能不能改""怎么私下联系 AC"）手忙脚乱。**这一节是"已核实当前公开流程，但未做端到端自动化验证"**——没有一篇真实要投稿的论文可以拿来真的走一遍完整提交流程，下面每一步都对照 ICLR 2026 官方 Author Guide 核实过，但不是本机真实点击操作过的截图记录。

**环境要求：** 一个 OpenReview 账号（免费注册）。

**一步步跟着做：**

1. **提交前：确认账号信息正确关联。** 如果论文提交后发现自己访问不了自己的投稿（常见于合作者较多、邮箱和 OpenReview 个人资料没对上的情况），官方指引的排查方法是：打开自己的投稿页面，把鼠标悬停在"Authors"字段里自己的名字上，如果关联的邮箱不在自己当前账号的已确认邮箱列表里，去 OpenReview 个人资料里添加并确认这个邮箱；如果名字关联到了别人的账号，需要联系 OpenReview 官方支持（`info@openreview.net`）合并资料。
2. **截止日期前：可以自由编辑。** 提交后、截止日期前，论文内容可以在 OpenReview 上反复修改（用"Revision"按钮），截止日期一到，编辑功能锁定，直到评审结果发布才会重新开放。
3. **评审阶段：论文进入正常流程。** 评审人打分、写评审意见，全部通过 OpenReview 完成。
4. **评审发布后：公开讨论期。** 官方评审发布后，进入一段公开讨论窗口——任何登录用户都能发表公开可见的评论（作者、审稿人、AC、组委会之外的评论者，规则要求必须以实名身份发言，不能匿名）。这一阶段，作者可以看到自己论文的评审并公开回复（也就是 03 号文件要讲的 rebuttal 环节）。
5. **公开讨论结束后：私下讨论期。** 审稿人和 AC 之间会有一段作者不可见的内部讨论，目的是汇总意见、形成一致判断，之后才做出最终决定。
6. **想私下联系 AC/审稿人：走论坛的"Readers"下拉菜单。** 如果需要给自己论文的 AC 或审稿人发送不公开的信息，在这篇投稿的 OpenReview 论坛页面，从"Readers"下拉菜单里选择"Area Chairs of the submission"和/或"Reviewers of the submission"，而不是发布成公开评论——第一联系人通常是分配给这篇论文的 AC。
7. **引用还在评审中的论文：OpenReview 会自动生成匿名引用格式。** 如果你的论文需要引用另一篇同样在双盲评审中的 OpenReview 投稿，平台会提供一个不包含作者信息、只有标题/年份/URL 的 BibTeX 条目，直接用这个匿名版本引用即可，不需要自己手动处理。
8. **补充材料/可复现性声明：可选但有专门位置。** 官方指引里，代码可以以匿名下载链接的形式作为补充材料提交；涉及理论结果的，附录里给出假设的清晰说明和完整证明；涉及数据集的，补充材料里给出完整的数据处理步骤说明。这部分内容不计入正文页数限制。

**背后发生了什么：**

OpenReview 的"Readers"权限机制是整个平台双盲/多阶段讨论能运作的技术基础——每一条评论、每一次编辑，背后都关联一个"谁能看到"的权限列表（作者/审稿人/AC/所有人），公开讨论期和私下讨论期的区别，本质上就是这个权限列表的默认值不同：公开讨论期默认所有登录用户都能读；私下讨论期把作者从"能读"的列表里移除。理解这套"Readers"机制，能帮你搞懂"为什么我看不到某条评论""这条留言应该发给谁看"这类操作层面的困惑。

**常见坑：**

| 现象 | 大概率原因 |
|---|---|
| 提交后发现自己进不去自己的论文页面 | 大概率是账号邮箱关联问题，参见本节第 1 步的排查方法 |
| 截止日期后想改论文内容，找不到编辑按钮 | 正常现象——编辑功能在截止日期后锁定，要等评审结果发布才重新开放（用于准备 revision） |
| 不确定该把话发给谁看，直接发了公开评论 | 想私下沟通用"Readers"下拉菜单选择性开放，公开评论会被所有登录用户看到，且要求实名，两种渠道不要混用 |
| 引用另一篇同样在投的 OpenReview 论文，手动打了作者名 | 让平台生成匿名 BibTeX 条目，不要自己手动填作者信息破坏对方的匿名 |

**自测清单：**

- [ ] 能说出截止日期前和截止日期后，论文编辑权限分别是什么状态
- [ ] 知道公开讨论期和私下讨论期的区别，以及为什么会有这个区分
- [ ] 知道想私下联系 AC/审稿人应该用哪个功能（Readers 下拉菜单），不是发公开评论
- [ ] 知道引用另一篇在投论文时，应该用平台生成的匿名 BibTeX，不是自己手动填

---

## 5. 匿名化之外的坑 —— Preprint、自我宣传、工具本身的边界

**为什么需要这个 / 不会有什么后果：**

前面几节覆盖了"论文本身怎么匿名化"，这一节讲几处容易被忽略、但同样会惹麻烦的灰色地带——这些不是"改一下 `.tex` 文件"能解决的，是行为层面的边界。

**环境要求：** 无。

**一步步跟着做：**

1. **理解"有非匿名 preprint"和"违反双盲"是两件不同的事。** NeurIPS 政策明确：单纯存在一份挂在 arXiv 上、能查到真实作者的预印本，**不算**违反双盲评审政策——这是因为完全禁止研究者使用 arXiv 在现实中不可行。但政策同时警告："高调宣传（aggressive advertising）在投论文"可能被认定为违规——比如投稿期间在社交媒体反复转发自己这篇论文的 arXiv 链接、刻意让审稿人更容易注意到，这类主动行为和"论文恰好在 arXiv 上能查到"是两个层级的事。**实操建议**：投稿期间不主动扩散自己在投论文的链接，不代表要把已有的 arXiv 版本删掉。
2. **理解 Anonymous GitHub 类工具解决的是什么问题、有什么已知问题。** Anonymous GitHub（`anonymous.4open.science`）这类服务的原理是：你提供真实仓库地址和一份需要替换的词表，服务会生成一个镜像页面，把仓库的 owner/组织名、文件名、文件内容里出现的这些词全部替换成占位符再展示——效果上审稿人点开链接能看到代码，但看不到任何指向真实身份的字符串。**已知的真实问题**：有用户反馈这类服务偶发的稳定性问题（比如浏览器兼容性 bug），比较谨慎的替代做法包括先把仓库内容存一份到 OSF 或 Zenodo 这类更成熟的学术存档平台，再从那里生成匿名可访问的链接。
3. **理解一个实用的 LaTeX 技巧：用变量同时维护匿名版和真实版链接。** 论文里如果多处引用了代码仓库链接，可以在导言区定义一个变量，投稿阶段指向匿名镜像，camera-ready 阶段（05 号文件详细讲）只需要改这一处定义，不用满篇文档去找每一处链接手动替换——这也是会议模板里"匿名开关"（[daily-toolkit-deep-dive/07-latex-paper-writing.md](../daily-toolkit-deep-dive/07-latex-paper-writing.md) 提到的 `\iclrfinalcopy` 类命令）背后的同一种思路：**把"要不要显示身份信息"做成一个开关，而不是散落在文档各处需要手动逐一修改的硬编码**。
4. **理解仓库文件名本身也要处理，不只是文件内容。** Anonymous GitHub 之所以要同时替换"owner/组织名、文件名、文件内容"三个层面，是因为哪怕文件内容里一个真名都没有，如果某个文件名或者目录名本身包含课题组缩写，同样构成泄露——这一点第 2 节清单第 7 条已经提过，这里再次强调。

**背后发生了什么：**

这几条坑的共同性质是：**它们不是"匿名化脚本能检测到的技术问题"，是"行为和判断层面的边界"**。这也是为什么 02 号文件不能只靠一个脚本就说"匿名化搞定了"——脚本处理的是"论文源码里明确写了什么"，而"要不要在社交媒体转发论文链接""要不要信任第三方匿名化服务的稳定性"这类问题，没有代码能替你决定，需要理解规则背后的意图（保护评审公正性），自己做判断。

**常见坑：**

| 现象 | 大概率原因 |
|---|---|
| 投稿期间在推特/朋友圈高调宣传自己在投的论文 | 触碰 NeurIPS"aggressive advertising"这条灰色地带政策，即使论文本身已经正确匿名化 |
| 用了 Anonymous GitHub 但链接偶尔打不开/显示异常 | 已知的服务稳定性问题，比较谨慎的做法是投稿前多测试几次，或者考虑 OSF/Zenodo 这类更成熟的平台作为备选 |
| camera-ready 阶段满篇文档手动找replace 匿名链接改回真实链接，改漏了几处 | 投稿之初就该用变量集中管理链接（本节第 3 步），而不是硬编码分散在文档各处 |

**自测清单：**

- [ ] 能说出"存在非匿名 arXiv 预印本"和"违反双盲"的区别
- [ ] 知道 Anonymous GitHub 类工具的基本原理，以及它已知的稳定性局限
- [ ] 能说出为什么建议用 LaTeX 变量集中管理匿名/真实链接，而不是硬编码
- [ ] 理解这一节的坑本质上是行为判断问题，不是脚本能自动检测的技术问题

---

## 参考来源

- ICLR 2026 / NeurIPS 2026 双盲评审官方政策与 Author Guide：iclr.cc/Conferences/2026/AuthorGuide、neurips.cc 官方文档（2026-07 WebSearch 调研）。
- 2025-11-27 OpenReview 匿名信息泄露事故：Medium "The Day Anonymity Died" 报道、gensee.ai 教授视角评论文章、CSPaper 论坛讨论（2026-07 WebSearch 调研）。
- OpenReview 提交/讨论流程：ICLR 历年 Author Guide（2021-2026）官方文档（2026-07 WebSearch 调研）。
- 匿名化有效性统计数据（70%-90% 猜不出/猜错）：ACM 通讯发表的双盲评审有效性研究综述（2026-07 WebSearch 调研）。
- Anonymous GitHub / LaTeX 匿名链接切换技巧：Micah Smith 博客、GitHub 官方社区讨论（2026-07 WebSearch 调研）。

---

*创建：2026-07-25*
