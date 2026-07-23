# 07 · LaTeX 论文写作实操 —— 从零到能编译出 ICLR 投稿

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一章直接服务于你自己接下来要写的 ICLR 论文——不是"顺便学一门排版语言"，是"两个月后你会真的用这些操作产出投稿 PDF"。

**为什么这一章格外重要:** 你在 Weikai Lin 老师指导下做的 world model 想象预算分配控制器研究，目标就是投 ICLR。写论文这件事本身（不是做研究本身）也是一项要花真金白银时间成本的技能——如果到截止日期前几天才第一次摸 LaTeX，光是排版报错、参考文献格式对不上、图表卡在奇怪的位置这些"操作层面"的问题，就足以吃掉你本该用来打磨论文内容的时间。这一章的目标是让你现在、不慌不忙的时候，就把这些操作层面的东西过一遍。

**这一章和仓库里"深挖系列"的区别，以及和 `_build_pdf.py` 的区别（避免混淆）:**
- 深挖系列（`alignment-algorithms-deep-dive` 这些）回答"这个机制为什么这样设计"；这一章回答"具体怎么操作、操作时该看到什么、卡住了怎么办"，参见 [00-roadmap.md](00-roadmap.md) 的系列定位说明。
- `research/world-model-imagination-controller/_build_pdf.py` 是一条"把 Markdown 讲义转成 PDF"的流水线（pandoc + xelatex，用来把会前简报这类内部文档排版好看），它解决的是完全不同的问题，路径是 `Markdown → pandoc → xelatex → PDF`。**你写 ICLR 论文不会走这条路**，会议要求的是**直接手写的 LaTeX 源文件**（`.tex`），路径是 `你手写的 .tex → xelatex/pdflatex → PDF`，中间没有 pandoc、也不是从 Markdown 转换来的。这一章聚焦的是后者，两者除了都用到 "xelatex" 这个排版引擎之外，没有其他共享的地方。

**验证环境（真实确认，不是假设）:** 本机装的是 MiKTeX，以下版本都已现场跑 `--version` 确认可用：

```text
xelatex : MiKTeX-XeTeX 4.8 (MiKTeX 22.3)
pdflatex: MiKTeX-pdfTeX 4.10 (MiKTeX 22.3)
bibtex  : MiKTeX-BibTeX 4.1 (MiKTeX 22.3)
```

本章**几乎每一节都真实编译出了 PDF**，编译产物（`.tex` 源文件、编译出的 `.pdf`、以及 `.aux`/`.log`/`.bbl`/`.blg` 这些编译过程中的中间文件）全部保留在 [`_assets/07-latex-demos/`](_assets/07-latex-demos/) 目录下，按小节分成 5 个子目录，本文所有"真实编译"的表述都可以在那里找到对应的原始产物核对。图表那一节用到的占位图是用仓库 `.venv` 里的 matplotlib（3.10.9）现场生成的，不是网上下载的。

**MiKTeX 缺包自动安装怎么处理（写在最前面，因为后面每一节都依赖这个）:** 本机 MiKTeX 的包管理器配置了自动安装（用 `initexmf --show-config-value=[MPM]AutoInstall` 查询，返回 `1`），额外在每条编译命令上加了 `-enable-installer` 参数（xelatex/pdflatex 都认这个参数，作用是"缺什么宏包就自动装，不要弹窗问我"）。本章所有编译命令的真实形态是：

```bash
xelatex  -enable-installer -interaction=nonstopmode -halt-on-error 文件名.tex
pdflatex -enable-installer -interaction=nonstopmode -halt-on-error 文件名.tex
```

`-interaction=nonstopmode` 让编译器遇到问题不要停下来等你在终端里按键确认（这在你自己电脑上手动敲命令时不加也行，但写进脚本/自动化流程里必须加，否则一旦出错整个终端会卡住等一个永远不会来的按键输入）。`-halt-on-error` 让编译器一遇到第一个致命错误就停止，方便看清最原始的报错（不加这个参数它会尝试硬着头皮往下编，经常在后面级联出一堆看起来吓人、实际上都是同一个根因的连锁报错）。本章十几次真实编译过程中，用这套参数没有出现过一次卡死或交互式弹窗。诚实说明一点：本机需要的几个宏包（`amsmath`/`amssymb`/`graphicx`/`booktabs`/`hyperref`）本身是 LaTeX 里最常用的基础宏包，编译时发现它们已经在本机装好了，所以并没有亲眼看到一次"正在下载 xxx.sty"的实时安装过程——但确认了 `-enable-installer` 这个参数在这台机器上真实生效、不会卡死，如果你自己机器上真的触发了缺包，这个参数会让 MiKTeX 自动装完继续编译，而不是弹一个对话框等你点"是"。

---

## 1. 为什么用 LaTeX 写论文而不是 Word

**为什么需要这个 / 不会有什么后果:**

如果你没搞清楚这一步就直接跳去学 LaTeX 语法，会陷入"学一门更麻烦的排版语言，就因为大家都说它'更专业'"这种空洞印象——这撑不起你熬夜改语法报错时的耐心。你需要几条具体到"少了它会怎样"的理由：

1. **数学公式排版质量。** LaTeX 的公式排版算法是 Donald Knuth 专门为数学排版设计的（这也是"TeX"这个词的来源），自动处理分数线粗细、根号包裹大小、上下标字号、括号自动缩放这些细节，几十个公式下来风格完全一致。Word 的公式编辑器（无论是自带的还是 MathType）是"所见即所得"手动调间距，公式一多，风格不一致、复制粘贴错位这些问题会指数级增多——你的 ICLR 论文里大概率会有十几到几十个公式（loss 函数、算法推导），这个差距会被放大到肉眼可见。
2. **参考文献自动管理。** 用 BibTeX（第 5 节详细讲），你维护一个文献数据库文件（`.bib`），正文里只写 `\cite{key}`，编号、排序、格式全部自动生成——加一篇新引用只需要加一条数据库条目，不需要手动重新数编号、重新排文末列表。Word 的引用功能能做类似的事，但格式经常在"引用管理器版本升级"或者"多人协作"时出岔子，手动改的引用列表几乎必然会出现"正文写的是 [7]，文末列表却对不上"这种低级但致命的错误。
3. **大部分顶会要求 LaTeX 模板投稿。** ICLR/NeurIPS/ICML/ACL 这些顶会的页边距、字号、页数限制、匿名审稿要求，都是通过官方发布的 LaTeX 宏包（`.sty` 文件）强制实现的（第 7 节详细讲）。这些会议要么没有对应的 Word 模板，要么 Word 模板年久失修、格式经常出错。用 Word 写完论文再手动排成会议要求的格式，几乎不可能不出错；用官方 LaTeX 模板，这些格式要求是"写论文之前就已经锁死"的，你不需要（也不被允许）自己去调。
4. **纯文本、对 Git 友好。** `.tex` 是纯文本文件，`git diff` 能清楚看到每次改动了哪一行——这跟你已经在 Git/GitHub 协作里体会到的"能追溯每一处修改"的价值是一回事。Word 的 `.docx` 本质上是压缩过的 XML，虽然技术上也能塞进 Git，但 diff 出来是不可读的乱码，多人协作、多轮修改时的"改了什么"完全没法追溯，只能依赖 Word 自己的"批注 + 修订模式"，而这套机制在多人来回传文件时经常冲突或丢失修改记录。

**环境要求:** 这一节是动机层面的内容，不需要装任何东西；如果你想现在就确认自己机器的 LaTeX 环境是否就绪，跳到第 2 节的"环境要求"直接执行那里的检查命令。

**一步步跟着做:** 这一节没有操作步骤——它的作用是在你开始敲语法之前，先在脑子里把"为什么要用这个工具"这件事钉死，避免学了一堆语法却说不出"比 Word 好在哪"。

**背后发生了什么:**

LaTeX 的工作模式和 Word 有本质区别，值得用你已经很熟悉的类比说清楚：**LaTeX 论文写作之于最终 PDF，相当于 C 源代码之于可执行程序。** 你写的 `.tex` 文件是"标记 + 指令"（类似源代码），排版引擎（`xelatex`/`pdflatex`，相当于编译器）把它转换成最终产物（PDF），你在写作过程中看到的是标记文本本身，不是最终排好版的样子。Word 是所见即所得（WYSIWYG）——你在屏幕上看到的，近似就是打印出来的样子，编辑和产物之间没有一个独立的"编译"步骤。这个区别决定了前面 4 条理由为什么成立：正因为 LaTeX 有一个独立的编译步骤，排版细节（公式间距、引用编号、页边距）才能被算法或者宏包统一接管，而不需要你手动一个个调。

**常见坑:**

| 现象 | 解释 |
|---|---|
| 觉得"LaTeX 就是加了很多反斜杠的 Word" | 低估了两者工作模式的本质区别；LaTeX 没有"实时预览最终样子"这件事（少数编辑器有近似的实时预览，但原理仍然是后台不断重新编译，不是真正意义的所见即所得） |
| 觉得必须先啃完整套 LaTeX 语法才能开始写论文 | 不需要，本章后面几节要用到的语法覆盖了写一篇 ML 论文 90% 的日常需求；渐进学习，先能编译最小文档（第 2 节），再一点点加内容 |
| 觉得"专业感"是唯一理由 | 本节列的 4 条都是具体到"少了它会怎样"的工程理由，不是审美偏好 |

**自测清单:**
- 你能不看这一节，自己说出至少两条"为什么 ML 论文用 LaTeX 而不是 Word"的具体理由（不是"更专业"这种空话）吗？
- 你能说清楚"LaTeX 编译"和"Word 所见即所得"这两种工作流程的本质区别吗？
- 你知道为什么 ICLR 这类会议要求 LaTeX 模板投稿，而不太依赖 Word 模板吗？
- 你确认了本机 `xelatex`/`pdflatex`/`bibtex` 都可用了吗（第 2 节会给出检查命令）？

---

## 2. 最小可编译文档结构

**为什么需要这个 / 不会有什么后果:**

任何论文，不管最终多复杂（几十个公式、十几张图表、几十条参考文献），骨架永远是同一个三段结构：`\documentclass{...}` 声明文档类型，`\begin{document}...\end{document}` 界定正文范围。不先把这个骨架吃透，直接去看 ICLR 官方模板那几十行密密麻麻的导言区设置，会觉得无从下手——但那些复杂设置全部是在这个骨架基础上"加装饰"，骨架本身极其简单。类比你已经熟悉的 C 语言：这就像 `int main() { return 0; }`——是能编译运行的最小单元，后面所有复杂程序都是在这个骨架里继续填内容，不理解这个最小单元，直接看一个几千行的真实项目会晕头转向。

**环境要求:**

需要一个 TeX 发行版（本机是 MiKTeX），核实的命令和真实输出：

```text
$ xelatex --version
MiKTeX-XeTeX 4.8 (MiKTeX 22.3)

$ pdflatex --version
MiKTeX-pdfTeX 4.10 (MiKTeX 22.3)

$ bibtex --version
MiKTeX-BibTeX 4.1 (MiKTeX 22.3)
```

这一节不需要额外宏包——`article` 是 LaTeX 核心自带的文档类，不用 `\usepackage` 引入。

**一步步跟着做:**

完整文件（[`_assets/07-latex-demos/01-minimal-document/minimal.tex`](_assets/07-latex-demos/01-minimal-document/minimal.tex)）：

```latex
\documentclass[11pt]{article}

\title{A Minimal Compilable Document}
\author{Daily Toolkit Deep Dive --- Section 2}
\date{\today}

\begin{document}

\maketitle

Hello, LaTeX! This one paragraph is the entire body of the simplest
document that still compiles into a real, standalone PDF file.

\end{document}
```

编译命令（真实执行过）：

```bash
pdflatex -enable-installer -interaction=nonstopmode -halt-on-error minimal.tex
```

应该看到的现象——终端先滚过一堆字体/宏包加载信息，最后几行是关键：

```text
[1{C:/Users/ericp/AppData/Local/MiKTeX/fonts/map/pdftex/pdftex.map}]
(minimal.aux) )<.../cmr10.pfb><.../cmr12.pfb><.../cmr17.pfb>
Output written on minimal.pdf (1 page, 42118 bytes).
Transcript written on minimal.log.
```

真实产出：`_assets/07-latex-demos/01-minimal-document/minimal.pdf`，1 页，**42118 字节**，退出码 `0`。同目录下还多出了 `minimal.aux`（记录标签/引用信息，第 3 节会展开它的作用）和 `minimal.log`（完整编译日志，排查报错时要看这个，第 6 节会用到）。打开这个 PDF，应该看到居中的标题 "A Minimal Compilable Document"、作者行、日期，下方是那一段正文。

**背后发生了什么:**

- `\documentclass[11pt]{article}`：告诉排版引擎用哪种"文档类型"（`article`/`report`/`book` 等预定义了页面尺寸、章节层级、默认间距等一整套排版规则），`[11pt]` 是可选参数，指定基础字号。
- `\title{}`/`\author{}`/`\date{}` 只是"登记"这些信息，本身**不会**自动排版出来——必须调用 `\maketitle` 才会真正在文档里生成标题区块。这是新手最容易踩的第一个坑：写了 `\title{}` 但 PDF 里没有标题，往往是忘了 `\maketitle`。
- `\begin{document}...\end{document}` 界定"正文"范围。之前的部分叫**导言区（preamble）**，只能写宏包引入、全局设置这类"配置"，不能写正文内容——这是一个硬性边界，写错了会直接报错（第 6 节的错误排查方法同样适用）。
- 编译过程本质：`pdflatex` 读取 `.tex` 文件，从头到尾依次执行里面的宏指令：遇到 `\documentclass` 加载对应的类文件，遇到 `\usepackage` 加载对应宏包，排版引擎计算每个字符/公式/图表的位置，最终直接输出 PDF 字节流。这和 C 编译器把源码变成机器码是同一种"一次性转换"的思路，区别是 C 编译器产出的是可执行程序，这里产出的直接就是最终成品文档。
- 编译产生的副产品文件 `.aux`：记录标签、引用、章节编号等信息，供**下一遍**编译读取——这是第 3、5 两节要重点解释的"多遍编译"机制的起点，这里先混个脸熟。

**常见坑:**

| 现象 | 原因 |
|---|---|
| 写了 `\title{}` 但 PDF 里没有标题 | 忘记调用 `\maketitle` |
| 在 `\begin{document}` 之前手滑写了正文文字 | 导言区不能放正文内容，会直接报错，报错的读法见第 6 节 |
| 终端最后出现一行 `pdflatex: major issue: User/administrator updates are out-of-sync.` | 本机真实遇到过（每一次编译都会出现），经查是 MiKTeX 用户级/管理员级包数据库版本不同步导致的一条**良性提示**，不影响编译结果——本章所有编译在这条提示存在的情况下依然退出码为 0、PDF 正常产出。第一次看到这行别慌，先看 PDF 是否真的生成了 |
| 文件名/路径带中文或空格 | 少数环境下会导致编译工具解析路径出错；本章所有 demo 文件名故意只用英文和连字符，绕开这个问题，如果你自己的论文项目路径带中文，建议改成纯英文路径 |

**自测清单:**
- 你能不看着抄、自己独立写出一个能编译的最小 LaTeX 文档吗？
- 你知道 `\maketitle` 的作用、忘记写会有什么后果吗？
- 你知道导言区和正文的边界（`\begin{document}`）是什么、为什么要有这个边界吗？
- 你自己编译出 PDF 了吗（文件确实存在、能打开）？

---

## 3. 数学公式 —— 用你自己已经学过的 DPO loss 练手

**为什么需要这个 / 不会有什么后果:**

ML 论文的核心内容往往就是几个公式（loss 函数、算法推导），公式排版质量直接影响论文的可读性和"专业感"——评审扫论文时，公式部分排版混乱（间距挤在一起、上下标看不清）会显著拉低第一印象。这也是你接下来写 ICLR 论文里最高频会用到的操作，不是可有可无的装饰技能。

**环境要求:**

需要 `amsmath` 宏包（几乎所有数学论文都会加载，提供 `equation`/`align` 等环境和更规范的公式命令）和 `amssymb`（提供 `\mathbb` 等符号字体，本例的 `\mathbb{E}` 期望符号要用到）。两者在本机编译时都被正常加载，没有触发任何缺包提示。

**一步步跟着做:**

这一节直接用你在 [`alignment-algorithms-deep-dive/01-dpo-foundations.md`](../alignment-algorithms-deep-dive/01-dpo-foundations.md) 第 95-114 行附近已经推过、也在代码里验证过的 DPO loss 公式练手，而不是随便找个陌生公式——公式本身：

```text
L_DPO(π_θ; π_ref) = -E_{(x,y_w,y_l)~D}[ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

完整文件（[`_assets/07-latex-demos/02-math-formulas/dpo-formula.tex`](_assets/07-latex-demos/02-math-formulas/dpo-formula.tex)）：

```latex
\documentclass[11pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}

\title{Typesetting the DPO Loss in \LaTeX}
\author{Daily Toolkit Deep Dive --- Section 3}
\date{\today}

\begin{document}
\maketitle

\section{Inline math}

The Direct Preference Optimization (DPO) loss trains the policy
$\pi_\theta$ directly on preference pairs $(x, y_w, y_l)$, where $y_w$
is the preferred (``chosen'') response and $y_l$ is the dispreferred
(``rejected'') response, without ever training a separate reward
model. The temperature $\beta > 0$ controls how far $\pi_\theta$ is
allowed to drift from the frozen reference policy $\pi_{\text{ref}}$.

\section{Display math: the DPO loss}

\begin{equation}
  \label{eq:dpo-loss}
  \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) =
  -\, \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}
  \left[
    \log \sigma \left(
      \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
      - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
    \right)
  \right]
\end{equation}

Equation~\eqref{eq:dpo-loss} is exactly the quantity implemented by
\texttt{dpo\_loss} in \texttt{dpo\_minimal.py}: the two log-ratio
terms inside $\sigma(\cdot)$ correspond to \texttt{log\_ratio\_w} and
\texttt{log\_ratio\_l}, and the whole bracketed expression before the
leading minus sign and expectation corresponds to \texttt{margin},
which is passed through \texttt{F.logsigmoid}.

\section{A multi-line derivation with \texttt{align}}

\begin{align}
  \hat r_w &= \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} \\
  \hat r_l &= \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\end{align}

so that Equation~\eqref{eq:dpo-loss} can be read compactly as
$\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log \sigma(\hat r_w - \hat r_l)]$.

\end{document}
```

编译（真实执行两遍，原因见下面"背后发生了什么"）：

```bash
pdflatex -enable-installer -interaction=nonstopmode -halt-on-error dpo-formula.tex   # 第一遍
pdflatex -enable-installer -interaction=nonstopmode -halt-on-error dpo-formula.tex   # 第二遍
```

**第一遍**编译真实输出里出现了这几行（摘录）：

```text
Overfull \hbox (9.90776pt too wide) detected at line 34
LaTeX Warning: Reference `eq:dpo-loss' on page 1 undefined on input line 36.
...
LaTeX Warning: There were undefined references.
LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.
Output written on dpo-formula.pdf (2 pages, 158000 bytes).
```

`Reference 'eq:dpo-loss' undefined` 不是我特意制造的错误——这是**第一次编译必然出现**的正常现象（下面解释原因），最后一行"Rerun to get cross-references right"就是 LaTeX 自己在提示你要再编译一次。**第二遍**编译后，这些警告全部消失，`\eqref{eq:dpo-loss}` 正确显示成 "Equation (1)"。最终稳定产物：`_assets/07-latex-demos/02-math-formulas/dpo-formula.pdf`，2 页，**147858 字节**（第二遍编译后的最终大小，第一遍产出的 158000 字节版本已被覆盖）。

**背后发生了什么:**

- `$...$` 是**行内数学模式**，公式和文字挤在同一行，字号、上下标会被压缩以适应行高，适合"$\beta > 0$ 控制偏离程度"这种嵌在句子里的短公式。
- `equation` 环境是**独立数学模式**，公式单独成行、居中、字号更舒展，比裸的 `\[...\]` 多了两个能力：自动编号（"(1)"、"(2)"……）和可以被 `\label` + `\eqref` 引用。
- `amsmath` 具体做了什么：重新定义了大量底层排版参数——分式的间距、`\left(`/`\right)` 自动缩放括号大小去包裹里面的内容、`align` 环境按 `&` 对齐符号（本例里两行的 `=` 号对得整整齐齐）。这些如果不用 `amsmath` 手动调，几乎不可能做到几十个公式风格统一。
- **交叉引用机制，也是"为什么第一遍编译会有 undefined 警告"的真正原因：** `\label{eq:dpo-loss}` 把"当前这个公式的编号"和你起的名字 `eq:dpo-loss` 绑在一起，这个绑定关系被写进 `.aux` 文件；`\eqref{eq:dpo-loss}` 在编译到那一行时，去 **上一次编译遗留下来的** `.aux` 文件里查这个名字对应的编号。第一次编译时，`.aux` 还是旧的（或者根本不存在），还没来得及写入这次新加的 `\label`，所以查不到，只能先显示成 "??" 并发出警告；必须再编译一次，让第二遍读到第一遍新写下的 `.aux`，才能正确替换成真正的编号。**这个"当前这一遍读不到当前这一遍自己产生的信息，只能读上一遍留下的快照"的机制，正是第 5 节 BibTeX 需要编译四遍的根本原因**——这里先建立直觉，第 5 节会把同一个机制讲得更完整。

**常见坑:**

| 现象 | 原因 |
|---|---|
| `\eqref{}` 显示成 "??" | 正常现象，见上——只编译了一遍，再编译一次即可，不是公式写错了 |
| 在行内 `$...$` 里塞 `\begin{align}` | `align` 是"独立数学模式专用"环境，报错，不能嵌进行内公式里 |
| 直接复制粘贴网页上的 "β" "σ" 这类 Unicode 符号，而不是敲 `\beta`/`\sigma` | `pdflatex` 对非 ASCII 字符处理能力有限，容易触发编码相关的报错或者显示异常；数学符号一律用 LaTeX 命令敲，不要复制粘贴 |
| Overfull \hbox 警告（本例第一遍真实出现过） | 某一行内容比可用宽度宽一点，LaTeX 硬着头皮排了，视觉上可能有一点点超出页边距；是排版警告不是错误，论文里如果出现太多，通常意味着这一段该重新组织断句，不影响本例的教学目的，此处不深入排查 |

**自测清单:**
- 你知道 `$...$` 和 `equation` 环境的用途区别、什么时候该用哪个吗？
- 你能独立把一个新公式（哪怕是自己论文里要用的）用 LaTeX 敲出来吗？
- 你能解释清楚为什么 `\eqref` 第一次编译经常显示成 "??"、为什么必须再编译一次吗？
- 你自己编译出了包含 DPO loss 公式的 PDF 了吗？

---

## 4. 图表插入 —— `graphicx` 插图 + `booktabs` 三线表

**为什么需要这个 / 不会有什么后果:**

ML 论文几乎必然包含实验结果曲线图和方法对比表格，这两样东西往往是评审除了摘要之外扫得最快的部分。排版糟糕的表格（Word 那种四面都是框线、每个格子都描边的风格）会显得业余；学术界的"三线表"（只有顶线、表头下分隔线、底线三条横线，完全没有竖线）是几乎所有顶会论文的标配。用错格式虽然不影响论文的科学内容，但会让评审下意识觉得"这个作者对学术写作规范不熟"，是一个不该在这种小事上失分的地方。

**环境要求:**

`graphicx`（插入图片）和 `booktabs`（三线表命令），两者编译时都被正常加载。另外需要一张真实存在的图片文件——这里用 matplotlib 现场生成一张占位曲线图，而不是用 TikZ 现画：科研图表几乎全部来自 matplotlib 之类的数据可视化工具（你自己以后跑实验画 loss 曲线也会用它），TikZ 更适合画流程图/示意图，这里选贴近你实际工作流的路径。

**一步步跟着做:**

第一步，生成占位图。脚本（[`_assets/07-latex-demos/03-figures-tables/make_placeholder_figure.py`](_assets/07-latex-demos/03-figures-tables/make_placeholder_figure.py)）：

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
steps = np.arange(0, 200)
loss = 2.5 * np.exp(-steps / 60) + 0.05 * rng.standard_normal(len(steps)) + 0.3

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(steps, loss, color="#1f77b4", linewidth=1.5)
ax.set_xlabel("Training step")
ax.set_ylabel("DPO loss")
ax.set_title("Placeholder training curve")
fig.tight_layout()
fig.savefig("placeholder-figure.png", dpi=200)
print("saved placeholder-figure.png")
```

用仓库 `.venv` 运行，真实输出：

```text
$ .venv/Scripts/python.exe make_placeholder_figure.py
saved placeholder-figure.png
```

产出 `placeholder-figure.png`，**41699 字节**——注意这是一条合成的衰减曲线加噪声，不是真实实验数据，这里只是为了验证 `\includegraphics` 机制本身。

第二步，写 LaTeX 文档插入这张图和一张三线表（[`_assets/07-latex-demos/03-figures-tables/figures-tables.tex`](_assets/07-latex-demos/03-figures-tables/figures-tables.tex)）：

```latex
\documentclass[11pt]{article}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}

\title{Figures and Tables in \LaTeX}
\author{Daily Toolkit Deep Dive --- Section 4}
\date{\today}

\begin{document}
\maketitle

\section{Inserting a figure with \texttt{graphicx}}

\begin{figure}[h]
  \centering
  \includegraphics[width=0.6\textwidth]{placeholder-figure.png}
  \caption{A placeholder training curve generated by matplotlib and
    embedded with \texttt{graphicx}.}
  \label{fig:placeholder}
\end{figure}

\section{A three-line table with \texttt{booktabs}}

\begin{table}[h]
  \centering
  \caption{Toy comparison table (numbers are illustrative, not from a
    real experiment).}
  \label{tab:toy-results}
  \begin{tabular}{lcc}
    \toprule
    Method & Win rate (\%) & KL to reference \\
    \midrule
    SFT only          & 50.0 & 0.00 \\
    DPO ($\beta=0.1$) & 68.4 & 3.21 \\
    DPO ($\beta=0.5$) & 61.2 & 1.05 \\
    \bottomrule
  \end{tabular}
\end{table}

\end{document}
```

编译（同样真实跑两遍，让浮动体编号和引用都稳定下来）：

```bash
pdflatex -enable-installer -interaction=nonstopmode -halt-on-error figures-tables.tex
pdflatex -enable-installer -interaction=nonstopmode -halt-on-error figures-tables.tex
```

第一遍日志里能明确看到图片被真实载入（不是占位符）：

```text
[1{.../pdftex.map} <./placeholder-figure.png>]
```

第一遍同样出现了 `Reference 'fig:placeholder' undefined` / `Reference 'tab:toy-results' undefined` 警告（原因和第 3 节完全一样），第二遍消失。最终稳定产物：`_assets/07-latex-demos/03-figures-tables/figures-tables.pdf`，2 页，**138798 字节**。打开应该看到：第一页是标题 + 一张居中的曲线图（下方带 "Figure 1: ..." 说明文字），第二页是一张只有三条横线、没有任何竖线的表格。

**背后发生了什么:**

- `figure`/`table` 是**浮动体（float）**环境——图片/表格不一定精确出现在你写 `\includegraphics` 那一行对应的位置，LaTeX 的排版算法会根据当前页面剩余空间自动决定它实际渲染在哪（可能被推到本页顶部/底部，甚至挤到下一页）。这是为了避免"一张图卡在段落正中间，把一行字切成两半"这种糟糕排版。`[h]`（here，"就放这里"）只是一个建议，不是强制指令——这也是为什么真实论文里经常看到图表出现在了"看起来不太对"的位置，这是排版算法的正常行为，不是你的错误。
- `\includegraphics[width=0.6\textwidth]{...}` 里的 `width=0.6\textwidth` 是**相对单位**（当前文本宽度的 60%），不是绝对像素值——这样即使论文换了模板、字号、页边距（比如从你自己的 `article` 模板换成 ICLR 官方模板），图片也会自动适配新的版心宽度，不会跑版。
- `booktabs` 的三条线粗细并不一样（`\toprule`/`\bottomrule` 比 `\midrule` 粗一点），这是"三线表"排版传统的一部分——用尽量少的横线、完全不用竖线，比起处处描边的表格更干净、噪声更少，`booktabs` 把这套约定封装成三个命令，不需要手动调线宽。
- 图/表的编号（"Figure 1"、"Table 1"）和 `\label`/`\ref` 引用机制与第 3 节的公式编号是**同一套底层机制**，同样依赖多遍编译才能稳定。

**常见坑:**

| 现象 | 原因 |
|---|---|
| `\includegraphics` 报 "File not found" | 图片文件路径写错，或者文件根本不存在——检查文件名拼写、相对路径是否相对于 `.tex` 文件本身 |
| 表格里到处是竖线和 `\hline` | 能编译通过、不算语法错误，但破坏了三线表规范，是"能跑但不符合学术写作惯例"的坑，评审看多了会觉得不专业 |
| 用了 `\toprule` 却没 `\usepackage{booktabs}` | 直接报 "Undefined control sequence" 硬错误——第 6 节会真实触发这个错误并展示报错长什么样 |
| 图片被排到了很远的地方（比如推到几页之后） | 浮动体算法的正常行为，不是 bug；如果论文快接近页数上限、图表位置乱飘，通常是内容太满导致的，不是靠死磕 `[h]` 能解决的 |

**自测清单:**
- 你知道 `figure`/`table` 是"浮动体"、为什么它们不一定出现在你写代码的那个精确位置吗？
- 你能独立写出一个三线表（不抄）吗？
- 你知道 `\includegraphics` 的 `width=0.6\textwidth` 是什么意思、为什么用相对单位而不是绝对像素吗？
- 你自己编译出了包含图 + 表的 PDF 了吗？log 里能找到图片被加载的那一行吗？

---

## 5. BibTeX / 参考文献管理 —— 为什么必须编译四遍

**为什么需要这个 / 不会有什么后果:**

一篇 ICLR 论文引用几十篇文献很常见。手动维护"正文里的引用编号"和"文末参考文献列表"的一致性——每加一篇引用就要手动重排后面所有编号、手动保证格式统一——几乎不可能不出错，这正是第 1 节提到的"Word 引用管理痛点"的具体化。BibTeX 把"文献数据库"和"论文正文"彻底解耦：你维护一个 `.bib` 文件（文献数据库，可以从 Google Scholar/Semantic Scholar 之类的网站直接导出条目，一次建好长期复用），正文里只需要 `\cite{key}`，编号、排序、格式全部自动生成。

**环境要求:**

`bibtex` 命令行工具（已确认可用，`MiKTeX-BibTeX 4.1`），一个 `.bib` 文件，`.tex` 文件里指定一个 `\bibliographystyle{}`（这里用最基础的 `plain`；真实投稿会用会议指定的样式文件，比如假设的 `iclr2026_conference.bst`，第 7 节会提到，这里先用最基础的样式把机制讲清楚）。

**一步步跟着做:**

文献数据库（[`_assets/07-latex-demos/04-bibliography/references.bib`](_assets/07-latex-demos/04-bibliography/references.bib)），两条真实文献条目，取的正好是你 `alignment-algorithms-deep-dive` 系列已经用过的两篇论文：

```bibtex
@article{rafailov2023dpo,
  title   = {Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author  = {Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Manning, Christopher D. and Ermon, Stefano and Finn, Chelsea},
  journal = {arXiv preprint arXiv:2305.18290},
  year    = {2023}
}

@inproceedings{ouyang2022instructgpt,
  title     = {Training language models to follow instructions with human feedback},
  author    = {Ouyang, Long and Wu, Jeff and Jiang, Xu and Almeida, Diogo and Wainwright, Carroll L. and Mishkin, Pamela and Zhang, Chong and Agarwal, Sandhini and Slama, Katarina and Ray, Alex and Schulman, John and Hilton, Jacob and Kelton, Fraser and Miller, Luke and Simens, Maddie and Askell, Amanda and Welinder, Peter and Christiano, Paul and Leike, Jan and Lowe, Ryan},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {27730--27744},
  year      = {2022}
}
```

`@article{rafailov2023dpo, ...}` 里，`article` 是文献类型（期刊/预印本用 `article`，会议论文用 `inproceedings`），`rafailov2023dpo` 是你自己起的 **key**（引用时用它，本身不出现在最终排版结果里，约定俗成用"第一作者姓 + 年份 + 关键词"方便记忆），花括号里是一串 `字段 = {值}`。

正文文件（[`_assets/07-latex-demos/04-bibliography/paper.tex`](_assets/07-latex-demos/04-bibliography/paper.tex)）：

```latex
\documentclass[11pt]{article}
\usepackage{amsmath}

\title{A Minimal BibTeX Example}
\author{Daily Toolkit Deep Dive --- Section 5}
\date{\today}

\begin{document}
\maketitle

Direct Preference Optimization~\cite{rafailov2023dpo} reparameterizes
the reward model implicitly inside the policy, building on the RLHF
pipeline used to train InstructGPT~\cite{ouyang2022instructgpt}. Both
papers are used throughout this repository's
\texttt{alignment-algorithms-deep-dive} series.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
```

**完整四步编译，每一步都真实跑过、真实截取了输出：**

```bash
xelatex -enable-installer -interaction=nonstopmode -halt-on-error paper.tex   # 第 1 遍
bibtex paper                                                                  # 第 2 遍（注意不带 .tex 后缀）
xelatex -enable-installer -interaction=nonstopmode -halt-on-error paper.tex   # 第 3 遍
xelatex -enable-installer -interaction=nonstopmode -halt-on-error paper.tex   # 第 4 遍
```

第 1 遍（`xelatex`）真实输出摘录——两条引用都报"undefined"：

```text
LaTeX Warning: Citation `rafailov2023dpo' on page 1 undefined on input line 12.
LaTeX Warning: Citation `ouyang2022instructgpt' on page 1 undefined on input line ...
LaTeX Warning: There were undefined references.
```

这一遍结束后，`paper.aux` 的真实内容是：

```text
\relax
\citation{rafailov2023dpo}
\citation{ouyang2022instructgpt}
\bibstyle{plain}
\bibdata{references}
\gdef \@abspage@last{1}
```

第 2 遍（`bibtex paper`）真实输出：

```text
This is BibTeX, Version 0.99d (MiKTeX 22.3)
The top-level auxiliary file: paper.aux
The style file: plain.bst
Database file #1: references.bib
```

这一遍产出了 `paper.bbl`，真实内容是已经排好版的参考文献列表（按 `plain` 样式的作者姓氏字母序排列，`ouyang` 在 `rafailov` 前面）：

```text
\begin{thebibliography}{1}

\bibitem{ouyang2022instructgpt}
Long Ouyang, Jeff Wu, ... and Ryan Lowe.
\newblock Training language models to follow instructions with human feedback.
\newblock In {\em Advances in Neural Information Processing Systems},
  volume~35, pages 27730--27744, 2022.

\bibitem{rafailov2023dpo}
Rafael Rafailov, Archit Sharma, ... and Chelsea Finn.
\newblock Direct preference optimization: Your language model is secretly a
  reward model.
\newblock {\em arXiv preprint arXiv:2305.18290}, 2023.

\end{thebibliography}
```

第 3 遍（`xelatex`，bibtex 跑完之后的第一次）——**关键教学点**：这一遍**仍然**警告引用 undefined，尽管 `.bbl` 已经生成好了：

```text
LaTeX Warning: Citation `rafailov2023dpo' on page 1 undefined on input line 12.
LaTeX Warning: Citation `ouyang2022instructgpt' on page 1 undefined on input line ...
LaTeX Warning: There were undefined references.
LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.
```

第 4 遍（`xelatex`，最后一遍）——终于干净，没有任何警告输出。最终稳定产物 `_assets/07-latex-demos/04-bibliography/paper.pdf`，**26330 字节**。用 `pypdf` 从最终 PDF 里提取文本做了真实核对，正文和文末列表编号完全对上：

```text
Direct Preference Optimization [2] reparameterizes the reward model
implicitly inside the policy, building on the RLHF pipeline used to train In-
structGPT [1]. ...

References
[1] Long Ouyang, Jeff Wu, ... 2022.
[2] Rafael Rafailov, Archit Sharma, ... 2023.
```

（编号是 [1]/[2] 而不是引用出现顺序，因为 `plain` 样式按作者姓氏字母序排列参考文献列表，再按这个顺序编号——这也是一个值得记住的细节：`\cite` 在正文里出现的先后顺序，不直接决定文末编号顺序，编号顺序由 `\bibliographystyle` 决定。）

**背后发生了什么（这是本节最核心的部分，必须讲清楚"为什么"，不能只说"要编译好几次"）:**

核心机制只有一条：**`.aux` 文件是"这一遍编译产生的、只有下一遍编译才能读到"的中间产物。** 每个工具（`xelatex`/`bibtex`）在运行时，只能读到**上一次**运行结束时留下的 `.aux` 快照，读不到"当前这一遍会写出什么"——这一点在第 3 节讲 `\eqref` 时已经埋过伏笔，BibTeX 只是把同一个机制在更长的依赖链上体现出来：

- **第 1 遍（xelatex）**：完全不知道 `references.bib` 里有什么内容，也不知道该怎么给引用编号——它只是老老实实把正文里每个 `\cite{key}` 记录成一行 `\citation{key}` 写进 `.aux`，把 `\bibliographystyle`/`\bibliography` 的信息记成 `\bibstyle{}`/`\bibdata{}`，因为查不到编号，只能在正文位置先摆一个占位符（对应显示为警告 + 最终 PDF 里的 "?"）。
- **第 2 遍（bibtex）**：这时候 `.aux` 已经有了上一遍留下的"引用了哪些 key" + "用哪个数据库/哪种样式"这些信息。**这里有个新手常见的误解要纠正：bibtex 从来不直接读 `.tex` 文件**，它只读 `.aux`。bibtex 拿着这份 key 列表去 `references.bib` 里逐条查完整文献信息，按 `plain` 样式排好版（作者字母序、"[数字]"编号），写成一个新文件 `paper.bbl`——这个文件本质上是"预先排好版的 `\bibitem{}` 列表"，用 LaTeX 原生命令写成，下一遍 `xelatex` 可以直接读取排版。
- **第 3 遍（xelatex，bibtex 之后第一次）**：这一遍能读到 `.bbl` 了，所以**参考文献列表本身**（文末那一块）已经能正确排出来；但正文里 `\cite{key}` 要显示的编号，需要一个 `\bibcite{key}{编号}` 映射——这个映射是 bibtex 处理 `.bbl` 排序结果之后，在**这一遍编译过程中才被写回** `.aux` 的，所以直到这一遍结束前，正文里的引用**依然找不到编号**，继续显示警告——这不是我编出来的理论，是本节真实复现出来的现象（见上面第 3 遍的真实输出）。
- **第 4 遍（xelatex，最后一次）**：这一遍开始时终于能读到上一遍新写好的、包含 `\bibcite{}` 映射的 `.aux`，正文里的编号才终于填上正确数字——干净、无警告。

所以"编译四次"不是玄学习惯或者"不知道具体几次就多编译几次保平安"的经验主义，而是**"每个工具只能读上一次运行留下的快照"这个底层机制决定的最短依赖链**。如果你后续只是改了论文正文的某个措辞、没有增删引用，通常重新跑一到两次 `xelatex` 就够了（不需要重新跑 `bibtex`）；但只要 `.bib` 内容或者引用关系变了（加了新引用、删了旧引用），这个四步序列要完整重新走一遍。

**常见坑:**

| 现象 | 原因 |
|---|---|
| 只编译一遍就去看 PDF，参考文献是空的、引用全是 "?" | 四步没走完，不是 LaTeX 坏了——回到上面的四步顺序重新走 |
| `bibtex paper.tex` 报错找不到文件 | `bibtex` 命令的参数是文件名**不带** `.tex` 后缀（`bibtex paper`），因为它要找的是 `paper.aux` 不是 `paper.tex` |
| 引用了 `.bib` 里不存在的 key，PDF 里永久显示 "?" | 这个"?"不会随着继续编译消失，跟"编译次数不够"的暂时性 "?" 要能区分开——第 6 节会真实复现这种情况并展示怎么从报错定位 |
| `.bib` 条目 key 拼写/大小写和 `\cite{}` 里不完全一致 | bibtex 精确匹配 key 字符串，大小写不同也算不匹配 |

**自测清单:**
- 你能独立说出 `xelatex → bibtex → xelatex → xelatex` 这四步各自在做什么、为什么必须是这个顺序吗？
- 你知道 `bibtex` 读的是 `.aux` 文件而不是 `.tex` 本身吗？
- 你能解释清楚为什么"只编译一遍"参考文献会是空的/问号吗？
- 你自己跑完这四步、拿到了一份正文编号和文末列表完全对得上的 PDF 了吗（比如用文本提取工具或者肉眼打开核对过）？

---

## 6. 常见编译报错排查 —— 学会看报错定位问题，而不是死记几种错误

**为什么需要这个 / 不会有什么后果:**

真实写论文时，绝大多数"卡住"的时刻不是不会某个语法，而是编译报错但看不懂报错信息在说什么，于是从头到尾瞎改却找不到问题根源。这一节的目标不是让你记住三种错误的解法——是学会"看报错定位问题"这个通用能力。这个能力比记住任何单个错误的解法都重要，因为你以后写论文一定会遇到本节没覆盖过的新错误，到时候能不能自己看懂 log、而不是把整段代码删了重写，是效率差异的关键。

**环境要求:** 复用前面几节已确认可用的环境，这一节不需要新宏包——三个错误都是刻意制造的，用来触发真实报错。

**一步步跟着做（三个真实错误，每一个都真实触发过，报错文本是编译器原始输出，不是转述）:**

### 错误 1：公式里少一个 `$`

文件（[`_assets/07-latex-demos/05-common-errors/err1-missing-dollar.tex`](_assets/07-latex-demos/05-common-errors/err1-missing-dollar.tex)）：

```latex
\documentclass[11pt]{article}
\usepackage{amsmath}
\begin{document}

The DPO loss uses a temperature $\beta that scales the log-ratio
margin before it is passed through the sigmoid function.

This is a second paragraph, added only so that \LaTeX{} hits a
paragraph break while it is still ``inside'' math mode from the
unclosed dollar sign above.

\end{document}
```

（注意：`$\beta` 开了行内数学模式，但后面一直没有配对的 `$` 把它关掉。）

编译命令与真实报错输出：

```bash
pdflatex -enable-installer -interaction=nonstopmode -halt-on-error err1-missing-dollar.tex
```

```text
! Missing $ inserted.
<inserted text>
                $
l.7

!  ==> Fatal error occurred, no output PDF file produced!
Transcript written on err1-missing-dollar.log.
```

**退出码 1，没有产出 PDF。**

怎么读这段报错：`!` 开头的行是错误类型本身——"Missing $ inserted" 意思是 TeX 发现自己"卡在数学模式里出不来了"，只好在它认为合适的位置**自己插入**一个 `$` 来强行了结。`l.7` 指出问题是在处理到第 7 行（也就是两段之间那个空行，触发了段落结束 `\par`）时才**被发现**的——注意这不等于"错误就是从第 7 行开始的"，真正漏掉 `$` 的位置在第 5 行。这是新手最容易被误导的地方：**报错行号是"TeX 撑不下去、终于报错的位置"，不一定是"问题真正开始的位置"**，尤其是缺 `$` 这类错误，TeX 会一直尝试"凑成合法数学模式"直到遇到 `\par`（段落结束）才放弃，所以报错行经常和真正漏掉符号的那一行隔着一段距离。定位方法：从报错行往上找，找最近的一个孤零零、没有配对的 `$`。

### 错误 2：引用了不存在的 `\cite`

文件（[`_assets/07-latex-demos/05-common-errors/err2-undefined-cite.tex`](_assets/07-latex-demos/05-common-errors/err2-undefined-cite.tex)）：

```latex
\documentclass[11pt]{article}
\begin{document}

Direct Preference Optimization was proposed by \cite{rafailov2023dpo}.
A follow-up idea, reward-free contrastive preference optimization, is
described in \cite{this_key_does_not_exist_2099}.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
```

（`rafailov2023dpo` 在 `references.bib` 里真实存在，`this_key_does_not_exist_2099` 是故意编的、数据库里没有这个 key。）

按第 5 节的四步序列编译，两个工具在两个不同层面都给出了报错信息：

`xelatex`/`pdflatex` 层面（**只是警告，编译依然"成功"、依然产出 PDF**）：

```text
LaTeX Warning: Citation `rafailov2023dpo' on page 1 undefined on input line 4.
LaTeX Warning: Citation `this_key_does_not_exist_2099' on page 1 undefined on input line ...
```

`bibtex` 层面（**更精确，直接点名是哪个 key 找不到**）：

```text
Warning--I didn't find a database entry for "this_key_does_not_exist_2099"
(There was 1 warning)
```

按完整四步走完之后，用 `pypdf` 提取最终 PDF 文本，真实结果是：

```text
Direct Preference Optimization was proposed by [1]. A follow-up idea,
reward-free contrastive preference optimization, is described in [?].

References
[1] Rafael Rafailov, Archit Sharma, ... 2023.
```

**这是本节最重要的一个区分点：** 真实存在的 key（`rafailov2023dpo`）最终正确解析成了 `[1]`；编造的 key（`this_key_does_not_exist_2099`）无论再编译多少遍，**永远**停留在 `[?]`——因为 `bibtex` 从一开始就没能在 `references.bib` 里找到匹配条目，从未给它生成 `\bibcite{}` 映射，后面编译多少遍都没用。这跟第 3、5 两节讲的"第一遍编译暂时性的 `??`（多编译几次会消失）"是两种性质完全不同的问题，必须能区分：**暂时性的未定义**是"还没编译够遍数"，**永久性的未定义**是"引用的 key 本身就是错的，需要去检查拼写或者去 `.bib` 里补上这条文献"。

### 错误 3：用了没导入的宏包命令

文件（[`_assets/07-latex-demos/05-common-errors/err3-missing-package.tex`](_assets/07-latex-demos/05-common-errors/err3-missing-package.tex)）：

```latex
\documentclass[11pt]{article}
\begin{document}

\begin{tabular}{lcc}
  \toprule
  Method & Accuracy \\
  \midrule
  Baseline & 50.0 \\
  \bottomrule
\end{tabular}

\end{document}
```

（用了 `\toprule`/`\midrule`/`\bottomrule`，但导言区没有 `\usepackage{booktabs}`。）

真实报错输出：

```text
! Undefined control sequence.
l.5   \toprule

!  ==> Fatal error occurred, no output PDF file produced!
```

**退出码 1，没有产出 PDF。**

怎么读："Undefined control sequence" 意思是 TeX 遇到了一个它完全不认识的命令名，`l.5 \toprule` 精确指出了是哪一行、哪个命令。这类报错几乎总是两种可能之一：(a) 命令名本身打错字了（比如把 `\includegraphics` 敲成 `\includegraphcis`）；(b) 命令来自某个宏包但忘了 `\usepackage` 导入——本例是情况 (b)。区分方法：先检查有没有拼写错误，排除之后，去搜"LaTeX `\toprule` package"确认它属于哪个宏包，补上对应的 `\usepackage{}`。

**背后发生了什么:**

- LaTeX 的报错机制本质：TeX 是一个"边读边执行"的宏展开引擎，不像很多现代编程语言的编译器那样有独立的"先做完整语法分析、再执行"的阶段。它是"跑到哪里、什么时候真正发现状态不对，就在哪里报错"——这就是为什么报错行号有时候和"人直觉认为的错误位置"对不上（错误 1 就是活生生的例子：漏掉的 `$` 在第 5 行，报错却在第 7 行）。
- `-halt-on-error` 让编译器一遇到第一个致命错误就停，本节三个错误的 log 都很干净、只有一个核心问题。**不加这个参数**（默认 `nonstopmode` 行为下继续跑），TeX 会尝试"猜测性恢复"继续往下编，经常在后面级联出一大串看起来吓人、实际上都是同一个根因的连锁报错——这是一个值得记住的排错习惯：**看到一大串报错先别慌，通常只需要处理 log 里第一个 `!` 开头的错误，后面一串很可能是它的连锁反应。**
- **退出码（exit code）是判断"这次编译到底成不成功"最可靠的信号。** 错误 1 和错误 3 都是退出码 `1`（致命错误，没有产出 PDF）；错误 2 是退出码 `0`（只有警告，PDF 正常产出，但内容里悄悄藏着一个 `[?]`）。以后如果要写脚本自动化编译论文（比如给论文仓库配一个 CI，每次 push 自动编译检查能不能过），判断"构建是否成功"应该检查退出码 + "PDF 文件是否真的生成了"，而不是简单粗暴地在 log 里搜 "error" 这个词——因为 log 里到处会出现这个词的正常提及（宏包名字、说明文字里都可能带），会有大量误报。

**常见坑:**

| 现象 | 原因 |
|---|---|
| 看到一堆报错就慌乱地大改一通 | 正确做法是从 log 文件里**第一个** `!` 开始看，先解决最早的那个，后面一大串报错很可能是它的连锁反应 |
| 把 warning 和 fatal error 混为一谈 | Fatal error 会导致编译失败（无 PDF 产出/退出码非 0）；warning 不会（编译仍"成功"，但内容可能不是你想要的，比如引用显示成 `[?]`）——两者都要重视，但处理优先级不同 |
| 只看终端最后几行 | 终端经常被大量宏包加载信息刷屏，真正的错误信息可能滚动在中间被错过；更可靠的做法是直接在 `.log` 文件里搜索 `! `（感叹号 + 空格） |
| 把"暂时性 ??"和"永久性 ??"当成同一回事 | 见错误 2 的详细区分；遇到 `[?]`，先确认是不是四步编译流程没走完，走完之后还是 `[?]`，才去检查 `.bib` 里的 key |

**自测清单:**
- 拿到一段没见过的 LaTeX 报错，你知道该看 `.log` 文件的哪一部分、怎么找到最早的那个 `!` 吗？
- 你能说出 "warning" 和 "fatal error" 在编译结果上的本质区别吗（一个继续产出 PDF、一个不产出）？
- 你能区分"citation undefined"警告什么时候是暂时的（多编译几次会消失）、什么时候是永久的（key 本身错了）吗？
- 你自己真实触发过这三种报错、并且看懂了 log 在说什么、知道怎么修吗？

---

## 7. 会议模板实战 —— 用 ICLR 官方模板投稿是怎么回事

**为什么需要这个 / 不会有什么后果:**

你的最终目标是投 ICLR，顶会投稿几乎不允许"自由排版"——页数限制、页边距、字号、双盲评审匿名要求，这些规则会议方通过官方发布的 LaTeX 宏包强制实现。不用官方模板、或者用了过时版本的模板，轻则格式检查不通过被要求重新提交，重则可能因为不满足格式要求被直接拒稿（是的，这是真实存在的、和论文内容质量完全无关的拒稿原因，纯粹因为格式没按要求来）。这一节要讲清楚"会议模板大概是怎么回事"，为你两个月后真正下载 ICLR 官方模板那一刻做准备。

**环境要求:** 无新工具需求。这一节**不实际下载 ICLR 官方模板文件**，原因：(1) 避免这篇教程绑定一个会很快过时的具体版本——ICLR 模板年年可能有细节调整，此刻下载的版本到你真正投稿时很可能已经不是最新的；(2) 避免不必要的外部依赖和版权问题。这一节讲的是"会议模板机制的通用说明"，不是"ICLR 2026 模板的具体内容"，等你真正开始写论文时，请直接去 ICLR 官网下载当年最新版本。

**一步步跟着做（这一节是通用流程说明，不是可编译的真实操作，如实标注）:**

真实投稿时的一般流程：

1. 去会议官网找 "Call for Papers" 或 "Author Instructions" 页面，下载官方模板压缩包（通常包含 `.sty`/`.bst` 宏包文件、一个示例 `.tex`、一份说明 PDF 或 README）。
2. 解压到你的论文项目目录，**用它提供的示例 `.tex` 作为起点**，而不是自己从 `\documentclass{article}` 从零写——示例文件里已经预置了正确的 `\documentclass` 参数、`\usepackage{会议专属宏包}`、标题/作者/摘要的标准格式宏,以及双盲匿名相关的开关。
3. 把你自己的内容逐段填进示例文件对应的位置。参考文献样式（`\bibliographystyle{}`）通常也是模板指定好的，不要自己改成别的样式。
4. 提交前重新检查一遍模板自带的格式要求清单——不少会议（包括 ICLR）要求论文末尾附一个格式化的 "checklist"，逐项确认你的论文满足可复现性、伦理声明等方面的要求。

一个典型的会议模板导言区大概长什么样——**下面这段是示意写法，不是从任何官网复制来的真实文件内容**，字段名和真实文件几乎肯定对不上，只是让你对"用模板和从零写有什么不同"有具体印象：

```latex
\documentclass{article}
\usepackage{iclr20XX_conference,times}   % 会议提供的宏包，不同年份文件名不同
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{natbib}
% \iclrfinalcopy   % 定稿阶段取消匿名，投稿阶段这一行通常要保持注释，维持双盲匿名
```

**背后发生了什么:**

- 会议提供的 `.sty` 宏包本质上就是"提前写好的一堆 `\newcommand`/`\renewcommand` 和排版参数设置"，它通过 `\usepackage{}` 被加载后，会"接管"页边距、字体、章节标题样式等原本由 `article` 类决定的默认值，还常常带一些会议专属命令（比如控制作者信息匿名/显示的开关、控制 "under review" 水印的开关）。
- **用模板和从零写的本质区别：** 从零写 `\documentclass{article}` 只能保证"编译出一份格式自洽的 PDF"，但版心宽度、页边距、字号这些具体数值是 `article` 类的通用默认值，几乎肯定不符合任何特定会议的强制要求。用官方模板则是把这些数值全部锁定成会议要求的值——你只需要专注内容，这些排版参数不需要、也不被允许你自己去调。这也是为什么第 2 节讲的"最小可编译文档"骨架和真实投稿论文的骨架，本质上是同一个骨架，只是导言区的宏包和参数被会议模板替换掉了。
- 页数限制的落地方式因会议而异：多数情况下是"文档说明里写明不能超过 N 页"，超没超需要你自己编译完看总页数去核对；少数模板会有更强制的技术检测手段。不管是哪种，"编译完看一下总页数是否在限制内"都是投稿前必须做的最后一步核对，不能想当然。

**常见坑:**

| 现象 | 后果 |
|---|---|
| 用了旧版本的会议模板（比如去年下载的，今年直接拿来用） | 格式细节年年可能有微调（页边距、checklist 要求、匿名规则），旧版本可能在细节上不满足今年的要求，投稿前务必重新去官网确认下载的是当年最新版本 |
| 手动修改模板里"不该碰"的排版参数（比如自己偷偷改页边距想多塞点内容） | 大部分会议明确禁止修改这些参数，一旦被发现（格式检查经常是自动化脚本，很容易查出来），可能被要求重新提交甚至直接拒稿 |
| 投稿阶段忘记保持匿名（作者信息、致谢、GitHub 链接暴露身份） | 双盲评审最常见的技术性翻车原因之一——模板通常提供了对应开关（比如上面示意的 `\iclrfinalcopy`），但开关用没用对是作者自己的责任，模板不会替你检查 |
| 从网上搜到"某个人分享的模板压缩包"，而不是从官网直接下载 | 存在版本过时、内容被篡改的风险；官方渠道是唯一可信来源 |

**自测清单:**
- 你知道为什么顶会要求用官方 LaTeX 模板投稿、不能自己随便排版吗？
- 你能说清楚"用模板写"和"从零用 `article` 写"两者在最终产出上的本质区别吗？
- 你知道投稿前应该去哪里确认自己用的是"当年最新"模板吗？
- 你知道双盲评审对模板使用有什么额外要求（匿名开关）、以及用错的后果吗？

---

## 附:本章全部真实编译产物一览

以下文件全部在本机真实用 `xelatex`/`pdflatex`/`bibtex` 编译产生，不是手写伪造，均可在 [`_assets/07-latex-demos/`](_assets/07-latex-demos/) 下找到并自行用 `-enable-installer -interaction=nonstopmode` 重新编译核对：

| 小节 | 源文件 | 最终 PDF | 大小 | 说明 |
|---|---|---|---|---|
| 2 | `01-minimal-document/minimal.tex` | `minimal.pdf` | 42118 字节 | 1 页，最小骨架 |
| 3 | `02-math-formulas/dpo-formula.tex` | `dpo-formula.pdf` | 147858 字节 | 2 页，含 DPO loss 公式，2 遍编译后稳定 |
| 4 | `03-figures-tables/figures-tables.tex` | `figures-tables.pdf` | 138798 字节 | 2 页，含 matplotlib 生成的图 + 三线表，2 遍编译后稳定 |
| 5 | `04-bibliography/paper.tex` | `paper.pdf` | 26330 字节 | 1 页，完整 4 遍编译（xelatex→bibtex→xelatex→xelatex），引用编号已核对正确 |
| 6 | `05-common-errors/err1-missing-dollar.tex` | *(无，致命错误，退出码 1)* | — | "Missing $ inserted" |
| 6 | `05-common-errors/err2-undefined-cite.tex` | `err2-undefined-cite.pdf` | 53299 字节 | 编译"成功"但含 1 个永久 `[?]`，用于对比暂时性/永久性未定义引用 |
| 6 | `05-common-errors/err3-missing-package.tex` | *(无，致命错误，退出码 1)* | — | "Undefined control sequence" |

错误 1、3 特意"没有产出 PDF"——这本身就是这两节要展示的教学内容（致命错误 vs 警告的区别），不是遗漏。

---

*创建:2026-07-23*
