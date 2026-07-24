# 论文投稿与评审全流程实操 —— 路线图与进度表

> 目标:不是"讲道理"，是"手把手教会你怎么把一篇论文从"写完了"送到"被接收"这条真实流程走完一遍"——venue 怎么选、双盲匿名化怎么做（含一个真的能跑的检查脚本）、rebuttal 这件事怎么组织、被拒了怎么办、camera-ready 怎么收尾，5 个分类，由浅入深。
> 背景：2026-07-25，用户转达导师（Weikai Lin）两轮反馈，第一轮要求补充科研写作/学术绘图/论文投递相关的手把手教程，第二轮追加点名要求覆盖 rebuttal（重点强调）、deal with reject、camera ready 等具体环节。这条系列是响应第二轮反馈新增的 5 个"论文发表系列"之一，专门覆盖"怎么走完投稿评审全流程"这一段——完整设计见 [`docs/superpowers/specs/2026-07-25-paper-publication-series-design.md`](../../docs/superpowers/specs/2026-07-25-paper-publication-series-design.md) 的"系列 3：投稿与评审"一节。

---

## 这条系列是什么、不是什么

**这不是"深挖系列"（numpy/torch/dsa 那种），也不是纯"操作系列"（daily-toolkit 那种）——它介于两者之间。** 深挖系列回答"这个函数/机制是什么、为什么这样设计"；纯操作系列回答"这个工具具体怎么点、敲什么命令"。这条系列要回答的问题更接近后者的**形式**（一步步教你做），但内容本身经常没有唯一"正确命令"——"这个 venue 该不该投""这条审稿意见是审稿人没看懂还是我真有硬伤"，这些判断没有 `command --flag` 式的标准答案。处理方式：**能给出操作步骤的地方（比如怎么用检查脚本、OpenReview 界面在哪点）严格按操作类模板讲清楚"输入什么→应该看到什么"；需要判断力的地方（venue fit、reject 之后怎么办），把"怎么判断"本身拆成可操作的检查清单和真实数据支撑的决策依据，不用"看情况""因人而异"这种空话搪塞。**

**和仓库里其他"论文发表系列"的边界**（分工写清楚，避免互相重复）：

- **[`daily-toolkit-deep-dive/07-latex-paper-writing.md`](../daily-toolkit-deep-dive/07-latex-paper-writing.md)** 已经讲过 LaTeX 怎么编译、会议官方模板机制是怎么回事（`\usepackage{会议宏包}` 怎么接管排版参数、`\iclrfinalcopy` 这类匿名开关的存在）——**这条系列不重复这部分**，02 号文件涉及匿名化时会直接引用那篇第 7 节，聚焦那边完全没讲的"匿名化具体要检查哪些内容、用什么工具查、OpenReview 投稿系统本身怎么操作"。
- **`research-writing-deep-dive/`**（另一位协作者同步建设中的姊妹系列，本文写作时其编号尚未最终定稿，交叉引用只到目录层级，不锁定具体文件号）负责"rebuttal 该**怎么写**"（措辞、逐条回应的文字技巧、字数限制内怎么说服人）——**这条系列的 03 号文件不重复这部分**，讲的是 rebuttal 这件事**怎么组织**（时间线、协作分工、word limit 怎么在多条意见之间分配）。两者是"写什么"和"怎么把这件事办成"的关系，不是同一件事讲两遍。
- **`academic-presentation-deep-dive/`、`research-release-deep-dive/`**（同批新建的姊妹系列）覆盖的是投稿评审结束**之后**的事——会议现场怎么讲、代码怎么发布、怎么和社区互动。这条系列的范围明确停在"论文被接收，camera-ready 交上去"这一刻，之后的事不属于这里。

**这条系列不覆盖什么（明确的范围边界，避免写着写着跑偏）**：不逐一覆盖 ICML/NeurIPS/ICLR 每年具体的页数/格式要求年年会变，这里只讲判断框架和检查清单该怎么用，具体数字请读者自己查当年 CFP（Call for Papers）；不代写用户真实论文的具体内容；不做真实的 OpenReview 端到端网络提交验证（没有一篇真的要拿去投稿的论文可以用来练手）。

---

## 每个知识点的固定讲解结构（6 步，复用 daily-toolkit-deep-dive 已验证的操作类模板）

和 [daily-toolkit-deep-dive/00-roadmap.md](../daily-toolkit-deep-dive/00-roadmap.md) 完全一致的 6 步，这里针对本系列"部分内容是判断力而不是命令"的特点做一点说明：

1. **为什么需要这个 / 不会有什么后果**——从零建立动机，不假设读者已经理解这件事解决什么问题。吸取 rhcsa-bash-deep-dive 的教训（该系列因为跳过"是什么"直接讲危险操作对比，导致进程/权限模型这些基础概念从未被真正建立），这一步必须先把概念本身讲清楚。
2. **环境要求**——操作类内容指"需要装什么工具/账号"；判断类内容指"你动手判断之前，手上应该已经准备好哪些信息"（比如：判断 venue fit 之前，你需要先知道自己论文的成熟度和"这个领域的人会去哪读论文"）。
3. **一步步跟着做**——操作类内容是真实命令+真实输出；判断类内容是一套可以真的照着走一遍的判断步骤（给决策依据，不是"自己感觉"）。能用代码验证的地方（本系列最重要的例子：02 号文件的匿名化检查脚本）一律在 `.venv` 里真实跑过，贴真实输出。
4. **背后发生了什么**——操作类讲机制；判断类讲"这条判断依据背后是什么更根本的原因"（比如影响因子迷思背后是 ML 社区独特的"会议优先"文化史）。
5. **常见坑**——真实会踩的坑，风格上"被拒了怎么办"这类场景化知识点采用 [rhcsa-bash-deep-dive/10-advanced-interview-depth.md](../rhcsa-bash-deep-dive/10-advanced-interview-depth.md) 首创的"故障排查语序"（现象→判断→行动→原因，不是"面试官/候选人"对话体）。
6. **自测清单**——3-5 条读者自己能校验的标准。

---

## 验证环境与真实性声明

不同类型的内容验证颗粒度不同，分开说清楚（延续 daily-toolkit-deep-dive 的诚实标注纪律）：

- **匿名化检查脚本**（02 号文件核心交付物）：真实代码，[`_assets/02-anon-checker/check_anonymity.py`](_assets/02-anon-checker/check_anonymity.py)，在 `e:\Workspace\dummy\.venv\Scripts\python.exe`（Python 3.13.9）里真实跑通——包括对着一份真实构造的、含 8 类身份泄露信号的 [`leaky_example.tex`](_assets/02-anon-checker/leaky_example.tex) 和一份真实通过匿名检查的 [`clean_example.tex`](_assets/02-anon-checker/clean_example.tex) 分别运行，命令行调用和内置自测两条路径都真实执行过，退出码、检测条数、每一行的检测结果全部是这次真实运行产生的，不是手写的示意输出。撰写过程中脚本本身有一处真实 bug 被这次验证现场揪出来并修复，02 号文件会如实记录这个过程（不是编出来的教学案例）。
- **Rebuttal 字数预算分配小工具**（03 号文件）：真实代码，在 `.venv` 里真实跑通，用真实的 `round()` 四舍五入结果，不是手算凑出来的"整数刚好对上"。
- **OpenReview / 会议投稿系统的图形界面操作**（02 号文件部分内容）：无法像代码一样自动化验证——已通过 WebSearch 核实 ICLR/NeurIPS 官方 Author Guide 当前公开的真实流程（截至 2026-07 的公开信息），但没有一篇真实要投稿的论文可以用来做端到端提交验证，如实标注为"已核实当前公开流程，但未做端到端自动化验证"，参照 [daily-toolkit-deep-dive/03-ssh-and-remote-servers.md](../daily-toolkit-deep-dive/03-ssh-and-remote-servers.md)"没有真实远程服务器，语法已验证但连接效果需要读者自己验证"的诚实标注方式。
- **真实数据来源**：ICLR 2024-2025 rebuttal 效果统计（19000+ 篇论文、74000+ 条评审的大规模研究）、JMLR vs IEEE TNNLS 影响因子与社区地位对比、2025-2026 年真实发生的 OpenReview 匿名泄露事故（2025-11-27）、NeurIPS/ICML 官方 paper checklist 与 supplementary material 规范、双盲匿名化真实违规案例与处理方式，全部来自这次真实执行的 WebSearch 调研，每个数字标注来源，不臆造。完整来源列表见各文件"参考来源"小节。
- **真实项目素材**：案例场景取材 [`research/world-model-imagination-controller/`](../../research/world-model-imagination-controller/)（用户真实在研、准备投稿 ICLR 的项目——测试时想象预算自适应分配控制器），已实地读取 [`01-meeting-briefing.md`](../../research/world-model-imagination-controller/01-meeting-briefing.md) 和 [`04-sharpened-recommendation.md`](../../research/world-model-imagination-controller/04-sharpened-recommendation.md) 了解真实技术内容和真实处境（AAAI 已被排除、按 ICLR 时间线倒推约 2 个月、pilot 阶段还在合成小环境里）。**06 号 capstone 是虚构的模拟时间线，不是这篇论文真实的投稿结果**——这篇论文在本系列写作时还没有真实投出、更没有真实的审稿决定，capstone 只是借用其真实技术设定和真实自我识别的风险点（比如"pilot 还没有在真实高维 world model 上验证过"）设计一条有真实技术质感的虚构故事线，文件开头会有醒目声明。

---

## 进度表

| # | 分类 | 文件 | 状态 |
|---|------|------|------|
| 01 | Venue 选择与 fit 度判断 | [01-venue-selection-and-fit.md](01-venue-selection-and-fit.md) | 完成（5 节均按 6 步模板；影响因子迷思用 JMLR vs IEEE TNNLS 真实数据展开，track 选择用真实的 full/workshop/journal track 三方对比，红旗信号含真实的 predatory conference 检查清单，收尾用 world-model-imagination-controller 项目真实的 venue 判断过程——AAAI 已被排除、锁定 ICLR——作为案例） |
| 02 | 双盲匿名化与投稿系统操作 | [02-double-blind-anonymization-and-submission-systems.md](02-double-blind-anonymization-and-submission-systems.md) | 完成（5 节均按 6 步模板；核心交付物是真实可用、真实跑通的匿名化检查脚本 [`check_anonymity.py`](_assets/02-anon-checker/check_anonymity.py)，含撰写过程中真实发现并修复的一处误判 bug；OpenReview 提交流程按官方 2026 Author Guide 真实核实，图形界面操作如实标注未做端到端验证；引用 2025-11-27 真实发生的 OpenReview 匿名信息泄露事故作为"为什么要认真对待双盲"的真实反面教材） |
| 03 | Rebuttal 时间与协作管理 | [03-rebuttal-time-and-collaboration-management.md](03-rebuttal-time-and-collaboration-management.md) | 完成（5 节均按 6 步模板；开篇用 ICLR 2024-2025 真实统计数据——分数提升论文录用率 55.7%-57.6% vs 不提升只有 7.8%-12.4%——建立动机；含真实跑通的 rebuttal 字数预算分配小工具；和 research-writing-deep-dive 的分工边界在文件开头明确声明） |
| 04 | Deal with Reject 深挖 | [04-deal-with-reject.md](04-deal-with-reject.md) | 完成（5 节均按 6 步模板；"审稿人真没看懂 vs 真有硬伤"和"resubmit 策略"两节采用 rhcsa-bash-deep-dive 首创的"故障排查语序"叙事，不是面试对话体；红旗信号数据来自真实调研） |
| 05 | Camera-ready 与 Supplementary 组织 | [05-camera-ready-and-supplementary.md](05-camera-ready-and-supplementary.md) | 完成（5 节均按 6 步模板；checklist 一节直接对照 NeurIPS 官方 paper checklist 真实结构和真实文档顺序要求；和 02 号文件的匿名化脚本形成"投稿时匿名→camera-ready 时反匿名"的完整闭环） |
| 06 | 教程外追加：模拟投稿全流程复盘（capstone） | [06-mock-submission-journey-capstone.md](06-mock-submission-journey-capstone.md) | 完成（叙事体，开篇有醒目声明"这是虚构模拟，不是真实投稿结果"；"首投被拒→冷静拆解审稿意见→调整方向重投→录用"完整线索，串联 01-05 全部 5 个分类的方法论，审稿意见基于 world-model-imagination-controller 项目自己在 `04-sharpened-recommendation.md` 里已经诚实列出的真实风险点设计，不是凭空编的批评） |

---

## 完成小结

5 个分类 + 1 篇 capstone 全部完成。这条系列和纯操作类系列（daily-toolkit）最大的不同：约一半内容天然是判断力而非命令，处理方式是把"怎么判断"本身拆解成可核验的检查清单和真实数据支撑的依据，不用"看情况"含糊带过；另一半内容（匿名化检查、rebuttal 字数分配）和纯操作类系列一样，能跑代码的地方严格做到真实跑通、真实断言，不写伪代码。

撰写过程中一处最值得记住的真实发现：02 号文件的匿名化检查脚本第一版用"内容是否精确等于某个占位符字符串"的方式判断 `\author{}` 块是否已经匿名化，结果对着自己构造的"干净"样例（`\author{Anonymous Author(s) \\ Affiliation withheld for double-blind review}`——这其实是很多真实会议模板的标准写法，比如 ICLR 官方模板匿名分支常见的 "Anonymous authors\\Paper under double-blind review"）现场跑出了假阳性：脚本把这段本该被认可的标准匿名占位符误判成"还没脱敏"。原因是最初的实现要求整段内容逐字命中一个很短的白名单，而真实世界的匿名占位符经常是"核心短语 + 补充说明"这种拼接写法。修复方式是把"整段精确匹配"换成"从内容里逐个抠掉已知的匿名占位符词块，抠完看还剩不剩实质内容"——这个修复过程本身就是一个很好的教学案例：**写一个"检测不匿名"的工具，工具自己先犯了一次"把已经匿名的内容当成没匿名"的错误**，02 号文件会如实记录这整个过程，而不是只展示修好之后的最终版本。另外，脚本正常运行时还带出了一个良性的"意外命中"：真实泄露样例里的 GitHub 链接 `github.com/epeng-riverside-lab/...`，因为 owner 名字本身包含 "lab" 这个词，被机构关键词检测顺带命中了一次——这不是 bug，是同一处泄露被两条独立规则各自发现了一次，02 号文件"常见坑"一节会讨论这种"多条规则命中同一行"的现象，以及反过来的、真实测出的假阳性例子（"we evaluate our controller in a simulated **laboratory** environment"这种无害的日常用词，被"机构关键词"规则错误命中）。

另一处环境层面的真实发现（用 `_verify_md.py` 验证 02/03 号文件的 Python 代码块时撞到）：直接跑 `python _verify_md.py xxx.md`（不带任何环境变量）会在后台读取子进程输出的线程里抛出 `UnicodeDecodeError: 'gbk' codec can't decode byte ...`——`_verify_md.py` 内部已经给**子进程**的环境变量设置了 `PYTHONUTF8=1`（脚本第 16 行 `env = {**os.environ, "PYTHONUTF8": "1"}`），子进程因此用 UTF-8 输出包含中文的 `print()` 内容（本系列两个代码块的输出里都有中文，比如"字符"、"投稿阶段这类信息不能出现在正文或作者块里"），但**外层**负责读取这份子进程输出的 `_verify_md.py` 自己这个 Python 进程，如果没有同样处于 UTF-8 模式，会按这台机器默认的 GBK locale 去解码收到的 UTF-8 字节，从而在多字节序列处崩溃。真实验证：这个 `UnicodeDecodeError` 只发生在后台读取线程里、被 `threading` 模块的默认异常钩子打印成一段吓人的 traceback，但**不影响最终判定**——`result.returncode` 的获取和 `[ok]`/`[FAIL]` 的判断逻辑不依赖这次解码是否成功，两个文件最终都正确报出 `ALL 1 blocks passed independently.`；给外层调用本身也加上 `PYTHONUTF8=1`（即 `PYTHONUTF8=1 python _verify_md.py xxx.md`）之后，traceback 完全消失，输出干净。这条经验值得记录：**`_verify_md.py` 内部已经替子进程设置了 UTF-8 环境，但调用它本身这一层同样需要处于 UTF-8 模式**，两层缺一不可，这是这次撰写过程中新发现的、这个复用自 dsa-deep-dive 的验证脚本在 Windows 中文 locale 环境下此前没有被文档记录过的细节。

---

*创建：2026-07-25*
