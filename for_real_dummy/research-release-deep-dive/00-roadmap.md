# 成果发布与社区实操 —— 路线图与进度表

> 目标:论文写完了、投出去了(甚至还没到那一步),代码也能跑了——这条系列回答再往后一步的问题:**代码仓库作为一个"成品"该长什么样、模型/数据怎么发布到 HuggingFace 让人愿意用、论文公开之后怎么和社区打交道**。3 个分类,由浅入深。
> 背景:2026-07-25,用户转达导师(与 [[worldmodel-imagination-controller-research]] 同一位导师 Weikai Lin)两轮反馈,第二轮里明确点名 "release code"“hugface release”“interact with community”这几项——这条系列直接响应这几项。详细背景见 [`docs/superpowers/specs/2026-07-25-paper-publication-series-design.md`](../../docs/superpowers/specs/2026-07-25-paper-publication-series-design.md) 的"系列 5:成果发布与社区"一节。

---

## 这条系列在仓库里的位置

导师第二轮反馈涉及的内容,被拆成了 5 个独立系列同时构建("论文发表系列",从写作到社区回响):科研写作、学术绘图、投稿与评审、学术演讲与参会、以及这一条(成果发布与社区)。5 个系列彼此独立、分属不同目录,按时间线大致排在论文生命周期的不同阶段——这条排在最后:**论文和代码都已经完成(甚至已经被接收)之后,怎么让工作被更多人看见、用起来**。这份路线图只负责自己这一条系列,不去核实、不去修改其余 4 个系列的内容——**曾经在设计文档里读到的其余系列名字(科研写作/学术绘图/投稿与评审/学术演讲与参会)在撰写这条系列时确实已经能在仓库里找到对应目录**,但它们各自是否已经完整收尾,不属于这份路线图能判断的范围,本文因此只在少数确认过真实存在、且直接相关的地方做了谨慎的交叉引用(比如下方 03 号文件提到投稿系列的双盲匿名化那一节),没有把 5 个系列的内容互相糅合。

**这不是"深挖系列"(numpy/torch/dsa 那种回答"这个函数/机制是什么"),也不完全是 [daily-toolkit-deep-dive](../daily-toolkit-deep-dive/00-roadmap.md) 那种纯操作系列**——比较接近后者(6 步操作类模板复用自那条系列),但内容性质更接近"判断力+范例"而不是纯粹的工具操作步骤,这一点在第 3 号文件(学术社区互动)体现得最明显。

### 和 daily-toolkit-deep-dive 04 号文件(Git 协作)的边界(必须先说清楚)

[`daily-toolkit-deep-dive/04-git-collaboration-workflow.md`](../daily-toolkit-deep-dive/04-git-collaboration-workflow.md) 已经完整讲过 git 这个工具本身怎么用——分支、commit 粒度、真实构造并解决的 merge conflict、`.gitignore`、PR/review、`git reflog`。**这条系列的 01 号文件完全不重复这些内容**,处理的是另一个问题:git 操作都会了、代码也确实推上了 GitHub,但作为一个**要给审稿人、给陌生读者看的成品**,这个仓库应该长什么样(README 该写什么、License 怎么选、复现说明该包含哪些确切步骤)。需要具体的 git 操作(比如怎么处理不小心提交进仓库的大文件),01 号文件会直接指回 04 号文件,不重写一遍。

---

## 格式模板(复用 daily-toolkit-deep-dive 已验证的 6 步操作类模板)

不重新设计新模板——[设计文档](../../docs/superpowers/specs/2026-07-25-paper-publication-series-design.md)里"系列 5"这一节明确写了"格式模板:复用 6 步操作类模板(和系列 3、daily-toolkit 一致)"。具体是 [`daily-toolkit-deep-dive/01-vscode-editor-workflow.md`](../daily-toolkit-deep-dive/01-vscode-editor-workflow.md) 确立的这 6 步:

1. **为什么需要这个 / 不会有什么后果**——从零建立动机
2. **环境要求**——需要什么、怎么确认自己已经有了
3. **一步步跟着做**——真正的操作核心,能真实跑的一律真实跑过、贴真实输出
4. **背后发生了什么**——关键步骤在底层做了什么
5. **常见坑**——真实会卡住的地方
6. **自测清单**——读者自己校验有没有真的掌握

3 个分类文件(01-03)每篇内部拆成 4-7 个编号小节,每个小节各自套用这 6 步——这一点也复用了 04 号 git 文件"多个 `## N. 主题` 顶层小节、每节内部走一遍 6 步"的组织方式,不是每篇文章从头到尾只走一次 6 步。

---

## 验证环境(必须诚实交代,这是本系列执行难度最高的部分)

- **Python 环境**:仓库根目录 `.venv`(`python 3.13.9`),真实确认已装 `huggingface_hub==1.18.0`。
- **HuggingFace 账号/token:本机没有真实可用的**——用 `huggingface_hub.HfApi().whoami()` 现场验证,真实返回 `LocalTokenNotFoundError`;环境变量 `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` 也现场确认均未设置。这意味着任何需要写权限的 Hub 操作(建仓库、上传文件)在这台机器上客观无法执行,[02 号文件](02-huggingface-release.md)从第 0 节开始就把"哪些内容本地可验证、哪些内容因为没有账号完全没做"分开标注,不含糊带过。
- **一个撰写过程中才弄清楚的重要细节:这台机器确实有真实的出站网络访问**,而且 `huggingface_hub` 有一个不需要账号、只读校验性质的公开接口(`ModelCard.validate()`,真实 POST 到 `https://huggingface.co/api/validate-yaml`)——这次撰写过程中真实调用过,真实耗时约 1.1 秒,真实拿到了 Hub 服务端当前认可的完整 license 标识符列表。这不是"上传",是"问一下 Hub 这份元数据写得对不对",[02 号文件第 6 节](02-huggingface-release.md)把这条边界画得很具体:本地元数据构造/解析(离线、纳入 `_verify_md.py` 自动化验证范围)、一次性真实网络校验(不需要账号,但依赖网络在未来仍然可用,展示为带时间戳的真实记录,不纳入自动化范围)、真正的上传(需要账号,本文完全没有触碰,也没有冒充触碰过)——三种颗粒度分得很清楚,不是简单地说"没验证"或者"已验证"两个词能概括的。
- **真实项目素材**:`research/world-model-imagination-controller/`(用户真实在研、即将投稿 ICLR 的项目)。撰写前完整读过其中的调研文档(`00-brainstorm-10-ideas.md`、`04-sharpened-recommendation.md`、`07-baseline-reproducibility-audit.md` 等)和真实存在的 pilot 代码(`eval-protocol/` 目录下 7 个 `.py` 文件、`PROTOCOL.md`、3 份 `RESULTS*.md`),04 号文件对这个项目做了一次真实的发布清单审计(不是编造的场景)。
- **WebSearch 真实调研**:Papers with Code 官方 checklist 与 README 模板、HuggingFace 官方 Model Card / Dataset Card 文档、Zenodo GitHub 归档官方文档、HuggingFace Papers 页面机制、开源许可证选择的社区共识、NeurIPS Paper Checklist 关于随机种子/误差棒的要求、学术论文 X/Twitter 宣传帖写作建议、GitHub issue 复现性反馈的社区规范——每篇文件末尾单独列出参考来源链接,不笼统概括。

---

## 进度表

| # | 分类 | 文件 | 状态 |
|---|------|------|------|
| 01 | 开源代码发布规范(README/依赖锁定/随机种子/License/DOI) | [01-open-source-code-release.md](01-open-source-code-release.md) | ✅ 已完成(6 个小节均按 6 步模板;真实发布清单扫描器、依赖锁定检查器、种子/方差 demo 三段代码全部离线可复现验证通过;真实项目对照数据引自 `07-baseline-reproducibility-audit.md` 里已经一手核查过的 11 篇竞品仓库真实结论,不重复核实第三方仓库;License/DOI 两节如实标注验证颗粒度——前者是社区共识而非法律判断,后者按 Zenodo 官方文档描述、未做端到端真实验证) |
| 02 | HuggingFace Release(Model Card / Dataset Card / 上传前检查) | [02-huggingface-release.md](02-huggingface-release.md) | ✅ 已完成(全系列诚实标注难度最高的一篇,第 0 节+第 6 节把"本地语法可验证"“真实但一次性的网络校验"“完全没做的真实上传"三个颗粒度分开说清楚;7 段代码离线自动化验证通过,含一次真实撞到、通过读源码精确定位机制的坑——`tags`/`datasets` 字段传入裸字符串时静默失败的方式不一样,前者被 `_to_unique_list` 逐字符拆开再去重,后者原样保留成非 list 字符串) |
| 03 | 学术社区互动(挂号渠道/宣传帖/GitHub issue/邮件/长线关系) | [03-community-interaction.md](03-community-interaction.md) | ✅ 已完成(全系列"可运行代码"占比最低、判断力占比最高的一篇,如实按这个特点处理,不硬凑代码;两段真实的文字机械检查器——宣传帖字数/夸大用词检查、issue 回复善意/具体信息检查——离线验证通过;示例宣传帖/示例 issue 场景取材于真实项目的真实研究动机和真实 pilot 代码,明确标注"教学示例,非真实发生") |
| 04 | 发布清单串讲(拿真实项目走一遍) | [04-release-checklist-walkthrough.md](04-release-checklist-walkthrough.md) | ✅ 已完成(判断详见下方"关于 04 号文件的决定"一节;两段代码真实扫描 `research/world-model-imagination-controller/` 这个真实目录,只断言返回结构、不断言具体结果,保证项目状态变化后依然能正常运行;真实发现:项目当前没有 README/LICENSE/依赖清单,但随机种子+方差报告方法论已经提前做对,且缺失的依赖清单实际只需要 `numpy` 一行——这些是这次真实审计的结果,不是假设的教学场景) |

---

## 关于 04 号文件的决定(为什么没有直接把清单塞进本文件收尾)

[设计文档](../../docs/superpowers/specs/2026-07-25-paper-publication-series-design.md)里"系列 5"这一节的原话是:"规模小,不强行配一篇同等重量的叙事 capstone,收尾用一份可勾选的'发布清单串讲'……这个规模判断如果后续发现内容其实很丰富,可以再补,不提前假设。"

实际撰写到这一步时,判断是:**这份"清单串讲"如果只是把 01-03 号文件的检查项目重新罗列一遍,那确实应该直接并进本文件的收尾小节,没必要单独开文件**——但真实翻查 `research/world-model-imagination-controller/` 之后,发现这个真实项目本身就是一个现成的、有真实缺口(没有 LICENSE、没有依赖清单)和真实亮点(随机种子方法论提前做对了)的审计对象,值得用真实代码扫一遍、拿真实结果说话,而不是空对空地重复清单条目——这部分内容(真实扫描代码+真实发现+两类现状分析)有独立的信息量,和 01-03 号文件"讲清楚原则"的性质不一样,所以最终决定单独开 [04-release-checklist-walkthrough.md](04-release-checklist-walkthrough.md),而不是塞进这里。

---

## 已知教训(写给这条系列自己,也写给以后可能翻到这里的人)

1. **这台机器上,`_verify_md.py` 自动执行的 `python` 代码块,只要输出(包括捕获到的报错信息里包含的源码片段)里出现非 ASCII 字符,就有真实概率触发父进程用 `gbk` 编码解码 `UTF-8` 字节流失败(`UnicodeDecodeError: 'gbk' codec can't decode byte ... illegal multibyte sequence`)**——这条系列撰写过程中真实撞到过两次:一次是 `print()` 语句里直接写了中文,另一次更隐蔽,是一处 `assert` 失败后,Python traceback 自动打印了那一行**源码**(其中恰好带着一段中文行内注释),同样触发了这个解码错误。第二次尤其提醒人:**不能只检查 `print()` 输出是不是纯 ASCII,还要考虑"如果这一行断言真的失败了,traceback 会不会连带把这一行的中文注释也打印出来"**——最终解决方式是让所有会被 `_verify_md.py` 自动执行的 `python` 代码块(包括注释)全程保持纯 ASCII,中文说明一律放在代码块外的正文里。这和 [daily-toolkit-deep-dive/06-monitoring-and-debugging.md](../daily-toolkit-deep-dive/06-monitoring-and-debugging.md) 撰写时记录的"pdb 交互和标准输出重定向进文件,中文会乱码"是同一类环境限制在不同场景下的复现,值得任何以后在这台机器上继续给这个仓库写"真实可执行 Python 代码块"的人提前知道,不用重新踩一遍。
2. **调试这类坑之前,不要急着归因于"编码问题就是显示乱码而已"**——第一反应可能以为只是控制台显示不好看,但真实观察到的是**代码块被判定为 FAIL、`result.stderr` 变成 `None`**,是会让 `_verify_md.py` 报出假阴性(明明逻辑是对的,报告却是"这段代码跑不通")的真实故障,不是单纯的视觉问题,必须当真的执行失败来修。
3. **测试一个不熟悉的库的字段校验行为时,不要只满足于"观察到了不报错"就下结论,要么读一遍源码,要么把断言写得足够具体去撞验证**——`ModelCardData(tags="字符串")` 最初以为只是"被当可迭代对象逐字符拆开",直到断言写成精确匹配 `list("world-models")` 才发现真实结果其实还被去重了(读 `huggingface_hub` 源码确认是内部调用了 `_to_unique_list`);这个更精确的发现本身也说明:遇到"反直觉但没报错"的库行为,值得花两分钟读一下源码,比停留在"反正它把字符串拆开了"这种笼统印象更可靠。

---

*创建:2026-07-25*
