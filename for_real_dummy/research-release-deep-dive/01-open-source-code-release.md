# 01 · 开源代码发布规范 —— README、依赖锁定、随机种子、License、代码引用

> 总览见 [00-roadmap.md](00-roadmap.md)
> **和 [daily-toolkit-deep-dive/04-git-collaboration-workflow.md](../daily-toolkit-deep-dive/04-git-collaboration-workflow.md) 的边界(先说清楚)**:那篇讲的是"怎么用 git 这个工具"——分支、commit、merge conflict、PR、`.gitignore`、`git reflog`,操作层面的东西。**这篇完全不重复那些内容**,回答的是另一个问题:你的论文代码仓库,git 操作全都会了、也确实推上了 GitHub,但作为一个**要给审稿人 / 陌生读者看的成品**,它应该长什么样?两篇的关系是:04 号文件教你怎么把改动安全地存进仓库,这篇教你仓库存好之后,以"发布"为目的该补什么。需要具体的 `.gitignore` 写法、大文件怎么处理,回 04 号文件第 4 节,这篇不再重复。

---

## 0. 这篇文章是怎么验证的(先说清楚)

- **调研来源全部真实可查**:Papers with Code 官方的 [`releasing-research-code`](https://github.com/paperswithcode/releasing-research-code) 仓库(NeurIPS 2021 官方推荐)、NeurIPS 官方 Paper Checklist 关于随机种子/误差棒的要求、Zenodo 官方文档关于 GitHub Release 存档流程、开源许可证选择的社区共识——下文每处引用都标了来源,不是凭印象转述。
- **真实项目素材**:第 2、4 节用到的"发布得好 vs 发布得差"对比案例,取材于 `research/world-model-imagination-controller/07-baseline-reproducibility-audit.md`——这是用户导师指导下的真实调研文档,里面对 11 篇最近邻竞品论文的代码仓库做过一手核查(真实 star 数、真实 commit 时间、真实 checkpoint 链接,调研文档原文写明"全部用 WebSearch/WebFetch/GitHub API 做一手核查...不是转述")。这里直接引用调研文档已经查证过的结论,不重新核实这些第三方仓库(没有必要重复劳动),但会如实标注"这是转引自项目自己的调研文档,时间点是该文档撰写时"。
- **可运行代码全部在 `.venv` 里真实跑过,用 `_verify_md.py 01-open-source-code-release.md` 独立复验**:第 2 节的"发布清单扫描器"、第 4 节的"种子与方差"demo、第 3 节的"依赖锁定检查",全部是真实执行、真实断言过的代码,不是伪代码。这几段代码为了能被独立重复验证,构造的是**临时目录里的合成示例**(不是去扫描某个会随时间变化的真实仓库)——真实项目的扫描结果放在 [04-release-checklist-walkthrough.md](04-release-checklist-walkthrough.md) 里,以"某个具体日期的真实快照"呈现,不掺进这里的可重复断言。
- **License 小节和"打 DOI"小节如实标注验证颗粒度**:License 文本本身的法律效力这篇不做也不可能做法律判断(不是律师),只讲工程社区的选择共识;"给代码打 DOI"这一步涉及 Zenodo 账号授权 + 真实 GitHub Release 触发的 webhook,本机没有可用于测试的 Zenodo 账号,流程按官方文档如实描述,标注"未做端到端验证"。

---

## 1. 为什么"能在你自己电脑上跑通"不等于"可以发布"

**为什么需要这个 / 不会有什么后果:**

`research/world-model-imagination-controller/07-baseline-reproducibility-audit.md`——用户自己项目里一份真实的调研文档——做过一件很直接的事:把 11 篇最近邻竞品论文的代码仓库,按"代码是否公开 / 权重是否可下载 / 依赖的 benchmark 是否公开"三条标准过了一遍。结论很扎眼:**只有 2 篇(Video-T1、Finding the Time to Think)代码+权重+benchmark 全部齐全,可以直接拿来当"确实能跑"的 baseline**;另有几篇"半发布"(核心模块权重没放出来);还有几篇"只能引用论文数字"——不是代码没写完,是压根没打算发布,或者发布了但缺了复现所需的关键信息(比如 FFDC 的调研记录写着"verifier 关键超参数(层数/参数量/数据规模)论文均未披露")。

这条真实调研提前替你验证了一件事:**"这篇论文有没有开源代码"和"这份代码能不能被别人真的跑起来、跑出论文里报告的数字",是两个不同的问题**,后者的达标率明显更低。你自己写代码的时候,天然知道该在哪个目录跑、该传什么参数、该激活哪个 conda 环境——这些知识对你来说是隐性的、不需要写下来的。但发布代码的目标读者,是审稿人、是想在你工作基础上做后续研究的陌生人、是几个月后已经忘记这些细节的你自己——对他们来说,你没写下来的东西,就是不存在的。

Papers with Code(现在是 NeurIPS 官方推荐的一部分)做过一次更大规模的量化:他们评估了 NeurIPS 2019 之后放出的仓库,按 5 项核心标准(依赖说明、训练脚本+超参数、复现指令、结果表格、预训练权重)打分——**一项都不满足的仓库,GitHub star 数中位数只有 1.5;5 项全部满足的仓库,中位数是 196.5**——但只有 9% 的仓库做到了全部 5 项([Papers with Code 官方博客](https://medium.com/paperswithcode/ml-code-completeness-checklist-e9127b168501),[官方 checklist 仓库](https://github.com/paperswithcode/releasing-research-code))。不会有什么后果——如果不管这件事:代码本身可能完全正确,但审稿人打开仓库看不懂怎么跑,会直接影响他们对你工作严谨性的印象;论文接收之后,别人想在你的方法上做后续对比实验,卡在"装不上环境""跑出来的数字对不上论文"这类本可以避免的问题上,你的工作被引用、被建立在其上的机会也跟着变少。

**环境要求:** 无(这一节是建立动机,不涉及操作)。

**背后发生了什么:** Papers with Code 这份 checklist 不是凭空定的规则,是**反向从"哪些仓库在社区里获得了最多认可"里提炼出的共同点**——"基于分析 200+ 个 ML 仓库,看哪些仓库反响最好,再看这些仓库共同做对了什么"。这也是为什么它后来被 NeurIPS 2021 收编为官方推荐:不是审美偏好,是有相关性数据支撑的经验规律。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 觉得"我论文里写了实现细节,代码就不用写文档了" | 论文页数有限,很多工程细节(具体版本号、启动命令的完整参数)不会写进论文正文 | 论文和代码仓库服务不同的读者场景——论文让人理解"为什么这么做",仓库让人真的跑起来,两者不能互相替代 |
| 觉得"等论文中了再整理仓库,现在先随便放" | 低估了整理的工作量,截止日期前手忙脚乱 | 第 2-6 节的清单越早对照越好,不必等到 camera-ready 前一天 |

**自测清单:**

- [ ] 能说出 Papers with Code 5 项核心标准分别是什么(依赖说明、训练脚本+超参数、复现指令、结果表格、预训练权重)
- [ ] 能举出至少一个"代码公开了,但复现门槛依然很高"的真实原因(缺超参数/缺 checkpoint/依赖未公开的第三方资源)
- [ ] 能说清楚"论文里写清楚了"和"仓库里能跑起来"为什么不能互相替代

---

## 2. README 该写什么、按什么顺序 —— 一个真实扫描器 + 两组真实对照

**为什么需要这个 / 不会有什么后果:**

README 是仓库的门面,大多数人(包括审稿人)决定要不要继续看下去,只花几秒钟扫一眼 README。Papers with Code 的 5 项核心标准里,除了"预训练权重"之外,其余 4 项(依赖说明、训练脚本+超参数、复现指令、结果表格)本质上都是"README 里有没有讲清楚"的问题。

`07-baseline-reproducibility-audit.md` 里两组真实对照,正好对应"做得好"和"做得有缺口"两种真实状态(均为该调研文档一手核查所得,不是转述):

- **做得好的例子**:Video-T1(GitHub `liuff19/Video-T1`,317★,17 forks,MIT 协议,持续维护 1 年以上)——调研记录明确写"11篇里门槛最低";Finding the Time to Think(GitHub `Aneeshers/realtime-rl-code`,14★)——调研记录特别指出它的 README "专门有 'Caveats & known limitations' 一节讨论跨 GPU 型号(H100/A100/A40)的复现方差,CPU-only 未测试过",原话评价是"这种坦诚记录本身是'真实可跑的研究代码'的佐证"。star 数不是目标本身,但这里能看出一个规律:**主动交代"什么条件下可能复现不出来",反而增加可信度,不是减分项**。
- **有缺口的例子**:FFDC 的 backbone 虽然开源,但"verifier 关键超参数(层数/隐藏维度/学习率/数据规模)论文均未披露";ROI-Reasoning "唯一提及'publicly available'的地方是 Ethics 部分说明'实验都在公开 benchmark 上进行'(指数据集,不是作者自己的代码)"——代码本身没有公开。这两个例子的问题不是"态度不认真",大概率是没把"发布代码"当成论文工作里需要投入时间的一部分,截止日期一到,这部分就被牺牲了。

**环境要求:** 无特殊环境,能读写文本文件即可;下面的扫描器用仓库已有的 `.venv`(`python 3.13`)跑。

**一步步跟着做:**

**第一步:一份能直接套用的最小 README 骨架**(顺序参照 Papers with Code 模板 + 上面两组真实对照总结出的顺序):

```markdown
# 项目名 —— 一句话说清楚这是什么

简短的项目描述(2-4 句):解决什么问题、方法一句话、指向论文/arXiv 链接。

## 安装

假设读者对这个项目的技术栈毫无背景知识:
\`\`\`bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt   # 或 conda env create -f environment.yml
\`\`\`

## 复现论文结果

给出能直接抄进终端跑的确切命令,不要让读者自己猜参数:
\`\`\`bash
python train.py --config configs/main.yaml --seed 0
python eval.py --checkpoint checkpoints/main.pt
\`\`\`
预期输出 / 结果表格(哪怕只贴论文 Table 1 的一个子集):

| 方法 | 指标 A | 指标 B |
|---|---|---|
| Baseline | ... | ... |
| Ours | ... | ... |

## 已知局限 / 复现注意事项

老老实实写:哪些环境下测过、哪些没测过、哪些结果有已知方差。

## 引用

\`\`\`bibtex
@inproceedings{...}
\`\`\`

## License

见 [LICENSE](LICENSE)(第 5 节详细展开怎么选)。
```

**第二步:一个真实的"发布清单扫描器"**,用来快速自查一个目录有没有踩到 Papers with Code checklist 里能用文件系统直接判断的几条信号(不追求语义理解,只做粗筛——第 4 节会诚实指出这类脚本的假阳性风险):

```python
import tempfile
from pathlib import Path


def scan_release_checklist(repo_dir: Path) -> dict:
    """极简版 ML Code Completeness Checklist 扫描器：只查文件系统层面的信号。"""
    files = {p.name.lower() for p in repo_dir.iterdir() if p.is_file()}
    readme = any(n.startswith("readme") for n in files)
    license_ = any(n.startswith("license") for n in files)
    deps = any(n in {"requirements.txt", "environment.yml", "pyproject.toml"} for n in files)
    repro_keywords = ("reproduce our results", "how to reproduce", "how to run", "复现")
    mentions_repro = False
    if readme:
        for p in repo_dir.iterdir():
            if p.is_file() and p.name.lower().startswith("readme"):
                text = p.read_text(encoding="utf-8", errors="ignore").lower()
                mentions_repro = any(kw in text for kw in repro_keywords)
    return {
        "has_readme": readme,
        "has_license": license_,
        "has_dependency_manifest": deps,
        "readme_mentions_reproduction": mentions_repro,
    }


# 场景一:一个只写了半截的仓库
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    (root / "README.md").write_text(
        "# My Project\n\nA short description of what this repo does. No instructions included yet.\n",
        encoding="utf-8",
    )
    (root / "train.py").write_text("print('hello')\n", encoding="utf-8")
    report = scan_release_checklist(root)
    print("incomplete repo scan result:", report)
    assert report == {
        "has_readme": True,
        "has_license": False,
        "has_dependency_manifest": False,
        "readme_mentions_reproduction": False,
    }

# 场景二:一个补齐了基本要素的仓库
with tempfile.TemporaryDirectory() as d2:
    root2 = Path(d2)
    (root2 / "README.md").write_text(
        "# My Project\n\n## How to reproduce\n\nRun `python train.py --seed 0`.\n", encoding="utf-8"
    )
    (root2 / "LICENSE").write_text("MIT License...\n", encoding="utf-8")
    (root2 / "requirements.txt").write_text("torch==2.4.1\n", encoding="utf-8")
    report2 = scan_release_checklist(root2)
    print("completed repo scan result:", report2)
    assert report2 == {
        "has_readme": True,
        "has_license": True,
        "has_dependency_manifest": True,
        "readme_mentions_reproduction": True,
    }

print("OK: both repo states matched the expected checklist result")
```

真实运行结果(用 `_verify_md.py` 独立跑过,两个断言都通过):半成品仓库四项全部标记缺失或 `False`,补齐后的仓库四项全部为 `True`。这个脚本你可以直接拿去扫自己真实的项目目录(把 `repo_dir` 换成真实路径),[04-release-checklist-walkthrough.md](04-release-checklist-walkthrough.md) 就是真的这么做的,扫的是用户自己那个即将投稿 ICLR 的项目目录,得到的是真实结果而不是这里为了演示脚本逻辑而构造的临时目录。

**背后发生了什么:** 上面骨架里"复现论文结果"这一节刻意要求"能直接抄进终端跑的确切命令",而不是"运行训练脚本即可"这种模糊描述——原因是 Papers with Code 的原始建议就是"exact steps"(确切步骤),背后的逻辑是:你确实知道该传哪些参数,但读者不知道,任何"应该显而易见"的省略,对第一次接触这份代码的人都是一个可能卡住的坑。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 结果表格贴了,但复现出来的数字对不上 | 表格里的数字和某个特定超参数/种子/checkpoint 绑定,但没写清楚是哪一组 | 结果表格旁边直接注明对应的确切命令或 config 文件名 |
| README 写了"local development"但读者根本没有你本地那台机器的隐性配置(CUDA 版本、系统依赖) | 把"在我机器上有效"误当成"通用步骤" | 假设读者是一台全新的机器,能列出的系统级依赖(CUDA 版本、`apt` 包)也列出来 |
| 关键词扫描类工具(比如上面的 `scan_release_checklist`)显示"有复现说明",但内容其实很敷衍 | 这类脚本只能做文件存在性/关键词命中的粗筛,不理解语义 | 自动化扫描只作为第一道筛子,发布前仍需要自己或找人从头到尾走一遍安装+复现流程 |

**自测清单:**

- [ ] 能不看这篇文章,默写出 README 至少 5 个必须有的部分(简介、安装、复现命令、结果、License)
- [ ] 能说出为什么"Finding the Time to Think"在真实调研里被认为是"可信"的例子——诚实写局限反而加分
- [ ] 能用上面的扫描器脚本,对自己任意一个项目目录跑一遍,说出真实缺了哪几项

---

## 3. 依赖锁定 —— 为什么"在我电脑上能跑"不是复现说明

**为什么需要这个 / 不会有什么后果:**

`pip install torch` 这条命令,今天跑和三个月后跑,拿到的 PyTorch 版本可能完全不同——如果版本之间有任何行为差异(哪怕只是默认参数变了),你今天测出来的结果,三个月后的读者用同一份代码可能根本复现不出来,而且**双方都不知道问题出在版本上**,只会怀疑是不是哪里操作错了。这不是假设的场景:上文调研文档里 Finding the Time to Think 的 README 专门讨论跨 GPU 型号的复现方差,是同一类问题的另一个维度(硬件而不是软件版本),说明这类"环境细节导致复现有出入"是真实、常见的困扰,不是小题大做。

**环境要求:** 无特殊环境,`.venv` 里的 `pip` 即可跑下面的检查脚本。

**一步步跟着做:**

**依赖清单三选一**(不是必须全都提供,选一个和你项目技术栈匹配的):

- `requirements.txt`(pip):最简单,适合大多数 Python 项目
- `environment.yml`(conda):项目依赖非 Python 的系统级库(比如特定版本的 CUDA、编译好的二进制)时更合适
- `pyproject.toml`(如果代码本身要打包成一个可安装的库,而不只是一堆脚本)

**关键区分:"锁定精确版本"和"只写包名"是两种不同的承诺**——`torch` 不锁版本,意味着"能装上什么版本就用什么版本";`torch==2.4.1` 意味着"就是这个版本跑出的论文结果"。下面这个脚本可以快速检查一份 `requirements.txt` 里锁没锁版本:

```python
def check_pins(requirements_text: str) -> dict:
    """检查 requirements.txt 里哪些依赖锁定了精确版本(==),哪些没有。"""
    lines = [
        line.strip()
        for line in requirements_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    pinned, unpinned = [], []
    for line in lines:
        if "==" in line:
            pinned.append(line)
        else:
            unpinned.append(line)
    return {"pinned": pinned, "unpinned": unpinned}


sample = """
torch==2.4.1
numpy
gymnasium>=0.29
huggingface_hub==1.18.0
"""
report = check_pins(sample)
print(report)
assert report["pinned"] == ["torch==2.4.1", "huggingface_hub==1.18.0"]
assert report["unpinned"] == ["numpy", "gymnasium>=0.29"]
total = len(report["pinned"]) + len(report["unpinned"])
print(f"{len(report['unpinned'])}/{total} dependencies are NOT pinned to an exact version")
```

真实运行结果(`_verify_md.py` 独立验证通过):4 条依赖里 2 条锁定、2 条没锁定(`numpy` 完全没有版本约束,`gymnasium>=0.29` 只锁了下界,上界依然是开放的)。

**背后发生了什么:** `pip freeze > requirements.txt` 是最常见的"生成依赖清单"方式,它会把你当前环境里**每一个**已安装的包(包括间接依赖、和这个项目其实没关系的包)全部锁定精确版本——好处是完全可复现,坏处是清单可能有几十上百行,读者装的时候如果自己机器的操作系统/Python 版本和你不同,某些精确锁定的包可能根本装不上(比如某个包只有特定平台的预编译轮子)。工程实践里常见的折中是:核心依赖(直接决定实验结果的库,比如 `torch`/`numpy`)精确锁定,外围工具类依赖(比如日志/进度条库)可以放宽,这不是非黑即白的规则,而是"复现精确性"和"跨环境可安装性"之间的权衡。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| `pip install -r requirements.txt` 报错,提示找不到某个精确版本 | `pip freeze` 生成的清单里锁了一个平台特定的构建(比如带 CUDA 版本后缀的 torch wheel),换了平台装不上 | 核心依赖单独写清楚"怎么装"(比如 PyTorch 官网按 CUDA 版本选择安装命令的那一段),不要指望一条 `pip install -r` 能覆盖所有平台 |
| 读者反馈"结果和论文报告的对不上",但代码逻辑确认没问题 | 依赖没锁版本,读者装到了行为不同的新版本 | 先问读者 `pip list` 里几个核心库的版本,和你论文实验时的版本对比 |
| requirements.txt 有几十行,大部分是间接依赖 | 直接用 `pip freeze` 全量导出 | 可以保留(不算错),但建议在 README 里点出"核心依赖是哪几个",不要让读者在几十行里自己猜 |

**自测清单:**

- [ ] 能说出 `requirements.txt`/`environment.yml`/`pyproject.toml` 三者分别适合什么场景
- [ ] 能说清楚"锁定精确版本"和"只写包名不锁版本"之间,复现精确性和可安装性的权衡
- [ ] 能用 `check_pins` 这类思路,自查一份真实 `requirements.txt` 里有没有锁定核心依赖的版本

---

## 4. 随机种子与结果方差 —— 一个数字背后有没有运气成分

**为什么需要这个 / 不会有什么后果:**

深度学习训练本身包含大量随机性来源(权重初始化、数据打乱顺序、dropout、某些 GPU kernel 的非确定性实现)——同一份代码、同一份数据,换一个随机种子跑,结果可能有肉眼可见的波动。如果论文只报告"跑了一次、正好很好看"的那个数字,读者复现时随手换了个种子,拿到一个明显更差的数字,会合理怀疑论文数字是不是运气好挑出来的("cherry-picked seed")。

NeurIPS 官方 Paper Checklist 明确把这一条写成了必答题:**要求作者回答"是否报告了误差棒(比如相对随机种子多次运行的误差棒)"**([NeurIPS Paper Checklist Guidelines](https://neurips.cc/public/guides/PaperChecklist))——不满足会被认为复现性信息不完整。这个要求不是走形式:NeurIPS 2019 Reproducibility Program 的报告(Pineau 等人)正是这一整套 checklist 机制的起点,该项目本身就是为了系统性回应"机器学习有自己的复现危机"这个问题。

**环境要求:** `.venv` 即可,不需要真的训练神经网络——下面用一个"跑起来很快但确实带随机性"的玩具函数演示方法论,不是要教你怎么训练模型。

**一步步跟着做:**

```python
import random
import statistics


def toy_noisy_score(seed: int) -> float:
    """模拟一次"训练+评测"得到一个分数,带真实的随机性来源(这里简化成随机数生成器本身)。"""
    rng = random.Random(seed)
    return sum(rng.random() for _ in range(1000)) / 1000


SEEDS = [0, 1, 2, 3, 4]  # 和真实项目 eval-protocol 脚本里的写法一致(见下方引用)
scores = [toy_noisy_score(s) for s in SEEDS]
mean = statistics.mean(scores)
std = statistics.pstdev(scores)
print("scores across 5 seeds:", [round(s, 4) for s in scores])
print(f"mean +/- std = {mean:.4f} +/- {std:.4f}")

# assertion 1: the same seed run twice must give exactly the same number -- the basic promise of a fixed seed
assert toy_noisy_score(0) == toy_noisy_score(0)
# assertion 2: different seeds should give different numbers -- catches the bug where a --seed flag
# is parsed but never actually wired into the random source (a "fake" fixed seed)
assert len(set(scores)) == len(SEEDS)
print("OK: same seed reproduces exactly, different seeds give different numbers")
```

真实运行结果(`_verify_md.py` 独立验证通过):`mean +/- std = 0.5037 +/- 0.0068`(具体小数点后几位每次运行都一样,因为种子固定)。

**这不是凭空编的方法论**——`research/world-model-imagination-controller/eval-protocol/` 下三份真实的 pilot 脚本(`run_pilot_study.py`、`run_pilot_study_neural.py`、`task_conditioning_pilot.py`)里都真实写着 `SEEDS = [0, 1, 2, 3, 4]`,并且 `PROTOCOL.md` 里明确写"每组配置独立重复 5 个随机种子(数据采样+模型学习+评测全部重新来一遍),报告均值±标准差"——这正是上面 NeurIPS checklist 要求的具体实践,项目在写论文之前的 pilot 阶段就已经在做了,不是临时补的。[04-release-checklist-walkthrough.md](04-release-checklist-walkthrough.md) 会展示这一点在真实项目里的具体样子。

**背后发生了什么:** `random.Random(seed)` 创建的是一个**独立的、状态和全局随机数生成器无关**的生成器实例——这保证了"给定同一个种子,不管这段代码在什么上下文里被调用,输出都完全一样"。真实的深度学习训练要更复杂:除了 Python 的 `random`,还有 `numpy` 的随机源、`torch` 的 CPU/GPU 随机源,分别有各自的种子设置接口,漏设了任何一个,"固定种子"就不是真的完全固定——这也是为什么"声称固定了种子但复现不出相同数字"是一类真实存在的坑(NeurIPS 相关研究里明确提到,卷积层不开启 `cudnn.deterministic`、不关闭 `cudnn.benchmark`,同一个种子也不能保证完全复现)。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 论文只报告一个数字,没有 ± 误差范围 | 图省事,或者只跑了一次 | 至少补跑 3-5 个种子,报告 mean±std;如果时间/算力实在有限,在局限性部分诚实说明只跑了单次 |
| 写了 `--seed` 参数,但复现出来数字还是不一样 | `--seed` 只设置了 Python 的 `random`,没设置 `numpy`/`torch`,或者用了非确定性的 GPU kernel | 检查是否所有随机源(`random`/`numpy.random`/`torch.manual_seed`/`torch.cuda.manual_seed_all`)都设置了;GPU 上完全确定性可能还需要额外的 `cudnn` 设置,代价是可能变慢 |
| 挑了"跑出来最好看"的那个种子的数字写进论文 | 主观上没意识到这是"挑数字",客观上确实是 | 提前决定好种子集合(比如固定用 0-4),不要跑出结果之后再挑,mean±std 的报告方式天然能防住这个问题 |

**自测清单:**

- [ ] 能说出 NeurIPS Paper Checklist 对随机种子/误差棒的具体要求是什么
- [ ] 能说清楚"固定了 Python 的 `random` 种子"和"真的完全复现"之间为什么可能还有差距(numpy/torch/cudnn 各自的随机源)
- [ ] 能独立写一个"多种子跑 + 报告 mean±std"的最小示例,不需要照抄这篇文章的代码

---

## 5. License 怎么选 —— 代码和论文文本不是一回事

**为什么需要这个 / 不会有什么后果:**

没有 License 的代码仓库,严格来说**默认版权保留、别人无权使用你的代码**——哪怕你把仓库设成 public、哪怉写了"欢迎使用",没有一份明确的 License 文件,法律意义上的"允许别人做什么"依然是模糊的。对于学术代码,这通常不是故意要限制别人使用(大多数研究者希望自己的方法被更多人用、被引用),而是单纯忘了加。

**环境要求:** 无,这一节是判断力而非操作。

**一步步跟着做(该怎么判断):**

先厘清一个容易混的点:**License 有两个不同的对象**——代码用软件许可证(MIT/Apache-2.0/BSD 这一类),论文文本本身(如果你要单独发布 PDF 或者预印本的再分发权限)用 Creative Commons 这一类内容许可证(CC-BY/CC-BY-SA)。这篇只讨论代码这一半。

学术研究代码最常见的两个选择:

| | MIT | Apache 2.0 |
|---|---|---|
| 长度/复杂度 | 极短(约 170 词),几乎人人能一眼读完 | 更长、更正式,包含专利授权条款 |
| 核心许可 | 允许使用/复制/修改/分发,几乎没有限制,不提供任何担保 | 同样宽松,额外包含明确的专利授权条款 |
| 什么时候更合适 | 优先简单、想要最大化被采用/被建立在其上的研究代码,是学术界的常见默认选择 | 项目可能涉及专利相关的技术,或者希望给"贡献者/使用者"更明确的专利保护时更合适 |
| 兼容性 | 可以被合并进几乎任何其他许可证的项目(包括 GPL) | 和 GPLv3 兼容,但和 GPLv2 不兼容(GPLv2 不接受 Apache 2.0 的专利条款) |

多个来源(见文末引用)一致的判断是:**MIT 是学术/研究代码的常见默认选择**,原因正是"简单、法律门槛低、有利于被更多人采用"——这和上面第 1 节"你希望别人在你工作基础上做后续研究"的目标是一致的。Apache 2.0 在需要专利保护或者面向更长期商业化部署场景更合适,但对大多数论文配套代码来说不是必需的复杂度。

**具体操作**:GitHub 建仓库或者在已有仓库根目录直接加一个 `LICENSE` 文件,GitHub 网页端 "Add file" 时会有一个官方模板选择器,选中 "MIT License" 或 "Apache License 2.0" 会自动帮你填好年份和你的名字,不需要自己手打法律文本。

**背后发生了什么:** 上面 Hugging Face Hub 官方接受的 License 标识符列表里(第 2 篇 HuggingFace Release 文件会用到这份真实列表,来自 Hub 服务端实际返回的校验结果),`mit` 和 `apache-2.0` 都在其中,和 `cc-by-4.0`(内容类)、`gpl-3.0`(著佐权类,要求衍生作品必须同样开源)并列——这从另一个角度印证了"代码许可证"和"内容许可证"确实是两套不同的体系,不能张冠李戴。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 仓库里完全没有 LICENSE 文件 | 单纯忘了加,或者不知道"默认版权保留"意味着什么 | 尽早加上,不需要等论文中了再补——[04-release-checklist-walkthrough.md](04-release-checklist-walkthrough.md) 会展示这正是真实项目当前的一个真实缺口 |
| 依赖的某个第三方库是 GPL,但自己代码想用更宽松的 MIT | GPL 有"传染性"(衍生作品通常也要求开源),和更宽松的许可证直接混用可能有法律上的不一致 | 这类情况建议真的去查一下具体许可证的兼容性(或咨询学校的技术转移办公室),这篇不代替法律判断 |
| 论文 PDF 和代码用了同一份 License 文件 | 没意识到两者是不同的许可证体系 | 代码仓库根目录放软件许可证(如 MIT);论文本身的版权通常由投稿的会议/出版社条款决定(部分会议允许作者同时以 CC-BY 挂到 arXiv),不要混着写 |

**自测清单:**

- [ ] 能说出"没有 LICENSE 文件"在默认版权规则下意味着什么
- [ ] 能说出 MIT 和 Apache 2.0 至少一个实质性区别(专利条款)
- [ ] 能说清楚"代码许可证"和"论文/内容许可证"是两个不同的体系,不能互相代替

---

## 6. 给代码打一个可引用的 DOI(进阶,可选)—— Zenodo 是干什么的

**为什么需要这个 / 不会有什么后果:**

`github.com/你的用户名/项目名` 这个链接,看起来是一个稳定地址,但它并不是学术意义上"可引用"的东西——仓库可以被删除、改名、设为 private,里面的代码可以在你不知情的情况下被后续 commit 悄悄改变(引用的人如果只写了仓库链接,没写具体是哪个版本,几年后可能已经对应不上当初被引用的那份代码)。会议/期刊在要求"代码可获得性"的时候,更规范的做法是给某个**确定版本的代码**一个永久、不会变的标识符——这正是 DOI(Digital Object Identifier)解决的问题,论文本身通常也有一个 DOI,道理是一样的。

**环境要求:** 一个 GitHub 账号(有仓库管理权限)+ 一个 Zenodo 账号(可以直接用 GitHub 账号登录授权)。**本机没有配置用于测试的 Zenodo 账号**,下面的步骤按 [Zenodo 官方文档](https://help.zenodo.org/docs/github/archive-software/github-upload/) 如实描述,**没有做端到端真实验证**——如果你要用,建议照着官方文档现场走一遍,不要假设这里的每个界面细节和你实际看到的完全一致(这类账号授权类操作,和 daily-toolkit 系列里 GitHub 网页操作的验证颗粒度是同一类,如实标注不冒充)。

**一步步跟着做(按官方文档描述,未做自动化验证):**

1. 仓库本身要 public,并且根目录已经有 README + LICENSE(第 2、5 节)。
2. 用 GitHub 账号登录 [zenodo.org](https://zenodo.org),在个人设置里找到 GitHub 集成入口,点 "Sync now" 同步你的仓库列表。
3. 找到你想要归档的仓库,把开关切到 "on"——这一步会在 GitHub 和 Zenodo 之间建立一个 webhook。
4. (推荐但非必须)在仓库根目录加一个 `CITATION.cff` 文件,写清楚作者信息(包括 ORCID,如果有的话)——Zenodo 抓取元数据时会优先用这个文件,而不是简单地从 GitHub 的 contributor 列表里猜。
5. 回到 GitHub,在仓库的 "Releases" 标签页创建一个新的 Release(比如 `v1.0.0`,对应论文 camera-ready 那个版本的代码)。
6. Zenodo 会自动检测到这次 Release,生成一个专属的 DOI,并把这个版本的代码完整存档——即使你之后删除了 GitHub 仓库,Zenodo 上归档的这份副本依然存在。
7. 之后每次打新的 Release,都会生成一个新版本的 DOI,但 Zenodo 提供一个"概念 DOI"(concept DOI)始终指向最新版本,方便引用时选择"引用某个具体版本"还是"引用这个项目本身"。
8. 把生成的 DOI 徽章(Zenodo 会给你一段现成的 Markdown/HTML)贴回 README 顶部,再补一段人类可读的引用格式(不是所有读者都熟悉怎么用 DOI 生成引用),方便别人复制粘贴。

**背后发生了什么:** Zenodo 抓取元数据的优先级是 `.zenodo.json` > `CITATION.cff` > `LICENSE`——如果你的仓库同时有 `.zenodo.json` 和 `CITATION.cff`,Zenodo 会完全忽略 `CITATION.cff`,这是官方文档明确写的行为,不是 bug。GitHub Release 触发 webhook、Zenodo 拉取这次 Release 对应的代码快照并单独存档,是这套集成能保证"即使原仓库被删,归档副本还在"的关键机制——它存的是一份独立快照,不是简单地指向 GitHub 那个可能会变的链接。

**常见坑(按官方文档整理,未做真实故障复现):**

| 现象 | 原因 | 怎么办(按官方文档) |
|---|---|---|
| 打了 Release,Zenodo 上却没有生成对应记录 | 归档开关没有真的切到 "on",或者 webhook 配置有问题 | 检查 Zenodo 上这个仓库的归档状态页,官方文档提到可以看 "External resources > Archived in" 确认 |
| 元数据(作者列表)不完整或者不对 | 只依赖 GitHub contributor 列表自动生成,没有提供 `CITATION.cff`/`.zenodo.json` | 补上 `CITATION.cff`,列全作者和 ORCID |
| 不确定该不该现在就做这件事 | 这一步通常在论文接收、准备 camera-ready 阶段才需要,不是投稿阶段的必需品 | 大多数会议不会在投稿阶段强制要求 DOI,可以放在"接收之后要做的事"清单里(和 [04-release-checklist-walkthrough.md](04-release-checklist-walkthrough.md) 的收尾清单对应) |

**自测清单:**

- [ ] 能说出 `github.com/.../repo` 这个链接和一个 Zenodo DOI 的本质区别(前者可变,后者指向一份永久快照)
- [ ] 知道触发归档的具体动作是"在 GitHub 上创建一个 Release",不是随便一次 push
- [ ] 知道这一步大概该在项目时间线的哪个阶段做(camera-ready 前后,不是投稿时的硬性要求)

---

*参考来源:[Papers with Code — ML Code Completeness Checklist](https://medium.com/paperswithcode/ml-code-completeness-checklist-e9127b168501)、[paperswithcode/releasing-research-code](https://github.com/paperswithcode/releasing-research-code)、[NeurIPS Paper Checklist Guidelines](https://neurips.cc/public/guides/PaperChecklist)、[Zenodo GitHub 归档官方文档](https://help.zenodo.org/docs/github/archive-software/github-upload/)、License 选择的社区共识综合多篇公开指南。真实项目对照数据引自 `research/world-model-imagination-controller/07-baseline-reproducibility-audit.md`。*
*创建:2026-07-25*
