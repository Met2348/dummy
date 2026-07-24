# 04 · 发布清单串讲 —— 拿真实项目走一遍

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这个系列规模不大(3 个分类),按 [设计文档](../../docs/superpowers/specs/2026-07-25-paper-publication-series-design.md) 的建议,不硬凑一篇同等重量的叙事体 capstone,收尾用一份"发布清单串讲"——但这篇不是把 01-03 号文件的清单原样抄一遍,而是真的拿 `research/world-model-imagination-controller/`(用户真实在研、即将投稿 ICLR 的项目)当一次真实的审计对象,一条一条过。下面每一项都是这次真实检查得到的结果,不是假设的场景。

---

## 0. 这篇文章是怎么验证的(先说清楚,这篇和前三篇性质不一样)

前三篇(01-03)的可运行代码,验证目标是"这段逻辑本身对不对",所以都用临时构造的合成目录/文本,确保不管什么时候重新运行 `_verify_md.py`,结果都应该一致。**这篇的目标不一样,是"这个真实项目现在处于什么状态"**——这个问题的答案会随项目进展改变(比如某天真的补上了 LICENSE,下面第 4 节的结论就会过期,而且是朝好的方向过期)。所以这篇分两类内容处理,不混在一起:

- **下面几个 `python` 代码块,真的会去读 `E:/Workspace/dummy/research/world-model-imagination-controller/` 这个真实目录**——不是编造路径。但代码里的断言只检查"这个报告的结构对不对"(比如"返回的是不是一个字典、字段是不是布尔值"),**不断言具体的检查结果**,所以这些代码在项目状态改变之后依然能正常跑通、依然会给出当时的真实结果,不会因为项目变好了而报错。
- **正文里对这次真实运行结果的具体解读("现在有 LICENSE 吗"之类的结论),标注了明确的日期(2026-07-25)**,当成一次快照来读,不是恒久声明。如果你现在重新跑这篇文章里的代码,得到的结果可能和正文描述的不一样——那不代表这篇文章写错了,代表项目往前推进了,这正是这份清单存在的意义:它应该被不断重新对照,不是读一次就过时的静态文档。

---

## 1. 先用第 1 号文件的扫描器,真的扫一遍这个项目

```python
from pathlib import Path


def scan_release_checklist(repo_dir: Path) -> dict:
    """与 01-open-source-code-release.md 第 2 节完全相同的扫描逻辑,这里指向真实目录。"""
    files = {p.name.lower() for p in repo_dir.iterdir() if p.is_file()}
    return {
        "has_readme": any(n.startswith("readme") for n in files),
        "has_license": any(n.startswith("license") for n in files),
        "has_dependency_manifest": any(
            n in {"requirements.txt", "environment.yml", "pyproject.toml"} for n in files
        ),
    }


project_root = Path("E:/Workspace/dummy/research/world-model-imagination-controller")
eval_protocol_dir = project_root / "eval-protocol"
expected_keys = {"has_readme", "has_license", "has_dependency_manifest"}

if project_root.exists():
    root_report = scan_release_checklist(project_root)
    eval_report = scan_release_checklist(eval_protocol_dir) if eval_protocol_dir.exists() else None
    print("project root scan:", root_report)
    print("eval-protocol/ scan:", eval_report)
    # structural checks only -- these keep passing no matter how the project's real
    # state changes over time, unlike a hardcoded expected-value assertion would
    assert set(root_report.keys()) == expected_keys
    assert all(isinstance(v, bool) for v in root_report.values())
    if eval_report is not None:
        assert set(eval_report.keys()) == expected_keys
else:
    print("skipped: real project directory not found in this checkout")
```

**真实运行结果(2026-07-25,`_verify_md.py` 独立验证通过)**:

```
project root scan: {'has_readme': False, 'has_license': False, 'has_dependency_manifest': False}
eval-protocol/ scan: {'has_readme': False, 'has_license': False, 'has_dependency_manifest': False}
```

三项全部是 `False`。这不是在说这个项目做得不好——它现在的性质是**一份导师指导下的科研调研+原型验证笔记**(腦暴、竞品调研、pilot 脚本),压根还没有到"要不要发布"这个决策点,所以完全没有 README/LICENSE/依赖清单是完全合理的当前状态,不是疏漏。这份扫描的价值在于:**先建立一个真实的基线,后面几节逐项讨论"等到真要发布的时候,这几项分别要花多少功夫补上"**。

---

## 2. 逐项对照 Papers with Code 5 要素,现在分别是什么状态

参照 [01 号文件第 1 节](01-open-source-code-release.md)引用的 5 项核心标准,逐项过一遍:

| # | 要素 | 现状(2026-07-25) | 说明 |
|---|---|---|---|
| 1 | 依赖说明 | ❌ 缺失 | 见第 5 节,实际修复成本很低 |
| 2 | 训练脚本 + 超参数 | ✅ 部分具备 | `eval-protocol/run_pilot_study.py` 等脚本本身就是"训练脚本"(这里的"训练"是给 `LearnedWorldModel` 学习),超参数(`H_SWEEP`/`K_SWEEP`/`N_TRAIN_TRANSITIONS` 等)以模块级常量的形式写在脚本顶部,不是命令行参数,但确实存在、确实可读 |
| 3 | 复现指令 | ⚠️ 不完整 | 没有一份专门的"怎么跑"说明(比如"先 `cd eval-protocol`,再 `python run_pilot_study.py`"这种确切步骤),`PROTOCOL.md` 讲的是研究方法论(为什么这么设计实验),不是操作步骤——这是两种不同的文档,现在只有前者 |
| 4 | 结果表格 | ✅ 具备 | `RESULTS.md`/`RESULTS-neural-ensemble.md`/`RESULTS-task-conditioning.md` 三份文档,真实记录了三轮 pilot 的数字结果 |
| 5 | 预训练权重 | 不适用 | 目前的"模型"是频次估计表格模型和小型神经网络 ensemble,训练成本本身很低(几分钟级别),不是需要预先训练好、动辄几 GB 的 checkpoint,这一项在当前阶段不构成门槛 |

**这次真实核查唯一的、值得单独指出的缺口是第 3 项**:方法论文档很完整(甚至比大多数已发表论文的配套仓库详细,`PROTOCOL.md` 明确写了"为什么这样设计""范围声明"这些通常会被省略的部分),但缺一份`01 号文件第 2 节`骨架里"复现论文结果"那样的确切操作步骤。这是一个**具体、低成本就能补上**的缺口,不是需要重新设计实验的大问题。

---

## 3. 随机种子与结果方差 —— 这一项已经提前做对了

[01 号文件第 4 节](01-open-source-code-release.md)讲的 NeurIPS checklist 要求(报告多个种子的均值±标准差),这个项目在写论文之前的 pilot 阶段就已经在做,值得作为一个正面例子记录下来:

```python
import re
from pathlib import Path

eval_protocol_dir = Path("E:/Workspace/dummy/research/world-model-imagination-controller/eval-protocol")

if eval_protocol_dir.exists():
    seed_pattern = re.compile(r"^SEEDS\s*=\s*(\[.*\])", re.MULTILINE)
    hits = {}
    for py_file in sorted(eval_protocol_dir.glob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        match = seed_pattern.search(text)
        if match:
            hits[py_file.name] = match.group(1)
    print("files with a top-level SEEDS = [...] constant:", hits)
    # structural check: this just confirms the scan produced a dict, not specific contents,
    # so it stays valid even if more pilot scripts are added later
    assert isinstance(hits, dict)
else:
    print("skipped: real project directory not found in this checkout")
```

**真实运行结果(2026-07-25,`_verify_md.py` 独立验证通过)**:

```
files with a top-level SEEDS = [...] constant: {
  'run_pilot_study.py': '[0, 1, 2, 3, 4]',
  'run_pilot_study_neural.py': '[0, 1, 2, 3, 4]',
  'task_conditioning_pilot.py': '[0, 1, 2, 3, 4]'
}
```

三份真实的 pilot 脚本,全部固定了同一组 5 个种子(`0-4`),`PROTOCOL.md` 里明确写着"每组配置独立重复 5 个随机种子(数据采样+模型学习+评测全部重新来一遍),报告均值±标准差"。等到正式论文/代码发布的时候,这部分方法论不需要额外补课,直接沿用即可——这也是这条系列想传达的一个具体教训:**复现性的好习惯,越早养成越好,不是发布前临时补的东西**,这个项目在这一项上已经提前做对了。

---

## 4. License —— 真实缺口,建议怎么补

第 1 节的扫描已经确认:项目根目录和 `eval-protocol/` 目录都**没有 LICENSE 文件**。参照 [01 号文件第 5 节](01-open-source-code-release.md)的判断框架:这是一份研究代码,目标是被更多人看到、用起来、建立在其上做后续工作,MIT 是最贴合这个目标的默认选择——不需要现在就做决定(项目还在 pilot 阶段,连要不要独立开一个仓库发布都还没定),但值得记在"论文接收前后要做的事"清单里,提前知道选哪个可以避免临到发布前才纠结。

## 5. 依赖清单 —— 真实缺口,但修复成本很低

同样没有 `requirements.txt`/`environment.yml`/`pyproject.toml`。但把 `eval-protocol/` 目录下全部 `.py` 文件的 import 语句真实过一遍(用 Grep 逐文件核实,不是猜测),结果是:**除了标准库(`random`/`statistics`/`collections`/`time`/`__future__`)之外,唯一的第三方依赖是 `numpy`(只有 `neural_ensemble_model.py` 一个文件用到,`import numpy as np`)**。这意味着,等真要补一份 `requirements.txt` 的时候,核心内容其实就一行:

```
numpy
```

(具体要不要锁定精确版本,参照 [01 号文件第 3 节](01-open-source-code-release.md)的判断——这份 pilot 代码目前不涉及任何 GPU/深度学习框架特定版本的行为差异,锁不锁版本的风险相对较低,但仍然建议锁,便宜的保险。)

这是一个很具体的例子,说明"依赖清单缺失"这类清单项,实际修复成本可能差异很大——不是每一项"❌"都代表"要花很多时间",有时候只是"还没有人花五分钟做"。

## 6. HuggingFace 发布 —— 现在还不适用,以及为什么

对照 [02 号文件](02-huggingface-release.md)整篇的前提:HuggingFace Hub 发布的是**模型权重**或者**数据集**。这个项目现在:

- 没有一个"训练好、要长期复用"的模型 checkpoint——`LearnedWorldModel`(频次估计表格)和 `NeuralEnsembleWorldModel`(小型神经网络)都是 pilot 脚本运行过程中几分钟内重新训练出来的中间产物,不是被当成最终交付物来管理的
- 没有一个被专门整理、打算独立分发的数据集——训练数据是 `collect_rollout_data` 在运行时用固定种子现场生成的合成轨迹,不是一份需要独立托管的静态文件

**这不代表以后用不上第 2 号文件**——`PROTOCOL.md` 自己写得很清楚:"下一步如果方向选定,需要把同一套测量协议搬到真实 world model checkpoint(如 DreamerV3、TD-MPC2)和真实任务(DMControl/Atari)上重新做一遍"。真的走到那一步、真的训练出规模有意义的 checkpoint 时,第 2 号文件的 model card 写法、`.safetensors` 格式建议、上传前检查清单会直接用得上。现在这个阶段,提前判断"用不上"本身也是一种诚实的清单结论,不是回避。

## 7. 学术社区互动 —— 现在还不适用,以及为什么

对照 [03 号文件](03-community-interaction.md):论文还没有投出去,更谈不上被接收、被公开、被人读到——第 3 号文件整篇讨论的"论文放出后怎么办",这个项目现在还没有到这个时间点。这一项的清单结论同样是诚实的"不适用",不是遗漏。

---

## 8. 汇总:一份可勾选的发布清单

把上面 7 节的真实结论,收成一份读者能直接照着走一遍的清单——前半部分是这个项目现在的真实状态(截至 2026-07-25),后半部分是任何一个新项目都可以直接拿去用的通用版本。

**这个项目现在的真实状态**:

- [x] 研究方法论文档完整(`PROTOCOL.md`,含"为什么这样设计"+"范围声明")
- [x] 结果记录完整(3 份 `RESULTS*.md`)
- [x] 随机种子固定 + 多种子方差报告(`SEEDS = [0, 1, 2, 3, 4]`,3 个文件一致)
- [ ] README(项目根目录和 `eval-protocol/` 均缺失)
- [ ] LICENSE(建议 MIT,尚未添加)
- [ ] 依赖清单(实际只需要 `numpy` 一行,尚未添加)
- [ ] 确切的"怎么跑"操作步骤(方法论文档不等于操作步骤,目前只有前者)
- [ ] HuggingFace 发布(当前阶段不适用,待有真实 world model checkpoint 后参照 [02 号文件](02-huggingface-release.md)重新评估)
- [ ] 社区互动(当前阶段不适用,论文公开后参照 [03 号文件](03-community-interaction.md)重新评估)

**任何项目都可以直接拿去用的通用清单**(把上面具体案例还原成通用条目):

- [ ] README 写清楚:是什么、怎么装、怎么跑出论文里的结果、结果长什么样([01 号文件第 2 节](01-open-source-code-release.md))
- [ ] 依赖清单存在且核心依赖锁定版本([01 号文件第 3 节](01-open-source-code-release.md))
- [ ] 多个随机种子独立跑过,报告 mean±std,不是单次结果([01 号文件第 4 节](01-open-source-code-release.md))
- [ ] LICENSE 文件存在,且和代码技术栈/目标匹配([01 号文件第 5 节](01-open-source-code-release.md))
- [ ] (可选,camera-ready 前后)给发布的代码版本打一个 DOI([01 号文件第 6 节](01-open-source-code-release.md))
- [ ] 如果有模型/数据集要发布,model card / dataset card 按官方结构写清楚,上传前本地检查过文件结构([02 号文件](02-huggingface-release.md))
- [ ] 论文公开当天,该"挂号"的地方都挂了号([03 号文件第 1 节](03-community-interaction.md))
- [ ] 想清楚宣传帖怎么写,不夸大不模糊([03 号文件第 2 节](03-community-interaction.md))
- [ ] 想清楚收到"复现不出来"反馈时,默认对方遇到了真实问题,准备好怎么回应([03 号文件第 3 节](03-community-interaction.md))

---

*本文件的项目状态检查全部在 2026-07-25 用真实代码扫描 `research/world-model-imagination-controller/` 得到,随项目推进会过期——这是设计使然,不是缺陷,参照第 0 节的说明重新读。*
*创建:2026-07-25*
