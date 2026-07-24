# 02 · HuggingFace Release —— Model Card / Dataset Card 怎么写、怎么让人愿意用你发布的东西

> 总览见 [00-roadmap.md](00-roadmap.md)
> 前提:已经看过 [01-open-source-code-release.md](01-open-source-code-release.md)——那篇讲的是代码仓库(README/License/依赖锁定)本身该长什么样。这篇讲另一件常被忽略的事:如果你的工作里有训练出来的模型权重、或者你自己整理/生成的数据集,只放在 GitHub 上(通常是一个几百 MB 到几十 GB 的文件,或者干脆没放,只在 README 里写"联系作者索取"),读者很难真正用起来。HuggingFace Hub 是目前 AI 研究领域事实上的标准发布地——这篇讲怎么把它做对。

---

## 0. 这篇文章是怎么验证的(先说清楚,这是全篇最重要的一节)

这是全系列里验证颗粒度最不均匀的一篇,必须在最前面把每一类内容分别标清楚,不能笼统地说"已验证"。

- **已确认这台机器没有真实可用的 HuggingFace 账号/token**:用 `huggingface_hub.HfApi().whoami()` 现场验证过,真实返回 `LocalTokenNotFoundError`(第 6 节会展示完整代码和真实计时结果)。环境变量 `HF_TOKEN`、`HUGGING_FACE_HUB_TOKEN` 也现场确认过都没有设置。这意味着**任何需要写权限的操作(建仓库、上传文件)在这台机器上都无法真实执行,更不能冒充执行过**。
- **本地语法/元数据构造与解析:真实跑过,可重复验证**。第 2、3、4、5 节的所有 Python 代码,都是用本机 `.venv` 里的 `huggingface_hub==1.18.0` 真实执行的,不需要网络、不需要账号——`ModelCardData`/`DatasetCardData`/`ModelCard`/`DatasetCard` 这几个类,负责的是"按 HuggingFace Hub 认可的 YAML 格式,在本地构造和解析一份 model card/dataset card 的元数据",这部分工作全部发生在你自己的电脑上,和有没有账号完全无关。这些代码块都在 `_verify_md.py 02-huggingface-release.md` 的自动化验证范围内。
- **有一处真实、需要网络、但不需要账号的例外**:`ModelCard.validate()` 这个方法会真的向 `https://huggingface.co/api/validate-yaml` 发一个网络请求,校验你的 YAML 元数据是否符合 Hub 的 schema——这个端点是公开的、只读性质的校验服务,**不需要登录**。第 6 节会如实展示这次真实发生过的网络往返(真实耗时、真实返回了 Hub 服务端当前认可的完整 license 标识符列表),并且清楚说明:这不是"上传",这是"问一下 Hub 服务器这份元数据写得对不对",两者是完全不同性质的操作,不要混为一谈。这段代码**不放进自动化验证范围**——原因和判断依据在第 6 节详细说明。
- **真正的"上传到 Hub"这一步,本文没有做,也没有冒充做过**:创建仓库(`create_repo`)、推送文件(`upload_file`/`upload_folder`)这些需要写权限的操作,本文只描述官方文档记录的调用方式,不在本机执行、不贴"运行成功"的假输出。哪些内容属于这一类,每处都会用加粗标注"未验证(需要账号/token)"。
- **调研来源**:HuggingFace 官方文档([Model Card Guidebook](https://huggingface.co/docs/hub/en/model-card-guidebook)、[Annotated Model Card](https://huggingface.co/docs/hub/model-card-annotated)、[Dataset Cards](https://huggingface.co/docs/hub/datasets-cards)、[Paper Pages](https://huggingface.co/docs/hub/en/paper-pages))真实检索获取,不是凭旧印象转述。

---

## 1. 为什么要发布到 HuggingFace Hub,不只是挂个 GitHub 链接

**为什么需要这个 / 不会有什么后果:**

[01 号文件](01-open-source-code-release.md)已经讲过"代码能跑"和"代码可发布"的区别。模型权重和数据集有一个 GitHub 处理不好的额外问题:**体积**。一个几百 MB 到几十 GB 的模型 checkpoint,塞进 git 仓库会让每一次 `clone` 都变得极慢([daily-toolkit-deep-dive/04-git-collaboration-workflow.md 第 4 节](../daily-toolkit-deep-dive/04-git-collaboration-workflow.md)已经讲过为什么大文件不该进 git,这里不重复);放进某个云盘链接、写"邮件联系我要权重",对读者来说是额外的、容易在几年后失效的摩擦(链接过期、你毕业换了邮箱、云盘账号到期)。

HuggingFace Hub 解决的正是"模型/数据集这类大文件的托管 + 发现"问题,而且它不只是一个文件托管服务——**Hub 上的 README.md 会被渲染成结构化的 model card/dataset card,读者能直接看到 license、任务类型、依赖的数据集这些元数据**,还能被 Hub 的搜索/筛选功能索引到,这是一个静态 GitHub README 做不到的。另外一个容易被忽略的联动:[HuggingFace Papers 页面](https://huggingface.co/docs/hub/en/paper-pages)会自动抓取"model card/dataset card 里链接了哪篇 arXiv 论文"这个信息——**只要你的 model card 里出现一个 arXiv 链接,Hub 会自动把这个仓库和对应的论文页面关联起来,不需要额外提交**,论文页面上会自动汇总"这篇论文关联了哪些模型/数据集/Space",第 7 节会展开说这件事具体怎么发生。

不会有什么后果——如果不这么做:模型/数据集本身依然可以分享(比如继续用云盘链接),但会错过 Hub 的可发现性和"关联论文自动聚合"这两个好处,对于一篇希望被更多人用起来的论文,这是白白放弃的曝光机会。

**环境要求:** 无,这一节是动机建立。

**背后发生了什么 / 常见坑 / 自测清单:** 这一节偏动机说明,具体操作和坑点在后面几节按类型分别展开(model card 在第 2-3 节,dataset card 在第 4 节,上传前检查在第 5 节),这里不重复列。

---

## 2. Model Card 怎么写 —— 用 `huggingface_hub` 库真实构造 + 本地解析校验

**为什么需要这个 / 不会有什么后果:**

Model card 本质上就是模型仓库根目录的 `README.md`,但开头有一段 YAML "front matter"(用 `---` 包起来的元数据块)——这段 YAML 不是随便写的自由格式,Hub 会解析它来驱动搜索筛选、显示 license 徽章、关联数据集页面。HuggingFace 官方文档明确写了 model card 应该说明:模型本身是什么、预期用途和已知局限(包括偏见和伦理考量)、训练参数、用了哪些数据集训练、评测结果([Model Card Guidebook](https://huggingface.co/docs/hub/en/model-card-guidebook))。不写清楚的后果很直接:别人不知道你的模型能不能用在他们的场景里,更谈不上正确复现你的评测数字。

**环境要求:** `.venv` 里的 `huggingface_hub==1.18.0`,不需要网络、不需要账号。

**一步步跟着做:**

`huggingface_hub` 库提供 `ModelCardData`(负责 YAML 元数据部分)和 `ModelCard`(负责整份文档,包含 YAML + 正文)两个类,两者都可以完全离线使用:

```python
from huggingface_hub import ModelCard, ModelCardData

card_data = ModelCardData(
    license="mit",
    library_name="pytorch",
    tags=["reinforcement-learning", "world-models", "imagination-controller"],
    datasets=["dmcontrol-benchmark-suite"],
    pipeline_tag="reinforcement-learning",
)
yaml_text = card_data.to_yaml()
print(yaml_text)

full_text = f"---\n{yaml_text}\n---\n\n# Imagination Budget Controller (demo card)\n\nDemo body.\n"
card = ModelCard(full_text)

assert card.data.license == "mit"
assert card.data.tags == ["reinforcement-learning", "world-models", "imagination-controller"]
assert card.data.datasets == ["dmcontrol-benchmark-suite"]
assert card.data.pipeline_tag == "reinforcement-learning"
print("OK: metadata written offline round-trips through ModelCard parsing unchanged")
```

真实运行结果(`_verify_md.py` 独立验证通过):`ModelCardData.to_yaml()` 把参数按字母序渲染成标准 YAML,`ModelCard` 能把这段 YAML + 正文重新解析回结构化字段,构造和解析这一去一回全部在本地完成,不发一个网络包。

**评测结果也可以写成结构化元数据**(而不只是正文里的一个表格),用 `EvalResult`——这样 Hub 能在仓库页面直接渲染出一块"评测结果"卡片:

```python
from huggingface_hub import ModelCard, ModelCardData, EvalResult

# 下面 metric_value=42.0 是演示格式用的占位数字,不是项目的真实实验结果——
# 这个系列的教学案例只借用真实项目的场景设定,不代写/冒用真实结论(设计文档里的纪律)。
card_data = ModelCardData(
    license="mit",
    tags=["world-models", "reinforcement-learning"],
    model_name="imagination-budget-controller-demo",
    eval_results=[
        EvalResult(
            task_type="reinforcement-learning",
            dataset_type="dmcontrol-walker-walk",
            dataset_name="DMControl Walker-Walk (placeholder)",
            metric_type="episode_return",
            metric_value=42.0,
        )
    ],
)
text = "---\n" + card_data.to_yaml() + "\n---\n\n# Demo\n"
card = ModelCard(text)
assert card.data.eval_results[0].dataset_name == "DMControl Walker-Walk (placeholder)"
assert card.data.eval_results[0].metric_type == "episode_return"
print("OK: eval_results metadata block parses back with the fields intact")
```

**背后发生了什么:** `ModelCardData` 底层就是一个会被序列化成 YAML 的数据类,`to_yaml()` 只是调用了标准 YAML 序列化;`EvalResult` 序列化出来其实是 Hugging Face 定义的 `model-index` 规范(参照社区已有的 model evaluation 元数据标准),渲染到 Hub 网页上会变成结构化的"结果表格"展示,而不是纯文本——这也是为什么 model card 建议尽量用结构化元数据而不是只把结果写进正文 Markdown:结构化的部分才能被 Hub 的界面/搜索利用。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 只写了正文里的结果表格,没填 `eval_results` 元数据 | 不知道 Hub 会渲染结构化评测结果这个功能 | 两者不冲突,可以同时保留——正文表格给人看,`eval_results` 给 Hub 界面和第三方工具解析 |
| model card 只写了"how to use"代码示例,没写训练数据/训练参数 | 把 model card 当成了纯粹的"调用示例",忽略了它作为文档的另一半职责 | 参照官方模板把"Training Data"/"Training Procedure"这两节也补上,即使写得简短 |
| 不知道该填哪些 `tags` | 没有参照 Hub 上同类模型的常见标签 | 去 Hub 上搜几个同任务类型的模型,看它们用了哪些 tags,不需要自己凭空发明 |

**自测清单:**

- [ ] 能说出 model card 的 YAML 元数据和正文 Markdown 各自的作用(元数据驱动搜索/筛选/结构化展示,正文给人看)
- [ ] 能独立写一段 `ModelCardData(...)`,填上 license/tags/library_name 三个基本字段
- [ ] 知道 `EvalResult` 存在,以及它和正文里手写的结果表格是互补关系不是二选一

---

## 3. 一个真实撞到的坑 —— `tags`/`datasets` 字段传错类型,不会报错

**为什么需要这个 / 不会有什么后果:**

写这篇文章的过程中,真实测试 `ModelCardData` 的字段校验行为时撞到了一个值得记录的坑:**如果手滑把 `tags` 参数传成一个裸字符串(忘了包成 list),`huggingface_hub` 不会报错**——它会把这个字符串当成一个可迭代对象,**逐个字符拆开、再按内部的去重逻辑丢掉重复字符**,变成一串不重复的单字符标签(具体机制下面"背后发生了什么"会给出真实读到的源码)。更麻烦的是:**`datasets` 字段遇到同样的错误,表现却完全不一样**——它不会拆字符也不会去重,而是把整个字符串原样保留(不是 list,是一个裸字符串)。同一类型的输入错误,两个字段的静默失败方式还不一样,这正是这类坑难查的原因:你在 Hub 网页上会看到一堆莫名其妙的单字符标签,或者一个格式不对的 `datasets` 字段,但代码本身完全没有报错提示你哪里错了。

**环境要求:** `.venv` 里的 `huggingface_hub==1.18.0`,离线可复现。

**一步步跟着做:**

```python
from huggingface_hub import ModelCardData

# correct usage: tags must be a list
correct = ModelCardData(license="mit", tags=["world-models"])
assert correct.tags == ["world-models"]

# a real footgun: passing a bare string to `tags=` instead of a list
wrong_tags = ModelCardData(license="mit", tags="world-models")
print("tags='world-models' (bare string) silently became:", wrong_tags.tags)
# huggingface_hub internally calls _to_unique_list(tags), which iterates the input and
# de-duplicates it (order-preserving). For a bare string this means: split into
# characters, THEN drop repeats. "world-models" has repeated o/d/l, so the result is
# the 9 unique characters in first-seen order, not all 12 characters.
assert wrong_tags.tags == ["w", "o", "r", "l", "d", "-", "m", "e", "s"]

# the SAME class of mistake on the `datasets` field fails differently: `datasets` is stored
# as-is with no list/unique wrapping at all, so a bare string is kept as a bare string.
wrong_datasets = ModelCardData(license="mit", datasets="my-dataset")
print("datasets='my-dataset' (bare string) silently became:", repr(wrong_datasets.datasets))
assert wrong_datasets.datasets == "my-dataset"
assert not isinstance(wrong_datasets.datasets, list)

print("OK: same mistake, two different silent failure modes -- neither one raises an error")
```

真实运行结果(`_verify_md.py` 独立验证通过):`tags="world-models"` 被拆成 `['w', 'o', 'r', 'l', 'd', '-', 'm', 'e', 's']`——**9 个不重复的字符**,不是全部 12 个字符(`world-models` 里 `o`/`d`/`l` 各出现两次,重复的都被丢掉了);`datasets="my-dataset"` 被原样保留成字符串 `'my-dataset'`,类型不是 `list`。

**背后发生了什么:** 直接读了本机安装的 `huggingface_hub==1.18.0` 源码(`ModelCardData.__init__` 里 `self.tags = _to_unique_list(tags)` 这一行),`_to_unique_list` 的实现是:

```python
def _to_unique_list(tags):
    if tags is None:
        return tags
    unique_tags = []  # make tags unique + keep order explicitly
    for tag in tags:
        if tag not in unique_tags:
            unique_tags.append(tag)
    return unique_tags
```

这段代码假设传入的 `tags` 已经是一个字符串列表,`for tag in tags` 逐个取出"每一个标签"、顺手去重。Python 里字符串本身就是一个可迭代对象(`for c in "abc"` 会依次拿到 `'a'`、`'b'`、`'c'`)——当你传入一个裸字符串,这段函数会把每一个**字符**当成一个"标签"逐个处理,连带把重复字符去掉,这是 Python 语言本身对"可迭代对象"一视同仁的行为,不是这个库故意设计的陷阱,但结果确实反直觉。反观 `datasets` 字段,`ModelCardData.__init__` 里对应的是 `self.datasets = datasets`——没有任何包装或校验,原样存下来,这就是为什么同一类输入错误在两个字段上表现不一样:不是巧合,是两个字段背后调用了不同的(或者说,一个有、一个没有)处理逻辑。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| Hub 网页上模型标签是一串毫无意义的单字符 | `tags=` 传了字符串而不是 list | 检查代码里是不是漏了方括号,`tags=["x"]` 不是 `tags="x"` |
| `datasets` 字段在 Hub 页面没有正确关联到数据集 | `datasets=` 传了字符串而不是 list | 同上,养成"这几个字段永远传 list,哪怕只有一个元素"的习惯 |
| 类似的字段(`language`、`metrics`)有没有同样的坑没有测过 | 这次只实测了 `tags`/`datasets` 两个字段 | 如实说明:这篇没有逐一测过 `ModelCardData` 的每一个字段,只报告了真实测到的这两个;写自己的 model card 时,新字段建议先用离线的 `ModelCardData(...)` 构造+打印确认一次,不要凭直觉假设行为一致 |

**自测清单:**

- [ ] 能复述这个坑的具体表现:`tags` 传字符串会被拆成不重复的单字符,`datasets` 传字符串会被原样保留成非 list
- [ ] 知道为什么会这样(字符串本身是可迭代对象;`tags` 字段内部调用 `_to_unique_list` 逐个取出可迭代对象里的元素并去重,`datasets` 字段没有任何包装,原样存下来)
- [ ] 养成"提交前用 `print(card_data.to_yaml())` 肉眼确认一遍"的习惯,而不是假设参数一定被正确处理

---

## 4. Dataset Card 怎么写(如果你发布的是数据,不只是模型)

**为什么需要这个 / 不会有什么后果:**

如果论文附带发布一个自己构造/整理的数据集(哪怕只是"跑 pilot 用的合成环境采样数据"这种规模不大的数据),同样应该有一份 dataset card,原因和 model card 类似:没有文档的数据集,别人不知道这份数据是怎么来的、有没有已知的偏差或局限、能不能用在他们的场景里。HuggingFace 官方文档说明 dataset card 的设计直接借鉴了 Mitchell et al. 2018 提出的 Model Card 概念,目的是"促进负责任的使用,并让用户了解数据集里可能存在的偏差"([Dataset Cards 官方文档](https://huggingface.co/docs/hub/datasets-cards))。

**环境要求:** 同上,`.venv` 里的 `huggingface_hub`,离线可复现。

**一步步跟着做:**

```python
from huggingface_hub import DatasetCard, DatasetCardData

card_data = DatasetCardData(
    license="cc-by-4.0",
    language=["en"],
    pretty_name="Imagination Budget Rollout Traces (toy)",
    task_categories=["reinforcement-learning"],
    size_categories=["1K<n<10K"],
)
text = "---\n" + card_data.to_yaml() + "\n---\n\n# Dataset Card\n\nDemo.\n"
card = DatasetCard(text)
assert card.data.license == "cc-by-4.0"
assert card.data.pretty_name == "Imagination Budget Rollout Traces (toy)"
assert card.data.task_categories == ["reinforcement-learning"]
print("OK: DatasetCardData round-trips through DatasetCard the same way ModelCardData does")
```

真实运行结果(`_verify_md.py` 独立验证通过):和 model card 完全一样的离线构造/解析流程。

**内容上该写什么**:官方文档给出的标准结构是 5 个部分——Dataset Description(数据集是什么)、Dataset Structure(字段/格式/划分)、Dataset Creation(数据怎么来的、标注流程)、Considerations for Using the Data(已知局限/偏差)、Additional Information(许可证/引用方式)。如果某部分暂时没法填,官方文档给出的建议做法是直接写 `[More Information Needed]`,而不是留空或者编一个——这和这个仓库一贯的"诚实标注,不冒充"纪律是同一个精神。

**背后发生了什么:** `DatasetCardData` 和 `ModelCardData` 是同一套设计模式的两个平行实现,字段不同(数据集关心 `task_categories`/`size_categories` 这类描述"这是什么数据"的元数据,模型关心 `pipeline_tag`/`library_name` 这类描述"这是什么模型"的元数据),但"YAML 元数据 + Markdown 正文"这个整体结构、"本地构造/解析不需要网络"这个特性完全一致——学会一个,另一个不需要重新学一遍设计思路。有一个值得注意的字段类型差异:`ModelCardData.license` 是单个字符串(一个模型通常只有一个许可证),而 `DatasetCardData.license` 可以是字符串或字符串列表(数据集有时候是多个来源合并的,可能对应不止一个许可证)——这个不对称在两边的函数签名里都能直接看到,不是猜测。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 用 model card 的字段名去写 dataset card(比如写了 `pipeline_tag`) | 以为两者字段通用 | 两者字段不完全相同,`DatasetCardData`/`ModelCardData` 各自有一套,写之前对照各自的构造函数签名 |
| 不确定某部分内容能不能公开(比如标注流程涉及众包平台的具体细节) | 数据集创建过程有些细节可能涉及第三方协议 | 参照官方"填不了就写 `[More Information Needed]`"的建议,不要为了填满而编造 |
| license 字段不知道该填字符串还是列表 | 没注意到 `DatasetCardData.license` 支持两种类型 | 单一来源的数据用字符串,多来源合并的数据集可以用列表,两种写法都合法 |

**自测清单:**

- [ ] 能说出 dataset card 官方建议的 5 个组成部分
- [ ] 能说出 `ModelCardData.license` 和 `DatasetCardData.license` 在字段类型上的一个具体差异
- [ ] 遇到暂时无法填写的部分,知道官方建议怎么诚实处理(`[More Information Needed]`,不是编一个)

---

## 5. 上传前的本地文件结构检查

**为什么需要这个 / 不会有什么后果:**

在真正执行上传(需要账号,本文没有做,见第 6 节)之前,先在本地把目录结构理顺,能避免"传上去发现漏了配置文件"这种来回折腾。一个典型的 HuggingFace 模型仓库,通常至少包含:一份 model card(`README.md`)、模型权重文件、一份 `config.json`(记录模型结构/超参数,供 `from_pretrained` 之类的加载函数使用)。权重格式上,Hub 生态目前明确推荐 `.safetensors` 而不是旧式的 PyTorch `.bin`(pickle 格式)——`.safetensors` 不能执行任意代码,加载更安全,这也是为什么很多新模型页面会看到"这个仓库同时有 `.bin` 和 `.safetensors`,后者是转换后的安全版本"这种过渡状态。

**环境要求:** 无网络需求,纯本地文件系统检查。

**一步步跟着做:**

```python
import tempfile
from pathlib import Path


def check_model_repo_structure(model_dir: Path) -> dict:
    """本地上传前检查:一个要推到 HF Hub 的模型目录,通常应该有哪些东西。"""
    names = {p.name for p in model_dir.iterdir() if p.is_file()}
    return {
        "has_readme_or_model_card": "README.md" in names,
        "has_weights": any(n.endswith((".safetensors", ".bin", ".pt", ".ckpt")) for n in names),
        "has_config": "config.json" in names,
        "weights_are_safetensors": any(n.endswith(".safetensors") for n in names)
        and not any(n.endswith(".bin") for n in names),
    }


with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    (root / "README.md").write_text("---\nlicense: mit\n---\n# demo\n", encoding="utf-8")
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"fake-weights")
    report = check_model_repo_structure(root)
    print(report)
    assert report == {
        "has_readme_or_model_card": True,
        "has_weights": True,
        "has_config": True,
        "weights_are_safetensors": True,
    }
print("OK: structure checker correctly reads a minimal well-formed local model directory")
```

真实运行结果(`_verify_md.py` 独立验证通过):这是一个非常粗略的本地检查工具,和第 1 号文件第 2 节的仓库扫描器是同一种思路——只做文件存在性检查,不理解内容语义,发布前仍然建议真的用 `from_pretrained` 之类的加载函数在本地实测一遍能不能加载成功。

**大文件怎么处理,交叉引用而不重复**:HF Hub 底层用 Git LFS 存储大文件,概念上和 [daily-toolkit 04 号文件第 4 节](../daily-toolkit-deep-dive/04-git-collaboration-workflow.md)讲的"哪些文件不该直接进普通 git 历史"是同一个问题的另一种解法(LFS 是"进 git 但用指针+外部存储"而不是"完全不进 git")——这篇不重复 LFS 底层原理,只强调一点:HF Hub 会自动检测常见的大文件格式(权重/数据集文件)并转成 LFS 追踪,大多数时候不需要你手动配置,但如果上传报错提示文件过大,官方文档建议检查 `.gitattributes` 里的 LFS 规则是否覆盖了你的文件类型。

**背后发生了什么:** 官方推荐 `.safetensors` 不是审美偏好——旧式 `.bin` 文件用 Python 的 `pickle` 序列化,加载一个 `.bin` 文件本质上是在执行文件里编码的任意代码(这是 pickle 格式的已知安全问题),如果你从不完全信任的来源下载一个 `.bin` 权重文件并加载,理论上存在被注入恶意代码的风险;`.safetensors` 从设计上就不支持任意代码执行,只存张量数据本身,这也是为什么它现在是 Hub 生态的默认推荐格式。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 只上传了权重文件,没有 `config.json` | 忘了模型加载通常还需要结构/超参数信息 | 上传前对照自己训练脚本里保存 checkpoint 的逻辑,确认配置也一并导出了 |
| 权重是 `.bin` 格式 | 用了较旧的 `torch.save` 直接保存方式 | 可以用 `safetensors` 库转换,或者训练框架本身如果支持,直接用 safetensors 格式保存 |
| 不确定上传大文件会不会超时/失败 | 网络环境或者文件确实很大 | 官方 SDK 的上传函数本身有分块上传/断点续传的设计,这篇没有真实网络环境去验证这一点,如实标注未验证 |

**自测清单:**

- [ ] 能说出一个典型 HF 模型仓库通常包含哪三类文件(model card、权重、config)
- [ ] 能说出为什么 `.safetensors` 比 `.bin` 更被推荐(安全性,不是性能)
- [ ] 知道大文件在 HF Hub 上是通过 Git LFS 类似机制处理的,和 [daily-toolkit 04 号文件](../daily-toolkit-deep-dive/04-git-collaboration-workflow.md)讲的"大文件不该进普通 git 历史"是同一个问题的不同解法

---

## 6. 诚实边界 —— 本机没有真实账号,以及一次真实的、需要网络但不需要账号的校验

**为什么需要这个 / 不会有什么后果:**

前面 5 节的代码全部离线可跑,容易让人以为"这就是 HuggingFace 发布的全部"——不是。model card 写对格式,离上传成功还差着"有没有权限往那个仓库写东西"这一整层。这一节把这条边界具体划清楚,不含糊带过。

**环境要求:** `.venv` 里的 `huggingface_hub`;本机确认没有配置任何 HF 凭据。

**一步步跟着做:**

**第一步,确认没有账号,而且这个确认本身是可以自动化验证的(纯本地检查,不发网络包)：**

```python
import time
from huggingface_hub import HfApi
from huggingface_hub.errors import LocalTokenNotFoundError

t0 = time.time()
try:
    HfApi().whoami()
    raise SystemExit("unexpectedly authenticated -- this machine should have no HF token configured")
except LocalTokenNotFoundError:
    elapsed = time.time() - t0
    print(f"whoami() raised LocalTokenNotFoundError in {elapsed:.4f}s (a local check, not a network round trip)")
    assert elapsed < 0.5  # 这是本地凭据文件检查,应该接近瞬间返回,不是网络超时
    print("OK: confirmed no HF token/account configured on this machine")
```

真实运行结果(`_verify_md.py` 独立验证通过):`LocalTokenNotFoundError` 在 0.0008 秒内被抛出——这个耗时本身就是证据:`huggingface_hub` 在真正尝试联网调用 `/whoami-v2` 接口之前,会先在本地找有没有存过 token(检查环境变量 `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`,以及本地凭据文件),找不到就直接在本地报错、根本不会真的发网络请求——错误类名 `LocalTokenNotFoundError` 里的 "Local" 不是随便起的。

**第二步(不属于自动化验证范围,原因见下):真实测试了 `ModelCard.validate()`——这个方法会真的发网络请求。** 读 `huggingface_hub` 源码能看到它的文档字符串写着"Using this function requires access to the internet"([源码里的说明](https://github.com/huggingface/huggingface_hub),`repocard.py` 的 `validate` 方法),它会 POST 到 `https://huggingface.co/api/validate-yaml`——这是一个**不需要登录**的公开校验端点(检查你的 YAML 是不是符合 Hub 认可的 schema,比如 `license` 字段的值是不是 Hub 认识的合法标识符),和"创建仓库"/"上传文件"这类需要账号的写操作是完全不同性质的调用。这次撰写过程中真实执行过,记录如下(以下是真实捕获的代码+输出,不是编造的示例):

```text
>>> from huggingface_hub import ModelCard, ModelCardData
>>> import time
>>> card_data = ModelCardData(license="mit", tags=["a"])
>>> text = "---\n" + card_data.to_yaml() + "\n---\n\n# Title\n"
>>> card = ModelCard(text)
>>> t0 = time.time(); card.validate(); print(f"passed, elapsed={time.time()-t0:.3f}s")
passed, elapsed=1.129s

>>> bad_card_data = ModelCardData(license="this-is-not-a-real-license-xyz", tags=["a"])
>>> bad_text = "---\n" + bad_card_data.to_yaml() + "\n---\n\n# Title\n"
>>> bad_card = ModelCard(bad_text)
>>> t0 = time.time()
>>> bad_card.validate()
Traceback (most recent call last):
    ...
ValueError: - Error: "license" must be one of [apache-2.0, mit, openrail, bigscience-openrail-m,
creativeml-openrail-m, bigscience-bloom-rail-1.0, bigcode-openrail-m, afl-3.0, artistic-2.0, bsl-1.0,
bsd, bsd-2-clause, bsd-3-clause, bsd-3-clause-clear, c-uda, cc, cc0-1.0, cc-by-2.0, cc-by-2.5, cc-by-3.0,
cc-by-4.0, cc-by-sa-3.0, cc-by-sa-4.0, cc-by-nc-2.0, cc-by-nc-3.0, cc-by-nc-4.0, cc-by-nd-4.0,
cc-by-nc-nd-3.0, cc-by-nc-nd-4.0, cc-by-nc-sa-2.0, cc-by-nc-sa-3.0, cc-by-nc-sa-4.0, cdla-sharing-1.0,
cdla-permissive-1.0, cdla-permissive-2.0, wtfpl, ecl-2.0, epl-1.0, epl-2.0, etalab-2.0, eupl-1.1,
eupl-1.2, agpl-3.0, gfdl, gpl, gpl-2.0, gpl-3.0, lgpl, lgpl-2.1, lgpl-3.0, isc, h-research,
intel-research, lppl-1.3c, ms-pl, apple-ascl, apple-amlr, mpl-2.0, odc-by, odbl, openmdw-1.0,
openmdw-1.1, openrail++, osl-3.0, postgresql, ofl-1.1, ncsa, unlicense, zlib, pddl, lgpl-lr,
deepfloyd-if-license, fair-noncommercial-research-license, llama2, llama3, llama3.1, llama3.2,
llama3.3, llama4, grok2-community, gemma, unknown, other, array]
elapsed=1.144s
```

**为什么这段没有放进 `_verify_md.py` 的自动化验证范围**:上面第 2-5 节所有 `python` 代码块,每次运行 `_verify_md.py` 时都会被重新真实执行一遍——这对纯本地逻辑没问题,但对一个依赖外部网络服务当前可用性的调用不合适:如果未来某次运行时网络不通,或者 Hub 服务端点变化,会让"这份文档的代码是不是还正确"这个问题和"这台机器现在能不能连上外网"这个完全无关的问题混在一起。所以这段展示为一次性、带时间戳的真实捕获记录(和 [daily-toolkit-deep-dive 里 03 号 SSH 文件、04 号 git 文件处理"真实但依赖外部环境"内容的方式一致](../daily-toolkit-deep-dive/03-ssh-and-remote-servers.md)),不伪装成可以无限次重新验证的确定性断言。

**背后发生了什么:** 合法 license 标识符列表(上面 `ValueError` 里那一长串)是 **Hub 服务端真实返回的**,不是这篇文章编的——这份列表包含 `mit`、`apache-2.0`、多个 `cc-by-*` 变体、`gpl-*` 系列等,和 [01 号文件第 5 节](01-open-source-code-release.md)讨论的 MIT/Apache 2.0/CC-BY 分类完全对得上——这也印证了那一节"代码许可证和内容许可证是两套体系,但 Hub 的 `license` 元数据字段两种都收"这个说法有真实依据,不是猜测。

**真正的"上传"需要什么、这篇为什么没有做:** 创建仓库要调用 `HfApi().create_repo(repo_id, ...)`,推送文件要调用 `upload_file`/`upload_folder`——这两类操作都要求一个有写权限的 token。本机 `whoami()` 已经证明没有这样的 token,如果强行调用 `create_repo`,预期结果是认证失败报错;**这篇选择完全不去调用这类写操作**(哪怕是"预期会失败"的调用也不去触发),因为唯一的价值只是再一次证明"没有 token 就是不能写",这个事实上面的 `whoami()` 测试已经证明过一次,没必要重复,更不能给读者留下"这里似乎也做过上传尝试"的印象。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 以为 `ModelCard.validate()` 是纯本地的语法检查 | 方法名听起来像本地校验,实际上文档字符串里明确写了需要联网 | 用之前读一下这个方法的 docstring/源码,不要凭方法名猜测它是否需要网络 |
| 真正要上传时,`create_repo`/`push_to_hub` 报认证错误 | 没有配置 token,或者 token 没有写权限 | 去 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 生成一个 token(需要真实账号),用 `huggingface-cli login` 或者 `HfApi(token=...)` 提供,这篇没有真实账号无法演示这一步 |
| 把"本地元数据构造没报错"当成"上传一定会成功" | 混淆了语法正确和权限充足这两件事 | 语法正确是上传成功的必要条件,不是充分条件——权限、网络、文件大小限制等其他因素都可能独立导致上传失败 |

**自测清单:**

- [ ] 能说出 `whoami()` 报 `LocalTokenNotFoundError` 是本地检查还是网络请求,以及怎么用耗时数据证明这一点
- [ ] 能说出 `ModelCard.validate()` 和"上传/创建仓库"的本质区别(前者不需要账号,后者需要)
- [ ] 能复述这篇文章诚实标注验证颗粒度的具体做法:本地逻辑离线自动验证,一次性网络调用作为带时间戳的真实记录展示,真正的写操作完全不触碰、不冒充

---

## 7. 和 HuggingFace Papers 页面的联动(选读)

**为什么需要这个 / 不会有什么后果:**

论文正式挂上 arXiv 之后,`hf.co/papers/<arxiv-id>` 会自动出现一个对应页面——不需要手动提交。触发机制是:**只要有任何一个 model card / dataset card / Space 的 README 里出现了这篇论文的 arXiv 链接,Hub 就会抓取这个 arXiv ID,自动把这个仓库和论文页面关联起来**([Paper Pages 官方文档](https://huggingface.co/docs/hub/en/paper-pages))。这意味着第 2 节 model card 里如果补一句"论文见 `arxiv.org/abs/xxxx.xxxxx`",不只是给读者一个链接,还会让你的模型出现在论文页面的"关联模型"列表里,论文页面本身也能被搜索/浏览到。

**一步步跟着做(按官方文档描述,本机没有真实 arXiv 论文可以端到端验证这个联动,如实标注未做真实验证):**

1. Model card / dataset card 的 YAML 头部或正文里,放一个指向 HF 论文页面或者 arXiv 摘要页/PDF 的链接。
2. Hub 会自动提取其中的 arXiv ID,加进这个仓库的 tags(格式是 `arxiv:<PAPER_ID>`)。
3. 点这个自动生成的 tag,能跳到论文页面,看到"哪些模型/数据集/Space 引用了同一篇论文"的汇总列表。
4. 如果论文页面还没被 Hub 索引过,可以直接访问 `hf.co/papers/<arxiv-id>` 触发索引,或者在 Papers 搜索页手动搜索论文标题/arXiv ID。
5. 作者本人可以在自己的论文页面点"claim authorship"认领作者身份(会跳转到审核流程,官方文档说明由 Hub 管理团队人工审核)。

**背后发生了什么:** 这套联动的关键在于"只需要一个 arXiv 链接"这个低门槛设计——不需要额外去 Papers 页面手动提交,Hub 选择了"从用户已经在做的事情里顺便提取信息",官方文档提到 arXiv 链接占了用户在仓库里实际引用的论文来源的 95% 以上,这是他们优先支持 arXiv 而不是别的论文托管平台的直接原因。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| model card 里写了论文名字,但没有自动关联 | 只写了论文标题文字,没有放实际的 arXiv 链接 | 确认链接格式是可点击的 `https://arxiv.org/abs/xxxx.xxxxx`,不是纯文字提及 |
| 论文页面上没有出现在自己组织(organization)主页 | 没有把 HF organization 链接也加进 model card | 官方文档提到"链接 HF 组织"也会让论文出现在组织主页,这是额外的一步,不是自动发生的 |
| 论文刚挂上 arXiv 几周,一直没能提交到 Papers 首页推荐(daily papers) | Daily papers 推荐有时间窗口限制 | 官方文档提到 daily papers 提交窗口是论文发布后 14 天内,过期后论文依然会被索引(能被搜到、能关联模型),只是不再进入首页推荐轮换 |

**自测清单:**

- [ ] 能说出触发"论文页面自动关联模型/数据集"的具体条件(model card 里出现 arXiv 链接)
- [ ] 知道这个联动不需要额外手动提交,是 Hub 自动抓取的
- [ ] 知道作者认领(claim authorship)是需要人工审核的流程,不是点一下就立即生效

---

*参考来源:[HuggingFace Model Card Guidebook](https://huggingface.co/docs/hub/en/model-card-guidebook)、[Annotated Model Card](https://huggingface.co/docs/hub/model-card-annotated)、[Dataset Cards](https://huggingface.co/docs/hub/datasets-cards)、[Create a dataset card](https://huggingface.co/docs/datasets/dataset_card)、[HuggingFace Paper Pages 官方文档](https://huggingface.co/docs/hub/en/paper-pages)。license 标识符列表、`LocalTokenNotFoundError` 行为、`tags`/`datasets` 字段的静默类型转换行为,均为本机 `huggingface_hub==1.18.0` 真实实测所得,实测日期 2026-07-25。*
*创建:2026-07-25*
