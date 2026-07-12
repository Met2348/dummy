# 11 · Hub 与模型分享机制(Hub & Model Sharing Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)。已在仓库根目录 `.venv` 真实跑通(`huggingface_hub==1.18.0`)。**本机没有配置 `HF_TOKEN`(`get_token()` 返回 `None`),知识点 3/6 涉及需要写权限/私有仓库权限的操作,如实标注验证到什么程度,不冒充完整跑通了需要认证的部分。**

---

## 1. `huggingface_hub` 库基础(`HfApi`)

**签名/是什么:**
```
from huggingface_hub import HfApi
api = HfApi()
api.model_info("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```
`huggingface_hub` 是比 `transformers` 更底层的库——`transformers` 的 `from_pretrained` 内部下载权重这一步,实际调用的就是 `huggingface_hub` 提供的能力。`HfApi` 是这个库里和 Hub 网站直接对话的客户端对象。

**一句话:** 不需要认证就能查询公开仓库的元数据(文件列表、commit 历史、模型卡片等),需要写权限(上传/修改)的操作才需要 token。

**底层机制/为什么这样设计:** `transformers`/`datasets`/`peft`/`trl` 这些上层库都不各自实现一套"怎么和 Hub 通信"的逻辑,而是统一依赖 `huggingface_hub` 这个基础库——这是标准的分层设计:`huggingface_hub` 只管"网络协议层"(认证、下载、上传、缓存、查询 API),各个上层库在此基础上加各自领域的语义(`transformers` 知道怎么把下载下来的文件组装成一个模型对象)。理解这一层的价值在于:很多"下载/缓存"相关的问题(比如 02 类讲过的权重缓存、`local_files_only`),根源都在 `huggingface_hub` 这一层,不是 `transformers` 自己的逻辑。

**AI 研究/工程场景:** 想批量查询/管理多个模型仓库(比如脚本化地检查一批模型的文件大小、最后更新时间),直接用 `HfApi` 比每个都走 `AutoModel.from_pretrained` 高效得多——后者会真的触发下载,前者只是查询元数据。

**可运行例子:**
```python
from huggingface_hub import HfApi, get_token

api = HfApi()

# 本机没有配置token,查询公开仓库的元数据完全不受影响
assert get_token() is None

info = api.model_info("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
assert info.id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
assert len(info.sha) == 40  # 完整的git commit sha

filenames = {s.rfilename for s in info.siblings}
assert "config.json" in filenames
assert "model.safetensors" in filenames  # 呼应02类讲过的safetensors格式
assert "README.md" in filenames          # 这就是知识点5要讲的model card源文件

print(f"OK: 未认证情况下查询到{len(filenames)}个文件,sha={info.sha[:12]}...")
```
本机实测:确认本机 `get_token()` 返回 `None`(未配置任何 token),但公开仓库的元数据查询完全正常,拿到真实的 40 位 commit sha 和完整文件列表。

**面试怎么问 + 追问链:** "`huggingface_hub` 和 `transformers` 是什么关系?" → 追问"如果 Hub 网站本身访问不了(比如公司内网屏蔽),`transformers` 还能用吗?"(能,但仅限于本地已缓存的模型 + `local_files_only=True`,或者配合企业内部搭建的镜像站点——`HF_ENDPOINT` 环境变量可以指向自建的镜像服务,这是很多国内团队实际会用到的配置)。

**常见坑:**
1. 不要把"查询元数据不需要 token"和"下载文件不需要 token"混为一谈——本知识点只验证了前者,knowledge point 6 会展示两者在 gated 仓库上的真实差异。
2. `HfApi()` 默认连接的是 `huggingface.co`,国内网络访问可能不稳定,`HF_ENDPOINT` 环境变量可以切换到镜像站点(比如社区维护的 `hf-mirror.com`),这是实际项目里常见的网络适配手段。

---

## 2. 缓存机制精解

**签名/是什么:**
```
from huggingface_hub import scan_cache_dir
info = scan_cache_dir()   # 扫描本机 ~/.cache/huggingface/hub 目录
```
和 02 类知识点 4 用过的是同一个函数,这里展开讲缓存目录本身的组织结构。

**一句话:** 缓存目录不是简单地"文件名对应仓库名",而是按 `模型仓库 → revision(commit) → 文件` 三层组织,同一个仓库的不同版本可以共存,靠 `refs`(比如 `main` 分支名)指向具体某个 revision。

**底层机制/为什么这样设计:** Hub 上的每个仓库本质是一个 git 仓库,`revision` 可以是分支名(`main`)、标签、或者具体的 commit sha。缓存系统用 `blobs`(按内容哈希存储实际文件数据,不同 revision 如果有完全相同的文件内容,会共享同一份物理存储,不重复占用磁盘)+ `snapshots`(每个 revision 一个目录,里面是指向 blobs 的符号链接,构成这个 revision 该有的完整文件树)这套结构实现"多版本共存但不重复存储相同内容"。**在 Windows 上,如果没有开发者模式/管理员权限,符号链接创建不了**,缓存系统会退化成直接拷贝文件(而不是链接),仍然能正确工作,只是失去了"内容去重"这个磁盘节省效果。

**AI 研究/工程场景:** 长期在同一台机器上做实验,缓存目录会持续增长(尤其是频繁切换不同模型/不同 revision 做对比实验时),理解这套目录结构能让你在磁盘紧张时有针对性地清理(`huggingface-cli delete-cache` 命令,或者直接用 `scan_cache_dir()` 返回的对象编程式地清理),而不是不清楚哪些能删就直接删掉整个缓存目录。

**可运行例子:**
```python
from huggingface_hub import scan_cache_dir

cache_info = scan_cache_dir()

target = None
for repo in cache_info.repos:
    if repo.repo_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        target = repo
        break

assert target is not None
assert target.repo_type == "model"

revisions = list(target.revisions)
assert len(revisions) >= 1
rev = revisions[0]
assert len(rev.commit_hash) == 40   # 完整commit sha,和知识点1查到的info.sha同源
assert "main" in rev.refs           # main分支名指向这个具体revision

print(f"OK: 缓存里{target.repo_id}的revision {rev.commit_hash[:12]}... 被'main'这个ref指向,占用{target.size_on_disk_str}")
```
本机实测:`TinyLlama` 仓库缓存的 revision `commit_hash` 前缀是 `fe8a4ea1ffed`,和知识点 1 里 `HfApi.model_info()` 查到的 `sha` 完全一致(证实缓存的版本确实对应 Hub 上 `main` 分支当前指向的那个 commit);占用磁盘 2.2G。运行过程中触发了真实警告:"your machine does not support them [symlinks]... requires Developer Mode or to run Python as an administrator"——确认了本机 Windows 环境确实退化成了拷贝模式。

**面试怎么问 + 追问链:** "为什么 Hub 缓存要设计成 blobs+snapshots 这种结构而不是直接按仓库名存文件?" → 追问"这套设计在 Windows 无符号链接权限的情况下,磁盘占用会有什么变化?"(正常情况下,如果两个不同 revision 之间有 95% 的文件没变化,符号链接机制只需要多存 5% 的新内容;退化成拷贝模式后,每个 revision 都是完整的一份拷贝,磁盘占用会显著更高——这是本机这类 Windows 开发环境需要留意的真实差异)。

**常见坑:**
1. 直接用文件管理器手动删除缓存目录下的某些文件,可能破坏 blobs/snapshots 之间的链接关系,导致缓存状态不一致——清理缓存应该用官方提供的工具(`scan_cache_dir()` 配合 `.delete_revisions()`,或者 `huggingface-cli delete-cache` 命令行工具),不要手动瞎删。
2. 符号链接在 Windows 上退化成拷贝这件事本身不影响功能正确性,只影响磁盘占用——不要误以为看到这条警告就意味着什么东西坏了。

---

## 3. `push_to_hub()` 机制

**签名/是什么:**
```
model.push_to_hub("my-username/my-model-name")
tokenizer.push_to_hub("my-username/my-model-name")
```
把本地训练/微调好的模型(或 tokenizer)上传到 Hub,变成一个可以被别人 `from_pretrained` 直接加载的仓库。

**一句话:** `push_to_hub()` 内部做的事情是"先 `save_pretrained()` 到一个临时目录,再把这个目录的内容作为一次 git commit 推送到 Hub 上对应的仓库"——**没有任何魔法,产出的文件和你本地 `save_pretrained()` 会得到的文件完全一样**,只是多了"上传"这一步。

**底层机制/为什么这样设计:** 这个设计让"分享模型"这件事和"本地保存模型"复用完全相同的序列化逻辑,不需要维护两套代码路径。这也意味着理解"push 上去的是什么",最直接的方法就是先在本地 `save_pretrained()` 看看产出了哪些文件——这是本知识点在没有写权限 token 的情况下依然能验证的部分。

**AI 研究/工程场景:** 09 类微调完成后,如果想把训练好的 adapter 分享给同学/团队复用,`push_to_hub()` 是比"发压缩包"更规范的方式(带版本管理、带 model card、别人可以直接一行代码加载)——但本系列的验证环境没有配置 Hub 账号 token,09 类的"微调产物保存与复现"知识点会用 `save_pretrained()`/`from_pretrained(本地路径)` 完成同等验证,`push_to_hub()` 只作机制性介绍。

**可运行例子:**
```python
import os
import tempfile
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 验证push_to_hub方法确实存在(继承自huggingface_hub的Mixin)
assert hasattr(tok, "push_to_hub")
assert callable(tok.push_to_hub)

# 本机没有写权限token,无法真实推送;但可以验证push_to_hub会上传的"内容"——
# 因为push_to_hub内部第一步就是save_pretrained到本地临时目录,这一步不需要认证
tmpdir = tempfile.mkdtemp()
tok.save_pretrained(tmpdir)
saved_files = set(os.listdir(tmpdir))

assert "tokenizer_config.json" in saved_files
assert "tokenizer.json" in saved_files  # 呼应01类,当前版本tokenizer是TokenizersBackend统一实现,
                                          # 词表/merges等信息整体序列化在这一个文件里

print(f"OK: push_to_hub方法存在;save_pretrained(等价于push前半段)产出文件: {sorted(saved_files)}")
```
本机实测:`save_pretrained` 产出 `chat_template.jinja`、`tokenizer.json`、`tokenizer_config.json` 三个文件——**这里有个值得记录的真实细节**:当前版本把 chat template 单独存成了 `.jinja` 文件,不是像旧版本那样整个内嵌在 `tokenizer_config.json` 里,这也是"不能凭旧版本记忆写细节"这条纪律的又一个例证。这三个文件就是 `push_to_hub()` 真正会上传的内容,没有额外的隐藏步骤。

**面试怎么问 + 追问链:** "`push_to_hub()` 失败了,应该从哪几个角度排查?" → 认证问题(token 没配置/权限不够)、网络问题(连不上 Hub)、仓库名冲突(目标仓库已存在且不属于你)是三个最常见的方向;追问"如果我想上传前先本地检查一遍会上传什么内容,该怎么做?"(就是本例展示的做法——本地 `save_pretrained()` 到临时目录,人工检查一遍文件内容,再决定要不要真的 `push_to_hub()`,这是稳妥的工程习惯)。

**常见坑:**
1. `push_to_hub()` 默认会创建**公开**仓库(除非显式传 `private=True`),不小心上传了不该公开的内容(比如还在实验阶段、包含敏感数据训练出来的模型)是真实的隐私事故来源,上传前确认可见性设置。
2. 大模型上传是真实的网络带宽和时间开销(GB 级别的文件),不要在网络不稳定的环境里发起大文件上传又没有断点续传的心理准备——`huggingface_hub` 对大文件上传有分块机制,但网络中断仍可能需要重试。

---

## 4. 模型版本管理(`revision` 参数)

**签名/是什么:**
```
AutoTokenizer.from_pretrained(MODEL, revision="main")                              # 默认:主分支最新
AutoTokenizer.from_pretrained(MODEL, revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6")  # pin到具体commit
```
`revision` 参数不仅接受分支名,也接受具体的 commit sha(或者 tag),把加载的版本锁定到某个确切的时间点。

**一句话:** 不传 `revision` 默认等价于 `revision="main"`——**这意味着同一份代码,如果模型仓库的 `main` 分支后续被作者更新了权重/配置,你下次运行拿到的可能不是当初调试时用的那个版本**,`revision` 参数就是解决这个"版本漂移"问题的机制。

**底层机制/为什么这样设计:** 和 02 类"缓存机制"讲过的 blobs/snapshots 结构直接对应——`revision` 参数最终就是知识点 2 提到的"哪个 revision 目录"这个选择。pin 到具体的 40 位 commit sha 是最严格的锁定方式(这个 sha 对应的文件内容永远不会变,git 的内容寻址特性保证了这一点);pin 到 tag(如果仓库作者发布了版本标签)是稍宽松但依然明确的锁定方式。

**AI 研究/工程场景:** 写论文/做正式实验报告时,复现性要求你能明确说清楚"用的是这个模型的哪个确切版本"——只写模型名字(不带 revision)在严谨的研究场景下是不够的,如果原作者之后更新了仓库内容,你的实验结果可能变得无法复现,pin 具体 commit sha 是对这个问题的标准解法。

**可运行例子:**
```python
from transformers import AutoTokenizer
from huggingface_hub import HfApi

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
api = HfApi()

# 拿到main分支当前指向的确切commit sha
current_sha = api.model_info(MODEL).sha
assert len(current_sha) == 40

# 显式pin到这个sha,和不传revision(默认main)应该加载出完全一样的内容
tok_default = AutoTokenizer.from_pretrained(MODEL)
tok_pinned = AutoTokenizer.from_pretrained(MODEL, revision=current_sha)
assert tok_default.vocab_size == tok_pinned.vocab_size == 32000

# 一个不存在的revision应该明确报错,而不是静默回退到默认版本
raised = False
try:
    AutoTokenizer.from_pretrained(MODEL, revision="nonexistent-branch-xyz-123")
except OSError:
    raised = True
assert raised

print(f"OK: pin到commit {current_sha[:12]}... 和默认main加载结果一致;不存在的revision正确报错而非静默回退")
```
本机实测:`main` 分支当前 sha 是 `fe8a4ea1ffedaf415f4da2f062534de366a451e6`,显式 pin 这个 sha 加载和默认加载结果一致;传一个不存在的分支名触发 `OSError`(不是静默使用默认版本,这一点很重要——如果它静默回退,pin 机制就完全失去了"保证复现性"的意义)。

**面试怎么问 + 追问链:** "为什么严谨的研究工作建议 pin 具体的 commit sha 而不是用分支名?" → 追问"如果我 pin 了一个 sha,但那个 revision 对应的文件已经从 Hub 上被删除了(比如作者删库了),会发生什么?"(本地如果已经缓存过,`local_files_only` 场景不受影响;但没缓存过、纯依赖网络重新拉取的场景会直接失败——这也是为什么重要实验的依赖,团队内部通常会有自己的镜像/归档策略,不能完全依赖第三方仓库永远存在)。

**常见坑:**
1. `revision` 传的 sha 前缀(比如只传前 8 位)在 git 里通常也能唯一定位到一个 commit,但 Hub 的 API 有些接口要求完整 sha,养成传完整 40 位 sha 的习惯更保险。
2. 不要把"pin 了 revision"和"这个模型内容一定安全可信"混为一谈——pin 只保证"版本不会变",不保证"这个版本本身没有问题",这是两个独立的关注点。

---

## 5. Model Card 机制

**签名/是什么:**
```
from huggingface_hub import ModelCard
card = ModelCard.load("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
card.data.license   # 结构化元数据(YAML front matter)
card.text            # 人类可读的正文(Markdown)
```
Model Card 就是模型仓库的 `README.md`,但 `ModelCard` 这个类能把文件开头的 YAML 元数据块(license、语言、任务标签等结构化信息)和后面的自由格式说明文字分开解析。

**一句话:** `README.md` 顶部用 `---` 包起来的部分是**给机器读的**结构化字段(Hub 网站用它来做筛选/展示标签),下面的部分是**给人读的**自由文本说明——`ModelCard` 类把这两部分分别暴露成 `.data` 和 `.text`。

**底层机制/为什么这样设计:** 这是经典的"YAML front matter"模式(很多静态博客生成器也用同样的约定)——把结构化元数据和自由文本内容放在同一个文件里,但用明确的分隔符区分开,既保持了"一个文件里能看到全部信息"的可读性,又能让程序精确提取出结构化的那一部分做进一步处理(比如 Hub 网站首页的"按 license 筛选"功能,背后读的就是这个字段)。

**AI 研究/工程场景:** 批量调研/筛选 Hub 上的模型时(比如想找"所有 Apache 2.0 协议的 1B 参数以下对话模型"),比起打开网页一个个看,用 `ModelCard`/`HfApi` 批量拉取结构化元数据脚本化筛选要高效得多;09 类选型 TinyLlama 时,"Apache 2.0"这个协议信息就是从这里核实的。

**可运行例子:**
```python
from huggingface_hub import ModelCard

card = ModelCard.load("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

assert isinstance(card, ModelCard)
assert card.data.license == "apache-2.0"   # 和00-roadmap.md环境声明记录的协议信息交叉验证一致
assert len(card.text) > 0
assert "TinyLlama" in card.text            # 正文确实提到了模型名字

# data部分是结构化对象,可以像访问普通属性一样取值
license_value = card.data.license
assert isinstance(license_value, str)

print(f"OK: license={license_value!r}(和环境声明记录的一致),正文长度{len(card.text)}字符")
```
本机实测:`card.data.license` 精确读出 `'apache-2.0'`——这是对 00-roadmap.md"模型/数据集选型"章节里"Apache 2.0"这条记录的独立交叉验证(不是道听途说,是直接从模型卡片的结构化元数据里读出来的);正文长度 2731 字符,开头确认是 TinyLlama 项目的说明文字。

**面试怎么问 + 追问链:** "Model Card 的 license 字段和实际的法律约束是什么关系?" → 这个字段是**仓库维护者自己填写**的声明,Hub 平台本身不做法律层面的强制核验——理论上存在填写错误/滞后更新的可能性,重要的商业/法律决策不应该只依赖这一个字段的字面值,必要时应该找到原始论文/官方声明做交叉核实(这也是为什么 00-roadmap.md 对 `no_robots` 数据集协议特意提到"最初以为是MIT,查证后是CC-BY-NC-4.0"这个真实的自我纠错案例)。

**常见坑:**
1. 不是所有仓库的 `README.md` 都规范地包含 YAML front matter——一些个人上传的仓库可能完全没有结构化元数据,`card.data` 对应字段会是 `None`,不能假设所有仓库都能读到完整信息。
2. `ModelCard.load()` 本身也是一次网络请求(除非已缓存),批量对大量仓库调用会有明显的网络开销,大规模筛选场景应该考虑 Hub 提供的搜索 API(能在服务端就按条件筛选,不需要客户端逐个拉取)而不是每个仓库都读一遍卡片。

---

## 6. 私有仓库与访问控制(`token` 参数)

**签名/是什么:**
```
AutoModelForCausalLM.from_pretrained("some-org/private-model", token="hf_xxxxx")
```
访问私有仓库或者需要申请授权的 "gated" 仓库(比如 Meta 的 Llama 系列官方仓库),需要显式传一个有权限的 `token`。

**一句话:** **"能查到仓库元数据"和"能下载仓库文件"是两个独立的权限层级**——本机的实测发现:不带 token 查询一个 gated 仓库的 `model_info()` 完全成功(能看到文件列表、能看到 `gated` 状态本身),但真的去下载里面任何一个文件,会被明确拒绝。

**底层机制/为什么这样设计:** Hub 的元数据(仓库存在、有哪些文件、gated 状态是什么)本身不算敏感信息,公开可查有助于用户在申请授权之前就了解这个仓库大概是什么样子;真正的内容(文件字节数据)才是需要授权保护的资产。这种"元数据公开、内容受控"的分层访问控制,和很多云存储服务(能看到文件名/大小,但没权限的话下载会被拒绝)是同一套设计思路。

**AI 研究/工程场景:** 使用 Llama 系列等需要申请授权的官方模型时,标准流程是:先在 Hub 网站上申请访问权限(通常需要同意使用协议)→ 生成一个有权限的 personal access token → 代码里通过 `token=` 参数或者 `huggingface-cli login` 配置这个 token → 才能真正下载权重。只看到"能查到这个仓库存在"不代表已经有下载权限,这是新手容易搞混的两件事。

**可运行例子:**
```python
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
GATED_REPO = "meta-llama/Llama-2-7b-hf"  # 知名的gated仓库,用来验证访问控制的两层语义

# 第一层:元数据查询——不需要token也能成功
info = api.model_info(GATED_REPO)
assert info.gated in ("manual", "auto", True)  # 确认这确实是一个gated仓库
assert len(info.siblings) > 0                   # 文件列表可见

# 第二层:真实下载——没有token/没有被授权,应该被明确拒绝
raised = False
try:
    hf_hub_download(GATED_REPO, filename="config.json")
except Exception as e:
    raised = True
    error_type = type(e).__name__

assert raised
assert error_type == "GatedRepoError"  # 明确的"gated仓库拒绝访问"错误类型,不是笼统的网络错误

print(f"OK: gated仓库{GATED_REPO}——元数据查询成功(gated={info.gated}),文件下载被拒绝({error_type})")
```
本机实测:`meta-llama/Llama-2-7b-hf` 的 `model_info()` 显示 `gated='manual'`(需要人工审核授权申请),17 个文件的元数据完全可见;真正尝试下载 `config.json` 触发 `GatedRepoError`(HTTP 401),错误信息明确写着"Access to model ... is restricted"——精确验证了"元数据公开、内容受控"这个两层访问控制模型。

**面试怎么问 + 追问链:** "怎么排查'为什么我加载不了这个模型,报了一个 gated 相关的错误'?" → 三步排查:① 确认自己是否已经在 Hub 网站上申请并获得了这个仓库的访问授权;② 确认本地/代码里配置的 token 确实是自己账号的、且没有过期;③ 确认这个 token 有读取权限(Hub 的 token 可以设置细粒度权限,有的 token 可能是只读某些资源、不包括这个特定仓库)。

**常见坑:**
1. 把 token 硬编码进代码/提交进 git 仓库是真实的安全事故来源——token 应该通过环境变量(`HF_TOKEN`)或者 `huggingface-cli login` 存到本地配置文件,不应该出现在任何会被提交到版本控制的代码里。
2. 申请 gated 仓库授权到审核通过之间通常有延迟(不是提交申请立刻生效),如果代码报错但你确信自己已经"申请过了",先去 Hub 网站确认申请状态是否真的已经是"approved",而不是反复怀疑自己代码写错了。

---

*本篇 6 个知识点在仓库根目录 `.venv` 真实验证通过,知识点 3(push_to_hub 实际推送)和知识点 6(gated 仓库实际下载)受限于本机未配置写权限/授权 token,如实标注验证到"能验证的边界"为止,未冒充完整跑通。*
