# 04 · Datasets 库机制(Datasets Library Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)。本篇例子统一以 `timdettmers/openassistant-guanaco` 为对象,已在仓库根目录 `.venv` 真实跑通(`datasets==5.0.0`)。**知识点 3 涉及 Windows 多进程 spawn 的一个真实陷阱,详见该知识点和 00-roadmap.md 环境声明,写自己的脚本时务必注意。**

---

## 1. `load_dataset` 内部机制(Arrow 格式与内存映射)

**签名/是什么:**
```
from datasets import load_dataset
ds = load_dataset("timdettmers/openassistant-guanaco", split="train")
```
`load_dataset` 把 Hub 上的数据集下载并转换成本地的 **Arrow 格式**文件,之后每次访问都是对这份 Arrow 文件做内存映射读取,不是把整个数据集常驻在 Python 进程内存里。

**一句话:** `datasets` 库能轻松处理比内存还大的数据集,秘诀就是"数据本身留在磁盘上的 Arrow 文件里,Python 对象只是一层内存映射视图",这和 `numpy`/普通 `list` 那种"数据整个读进内存"的直觉完全不同。

**底层机制/为什么这样设计:** Apache Arrow 是一种列式内存格式,设计目标之一就是"文件在磁盘上的字节布局,和进程访问它时用的内存布局是同一份"(通过 `mmap` 系统调用直接把文件映射进进程地址空间),读取数据不需要经过"从磁盘读→反序列化→构造 Python 对象"这个传统流程,而是直接按偏移量读取已经是目标格式的字节。这也是为什么 `ds[0]` 这种随机访问几乎是瞬时的,即使整个数据集文件有几十 GB——你访问的只是被映射的那一小块,操作系统按需把对应的磁盘页面调进物理内存,不需要的部分完全不占用内存。

**AI 研究/工程场景:** 处理数十 GB 甚至上百 GB 的预训练语料时,如果每次都要求整个数据集能放进内存,很多研究根本没法在普通工作站上做;Arrow 的内存映射设计让"数据集比内存大"这件事变得完全不是问题,这也是为什么 `datasets` 库能同时兼顾"小规模实验"和"预训练级别语料处理"两种场景,不需要换一套工具。

**可运行例子:**
```python
from datasets import load_dataset

ds = load_dataset("timdettmers/openassistant-guanaco", split="train")

assert type(ds).__name__ == "Dataset"
assert len(ds) == 9846
assert list(ds.features.keys()) == ["text"]
assert set(ds[0].keys()) == {"text"}

# 数据确实是磁盘上的Arrow文件,不是纯内存对象——cache_files能查到具体文件路径
assert len(ds.cache_files) > 0
cache_path = ds.cache_files[0]["filename"]
assert cache_path.endswith(".arrow")

import os
assert os.path.exists(cache_path)  # 这个文件真实存在于磁盘上
file_size_mb = os.path.getsize(cache_path) / 1e6
assert file_size_mb > 1  # 真实文件,有实际大小,不是空占位符

print(f"OK: 9846条数据,后端是磁盘上的Arrow文件({file_size_mb:.1f}MB),不是纯内存list")
```
本机实测:`len(ds)==9846`,`ds.cache_files[0]["filename"]` 指向真实存在的 `.arrow` 文件(路径类似 `~/.cache/huggingface/datasets/timdettmers___openassistant-guanaco/.../openassistant-guanaco-train.arrow`)。

**面试怎么问 + 追问链:** "`datasets` 库怎么处理比内存还大的数据集?" → 追问"内存映射和'把文件整个读进内存的 `bytes` 对象'有什么本质区别?"(内存映射是**惰性**的,操作系统按需调页,理论上能映射比物理内存大得多的文件;读进 `bytes` 对象是**立即**把全部内容物化进内存,受物理内存严格限制)→ 深挖"内存映射的数据是只读的吗?能修改吗?"(`Dataset` 对象本身设计成不可变,`.map()`/`.filter()` 这类操作都是产出**新的** Arrow 文件/新的 `Dataset` 对象,不是原地修改,这是理解 08 知识点"map缓存失效"的重要前提)。

**常见坑:**
1. 不要假设 `len(ds)` 或者遍历一个 `Dataset` 对象会有明显的内存开销——这正是 Arrow 内存映射设计要避免的问题,如果你发现内存占用异常增长,更可能是代码里某处不小心把数据转成了普通 Python list/调用了会整体物化数据的操作。
2. 首次 `load_dataset` 需要下载+转换成 Arrow 格式,这一步确实要落盘、要花时间;第二次调用会直接复用已经转换好的本地 Arrow 缓存,速度差异很大,不要把"首次加载慢"误判为"这个库性能不行"。

---

## 2. `streaming=True` 流式加载

**签名/是什么:**
```
ds_stream = load_dataset("timdettmers/openassistant-guanaco", split="train", streaming=True)
for item in ds_stream:   # 逐条产出,不是先加载完整个数据集再迭代
    ...
```
`streaming=True` 返回的是 `IterableDataset`,不是知识点 1 讲的 `Dataset`——**连"下载转换成本地 Arrow 文件"这一步都跳过了**,数据边下载边处理。

**一句话:** 知识点 1 的 Arrow 内存映射已经能优雅处理"比内存大但能放进本地磁盘"的数据集;`streaming=True` 解决的是更极端的情况——**连本地磁盘都放不下**(或者根本不想等全量下载完成)的超大数据集/持续增长的数据流。

**底层机制/为什么这样设计:** `IterableDataset` 内部不是"数据已经在某个地方、我给你一个访问视图"(那是 `Dataset` 的设计),而是一个真正的**惰性生成器管道**——`.map()`/`.filter()` 这类操作在流式模式下不会立即执行,而是被记录成"等真正迭代到某一条数据时,再对这一条数据依次应用这些变换"。这个设计意味着 `IterableDataset` **没有明确的长度**(除非数据源本身提供了元数据),因为在真正迭代完之前,没有人知道总共有多少条——这是它和 `Dataset` 最大的行为差异,很多为 `Dataset` 设计的 API(比如按下标随机访问)在 `IterableDataset` 上根本不存在。

**AI 研究/工程场景:** 预训练超大规模语料(TB 级别)时,`streaming=True` + 边下载边训练是标准做法,不可能等价的效果情况下先把整个语料下载落盘;09 类的实验规模远达不到需要流式加载的程度(用的是子采样的小数据集),但理解这个机制,是未来真做大规模训练时的必备知识。

**可运行例子:**
```python
from datasets import load_dataset, IterableDataset

ds_stream = load_dataset("timdettmers/openassistant-guanaco", split="train", streaming=True)

assert isinstance(ds_stream, IterableDataset)
assert not hasattr(ds_stream, "__len__")  # 流式数据集没有明确长度,这是和Dataset最大的区别

# 只能用迭代的方式访问,不支持 ds_stream[0] 这种随机下标访问
first_item = next(iter(ds_stream))
assert "text" in first_item
assert isinstance(first_item["text"], str)

# take() 可以取前N条,常用于流式模式下的小规模预览/调试
first_three = list(ds_stream.take(3))
assert len(first_three) == 3

print(f"OK: 流式数据集类型={type(ds_stream).__name__},无__len__,take(3)拿到{len(first_three)}条预览")
```
本机实测:`streaming=True` 返回 `IterableDataset`,确认没有 `__len__`,`next(iter(...))` 能正常取出第一条,`.take(3)` 能拿到指定条数的预览。

**面试怎么问 + 追问链:** "流式数据集为什么不能直接用下标随机访问?" → 这是"惰性生成器"这个设计选择的直接推论(生成器本来就不支持随机访问,只能顺序消费)→ 追问"流式模式下怎么做训练/验证集切分?"(不能像 `Dataset` 那样先知道总长度再按比例切,通常需要用 `.take(N)`/`.skip(N)` 组合,或者提前知道数据源自带的切分标记)。

**常见坑:**
1. 对一个 `IterableDataset` 调用 `len()` 会直接报错(`TypeError`),这是新手从 `Dataset` 切换过来最容易踩的坑——两者接口看起来相似,但底层语义完全不同,不能无脑套用同一套代码。
2. 流式模式下的 `.shuffle()` 不是真正的全局随机打乱(不可能,因为不知道全部数据),而是维护一个固定大小的"缓冲区"做近似打乱——这个近似程度取决于缓冲区大小,不了解这一点可能会误以为拿到的是完全随机的顺序。

---

## 3. `map()` 并行处理机制(含 Windows 多进程真实陷阱)

**签名/是什么:**
```
# 正确的、安全的写法:num_proc必须配合 __main__ 保护(Windows下强制要求,不是可选项)
if __name__ == "__main__":
    ds.map(fn, num_proc=4)

# 不需要多进程时,batched=True 在单进程内做批量处理,不需要任何保护
ds.map(batched_fn, batched=True, batch_size=1000)
```
`.map()` 有两种独立的加速手段:`num_proc`(多进程并行,每个进程处理数据的一部分)和 `batched=True`(单进程内一次处理一批而不是逐条调用函数,减少 Python 函数调用开销)。

**一句话:** **这两种加速手段可以叠加使用,但 `num_proc` 在 Windows 上有一个真实的、危险的陷阱**——本篇撰写过程中现场触发过一次进程数量失控增长的真实事故,详情见下方"常见坑"。

**底层机制/为什么这样设计:** `num_proc=N` 内部用 `multiprocessing.Pool(N)` 把数据集切成 N 份,分给 N 个子进程各自独立处理,最后合并结果——这是标准的"数据并行"模式,能利用多核 CPU 加速 CPU 密集型的处理函数。**关键问题在于 Windows 没有 `fork()` 系统调用**,`multiprocessing` 在 Windows 上被迫用 `spawn` 方式创建子进程:子进程不是"复制"父进程当前的内存状态(那是 `fork` 的语义),而是**从头启动一个新的 Python 解释器,重新 import 主脚本模块**来重建运行环境。如果调用 `.map(num_proc=...)` 的代码写在脚本顶层(没有 `if __name__ == "__main__":` 保护),子进程重新执行主脚本时会**再次执行到同一句 `.map(num_proc=...)`**,于是又 spawn 出新的子进程,新子进程又重复这个过程——**这是一个真实的递归 spawn 炸弹,不是理论风险**。`if __name__ == "__main__":` 保护有效的原因是:子进程重新执行主脚本时,Python 会把这次执行的模块名设成 `"__mp_main__"` 而不是 `"__main__"`,守卫条件不成立,不会再次触发 `.map()` 调用。

**AI 研究/工程场景:** 数据预处理阶段(分词、清洗、格式转换)是很多训练 pipeline 里 CPU 成为瓶颈的地方,`num_proc` 是标准的加速手段;但任何在 Windows 上写训练脚本的人,只要用到 `num_proc`(不只是 `datasets.map`,`torch.utils.data.DataLoader(num_workers=N)` 也是同样的 `spawn` 语义,同样需要这个保护),都必须养成把主逻辑放进 `if __name__ == "__main__":` 的习惯——这不是这个系列独有的坑,是 Windows 平台本身的性质。

**可运行例子:**
```python
from datasets import load_dataset
import time

ds = load_dataset("timdettmers/openassistant-guanaco", split="train")
small = ds.select(range(20))

# 本例只演示 batched=True 这种不涉及多进程、单进程内安全可跑的加速方式
# (num_proc 的用法和真实踩坑经过在上面"底层机制"段落和下面"常见坑"里详细说明,
#  出于安全考虑,不在这个会被独立执行的代码块里直接调用 num_proc,避免示例本身
#  被复制粘贴到没有 __main__ 保护的脚本里时构成风险)

def single_fn(x):
    return {"length": len(x["text"])}

def batched_fn(batch):
    return {"length": [len(t) for t in batch["text"]]}

mapped_single = small.map(single_fn)                          # 逐条调用
mapped_batched = small.map(batched_fn, batched=True, batch_size=4)  # 分批调用

assert mapped_single[0]["length"] == mapped_batched[0]["length"]  # 两种方式结果一致
assert mapped_single[5]["length"] == mapped_batched[5]["length"]

print(f"OK: 单条/批量map结果一致,第0条长度={mapped_single[0]['length']}")
```
**本机实测(在独立的、正确加了 `if __name__=="__main__":` 保护的脚本里跑的,不是上面这个代码块)**:20 条数据 `num_proc=2` 耗时 4.179 秒——**这个数字本身就是一个真实的教学点**:对这么小的数据量,启动两个子进程的固定开销(每个子进程都要重新初始化 Python 解释器、重新 import `datasets` 等依赖库)远超实际处理数据的耗时,`num_proc` 只有在数据量/单条处理耗时都足够大时才划算,这也是为什么本知识点的可运行例子刻意选择演示 `batched=True` 而不是 `num_proc`。

**面试怎么问 + 追问链:** "`num_proc` 和 `batched=True` 该怎么选?" → 追问"如果我的处理函数本身很轻量(比如只是字符串长度计算),要不要开多进程?"(不该,本例的真实耗时对比就是证据:轻量处理函数配小数据集,多进程的固定开销反而拖慢整体;`num_proc` 适合"单条处理确实很重"(比如复杂的文本清洗规则、调用外部工具)且数据量足够大的场景)。

**常见坑:**
1. **(本系列现场真实事故)** 在 Windows 上,`.map(num_proc=N)`(**包括 `N=1`**,实测确认 `N=1` 也会走 `multiprocessing.Pool` 代码路径)如果不放进 `if __name__ == "__main__":` 保护块,会触发递归 spawn 子进程——本篇撰写时现场复现,表现为同一段日志反复打印、进程数量持续增长,已立即用 `TaskStop` 终止并确认无残留进程。这不是"可能出现的边缘情况",是**必然复现**的行为,任何在 Windows 上直接跑（不加保护）的脚本都会触发。
2. Linux/WSL2 用 `fork()`,子进程是内存状态的直接复制,不会重新 import 主模块,所以这个坑在那些平台上不会出现——这也是为什么这个坑在网上的资料里很少被提到(大部分教程/文档作者是在 Linux 环境下写的,根本没遇到过)。

---

## 4. `DatasetDict` 结构与 train/test 切分

**签名/是什么:**
```
full = load_dataset("timdettmers/openassistant-guanaco")   # 不传 split
full["train"], full["test"]
```
不给 `load_dataset` 传 `split` 参数,拿到的是 `DatasetDict`——一个"切分名字 → `Dataset`"的字典容器,而不是单个 `Dataset`。

**一句话:** 一个数据集仓库通常自带多个切分(train/validation/test),`DatasetDict` 是把这些切分统一打包在一起的容器,传 `split="train"` 相当于直接从这个字典里取出 `"train"` 这一份,跳过打包步骤。

**底层机制/为什么这样设计:** `DatasetDict` 本质就是 Python `dict` 的一个子类,`key` 是切分名字(字符串),`value` 是知识点 1 讲的 `Dataset` 对象。这样设计让"这个数据集仓库有哪些切分"这件事对用户完全透明可查(`full.keys()`),同时保持每个切分内部依然是知识点 1 讲的那套 Arrow 内存映射机制,没有引入额外的复杂度。

**AI 研究/工程场景:** 09 类的微调实验会用到 `openassistant-guanaco` 自带的 train(9,846条)/test(518条)切分,`Trainer`(05类)的 `eval_dataset` 参数直接吃 `DatasetDict["test"]` 这种切好的 `Dataset` 对象,不需要自己手写切分逻辑。

**可运行例子:**
```python
from datasets import load_dataset, DatasetDict, Dataset

full = load_dataset("timdettmers/openassistant-guanaco")

assert isinstance(full, DatasetDict)
assert set(full.keys()) == {"train", "test"}
assert isinstance(full["train"], Dataset)  # 每个切分本身就是知识点1讲的Dataset类型
assert len(full["train"]) == 9846
assert len(full["test"]) == 518

# DatasetDict支持类似dict的批量操作,比如同时给所有切分做map
mapped_all = full.map(lambda x: {"len": len(x["text"])})
assert isinstance(mapped_all, DatasetDict)
assert set(mapped_all.keys()) == {"train", "test"}  # map之后切分结构保持不变

print(f"OK: DatasetDict包含{list(full.keys())}两个切分,train={len(full['train'])}条,test={len(full['test'])}条")
```
本机实测:`full.keys()` 确认是 `{'train', 'test'}`,长度分别是 9846 和 518,和 00-roadmap.md 环境声明里记录的数字完全一致。

**面试怎么问 + 追问链:** "如果一个数据集仓库本身没有提供 validation 切分,该怎么办?" → 追问"`Dataset.train_test_split()` 内部是怎么切分的,有没有分层抽样(stratified)选项?"(`train_test_split` 默认是随机切分,可以传 `stratify_by_column` 参数按某个标签列做分层抽样,保证切分后各类别比例和原数据集一致,这在类别不均衡的分类数据集上是常见的正确做法)。

**常见坑:**
1. 不是所有数据集都有 `"test"` 切分,有的只有 `"train"`,有的用 `"validation"` 而不是 `"test"` 这个名字——写通用代码前先 `.keys()` 确认实际有哪些切分,不要硬编码假设。
2. 对 `DatasetDict` 整体调用 `.map()`/`.filter()` 会对**每个切分分别独立**应用同一个函数,不会把不同切分的数据混在一起处理,这个语义容易被误解成"先合并再处理"。

---

## 5. `.with_format("torch")` 桥接 PyTorch

**签名/是什么:**
```
torch_ds = ds.with_format("torch")
torch_ds[0]  # 数值型字段自动变成 torch.Tensor,非数值字段(如字符串)保持原样
```
`.with_format()` 不改变底层数据,只改变"通过下标访问时,返回给你的是什么类型的对象"。

**一句话:** 默认情况下 `ds[i]` 返回的字段是 Python 原生类型(`int`/`float`/`str`/`list`);`.with_format("torch")` 之后,**数值型**字段会自动转换成 `torch.Tensor`,但**字符串字段不受影响,还是普通 `str`**——这个细节很容易被想当然地误解。

**底层机制/为什么这样设计:** `.with_format()` 是一个轻量的、可撤销的视图切换(不会重新生成底层 Arrow 数据,随时可以 `.with_format(None)` 切回原生 Python 类型),它只影响"从 Arrow 列式数据取出一行时,用什么函数把 Arrow 的原生类型转换成最终返回给你的 Python/Tensor 对象"这一步。字符串类型在 PyTorch 里没有对应的 tensor 表示(`torch.Tensor` 只能存数值),所以 `with_format("torch")` 对字符串字段无能为力,只能原样透传——这不是 bug,是数值类型系统本身的限制。

**AI 研究/工程场景:** `guanaco` 数据集只有一个 `text` 字符串字段,`.with_format("torch")` 在这个具体数据集上其实用不太上(因为没有数值字段可转换);但如果数据集本身有预先分好词的 `input_ids`(整数列表)字段,`.with_format("torch")` 能让 `DataLoader` 迭代取出的每个 batch 直接是可以喂给模型的 `torch.Tensor`,省掉手动转换这一步,05 类的 `DataCollator` 讨论会涉及这个衔接点。

**可运行例子:**
```python
from datasets import load_dataset, Dataset

ds = load_dataset("timdettmers/openassistant-guanaco", split="train")
small = ds.select(range(5))

torch_ds = small.with_format("torch")
assert torch_ds.format["type"] == "torch"

item = torch_ds[0]
# text是字符串字段,with_format("torch")对它无能为力,依然是普通str
assert isinstance(item["text"], str)

# 构造一个含数值字段的小数据集,验证数值字段才会真正被转成tensor
import torch
numeric_ds = Dataset.from_dict({"value": [1, 2, 3], "label": ["a", "b", "c"]})
numeric_torch = numeric_ds.with_format("torch")
numeric_item = numeric_torch[0]
assert isinstance(numeric_item["value"], torch.Tensor)  # 数值字段:真正变成tensor
assert isinstance(numeric_item["label"], str)            # 字符串字段:with_format对它没有作用,原样透传

# 撤销格式设置,恢复原生Python类型
back_to_plain = numeric_torch.with_format(None)
assert isinstance(back_to_plain[0]["value"], int)  # 恢复成普通int,不再是tensor

print("OK: with_format('torch')只转换数值字段为tensor,字符串字段原样透传,且可用None撤销")
```
本机实测:`guanaco` 数据集的 `text` 字段在 `with_format("torch")` 后依然是 `str`;自己构造的含数值字段的小数据集验证了数值字段确实会变成 `torch.Tensor`,字符串字段确实原样透传,`with_format(None)` 确认能撤销回原生类型。

**面试怎么问 + 追问链:** "`.with_format("torch")` 之后,字符串字段为什么没有变成 tensor?" → 这是本知识点最容易被问到的细节题,答案要点是"tensor 只能存数值,字符串类型系统里没有对应表示"→ 追问"那文本数据最终要怎么变成能喂给模型的 tensor?"(需要先经过 01 类讲的 tokenizer 编码,把字符串转成整数 id 列表,这个整数字段才能被 `with_format("torch")` 转换成 tensor——`with_format` 不能替代 tokenize 这一步,两者是流水线里前后相继的两个环节)。

**常见坑:**
1. 误以为 `.with_format("torch")` 能让文本数据"自动"变成模型能用的输入——它只是类型转换的最后一步,真正的"文本变数字"工作(tokenize)必须自己先做,`with_format` 只是省去"tokenize完的整数列表再手动转成tensor"这一步的样板代码。
2. `.with_format()` 设置的是整个 `Dataset` 对象的默认返回格式,如果代码里多处共享同一个数据集对象引用,某处改了 format 会影响所有引用同一对象的地方读到的数据类型——这和 01 类"padding_side是可变属性,共享实例要小心"是同一类坑。

---

## 6. `.filter()` / `.select()` / `.shuffle()` 常用操作

**签名/是什么:**
```
ds.filter(lambda x: len(x["text"]) < 200)   # 按条件筛选,产出新Dataset
ds.select([0, 5, 10])                        # 按下标列表取子集
ds.shuffle(seed=42)                          # 打乱顺序,产出新Dataset(seed保证可复现)
```
三个操作都不修改原 `Dataset` 对象,而是各自返回一个新的 `Dataset`。

**一句话:** 这三个操作的共同点是**都不改变知识点 1 讲的底层 Arrow 数据本身**,`filter`/`select` 实际上是维护一份"哪些行是有效的"的索引映射,`shuffle` 是维护一份"下标该按什么顺序读取"的排列映射——都是"换个方式看同一份底层数据",不是真正拷贝/重写数据。

**底层机制/为什么这样设计:** 如果 `filter`/`select`/`shuffle` 每次都要重写一份新的 Arrow 文件,数据量大的时候会非常昂贵(涉及磁盘 I/O);用"索引映射"而不是"物理拷贝"实现这些操作,让它们在任意大小的数据集上都是轻量、快速的——这也是知识点 1"数据在磁盘 Arrow 文件里,Python 对象只是一层视图"这个设计哲学的延续。**这也解释了为什么这几个操作可以自由链式调用**(`ds.filter(...).shuffle(...).select(...)`)而不用担心中间产生大量临时大文件。

**AI 研究/工程场景:** 09 类"数据规模对微调效果的影响"知识点会用 `.select(range(N))` 从完整数据集里子采样不同规模做对比实验;`.shuffle(seed=42)` 里显式传固定 `seed` 是保证实验可复现的标准做法,不传 seed 每次运行顺序都不一样,会给"为什么这次结果和上次不一样"的排查徒增困扰。

**可运行例子:**
```python
from datasets import load_dataset

ds = load_dataset("timdettmers/openassistant-guanaco", split="train")

filtered = ds.filter(lambda x: len(x["text"]) < 200)
assert len(filtered) < len(ds)          # 筛选后条数应该变少
assert all(len(x["text"]) < 200 for x in filtered.select(range(min(50, len(filtered)))))  # 抽查前50条确实满足条件

selected = ds.select([0, 5, 10])
assert len(selected) == 3
assert selected[0]["text"] == ds[0]["text"]   # select保持原始内容,只是取子集
assert selected[1]["text"] == ds[5]["text"]   # 下标1对应原数据集下标5

shuffled = ds.shuffle(seed=42)
assert len(shuffled) == len(ds)               # 打乱不改变条数
assert shuffled[0]["text"] != ds[0]["text"]   # 顺序确实变了(极小概率巧合相同,可忽略)

# seed相同,shuffle结果可复现
shuffled_again = ds.shuffle(seed=42)
assert shuffled[0]["text"] == shuffled_again[0]["text"]  # 同seed,结果完全一致

print(f"OK: filter后{len(filtered)}条(原{len(ds)}条),select精确取出对应下标,shuffle(seed=42)可复现")
```
本机实测:`filter(text<200字符)` 后剩 478 条(原 9846 条);`select([0,5,10])` 精确对应原数据集的这几个下标;`shuffle(seed=42)` 两次调用结果完全一致,确认可复现性。

**面试怎么问 + 追问链:** "`shuffle(seed=42)` 的可复现性在分布式训练场景下还成立吗?" → 追问"多个 worker 各自读取数据分片时,怎么保证既随机又不重复?"(这已经触及 06 类 accelerate/分布式数据加载的话题——标准做法通常是每个 worker 用"同一个 seed + 自己的 rank 编号"生成确定性但各不相同的打乱顺序/分片方式,保证整体覆盖所有数据且互不重复,这是分布式数据加载正确性的一个容易被忽视的细节)。

**常见坑:**
1. `.filter()` 需要对**每一行**都跑一遍判断函数,数据量大时这本身是一次完整遍历,不是"瞬时"操作——如果要多次用不同条件筛选同一份数据,考虑是否能把多个条件合并成一次遍历,而不是链式调用多次 `.filter()`。
2. `ds.select(range(N))` 和 `ds.shuffle().select(range(N))` 是完全不同的操作——前者取的是数据集**原始顺序**的前 N 条,后者是打乱之后的前 N 条(相当于随机采样 N 条)——09 类做数据子采样时要清楚自己想要的是哪一种。

---

## 7. 自定义本地数据集加载

**签名/是什么:**
```
load_dataset("json", data_files="my_data.jsonl", split="train")
load_dataset("csv", data_files="my_data.csv", split="train")
```
`load_dataset` 第一个参数除了 Hub 上的仓库名,也可以是内置的"数据格式加载器"名字(`"json"`/`"csv"`/`"text"`/`"parquet"` 等),配合 `data_files` 指向本地文件。

**一句话:** 不是所有数据都恰好已经打包成 Hub 仓库的形式,`load_dataset("json"/"csv", data_files=...)` 让任意本地结构化文件都能享受到知识点 1-6 讲的全套 `Dataset` 能力(Arrow 内存映射、`.map()`、`.filter()`、切分管理等)。

**底层机制/为什么这样设计:** 内部行为是"用对应的格式解析器(json/csv library)读取本地文件 → 转换成 Arrow 格式 → 之后的行为和从 Hub 下载的数据集完全一样"。这个设计的价值在于**统一了"数据来源"和"后续处理能力"这两个维度**——不管数据最初是从 Hub 下载的、还是自己攒的本地文件,一旦变成 `Dataset` 对象,能用的 API 是完全一致的一套,不需要为"本地数据"单独学一套不同的处理流程。

**AI 研究/工程场景:** 09 类的补充数据集(`databricks-dolly-15k`)如果要做"自定义格式转换"演示,或者研究中自己标注/生成的数据(比如从模型输出蒸馏出来的训练数据),都是先落成本地 json/jsonl 文件,再用这个机制接入标准的 `Dataset` 处理流程——这是真实工程里"私有数据"和"公开数据集"统一处理的标准路径。

**可运行例子:**
```python
import json
import tempfile
import pathlib
from datasets import load_dataset

tmpdir = pathlib.Path(tempfile.mkdtemp())
data = [
    {"instruction": "What is 2+2?", "response": "4"},
    {"instruction": "Capital of France?", "response": "Paris"},
]
jsonl_path = tmpdir / "custom.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for row in data:
        f.write(json.dumps(row) + "\n")

custom_ds = load_dataset("json", data_files=str(jsonl_path), split="train")

assert len(custom_ds) == 2
assert set(custom_ds.features.keys()) == {"instruction", "response"}
assert custom_ds[0]["instruction"] == "What is 2+2?"
assert custom_ds[0]["response"] == "4"

# 加载出来的是标准Dataset对象,知识点1-6讲的能力全部可用,不是简配版
assert type(custom_ds).__name__ == "Dataset"
mapped = custom_ds.map(lambda x: {"combined": x["instruction"] + " -> " + x["response"]})
assert mapped[0]["combined"] == "What is 2+2? -> 4"

print(f"OK: 本地jsonl加载出{len(custom_ds)}条标准Dataset对象,.map()等能力完全可用")
```
本机实测:自建的 2 行 jsonl 文件正确加载,字段名自动从 JSON key 推断(`instruction`/`response`),`.map()` 能力和从 Hub 加载的数据集完全一致。

**面试怎么问 + 追问链:** "自己的数据是一个大的 JSON 数组文件(不是 jsonl 逐行格式),能直接用吗?" → 能,`load_dataset("json", data_files=...)` 两种格式都支持,库内部会自动探测;追问"如果本地数据分散在很多个文件(比如每天一个文件)怎么办?"(`data_files` 可以传一个列表或者 glob 通配符模式,一次性加载多个文件合并成一个 `Dataset`)。

**常见坑:**
1. `data_files` 传的路径是**相对于当前工作目录**还是**绝对路径**容易搞混,尤其是脚本从不同目录被调用时——生产代码建议统一用绝对路径,或者显式基于脚本自身位置(`__file__`)构造路径,避免"在我机器上能跑"的问题。
2. JSON/CSV 格式解析对字段类型的推断是自动的,如果不同行同一个字段类型不一致(比如某几行这个字段是数字、其他行是字符串),加载时可能报类型冲突错误或者被强制转换成某种类型——数据质量问题最好在生成阶段就保证一致,不要依赖加载器的自动纠错。

---

## 8. `.map()` 缓存失效机制

**签名/是什么:**
```
mapped1 = ds.map(fn)   # 第一次调用:真正执行fn,结果连同"指纹"一起缓存到磁盘
mapped2 = ds.map(fn)   # 第二次调用,fn函数体没变:直接复用缓存,不重新执行
```
`.map()` 的结果会被缓存(写成新的 Arrow 文件),**下次用同样的输入数据集 + 同样的处理函数再调用一次,会直接读缓存,不会重新跑一遍处理函数**。

**一句话:** 判断"能不能复用缓存"靠的是给这次调用计算一个**指纹(fingerprint)**——只要指纹相同就复用缓存,指纹不同(哪怕只是函数内部逻辑改了一个字符)就会重新计算,这个机制既能节省重复计算,也可能在你没意识到的情况下"用了旧缓存而不自知"。

**底层机制/为什么这样设计:** 指纹是基于"输入数据集本身的哈希 + 处理函数的序列化表示 + 调用参数"综合计算出来的一个哈希值。这样设计是为了自动化"这次调用和之前是不是完全一样"的判断,不需要用户手动管理缓存文件名/版本号。**代价是这套自动判断不是万能的**——如果处理函数依赖了某个外部状态(比如引用了一个全局变量,而这个变量的值在两次调用之间变了,但函数本身的代码没变),指纹可能算出"一样",于是错误地复用了本该失效的旧缓存,这是一个真实存在的边界情况,不是这个机制的 bug,是"自动指纹判断"这种设计的固有局限。

**AI 研究/工程场景:** 数据预处理步骤(尤其涉及慢速操作,比如调用外部 API 做数据增强)如果每次调试脚本的其他部分都要重新跑一遍,会严重拖慢迭代速度;`.map()` 的自动缓存机制让"改了下游代码,重跑脚本"这件事不需要重新计算上游已经跑过的数据处理,这也是为什么本篇很多知识点的例子在重复调用时会明显更快。

**可运行例子:**
```python
from datasets import load_dataset

ds = load_dataset("timdettmers/openassistant-guanaco", split="train")
small = ds.select(range(10))

def to_upper(x):
    return {"upper": x["text"].upper()}

mapped1 = small.map(to_upper)
fp1 = mapped1._fingerprint

mapped2 = small.map(to_upper)  # 同一个函数对象,再调用一次
fp2 = mapped2._fingerprint

assert fp1 == fp2  # 指纹相同,说明第二次调用识别出"这是同一次计算",会复用缓存

# 换一个函数体不同的函数(哪怕逻辑效果类似),指纹应该不同
def to_upper_v2(x):
    return {"upper": x["text"].upper() + ""}  # 故意加一个无意义的空字符串拼接,函数体变了

mapped3 = small.map(to_upper_v2)
fp3 = mapped3._fingerprint
assert fp1 != fp3  # 函数体变了,指纹不同,不会错误复用mapped1的缓存

print(f"OK: 同一函数两次调用指纹相同({fp1[:8]}...),函数体变化后指纹不同({fp3[:8]}...)")
```
本机实测:同一个 `to_upper` 函数两次调用产出相同的 `_fingerprint` 值,函数体哪怕只改了一个无意义的空字符串拼接,`_fingerprint` 也会变化——确认指纹机制精确到函数字节码层面,不是粗粒度的"看起来像不像"。

**面试怎么问 + 追问链:** "`.map()` 的缓存机制在什么情况下会'骗'到你?" → 这是一道很好的"知道原理边界"考题,答案要点:处理函数如果依赖闭包捕获的外部可变状态(比如引用了一个之后被修改的列表/字典,而不是把它当参数传进来),指纹计算不到这种"隐藏依赖",可能错误复用旧缓存——正确做法是让处理函数是**纯函数**(所有依赖都通过参数显式传入),或者调试时确实怀疑缓存有问题,用 `load_dataset(..., download_mode="force_redownload")`/`.map(..., load_from_cache_file=False)` 强制跳过缓存重新计算。

**常见坑:**
1. 调试处理函数时如果发现"明明改了代码,结果却没变",第一反应应该怀疑是不是命中了旧缓存(尤其是用 lambda 或者闭包引用了外部状态的写法),`load_from_cache_file=False` 是快速排除这个可能性的手段。
2. 缓存文件会持续占用磁盘空间(尤其是反复用不同参数调试同一个处理流程时,每次不同的指纹都会产生一份新的缓存文件),长期开发同一个数据处理脚本,偶尔清理 `~/.cache/huggingface/datasets` 下的旧缓存文件是必要的磁盘管理。

---

*本篇 8 个知识点全部在仓库根目录 `.venv` 真实验证通过。知识点3现场触发并记录了一次真实的 Windows 多进程 spawn 事故,详见该知识点和 00-roadmap.md 环境声明。*
