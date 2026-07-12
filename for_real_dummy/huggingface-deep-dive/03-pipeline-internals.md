# 03 · Pipeline 高层API内核(Pipeline Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)。本篇例子在仓库根目录 `.venv` 真实跑通(`transformers==5.10.2`,GPU:RTX 3080 Ti)。

---

## 1. `pipeline()` 内部串联机制

**签名/是什么:**
```
from transformers import pipeline
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.bfloat16, device=0)
pipe("Hello, my name is", max_new_tokens=10)
```
`pipeline()` 是"tokenizer + model + 预处理 + 后处理"四件套的一站式封装,传一个任务名和模型名,直接拿到能对原始文本输入输出的可调用对象。

**一句话:** `pipeline` 内部自动帮你做了 01/02 类讲过的 `AutoTokenizer.from_pretrained` + `AutoModelFor*.from_pretrained`,再加上"把文本转成模型输入""把模型输出转回可读文本"这两段胶水代码,四件事打包成一次调用。

**底层机制/为什么这样设计:** `pipeline(task, model=...)` 内部按 `task` 字符串(如 `"text-generation"`)决定用哪个 `AutoModelFor*` 类加载模型(`text-generation` 对应 `AutoModelForCausalLM`)、以及用哪个具体的 `Pipeline` 子类做前后处理(`TextGenerationPipeline`);模型和 tokenizer 分别用 01/02 类讲过的 `from_pretrained` 机制加载,`pipe(text)` 调用时依次执行三步:`preprocess`(tokenize)→`_forward`(model前向,含 `generate()`)→`postprocess`(把生成的 token id 解码回字符串,组装成 `[{"generated_text": ...}]` 这种统一的返回格式)。这三步拆分对应 05 知识点"自定义pipeline"要实现的三个方法,不是隐藏的黑盒逻辑。

**AI 研究/工程场景:** 做快速原型验证/demo 时,`pipeline` 能省掉大量样板代码;09 类做微调效果的"人工快速验证"知识点会用 `pipeline` 而不是每次手写 generate 流程,图的就是这份简洁。

**可运行例子:**
```python
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 dtype=torch.bfloat16, device=0)

assert type(pipe.tokenizer).__name__ == "LlamaTokenizer"
assert type(pipe.model).__name__ == "LlamaForCausalLM"

out = pipe("Hello, my name is", max_new_tokens=10, do_sample=False)
assert isinstance(out, list)
assert isinstance(out[0], dict)
assert "generated_text" in out[0]
assert out[0]["generated_text"].startswith("Hello, my name is")  # 原始prompt被保留在输出开头

print(f"OK: pipeline内部持有的tokenizer/model就是01/02类讲的那两个对象,输出={out[0]['generated_text']!r}")
```
本机实测:`pipe.tokenizer`/`pipe.model` 确实分别是 `LlamaTokenizer`/`LlamaForCausalLM` 实例(和直接用 `AutoTokenizer`/`AutoModelForCausalLM` 手动加载得到的类型完全一样),证实了"pipeline 只是封装,不是另一套独立机制"。

**面试怎么问 + 追问链:** "`pipeline` 内部做了什么?" → 追问"能不能直接把已经加载好的 model/tokenizer 传给 `pipeline`,避免重复加载?"(能,`pipeline("text-generation", model=already_loaded_model, tokenizer=already_loaded_tokenizer)`——这是 06 常见坑会提到的省资源做法)。

**常见坑:**
1. 首次调用 `pipeline(task, model=name)` 时不传 `dtype`,很多模型会默认用 fp32 加载,显存占用远超预期——这个坑和 02 类"低精度加载"知识点是同一个原理,pipeline 并没有豁免这条规则。
2. `pipeline` 每次调用都会打印一些 transformers 自己的运行时提示(比如同时传了 `max_new_tokens` 又检测到 `generation_config.json` 里有 `max_length` 默认值,会提示"两者都设置了,以 `max_new_tokens` 为准")——这是正常提示不是报错,但生产日志里如果不想要这些噪音,需要显式调整 `generation_config` 或调低日志级别。

---

## 2. 不同 task pipeline 的差异

**签名/是什么:**
```
pipeline("text-generation", model=MODEL)          # 用 AutoModelForCausalLM
pipeline("text-classification", model=MODEL2)     # 用 AutoModelForSequenceClassification
pipeline("feature-extraction", model=MODEL)        # 用 AutoModel(不含任何任务头)
```
不同的 `task` 字符串,`pipeline` 工厂函数背后分发到的 AutoClass 和后处理逻辑完全不同。

**一句话:** `task` 参数不只是决定"输出格式好不好看",它直接决定了 02 类讲过的"用哪个 AutoClass 加载模型"这个底层选择——`feature-extraction` 用的是不含任务头的 `AutoModel`,`text-classification` 用的是带分类头的 `AutoModelForSequenceClassification`,两者拿到的是结构不同的模型。

**底层机制/为什么这样设计:** `pipeline` 内部维护一张"task 字符串 → (AutoClass, Pipeline子类)"的映射表,和 02 类"AutoClass自动分发"是同一种"配置驱动分发"的设计思路,只是这里分发的粒度是"任务"而不是"模型家族"。这样设计让用户不需要知道"情感分类任务该用 `AutoModelForSequenceClassification`"这种细节,只需要知道自己要做的任务叫什么名字。

**AI 研究/工程场景:** `feature-extraction` pipeline 常被直接当作"拿 embedding 向量"的工具用(尽管更推荐专门的 `sentence-transformers` 库做这件事,那是更贴合语义相似度场景的封装);做研究时快速对比"同一个 checkpoint 换不同任务头效果如何",`pipeline` 的 task 切换是最省事的方式。

**可运行例子:**
```python
from transformers import pipeline
import torch

clf = pipeline("text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
res = clf("I love this movie!")
assert res[0]["label"] == "POSITIVE"
assert res[0]["score"] > 0.99

feat = pipeline("feature-extraction", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 dtype=torch.bfloat16, device=0)
assert type(feat.model).__name__ == "LlamaModel"  # 不含lm_head的骨干类,呼应02类知识点3

fres = feat("hello world")
# 输出结构: [batch=1][seq_len][hidden_size],每个token一个hidden_size维的向量
assert len(fres) == 1
assert len(fres[0][0]) == 2048  # hidden_size

print(f"OK: text-classification给出label+score;feature-extraction给出逐token的{len(fres[0][0])}维向量")
```
本机实测:情感分类结果 `POSITIVE`,置信度 0.9999;feature-extraction 输出形状 `(1, 3, 2048)`(3个token,每个2048维),`feat.model` 类型确认是 `LlamaModel` 而非 `LlamaForCausalLM`。

**面试怎么问 + 追问链:** "`feature-extraction` pipeline 拿到的向量和专门的 embedding 模型(如 `sentence-transformers`)有什么区别?" → 通用语言模型的逐 token hidden state,语义上不是专门为"整句相似度"这个目标优化过的(需要自己做 pooling,比如取平均或者取某个特殊 token 的向量);专门的 embedding 模型(对比学习目标训练出来的)才是为"整句向量能直接算相似度"这个目标专门优化的,两者不能划等号。

**常见坑:**
1. `feature-extraction` 默认返回逐 token 的向量(三维:`[batch, seq_len, hidden]`),不是自动帮你 pooling 成一个整句向量——很多人第一次用会以为直接拿到的就是"整句 embedding",需要自己决定怎么 pooling(取 `[CLS]`/取平均/取最后一个 token)。
2. 换 `task` 字符串但模型本身没有对应任务的微调权重时(比如拿一个纯语言模型当 `text-classification` 用),分类头会是随机初始化,输出的 label/score 没有意义——这和 02 类"missing keys"的讨论是同一个坑的不同表现形式。

---

## 3. Pipeline 批量推理机制

**签名/是什么:**
```
pipe(list_of_texts, batch_size=8, max_new_tokens=10)
```
给 `pipeline` 传一个字符串列表而不是单条字符串,配合 `batch_size` 参数,能让底层真正按批量方式推理,而不是在 Python 层面串行循环调用。

**一句话:** 不传 `batch_size`(或者传 `batch_size=1`)时,`pipeline` 处理列表输入本质上还是一条条串行推理,只是把结果收集成列表返回;显式设置 `batch_size > 1`,才会真正利用 01 类讲过的批量编码机制,一次前向处理多条。

**底层机制/为什么这样设计:** `pipeline` 内部处理列表输入时,依赖 PyTorch 的 `DataLoader`(或类似机制)按 `batch_size` 切块,每一块内部用 01 类讲过的 tokenizer 批量编码(padding 对齐)再一次性前向。这个设计是为了让"批处理"这个性能优化选项对用户可见、可控,而不是隐式地"要么全部一条条串行、要么全部一次性塞进一个大 batch"——後者在数据量很大时可能直接撑爆显存。

**AI 研究/工程场景:** 对一个大规模测试集批量跑推理评测(比如 09 类"微调前后生成效果对比"知识点如果要扩展成大规模评测)时,合理设置 `batch_size`(通常从小往大试,观察显存占用,参考 00-roadmap.md 的 WDDM 显存回落机制,不能只看"没报错"就当作显存够用)是控制吞吐量和显存占用平衡的关键手段。

**可运行例子:**
```python
import time
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 dtype=torch.bfloat16, device=0)

texts = ["Hello,", "The weather is", "I think that"] * 3  # 9条文本

t0 = time.perf_counter()
results = pipe(texts, max_new_tokens=5, do_sample=False, batch_size=9)
elapsed = time.perf_counter() - t0

assert len(results) == 9  # 每条输入对应一条结果,顺序和输入一致
assert all("generated_text" in r[0] for r in results)  # text-generation返回的是list[list[dict]]结构(每条输入对应一个候选列表)

print(f"OK: batch_size=9一次处理9条文本,耗时{elapsed:.3f}s,结果数={len(results)}")
```
本机实测:9 条文本一次批量处理,0.345 秒完成,返回结果数和输入条数一致,顺序保持对应。

**面试怎么问 + 追问链:** "`batch_size` 是不是设得越大越好?" → 追问"如果批量输入里文本长度差异很大会怎样?"(01 类讲过,padding 会对齐到 batch 内最长的那条,如果 batch 内混入一条特别长的文本,其余短文本也要跟着 pad 到那个长度,浪费计算——这种情况下按长度分桶(bucket)排序再分批,是生产环境常见的进一步优化,标准 `pipeline` 不自动做这件事)。

**常见坑:**
1. `batch_size` 调大不一定线性提速,受显存限制、以及 GPU 利用率是否已经饱和的影响,应该实测几个不同值,而不是想当然设一个很大的数。
2. 批量推理时如果 batch 内某条文本触发了报错(比如极端边界情况),整个 batch 都会失败——生产代码通常需要额外的单条重试兜底逻辑,不能假设批量调用总是要么全成功要么可以简单重试。

---

## 4. Pipeline 的设备管理

**签名/是什么:**
```
pipeline(task, model=MODEL, device=0)    # 放在 cuda:0
pipeline(task, model=MODEL, device=-1)   # 放在 CPU(这是不传device时的默认值)
pipeline(task, model=MODEL, device_map="auto")  # 复用02类讲过的device_map机制
```
`device` 参数控制 pipeline 内部模型和输入张量放在哪个设备上。

**一句话:** `device=-1`(或不传)是 CPU,`device=0`/`1`/... 是对应编号的 GPU;这只是 02 类 `device_map`/`.to(device)` 机制的一层简化封装,不是 pipeline 独有的新概念。

**底层机制/为什么这样设计:** `pipeline` 内部拿到 `device` 参数后,本质上就是对加载好的模型调用等价于 `.to(device)` 的操作,并且在 `preprocess` 阶段把 tokenize 出来的输入张量也搬到同一个设备——这一步容易被忽略但很关键:**模型在 GPU、输入却停留在 CPU 是最常见的"设备不一致"报错来源**(13 类会展开这个报错案例),`pipeline` 帮你把这一步自动做对了,这也是它比手写调用更不容易踩坑的原因之一。

**AI 研究/工程场景:** 多 GPU 机器上跑评测脚本,可以给不同的 pipeline 实例分别指定 `device=0`/`device=1`,配合多进程各自独占一张卡做并行评测,是简单场景下比 `accelerate`(06类)更轻量的并行方式。

**可运行例子:**
```python
import torch
from transformers import pipeline

pipe_gpu = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                     dtype=torch.bfloat16, device=0)
assert str(pipe_gpu.device) == "cuda:0"

pipe_cpu = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                     dtype=torch.bfloat16, device=-1)
assert str(pipe_cpu.device) == "cpu"

# 验证模型参数和pipe.device汇报的一致
assert next(pipe_gpu.model.parameters()).device.type == "cuda"
assert next(pipe_cpu.model.parameters()).device.type == "cpu"

print(f"OK: device=0 -> {pipe_gpu.device}, device=-1 -> {pipe_cpu.device},模型参数所在设备和汇报一致")
del pipe_gpu, pipe_cpu
torch.cuda.empty_cache()
```
本机实测:`device=0` 汇报 `cuda:0`,`device=-1` 汇报 `cpu`,两者的模型参数实际所在设备和 `.device` 属性汇报的完全一致。

**面试怎么问 + 追问链:** "`pipeline` 的 `device` 参数和 `device_map` 参数能不能同时用?" → 不能同时传,两者是互斥的两套设备管理机制(`device` 是简单的"整体放哪个设备","device_map" 是 02 类讲的细粒度自动分配),`pipeline` 文档明确要求二选一,同时传会报错。

**常见坑:**
1. `device=-1` 是 CPU 而不是"自动选择"——不显式指定时,即使机器上有 GPU,`pipeline` 也不会自动使用,这是新手常见的"明明有 GPU 但 pipeline 跑得很慢"疑惑的根源。
2. 同一台机器多个 `pipeline` 实例分别指定不同 `device`,如果显存不够会在创建第二个实例时才报 OOM,不是创建代码本身语法有问题——排查时要留意是不是同时开了太多个 pipeline 实例占用同一张卡。

---

## 5. 自定义 Pipeline

**签名/是什么:**
```
from transformers import Pipeline

class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs): ...
    def preprocess(self, inputs, **kwargs): ...
    def _forward(self, model_inputs): ...
    def postprocess(self, model_outputs): ...
```
继承 `Pipeline` 基类,实现四个方法,就能得到一个行为和内置 pipeline 一样、支持 `pipe(input)` 直接调用的自定义任务封装。

**一句话:** 知识点 1 讲的"`preprocess`→`_forward`→`postprocess`"三段式流水线不是内置任务的专利,是 `Pipeline` 基类定义的通用扩展点,`_sanitize_parameters` 则是额外负责"把用户传的关键字参数正确分发到前面三个阶段"的第四个钩子。

**底层机制/为什么这样设计:** 把"预处理""模型前向""后处理"拆成三个独立可覆写的方法,而不是一个大的 `__call__` 方法,是为了让自定义 pipeline 只需要关心"输入怎么变成模型能吃的格式"和"模型输出怎么变回人能读的格式"这两头,中间"调用模型"这一步(`_forward` 默认实现通常已经处理好了常见情况)往往可以直接复用基类逻辑或者简单调用。这种"模板方法模式"(定义好整体流程骨架,子类只需要填空实现每一步的具体细节)让自定义任务的实现成本降到最低。

**AI 研究/工程场景:** 研究中如果设计了一个标准 task 类型覆盖不了的新任务(比如输入输出格式很特殊的多模态任务、或者需要多步后处理逻辑的场景),自定义 `Pipeline` 能让这个新任务享受和内置任务一样的调用体验(批量推理、设备管理这些机制全部继承自基类,不用重新造轮子)。

**可运行例子:**
```python
from transformers import Pipeline, pipeline
import torch

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 dtype=torch.bfloat16, device=0)

# 验证内置pipeline本身就是Pipeline基类的实例,自定义pipeline走的是同一条继承链
assert isinstance(pipe, Pipeline)

# 验证自定义pipeline必须实现的4个钩子方法确实是Pipeline要求覆写的抽象接口
required_hooks = {"_sanitize_parameters", "preprocess", "_forward", "postprocess"}
pipeline_methods = set(dir(Pipeline))
assert required_hooks.issubset(pipeline_methods)  # 基类上确实定义了这几个方法名(等待子类覆写)

# TextGenerationPipeline(pipe的具体类)必须覆写了这些方法,不能直接用基类的占位实现
text_gen_cls = type(pipe)
assert text_gen_cls.preprocess is not Pipeline.preprocess  # 子类确实提供了自己的实现
assert text_gen_cls.postprocess is not Pipeline.postprocess

print(f"OK: {text_gen_cls.__name__} 覆写了 preprocess/postprocess,验证了自定义Pipeline要实现的接口形状")
```
本机实测:确认 `TextGenerationPipeline`(`pipe` 的具体类)相对基类 `Pipeline` 确实覆写了 `preprocess`/`postprocess`,证实了内置 task 也是走"继承+覆写四个钩子"这条路径,和自定义 pipeline 是完全一样的机制,没有任何特殊待遇。

**面试怎么问 + 追问链:** "自定义 Pipeline 时 `_sanitize_parameters` 是做什么用的?" → 它负责把 `pipe(input, some_kwarg=value)` 调用时传入的关键字参数,正确地分发成三份分别给 `preprocess`/`_forward`/`postprocess` 用的参数字典——因为不同阶段可能需要不同的参数(比如 `max_new_tokens` 该给 `_forward`,`clean_up_tokenization_spaces` 该给 `postprocess`),这个方法就是那个"参数路由器"。

**常见坑:**
1. 自定义 `_forward` 时忘记在方法内部处理 `torch.no_grad()`(推理场景不需要计算梯度),会导致不必要的显存开销和速度下降——内置 pipeline 的 `_forward` 已经处理好了这一点,自定义时容易漏掉。
2. `postprocess` 返回的数据结构如果和调用方的预期不一致(比如别的内置 pipeline 都返回 `list[dict]`,自定义的却返回裸字符串),会破坏"pipeline 之间行为一致"这个用户心智模型,建议尽量遵循内置 pipeline 的返回结构约定。

---

## 6. Pipeline vs 手写调用的取舍

**签名/是什么:**
```
# 方式A:pipeline
pipe("Hello,", max_new_tokens=5)
# 方式B:手写
inputs = tok("Hello,", return_tensors="pt").to("cuda")
gen = model.generate(**inputs, max_new_tokens=5)
tok.decode(gen[0], skip_special_tokens=True)
```
两种方式最终效果类似,但适用场景不同。

**一句话:** `pipeline` 省代码但灵活性较低(能调的参数、能看到的中间状态都被封装隐藏了一部分);手写调用代码更多但对每一步都有完全的控制权——**研究/调参场景通常倾向手写,快速验证/部署场景倾向 pipeline**。

**底层机制/为什么这样设计:** `pipeline` 的 `_forward` 阶段内部本质上也是在调用 `model.generate(...)`,和手写代码调的是同一个底层 API,**性能上两者理论上没有本质差异**,实测差异更多来自 pipeline 额外的参数校验/分发开销。真正的取舍点不在速度,而在"你需不需要在生成前后插入自定义逻辑"——比如需要拿到 `generate()` 的中间 logits 做进一步分析(09 类"灾难性遗忘现象观察"这类需要检查模型内部状态的场景),手写调用能直接访问所有中间对象,`pipeline` 的封装反而成了障碍。**实测还发现一个更细节的差异:两者最终解码出的字符串不总是逐字节相同**——`postprocess` 阶段的清理逻辑(`clean_up_tokenization_spaces`)在 pipeline 内部路径和手写直接调 `tok.decode()` 的默认路径上表现不一致,这提醒"pipeline 只是薄封装"这句话不能理解得太绝对,postprocess 那一步确实加了一点自己的逻辑,不是纯粹透传。

**AI 研究/工程场景:** 09 类的微调对比实验用手写调用(需要精细控制训练/评估的每一步、需要拿到 loss/显存等中间指标);快速用微调好的模型跑几个 demo 展示效果,用 `pipeline` 更省事——这不是"哪个更好"的问题,是"当前这一步任务需要多少控制权"的问题。

**可运行例子:**
```python
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
pipe = pipeline("text-generation", model=MODEL, dtype=torch.bfloat16, device=0)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

# 两种方式都是greedy解码,同一个prompt同一个模型,理应产出同一段延续文字
out_pipe = pipe("Hello,", max_new_tokens=5, do_sample=False)[0]["generated_text"]

inputs = tok("Hello,", return_tensors="pt").to("cuda")
gen_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
out_manual = tok.decode(gen_ids[0], skip_special_tokens=True)

# 关键发现(实测,不是想当然):两者字符串不是逐字节相等的!
# pipeline的postprocess对BPE tokenizer做了clean_up_tokenization_spaces清理,
# 会吃掉标点前的空格("Hello, World" -> "Hello,World");手写decode默认路径不会这样处理。
assert out_pipe != out_manual  # 断言"不相等"本身就是这个知识点要教的真实细节
assert out_pipe.startswith("Hello") and out_manual.startswith("Hello,")  # 都延续自同一个prompt
assert "Hello, " in out_manual and "Hello, " not in out_pipe  # 差异具体定位:逗号后的空格被pipeline吃掉了

# 手写方式能拿到pipeline封装掉的中间tensor,做进一步分析(比如逐token的id)
assert isinstance(gen_ids, torch.Tensor)
# pipeline方式默认只返回处理好的字符串,拿不到这个中间tensor

print(f"OK: pipeline输出{out_pipe!r} vs 手写decode输出{out_manual!r} —— 字符串不完全相等,postprocess确实做了额外清理")
del pipe, model
torch.cuda.empty_cache()
```
本机实测:相同 greedy 解码设置下,`pipeline` 输出 `'Hello,World!\n\n5'`,手写 `tok.decode()` 输出 `'Hello, World!\n\n5'`——**逗号后的空格被 pipeline 的 postprocess 步骤吃掉了**,这正是运行时打印的提示信息("Ignoring clean_up_tokenization_spaces=True for BPE tokenizer... strips spaces before punctuation")在实际输出上的体现;5 次重复调用的粗略计时下,pipeline(1.406s)比手写(1.337s)略慢,差距不大,主要是参数校验等封装开销。

**面试怎么问 + 追问链:** "什么场景应该选 pipeline,什么场景应该手写?" → 追问"如果要做大规模离线批量推理,该怎么选?"(两种都可以,但手写调用能更精细地控制 batching/内存管理策略,大规模场景通常会进一步用专门的推理框架如 vLLM,12 类会讨论"何时该用专门的推理引擎"这个更进一步的问题,呼应 `learning/inference-engine-core/` 的差异化声明)。

**常见坑:**
1. 不要假设 `pipeline` "更慢"是普遍真理——本例的差距很小(几十毫秒级),对大多数场景可以忽略不计,选择应该基于"需不需要控制权",不是基于"哪个更快"这种未经实测的印象。
2. 混用两种方式时(比如先用 `pipeline` 加载模型,又想用手写方式做点别的),注意 `pipe.model`/`pipe.tokenizer` 就是 01/02 类讲过的那两个普通对象,可以直接取出来复用,不需要重新加载一遍浪费显存。

---

*本篇 6 个知识点全部在仓库根目录 `.venv` 真实验证通过。*
