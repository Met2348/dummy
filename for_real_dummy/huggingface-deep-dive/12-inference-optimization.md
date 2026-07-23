# 12 · 推理优化基础(Inference Optimization Basics)

> 总览见 [00-roadmap.md](00-roadmap.md)。已在仓库根目录 `.venv` 真实跑通。**差异声明:本篇讲怎么正确调用 `transformers.generate()` 把这一层用好,不是从零造一个推理引擎——系统工程视角(PagedAttention/CUDA kernel/continuous batching/自己实现KV cache管理)见 [`learning/inference-engine-core/`](../../learning/inference-engine-core/),两者完全不是一个海拔,这里的"优化"是"用对官方提供的参数和机制",不是"重新发明轮子"。**

---

## 1. `generate()` 内部机制总览

**签名/是什么:**
```
output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
```
`generate()` 是自回归语言模型做"续写"的标准入口——它不是一次前向调用,而是一个循环:每次前向拿到下一个 token 的概率分布,选一个 token 拼回输入序列,再喂回去做下一次前向,直到达到停止条件。

**一句话:** 模型本身(`model(input_ids)`)一次前向调用只能给出"下一个token的概率分布",不会自己产出一整段续写文字——`generate()` 就是把"预测下一个token→拼接→再预测"这个循环封装起来的工具函数,这也是为什么它的返回值是完整的 token 序列(输入+新生成的部分拼在一起),不是单步的 logits。

**底层机制/为什么这样设计:** 语言模型训练时学的是"给定前缀,预测下一个token"这个单步任务,生成连贯的多token文本必须靠**重复调用**这个单步能力来实现——`generate()` 内部按配置好的解码策略(knowledge point 2)、结合 KV cache(knowledge point 3)优化重复调用的效率、检查停止条件(遇到EOS或者达到`max_new_tokens`),把这一整套循环逻辑封装成一次函数调用,用户不需要自己手写这个循环。

**AI 研究/工程场景:** 03 类的 pipeline `_forward` 阶段内部调的就是 `generate()`;09 类"微调前后生成效果对比"、"灾难性遗忘现象观察"这些知识点都要用 `generate()` 产出模型的真实输出文本做人工/自动评估。

**可运行例子:**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
input_len = inputs["input_ids"].shape[1]

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

# generate()返回的是"输入+新生成内容"拼在一起的完整序列,不是只有新生成的部分
assert output_ids.shape[1] == input_len + 10
assert torch.equal(output_ids[0, :input_len], inputs["input_ids"][0])  # 前缀部分和输入完全一致

decoded = tok.decode(output_ids[0], skip_special_tokens=True)
assert decoded.startswith("The capital of France is")

print(f"OK: generate()输出长度={output_ids.shape[1]}(输入{input_len}+新增10),前缀确认和输入一致,文本={decoded!r}")
```
本机实测:输入 6 个 token,`max_new_tokens=10` 后输出恰好 16 个 token;输出序列的前 6 个 token 和输入逐 token 相等(确认 `generate()` 是"拼接"而不是"只返回新内容")。真实生成文本:`"The capital of France is Paris.\n\n2. B. The capital"`(小模型贪心解码的真实输出,续写质量不追求完美,重点是验证机制)。

**面试怎么问 + 追问链:** "`generate()` 返回的序列长度一定是 `输入长度+max_new_tokens` 吗?" → 追问"什么情况下实际长度会更短?"(如果模型在达到 `max_new_tokens` 之前就生成了 EOS token,默认行为是提前停止——这种情况下输出长度会小于"输入+max_new_tokens",knowledge point 6 的 `StoppingCriteria` 是这个"提前停止"机制的可定制扩展)。

**常见坑:**
1. 忘记 `skip_special_tokens=True` 解码,输出文本里会混入 `<s>`/`</s>` 这类特殊标记的字面文本,容易被误认为是模型生成的内容。
2. `generate()` 默认不会计算梯度,但如果忘记包在 `torch.no_grad()` 里,依然会额外占用显存构建不需要的计算图——虽然不会报错,但是纯粹的资源浪费,推理场景养成显式 `torch.no_grad()` 的习惯。

---

## 2. 解码策略对比

**签名/是什么:**
```
model.generate(**inputs, do_sample=False)                              # greedy:每步选概率最高的token
model.generate(**inputs, do_sample=True, temperature=1.5, top_k=50)     # 采样:引入随机性
```
`do_sample` 是控制"每一步该怎么从概率分布里选出下一个token"这件事的总开关。

**一句话:** **greedy 解码是完全确定性的**(同样的输入、同样的模型,反复跑结果逐字节相同);**采样解码引入随机性**(`temperature`/`top_k`/`top_p` 控制随机性的"温度"和候选范围),不设固定随机种子的话,同样的输入每次跑出来的文本可能不一样。

**底层机制/为什么这样设计:** 每一步模型输出的是一个覆盖全部词表的概率分布(经过 softmax),greedy 解码直接取 `argmax`(概率最高的那个token,无随机性来源);采样解码则是把这个分布当成一个真正的概率分布去抽样——`temperature` 在 softmax 之前对 logits 做缩放(温度越高,分布越平坦,越倾向于"雨露均沾"、随机性越强;温度越低越接近 greedy 的确定性行为),`top_k`/`top_p` 则是在抽样前先把概率分布截断到"最可能的k个候选"或者"累计概率达到p的候选集合",避免采样到极小概率的荒谬续写。

**AI 研究/工程场景:** 09 类的"微调前后生成效果对比"用 `do_sample=False`(greedy)确保多次跑同一个 prompt 得到可复现、可直接比较的结果;需要展示"模型创造力/多样性"的场景(比如故事续写)则更适合用采样解码。

**可运行例子:**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
inputs = tok("The capital of France is", return_tensors="pt").to("cuda")

with torch.no_grad():
    out_greedy_1 = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    out_greedy_2 = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    assert torch.equal(out_greedy_1, out_greedy_2)   # greedy:两次跑结果逐token相同

    torch.manual_seed(42)
    out_sample_1 = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.5, top_k=50)
    torch.manual_seed(123)
    out_sample_2 = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.5, top_k=50)
    assert not torch.equal(out_sample_1, out_sample_2)  # 采样:不同随机种子,结果大概率不同

print("OK: greedy解码可复现(2次结果逐token相等),采样解码在不同随机种子下产出不同结果")
```
本机实测:greedy 两次调用结果确认逐 token 相等;采样(temperature=1.5, top_k=50)在两个不同随机种子下确认产出不同结果。

**面试怎么问 + 追问链:** "为什么很多线上对话产品的默认设置既不是纯greedy也不是很高的temperature,而是中等的temperature+top_p组合?" → 纯 greedy 容易陷入重复/乏味的模式(尤其长文本生成时,greedy 有陷入循环重复的已知倾向);温度太高容易产出不连贯的胡言乱语——中等 temperature 配合 top_p(核采样)是在"确定性/连贯性"和"多样性/创造力"之间找一个经验上效果较好的折中点,这是工程实践里反复调优出来的经验,不是理论上唯一正确的选择。

**常见坑:**
1. 想要"可复现的采样结果"必须显式设置随机种子(`torch.manual_seed`),很多人以为"设置了 `do_sample=True` 但忘了固定种子"的不可复现是bug,实际是采样解码的正常特性。
2. `temperature=0` 在数学上会导致除零错误(温度是除数),很多实现里 `temperature=0` 会被特殊处理成等价于greedy解码,但不是所有版本/所有代码路径都保证这个特殊处理,想要greedy应该直接用 `do_sample=False`,不要依赖 `temperature=0` 这种间接写法。

---

## 3. KV Cache 机制

**签名/是什么:**
```
model.generate(**inputs, use_cache=True)    # 默认开启
```
自回归生成每一步都要重新计算 attention,`use_cache=True` 让模型**缓存**之前步骤已经算过的 Key/Value 张量,避免重复计算。

**一句话:** 不用 KV cache 时,生成第 N 个 token 需要对**从头到第N-1个token的全部序列**重新做一遍完整的前向计算;用 KV cache,只需要对**新增的这一个token**做计算,之前所有token的Key/Value直接复用缓存结果——这是自回归生成能实用化的关键优化,不开启的话生成长文本的耗时会随长度平方增长。

**底层机制/为什么这样设计:** self-attention 里每个 token 的 Key/Value 向量,只取决于**这个 token 自己**和它之前的上下文(causal attention 的性质决定了它不会看到未来的token),这意味着"已经算过的token,它的Key/Value不会因为后面新增了token而改变"——这个性质保证了缓存的正确性:可以放心复用之前算好的Key/Value,不用担心过期。`generate()` 返回值里的 `past_key_values` 字段(本例的可运行例子会展示)就是这份缓存的容器,每一步生成后被更新、传给下一步。

具体每一步到底省了哪部分计算,拿生成 `t4`、`t5` 这两步举例对比会比纯文字描述更直接(前提是已经生成了 `t1 t2 t3`):

```
不用 KV cache——每一步都要把"从头到当前位置"整段重新算一遍:
  生成 t4 这一步: 输入 [t1 t2 t3]      → 重新算 t1,t2,t3 三个位置的K/V(全是重复劳动)
  生成 t5 这一步: 输入 [t1 t2 t3 t4]   → 重新算 t1,t2,t3,t4 四个位置的K/V(前3个又重复算了一遍)
  ...
  生成第N个新token时要重算N-1个位置 → 总计算量 ≈ 1+2+...+N,是N的平方量级

用 KV cache——只算新增的那一个token,其余直接从缓存里取:
  生成 t4 这一步: cache里已有t1,t2,t3的K/V → 只新算t4的K/V,追加进cache
  生成 t5 这一步: cache里已有t1,t2,t3,t4的K/V → 只新算t5的K/V,追加进cache
  ...
  每一步都只新算1个位置 → 总计算量 ≈ N,是N的线性量级
```

这就是"一句话"里"不开启的话生成长文本的耗时会随长度平方增长"这个结论的直接来源:不用 cache 时每步的重复计算随位置数线性增加、累加起来是平方级;用 cache 后每步是常数计算量,累加起来是线性级——序列越长,平方和线性之间的差距被拉得越开,这也是下面"本机实测"提到的"50 token 这个规模还不足以体现量级差异,序列越长优化收益越明显"的具体原因。

**AI 研究/工程场景:** 长对话/长文档生成场景,KV cache 是让响应速度保持可用的必要机制;但缓存本身占用显存(缓存大小随序列长度、层数、注意力头数增长),这也是为什么长上下文模型的显存需求会显著高于短上下文——KV cache 本身也是一块不随权重量化而减少的显存开销(呼应 08 类"量化对显存的真实收益比例受非权重开销影响"的讨论)。

**可运行例子:**
```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
inputs = tok("The capital of France is", return_tensors="pt").to("cuda")

def timed_generate(use_cache, n_tokens=50):
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=use_cache)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False, use_cache=use_cache)
    torch.cuda.synchronize()
    return time.perf_counter() - t0

t_with_cache = timed_generate(use_cache=True)
t_without_cache = timed_generate(use_cache=False)

assert t_without_cache > t_with_cache   # 方向性断言:不用cache确实更慢,这是KV cache存在的意义

# 确认generate()返回对象里真的暴露了past_key_values这个缓存容器
with torch.no_grad():
    result = model.generate(**inputs, max_new_tokens=5, do_sample=False,
                             return_dict_in_generate=True)
assert "past_key_values" in result.keys()

print(f"OK: use_cache=True耗时{t_with_cache:.3f}s,use_cache=False耗时{t_without_cache:.3f}s(更慢),"
      f"generate()返回对象确认暴露past_key_values")
```
本机实测:生成 50 个新 token,`use_cache=True` 耗时 1.180s,`use_cache=False` 耗时 1.449s(约 1.23 倍慢)——**这个加速比例在这台机器、这个短序列长度下不算戏剧性**,序列越长、模型越大,不用KV cache的"重新计算全部历史"的代价会越发凸显,50 token 这个规模还不足以体现KV cache带来的量级差异,但方向性结论(有cache更快)已经真实确认。

**面试怎么问 + 追问链:** "KV cache的显存占用和什么因素成正比?" → 序列长度 × 层数 × 注意力头数(或者GQA场景下的KV头数,呼应01/02类讲过TinyLlama是GQA架构,KV头数远小于Q头数)× 每个头的维度——GQA架构本身就是为了压缩KV cache显存(减少KV头数量)而设计的,这是理解"为什么现代LLM大多采用GQA而不是标准MHA"的一个直接动机,和 knowledge point 3 这里的机制是同一件事的两个角度。

**常见坑:**
1. `use_cache=False` 不会报错,只是变慢——这是一个"性能而不是正确性"的坑,容易在代码审查中被忽略,除非专门做过性能测试。
2. 极长上下文场景下,KV cache本身可能成为显存瓶颈(比"权重占用"更大),这种场景下"要不要用更激进的KV cache压缩技术(量化KV cache等)"是一个专门的优化话题,不属于本知识点的基础范围,属于 `learning/inference-engine-core/` 更系统工程视角的内容。

---

## 4. `torch.compile` 与 HF 模型结合

**签名/是什么:**
```
compiled_model = torch.compile(model)
compiled_model.generate(**inputs, ...)   # 用法和普通model一样
```
`torch.compile`(呼应 torch-deep-dive 相关内容)把 PyTorch 的 eager 模式执行,通过 JIT 编译成更高效的融合 kernel。

**一句话:** **本机实测的真实结果和"编译一次、后续更快"这个教科书式预期不完全吻合**——`generate()` 这种自回归循环里,每一步的输入序列长度都在变化(动态shape),这会导致 `torch.compile` 反复触发重新编译,不是"预热一次、之后一直快"这么简单,本知识点如实记录了这个真实的复杂性,不美化成一个干净的成功案例。

**底层机制/为什么这样设计:** `torch.compile` 的加速原理是"针对具体的张量形状编译出优化过的计算图",遇到新的输入形状需要重新编译(或者退回eager模式)。`generate()` 自回归循环里,虽然"新增1个token"这个单步计算本身形状是固定的(配合KV cache,每步只处理1个新token),但结合 KV cache 增长、attention mask 变化等因素,`torch.compile` 在处理这类动态生成场景时不像处理"固定形状的单次前向"(比如做分类推理)那样能获得干净、可预测的加速效果——这是当前 `torch.compile` 生态在生成式场景的一个真实局限,不是配置错误导致的。

**AI 研究/工程场景:** 对**固定输入形状的场景**(比如批量对固定长度文本做一次性特征提取、不涉及自回归生成),`torch.compile` 通常能拿到更干净的加速效果;对 `generate()` 这类动态自回归场景,更成熟的加速方案通常来自专门的推理引擎(`learning/inference-engine-core/` 涉及的 continuous batching、CUDA graphs 等技术),而不是简单地在 `generate()` 外面套一层 `torch.compile`。

**可运行例子:**
```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
inputs = tok("The capital of France is", return_tensors="pt").to("cuda")

# 验证torch.compile不会破坏正确性(这是比"一定更快"更重要、更应该验证的基本要求)
with torch.no_grad():
    out_eager = model.generate(**inputs, max_new_tokens=5, do_sample=False)

compiled_model = torch.compile(model)
with torch.no_grad():
    out_compiled = compiled_model.generate(**inputs, max_new_tokens=5, do_sample=False)

assert torch.equal(out_eager, out_compiled)   # 核心要求:编译前后生成结果必须一致,不能为了"快"牺牲正确性

print("OK: torch.compile包装后generate()结果和eager模式完全一致(正确性优先于本例未强行断言的速度提升)")
```
本机实测:`torch.compile` 包装后的 `generate()` 输出和未编译版本逐 token 相等(正确性得到保证);**速度方面的真实观察**——"首次调用(含编译开销)"耗时 0.109s,"第二次调用"耗时 0.127s,并没有观察到教科书描述的"首次慢、后续明显更快"模式,这大概率和上面"底层机制"讨论的动态shape/KV cache增长导致的反复(部分)重新编译有关。**本知识点刻意不断言"compile更快"这个方向性结论**,因为实测数据不支持这个结论,只断言了"结果正确性不受影响"这个更有把握的事实——这本身也是一个真实、有价值的工程认知:`torch.compile` 不是"无脑加速"的万能开关,在自回归生成场景需要更细致的评估,不能想当然。

**面试怎么问 + 追问链:** "`torch.compile` 对 `generate()` 场景效果不明显,该怎么进一步优化?" → 追问"这和专门的推理引擎(vLLM等)解决这个问题的思路有什么不同?"(专门推理引擎通常会用 CUDA Graphs 之类的技术针对"固定的单步decode计算模式"做更针对性的图捕获和复用,而不是依赖通用的 `torch.compile` 动态编译机制去适配一个本质上形状会变化的循环——这是"通用JIT编译工具"和"领域专用推理引擎"在处理同一个问题时的不同工程路线,`learning/inference-engine-core/` 会更系统地讲这条路线)。

**常见坑:**
1. 不要看到 `torch.compile` 在别的固定形状场景(比如图像分类模型推理)有效,就默认它对任意模型/任意使用方式都有同等效果——本知识点是一个真实的反例,场景差异(固定形状 vs 动态自回归)会显著影响实际收益。
2. `torch.compile` 首次调用的编译过程本身可能耗时较长(尤其模型更大、更复杂时),如果场景是"只调用一两次就结束"(比如单次脚本任务),编译开销可能完全抵消甚至超过它带来的收益,值得在决定使用前用类似本例的方式实测对比,而不是假设"编译了就一定划算"。

---

## 5. 批量推理的 Padding 与 Attention Mask 处理

**签名/是什么:**
```
tok.padding_side = "left"    # 呼应01类:生成场景必须左padding
batch = tok(["Hi", "The capital of France is"], return_tensors="pt", padding=True)
model.generate(**batch, max_new_tokens=5)
```
把 01 类讲过的"padding_side"知识,落地到 `generate()` 批量生成这个真实场景里。

**一句话:** 批量生成时,`generate()` 会读取 `attention_mask` 来正确处理每条序列的 padding 部分,**必须**同时正确设置 `padding_side="left"` 和传入 `attention_mask`,两者缺一都会导致批量生成结果出问题(要么报错,要么生成质量诡异但不报错)。

**底层机制/为什么这样设计:** 这是 01 类"padding策略"知识点的直接应用场景——`generate()` 内部依赖 `attention_mask` 知道"每条序列真实内容从哪里开始"(左填充时,pad token在前,attention_mask前几位是0);同时依赖左填充这个物理布局保证"每一行的最后一个位置都是这一行的真实最后一个token",从而能对整个batch统一地"从最后一个位置续写"。两个机制配合,才能让batch里长度不同的多条prompt同时正确生成,互不干扰。

用可运行例子里实际用到的两条 prompt 具体画一下这个"对齐"长什么样(`P`=pad token,数字下标是这条 prompt 自己的第几个真实token,batch=2、`padding_side="left"`):

```
行0 "Hi"                      (2个真实token):  [ P  P  P  P  h1 h2]
行1 "The capital of France is"(6个真实token):  [t1 t2 t3 t4 t5 t6]
                                                                ↑
                                          每一行最后一列,都是"这一行自己最后一个真实token"

对应的 attention_mask(0=这个位置是pad,不参与attention;1=真实内容):
行0: [0 0 0 0 1 1]
行1: [1 1 1 1 1 1]
```

`generate()` 每一步都从"当前序列的最后一列"往右续写——左padding保证了这一列在每一行都对齐到"这一行真实内容的末尾",所以整个 batch 能统一地"从最后一列续写"。如果换成右padding,行0 会变成 `[h1 h2 P P P P]`,最后一列是 pad 不是真实内容,统一从"最后一列"续写就变成了从 pad token 续写——这正是下面常见坑1、以及 13 类知识点4会实测到的"生成结果异常"的直接成因。

**AI 研究/工程场景:** 09 类如果要对多个测试样本批量做生成评估(而不是一条条串行调用,03类讨论过的效率考虑),必须正确配置 padding,否则批量结果里较短的那些prompt的生成质量会不可靠。

**可运行例子:**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
tok.padding_side = "left"   # 生成场景的正确设置,呼应01类
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

prompts = ["Hi", "The capital of France is"]   # 长度明显不同的两条prompt
batch_inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda")

with torch.no_grad():
    batch_out = model.generate(**batch_inputs, max_new_tokens=5, do_sample=False)

assert batch_out.shape[0] == 2  # batch维度保留,两条结果都在

decoded = [tok.decode(batch_out[i], skip_special_tokens=True) for i in range(2)]
# 关键验证:每条结果都应该是各自prompt的合理延续,不应该因为padding而串扰/损坏
assert decoded[0].startswith("Hi")
assert decoded[1].startswith("The capital of France is")

# 对照组:单条单独生成的结果,理论上和batch生成的对应结果应该一致(相同的greedy解码)
single_out = model.generate(**tok(prompts[1], return_tensors="pt").to("cuda"), max_new_tokens=5, do_sample=False)
single_decoded = tok.decode(single_out[0], skip_special_tokens=True)
assert decoded[1] == single_decoded  # batch里的结果和单独生成结果完全一致,证明padding没有污染这条结果

print(f"OK: batch生成两条结果={decoded},且batch中较长prompt的结果和单独生成完全一致(padding正确隔离)")
```
本机实测:batch 输出 `["Hi, I'm Sarah", "The capital of France is Paris.\n\n2"]`,两条结果各自合理延续自己的prompt;较长的那条prompt在batch里的生成结果和单独生成完全一致,确认padding正确隔离、没有互相污染。

**面试怎么问 + 追问链:** "如果batch生成时用了右padding(padding_side='right')会发生什么?" → 追问"这个问题会报错,还是静默产出错误结果?"(通常**不会报错**,但较短那条prompt的生成会从错误的位置续写——因为01类讲过右padding下"最后一个位置"是pad而不是真实内容,模型会从pad token的位置续写,产出的内容和这条prompt的真实语义脱节;"不报错但结果错误"比"直接报错"更危险,这是本篇和01类共同强调的一个真实、容易被忽视的正确性陷阱)。

**常见坑:**
1. 忘记设置 `tok.pad_token`(某些tokenizer默认没有,呼应01类)会在 `padding=True` 这一步就直接报错,顺序上要先确认pad_token已设置。
2. 批量生成时不同长度的输出(有的prompt提前触发EOS,有的没有)会让batch内的序列在生成过程中长度不一致,`generate()` 内部会自动处理这种情况(用pad填充已完成的序列,继续生成未完成的),但解码时同样需要留意 `skip_special_tokens=True` 正确剥离这些填充产生的特殊token。

---

## 6. `StoppingCriteria` 自定义停止条件

**签名/是什么:**
```
from transformers import StoppingCriteria, StoppingCriteriaList

class MyStop(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return <某个条件为True时返回True>

model.generate(**inputs, stopping_criteria=StoppingCriteriaList([MyStop()]))
```
除了"达到max_new_tokens"或者"生成了EOS"这两种内置停止条件,`StoppingCriteria` 让你自定义"什么情况下该提前结束生成"。

**一句话:** 每生成一个新token,`generate()` 循环都会调用一遍 `stopping_criteria` 里注册的每个判断函数,只要有一个返回 `True`,生成就立即停止——这是一个和 05 类 `TrainerCallback` 思路相似的"在固定时间点插入自定义判断逻辑"的扩展机制。

**底层机制/为什么这样设计:** 内置的停止条件(EOS/max_length)覆盖不了所有场景——比如想在模型生成到某个特定关键词就停(而不是等它自己判断该结束)、或者想实现"生成到平衡的括号/引号就停"这类结构化的停止逻辑。把停止判断做成可插拔的回调,`generate()` 核心循环不需要为每一种可能的停止需求都各写一个专门参数,只需要提供这一个通用扩展点。

**AI 研究/工程场景:** 需要模型生成结构化输出(比如只要一个JSON对象,不要额外的解释文字)时,自定义停止条件(检测到匹配的右括号就停)能避免模型"话痨"式地生成大量不需要的后续内容,省下真实的推理时间和token开销。

**可运行例子:**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
inputs = tok("The capital of France is", return_tensors="pt").to("cuda")

# 先确定"Paris"这个词在这个tokenizer下对应的token id(不同tokenizer/不同上下文可能不同,现场查而不是猜)
paris_token_id = tok(" Paris", add_special_tokens=False)["input_ids"][0]

class StopOnToken(StoppingCriteria):
    def __init__(self, stop_id):
        self.stop_id = stop_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() == self.stop_id   # 一旦最新生成的token是目标token,就停

criteria = StoppingCriteriaList([StopOnToken(paris_token_id)])

with torch.no_grad():
    out_normal = model.generate(**inputs, max_new_tokens=30, do_sample=False)          # 不加停止条件,跑满30个新token
    out_stopped = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                   stopping_criteria=criteria)                          # 加了自定义停止条件

assert out_stopped.shape[1] < out_normal.shape[1]   # 提前停止:长度明显更短
assert out_stopped[0, -1].item() == paris_token_id   # 确认停止的确切位置就是目标token

decoded_stopped = tok.decode(out_stopped[0], skip_special_tokens=True)
print(f"OK: 不加停止条件长度={out_normal.shape[1]},加了停止条件在生成'Paris'后立即停止,长度={out_stopped.shape[1]},文本={decoded_stopped!r}")
```
本机实测:目标 token(" Paris"的第一个子词)id 是 3681;不加停止条件生成 30 个新 token(总长36);加了自定义停止条件后,总长精确为 7(生成到"Paris"这个token就立即停止,没有继续生成后续的句号/换行等内容),生成文本 `"The capital of France is Paris"`。

**面试怎么问 + 追问链:** "`StoppingCriteria` 检查的粒度是什么?每生成一个token都要检查一遍所有条件吗?" → 是的,每步生成后都会遍历调用注册的每个 `StoppingCriteria` 实例——如果判断逻辑本身很重(比如涉及复杂的字符串解析),会给生成过程带来额外的、逐步累加的开销,写自定义停止条件时要注意判断逻辑本身的效率,不能假设这里"几乎零成本"。

**常见坑:**
1. `input_ids` 在 `StoppingCriteria.__call__` 里是当前**已经生成到目前为止的完整序列**(输入+目前已生成的部分),不是"只有最新那个token"——本例用 `input_ids[0, -1]` 精确取"最新token"这个切片操作是必须的,直接用整个 `input_ids` 做判断逻辑会出错。
2. 批量生成场景下(knowledge point 5),batch 里不同样本可能在不同步数触发各自的停止条件,但 `StoppingCriteria` 的整体机制是"batch里全部样本都满足条件才真正停止整个batch循环"(不会出现"batch里第0条先停、第1条继续单独跑"这种精细的单条控制),这是批量生成场景下需要了解的一个真实限制。

---

## 7. 流式生成(`TextIteratorStreamer`)

**签名/是什么:**
```
from transformers import TextIteratorStreamer
import threading

streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
thread = threading.Thread(target=model.generate, kwargs=dict(**inputs, streamer=streamer, max_new_tokens=10))
thread.start()
for chunk in streamer:   # 生成过程中逐块拿到文本,不用等整个生成完成
    print(chunk, end="")
thread.join()
```
让调用方能在生成**过程中**逐步拿到已经生成好的文本片段,而不是等 `generate()` 整个函数调用返回才能看到任何内容——这是聊天类产品"打字机效果"背后的标准实现方式。

**一句话:** `generate()` 本身是一个阻塞调用(要等全部token生成完才返回),`TextIteratorStreamer` 通过"把 `generate()` 放到后台线程跑,通过一个队列把每步新生成的文本片段实时传给主线程"这个技巧,实现了不阻塞地逐步获取输出。

**底层机制/为什么这样设计:** `generate()` 内部循环每生成一个新token就会调用一次 streamer 的回调,把这个token解码好的文本塞进一个线程安全的队列;`TextIteratorStreamer` 实现了 Python 的迭代器协议,主线程通过 `for chunk in streamer:` 不断从队列取数据——因为 `generate()` 本身运行在**另一个线程**里,主线程的迭代不会被 `generate()` 的整体运行时长阻塞,能一有新内容就立刻拿到。这是"生产者(后台线程跑generate)-消费者(主线程消费队列)"模式的一个具体应用。

**AI 研究/工程场景:** 任何面向最终用户的对话式产品,流式输出几乎是标配体验(用户不需要等待完整回复生成完,能看到文字逐步出现)——`TextStreamer`(直接打印到控制台,不需要手动管理线程,适合脚本调试场景)和 `TextIteratorStreamer`(本知识点用的,适合需要把流式内容进一步传递给别的系统,比如Web服务的场景)是两个常用的具体实现。

**可运行例子:**
```python
import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
inputs = tok("The capital of France is", return_tensors="pt").to("cuda")

streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
gen_kwargs = dict(**inputs, max_new_tokens=10, do_sample=False, streamer=streamer)

thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()

chunks = []
for chunk in streamer:
    chunks.append(chunk)   # 每次循环拿到的是新解码出来的一小段文本
thread.join()

assert len(chunks) > 1   # 确认真的是分多次拿到的(流式),不是一次性拿到整个结果

streamed_full_text = "".join(chunks)

# 对照:非流式的常规generate,拼出的文本应该和流式拼接的结果一致
with torch.no_grad():
    normal_out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
normal_text = tok.decode(normal_out[0], skip_special_tokens=True)

# 关键发现(实测,推翻了"streamer默认不含prompt"这个直觉假设):
# TextIteratorStreamer 默认 skip_prompt=False,流式内容本身就已经包含了prompt回显,
# 不需要(也不应该)再手动把prompt拼到streamed_full_text前面
assert streamed_full_text == normal_text

# 验证skip_prompt=True是"只要新生成内容,不要prompt回显"的开关
streamer2 = TextIteratorStreamer(tok, skip_special_tokens=True, skip_prompt=True)
gen_kwargs2 = dict(**inputs, max_new_tokens=10, do_sample=False, streamer=streamer2)
thread2 = threading.Thread(target=model.generate, kwargs=gen_kwargs2)
thread2.start()
chunks2 = [c for c in streamer2]
thread2.join()
streamed_only_new = "".join(chunks2)
prompt_text = tok.decode(inputs["input_ids"][0], skip_special_tokens=True)

# 关键发现之二(和03类pipeline knowledge point同源的问题):
# skip_prompt=True时,新内容第一个词前的空格会丢失("is"+"Paris"之间的空格没了),
# 因为这个空格在SentencePiece里是编码在"Paris"这个token前缀(▁Paris)里的,
# 但跨越prompt/generation边界单独解码"新增部分"时,tokenizer不一定能正确重建这个边界空格。
assert not streamed_only_new.startswith(" ")   # 新内容开头没有空格(真实观察,不是我们想要的效果)
assert prompt_text + streamed_only_new != normal_text          # 因此直接拼接对不上normal_text
assert prompt_text + " " + streamed_only_new == normal_text    # 手动补一个空格才能对上——这正是要记录的真实坑

print(f"OK: 默认skip_prompt=False,流式内容({len(chunks)}个chunk)已含prompt回显,和非流式结果直接相等;"
      f"skip_prompt=True时新内容开头丢失了边界空格,需要调用方自己留意拼接时补上")
```
本机实测:10 个新 token 的生成过程,`streamer` 分 12 次(chunk数量和token数量不完全一一对应,取决于解码时token到文本片段的切分方式)产出内容;拼接后的完整文本和非流式 `generate()` 直接解码的结果完全一致——确认流式只是"分批交付同一份结果",不是另一套独立的生成逻辑。**两个推翻了初始假设的真实发现**:① `TextIteratorStreamer` 默认 `skip_prompt=False`,第一批 chunk(实测是 `'The capital of France '`)就已经包含了 prompt 本身的回显,不是只流式输出新生成的部分;② 改用 `skip_prompt=True` 后,新内容开头丢失了一个空格(`'Paris.\n...'` 而不是 `' Paris.\n...'`)——根因和 03 类 pipeline 那个知识点是同一类问题:空格信息编码在 SentencePiece token 自己的 `▁` 前缀里,跨越"prompt在哪结束、生成从哪开始"这条边界单独解码后半段时,tokenizer 不总能正确重建边界处的空格,这是聊天应用做流式UI时必须手动处理的一个真实细节,不能假设"分段解码再拼接"和"整体解码"永远等价。

**面试怎么问 + 追问链:** "为什么流式生成要开一个新线程,而不是直接在主线程里一边生成一边yield?" → `generate()` 函数本身的实现是一次性跑完整个循环再返回,内部没有设计成"生成器函数"(用 `yield` 逐步返回)——要在不修改 `generate()` 核心实现的前提下拿到"边生成边可读取"的效果,让它在独立线程里正常跑到底、通过线程安全的队列往外"泄漏"中间结果,是一个不侵入核心逻辑的工程技巧,不是最优雅但是实用的解法。

**常见坑:**
1. 忘记 `thread.join()` 可能导致主线程提前退出时后台生成线程还没跑完(尤其是脚本场景),养成显式等待线程结束的习惯。
2. **`TextIteratorStreamer` 默认 `skip_prompt=False`,会把 prompt 也一起流式吐出来**——这是本知识点现场验证纠正的一个真实误区(最初凭直觉假设"流式当然只给新内容",实测证明默认行为并非如此)。聊天类应用如果不显式设置 `skip_prompt=True`,会在界面上把用户刚输入的内容又重复展示一遍,这是一个真实会被用户注意到的体验问题,不是无关紧要的细节。

---

*本篇 7 个知识点全部在仓库根目录 `.venv` 真实验证通过(每个知识点独立进程验证)。*
