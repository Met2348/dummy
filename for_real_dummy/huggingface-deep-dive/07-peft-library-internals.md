# 07 · PEFT 库工程机制(PEFT Library Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)。已在仓库根目录 `.venv` 真实跑通(`peft==0.19.1`)。**差异声明:本篇讲 `peft` 这个库怎么在工程上实现 LoRA(module怎么被替换、参数怎么冻结、怎么保存/合并),不讲 LoRA 算法本身为什么有效——算法/数学部分见 [`learning/lora-family/lectures/01-lora.md`](../../learning/lora-family/lectures/01-lora.md)。类比 torch-deep-dive 对 PyTorch 的处理:不重复"什么是反向传播",只讲 autograd 库内部怎么实现。**

---

## 1. `get_peft_model()` 内部机制

**签名/是什么:**
```
from peft import LoraConfig, get_peft_model
peft_model = get_peft_model(model, lora_config)
```
把一个普通的 `transformers` 模型和一份 LoRA 配置传进去,拿到一个"外表和普通模型很像,但内部特定层被替换成了 LoRA 版本"的新对象。

**一句话:** `get_peft_model()` 不是复制一份新模型,而是**原地遍历模型的 module 树,把匹配 `target_modules` 的 `nn.Linear` 层替换成对应的 LoRA 版本**,其余层保持不变,最后把整个模型包一层 `PeftModel` 外壳。

**底层机制/为什么这样设计:** LoRA 的核心思想是"在某些线性层旁边加一条低秩旁路(`B @ A`),原始权重冻结不动,只训练这条旁路",工程实现上最直接的方式就是把 `nn.Linear` 替换成一个行为兼容但内部多了旁路逻辑的子类。这样设计的好处是**对模型其余代码完全透明**——`LlamaForCausalLM.forward()` 里调用 `self.q_proj(x)` 的代码一行都不用改,因为替换后的对象依然是"看起来像"`nn.Linear` 的东西(前向接口兼容),只是内部多做了旁路计算。这是"猴子补丁"(monkey patching)思路在模块化系统里的规范化实现。

**AI 研究/工程场景:** 理解"是替换 module 而不是包一层新模型"这件事,能解释很多 PEFT 库的行为——比如为什么 `peft_model.model` 能访问到原始的 `LlamaForCausalLM` 结构(model 树基本没变,只是部分叶子节点被换了),这对调试"LoRA 到底加在了哪些层"非常关键。

**可运行例子:**
```python
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
assert type(model).__name__ == "LlamaForCausalLM"

config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                     lora_dropout=0.0, task_type=TaskType.CAUSAL_LM)
peft_model = get_peft_model(model, config)

assert type(peft_model).__name__ == "PeftModelForCausalLM"   # 外层包装
assert type(peft_model.base_model).__name__ == "LoraModel"    # 中间层:管理LoRA注入的容器

# 找到一个被替换的q_proj层,确认它不是原生torch.nn.Linear,而是peft自己的Linear子类
for name, module in peft_model.named_modules():
    if name.endswith("q_proj"):
        assert type(module).__name__ == "Linear"                       # 类名相同(容易误认成原生Linear)
        assert type(module).__module__ == "peft.tuners.lora.layer"      # 但实际是peft自己的类,不是torch.nn
        break
else:
    raise AssertionError("没找到q_proj层")

print("OK: get_peft_model()把LlamaForCausalLM包装成PeftModelForCausalLM,q_proj被原地替换成peft.tuners.lora.layer.Linear")
```
本机实测:替换后的 `q_proj` 层 `type().__name__` 显示为 `'Linear'`(容易让人误以为还是原生 `torch.nn.Linear`),但 `__module__` 确认是 `peft.tuners.lora.layer`——**类名相同但类完全不同**,这是一个容易造成误判的真实细节,判断"这层是不是被 LoRA 接管了"不能只看类名,要么看 `__module__`,要么直接看有没有 `lora_A`/`lora_B` 这两个子模块(knowledge point 3 会展开)。

**面试怎么问 + 追问链:** "`get_peft_model()` 返回的对象和原始模型是同一个对象吗?" → 追问"如果我对原始 `model` 变量做操作(比如 `model.eval()`),会影响 `peft_model` 吗?"(会,因为底层 module 树是同一份内存中的对象,`get_peft_model` 只是替换了其中部分子模块的引用、又包了一层外壳,原始 `model` 变量和 `peft_model.base_model.model` 指向的是同一批被部分替换过的模块——这也是为什么替换是"原地"的这件事很重要,不理解这一点容易在同时持有 `model`/`peft_model` 两个变量时产生困惑)。

**常见坑:**
1. 判断"这个模型是不是已经被 PEFT 包装过"不能只看类名里有没有"Peft"字样想当然——本例已经展示了 `q_proj` 这个内部替换层的类名"Linear"具有欺骗性,统一用 `isinstance(model, PeftModel)` 或者检查 `hasattr(model, 'peft_config')` 更可靠。
2. `get_peft_model()` 调用之后,原始的 `model` 变量的部分子模块已经被原地替换,如果代码逻辑还假设 `model` 是"纯净的、未被修改的"原始模型,会遇到意料之外的行为——这不是 bug,是"原地替换"这个设计选择的直接后果。

---

## 2. `LoraConfig` 关键参数

**签名/是什么:**
```
LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
           lora_dropout=0.0, task_type=TaskType.CAUSAL_LM)
```
LoRA 旁路本身的形状(秩、缩放系数)和"加在哪些层"的配置对象。

**一句话:** `r`(秩)决定旁路矩阵 `B@A` 的"瓶颈宽度"(越小可训练参数越少),`lora_alpha` 是一个缩放系数(旁路输出会乘以 `lora_alpha/r`),`target_modules` 决定模型里哪些线性层会被知识点 1 讲的机制替换掉。

**底层机制/为什么这样设计:** `r` 直接决定了 knowledge point 4 会讲的可训练参数量——这是 LoRA"参数高效"这个核心卖点的直接来源。`lora_alpha/r` 这个缩放设计是为了让"改变 `r`"这件事(做超参搜索时很常见)不会同时意外改变旁路对输出的影响强度尺度——如果不做这个缩放,`r` 越大旁路的数值贡献可能不成比例地暴涨,`alpha` 提供了一个独立于 `r` 的"旁路影响力"旋钮。**这里只讲 `peft` 库把这个系数暴露成了 `lora_alpha`/`r` 两个独立参数这件工程事实;"`r` 个低秩分量累加起来幅度为什么会按 `√r` 增长、`alpha/r` 缩放在大 `r` 下具体会怎么失效"这套数学推导,[peft-deep-dive/01-lora-core.md](../peft-deep-dive/01-lora-core.md) 第1、2点已经用真实张量实验从零推导并验证过,这里不重复。** `target_modules` 接受的字符串是**模块名的后缀匹配**(不是完整路径),这也是 knowledge point 3 要展开的内容。

**AI 研究/工程场景:** 09 类"超参敏感度扫描"知识点会对比不同 `r` 值对微调效果的影响,`LoraConfig` 这几个参数就是那组对比实验唯一改变的变量。

**可运行例子:**
```python
from peft import LoraConfig, TaskType

config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                     lora_dropout=0.0, task_type=TaskType.CAUSAL_LM)

assert config.r == 8
assert config.lora_alpha == 16
# target_modules在LoraConfig内部被规范化成set,不保证保留原始传入的顺序
assert config.target_modules == {"q_proj", "v_proj"}
assert isinstance(config.target_modules, set)
assert config.lora_dropout == 0.0

# 缩放系数 alpha/r,决定旁路输出对最终结果的影响强度
scaling = config.lora_alpha / config.r
assert scaling == 2.0

print(f"OK: r={config.r}, lora_alpha={config.lora_alpha}, 缩放系数alpha/r={scaling}, target_modules规范化为set: {config.target_modules}")
```
本机实测:确认 `target_modules` 传入时是 `list`,但 `LoraConfig` 内部存储时变成了 `set`(`{'v_proj', 'q_proj'}`,顺序不保证)——这是配置对象内部规范化的一个真实细节,如果代码依赖 `target_modules` 的顺序会有问题。

**面试怎么问 + 追问链:** "`r` 该怎么选,越大越好吗?" → 追问"`r` 增大到接近原始权重矩阵的维度会发生什么?"(`r` 越大,旁路矩阵的表达能力越强、越接近全参微调的效果,但可训练参数也线性增长,失去 LoRA"参数高效"的优势;`r` 太小则表达能力受限,可能欠拟合——这是一个真实的容量/效率权衡,09 类的超参扫描会给出这台机器上的实测数据,不是纸上谈兵)。

**常见坑:**
1. 不同论文/教程对 `lora_alpha` 的推荐设置不一致(有的建议 `alpha=r`,有的建议 `alpha=2r`),没有放之四海皆准的黄金比例,需要针对具体任务做实验——本例用的 `alpha=16, r=8`(缩放系数2)是一个常见起点,不是唯一正确答案。
2. `target_modules` 如果写错模块名(比如把 `q_proj` 拼成 `q_proj_`),不会报错,只是这个配置匹配不到任何层,LoRA 实际上没有生效——knowledge point 3 会展示怎么验证"确实匹配到了预期数量的层"。

---

## 3. `target_modules` 怎么匹配到具体 Linear 层

**签名/是什么:**
```
target_modules=["q_proj", "v_proj"]   # 匹配所有名字以这些字符串结尾的子模块
```
`target_modules` 里的字符串是对模型 module 树里每个子模块**名字**的匹配规则,不是完整路径。

**一句话:** TinyLlama(LLaMA架构)每一层 transformer block 里都有 `q_proj`/`k_proj`/`v_proj`/`o_proj`(attention)和 `gate_proj`/`up_proj`/`down_proj`(MLP)这些线性层,`target_modules=["q_proj", "v_proj"]` 会让**每一层**里名字匹配的这两个层都被替换,22 层模型总共产生 44 个被 LoRA 接管的层(22 × 2)。

**底层机制/为什么这样设计:** 匹配逻辑是"模块名的后缀匹配"(而不是要求写出 `model.layers.0.self_attn.q_proj` 这种完整路径),这让同一份 `target_modules` 配置可以不用感知模型具体有多少层就能"广播"到所有层——写配置的人只需要关心"这一**类**层(不管在哪一层出现)要不要加LoRA",不需要为每一层单独写一条配置。`peft` 库内置了 `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING` 这张官方推荐映射表(针对不同模型架构给出默认该对哪些层做LoRA的建议),`target_modules` 不显式指定时会查这张表。

**AI 研究/工程场景:** 09 类"超参敏感度扫描"里"target_modules选择的影响"这一组对比实验,比较的就是"只对attention的q/v做LoRA" vs "attention全做" vs "连MLP的gate/up/down也做"这几种配置——`target_modules` 参数是控制这组实验范围的开关。

**可运行例子:**
```python
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# peft官方对llama架构的推荐target_modules
official_llama_targets = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["llama"]
assert official_llama_targets == ["q_proj", "v_proj"]

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
peft_model = get_peft_model(model, config)

# 精确统计:名字以q_proj或v_proj结尾的module,应该恰好是 22层 x 2个目标 = 44个
adapted_layers = [n for n, m in peft_model.named_modules() if n.endswith("q_proj") or n.endswith("v_proj")]
assert len(adapted_layers) == 44

# 每个被匹配到的层,内部应该都出现了lora_A/lora_B这两个新增的子模块(旁路的两个矩阵)
sample_layer_name = adapted_layers[0]
sample_layer = dict(peft_model.named_modules())[sample_layer_name]
assert hasattr(sample_layer, "lora_A")
assert hasattr(sample_layer, "lora_B")

# 没有被匹配的层(比如o_proj)应该保持原生torch.nn.Linear,没有被接管
o_proj_layer = None
for n, m in peft_model.named_modules():
    if n.endswith("o_proj"):
        o_proj_layer = m
        break
assert not hasattr(o_proj_layer, "lora_A")   # o_proj不在target_modules里,原封不动

print(f"OK: target_modules=['q_proj','v_proj']精确匹配到{len(adapted_layers)}层(22层x2),o_proj未被接管")
```
本机实测:精确匹配到 44 层(22 层 × 2 个目标模块名),每层内部确认存在 `lora_A`/`lora_B` 子模块;未在 `target_modules` 里的 `o_proj` 确认没有被接管(没有 `lora_A` 属性)。

**面试怎么问 + 追问链:** "如果 `target_modules` 写的字符串在模型里根本不存在会怎样?" → `peft` 不会报错,只是匹配不到任何层,`get_peft_model()` 依然能成功返回一个"看起来正常"的对象,但实际上没有任何一层被 LoRA 接管,`print_trainable_parameters()`(knowledge point 4)会显示可训练参数为 0 或者只有极少数其他默认可训练的参数——这是一个真实存在的、容易被忽视的静默失败模式,验证"确实匹配到了预期层数"(如本例)是排查这类问题的关键一步。

**常见坑:**
1. 不同模型架构的线性层命名不一样(LLaMA 系是 `q_proj`/`v_proj`,有些模型可能是 `query`/`value` 或其他命名),照抄别的模型架构的 `target_modules` 配置换到不同架构上很可能匹配不到任何层——`peft.utils.constants` 里的官方映射表是针对特定架构验证过的,换架构要重新确认。
2. 用后缀匹配意味着如果模型里有两个不同层级但名字后缀恰好相同的模块(不太常见,但架构复杂的模型可能出现),`target_modules` 可能匹配到超出预期的层——精确统计匹配到的层数(如本例）是确保配置符合预期的必要步骤,不能只凭配置字符串"看起来对"就信任。

---

## 4. 参数冻结机制(`print_trainable_parameters()`)

**签名/是什么:**
```
peft_model.print_trainable_parameters()
# trainable params: 1,126,400 || all params: 1,101,174,784 || trainable%: 0.1023
```
LoRA 的"参数高效"不是一句口号,是可以现场量化验证的具体数字。

**一句话:** `get_peft_model()` 除了 knowledge point 1 讲的"替换特定层",还做了另一件同样重要的事——**把原始模型的全部参数 `requires_grad` 设为 `False`(冻结),只有新注入的 `lora_A`/`lora_B` 参数保持 `requires_grad=True`**,这两件事合起来才是 LoRA"只训练一小部分参数"的完整机制。

**底层机制/为什么这样设计:** 只是"新增一条旁路"不足以实现参数高效——如果原始权重依然 `requires_grad=True`,反向传播时依然会对**全部**参数计算梯度、优化器依然要维护全部参数的动量状态,显存和计算开销和全参微调没有本质区别,LoRA 的旁路设计就失去了意义。`get_peft_model()` 在替换层结构的同时,遍历模型全部参数,精确地只把新注入的旁路参数标记为可训练,这是"参数高效"效果真正实现的地方。

**AI 研究/工程场景:** 09 类的显存对比实验里,LoRA 配置显著更低的显存占用(相对全参微调),根源就在这里——优化器(比如 Adam)需要为每个可训练参数额外维护一阶/二阶矩估计,可训练参数量从 11 亿降到 112 万(本例的真实数字),优化器状态的显存占用几乎可以忽略不计。

**可运行例子:**
```python
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
peft_model = get_peft_model(model, config)

trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in peft_model.parameters())

assert total == 1_100_048_384 + 1_126_400   # 原始1.1B参数(01/02类记录的1,100,048,384)+ 新增的LoRA旁路参数
assert trainable == 1_126_400                # 只有LoRA旁路参数可训练
pct = 100 * trainable / total
assert pct < 0.2   # 可训练比例远小于1%,这就是"参数高效"的量化证据

# 原始的q_proj主权重(不是lora_A/B)应该被冻结了
for name, param in peft_model.named_parameters():
    if name.endswith("q_proj.base_layer.weight"):
        assert param.requires_grad is False
        break

print(f"OK: 可训练参数{trainable:,} / 总参数{total:,} = {pct:.4f}%,原始权重requires_grad=False")
```
本机实测:base model 原始参数 1,100,048,384(和 01/02 类记录的数字一致)+ 新增 LoRA 旁路参数 1,126,400 = 总参数 1,101,174,784;可训练参数精确为 1,126,400,占比 0.1023%——原始 `q_proj` 主权重确认 `requires_grad=False`。

**面试怎么问 + 追问链:** "1,126,400 这个可训练参数量是怎么算出来的,能手推吗?" → 每个被接管的 Linear 层,原始权重形状是 `[out_features, in_features]`,LoRA 旁路是两个矩阵 `A: [r, in_features]` 和 `B: [out_features, r]`;TinyLlama 的 `q_proj`/`v_proj` 具体维度(hidden_size=2048,GQA下v_proj的输出维度因KV头数较少而更小)决定了每层旁路的具体参数量,乘以 44 层(22层×2个目标)就是总数——这是一道能检验"是否真的理解LoRA参数量从哪来"的手推题,不是死记数字。

**常见坑:**
1. `requires_grad=False` 只是"不参与梯度更新",不代表这些参数不参与前向计算——原始权重依然完整参与前向传播(LoRA旁路是"加在"原始计算结果上,不是替代),这一点容易和"冻结=不参与计算"的错误直觉混淆。
2. 如果自定义训练代码里手动遍历参数做了一些操作(比如自定义的权重衰减逻辑),忘记检查 `requires_grad` 就统一处理所有参数,可能意外地对已冻结的原始权重做了不该做的操作——PEFT模型的参数列表里"可训练"和"冻结"两类参数是混在一起的,不能假设 `model.parameters()` 返回的都是可训练的。

---

## 5. Adapter 权重单独保存

**签名/是什么:**
```
peft_model.save_pretrained("my_adapter_dir")
# 只保存LoRA旁路的参数,不保存11亿参数的base model
```
和知识点 4 呼应——既然只有一小部分参数是可训练/被改变的,保存的时候自然也只需要保存这一小部分。

**一句话:** `peft_model.save_pretrained()` 产出的文件体积和 base model 的大小**无关**,只和 LoRA 旁路的参数量(knowledge point 4 讲的那 112 万个参数)相关——本例产出的 adapter 文件只有几 MB,而不是 base model 那样的 GB 级别。

**底层机制/为什么这样设计:** 这是"参数高效微调"这个理念在**存储**层面的延伸——如果每次微调都要保存一份完整的模型副本,LoRA 在训练时节省的显存优势,会在存储和分发环节被完全抵消(想象保存100个不同任务的LoRA微调版本,如果每个都是完整模型,存储成本是天文数字;如果只存adapter,100个adapter的总体积可能比1个完整模型还小)。保存的内容只有 `adapter_config.json`(记录 `LoraConfig` 的配置,knowledge point 8 加载时需要靠它重建结构)和 `adapter_model.safetensors`(旁路权重本身)。

**AI 研究/工程场景:** 09 类"微调产物保存与复现"知识点会用这个机制保存每一组对比实验的产物;团队协作场景下,共享一个几 MB 的 adapter 文件比共享几 GB 的完整模型副本高效得多。

**可运行例子:**
```python
import torch
import os
import tempfile
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
peft_model = get_peft_model(model, config)

tmpdir = tempfile.mkdtemp()
peft_model.save_pretrained(tmpdir)

saved_files = set(os.listdir(tmpdir))
assert "adapter_config.json" in saved_files
assert "adapter_model.safetensors" in saved_files
assert "pytorch_model.bin" not in saved_files      # 不是完整模型的保存格式
assert not any(f.startswith("model") and f.endswith(".safetensors") for f in saved_files)  # 没有base model的权重文件

adapter_size_mb = os.path.getsize(os.path.join(tmpdir, "adapter_model.safetensors")) / 1e6
assert adapter_size_mb < 10   # 几MB级别,而不是base model那样的GB级别(TinyLlama本身约2.2GB)

print(f"OK: adapter文件={saved_files},adapter_model.safetensors体积仅{adapter_size_mb:.2f}MB(对比base model约2200MB)")
```
本机实测:adapter 文件夹只含 `adapter_config.json`/`adapter_model.safetensors`(还额外生成了一份 `README.md` model card,呼应 11 类)三个文件,`adapter_model.safetensors` 体积 4.52MB——和 base model 的 2.2GB 相比,体积差距接近 500 倍。

**面试怎么问 + 追问链:** "只保存 adapter,推理时怎么用?" → 需要先加载 base model,再用 knowledge point 8 讲的 `PeftModel.from_pretrained(base_model, adapter_dir)` 把 adapter 接回去——adapter 文件本身不是一个可以独立运行的完整模型,它是"base model + 这份 adapter"这个组合才构成完整的可用模型,这也是为什么分发 adapter 时必须同时说明"这个 adapter 配哪个 base model 用"。

**常见坑:**
1. `adapter_config.json` 里记录了训练这个 adapter 时用的 base model 名字(`base_model_name_or_path`),如果之后 base model 的版本发生了变化(比如 knowledge point 4 类的 `revision` 变了),用不匹配的 base model 加载这个 adapter,行为是未定义的,可能报错也可能悄悄跑出低质量结果——不报错不代表组合是对的。
2. 不要把"adapter文件很小"误解为"adapter包含的信息不重要"——4.52MB 虽然远小于 base model,但正是这部分信息决定了微调带来的所有改变,弄丢这个文件等于弄丢了整次微调的全部成果。

---

## 6. `merge_and_unload()` 合并机制

**签名/是什么:**
```
merged_model = peft_model.merge_and_unload()
# 返回一个普通的、不再有LoRA结构的模型,推理时和原生模型零延迟差异
```
把 LoRA 旁路的效果**数学上精确地**吸收进原始权重,之后 LoRA 相关的额外结构完全消失。

**一句话:** LoRA 推理时("不合并"的状态下)每次前向都要额外算一遍旁路(`x @ A^T @ B^T * scaling`)再加到原始输出上,这部分计算是真实的额外开销;`merge_and_unload()` 利用"矩阵乘法可结合"这个数学性质,把 `W + scaling*B@A` 预先算成一个新的权重矩阵,合并之后的模型做前向时,和一个从来没有用过 LoRA 的原生模型在计算路径上完全一样,**推理阶段的额外开销归零**。

**底层机制/为什么这样设计:** 这是 LoRA 相比其他一些 PEFT 方法(比如串联型 adapter,需要在网络中插入额外的非线性层)的一个结构性优势——因为 LoRA 的旁路是**线性**的、且加在和原始权重**同一个位置**(都是对同一个输入做线性变换再相加),数学上可以合并成一个等价的线性变换,不像非线性 adapter 那样合并后还会改变原本的计算路径。这也是 00-roadmap.md 差异声明里提到的、`learning/adapter-tuning-family` 的"10-peft-next-step"总结里"LoRA范式吞噬了Adapter范式"这个结论的技术根源之一。

**AI 研究/工程场景:** 微调实验阶段用未合并的 LoRA 形式(方便随时 `set_adapter`/切换不同 adapter 对比效果);确定要**部署**某一个版本时,`merge_and_unload()` 消除推理时的额外计算开销,是从"实验形态"切换到"生产形态"的标准最后一步。

**可运行例子:**
```python
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
peft_model = get_peft_model(model, config)

dummy_input = torch.randint(0, 32000, (1, 8)).to("cuda")
with torch.no_grad():
    logits_before = peft_model(dummy_input).logits

merged = peft_model.merge_and_unload()

assert type(merged).__name__ == "LlamaForCausalLM"   # 合并后变回普通模型类型,不再是PeftModel
assert not any("lora" in name.lower() for name, _ in merged.named_parameters())  # 结构上lora参数彻底消失

with torch.no_grad():
    logits_after = merged(dummy_input).logits

# 合并前后,同样输入的输出在数值上应该(近似)相等——这是"合并"这个操作正确性的核心验证
assert torch.allclose(logits_before.float(), logits_after.float(), atol=1e-2)

print("OK: merge_and_unload()后模型变回LlamaForCausalLM,无lora参数残留,合并前后推理结果数值一致")
```
本机实测:合并前后同一个输入的 logits 用 `torch.allclose`(容差 1e-2,bf16 精度下的合理误差范围)确认数值一致;合并后的模型确认不再含任何名字带"lora"的参数,类型变回原生 `LlamaForCausalLM`。

**面试怎么问 + 追问链:** "合并之后还能撤销、切换回LoRA形式吗?" → 不能,`merge_and_unload()` 是破坏性操作(直接修改了权重数值,原始未合并的权重信息丢失了),如果之后还需要切换/对比不同 adapter,应该在**原始未合并**的 `peft_model` 上操作(或者提前保存好合并前的 adapter 文件,merged 模型是一个"用后即焚"的部署产物,不是实验阶段该用的形态)。

**常见坑:**
1. 对同一个 `peft_model` 对象调用 `merge_and_unload()` 会原地修改基座权重,之后这个对象已经不再是"可以切换adapter"的PEFT模型——如果还需要保留可切换的原始状态,应该在合并前用 `copy.deepcopy` 或者重新加载一份,不要在唯一的实验模型对象上直接调用。
2. `atol=1e-2` 这个误差容忍度不是随便设的——bf16 精度本身的数值误差量级就在这个范围,如果换成 fp32 计算,应该能观察到更接近于 0 的合并误差;用过于严苛的容差(比如 `atol=1e-8`)去验证 bf16 计算的一致性会得到误报的"不一致"结论。

---

## 7. 多 Adapter 切换机制

**签名/是什么:**
```
peft_model = get_peft_model(model, config_a, adapter_name="adapter_a")
peft_model.add_adapter("adapter_b", config_b)
peft_model.set_adapter("adapter_b")   # 切换当前生效的adapter
```
同一个 base model 上可以同时挂多个不同的 LoRA adapter,运行时切换"当前用哪一个"。

**一句话:** 多个 adapter 共享同一份被冻结的 base model 权重,只有各自的旁路参数(`lora_A`/`lora_B`)是独立存储的——`set_adapter()` 切换的是"前向计算时应该用哪一组旁路参数",不需要重新加载 base model。

**底层机制/为什么这样设计:** 如果每个 adapter 都要求重新加载一份完整的 base model,在"需要频繁切换多个任务/风格"的场景下(比如一个服务同时支持多种微调风格,按请求切换),显存和加载时间开销会难以承受。PEFT 库把"同一个 base model,多套旁路参数"设计成一等公民支持的场景——每个被接管的层内部,`lora_A`/`lora_B` 实际上是 `nn.ModuleDict`(以 adapter 名字为 key),`set_adapter()` 本质上是切换这个字典的"当前激活key",底层 base model 的权重从始至终只有一份,不会被重复加载。

**AI 研究/工程场景:** 研究中如果要系统比较"同一个base model,不同超参/不同训练数据训出的多个LoRA版本各自的效果",在同一个 Python session 里加载一次 base model、`add_adapter` 挂多个候选,用 `set_adapter` 切换着对比,比每次都重新加载一遍完整模型高效得多——这在 09 类"超参敏感度扫描"这种需要对比多组结果的场景里是真实有用的技巧。

**可运行例子:**
```python
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

config_a = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
peft_model = get_peft_model(model, config_a, adapter_name="adapter_a")

config_b = LoraConfig(r=8, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
peft_model.add_adapter("adapter_b", config_b)

assert set(peft_model.peft_config.keys()) == {"adapter_a", "adapter_b"}
assert peft_model.peft_config["adapter_a"].r == 4   # 两个adapter配置(比如rank)可以互不相同
assert peft_model.peft_config["adapter_b"].r == 8

# 默认激活的应该是第一个创建的adapter
assert peft_model.active_adapter == "adapter_a"

peft_model.set_adapter("adapter_b")
assert peft_model.active_adapter == "adapter_b"   # 切换生效

print(f"OK: 同一base model挂了{list(peft_model.peft_config.keys())}两个adapter(r分别为4和8),set_adapter成功切换激活状态")
```
本机实测:两个 adapter(`r=4` 和 `r=8`)成功共存在同一个 `peft_model` 对象上,`active_adapter` 属性确认切换生效。

**面试怎么问 + 追问链:** "多adapter场景下,base model的显存只占用一份吗?" → 是的,这正是这个机制的核心价值——`base_model` 的权重在内存里只有一份物理拷贝,多个 adapter 只是各自额外的、体积很小的旁路参数(knowledge point 5 讲过体积量级);追问"能不能同时激活多个adapter的效果叠加?"(标准的 `set_adapter` 是互斥切换,但 `peft` 也提供了更高级的多 adapter 组合/加权融合机制,不属于本知识点的基础范围,面试如果被深挖到这里,如实说明"基础机制是互斥切换,组合调用是进阶话题"是诚实的回答)。

**常见坑:**
1. `add_adapter` 添加的新 adapter,如果 `target_modules` 和已有 adapter 不完全一致(比如adapter_a只接管q_proj/v_proj,adapter_b还想接管o_proj),需要确认 `peft` 版本对这种"不同adapter接管不同层集合"的场景支持程度,不是所有版本/所有场景都完美支持,复杂配置建议先用小规模测试验证行为符合预期。
2. `set_adapter()` 切换的只是"哪组旁路参数参与计算",不会自动帮你处理"不同adapter可能需要配合不同的推理超参(比如不同的prompt模板)"这类上层业务逻辑,这部分依然需要调用方自己管理。

---

## 8. `PeftModel.from_pretrained` 加载 Adapter 权重

**签名/是什么:**
```
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
peft_model = PeftModel.from_pretrained(base_model, "my_adapter_dir")
```
和知识点 1 的 `get_peft_model()`(从零开始注入一个**新的、未训练**的LoRA结构)不同,`PeftModel.from_pretrained` 是加载一个**已经训练好、保存过**的 adapter,重新接回 base model 上。

**一句话:** 这个函数做的事情是"先读 `adapter_config.json` 重建出对应的 `LoraConfig`(knowledge point 1 的注入逻辑),再把 `adapter_model.safetensors` 里保存的旁路权重数值精确加载进对应位置"——本质上是 knowledge point 1(注入结构)+ knowledge point 5(保存的产物)这两个机制的组合复用,不是全新的逻辑。

**底层机制/为什么这样设计:** 把"重建LoRA结构"和"加载训练好的LoRA权重数值"这两步骤自动化成一次调用,用户不需要手动读取 `adapter_config.json`、手动构造匹配的 `LoraConfig`、再手动调用 `get_peft_model()`——这些步骤 `from_pretrained` 内部全部自动完成,只需要提供 base model 和 adapter 目录路径。这也解释了为什么这个函数**必须**接收一个已经加载好的 `base_model` 参数(它不像 `AutoModel.from_pretrained` 那样"从零开始加载一整个模型",它假设 base model 是调用方已经准备好的,自己只负责"把adapter接上去")。

**AI 研究/工程场景:** 09 类"微调产物保存与复现"知识点的核心验证逻辑就是这个函数——保存一个训练过的 adapter,重新加载,确认生成效果和保存前一致,这是验证整个微调流程"确实把训练成果正确持久化了"的关键一步,不能跳过。

**可运行例子:**
```python
import torch
import tempfile
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 第一阶段:创建一个LoRA adapter并保存(模拟"训练完成后保存"这一步,不实际训练)
model1 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
peft_model1 = get_peft_model(model1, config)
tmpdir = tempfile.mkdtemp()
peft_model1.save_pretrained(tmpdir)

# 第二阶段:重新加载一个全新的base model,把保存的adapter接回去
model2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
peft_model2 = PeftModel.from_pretrained(model2, tmpdir)

assert type(peft_model2).__name__ == "PeftModelForCausalLM"
assert peft_model2.peft_config["default"].r == 8   # 配置从adapter_config.json正确重建
assert peft_model2.peft_config["default"].target_modules == {"q_proj", "v_proj"}

# 核心验证:重新加载出来的lora权重数值,应该和保存前完全一致(逐bit相等,不是近似)
w1 = dict(peft_model1.named_parameters())["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"]
w2 = dict(peft_model2.named_parameters())["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"]
assert torch.equal(w1, w2)

print("OK: PeftModel.from_pretrained正确重建LoraConfig(r=8)并加载出和保存前逐bit相同的旁路权重")
```
本机实测:重新加载出的 `peft_config["default"].r` 精确为 8(从 `adapter_config.json` 正确重建);`lora_A` 权重张量用 `torch.equal` 确认和保存前逐 bit 相等——这是"保存/加载没有精度损失"的严格验证,不是近似意义上的"差不多"。

**面试怎么问 + 追问链:** "`PeftModel.from_pretrained` 加载时,如果 `base_model` 和保存 adapter 时用的不是同一个模型会怎样?" → 如果模型架构不兼容(比如 `target_modules` 指向的层在这个 `base_model` 里根本不存在,或者维度对不上),加载时会报错;但如果只是**版本**不同(比如同一个 `TinyLlama` 但 revision 变了,权重数值微调过),不一定会报错,而是"能加载但效果未必正确"——这也是 11 类强调"记录/pin住训练时用的确切base model版本"的实际意义所在。

**常见坑:**
1. `PeftModel.from_pretrained` 要求传入的 `base_model` 已经是加载好的模型对象(不是模型名字符串)——这个 API 设计和 `AutoModel.from_pretrained` 直接传名字很不一样,容易搞混两种调用方式。
2. 忘记先把 `base_model` 搬到目标设备(比如忘记 `.to("cuda")`)就直接 `PeftModel.from_pretrained`,加载出的 `peft_model` 依然会停留在 base model 所在的设备(通常是CPU),后续推理会因为输入张量和模型不在同一设备而报错——这个坑本质上和 02 类讲过的设备管理问题是同一类。

---

*本篇 8 个知识点全部在仓库根目录 `.venv` 真实验证通过(每个知识点独立进程验证)。*
