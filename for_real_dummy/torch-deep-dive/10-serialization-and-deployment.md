# 10 · 序列化与部署基础(Serialization and Deployment)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 训练完一个模型,怎么把它安全地存下来、换一台机器/换个设备加载回来、交给别的框架(ONNX 运行时)或者脱离 Python 环境(TorchScript/C++)跑起来——这一批讲的是"模型训练完之后"这一段经常被忽视、但线上事故率不低的机制。`state_dict()` 到底存的是什么、`torch.save(model)` 为什么现在默认加载不了、`strict=False` 到底能不能救回一个改过结构的 checkpoint——这些问题看似是"背命令行参数"的层面,但背后的机制(Python pickle 的工作原理、PyTorch 2.6 起收紧的安全默认值)是真实的面试深挖区,也是真实会导致"模型救不回来"的生产事故根源。

**本篇和前几批的关系:** `state_dict()` 里的每个 tensor 本身怎么存/取,复用 [01-tensor-memory-model.md](01-tensor-memory-model.md) 的内存模型;`strict` 加载失败时具体报什么错、ONNX 导出遇到控制流会发生什么,分别呼应 [11-debugging-and-common-errors.md](11-debugging-and-common-errors.md) 的报错解读方法论和 [08-memory-and-performance.md](08-memory-and-performance.md) 第 4 节 `jit.trace`/`jit.script` 的控制流问题——本篇第 5 节会看到,现在的 ONNX 导出器処理数据依赖控制流的方式和 08 篇讲的 `jit.trace` 已经不一样了。

**验证方法论(与前几批一致):** 本文所有代码已在仓库 `.venv`(torch 2.11.0+cu128)下实际跑通验证,涉及"文件损坏/依赖缺失才会触发"的报错路径,会现场真实构造出这个条件(比如真的删掉一个 Python 模块文件)来触发,而不是转述文档描述的行为。

**本篇统一结构(同前几批):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(现场验证,不转述)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. `state_dict()` 的本质 —— 一个和模型参数共享底层存储的 `OrderedDict`

**是什么:**
```python
sd = model.state_dict()          # 返回一个 OrderedDict[str, Tensor]
model.load_state_dict(sd)        # 把这些tensor的值拷贝进模型对应的参数/buffer里
```

**一句话:** `state_dict()` 不是模型的"快照"或者深拷贝——它返回的 `OrderedDict`,键是参数/buffer 的层级名字(比如 `"backbone.0.weight"`),值就是模型当前那些参数 tensor **本身**(共享同一块底层存储),这一点在"要不要在存之前手动 clone"这类问题上是决定性的细节。

**底层机制/为什么这样设计:**

`nn.Module.state_dict()` 内部遍历 `self._parameters`/`self._buffers`/子模块(呼应 [03-nn-module-internals.md](03-nn-module-internals.md) 第 2、3 节的自动注册机制),把每个 tensor 按层级路径拼出的 key 直接放进一个 `OrderedDict` 里——这一步是**浅拷贝**,只是把已经存在的 tensor 对象引用装进一个新的字典容器,不会新分配一份 tensor 数据。这样设计的好处是获取 `state_dict()` 本身几乎零开销(不管模型多大,只是拷贝了一堆引用),但代价是:如果你后续原地修改了模型参数,之前拿到的 `state_dict()` 引用会跟着一起变(因为本来就是同一块内存)。

**可运行例子:**

```python
import torch, torch.nn as nn

model = nn.Linear(3, 2)
sd = model.state_dict()

# 验证:sd里的tensor和model参数是同一块存储,不是拷贝
assert model.weight.data_ptr() == sd['weight'].data_ptr()

with torch.no_grad():
    model.weight[0, 0] = 999.0
assert sd['weight'][0, 0].item() == 999.0    # 拿到state_dict之后修改模型,sd也跟着变了!

# 真的想要一份"当前时刻的快照"(比如保存最优checkpoint、后面还要继续训练),必须手动clone
sd_snapshot = {k: v.clone() for k, v in model.state_dict().items()}
with torch.no_grad():
    model.weight[0, 0] = -1.0
assert sd_snapshot['weight'][0, 0].item() == 999.0   # 快照不受后续修改影响
assert model.weight[0, 0].item() == -1.0

# torch.save 内部会把tensor真正序列化写入磁盘(不是引用),所以"存下来的那一刻"是安全的快照,
# 这里的坑只发生在"save之前,你还在内存里拿着这个dict做其他事"的中间过程
```

**AI 研究场景:** 训练循环里如果想"保存历史最优的几个 checkpoint 到内存里做模型平均(model soup / SWA 类技术)而不是每次都写磁盘",必须记得手动 `clone()` 每个 tensor,直接 `state_dict()` 存进一个列表会导致列表里所有"历史快照"最终都指向同一份、也就是训练结束时最新的参数——这是一个很容易在实现 EMA/SWA 这类技术时踩到的真实坑。

**面试怎么问 + 追问链:**
- **Q:** "`model.state_dict()` 返回的东西,和模型当前的参数是什么关系?"—— 期望答"共享底层存储的引用,不是拷贝"。
- **追问 1(核心):** "如果我拿到 `state_dict()` 之后又继续训练了几步,这个 dict 里的值会变吗?"—— 期望答"会,因为是同一份内存,除非手动 `.clone()`",最好能现场举出上面这类"实现模型平均忘记clone"的例子。
- **追问 2:** "那 `torch.save(model.state_dict(), path)` 这个调用本身安全吗,会不会存下一个'半成品'?"—— 期望能区分"内存里持有引用"和"序列化写入磁盘"是两件事:`torch.save` 执行的那一刻,会把当时的 tensor 值真实写进文件,写完之后就和内存里的模型状态无关了,这一步操作本身是安全的快照。

**常见坑:** 把 `state_dict()` 存进一个 Python list/dict 当"历史快照"收集起来,却没有 clone,导致训练结束后发现所有"快照"其实都是同一份最终参数——这在实现 checkpoint 平均、EMA 手动实现这类需要长期持有多份历史参数的场景里是真实高发的 bug,而且不会报错,只会在最后对比几个"快照"发现数值完全一样时才会意识到问题。

---

## 2. `strict=False` —— 精确按 key 名字匹配,不做任何"智能"推断

**是什么:**
```python
result = model.load_state_dict(checkpoint_sd, strict=False)
result.missing_keys       # checkpoint里没有、但模型需要的key
result.unexpected_keys    # checkpoint里有、但模型不认识的key
```

**一句话:** 默认 `strict=True` 要求 checkpoint 里的 key 集合和模型当前的 key 集合**完全一致**,少一个多一个都直接报错;`strict=False` 放宽这个要求,只加载"两边都有的、且形状匹配的" key,并把"该有没有的"和"有但用不上的" key 分别收集进 `missing_keys`/`unexpected_keys` 返回给你,但**不会做任何猜测性的名字匹配**——这是本节最容易被误解的一点。

**底层机制/为什么这样设计:**

`load_state_dict` 内部本质是逐个 key 做字符串精确匹配,`strict=False` 只是把"匹配不上就报错退出"改成"匹配不上就记录下来,继续处理其它 key",匹配逻辑本身没有变得更聪明。这意味着如果你把某一层从 `self.old_name` 改名成 `self.new_name`,`strict=False` 并不会意识到"这其实是同一层、只是换了个名字"——它会把 `old_name.*` 计入 `unexpected_keys`(checkpoint里有,新模型不认识,直接丢弃)、把 `new_name.*` 计入 `missing_keys`(新模型需要,checkpoint里没有,保持随机初始化),旧权重就这样安静地丢失了,不会有任何报错提示你"其实这两个是同一层"。

**可运行例子:**

```python
import torch, torch.nn as nn

class OldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)

class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)   # 名字没变,能对上
        self.head = nn.Linear(4, 2)        # 新增的层

old_sd = OldModel().state_dict()
new_model = NewModel()

try:
    new_model.load_state_dict(old_sd)             # strict=True(默认)
    assert False, "应该要报错"
except RuntimeError as e:
    assert 'Missing key(s)' in str(e)
    assert 'head.weight' in str(e)

result = new_model.load_state_dict(old_sd, strict=False)
assert result.missing_keys == ['head.weight', 'head.bias']
assert result.unexpected_keys == []
assert torch.equal(new_model.backbone.weight, OldModel().backbone.weight) is False  # 不同随机初始化,仅结构对比
# backbone权重确实被正确加载了(和old_sd里的值一致):
old_backbone_weight = old_sd['backbone.weight']
assert torch.equal(new_model.backbone.weight, old_backbone_weight)

# 改名场景:strict=False 不会做"智能匹配",旧key进unexpected,新key进missing,权重直接丢失
class RenamedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.new_name = nn.Linear(4, 4)   # 从 backbone 改名成 new_name
        self.head = nn.Linear(4, 2)

renamed = RenamedModel()
result2 = renamed.load_state_dict(old_sd, strict=False)
assert 'new_name.weight' in result2.missing_keys       # 新名字:找不到对应的旧key
assert 'backbone.weight' in result2.unexpected_keys      # 旧名字:新模型不认识
# renamed.new_name 的权重依然是随机初始化,checkpoint里的backbone权重被彻底忽略了
assert not torch.equal(renamed.new_name.weight, old_backbone_weight)
```

**AI 研究场景:** 最常见的正当用法是**迁移学习/微调**——用一个预训练 backbone 的 checkpoint,加载进一个"backbone结构相同 + 新增了自定义任务头"的模型,`strict=False` 让 backbone 部分正常加载、新增的 head 保持随机初始化等待训练,这是加载预训练权重的标准写法;但如果 checkpoint 来自一个经过重构(改过层名字)的旧版本代码,`strict=False` 不能自动"接上"改名前后的权重,需要手动写一个 key 名字的映射(重命名 `state_dict` 里的 key)再加载,不能指望 `strict=False` 帮你做这件事。

**面试怎么问 + 追问链:**
- **Q:** "`strict=False` 是做什么用的,典型场景是什么?"—— 期望提到"迁移学习加载预训练backbone、模型新增了层"这个典型场景。
- **追问 1(陷阱题,区分度很高):** "如果我把模型里某一层改了个名字,`strict=False` 能不能把 checkpoint 里对应的旧权重正确加载到新名字上?"—— 期望答"不能,`strict=False` 只做精确字符串匹配,改名后的层会被当成'全新的、checkpoint里没有'处理,旧权重进 unexpected_keys 被丢弃,新名字进 missing_keys 保持随机初始化"——这正是上面验证的现象,很多人会想当然认为"名字类似"或者"形状匹配"就能对上,这是错的。
- **追问 2(工程向):** "如果确实需要处理改名的情况,应该怎么做?"—— 期望能想到"手动构造一个新的 dict,把旧 key 名字映射成新 key 名字之后再 `load_state_dict`",比如 `{k.replace('backbone', 'new_name'): v for k, v in old_sd.items()}`。
- **追问 3:** "`missing_keys`/`unexpected_keys` 都是空列表,是不是就说明加载完全正确了?"—— 期望能提到"key 名字对上、形状也对上不等于语义对上"——比如两个不同任务预训练出来的模型,即使层名字和形状恰好完全一致,加载后权重的"含义"也可能完全不适配当前任务,`load_state_dict` 不做任何语义层面的检查。

**常见坑:** 把 `strict=False` 当成"万能兜底开关",加载完不检查 `missing_keys`/`unexpected_keys` 就直接开始训练——如果因为改名、拼写错误等原因导致大部分权重其实没加载成功(比如预期只有新增的 head 该出现在 missing_keys 里,结果 backbone 也整个出现在里面),模型会用一份基本随机初始化的参数"看起来正常"地开始训练,不会报错,只会体现为"loss 下降得比预期慢很多、最终精度上不去"这种容易被误诊断成其他问题的现象。

---

## 3. `map_location` —— 跨设备加载 checkpoint

**是什么:**
```python
torch.load(path, map_location='cpu')                       # 强制加载到CPU
torch.load(path, map_location='cuda:0')                     # 强制加载到指定GPU
torch.load(path, map_location={'cuda:1': 'cuda:0'})         # 把"保存时的cuda:1"重映射到"当前的cuda:0"
torch.load(path, map_location=lambda storage, loc: storage) # 自定义映射逻辑
```

**一句话:** `torch.save` 会把每个 tensor 保存时所在的设备信息也记录进文件;不指定 `map_location` 时,`torch.load` 默认会尝试把 tensor 加载回**保存时那个设备**——如果保存checkpoint的机器有4张GPU、当前机器只有1张甚至没有GPU,不加 `map_location` 直接加载大概率会报错,`map_location` 就是用来覆盖这个"默认按原设备加载"行为的。

**底层机制/为什么这样设计:**

官方 docstring 原文(已验证):"storages... are first deserialized on the CPU and are then moved to the device they were saved from. If this fails (e.g. because the run time system doesn't have certain devices), an exception is raised."——反序列化本身永远先在 CPU 上完成,"move to original device"是额外的一步,`map_location` 本质是**替换掉这最后一步的目标设备**,而不是改变反序列化过程本身。

**可运行例子:**

```python
import torch, torch.nn as nn

model = nn.Linear(3, 2).to('cuda')
path = '_tmp_gpu_ckpt.pt'
torch.save(model.state_dict(), path)

sd_default = torch.load(path)                 # 不指定map_location
assert sd_default['weight'].device.type == 'cuda'   # 默认:回到保存时的设备(这里是cuda:0)

sd_cpu = torch.load(path, map_location='cpu')
assert sd_cpu['weight'].device.type == 'cpu'

sd_lambda = torch.load(path, map_location=lambda storage, loc: storage)
assert sd_lambda['weight'].device.type == 'cpu'   # lambda原样返回反序列化时的storage(已经在CPU上)

sd_dict = torch.load(path, map_location={'cuda:0': 'cpu'})
assert sd_dict['weight'].device.type == 'cpu'

import os; os.remove(path)
```

**AI 研究场景:** 最典型的场景是"在一台多卡机器上用 `cuda:2` 训练并保存了 checkpoint,后续想在一台只有 `cuda:0` 甚至没有 GPU 的机器上做推理/继续训练"——不加 `map_location='cpu'`(或者对应的目标GPU)直接 `torch.load`,会因为目标机器没有 `cuda:2` 这个设备直接抛异常;`map_location=torch.device('cpu')` 是最常见、最保险的写法(先加载到CPU,后续再用 `model.to(device)` 显式挪到你真正想用的设备,把"加载"和"放到哪张卡"这两件事拆开处理,不容易出错)。

**面试怎么问 + 追问链:**
- **Q:** "在一台只有1张卡(或没有GPU)的机器上,加载一个用4卡训练保存的checkpoint,会发生什么?为什么?"—— 期望答"如果不指定 `map_location`,大概率会报错,因为默认行为是尝试把tensor放回保存时的原设备,原设备在当前机器上不存在"。
- **追问 1:** "推荐的、最稳妥的做法是什么?"—— 期望答"先用 `map_location='cpu'` 加载到CPU,再用 `model.to(device)` 显式挪到目标设备,不要依赖默认的'自动放回原设备'行为"。
- **追问 2(容易漏答):** "`map_location` 只影响 `torch.load` 拿到的 tensor 在哪个设备,还是会影响模型代码本身的行为?"—— 期望能区分清楚:`map_location` 只决定 `state_dict` 里 tensor 的存放设备,`model.load_state_dict()` 会把这些值拷贝进模型已有的参数(参数本身在哪个设备,取决于你之前有没有对 `model` 调用过 `.to(device)`,和 `map_location` 无关)——两者是独立的两步。

**常见坑:** 以为不指定 `map_location` 就是"默认加载到CPU"——实际默认行为是"尝试加载回保存时的设备",这在单机单卡开发时不容易发现问题(保存和加载往往是同一张卡),一旦换机器/换卡数就容易在没有防备的情况下直接报错。

---

## 4. 只存 `state_dict()` vs 存整个模型对象 —— 不只是"最佳实践",现在是默认就跑不起来

**是什么:**
```python
torch.save(model.state_dict(), path)   # 官方推荐:只存参数/buffer这份纯数据
torch.save(model, path)                 # 存整个模型对象(包含类结构、方法等)
```

**一句话:** `torch.save(model)` 存的不只是参数数值,还包括"这是哪个 Python 类的实例"这份信息(通过 pickle 协议记录模块路径+类名);`torch.load` 加载这种"整个对象"存档时,本质是在**反序列化并重建一个任意 Python 对象**,这正是 PyTorch 从 2.6 版本起把 `torch.load` 的 `weights_only` 参数默认值从 `False` 改成 `True`(已验证:本机 `torch.load` 签名及 docstring 都显示当前默认就是 `weights_only=True`)所要防范的风险——`state_dict()` 只是一个纯粹的 `tensor` 字典,不涉及重建任意对象,天然对得上 `weights_only=True` 的安全要求。

**底层机制/为什么这样设计:**

Python 的 `pickle` 序列化协议,对于自定义类的实例,存的是"这个对象属于哪个模块的哪个类 + 构造/还原它需要的数据",**不是**类的源代码本身——反序列化时,pickle 会尝试 `import` 这个模块、找到这个类,再用存的数据重建实例。这带来两个独立的真实问题:

**问题①安全性(现场验证 PyTorch 2.6 起的默认防御):**

```python
import torch, torch.nn as nn

model = nn.Linear(3, 2)
torch.save(model, 'whole_model.pt')     # 存整个对象

try:
    torch.load('whole_model.pt')          # 默认 weights_only=True(已验证当前版本默认值)
    assert False
except Exception as e:
    assert type(e).__name__ == 'UnpicklingError'
    # 实测报错原文包含: "In PyTorch 2.6, we changed the default value of the `weights_only`
    # argument in `torch.load` from `False` to `True`... it can result in arbitrary code execution"
    assert 'arbitrary code execution' in str(e)
```

这不是一个晦涩的边界情况——**默认配置下,`torch.load` 一个整模型对象的 checkpoint 直接就加载不了**,因为反序列化任意类实例这件事本身,理论上可以被恶意构造的文件用来在你的机器上执行任意代码(pickle 协议允许在反序列化过程中调用任意可调用对象),`weights_only=True` 通过只允许反序列化 tensor/基本容器这类"已知安全"的类型,从根子上堵住这个口子——`state_dict()` 恰好就是这样一个安全类型,不受影响。

**问题②即使显式绕开安全检查,还要求原始类定义在加载时依然可 import(现场真实构造这个场景验证,不是推测):**

```python
# 步骤1:在一个独立模块里定义类,存整个模型对象
# custom_model_def.py:
#     class CustomModel(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.fc = nn.Linear(3, 3)
# 保存: torch.save(CustomModel(), 'whole_model_custom.pt')

# 步骤2:删掉 custom_model_def.py,模拟"代码库重构后旧模块被移除/换了路径"这个真实场景

# 步骤3:即使显式传 weights_only=False 绕开安全限制,依然会失败:
try:
    torch.load('whole_model_custom.pt', weights_only=False)
    assert False
except ModuleNotFoundError as e:
    assert "No module named 'custom_model_def'" in str(e)
# 实测:确实精确抛出这个错误 —— pickle 反序列化时尝试 import 原始模块,
# 模块文件已经不存在,不管信不信任这个文件来源,这份 checkpoint 都救不回来了
```

**AI 研究场景:** 这也是为什么 HuggingFace 等生态的模型分发格式(`safetensors`)进一步收紧了这个思路——`safetensors` 从格式设计上就**只能**存 tensor 数据,连 pickle 协议都不用,连"理论上支持反序列化任意类型"的可能性都不存在,读取速度也因为跳过了 pickle 的通用反序列化逻辑而更快;理解"为什么模型分发生态都在往'只存纯数据'的方向收敛",本质就是本节验证的这两个问题。

**面试怎么问 + 追问链:**
- **Q:** "官方为什么推荐只保存 `state_dict()`,而不是整个模型对象?"—— 期望提到"安全性"和"代码依赖"两个独立的原因,而不是只说"这是最佳实践"。
- **追问 1(时效性强,考察是否跟进新版本变化):** "如果我现在(2026年,PyTorch 2.6+)直接 `torch.load` 一个别人给我的、存了整个模型对象的 `.pt` 文件,会发生什么?"—— 期望知道"默认情况下会直接报错(`weights_only=True` 是新默认值),需要显式设 `weights_only=False` 才能加载,而这个参数本身的存在就是在提醒你'这样做有风险'"。
- **追问 2(深挖,两个原因要分清楚):** "就算我确认这个文件绝对安全、也显式设了 `weights_only=False`,还有什么可能导致加载失败?"—— 期望答"如果保存时用的自定义模型类,在加载时的代码库里已经被删除/改了模块路径/改了类定义,pickle 反序列化会在 import 这个类的阶段直接失败(ModuleNotFoundError 或 AttributeError),这和信不信任文件来源是两个独立的问题",这正是上面第二个实验验证的场景。
- **追问 3(工程向):** "只存 `state_dict()` 是不是完全没有代价?"—— 期望答"不是免费的:加载时你必须自己先构造出一个结构完全匹配的模型实例,再 `load_state_dict` 进去——这要求'构造模型的代码'本身要被妥善保存/版本控制好,只是把'依赖模型类代码'这个问题从'pickle自动重建'转移成了'你自己要留着能重新构造模型的代码',并没有把这个依赖完全消除"。

**常见坑:**
- 从旧代码(PyTorch 2.6 之前)迁移过来的训练脚本,如果保存的是整个模型对象,升级 PyTorch 版本后重新加载会突然报错,容易被误判成"环境坏了""版本不兼容"这类无关问题,实际根因是 `weights_only` 默认值变了。
- 以为"信任这个 checkpoint 的来源"就可以放心用 `weights_only=False`——即使来源可信,只要保存时的自定义类代码后续被重构/删除过,一样加载不回来,"信任来源"只解决安全问题,不解决代码依赖问题。

---

## 5. ONNX 导出简介 —— 现在的导出器,对付不了的控制流会直接报错,不会悄悄导出错误模型

**是什么:**
```python
torch.onnx.export(model, (example_input,), "model.onnx", input_names=["x"], output_names=["y"],
                   dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}})
```
把一个 PyTorch 模型导出成 ONNX(Open Neural Network Exchange)格式——一种跨框架的模型交换格式,导出之后可以用 `onnxruntime` 等专门的推理引擎加载运行,不再依赖 PyTorch/Python 环境。

**一句话:** 本机安装的 torch 版本(2.11.0)默认走的是基于 `torch.export`(而不是老式 `jit.trace`)的导出路径——这个新路径会先尝试把模型转换成一张严格的计算图,如果模型里有依赖运行时数值的控制流(比如 [08-memory-and-performance.md](08-memory-and-performance.md) 第 4 节讲过的那种 `if x.sum() > 0` 分支),**导出会直接失败并报出清晰的错误**,而不是像老式 `jit.trace` 那样安静地把某一条分支焊死、导出一个"能跑但可能跑错"的模型——这是一个真实的、值得记录的改进。

**底层机制/为什么这样设计:**

`torch.export.export`(新导出路径的核心)在构图时,遇到需要根据一个"依赖具体输入数值"的判断(比如 `x.sum() > 0` 这种结果是 `True`/`False` 但取决于输入数据的表达式)来决定图结构走哪条分支时,不会像 `jit.trace` 那样"直接按这一次实际走的分支往下记录、假装这是唯一的路径",而是主动检测到"这个判断没法在不知道具体输入的情况下静态确定",并抛出 `GuardOnDataDependentSymNode` 异常终止导出——用一次"导出失败"换掉 08 篇讲过的"trace 悄悄导出错误模型"这个更危险的结果。

**可运行例子:**

```python
import torch, torch.nn as nn

# 场景1:含数据依赖控制流的模型,导出会直接失败(不是悄悄导出错的)
class BranchyModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x * 2
        else:
            return x * -1

m = BranchyModel(); m.eval()
pos_input = torch.tensor([1.0, 2.0, 3.0])
try:
    torch.onnx.export(m, (pos_input,), "branchy.onnx")
    assert False, "预期导出应该失败"
except Exception as e:
    # 实测精确报出: GuardOnDataDependentSymNode: Could not guard on data-dependent expression
    assert "GuardOnDataDependentSymNode" in str(e) or "data-dependent" in str(e)

# 场景2:不含数据依赖控制流的正常模型,导出成功,且用onnxruntime交叉验证数值一致
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
model.eval()
x = torch.randn(1, 4)
torch.onnx.export(model, (x,), "normal.onnx", input_names=["x"], output_names=["y"],
                   dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}})

import onnx
onnx_model = onnx.load("normal.onnx")
onnx.checker.check_model(onnx_model)     # 结构合法性检查,通过说明导出的图本身没问题

import onnxruntime as ort
sess = ort.InferenceSession("normal.onnx")
onnx_out = sess.run(None, {"x": x.numpy()})[0]
torch_out = model(x).detach().numpy()

import numpy as np
assert np.allclose(onnx_out, torch_out, atol=1e-5)   # ONNX Runtime算出来的结果和PyTorch一致

# dynamic_axes 生效验证:换一个不同的batch size,同一份onnx模型依然能跑
x_batch4 = torch.randn(4, 4)
onnx_out_batch4 = sess.run(None, {"x": x_batch4.numpy()})[0]
assert onnx_out_batch4.shape == (4, 2)
```

**AI 研究场景:** 把训练好的模型交给推理服务(尤其是非 Python 技术栈,比如 C++/Java 后端,或者需要用 TensorRT 这类专门推理引擎进一步优化)时,ONNX 是最通用的中间交换格式;上面验证的"含控制流会直接导出失败"这个特性,意味着**为了能导出 ONNX,模型的 `forward` 里应该尽量避免依赖输入数值的 Python 控制流**——这是一条真实影响模型代码写法的工程约束,不只是"导出工具的限制",很多推理友好的模型架构设计都会主动规避这类结构。

**面试怎么问 + 追问链:**
- **Q:** "ONNX 导出的基本流程是什么?能处理任意 PyTorch 模型吗?"—— 期望答"不能,依赖输入数值的控制流没法直接导出,需要模型结构本身对导出友好"。
- **追问 1(联系08篇,考察知识迁移):** "这和 08 篇讲的 `torch.jit.trace` 处理控制流的问题是同一回事吗?"—— 期望能对比:"都是'追踪型'机制在控制流上的局限,但现在的 ONNX 导出器遇到这种情况会直接报错终止,不像老式 `jit.trace` 那样悄悄固化一条分支、生成一个能跑但可能算错的模型——检测能力更强,是一个真实的改进方向"。
- **追问 2(工程向):** "如果模型里确实有必要的控制流(比如不同 batch 有没有 padding 走不同逻辑),怎么办?"—— 开放题,合理方向包括:改写成不依赖数据的、用张量运算表达的等价逻辑(比如用 mask 相乘代替 if 分支)、或者把有控制流的部分拆出来单独处理、不强行整体导出。
- **追问 3:** "`dynamic_axes` 是做什么的?不设置会怎样?"—— 期望答"声明哪些维度(通常是 batch size)在推理时可以变化,不设置的话导出的 ONNX 模型会把当时 example input 的具体形状写死,只能用完全一样的输入形状去推理"——上面例子验证了设置之后,同一份模型确实能接受不同 batch size。

**常见坑:** 忘记在导出前调用 `model.eval()`——本机实测会产生明确的警告("Exporting a model while it is in training mode... Calling model.eval() before export is recommended"),训练模式下 `Dropout`/`BatchNorm` 的行为和推理模式不一致(呼应 03 篇 `train()`/`eval()` 模式切换的内容),忘记切换会导出一个行为和"实际想部署的推理版本"不一致的模型,这类问题往往要等到线上推理结果和本地测试对不上才会被发现。

---

## 6. TorchScript 序列化 —— 存下来的是图,不是"这是哪个 Python 类"

**是什么:**
```python
scripted = torch.jit.script(model)   # 08篇第4节讲过:静态分析源码,正确处理控制流
scripted.save("model.pt")
loaded = torch.jit.load("model.pt")   # 加载完全不需要原始的 Python 类定义
```

**一句话:** `torch.jit.save`/`torch.jit.load` 存取的是 TorchScript 编译出的**计算图 + 权重**,不依赖 pickle 那种"记录类路径、加载时重新 import 原始类"的机制——这是它和"第 4 节讲的整模型对象 pickle 存储"最本质的区别,也是它能脱离 Python 环境、被 C++ (`libtorch`) 直接加载运行的原因。

**底层机制/为什么这样设计:**

`torch.jit.script` 把模型的 `forward`(及其调用到的方法)编译成 TorchScript 这种独立于 Python 解释器的中间表示(呼应 08 篇第 4 节验证过的 `prim::If` 这类图节点);`.save()` 序列化的是这份图结构本身加上参数数值,`.load()` 加载回来得到的是一个 `RecursiveScriptModule` 对象,它执行 `forward` 靠的是这份已编译的图,不需要重新 `import` 你写的原始 `nn.Module` 子类。

**可运行例子:**

```python
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
model.eval()
scripted = torch.jit.script(model)
scripted.save("scripted_model.pt")

loaded = torch.jit.load("scripted_model.pt")
x = torch.randn(3, 4)
assert torch.allclose(model(x), scripted(x))
assert torch.allclose(model(x), loaded(x))    # 加载回来的结果和原模型完全一致

# 关键区别:loaded 对象根本不属于原来定义模型的那个模块/类
assert loaded.__class__.__module__ == 'torch.jit._script'
assert loaded.__class__.__name__ == 'RecursiveScriptModule'
# 不是 'torch.nn.modules.container' 或任何用户自定义模块路径——
# 印证"加载不依赖原始nn.Module子类的Python源码"这一点

import os
os.remove("scripted_model.pt")
```

**AI 研究场景:** 需要把模型部署进不依赖 Python 运行时的环境(比如用 C++ `libtorch` 直接加载做推理、移动端部署)时,TorchScript 是官方长期支持的路径之一(和 ONNX 相比,TorchScript 更"原生",少了一层格式转换,但生态上不如 ONNX 通用,能对接的推理引擎/硬件厂商支持不如 ONNX 广);和第 4 节的"整模型对象 pickle"相比,TorchScript 序列化天然规避了"依赖原始类定义仍然可 import"这个问题,是官方认可的、可以拿去做跨环境部署的存储格式,而普通 `torch.save(model)` 明确不建议用来做这件事。

**面试怎么问 + 追问链:**
- **Q:** "TorchScript 的 `save`/`load` 和 `torch.save(model)`/`torch.load` 存整个模型对象,有什么本质区别?"—— 期望答"TorchScript 存的是编译好的计算图,加载不需要原始 Python 类定义;`torch.save(model)` 靠 pickle 记录类路径,加载时必须能重新 import 到那个类"。
- **追问 1:** "这意味着 TorchScript 存下来的模型,能不能脱离 Python 环境使用?"—— 期望答"能,这也是它存在的核心目的之一,C++ 的 `libtorch` 可以直接加载 TorchScript 模型做推理,不需要嵌入一个 Python 解释器"。
- **追问 2(联系08篇):** "`jit.script` 和 `jit.trace` 都能拿去 `save`/`load`,选哪个更好?"—— 期望能直接调用 08 篇第 4 节的结论:"优先 `script`,因为它能正确处理控制流;`trace` 遇到依赖数值的分支会把某条路径焊死,这个问题在'导出后脱离原始代码运行'的场景里更危险,因为运行环境往往没法再方便地用不同输入去交叉验证导出的模型对不对"。

**常见坑:** 以为只要能 `torch.jit.script(model)` 成功,模型就一定能被 `libtorch`/移动端环境正常加载运行——`script` 编译成功只保证"这段 Python 代码能被 TorchScript 的受限语法子集正确解析",不保证模型里用到的每一个算子在目标部署环境(尤其是移动端、某些精简过的推理运行时)都有对应实现,某些算子在特定后端缺失是部署阶段才会暴露的独立问题。

---

## 小结:这一批 6 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | `state_dict()` 本质 | 和模型参数共享底层存储的浅拷贝字典,不是快照;需要长期持有的历史副本必须手动`.clone()` |
| 2 | `strict=False` | 只做精确key字符串匹配,不做智能的改名识别;missing_keys/unexpected_keys需要主动检查,不能盲信 |
| 3 | `map_location` | 默认尝试加载回保存时的原设备,跨机器/跨卡数场景不指定容易直接报错;稳妥做法是先map到CPU再手动`.to(device)` |
| 4 | state_dict vs 整个模型对象 | PyTorch 2.6起`weights_only=True`成为默认值,整模型pickle默认加载不了;即使绕开安全检查,还依赖原始类代码仍可import(已用真实删除模块文件验证两种独立失败模式) |
| 5 | ONNX 导出 | 现行`torch.export`导出路径遇到数据依赖控制流会直接报错终止,不像`jit.trace`那样悄悄导出错误模型;正常模型可导出+用onnxruntime交叉验证数值一致 |
| 6 | TorchScript 序列化 | 存的是编译后的图,不依赖pickle的类路径机制,加载不需要原始Python类定义,可脱离Python环境(如libtorch)运行 |

下一批:[11-debugging-and-common-errors.md](11-debugging-and-common-errors.md) —— 调试与常见报错精解(也是这个约100个知识点系列的收尾)。

---

*更新:2026-07-07*
