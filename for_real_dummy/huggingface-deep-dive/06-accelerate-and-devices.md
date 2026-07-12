# 06 · Accelerate 分布式与设备机制(Accelerate Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)。已在仓库根目录 `.venv` 真实跑通(`accelerate==1.13.0`)。**`Accelerator` 的配置状态是进程级单例——同一个 Python 进程内创建过一次 `Accelerator()` 后,不能再用不同参数重新创建,本篇每个知识点的例子因此都设计成能在独立进程里运行,不共享状态。**

---

## 1. `accelerator.prepare()` 内部机制

**签名/是什么:**
```
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer = accelerator.prepare(model, optimizer)
```
`prepare()` 把普通的 PyTorch 对象(模型、优化器、DataLoader)包装/搬运成能在当前 accelerator 配置(单卡/多卡/混合精度等)下正确工作的版本。

**一句话:** 单卡场景下,`prepare()` 对模型做的事情本质上就是知识点等价于 `.to(accelerator.device)`;它真正的威力在于**同一行代码**在单卡、多卡、混合精度等不同配置下都能正确工作,不需要为每种场景写不同的搬运逻辑。

**底层机制/为什么这样设计:** `accelerator.device` 在初始化时已经根据当前环境(有没有 GPU、有没有配置分布式)确定好了应该用哪个设备;`prepare()` 内部按传入对象的类型分别处理:模型和张量做设备搬运,优化器需要在模型搬运**之后**重新关联参数(因为优化器内部持有的参数引用不能失效),DataLoader 在分布式场景下还会被包装成能自动给每个进程分发不同数据分片的版本(呼应知识点 6)。这一整套"看类型分发处理逻辑"的设计,让用户代码只需要写"给我 prepare 这几个对象"这一行,不需要关心背后具体要做哪几件不同的事。

**AI 研究/工程场景:** 09 类的微调实验代码,不管最终是在这台单卡机器上跑、还是未来换到多卡集群上跑,理论上都不需要改动"prepare 模型和优化器"这几行代码——这是 `accelerate` 存在的核心价值:同一份训练代码在不同硬件配置之间可移植。

**可运行例子:**
```python
import torch
from torch import nn
from accelerate import Accelerator

accelerator = Accelerator()
model = nn.Linear(10, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model2, optimizer2 = accelerator.prepare(model, optimizer)

assert accelerator.device.type == "cuda"  # 本机有GPU,accelerator自动选择了它
assert next(model2.parameters()).device.type == "cuda"  # 模型被自动搬到了正确设备

# 单卡场景下,prepare返回的就是原对象本身(搬了设备,但没有额外包装一层)
assert model2 is model

print(f"OK: accelerator.device={accelerator.device},prepare后模型已在该设备上,单卡场景返回同一对象")
```
本机实测:`accelerator.device` 自动解析为 `cuda`,`prepare()` 之后模型参数确实在 `cuda:0` 上,且 `model2 is model` 为 `True`(单卡场景下没有额外包装,这一点在分布式场景下会不同——`DistributedDataParallel` 包装会让返回对象和原对象不是同一个)。

**面试怎么问 + 追问链:** "为什么 `optimizer` 也需要 `prepare()`,不能只 prepare 模型吗?" → 优化器初始化时通过 `model.parameters()` 拿到的是参数张量的引用,如果模型后续被搬运/包装、参数对象的身份发生了变化(比如被 DDP 包装后,底层参数可能是不同的对象),优化器如果还持有搬运前的旧引用,更新的就是"死"的参数,不会真正影响模型——`prepare()` 必须处理优化器,确保它引用的是搬运后的、真正在起作用的那份参数。

**常见坑:**
1. `prepare()` 的调用顺序有讲究——必须先 `prepare` 模型,再用 prepare 后的模型参数构造优化器,或者像本例这样把模型和优化器一起传给 `prepare()`,不能先构造优化器(引用了搬运前的模型参数)再单独 `prepare` 模型,那样优化器持有的引用会失效。
2. 不要在业务代码里到处手写 `.to(accelerator.device)` 又同时用 `prepare()`——这两种设备管理方式混用容易导致部分张量在 CPU、部分在 GPU 的不一致状态,应该统一交给 `prepare()` 处理。

---

## 2. 混合精度自动处理(`mixed_precision` 参数)

**签名/是什么:**
```
accelerator = Accelerator(mixed_precision="bf16")   # 或 "fp16" / "no"
```
声明式地指定训练用什么混合精度策略,`accelerator.backward(loss)` 内部会自动处理对应的 loss scaling / autocast 逻辑,不需要像裸 PyTorch 那样手动管理 `GradScaler`。

**一句话:** **`AcceleratorState` 是整个 Python 进程级别的单例**——这不是一个无关紧要的实现细节,而是本知识点最重要的一条真实发现:同一个进程里已经创建过一次 `Accelerator()` 之后,不能再用不同的 `mixed_precision` 参数创建第二个实例,会直接报错要求"重启运行时"。

**底层机制/为什么这样设计:** 混合精度、设备配置这些属性一旦确定,会影响后续几乎所有 `accelerate` API 的行为,如果允许同一进程内反复用不同配置重新初始化,会给"当前到底处于什么状态"引入巨大的不确定性(尤其是分布式场景下,状态不一致可能导致不同进程之间的行为对不上)。把这类全局配置做成进程级单例(拿到的永远是同一个 `AcceleratorState` 实例,后续调用只是校验参数没有冲突,不会真的重新初始化),是用"更严格的约束"换"状态可预测性"的典型设计取舍。

**AI 研究/工程场景:** 如果你在同一个 Jupyter/交互式 session 里先跑了一次 `Accelerator()`(哪怕用的是默认配置),之后想换一个 `mixed_precision` 配置重新实验,**必须重启 Python 进程/kernel**,不能指望"重新执行一遍初始化代码"就能生效——这是实际调试时非常容易踩、又容易被误判成"代码写错了"的一个坑。

**可运行例子:**
```python
from accelerate import Accelerator
from accelerate.state import AcceleratorState

# 全新进程里第一次创建Accelerator,mixed_precision设置应该正确生效
accelerator = Accelerator(mixed_precision="bf16")

assert accelerator.mixed_precision == "bf16"
assert accelerator.state.mixed_precision == "bf16"
assert isinstance(accelerator.state, AcceleratorState)

# 验证"进程级单例"这个关键性质:再次调用Accelerator()(不传参数),
# 拿到的应该是同一个底层状态,而不是重置成默认值
accelerator2 = Accelerator()
assert accelerator2.mixed_precision == "bf16"  # 复用了第一次设置的状态,不是"no"默认值

print(f"OK: mixed_precision='bf16'生效;后续不传参数重新创建Accelerator()仍复用同一状态(单例)")
```
本机实测:全新进程里 `Accelerator(mixed_precision="bf16")` 正确设置;**在同一进程里如果先创建过默认配置的 `Accelerator()`,再尝试 `Accelerator(mixed_precision="bf16")` 会直接抛 `ValueError`**("AcceleratorState has already been initialized and cannot be changed, restart your runtime completely..."),这是本机现场触发的真实报错,不是文档描述,验证了"进程级单例"这条结论。

**面试怎么问 + 追问链:** "`accelerate` 的 `mixed_precision='bf16'` 和 PyTorch 原生的 `torch.autocast(dtype=torch.bfloat16)` 是什么关系?" → `accelerate` 内部就是基于 `autocast`/`GradScaler`(fp16场景需要,bf16因为动态范围更大通常不需要loss scaling)实现的,`mixed_precision` 参数只是把"选择哪种精度策略+要不要loss scaling"这套决策封装成了一个声明式配置,不需要用户在训练循环里手写 `with torch.autocast(...):`。

**常见坑:**
1. Jupyter Notebook 场景下反复修改 `Accelerator(...)` 配置重新跑 cell 是这个单例陷阱最容易发作的地方——报错信息虽然明确写了"restart your runtime",但第一次遇到容易摸不着头脑,以为是自己参数传错了。
2. 库/框架代码里如果在多个不同地方各自 `Accelerator()` 实例化(比如一个工具函数内部又建了一个),即使不传参数,也要清楚拿到的是同一份全局状态,不是各自独立的配置。

---

## 3. 梯度累加正确写法(`accelerator.accumulate`)

**签名/是什么:**
```
accelerator = Accelerator(gradient_accumulation_steps=4)
with accelerator.accumulate(model):
    loss = model(batch).loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```
`accelerator.accumulate(model)` 这个上下文管理器,让"每 N 个 micro-batch 才真正更新一次参数"这套逻辑不需要手写计数器/条件判断。

**一句话:** **`with accelerator.accumulate(model):` 内部按 `gradient_accumulation_steps` 自动决定"这一步是不是该真正 `sync` 梯度并调用 `optimizer.step()`"**,即使代码表面上每个 micro-batch 都写了 `optimizer.step()`,实际效果是每 N 步才有一次真正生效的参数更新。

**底层机制/为什么这样设计:** 手写梯度累加通常需要一个额外的计数器变量,判断"当前是不是累加周期的最后一步",容易和分布式场景下的梯度同步时机搅在一起出错(分布式训练里,每个 micro-batch 之间如果都做一次昂贵的 all-reduce 梯度同步,会严重浪费通信带宽——理想情况是只在真正要 `step()` 的那一刻才同步)。`accelerator.accumulate()` 把这套"什么时候该同步、什么时候不该"的判断封装成一个上下文管理器,内部通过 `accelerator.sync_gradients` 这个属性告诉你"这一步是不是真正生效的那一步",配合分布式场景下自动跳过非同步步骤的梯度 all-reduce,一举解决了正确性和效率两个问题。

**AI 研究/工程场景:** 09 类如果要在显存有限的情况下模拟更大的有效 batch size(比如显存只够放 batch=2,但想要 batch=8 的训练效果),梯度累加(`gradient_accumulation_steps=4`)是标准手段——`accelerator.accumulate` 让这个技巧的实现代码几乎和"不做累加"的代码长得一样,不用大改训练循环结构。

**可运行例子:**
```python
import torch
from torch import nn
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=4)
model = nn.Linear(5, 5).to(accelerator.device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model, optimizer = accelerator.prepare(model, optimizer)

sync_count = 0
for i in range(4):  # 4个micro-batch,累加步数也是4 -> 应该只有最后一步真正sync
    with accelerator.accumulate(model):
        out = model(torch.randn(2, 5).to(accelerator.device))
        loss = out.sum()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            sync_count += 1   # 只有这一步的optimizer.step()才是"真正生效"的
        optimizer.step()
        optimizer.zero_grad()

assert sync_count == 1  # 4个micro-batch,累加4步,只有第4步是真正的同步/生效步

print(f"OK: gradient_accumulation_steps=4,4个micro-batch里sync_gradients只在第{4}步为True,累加逻辑正确")
```
本机实测:4 个 micro-batch、`gradient_accumulation_steps=4` 的设置下,`accelerator.sync_gradients` 恰好只在第 4 步(最后一步)为 `True`,精确验证了累加周期的边界判断逻辑。

**面试怎么问 + 追问链:** "梯度累加对 BatchNorm 这类依赖 batch 统计量的层有什么影响?" → 这是一个容易被忽视但很实际的追问:BatchNorm 的均值/方差是按**每次前向调用**的 micro-batch 计算的,不会像梯度那样被"累加"——也就是说梯度累加能模拟更大 batch 的**梯度**效果,但不能模拟更大 batch 对 BatchNorm 统计量的效果,这是梯度累加这个技巧的一个已知局限,LayerNorm(不依赖 batch 统计量)不受这个问题影响。

**常见坑:**
1. 使用梯度累加时,损失函数如果用的是"求和"而不是"求平均"来聚合,需要额外除以累加步数,否则等效学习率会被放大——`accelerator.accumulate` 不会自动帮你处理这个数值细节,需要自己确认 loss 的归一化方式。
2. 累加周期内间隔调用 `accelerator.backward(loss)` 会持续累积梯度,不要在还没到同步步骤时手滑调用了 `optimizer.zero_grad()` 清空梯度,那会破坏累加语义。

---

## 4. `accelerate config` / `accelerate launch` 机制

**签名/是什么:**
```
accelerate config     # 交互式生成一份硬件/精度配置文件(~/.cache/huggingface/accelerate/default_config.yaml)
accelerate launch train.py   # 按配置文件启动训练脚本(自动设置好分布式相关的环境变量)
```
命令行工具,负责"把训练脚本按指定的硬件配置正确启动起来"这件事,不需要用户自己手写 `torchrun`/环境变量设置。

**一句话:** `accelerate launch` 本质是一个更友好的 `torch.distributed.run`(即 `torchrun`)包装,它读取 `accelerate config` 生成的配置文件,自动拼好该用哪些环境变量(`RANK`/`WORLD_SIZE`/`MASTER_ADDR` 等)去启动你的训练脚本。

**底层机制/为什么这样设计:** PyTorch 原生的分布式启动需要手动管理一堆环境变量和进程编号逻辑,对不熟悉分布式训练细节的用户很不友好。`accelerate config` 把"这台机器/这个集群该怎么配置"这个一次性的决策,固化成一份可复用的 YAML 文件;`accelerate launch` 每次运行时读这份配置,自动完成环境变量设置——用户的训练脚本本身完全不需要感知"我现在是单卡还是多卡、是第几个进程"这些细节(这些信息通过 `Accelerator()` 对象内部自动读取环境变量获得)。

**AI 研究/工程场景:** 团队内部把训练脚本从一台单卡开发机搬到多卡训练集群时,理想情况下只需要在集群上重新跑一次 `accelerate config`(选择多卡配置),训练脚本代码本身一行都不用改——这是 `accelerate` 整套设计"训练代码与硬件配置解耦"理念的具体体现。

**可运行例子:**
```python
import shutil
import subprocess

# accelerate CLI 是venv安装时生成的可执行文件,不一定在当前shell的PATH上
# (取决于venv是否被"激活"),但它确实存在于venv的Scripts/bin目录
accelerate_on_path = shutil.which("accelerate")

import sys, pathlib
venv_scripts_dir = pathlib.Path(sys.executable).parent  # python.exe所在目录,即Scripts/
accelerate_exe = venv_scripts_dir / "accelerate.exe"
assert accelerate_exe.exists()  # 确认CLI工具确实被正确安装了,只是没在PATH上

# 用完整路径调用,验证CLI本身可执行(--help不会有副作用,不需要真的launch什么)
result = subprocess.run([str(accelerate_exe), "launch", "--help"],
                         capture_output=True, text=True, timeout=30)
assert result.returncode == 0
assert "--num_processes" in result.stdout  # 确认输出的是真实的launch帮助信息

print(f"OK: accelerate CLI在PATH上={accelerate_on_path is not None},但确实存在于{accelerate_exe},可正常调用")
```
本机实测:`shutil.which("accelerate")` 返回 `None`(不在当前 shell 的 PATH 上,因为本系列用的是 venv 内 Python 解释器的完整路径调用方式,没有"激活"venv);但 `accelerate.exe` 确实存在于 `.venv/Scripts/` 目录,用完整路径调用 `accelerate launch --help` 能正常返回帮助信息。**额外真实发现**:`accelerate env`(官方文档常用来"打印当前环境诊断信息"的命令)在本机会直接崩溃,报 `UnicodeDecodeError`——根因是它内部用 `subprocess.check_output` 探测 `bash` 命令位置,解码那个子进程输出时没有处理好非 UTF-8 locale 的情况,这是一个和 00-roadmap.md 记录的 trl 编码问题同一类别、但发生在不同库里的真实 bug,13 类会收录这个案例。

**面试怎么问 + 追问链:** "`accelerate launch` 和直接 `python train.py` 有什么本质区别?" → `python train.py` 只启动**一个**进程;`accelerate launch`(多卡配置下)会启动**多个**进程(每张卡一个),并给每个进程设置好区分身份的环境变量,训练脚本内部的 `Accelerator()` 靠读取这些环境变量知道"我是第几个进程、总共几个进程"——单卡配置下两者启动的进程数量是一样的,区别不明显,这也是为什么本系列(单卡环境)大多数知识点直接用 `python` 运行脚本也没问题。

**常见坑:**
1. 忘记"激活"虚拟环境(或者用完整路径调用可执行文件)时,直接敲 `accelerate` 命令会得到"命令未找到",容易误判成"没装好这个包"——先确认包本身确实装了(`pip show accelerate`),再检查是不是 PATH 的问题。
2. `accelerate config` 生成的配置文件是全局的(默认路径在用户目录下),不同项目如果需要不同的硬件配置,应该用 `accelerate launch --config_file <路径>` 显式指定项目自己的配置文件,不要依赖可能被其他项目修改过的全局默认配置。

---

## 5. `save_state`/`load_state` checkpoint 机制

**签名/是什么:**
```
accelerator.save_state("checkpoint_dir")   # 保存模型+优化器+随机数状态等完整训练现场
accelerator.load_state("checkpoint_dir")   # 恢复到保存时的完整状态
```
和 02 类"权重缓存"讨论的"只存模型权重"不同,`save_state`/`load_state` 保存的是**完整的训练现场**,目标是"能从任意中断点无缝续训",不只是"能重新加载模型做推理"。

**一句话:** `save_state` 会把 `prepare()` 过的所有对象(模型、优化器)以及各种随机数生成器状态一起打包保存,`load_state` 恢复时,连"接下来生成的随机数序列"都能和没有中断过完全一致——这是"能推理"和"能无缝续训"两种不同保存粒度的关键区别。

**底层机制/为什么这样设计:** 训练中断后重新开始,如果只恢复了模型权重和优化器状态,但随机数生成器(数据洗牌顺序、dropout 的随机 mask 等)是从新的随机种子重新开始的,后续训练轨迹会和"假设没有中断过"产生的轨迹出现细微但真实的偏差——对追求严格可复现性的研究场景,这种偏差可能是不可接受的。`save_state` 把 PyTorch/CUDA 的随机数生成器状态也序列化保存,是为了让"中断再续训"和"从未中断过"在统计意义上等价。

**AI 研究/工程场景:** 长时间训练任务(尤其是可能被抢占/需要排队等资源的集群环境)必须支持"从最近一个 checkpoint 无缝续训",`save_state`/`load_state` 是这套机制的标准实现;和 05 类会讲的 `Trainer` 的 `resume_from_checkpoint` 本质上解决的是同一个问题,`Trainer` 内部很大程度上也是基于 `accelerate` 这套机制实现的。

**可运行例子:**
```python
import torch
import os
import tempfile
from torch import nn
from accelerate import Accelerator

accelerator = Accelerator()
model = nn.Linear(3, 3).to(accelerator.device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
model, optimizer = accelerator.prepare(model, optimizer)

original_weight = model.weight.clone()

tmpdir = tempfile.mkdtemp()
accelerator.save_state(tmpdir)
saved_files = set(os.listdir(tmpdir))
assert "model.safetensors" in saved_files       # 呼应02类:模型权重依然是safetensors格式
assert "optimizer.bin" in saved_files            # 优化器状态(动量等)单独保存
assert any("random_states" in f for f in saved_files)  # 随机数生成器状态,推理场景的save_pretrained不会有这个

# 人为破坏当前状态,模拟"训练中断,权重被搞乱了"
with torch.no_grad():
    model.weight.add_(1.0)
mutated_weight = model.weight.clone()
assert not torch.equal(original_weight, mutated_weight)

# 从checkpoint恢复,验证权重精确恢复到保存时的状态
accelerator.load_state(tmpdir)
restored_weight = model.weight.clone()
assert torch.equal(original_weight, restored_weight)

print(f"OK: save_state产出{saved_files},load_state后权重精确恢复(mutate后又还原)")
```
本机实测:`save_state` 产出 `model.safetensors`(模型权重)、`optimizer.bin`(优化器状态)、`random_states_0.pkl`(随机数生成器状态)三类文件;人为修改权重后 `load_state` 能精确恢复到保存时刻的数值,逐 bit 相等。

**面试怎么问 + 追问链:** "`save_state` 和知识点2类讲的 `save_pretrained` 有什么本质区别,分别该在什么场景用?" → `save_pretrained` 面向"分享/部署这个模型"(只要权重,干净、跨框架兼容性好,09类"微调产物保存与复现"会用这个);`save_state` 面向"这个具体训练任务的中断续训",保存的东西更全但也更"重"、和当前这次训练的具体配置(优化器类型等)强绑定,不适合用来分享模型给别人推理用。

**常见坑:**
1. `save_state` 产出的 checkpoint 目录体积比单纯的模型权重大不少(优化器状态对于 Adam 这类带动量的优化器,体积可能接近模型权重的 2-3 倍),长期训练任务如果不清理旧 checkpoint,磁盘占用会快速增长。
2. `load_state` 恢复的优化器状态和当前代码里优化器的**类型/超参配置**必须匹配(不能用 SGD 训练保存的 checkpoint 去恢复一个 Adam 优化器),这是和"只加载模型权重"相比更脆弱的地方,升级训练脚本时如果改了优化器类型,旧 checkpoint 的 `load_state` 可能失败或行为异常。

---

## 6. 单机多进程模拟分布式语义

**签名/是什么:**
```
import torch.distributed as dist
dist.is_gloo_available()   # True(CPU/GPU都能用,Windows可用)
dist.is_nccl_available()   # False(Windows原生不支持,GPU间高速通信后端)
```
本机(Windows 原生、单张 GPU)没有条件做真正的多卡分布式训练验证,本知识点如实说明能验证到什么程度,以及沿用哪一套已有的诚实验证框架。

**一句话:** **这台机器物理上只有一张 GPU,不管在 Windows 还是 Linux 上都测不出"多卡分布式"的真实效果**;能确认的是 Windows 原生环境下 `gloo` 后端可用、`nccl` 后端不可用,这个限制本身和"只有一张卡"是两个独立但都成立的事实。

**底层机制/为什么这样设计:** `torch.distributed` 支持多种通信后端,`nccl` 是 NVIDIA 专为多 GPU 间高速通信设计的后端(利用 NVLink/PCIe 等硬件特性),但**只支持 Linux**;`gloo` 是通用性更强、性能不如 nccl 但跨平台(包括 Windows)的后端。这意味着即使这台机器插了两张卡,想在 Windows 原生环境用 `accelerate`/`torch.distributed` 做真正的多卡训练,也只能退化到 `gloo` 后端,性能会明显低于 Linux+nccl 的组合——这是选择训练环境时一个真实需要考虑的因素,不是这个系列独有的限制。

**AI 研究/工程场景:** [`torch-deep-dive/09-distributed-training-basics.md`](../torch-deep-dive/09-distributed-training-basics.md) 已经建立并验证过一套"单机多**进程**(不是多**卡**)模拟 `torch.distributed` 通信语义"的方法——用 `gloo` 后端在同一台机器上跑多个 CPU 进程,验证 `all-reduce`/进程间通信的正确性,不需要真实多卡。本系列如果未来需要验证 `accelerate` 的分布式数据加载等机制,应该直接复用那一套方法论,不重新发明;凡是"必须物理多卡才能观测"的现象(比如 nccl 通信性能、真实的多卡显存分摊效果),明确标注"官方文档口径,未在本环境验证",不冒充实测过。

**可运行例子:**
```python
import torch.distributed as dist
import torch

assert dist.is_gloo_available() is True    # Windows可用,CPU/GPU通用
assert dist.is_nccl_available() is False   # Windows原生不支持,这是真实的平台限制

assert torch.cuda.device_count() == 1      # 如实confirm:这台机器只有一张卡,
                                             # 后续任何"多卡"相关内容都只能是机制性介绍,不是本机实测

print(f"OK: gloo可用(Windows兼容)、nccl不可用(Windows原生限制)、本机GPU数量={torch.cuda.device_count()}(单卡,如实标注多卡内容的验证边界)")
```
本机实测:`dist.is_gloo_available()` 为 `True`,`dist.is_nccl_available()` 为 `False`,`torch.cuda.device_count()` 确认为 `1`——三个事实共同确定了本系列关于"分布式训练"能验证到什么程度的边界,不做超出这个边界的验证声明。

**面试怎么问 + 追问链:** "在 Windows 上做多卡训练,除了 nccl 不可用,还有什么现实的限制?" → 追问"如果团队既有 Windows 开发机又有 Linux 训练集群,应该怎么组织工作流?"(常见做法是 Windows 机器用于代码开发/调试/小规模验证——本系列的 09 类"微调实战对比"就是这个定位,真正的大规模/多卡训练任务提交到 Linux 集群执行,`accelerate config` 的价值正在于让同一份训练代码在两种环境之间切换时几乎不需要改动)。

**常见坑:**
1. 不要仅凭"代码在单卡上跑通了"就假设"多卡场景下逻辑一定正确"——分布式场景引入的问题(比如某些操作忘记做跨进程同步、只在 rank 0 该执行的操作被所有进程重复执行)在单卡环境下完全不会暴露,这是这类"部分验证、部分标注未验证"内容的真实局限,不能靠单卡测试掩盖。
2. `WSL2` 虽然是 Linux 内核,但 GPU 直通场景下多卡支持情况和物理 Linux 机器可能仍有差异(取决于具体的虚拟化/驱动配置),不能想当然地认为"用了WSL2就等价于真Linux多卡环境",这类结论如果要下,应该额外验证,不能靠推断。

---

*本篇 6 个知识点全部在仓库根目录 `.venv` 真实验证通过(每个知识点独立进程验证,避免`AcceleratorState`单例状态互相干扰)。知识点6如实标注"单卡环境"这一验证边界,不冒充真实多卡场景已验证。*
