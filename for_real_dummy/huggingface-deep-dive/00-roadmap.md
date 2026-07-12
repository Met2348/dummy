# HuggingFace 生态深挖 —— 路线图与进度表

> 目标:100 个 HuggingFace 生态知识点,由浅入深,分批次完成,深度对标 [torch-deep-dive/](../torch-deep-dive/00-roadmap.md) 和 [tensorflow-deep-dive/](../tensorflow-deep-dive/00-roadmap.md)(面试二三四面深度,不是"这个函数怎么调")。
> 定位:这是仓库"深挖系列"里第六条(numpy/python-advanced/python-idioms/torch/tensorflow 之后;`rhcsa-bash-deep-dive` 是独立技能系列,不计入"深挖系列"总数,见 README.md 的两张表格分工)。前五条深挖系列合计约 366 个知识点(不含本系列),本系列实际完成 101 个(比 100 的目标多 1 个,详见下方进度表汇总说明),完成后深挖系列总量约 467 个。torch/tensorflow 讲的是深度学习框架本身的底层机制,本系列讲的是**建在框架之上的 HuggingFace 生态**——transformers/tokenizers/datasets/accelerate/peft/bitsandbytes/trl/huggingface_hub 这些库的**工程内核**,不是算法论文。

---

## 与仓库里其余相关系列的关系(差异化声明,必须先读)

本仓库 `learning/` 目录下已经有多条**论文/算法精读**风格的系列涉及相近关键词,本系列和它们的分工如下,写作时在对应类目开头会重复这句话,这里先统一交代:

- **[`learning/lora-family/`](../../learning/lora-family/)**(LoRA/AdaLoRA/PiSSA/VeRA/LoHa-LoKr/QLoRA/LoFTQ/DoRA)、**[`learning/adapter-tuning-family/`](../../learning/adapter-tuning-family/)**(Houlsby/Pfeiffer/AdapterFusion/IA3/MAM/...)、**[`learning/dpo-family/`](../../learning/dpo-family/)**(DPO/ORPO/SimPO/RainbowPO):讲的是这些方法的**论文/算法本身**——"为什么这个数学设计有效"。本系列 07(PEFT库工程机制)、08(量化机制)、10(TRL训练器抽象)三类讲的是"**这些算法被 HuggingFace 生态封装成库之后,Python API 怎么设计、内部怎么实现、真实调用会发生什么**"——工程内核,不重复算法数学。类比 torch-deep-dive 对 PyTorch 的处理方式:不重复"什么是反向传播",只讲 autograd 库内部怎么实现。
- **[`learning/quantization-deploy/`](../../learning/quantization-deploy/)**(INT8/GPTQ/AWQ/SmoothQuant/LLM.int8()/FP8 等量化算法数学):本系列 08 类讲 `BitsAndBytesConfig` 参数设计、怎么接入 `from_pretrained`、真实显存/速度测量、`prepare_model_for_kbit_training` 做了什么——不重复量化数学。
- **[`learning/inference-engine-core/`](../../learning/inference-engine-core/)**(从零造推理引擎:PagedAttention/CUDA kernel/continuous batching/mini-vLLM):系统工程视角,造推理引擎本身。本系列 12 类讲"怎么正确调用 `transformers.generate()`"——完全不是一个海拔。
- **[`learning/agent-framework-stack/`](../../learning/agent-framework-stack/)**、**[`learning/rag-essential/`](../../learning/rag-essential/)**:LangChain/LangGraph/LlamaIndex/RAG 已经系统覆盖,本系列不涉及。
- **[torch-deep-dive/](../torch-deep-dive/00-roadmap.md)**:本系列大量复用其已建立的诚实验证框架,尤其 06 类(Accelerate)直接复用 [`torch-deep-dive/09-distributed-training-basics.md`](../torch-deep-dive/09-distributed-training-basics.md) 已验证过的"单机多进程模拟分布式语义,物理多卡才能观测的现象明确标注引用官方文档未实测"这套方法论。

---

## 环境声明(先读,决定了本系列所有例子的真实行为)

**运行环境:** 仓库根目录 `e:\Workspace\dummy\.venv`(**Windows 原生**,Python 3.13.9)。**不需要 WSL2**——这点和 tensorflow-deep-dive 不同:TensorFlow 2.11 起 Windows 原生不支持 GPU 才需要 WSL2,但 PyTorch **原生支持 Windows CUDA**,已实测确认 `torch.cuda.is_available()` 为 `True`。

**已装版本(规划阶段逐个实测核对,不是查文档假设的):**
- torch **2.11.0+cu128**、transformers **5.10.2**、huggingface_hub **1.18.0**、tokenizers **0.22.2**、datasets **5.0.0**、accelerate **1.13.0**、peft **0.19.1**、bitsandbytes **0.49.2**、trl **1.5.1**、sentence-transformers **5.5.1**
- GPU:**NVIDIA GeForce RTX 3080 Ti Laptop GPU**,显存 **17.2GB**(`torch.cuda.get_device_properties(0).total_memory` 实测值,和显卡宣传的"16GB"有正常的进制换算差异)

**⚠️ 强制前置步骤(阻塞性,涉及 trl 的知识点必须遵守):**
整个验证 session 的 shell 必须在**启动 Python 解释器之前**设置环境变量 `PYTHONUTF8=1`(进程内 `os.environ["PYTHONUTF8"]="1"` 之后再 `import` 对已初始化的解释器无效,必须是解释器启动前的 shell 环境变量)。

原因:trl 1.5.1 的 `chat_template_utils.py` 读取内置 `.jinja` 聊天模板文件时没有显式传 `encoding="utf-8"`,而 `pathlib.Path.read_text()` 不传 encoding 时默认走系统 locale 编码——中文 Windows 默认是 `cp936`(GBK),这个 `.jinja` 文件本身是 UTF-8 存的,GBK 解码失败,`from trl import SFTTrainer` **直接抛 `UnicodeDecodeError` 崩溃**。这是任何第一次跟着教程敲代码的中国学生几乎必然会撞到的坑,不是边缘情况——已现场复现,设置 `PYTHONUTF8=1` 后验证修复有效。`DPOTrainer`/`GRPOTrainer` 同样受影响(触发同一个模块)。10 类开头会再次显著标注。

**⚠️ 重要机制说明(Windows WDDM 显存回落,全系列显存测量的前提认知):**
Windows 消费级 GPU 默认 **WDDM** 驱动模式下有一个叫 **CUDA Sysmem Fallback Policy** 的机制(NVIDIA 控制面板可调,默认启用):显存分配超过物理上限时,驱动会**静默地**把多出来的部分换页到系统内存,**不会抛出 `torch.cuda.OutOfMemoryError`,脚本会正常跑完,但性能可能断崖式下降**。这在 Linux/WSL2(通常跑 TCC 模式或压根没有 WDDM 这层)上不存在,会正常报错。

已现场复现:TinyLlama-1.1B 用"fp32 主权重 + `autocast(bf16)` + fp32 Adam 状态"这个教科书式混合精度配方做全参微调时,PyTorch 自报 `reserved=23.76GB`(超过本机 17.2GB 物理显存),**但脚本没有报错、正常跑完**;同一时刻用 `nvidia-smi` 查询,`memory.used` 显示的数字和 PyTorch 自报的对不上。**结论:本系列所有显存测量必须交叉核对 `torch.cuda.max_memory_allocated()`/`max_memory_reserved()` 和 `nvidia-smi`(或任务管理器"专用GPU内存"曲线),不能只信任一个数字**——09 类的显存对比实验尤其要遵守这一条,全参微调基线采用"整个模型强制转 bf16、不留 fp32 主权重"的配方(已实测峰值 11.13GB,舒服放进显存)。这个机制本身在 13 类单独成一条知识点。

**其余环境要点:**
- bitsandbytes 0.49.2 起 **Windows 是官方一等支持**(预编译 CUDA 库直接打进 PyPI wheel),不需要理会网上早年"Windows 装不了 bitsandbytes"的过时经验,已实测 NF4 4bit 加载 + `generate()` 正常。
- **DeepSpeed 未安装**,Windows 支持历史上不稳定(算子需要 JIT 编译,MSVC 工具链上历来问题较多),本系列**不做 DeepSpeed 动手实操**;如提及 ZeRO-offload 等仅作机制性介绍并标注"官方文档口径,未在本环境验证"。
- 原生 Windows `torch.distributed` **不支持 NCCL 后端**,只能用 `gloo`;这台机器物理上也只有一张卡——06 类的分布式验证复用 torch-deep-dive/09 已经验证过的"单机多进程模拟"诚实框架,不重新发明,凡是"必须物理多卡才能观测"的现象明确标注引用官方文档未实测。
- **⚠️ Windows 多进程 spawn 陷阱(04 类撰写时现场触发,曾导致进程失控,必须显著记录)**:`datasets.map(num_proc=N)`(**哪怕 `N=1`**)内部无条件走 `multiprocessing.Pool`;Windows 没有 `fork()`,用的是 `spawn` 方式创建子进程——**子进程会重新 import/执行整个主脚本**。如果调用 `.map(num_proc=...)` 的代码不在 `if __name__ == "__main__":` 保护块里,子进程重新执行主脚本时会再次跑到同一个 `.map(num_proc=...)` 调用,又 spawn 新的子进程,如此递归,**现场复现为进程数量失控增长**,已立即终止并确认无残留进程。**本系列凡是用到 `datasets.map(num_proc=...)`(或任何触发多进程的 API)的验证脚本,一律把主逻辑放进 `if __name__ == "__main__":` 函数里,并调用 `multiprocessing.freeze_support()`**,这不是可选的最佳实践,是 Windows 下的强制要求,Linux/WSL2 用 `fork()` 不会有这个问题(这也是为什么很多教程从没提过这个坑)。
- 全系列示例代码统一用 `dtype=` 而非已废弃的 `torch_dtype=`(transformers 5.10.2 实测对旧参数名触发 `FutureWarning`);任何"内省属性"类代码撰写时必须对着 5.10.2 实测一遍,不能凭旧版本教程记忆写(例如网上常见的 `model.hf_device_map` 属性在当前版本**不存在**,已实测确认)。

---

## 模型 / 数据集选型(09 类微调实战对比的确定性选择,已实测验证,后续撰写不得随意更换)

**主选模型:`TinyLlama/TinyLlama-1.1B-Chat-v1.0`**
1.10B 参数(`sum(p.numel())` 实测 1,100,048,384),`LlamaForCausalLM` 架构,GQA(32 attention heads / 4 KV heads),hidden_size=2048,22 层,vocab=32000。Apache 2.0,非 gated,`from_pretrained` 无需 token。已缓存在本机(`C:\Users\ericp\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0`),且仓库内 [`learning/lora-family/src/qlora_peft.py`](../../learning/lora-family/src/qlora_peft.py) 已经用它做 QLoRA 冒烟测试(`target_modules=["q_proj","v_proj"]`,与 PEFT 官方 LLaMA 族映射表 `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['llama']` 实测一致)——选它不是凭空推荐,是仓库内已有的真实精度锚点。

已实测三组显存对照(batch=2, seq_len=512):

| 配置 | 峰值显存(allocated) | 峰值(reserved) |
|---|---|---|
| 全参微调,纯 bf16 | 11.13 GB | 12.30 GB |
| 全参微调,fp32主权重+autocast(bf16)(经典配方) | 22.13 GB | 24.67 GB(**超过17.2GB物理显存,WDDM回落**) |
| LoRA r=8(q_proj/v_proj),bf16 | 6.26 GB | 6.68 GB(可训练参数占比0.10%) |

**备用/降级模型:`Qwen/Qwen2.5-0.5B-Instruct`**——[`learning/dpo-family/lectures/01-dpo.md`](../../learning/dpo-family/lectures/01-dpo.md) 已有真实训练闭环 precedent。仅当 TinyLlama 全参微调在实际撰写中出现难以控制的问题时启用,须在对应知识点如实标注降级原因。

**主选数据集:`timdettmers/openassistant-guanaco`**
train 9,846 条 / test 518 条,单一 `text` 字段已是 `### Human: ... ### Assistant: ...` 格式,`SFTTrainer(dataset_text_field="text")` 零预处理直接吃。Apache 2.0,是 **QLoRA 论文自己的演示数据集**。多组对比实验用子采样 1,000～2,000 条保证快速迭代;"横向对比分析"收尾知识点在完整 9,846 条上跑一次作为压轴。已实测 `SFTTrainer+LoraConfig+TinyLlama` 5 步真实 loss 从 1.285 → 1.773 波动下降,峰值显存 2.545GB,端到端验证过可跑通。

**补充数据集(仅用于对应单点演示,不用于 09 类主实验):**
- `HuggingFaceH4/no_robots`:10K,`messages` 结构化字段,适合演示 `apply_chat_template()`。**协议是 CC-BY-NC-4.0(非商用)**,个人学习笔记场景使用没有问题,但必须原样标注协议,不能写错。
- `databricks/databricks-dolly-15k`:15,011 条,CC-BY-SA-3.0(宽松,人工撰写非LLM生成),裸 `instruction/context/response` 字段需要自己写 formatting function,用于演示自定义数据格式转换(guanaco 已经是现成 text 字段,反而没有格式转换的教学空间)。

---

## 知识点结构模板(七步,与 torch-deep-dive/tensorflow-deep-dive 完全一致)

1. **签名/是什么**——API 或概念定义,人话翻译
2. **一句话**——是什么
3. **底层机制/为什么这样设计**——不停在"怎么用",讲到"为什么必须是这样"、内部实际发生了什么
4. **AI 研究/工程场景**——具体在研究/工程代码里怎么用
5. **可运行例子**——带 `assert` 验证,能内省的地方现场打印内部状态,真实在仓库 `.venv` 里跑过
6. **面试怎么问 + 追问链**——面试官大概率怎么问,追问会往哪个方向深挖
7. **常见坑**

---

## 进度表(撰写顺序按风险递增,不是文件编号顺序;明细顺序见下方"撰写顺序"说明)

| # | 分类 | 文件 | 知识点数 | 状态 |
|---|------|------|---------|------|
| 01 | Tokenizer 核心机制 | [01-tokenizer-internals.md](01-tokenizer-internals.md) | 9 | ✅ 已完成(已验证,9/9代码块独立通过;现场发现2处真实差异:`use_fast=False`在当前版本对Llama/BERT已不产生真正slow实现、`apply_chat_template`默认返回`BatchEncoding`而非裸list) |
| 02 | Model 加载与 AutoClass 机制 | [02-model-loading-and-autoclass.md](02-model-loading-and-autoclass.md) | 9 | ✅ 已完成(已验证,9/9代码块独立通过;现场发现:`model.hf_device_map`当前版本不存在、`torch_dtype`弃用提示走transformers自己的logger而非Python warnings、默认`attn_implementation`是sdpa不依赖flash-attn) |
| 03 | Pipeline 高层API内核 | [03-pipeline-internals.md](03-pipeline-internals.md) | 6 | ✅ 已完成(已验证,6/6代码块独立通过;现场发现:pipeline的postprocess对BPE tokenizer做clean_up_tokenization_spaces清理,输出字符串和手写decode()不逐字节相等) |
| 04 | Datasets 库机制 | [04-datasets-mechanics.md](04-datasets-mechanics.md) | 8 | ✅ 已完成(已验证,8/8代码块独立通过;现场触发并修复Windows多进程spawn事故——`.map(num_proc=N)`哪怕N=1也无条件走`mp.Pool`,无`__main__`保护会递归spawn进程,已终止确认无残留并写入环境声明) |
| 05 | Trainer API 内核 | [05-trainer-api-internals.md](05-trainer-api-internals.md) | 9 | ✅ 已完成(已验证,9/9代码块独立通过;现场发现`trainer.accelerator.device`和`trainer.args.device`是不同的torch.device对象(带不带index),compute_metrics返回值自动加`eval_`前缀,checkpoint命名规则`checkpoint-{step}`) |
| 06 | Accelerate 分布式与设备机制 | [06-accelerate-and-devices.md](06-accelerate-and-devices.md) | 6 | ✅ 已完成(已验证,6/6代码块独立通过;现场发现`AcceleratorState`是进程级单例、`accelerate env`命令因bash探测的subprocess编码问题在本机崩溃、accelerate CLI存在于venv但默认不在PATH上) |
| 07 | PEFT 库工程机制 | [07-peft-library-internals.md](07-peft-library-internals.md) | 8 | ✅ 已完成(已验证,8/8代码块独立通过;现场发现peft的Linear包装类和torch.nn.Linear同名不同类、target_modules内部规范化成set、merge_and_unload前后logits数值一致) |
| 08 | 量化机制 bitsandbytes | [08-quantization-bitsandbytes.md](08-quantization-bitsandbytes.md) | 6 | ✅ 已完成(已验证,6/6代码块独立通过;现场发现`load_in_8bit=True`不能再直接传from_pretrained必须包BitsAndBytesConfig、4bit推理反而比bf16慢36%、量化模型`.to("cpu")`不报错但之后forward才报错的延迟陷阱) |
| 09 | 微调实战对比★重中之重 | [09-finetuning-comparison-lab.md](09-finetuning-comparison-lab.md) | 12 | ✅ 已完成(已验证,12/12代码块独立通过;真实跑通全参/LoRA/QLoRA三组200步训练对比(800条真实数据),核心发现QLoRA显存最省(1.66GB)但耗时反而最长(全参的2.9倍);规模从计划的完整9846条如实调整为800条/200步(计时探测后57分钟不现实);超参扫描/数据规模/失败模式等补充实验详见[09-lab-artifacts/](09-lab-artifacts/)) |
| 10 | TRL 训练器抽象 | [10-trl-trainers.md](10-trl-trainers.md) | 6 | ✅ 已完成(已验证,6/6代码块独立通过;现场发现`no_robots`数据集因prompt+messages列并存触发`SFTTrainer`格式检测歧义报`KeyError('completion')`、`SFTConfig`把继承的learning_rate默认值从5e-5改成2e-5、packing=True在非flash-attention实现下有真实的样本串扰风险警告) |
| 11 | Hub 与模型分享机制 | [11-hub-and-sharing.md](11-hub-and-sharing.md) | 6 | ✅ 已完成(已验证,6/6代码块独立通过;现场发现gated仓库"元数据可查、文件下载受限"两层访问控制的真实边界,license字段与00环境声明交叉验证一致;push_to_hub/gated下载受限于本机无写权限token,如实标注) |
| 12 | 推理优化基础 | [12-inference-optimization.md](12-inference-optimization.md) | 7 | ✅ 已完成(已验证,7/7代码块独立通过;现场发现`TextIteratorStreamer`默认`skip_prompt=False`会回显prompt、`skip_prompt=True`后新内容开头丢失边界空格(与03类pipeline同源的SentencePiece边界问题)、torch.compile对generate()动态shape场景无明显加速) |
| 13 | 调试与常见报错精解 | [13-debugging-and-common-errors.md](13-debugging-and-common-errors.md) | 9 | ✅ 已完成(已验证,9/9代码块独立通过;全系列最重要发现——越界token id触发CUDA设备端断言后,整个进程CUDA上下文被永久污染,后续任何无关操作都会失败,唯一恢复方式是重启进程,与可安全catch的标准OOM形成鲜明对比) |
| 14 | 进阶深度追加:5 个多级追问链案例 | [14-advanced-interview-depth.md](14-advanced-interview-depth.md) | 5案例(不计入101) | ✅ 已完成(已验证,7/7代码块独立通过;基于真实WebSearch调研的5条追问轴线撰写——全参→LoRA→QLoRA显存/耗时方案批判迭代(真实复算Adam优化器状态开销977倍差距+4bit压缩精确4倍,交叉验证09类真实训练数字QLoRA耗时是全参2.9倍)、Windows WDDM显存静默failure日志诊断(全新题型,建立在13类+00环境声明记录的真实事故,现场发现00-roadmap.md正文23.76GB与表格24.67GB数字不一致,统一采用与13类代码一致的24.67)、混合精度/梯度累加规模递增(真实测得accum=256时有效batch放大256倍但显存仅多6.9%,直接堆batch则暴涨3.46倍;bf16/fp32显存比例在5个数量级规模跨度里精确稳定等于2.0)、单卡模拟分布式的决策依据追问(3进程gloo all_reduce确定性验证,如实标注未测nccl的边界)、参数高效微调效率真实性验证追问(用09类真实JSON数据证明"效率"一词在LoRA/QLoRA上指向相反结论)) |

**合计:101 个知识点,13 篇 + 1 篇进阶深度追加(5 个案例,不计入 101),101/101 完成(13/13 篇)。🎉 全部完成。**(自查发现实际完成101个,比计划的100个多1个——各分类页面标题数逐个用`grep`核对过,101这个数字是真实文件内容,不是笔误;不因为101不是整数就将错就错标成100,如实更正)

**撰写顺序(按 root/GPU 依赖与耗时递增,不是文件编号顺序):**
1. 01(tokenizer)→ 02(model加载)→ 03(pipeline)→ 04(datasets)——纯 CPU/轻量 GPU,机制类,先跑顺流程。
2. 11(hub)→ 06(accelerate)——涉及网络/进程但不涉及真实长训练。
3. 05(Trainer)→ 07(peft)→ 08(量化)→ 12(推理优化)——单次前向/短训练验证,复杂度递增。
4. 10(TRL,**注意 `PYTHONUTF8=1` 前置**)→ 09(微调实战对比,全系列最耗时,多组真实训练闭环)——放最后。
5. 13(调试与常见报错精解)——收尾,汇总撰写全程真实撞到的坑(而不是凭空列常见错误)。

最终交付文件仍按 01→13 编号排列。

---

## 每一批具体覆盖哪些知识点(明细)

### 01 Tokenizer 核心机制
`AutoTokenizer.from_pretrained`内部机制(vocab/merges下载、tokenizer_config.json解析、类型分发) BPE分词算法实操演示(真实看tokenizer怎么切,不是背算法) fast(Rust) vs slow(Python) tokenizer行为/性能差异 special tokens机制(不同模型差异) padding策略与`padding_side`(左padding vs右padding,LLM生成为什么要左padding) truncation策略与`max_length` 批量编码机制(`__call__`返回结构,batch效率) chat template机制(`apply_chat_template`,不同模型对话格式差异) tokenizer保存与自定义词表增量添加(`add_tokens`/`resize_token_embeddings`联动关系)

### 02 Model 加载与 AutoClass 机制
`AutoModel.from_pretrained`内部机制(config解析→架构选择→权重下载缓存→state_dict对齐) `PretrainedConfig`体系与自定义config AutoClass自动分发机制 权重缓存机制与safetensors格式(为什么HF全面转向safetensors而不是pickle) `device_map="auto"`大模型多卡自动分配 低精度加载与`dtype=`参数(为什么从`torch_dtype`改名,fp16/bf16加载差异) `attn_implementation`选择机制(eager/sdpa/flash_attention_2,当前版本默认行为实测) `local_files_only`与离线模式(`HF_HUB_OFFLINE`) 模型架构注册机制(`AutoModel.register`扩展自定义模型)

### 03 Pipeline 高层API内核
`pipeline()`内部串联机制(tokenizer+model+预处理+后处理) 不同task pipeline差异(text-generation/text-classification/feature-extraction) 批量推理机制(`batch_size`参数) pipeline的设备管理(`device`参数) 自定义pipeline(继承`Pipeline`类) pipeline vs 手写调用的取舍(什么场景该绕开pipeline)

### 04 Datasets 库机制
`load_dataset`内部机制(Arrow格式、内存映射、为什么不会撑爆内存) `streaming=True`流式加载(超大数据集场景) `map()`并行处理机制(`num_proc`,batched模式) `DatasetDict`结构与train/test切分 `.with_format("torch")`桥接PyTorch Dataset/DataLoader `.filter()`/`.select()`/`.shuffle()`常用操作 自定义本地数据集加载(`load_dataset("json"/"csv", data_files=...)`,呼应dolly数据集的格式转换) `.map()`缓存失效坑

### 05 Trainer API 内核
`Trainer`整体架构(对比手写PyTorch训练循环,呼应torch-deep-dive07类) `TrainingArguments`关键参数解读 `Trainer.train()`内部循环到底做了什么 `compute_metrics`回调机制 checkpoint保存与`resume_from_checkpoint`机制 `DataCollator`机制(动态padding,`DataCollatorForLanguageModeling`) 自定义`Trainer`(覆写`compute_loss`) 内置callback机制(`EarlyStoppingCallback`等) `Trainer`与`accelerate`的关系(Trainer底层用accelerate)

### 06 Accelerate 分布式与设备机制
`accelerator.prepare()`内部机制(同一份代码单卡多卡都能跑的原理) 混合精度自动处理(`mixed_precision`参数) 梯度累加正确写法(`accelerator.accumulate`) `accelerate config`/`accelerate launch`机制 `save_state`/`load_state`checkpoint机制 单机多进程模拟分布式语义(复用torch-deep-dive09框架;Windows无NCCL只有gloo,物理多卡现象明确标注引用官方文档未实测)

### 07 PEFT 库工程机制
(差异声明:讲peft**库**怎么实现LoRA,不讲LoRA**算法**,算法见`learning/lora-family/lectures/01-lora.md`)
`get_peft_model()`内部机制(module替换/包装) `LoraConfig`关键参数(r/lora_alpha/target_modules/lora_dropout) `target_modules`怎么匹配到具体Linear层(module name匹配机制,PEFT官方LLaMA族映射表) 参数冻结机制(`print_trainable_parameters()`背后) adapter权重单独保存(不存base model) `merge_and_unload()`合并机制(合并后为什么推理零延迟) 多adapter切换机制(`set_adapter`,同一base model挂多个adapter) `PeftModel.from_pretrained`加载adapter权重

### 08 量化机制 bitsandbytes
(差异声明:讲`BitsAndBytesConfig`库工程实现,不讲量化数学,数学见`learning/quantization-deploy/`和`learning/lora-family/lectures/06-qlora.md`)
8bit量化实操(`load_in_8bit=True`) 4bit量化与`BitsAndBytesConfig`(nf4/双重量化) QLoRA怎么把量化+LoRA结合(`prepare_model_for_kbit_training`做了什么) 量化对显存占用的真实测量对比(fp16/8bit/4bit现场对比) 量化对推理速度的真实影响(现场测量) 量化常见报错与Windows一等支持现状说明

### 09 微调实战对比★重中之重
(开头注明:实验驱动统一使用`SFTTrainer`,建议先读10类"SFTTrainer内部机制"一点再进入本类动手部分)
实验设计与统一基准(模型/数据集/评估指标的确定性选型说明,呼应环境声明) 全参微调基线(真实loss曲线+真实显存+真实耗时,讲清"为什么必须强制bf16不留fp32主权重"这个真实工程决策) LoRA微调(同条件对比,r=8) QLoRA微调(同条件对比) 三者横向对比分析(表格化呈现,完整9846条数据集跑一次压轴) 超参敏感度扫描(rank/target_modules/学习率三线对比,合并为一点共享"为什么"论述) 微调前后生成效果人工对比 灾难性遗忘现象观察(简单探测) 训练日志与显存监控实操(`nvidia-smi`+`torch.cuda.memory_summary()`配合,呼应WDDM机制) 微调产物保存与复现(保存adapter+加载复现,确认结果一致) 数据规模对微调效果的影响 常见微调失败模式排查(loss不降/变nan/过拟合识别)

### 10 TRL 训练器抽象
(差异声明:讲trl库的Trainer封装工程实现,不重复DPO/RLHF算法,算法见`learning/dpo-family`/`learning/rlhf-classic`;**开头强制标注`PYTHONUTF8=1`前置设置**)
`SFTTrainer`内部机制(和裸`Trainer`的差异,数据格式化自动处理) chat格式数据集构造约定(`messages`格式,呼应no_robots数据集) `SFTConfig`关键参数 `DPOTrainer`工程角度简介(怎么传入chosen/rejected pair,不重复DPO loss数学) TRL+PEFT组合使用(`SFTTrainer`直接接`peft_config`) TRL常见坑(packing机制、序列长度设置、GBK编码崩溃案例回顾)

### 11 Hub 与模型分享机制
`huggingface_hub`库基础(`HfApi`,`login()`) 缓存机制精解(`~/.cache/huggingface/hub`目录结构,revision/etag机制,Windows符号链接默认不可用的真实警告) `push_to_hub()`机制 模型版本管理(`revision`参数pin到特定commit) model card机制 私有仓库与访问控制(`token`参数)

### 12 推理优化基础
(差异声明:讲怎么正确调用`transformers.generate()`,不是从零造推理引擎,系统工程视角见`learning/inference-engine-core/`)
`generate()`内部机制总览(自回归生成循环) 解码策略对比(greedy/beam/top-k/top-p/temperature) KV cache机制(`use_cache`参数,为什么能加速) `torch.compile`与HF模型结合(呼应torch-deep-dive) 批量推理的padding与attention_mask处理 `StoppingCriteria`自定义停止条件 流式生成(`TextStreamer`/`TextIteratorStreamer`)

### 13 调试与常见报错精解
标准CUDA OOM排查方法论 **Windows WDDM显存回落静默failure**(诊断方法:交叉核对`torch.cuda.max_memory_allocated()`与`nvidia-smi`/任务管理器,已现场复现) tokenizer/model不匹配报错(vocab size不对等) `padding_side`导致生成结果异常 device不一致报错 量化相关报错 `trust_remote_code=True`安全含义与使用场景 版本不兼容报错排查方法论 综合案例:哪些警告可以放心忽略(bitsandbytes FutureWarning等真实无害警告)

---

## 撰写与验证纪律

- 每个知识点的可运行例子必须在仓库根目录 `.venv` 真实跑通(`e:\Workspace\dummy\.venv\Scripts\python.exe`),涉及 trl 的代码块所在 shell 必须提前设置 `PYTHONUTF8=1`。
- 每个知识点必须自包含可独立运行(不依赖其他知识点残留的变量/进程状态)。
- 显存相关断言必须交叉核对 `torch.cuda.max_memory_allocated()` 和 `nvidia-smi` 输出,不能只信任一个数字。
- 大数据集/多组对比实验注意控制子采样规模,保证验证时间可控。
- 如实标注:DeepSpeed/多卡真实场景等确实无法在本环境验证的内容,标注"官方文档口径,未在本环境验证",不编造假验证记录。
- 每写完一批,在本文件进度表如实更新状态(生成中 → 验证通过才标"已完成"),不一次性提前标全部完成。

---

*创建:2026-07-12*
