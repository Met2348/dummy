# PEFT 深挖 —— 路线图与进度表

> 目标:约 24 个参数高效微调(Parameter-Efficient Fine-Tuning)知识点,由浅入深,深度对标 [torch-deep-dive/](../torch-deep-dive/00-roadmap.md)/[huggingface-deep-dive/](../huggingface-deep-dive/00-roadmap.md)(面试二三四面深度,不是"这个函数怎么调")。
> 定位:仓库"专题精读系列"第 4 条(四条新系列里规模最大的一条),直接对应 `learning/lora-family/` + `learning/adapter-tuning-family/` 两个专题模块——合计 23 种 PEFT 方法(LoRA 家族 12 种 + Adapter 家族 11 种),已逐文件读过两个模块的全部 `src/*.py`(含行数、每个文件实现什么、真训练 vs 纯数值/mock 模拟的精确分界、已有的 pytest 断言、`adapters` 库版本冲突导致的 skip 情况),本系列直接复用那批调研结果。

---

## 和 `learning/lora-family/`+`learning/adapter-tuning-family/` 的关系(差异化声明,必须先读)

**规模取舍(必须先说清楚)**:源模块合计 23 种方法,但本系列**不要求每种方法都单独展开成一个知识点**——按"家族关系"组织:核心方法(LoRA/QLoRA/DoRA/Houlsby/IA3)给足深度(完整七步),次要变体(rsLoRA/LoRA+/PiSSA/VeRA/LoHa/LoKr/AdapterFusion/Compacter/Parallel/AdapterDrop/MAM/K-Adapter/MAD-X/AdaMix 等)归入相关知识点内做横向对比说明,不逐一铺开成 23 个孤立小节——这是为了保持系列规模可控,不是遗漏或偷懒,每个明细小节里会写清楚具体怎么分组。

每个知识点从"最笨的想法"讲起(比如先问"全参数微调一个几十亿参数的模型,笔记本电脑或者单卡消费级显卡为什么根本训不动",再引出 LoRA 这类方法"只训一小撮新增参数"的思路),额外多两块:**底层机制/为什么这样设计** + **面试怎么问+追问链**。

---

## 环境声明

运行环境:仓库根目录 `.venv`(Windows 原生,Python 3.13,torch 2.11.0+cu128,transformers 5.10.2,peft 0.19.1,bitsandbytes 0.49.2)。**本系列是四条新系列里唯一涉及真实 GPU 训练的一条**——绝大多数知识点(LoRA 各变体的 minimal 实现、Adapter 各变体)是构造一个真实 GPT-2 模型 + 纯前向传播/参数量演示,不需要梯度训练;但两个地方例外:

1. **`qlora_minimal.py` 有真实 5 步训练循环**(手写 NF4 fake-quant + 真 `AdamW.step()`,CPU 可跑,不需要 GPU)。
2. **`qlora_peft.py` 需要真实 GPU**(真 bitsandbytes 4bit 量化,CPU 无对应路径,会 `print("[SKIP]")` 后直接返回)。

已用 `grep -r "adapters" learning/adapter-tuning-family/environment/requirements.txt` 确认:`adapter-tuning-family` 的 9 个 `*_adapters.py` 文件(每种方法一个)因为 `adapters` 库要求 `transformers<5.0`,和本仓库统一环境的 `transformers 5.10.2` 冲突,**全部 `tier:skip`,本地无法运行**——但每种方法都有对应的 `*_minimal.py`(手写)版本能正常跑,不影响本系列覆盖全部方法。`lora-family` 没有这类 skip,19/19 全部可跑。

---

## 知识点结构模板(七步,与 torch-deep-dive/huggingface-deep-dive 完全一致)

1. **签名/是什么** 2. **一句话** 3. **底层机制/为什么这样设计** 4. **AI 研究/工程场景** 5. **可运行例子**(带 assert,真在 `.venv` 里跑过) 6. **面试怎么问 + 追问链** 7. **常见坑**

---

## 进度表

| # | 分类 | 文件 | 知识点数(约) | 状态 |
|---|------|------|-----------|------|
| 01 | LoRA 核心与初始化变体 | [01-lora-core.md](01-lora-core.md) | 7 | ✅ 已完成(已验证,含 `merge_weights()` docstring 与实现不符导致双重计数发现 + B 零初始化下 A 在 step 0 梯度精确为零 + AdaLoRA"剪枝"只是前向掩码不省算力三处发现) |
| 02 | 量化 + LoRA | [02-quantized-lora.md](02-quantized-lora.md) | 6 | ✅ 已完成(已验证,本机真实 RTX 3080 Ti GPU 跑通知识点 3 的真 bitsandbytes 4bit 路径;含 NF4 网格 7负+1零+8正非对称 + 手写 NF4 与真 bitsandbytes 位级一致 + DoRA 与量化实为零关联三处发现) |
| 03 | Adapter 家族核心 | [03-adapter-core.md](03-adapter-core.md) | 6 | ✅ 已完成(已验证,含 Compacter `main()` 打印语句除数错误(漏乘 n=4)+ IA3 手写/peft 两条路径参数总量相等系巧合、缩放张量结构完全不同(经 peft 源码直接核实)两处发现) |
| 04 | Adapter 进阶与统一视角 | [04-adapter-advanced.md](04-adapter-advanced.md) | 5 | ✅ 已完成(已验证,含 MAD-X 的 `InvertibleAdapter` 完全未接入前向图 + MAM 的 `P_k` 参数梯度近零(真实 LM loss 反传复现约 7e-10 量级)两处发现) |

**预计合计:约 24 个知识点。**

---

## 明细(源码路径,撰写时逐一核实文件路径/行号仍然准确)

### 01 LoRA 核心与初始化变体(源:`learning/lora-family/src/{lora_minimal,lora_extensions,pissa_minimal,vera_minimal,loha_minimal,lokr_minimal,adalora_minimal}.py`)
1. LoRA 数学与实现(`lora_minimal.py::LoRALinear`)—— `h=base(x)+α/r·BAx`,B 零初始化保证训练开始时输出等于原模型
2. rsLoRA(`lora_extensions.py::RSLoRALinear`)+ LoRA+(`lora_plus_param_groups`)—— rsLoRA 改缩放公式为 `α/√r`;LoRA+ 给 A、B 两组参数设不同学习率(B 的 lr 通常是 A 的 16 倍)
3. `merge_and_unload()` 与推理零延迟——为什么合并后 `W' = W + BA` 可以完全消除推理时的额外计算
4. PiSSA/OLoRA(`pissa_minimal.py`)—— 用真实 SVD/QR 分解 `W_0` 初始化 A、B(而不是随机初始化),`pissa_minimal.py` 是全系列少数带真 assert 的文件(SVD 重构精度检验)
5. VeRA(`vera_minimal.py`)—— 全部层共享同一份冻结的随机 A/B 矩阵,每层只训练两个低维缩放向量,参数量比 LoRA 更省
6. LoHa/LoKr(`loha_minimal.py`/`lokr_minimal.py`)—— Hadamard 乘积分解(有效秩可达 r²)/ Kronecker 乘积分解;`loha_lokr_peft.py` 会真实抛 `ValueError`(peft 库的 LoHa/LoKr 不支持 GPT-2 的 `Conv1D` 层),这是文档化的预期行为不是 bug
7. AdaLoRA(`adalora_minimal.py`)—— SVD 形式的 `ΔW=PΛQᵀ` + 基于 EMA 重要性分数的自适应秩剪枝(`cubic_schedule`)

### 02 量化 + LoRA(源:`learning/lora-family/src/{nf4_quant,qlora_minimal,qlora_peft,loftq_minimal,dora_minimal}.py`)
1. NF4 量化机制(`nf4_quant.py`)—— 纯 PyTorch 手写 4-bit 块量化(block_size=64)+ double quantization(对 absmax 本身也做量化)
2. QLoRA(`qlora_minimal.py`)—— NF4 fake-quant 冻结权重 + LoRA,**全系列唯一的真 5 步训练循环**,验证量化后的 base weight 训练前后保持不变(bit-identical)
3. QLoRA 真 4bit 路径(`qlora_peft.py`)—— 真实 `BitsAndBytesConfig` + `prepare_model_for_kbit_training` + TinyLlama,**GPU-only**,明确标注这是可选进阶验证
4. LoftQ(`loftq_minimal.py`)—— 交替最小化算法(5 轮 SVD(W−Q) → NF4 重新量化(W−BA)),记录真实收敛轨迹
5. DoRA(`dora_minimal.py`)—— 把权重分解成 `W=m·V/‖V‖`(magnitude × direction),只有方向部分参与 LoRA 式低秩更新,magnitude 单独学习
6. **"真4bit训练+真bitsandbytes从未在本仓库同时出现"这个精确边界**——`qlora_minimal.py` 的真训练循环用的是手写 fake-quant(不是 bitsandbytes),`qlora_peft.py` 的真 bitsandbytes 4bit 路径只做一次前向传播(不训练);这个调研阶段已发现的精确事实,避免学生误以为仓库里存在一个"真 bitsandbytes 4bit + 真多步训练"同时具备的例子

### 03 Adapter 家族核心(源:`learning/adapter-tuning-family/src/{adapter_original_minimal,houlsby_minimal,pfeiffer_minimal,adapterfusion_minimal,compacter_minimal,parallel_minimal,ia3_minimal,ia3_peft}.py`)
1. 原始 bottleneck adapter 数学(`adapter_original_minimal.py`)—— 纯 Python 无 torch 实现,down-project→非线性→up-project,up 投影零初始化;是全系列唯一用 `_self_test()`(而不是 `main()`)命名、且带真 assert 的文件
2. Houlsby vs Pfeiffer(`houlsby_minimal.py`/`pfeiffer_minimal.py`)—— Houlsby 每个 transformer block 插 2 个 adapter(attn 后+FFN 后),Pfeiffer 简化成只插 1 个(FFN 后),参数量少一半
3. AdapterFusion(`adapterfusion_minimal.py`)—— 用 attention 机制融合多个冻结 adapter 的输出(Q 来自 hidden state,K/V 来自各 adapter 输出),训练一个额外的 fusion 层学习怎么组合多任务 adapter
4. Compacter(`compacter_minimal.py`)—— PHM(Parameterized Hypercomplex Multiplication)+ 跨层共享的 Kronecker 分解参数,是本系列参数压缩率最高的方法之一
5. Parallel Adapter(`parallel_minimal.py`)—— 并联(加法式 `base(x)+scaling·adapter(x)`)而不是串联,是 LoRA 的重要前身/近亲设计
6. IA3(`ia3_minimal.py`+`ia3_peft.py`)—— 3 个对角缩放向量(不是低秩矩阵),全部 ones 初始化(等价恒等变换);全系列唯一有 3 条独立实现路径(手写/adapters库(skip)/peft库)互相交叉验证参数量一致的方法

### 04 Adapter 进阶与统一视角(源:`learning/adapter-tuning-family/src/{adapterdrop_minimal,mam_minimal,k_adapter_minimal,madx_minimal,adamix_minimal}.py` + `lectures/09-three-line-unification.md`)
1. AdapterDrop(`adapterdrop_minimal.py`)—— 训练时随机丢弃部分层的 adapter(加速训练),推理时永久跳过某些层的 adapter(加速推理)
2. MAM(`mam_minimal.py`)—— Mix-And-Match:Prefix-tuning 风格的 attention 部分(近似实现,不是真正的序列前置)+ Parallel Adapter 风格的 FFN 部分,统一视角论文的代表方法
3. K-Adapter / MAD-X(`k_adapter_minimal.py`/`madx_minimal.py`)—— K-Adapter 用多个并行"知识 adapter"分别编码不同类型知识(事实性/语言学性);MAD-X 用 invertible adapter + 每语言/每任务 adapter 做跨语言迁移;如实标注两个文件里各自定义了一些"从未被 `main()` 实际使用"的 toy 数据常量
4. AdaMix(`adamix_minimal.py`)—— Mixture-of-Adapters,训练时每次前向随机路由到 N 个 expert 里的 1 个,推理时平均所有 expert(`merge_experts()` 折叠回单个 adapter)
5. Prompt+LoRA+Adapter 三线统一公式(来自 `learning/adapter-tuning-family/lectures/09-three-line-unification.md`)—— 呼应仓库已有的 `learning/prompt-tuning-family/`,把三大类 PEFT 方法表示成同一个通用公式的不同实例化,收官知识点

---

## 撰写与验证纪律

- 每个知识点的可运行例子必须在仓库根目录 `.venv` 真实跑通;`qlora_peft.py` 真 4bit 路径明确标注为可选进阶验证(GPU-only)。
- `adapters` 库相关的 9 个 `*_adapters.py`(tier:skip)如果要展示,只能展示"确实会 ImportError/版本冲突"这个事实,不能假装本地能跑通。
- K-Adapter/MAD-X 提到的未使用 toy 数据常量,如实标注,不强行解释成"有意义的设计"。
- 每写完一批,在本文件进度表如实更新状态(⬜ 待撰写 → 🔧 撰写中 → ✅ 已完成,验证通过才标"已完成")。

---

*创建:2026-07-12*
