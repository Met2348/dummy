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
| 05 | 进阶深度追加:5 个多级追问链案例 | [05-advanced-interview-depth.md](05-advanced-interview-depth.md) | 5案例(不计入24) | ✅ 已完成(已验证,10/10代码块独立通过;基于 dsa-deep-dive/alignment-algorithms-deep-dive 已验证的 5 条追问轴线撰写——①LoRA→QLoRA→LoftQ→DoRA 方案批判迭代链(核心案例;独立复验发现 02 号文件"交替最小化每一步不会变差"这条 assert 通过的结论并非全称成立,40 个新种子里 34 个(85%)在同样配置下就会出现局部残差回升,02 号文件恰好用的种子 0 只是因为 n_iter=5 停得早、放宽到 15 轮同样违反,并溯源到 absmax=max(abs(·)) 不是 L2 误差最优缩放这一根因;另外现场手写 QDoRA 最小原型验证量化轴与 magnitude/direction 轴确实正交可组合)、②Houlsby→Pfeiffer→Compacter→(IA)³ 压缩谱系决策依据追问(核心案例;约束求解器验证"<60K 参数+需要非线性"在默认超参下无解,进一步验证这只是默认超参 r=16/n=4 下的结论,调大 Compacter 的 n 到 8 或调小 r 到 8 都能重新打开这个约束,区分"当前配置无解"与"方法家族无解"两种不同强度的结论)、③LoRA 多租户并发服务工程约束递增(核心案例;用 threading.Event 精确复现共享 base 权重被并发合并时的更新丢失,加锁后正确;批处理不合并方案 N=20 租户实测存储节省 15.7 倍)、④`merge_weights()`"删除"承诺真实性验证(核心案例;01 号文件已发现的 minimal 双重计数 bug,对照真实 peft 库验证出两层防御——`merge_adapter()` 靠 `merged` 运行时标志防御(`lora_A` 仍占显存但 forward 不再双重计数,可逆)、`merge_and_unload()` 做结构级卸载(模块类型从 `Linear` 变 `Conv1D`,`lora_A` 属性彻底消失,不可逆))、⑤QLoRA 4bit 训练边界+规模外推(真实性验证+规模递增双轴;本机真实 RTX 3080 Ti 测出干净的 0.5159 bytes/param 边际比例,用 02 号文件已实测的 TinyLlama 1049.3MB 做交叉检验发现 naive 外推低估 1.85 倍,修正后 7B 权重估算 6.68GB;并用 `torch.cuda.device_count()==1` 如实确认这台机器验证不到任何多卡行为)) |
| 06 | 手把手实战:从零搭一个迷你 LoRA 层 | [06-build-a-mini-lora-layer.md](06-build-a-mini-lora-layer.md) | 4阶段(不计入24) | ✅ 已完成(已验证,5/5代码块独立通过;不调用 `peft` 库,从 `nn.Linear` 手写冻结→LoRA分支→反向传播梯度验证→组装迷你两层MLP真实训练循环四阶段,全程 CPU 玩具规模秒级跑完;复用上一路 agent 撞 API 日限额中断前留下的 `_tmp_stage1.py`~`_tmp_stage4b.py` 五个临时脚本——逐一核实其 `MiniLoRALinear`/`TinyLoRAMLP` 实现、梯度验证逻辑与 [01 类知识点 1](01-lora-core.md) 的真实 `LoRALinear` 一致后,在 `.venv` 重新跑一遍确认输出不变才转写进正文,未重新推导;额外发现一处真实训练稳定性现象——同一个迷你 LoRA MLP,`lr=0.05` 时 loss 中途飙升到初始值 6 倍以上(`1.0451→6.7275`)且 10 步内未恢复,`lr=0.02` 时 11 个数字严格单调下降,已补 assert 收作花絮,如实标注"只训练两个小矩阵"不会让学习率选择自动免疫于不稳定) |

**预计合计:约 24 个知识点,4 篇 + 1 篇进阶深度追加(5 个案例,不计入 24)+ 1 篇教程体(4 阶段,不计入 24)。**

**关于 06 类的方法论说明:** dsa-deep-dive 系列试点验证"教程体"格式(读者动手从空文件开始写、每写一段就跑一次看真实效果,区别于 05 类"读者旁观、跟着面试官和候选人的推理链条走"的叙事体)之后,这一格式被推广到本系列,新增本文件。与 dsa-deep-dive 21 号文件的一处关键差异:dsa-deep-dive 全系列本来就是纯 Python 标准库、不涉及 GPU;peft-deep-dive 系列平时大量知识点依赖真实 GPU 验证(01~02 号文件在本机 RTX 3080 Ti 上跑过真实 bitsandbytes 4bit 量化),但这次新增的教程体内容被明确要求刻意控制成 CPU 也能跑的玩具规模(张量维度几十到几百,`torch.device("cpu")`,秒级跑完),不引入需要 GPU 的真实大模型微调——这是教程体"验证机制、不追求规模"这个定位本身决定的,不是能力不够,如实标注这一差异。撰写时复用了上一路 agent 因撞上 API 日限额中断而留在目录下的 `_tmp_stage1.py`~`_tmp_stage4b.py` 五个临时脚本:逐一读过确认其中的冻结/LoRA分支/梯度验证/参数量对比逻辑均和 [01 类知识点 1](01-lora-core.md) 已验证过的真实 `LoRALinear` 一致后,在 `.venv` 里重新跑一遍确认输出不变,再转写进正文,没有重新推导;`_tmp_stage4b.py` 是一份没有 assert 的学习率探索脚本,重新验证后确认"`lr=0.05` 导致 loss 中途飙升、`lr=0.02` 单调下降"这个现象是确定性、可复现的(固定随机种子下多次重跑数字完全一致),补上 assert 后收进正文作为训练稳定性的真实花絮,不是凭空设计的教学效果。

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
