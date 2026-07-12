# GPU 架构与 Kernel 工程深挖 —— 路线图与进度表

> 目标:约 19 个 GPU 架构/Kernel 工程知识点,由浅入深,深度对标 [torch-deep-dive/](../torch-deep-dive/00-roadmap.md)/[huggingface-deep-dive/](../huggingface-deep-dive/00-roadmap.md)(面试二三四面深度,不是"这个函数怎么调")。
> 定位:仓库"深挖系列"第 8 条,直接对应 `learning/gpu-architecture/` + `learning/kernel-engineering/` 两个专题模块——这两个模块我(Claude)在 2026-07-12 亲自逐脚本验证过(全仓 46 模块 runbook 验证任务的一部分),对其中每个函数的真实输出、公式、发现的坑都有第一手核实过的认知,本系列直接复用那批验证结果,不是凭空重新调研。

---

## 和 `learning/gpu-architecture/`+`learning/kernel-engineering/` 的关系(差异化声明,必须先读)

这两个 `learning/` 模块本身已经有很详尽的 README(专题概览/横向对比/cheatsheet/自测题一应俱全),但行文是"研究者对研究者"的密度,不铺垫、不做面试向的追问链。本系列讲**同一份代码**,但:
- 每个知识点从"最笨的想法"讲起(比如先问"为什么普通矩阵乘法在 GPU 上会算力打折",再引入 tensor core/roofline 的解释)
- 额外多两块:**底层机制/为什么这样设计** + **面试怎么问+追问链**
- 两个源模块的 README 已经反复强调一件事,本系列开头必须再次显著重申:**`src/` 下的代码不是可编译的真实 CUDA/Triton/CUTLASS kernel,是用可断言验证的纯 Python 数值/机制模拟去复现这些系统的行为**——Windows 工作站上的设计取舍,不是造轮子偷懒。真编译需要 Linux CUDA 工具链,不在本系列/`learning/` 任何模块的范围内。

---

## 环境声明

运行环境:仓库根目录 `.venv`(Windows 原生,Python 3.13)。**全系列纯 CPU、零第三方依赖**——`learning/gpu-architecture/src/` 和 `learning/kernel-engineering/src/` 下全部 14 个脚本只依赖 `dataclasses`/`math`/`__future__`,已在今天的 runbook 验证中逐个核实(见 `learning/gpu-architecture/README.md`/`learning/kernel-engineering/README.md` 的"环境配置"段),秒级跑完,不需要 GPU、不需要装包。

两个模块的脚本互相之间的 import 情况不同:`gpu-architecture` 的脚本会互相 `from common import ...`;`kernel-engineering` 的 6 个脚本彼此**零 import**,可以任意顺序单独跑。撰写时注意区分。

---

## 知识点结构模板(七步,与 torch-deep-dive/huggingface-deep-dive 完全一致)

1. **签名/是什么** 2. **一句话** 3. **底层机制/为什么这样设计** 4. **AI 研究/工程场景** 5. **可运行例子**(带 assert,真在 `.venv` 里跑过) 6. **面试怎么问 + 追问链** 7. **常见坑**

---

## 进度表

| # | 分类 | 文件 | 知识点数(约) | 状态 |
|---|------|------|-----------|------|
| 01 | GPU 硬件与存储层次 | [01-gpu-hardware-and-memory.md](01-gpu-hardware-and-memory.md) | 5 | ✅ 已完成(已验证,含 `cost_to_load` 千倍单位错误 + FP4/FP8 注释与实测不符两处额外发现) |
| 02 | Roofline 性能建模 | [02-roofline-model.md](02-roofline-model.md) | 4 | ✅ 已完成(已验证,ridge point 反直觉发现 + A100/H100 翻转案例) |
| 03 | Kernel 设计语言:Triton & CUTLASS | [03-kernel-design-triton-cutlass.md](03-kernel-design-triton-cutlass.md) | 5 | ✅ 已完成(已验证,发现 128×256/256×128 打分完全平局、纯靠列表顺序决出) |
| 04 | FlashAttention 与 Kernel Fusion | [04-flashattention-and-fusion.md](04-flashattention-and-fusion.md) | 5 | ✅ 已完成(已验证,发现 rmsnorm 数值差异根源是 Python 3.12+ `sum()` 补偿求和 vs 手写循环) |

**合计:19 个知识点,4 篇全部完成并独立验证。**

---

## 明细(源码路径 + 我今天亲自验证过的关键数字,撰写时仍需重新独立跑一遍确认没有回归)

### 01 GPU 硬件与存储层次(源:`learning/gpu-architecture/src/{common,memory_hierarchy,tensor_core,sm_occupancy,nvlink_topology}.py`)
1. GPU 规格数据表(`common.py::GPU_CATALOG`)—— H100/H200/B200 的 FLOPS/HBM 带宽/显存容量怎么组织成可编程查询的数据结构
2. 存储层次与 tier 推荐(`memory_hierarchy.py::recommend_tier`)—— 今天验证时修过一个死代码 bug(`if False else` 恒走 else 分支,行为不变但关键字用错),可以作为"读代码要留意惯用法陷阱"的真实案例
3. Tensor Core 吞吐建模(`tensor_core.py`)—— 为什么专用矩阵乘法单元比通用 ALU 快一个数量级
4. SM Occupancy(`sm_occupancy.py`)—— 寄存器/共享内存怎么限制一个 SM 上能同时驻留多少个线程块
5. NVLink 拓扑带宽(`nvlink_topology.py`)—— 多卡互联带宽建模,为跨专题衔接到 `cluster-networking` 打基础(仅点出关联,不重复讲)

### 02 Roofline 性能建模(源:`learning/gpu-architecture/src/{roofline,roofline_original_minimal}.py` + capstone)
1. Arithmetic Intensity 与 Roofline 分类(`roofline.py`)—— compute-bound vs memory-bound 怎么从一个比值判断
2. Ridge Point 计算 —— 算力和带宽的交叉点公式
3. **Ridge Point 非单调的反直觉发现**(H200 206.0 < H100 295.2,今天验证 `learning/gpu-architecture/` 时独立发现并写入该模块 README 的真实洞察,原因是带宽涨幅超过算力涨幅——直接复用这个发现,重新独立跑一遍数字确认没有回归)
4. Capstone:10 op × 4 GPU roofline zoo(把前面的公式应用到真实算子清单上,批量分类)

### 03 Kernel 设计语言:Triton & CUTLASS(源:`learning/kernel-engineering/src/{triton_style,cutlass_layout}.py`)
1. Triton autotune 决策(`triton_style.py`)—— SMEM 预算约束怎么算(`smem_bytes(cfg) = dtype_bytes × num_stages × (block_m×block_k + block_k×block_n)`)
2. Autotune 启发式打分公式(`score(cfg) = (block_m×block_n)/(block_m+block_n) × num_stages − parallelism_penalty`)—— 今天验证确认 4096³ 大 GEMM 会选中 128×256 tile,这个数字要重新独立跑一遍
3. CUTLASS/CuTe layout 代数(`cutlass_layout.py`)—— `offset(idx) = Σ idx_d × stride_d` 线性偏移公式,row-major/col-major 怎么用 stride 表示
4. Coalescing(内存合并访问)判定 —— 为什么某个轴的访问模式快、另一个轴慢
5. **`swizzle_32b()` 是诚实标注的 stub 而非 bug**(今天验证时独立发现:docstring 承诺 XOR-based swizzle 但函数体从未实现真正的 XOR 变换,只是恒等返回 row_major;已修复为显式 stub 说明+新增断言锁定当前行为,让这个 gap 从"静默未测试"变成"诚实且可测试")—— **这条本身是极好的教学素材**:怎么从"docstring 说了什么"和"`_self_test()` 到底断言了什么"这两者的落差,判断出一个函数是真实现还是名不副实的空壳,这是读陌生代码库(尤其是面试现场被要求读一段代码)时的核心技能

### 04 FlashAttention 与 Kernel Fusion(源:`learning/kernel-engineering/src/{flash_attention,fused_mlp,rmsnorm_kernel,capstone_attn_speedup}.py`)
1. Online Softmax 递推(`flash_attention.py`)—— `m_new=max(m,max(s_block))` → `rescale=exp(m−m_new)` → `l_new`/`o_new` 三步递推,为什么这个递推在数学上和标准 softmax **精确等价**(不是近似)
2. Naive vs Flash 的 HBM 流量对比 —— naive attention 需要把 N×N 的中间矩阵完整写入/读出 HBM,flash 全程不 materialize 这个矩阵
3. Kernel Fusion 的 HBM 流量核算(`fused_mlp.py`)—— 融合两个算子省下一次中间结果的写入+读回,今天验证的具体数字(30.8% HBM 流量节省)要重新独立跑一遍
4. RMSNorm + Linear Fusion(`rmsnorm_kernel.py`)—— fusion 不改变数学结果、只改变访存次数的具体例证
5. Capstone:128k 序列 1025× HBM 节省(`capstone_attn_speedup.py`)—— naive `O(N²)` vs flash `O(N·d)` 的渐近关系,序列越长节省越夸张,今天验证的 5 个序列长度对照表(512→131072,5.0×→1025.0×)要重新独立跑一遍确认没有回归

---

## 撰写与验证纪律

- 每个知识点的可运行例子必须在仓库根目录 `.venv` 真实跑通,全部纯 CPU、秒级。
- 我今天验证 `learning/gpu-architecture/`+`learning/kernel-engineering/` 时拿到的具体数字(ridge point 反直觉发现、swizzle_32b stub、1025× 加速等)在本系列撰写时**必须重新独立跑一遍确认没有回归**,不能因为"今天刚验证过"就直接照抄数字——两次验证之间源码不应该变,但复验永远是纪律,不是可选项。
- 每写完一批,在本文件进度表如实更新状态(⬜ 待撰写 → 🔧 撰写中 → ✅ 已完成,验证通过才标"已完成")。

---

*创建:2026-07-12*
