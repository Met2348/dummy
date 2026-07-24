# 长上下文技术深挖 —— 路线图与进度表

> 目标:约 17 个长上下文(long-context)技术知识点,由浅入深,深度对标 [torch-deep-dive/](../torch-deep-dive/00-roadmap.md)/[huggingface-deep-dive/](../huggingface-deep-dive/00-roadmap.md)(面试二三四面深度,不是"这个函数怎么调")。
> 定位:这是仓库"深挖系列"新的一批(第 7 条),和前 6 条(numpy/python-advanced/torch/huggingface/tensorflow/python-idioms)性质不同——前 6 条补的是"读懂 `learning/` 代码需要的通用框架/语言技能",不绑定具体研究专题;本系列**直接对应 `learning/long-context/` 这一个专题模块**,用更适合初学者/面试备考的讲解深度(7 步结构,从最笨的想法讲起,面试追问链)重新讲解**同一份代码**,不是重复造轮子。

---

## 和 `learning/long-context/` 的关系(差异化声明,必须先读)

`learning/long-context/` 是博士级研究笔记,面向已经懂 RoPE/attention 数学基础的读者,行文密度高、跳步快(比如直接给出 YaRN 的 ramp 函数不解释为什么要分段)。本系列讲**同一份代码**,但:
- 每个知识点从"最笨的想法"讲起再引入正式方案(比如先讲"直接外推 RoPE 会怎样",再引入 PI/NTK/YaRN 分别怎么解决这个问题)
- 每个例子都要求能在 `.venv` 里独立跑通并用 `assert` 验证,不是转述文档
- 额外多两块:**底层机制/为什么这样设计**(不停在怎么用)+ **面试怎么问+追问链**
- 论文导读见 `learning/long-context/paper/guide_01_yarn_context_extension.md`(503 行,YaRN 论文),本系列不重复推导论文数学,聚焦"代码怎么实现这个数学、和论文/生产库的真实差异在哪"

本系列读者应该把它当成"`learning/long-context/` 的精读伴读笔记"来用,遇到不确定的地方以 `learning/long-context/` 的源码为准。

---

## 环境声明

运行环境:仓库根目录 `.venv`(Windows 原生,Python 3.13)。**全系列纯 CPU**——`learning/long-context/src/` 下所有代码零 GPU 依赖(已用 `grep -rn ".cuda()\|device=\"cuda\""` 核实零命中),秒级跑完,不需要下载任何模型权重。唯一例外是 capstone 的 `--train` 模式(需要 gated Llama-3.2-1B 权重 + 大显存),本系列不要求跑这个模式,只讲清楚它在做什么、默认 dry-run 模式打印什么。

依赖:`torch>=2.5`、`einops`、`transformers>=5.0`,均已在仓库 `.venv` 装好(核实版本:torch 2.11.0+cu128、transformers 5.10.2、einops 0.8.2)。

---

## 知识点结构模板(七步,与 torch-deep-dive/huggingface-deep-dive 完全一致)

1. **签名/是什么** —— 函数/概念定义,人话翻译
2. **一句话** —— 是什么
3. **底层机制/为什么这样设计** —— 不停在"怎么用",讲到"为什么必须是这样"
4. **AI 研究/工程场景** —— 具体在研究/工程代码里怎么用
5. **可运行例子** —— 带 `assert` 验证,真的在仓库 `.venv` 里跑过
6. **面试怎么问 + 追问链** —— 面试官大概率怎么问,追问会往哪个方向深挖
7. **常见坑**

---

## 进度表

| # | 分类 | 文件 | 知识点数(约) | 状态 |
|---|------|------|-----------|------|
| 01 | RoPE 外推家族 | [01-rope-scaling-family.md](01-rope-scaling-family.md) | 6 | ✅ 已完成(已验证,含"YaRN 教学代码 vs 生产库"深挖 + 发现 `rope_yarn.py` 死代码;P/T/L/V/D 逐文件复核追加 M-RoPE head_dim 切分示意表格——原文只有文字描述,缺可视化) |
| 02 | 长上下文 Attention 架构 | [02-long-context-attention.md](02-long-context-attention.md) | 4 | ✅ 已完成(已验证;P/T/L/V/D 逐文件复核未发现需要改动之处) |
| 03 | 长上下文评测方法论 | [03-long-context-eval.md](03-long-context-eval.md) | 3 | ✅ 已完成(已验证,发现 `niah_grid` 类型注解错误 + `ruler_eval.py` 死变量;P/T/L/V/D 逐文件复核未发现需要改动之处) |
| 04 | 数据工程与 Capstone | [04-long-context-capstone.md](04-long-context-capstone.md) | 4 | ✅ 已完成(已验证,含 `make_doc_mask` padding 边界发现;P/T/L/V/D 逐文件复核追加 KV-cache"是什么/为什么需要"前置说明——原文直接给显存公式未解释概念本身 + LoRA r/alpha 最小释义与 peft-deep-dive 交叉引用) |
| 05 | 进阶深度追加:4 个多级追问链案例 | [05-advanced-interview-depth.md](05-advanced-interview-depth.md) | 4案例(不计入17) | ✅ 已完成(已验证,8/8代码块独立通过;基于真实WebSearch调研的5条追问轴线撰写——①RoPE外推方案批判迭代链(vanilla→PI→NTK→YaRN,每一步换方案的代价都现场算出具体弧度/百分比,不是"效果不好"空话)、②YaRN教学代码vs生产库真实性验证(多个factor取值排除单点巧合,证明差距随factor增大而扩大)、③KV-cache显存规模递增(上下文长度轴+并发数轴独立都能推爆单卡,两轴相乘不相加)、④Ring/Striped/Infini-Attention选型决策依据追问(GPU数量容量规划从"应该够用"变成现算数字);独立复验阶段额外用非2的幂次factor([3,6,12,24,48])+完全不同的hidden_size/heads组合重新验证案例2的"生产库无sqrt"结论(0/5不匹配,且证明attention_factor与hidden_size/heads完全无关,只是factor的函数),以及用70B级GQA架构+经典MHA架构(无GQA)重新验证案例3的KV-cache结论(70B权重本身就吃满H200预算;MHA因kv_heads是GQA的4倍,在128k就已超预算,而文档的8B GQA配置在128k还有余量——证明GQA的kv-head缩减是"128k装得下"这个结论成立的关键前提,不是巧合);P/T/L/V/D 逐文件复核未发现需要改动之处) |

**合计:17 个知识点,4 篇 + 1 篇进阶深度追加(4 个案例,不计入 17)+ 1 篇教程体(不计入 17),全部完成并独立验证。**

| 06 | 手把手实战:从零搭一个迷你NIAH评测器 | [06-build-a-mini-niah-evaluator.md](06-build-a-mini-niah-evaluator.md) | 4阶段(不计入17) | ✅ 已完成(5/5代码块独立通过;串联03类知识点1"造测试用例"/"depth_pct相对位置设计"、判分逻辑、Lost-in-the-Middle三件事,自己从零手写(不导入`learning/long-context/src/niah_eval.py`,函数名故意都不同,避免被误当成仓库源码行为的依据)。诚实标注复现的U型曲线是在判分函数里手写位置相关衰减权重人为构造的,不是接了真实模型跑出来的,延续03类知识点3已立的规矩) |

**关于 06 类的方法论说明:** "教程体"格式最早在 [dsa-deep-dive/21](../dsa-deep-dive/21-build-a-mini-search-engine.md) 试点,这是第一次推广到 dsa-deep-dive 之外的系列,验证了它不依赖具体专题——03 类讲的是评测方法论而不是数据结构,同样的"分阶段动手搭"节奏依然适用。是否继续推广到其余系列,留给后续单独决定。

---

## 明细(对应源码,撰写时逐一核实文件路径/行号仍然准确)

### 01 RoPE 外推家族(源:`learning/long-context/src/{common,rope_pi,rope_ntk,rope_yarn,rope_3d}.py`)
1. Vanilla RoPE 回顾(`common.py::inv_freq`/`build_cos_sin`/`apply_rope_interleaved`)—— 只回顾对本系列后续必需的部分,不重复 torch-deep-dive 已讲过的 tensor 机制
2. Position Interpolation(`rope_pi.py::pi_cos_sin`)—— 压缩位置而不是压缩频率;**代数恒等式**:`pi_cos_sin(t=8,dim=16,scale_factor=4)` 的 `cos[4,0]` 精确等于未缩放 `cos[1,0]`(4/4=1)
3. NTK-aware RoPE(`rope_ntk.py::ntk_cos_sin`)—— 缩放 base 而不是位置;最高频维度(index 0)完全不变、最低频维度被压缩约 scale_factor 倍
4. YaRN(`rope_yarn.py::yarn_cos_sin`+`_yarn_ramp`)—— NTK-by-parts 分段 ramp + attention temperature;讲清楚"为什么要按维度分段"(高频保留细节、低频允许外推)
5. **YaRN 教学代码 vs 真实 transformers 库的公式差异**(精确验证过:`rope_yarn.py` 用 `sqrt(0.1·ln(s)+1)`,真实 `transformers==5.10.2` 的 `modeling_rope_utils.py::get_mscale` 直接用 `0.1·ln(s)+1`,少了一个 sqrt)—— 这条不是造轮子,是"怎么用代码交叉核对文档/教学简化和生产实现"的方法论示范
6. 3D-RoPE / M-RoPE(`rope_3d.py::m_rope`)—— 多模态场景下把 head_dim 切成 3 段分别编码 t/h/w 坐标

### 02 长上下文 Attention 架构(源:`learning/long-context/src/{ring_attention_naive,ring_attention_lib,infini_attention}.py`)
1. Ring Attention(`ring_attention_naive.py`)—— 序列切块 + online softmax 递推(m/l/o)模拟多卡通信;已有 pytest 断言"和 vanilla attention 数值精确等价(diff 2.4e-07)"+"结果与 n_rank 无关"
2. Striped Attention 概念(仅 lecture 06,无对应 src)—— Ring Attention 的负载均衡改进,如实标注"本仓库只有概念讲解,没有单独实现"
3. Infini-Attention(`infini_attention.py::InfiniBlock`)—— 局部因果 attention + 压缩记忆(sigmoid 线性 attention 检索)+ 可学习门控混合
4. 三种长上下文 attention 架构对比选型(单卡够用 vs 多卡序列并行 vs 理论无限上下文)

### 03 长上下文评测方法论(源:`learning/long-context/src/{niah_eval,ruler_eval}.py` + lecture 12)
1. NIAH(Needle in a Haystack,`niah_eval.py`)—— 测试用例生成器 + 字符串匹配检查器,讲清楚"这是生成测试题,不是跑模型"
2. RULER(`ruler_eval.py`)—— NIAH 的扩展:single/multi-key/multi-value NIAH + variable tracking 4 个子任务
3. Lost in the Middle 现象(来自 lecture 12)—— U 型准确率曲线,为什么中间位置的信息容易被模型忽略

### 04 数据工程与 Capstone(源:`learning/long-context/src/{long_data_packing,capstone_yarn_llama32}.py`)
1. 文档打包与 attention mask(`long_data_packing.py::pack_documents`/`make_doc_mask`)—— first-fit-decreasing 装箱 + 块对角 mask 防止跨文档串扰;已验证 packing_efficiency 98.75%~100%
2. Curriculum learning(`long_data_packing.py::curriculum_lengths`)—— 4 阶段长度课程学习
3. Capstone:Llama-3.2-1B + YaRN(scale=4) + LoRA → 32k(`capstone_yarn_llama32.py`)—— **精确标注**:默认是 dry-run(不加载任何模型),`--train` 模式也只做到 LoRA setup 为止,注释里明确写"省略 data loader + Trainer 实例化",不是完整训练脚本
4. KV-cache 显存膨胀(来自 lecture 12)—— 长上下文场景下 KV cache 相对模型权重本身的显存占比可以达到 8 倍量级,配合具体公式演算

---

## 撰写与验证纪律

- 每个知识点的可运行例子必须在仓库根目录 `.venv` 真实跑通,全部纯 CPU、不依赖任何模型下载。
- 涉及"教学代码 vs 真实库"的对比(01-05),必须现场 import 已装的 `transformers` 包核实差异仍然成立,不能凭一次性调研结论直接写死。
- capstone 相关内容严禁暗示"这是一个完整可训练脚本"——如实标注 dry-run/LoRA-setup-only 的边界。
- 每写完一批,在本文件进度表如实更新状态(⬜ 待撰写 → 🔧 撰写中 → ✅ 已完成,验证通过才标"已完成")。

---

*创建:2026-07-12*
