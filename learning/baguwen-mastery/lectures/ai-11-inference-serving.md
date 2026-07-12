# 11 · 推理部署与服务化（KV cache/PagedAttention/投机采样/量化）

## 这一类在面试里的分量 + 2026 新趋势

如果说分布式训练考的是"你会不会炼大模型"，推理部署考的就是"你会不会把模型真正伺候上线"——这是 2026 年几乎所有和 LLM 落地相关的岗位（不管是研究岗还是工程岗）都绕不开的一块，因为几乎每个团队最终都要面对"这个模型怎么部署、多贵、多快"的问题。这一类的特点是概念之间环环相扣：KV cache 显存爆炸 -> PagedAttention 管理碎片 -> continuous batching 提升吞吐 -> 投机采样/量化进一步压延迟和成本，面试官往往会顺着这条链条一路追问下去，单点背熟没用，得理解清楚"为什么需要下一个技术"。2026 年的新趋势是 vLLM/SGLang/TensorRT-LLM 这类框架的核心技术（尤其是 PagedAttention 及其衍生的 prefix caching、chunked prefill）几乎成了"新八股"，很多 JD 直接点名要求"熟悉 vLLM 服务化经验"，比过去几年更看重候选人对推理框架内部机制的理解深度，而不只是会调 API。

## 追问链深挖：PagedAttention 到底解决了什么问题

**Q1（面试官）：你说 vLLM 的核心是 PagedAttention，它具体解决了什么问题？**

**A1**：解决的是 KV cache 的显存管理效率问题。传统做法给每个请求的 KV cache 预先分配一段连续显存（通常按最大可能生成长度预留），但实际生成长度事先不知道、不同请求长度又参差不齐，这会导致两类浪费：预留了用不完的部分（内部碎片），以及请求结束后留下大小不一、难以复用的空洞（外部碎片）。vLLM 论文统计传统系统的 KV cache 有效利用率往往只有 20%-40%。PagedAttention 借鉴操作系统虚拟内存分页的思路，把 KV cache 切成固定大小的 block（页），每个请求的 KV cache 由若干个可以不连续存放的 block 组成，用一张块表记录逻辑位置到物理 block 的映射，按需分配、用完释放，几乎消除外部碎片。

**Q2（追问）：这只是省了显存，为什么论文里说吞吐能提升 2-4 倍？省显存和吞吐提升是怎么联系起来的？**

**A2**：因为省下来的显存直接转化成了"能同时服务的并发请求数"。推理服务的吞吐量本质上取决于能同时塞进多大的 batch——如果 KV cache 管理粗放、大部分显存被碎片浪费掉，同一张卡能同时处理的请求数就很有限；PagedAttention 把显存利用率从 20%-40% 提升到接近最优后，同样的显存预算下能同时容纳多得多的并发请求，也就能开更大的 batch。而 batch 越大，GPU 在做矩阵运算（尤其是 decode 阶段那种小矩阵乘法）时的算力利用率越高，单位时间内能处理的 token 数自然就上去了。所以本质链条是：更精细的显存管理 -> 更高的显存利用率 -> 能支撑更大 batch -> 更高的 GPU 利用率 -> 更高吞吐，省显存只是第一步，真正的收益要通过"能开多大 batch"这个中间环节才能兑现。

**Q3（再追问）：这套 block 化管理除了省显存、撑更大 batch，还带来了什么额外的好处？**

**A3**：一个很重要的衍生收益是 KV cache 的**跨请求共享**变得自然了。因为 PagedAttention 把 KV cache 拆成了离散的 block、并且通过块表做间接寻址，如果两个请求的某一段 prompt 内容完全相同（比如共享同一个很长的 system prompt，或者同一份检索文档反复被不同用户的 RAG 请求引用），它们对应的 block 内容也是完全相同的，这时可以让多个请求的块表直接指向同一份物理 block，不需要真的复制一份，类似操作系统里"写时复制共享页"的思路——这就是 prefix caching（前缀缓存）能够高效实现的底层基础。如果没有这套 block 化的间接寻址机制，想做类似的 KV cache 共享会复杂得多（还是得面对"共享的内容多长、怎么对齐、复用后怎么安全释放"这些和连续内存分配天然冲突的问题）。这也是为什么现在提到 vLLM 通常会把 PagedAttention 和 prefix caching、continuous batching 放在一起讲——它们是同一套设计思想在不同层面的延伸，而不是三个孤立的技术点。

## 其它高频题速览（完整答案见 `src/ai_qa/qbank_inference_serving.py`）

- **KV cache 显存占用怎么算** —— 层数×2×hidden_size×seq_len×batch×字节数，长上下文/大并发下比参数本身更容易成为瓶颈（`ai-inf-01`）。
- **投机采样 speculative decoding** —— 小草稿模型起草多个token、大模型一次并行前向验证，接受率决定加速比，且不损失分布正确性（`ai-inf-03`）。
- **continuous batching** —— 从"整批必须一起结束"改成"逐迭代动态换入换出"，是吞吐提升的关键调度策略（`ai-inf-04`）。
- **INT8/AWQ/GPTQ 量化对比** —— 对称量化最通用；GPTQ 用二阶信息逐层补偿误差；AWQ 靠激活感知保护显著权重通道，目前工业界部署 int4 常优先选 AWQ（`ai-inf-05`）。
- **vLLM vs TensorRT-LLM 卖点** —— 一个卖易用+跨硬件灵活性，一个卖编译后针对 NVIDIA 硬件的极致稳态性能（`ai-inf-06`）。
- **prefill vs decode** —— compute-bound vs memory-bound，是理解后面几乎所有推理优化技术为何存在的基础（`ai-inf-07`）。
- **GQA/MQA** —— 用共享 KV 头压缩 KV cache，是 MHA 和 MQA 之间的效果/显存折中，Llama系列标配（`ai-inf-08`）。
- **前缀缓存 prefix caching** —— 复用重复 prompt 前缀的 KV cache，直接降低 TTFT 和重复计算成本（`ai-inf-09`）。
- **TTFT/TPOT/吞吐量的权衡** —— batch 越大吞吐越高但拉长单请求延迟，服务要在 SLA 约束下找平衡点（`ai-inf-10`）。
- **chunked prefill** —— 把长 prompt 的 prefill 切块和 decode 混合调度，避免长请求"插队"拖慢别人吐字速度（`ai-inf-11`）。
- **推理阶段的张量并行** —— 除了装下大模型，更重要的诉求是降延迟、叠加显存带宽应对 decode 阶段的 memory-bound（`ai-inf-12`）。

## 易错点 / 常见误区清单

1. **把 PagedAttention 说成"一种新的 attention 计算方式"**——它优化的是 KV cache 的**存储管理**方式，attention 的数学计算本身没有变化，只是通过间接寻址去读取分散存放的 KV。
2. **只会说投机采样"用小模型加速"，说不出为什么不损失质量**——关键是拒绝采样保证最终采出的 token 分布和只用大模型逐个采样完全等价，这是能让面试官区分"背过"和"理解"的地方。
3. **混淆 continuous batching 和 batch size 的关系**——continuous batching 是调度策略（何时换入换出请求），不是"把 batch size 调大"这么简单，它需要配合动态的 KV cache 分配机制。
4. **认为量化比特越低越好**——比特越低精度损失风险越大，AWQ/GPTQ 在 int4 已经是常见下限，再往下（int2/int3）精度损失通常难以接受，工程上要按实际评测结果决定量化到几比特。
5. **不区分 prefill 和 decode 的瓶颈类型**——很多推理优化技术（continuous batching、投机采样）主要是针对 decode 阶段的 memory-bound 问题，说不清这一点会显得对优化动机理解不深。
6. **认为 TTFT 和 TPOT 是一回事**——TTFT 是"等第一个字要多久"，TPOT 是"之后吐字快不快"，两者由不同阶段（prefill vs decode）主导，优化手段也不同。
