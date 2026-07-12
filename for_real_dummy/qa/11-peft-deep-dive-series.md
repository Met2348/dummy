# Q&A —— peft-deep-dive 系列的建立与验证(2026-07-12 ~ 2026-07-13)

## Q:alignment-algorithms-deep-dive 完成后,推进第四条、也是最后一条新系列 peft-deep-dive

延续同一套方法论,源模块是 `learning/lora-family/` + `learning/adapter-tuning-family/`——四条新系列里规模最大的一条,合计 23 种 PEFT 方法(LoRA 家族 12 种 + Adapter 家族 11 种),分 4 个文件、24 个知识点。这条系列的派发过程比前三条都更波折:首轮 4 个内容 agent 并行派发,3 个(01/02/03)先后遇到 429(5 小时限额),而且这次和之前几条系列不同——**磁盘上没有留下任何已写内容**,3 个 agent 都是在"读源码+核实细节"阶段就被打断,还没有开始写目标文件。判断限额可能会在短时间内恢复(而不是像 alignment-algorithms 03 号那次遇到的日限额那样需要长时间等待),没有立刻转向自己撰写,而是先派了一个"探路" agent 重试 01 号试探限额状态,确认能正常完成后,把 02、03 号的重试一起派发出去,最终 4 个文件全部由子代理完成撰写。

## 做了什么

建立了第 11 条深挖系列:**[peft-deep-dive/](../peft-deep-dive/00-roadmap.md)** —— 24 个知识点,4 个分类,覆盖 LoRA 核心与初始化变体(LoRA 数学与实现→rsLoRA/LoRA+→merge_and_unload→PiSSA/OLoRA→VeRA→LoHa/LoKr→AdaLoRA)→ 量化 + LoRA(NF4量化机制→QLoRA fake-quant训练循环→QLoRA真4bit路径→LoftQ→DoRA→"真4bit训练+真bitsandbytes从未同时出现"精确边界)→ Adapter 家族核心(原始bottleneck adapter→Houlsby vs Pfeiffer→AdapterFusion→Compacter→Parallel Adapter→IA3)→ Adapter 进阶与统一视角(AdapterDrop→MAM→K-Adapter/MAD-X→AdaMix→Prompt+LoRA+Adapter三线统一公式)。

## 怎么做的(首轮 3/4 撞 429、二轮全部成功的完整过程)

**首轮派发**:4 个内容 agent 并行派发(01-lora-core/02-quantized-lora/03-adapter-core/04-adapter-advanced),04 号顺利完成;01、02、03 号先后报告 `429 · api key 5小时限额已用完`,检查磁盘发现 `for_real_dummy/peft-deep-dive/` 目录下只有原来就存在的 `00-roadmap.md`,三个失败的 agent 都没能写出任何目标文件——和 alignment-algorithms 系列此前遇到的"文件写完但收尾报告被打断"是不同的失败模式,这次是真的什么都没生成。

**二轮恢复**:没有立即三个一起重新派发(避免再次同时撞上可能还没恢复的限额),先单独派了一个"探路"agent 重试 01 号,同时精简了 prompt 里的"验证纪律"部分(强调"不需要为了确认而反复重跑同一件事",减少不必要的来回)。探路 agent 成功后,直接把 02、03 号的重试一起派发,不再逐个串行等待——这次两个都顺利完成。

**04 号在首轮就顺利完成**,独立复验时用了和源文件不同的验证路径:`InvertibleAdapter` 未接入前向图这条结论,agent 自己是用"把参数改成大随机数、看输出变不变"验证的,复验时额外加了一层更直接的验证——用 `register_forward_hook` 直接统计 `InvertibleAdapter.forward` 在一次真实前向传播里被调用的次数,结果是 `0`,比"改参数看输出"更直接地证明了这个模块完全没有被接入计算图。`P_k` 梯度近零这条结论同理,agent 用的是手写隔离出来的因果自注意力,复验时改用 `MAMGPT2` 真实的语言模型 loss 做端到端反传,反而得到更极端的比值(约 4870 万倍,agent 报告的是约 1 亿倍量级,方向和数量级都一致)。

## 撰写过程中意外发现:这台机器其实有真实 GPU

02 号文件撰写前,原定计划(参照本文 prompt)是"这台机器很可能没有 GPU,如实标注未跑通"——但 agent 实际执行 `torch.cuda.is_available()` 发现结果是 `True`:一块 RTX 3080 Ti Laptop GPU(16GB,compute capability 8.6),`bitsandbytes 0.49.2` 已装,`TinyLlama-1.1B-Chat-v1.0` 本机 HF cache 已有——这和仓库当前分支名 `ERIC-3080Ti/paper-guides` 是吻合的,只是在这之前的所有系列撰写过程中都没有需要用到 GPU 的知识点,一直没有被验证到。这意味着知识点 3(QLoRA 真 4bit 路径)是**真实跑通**的,不是按最坏情况假设写的降级说明——复验时额外换了一层(`layers[5]` 而不是文档用的 `layers[0]`)、换了一个非-target 模块(`o_proj` 而不是 `k_proj`)重新验证了全部结构性事实(`peft.tuners.lora.bnb.Linear4bit` 包装 `bitsandbytes.nn.modules.Linear4bit`、`weight.dtype==torch.uint8`、`lora_A` 仍是 `float32`),结论完全一致。

## 这条系列独立发现的价值最高的几处(全部经过二次独立复验,不同参数/方法重新跑过)

- **01 号(LoRA核心)**:`merge_weights()` 的 docstring 写"删除 A、B",但实现代码没有真的删,继续误用会把 delta 双重计数(复验时换维度/种子重新验证,`diff` 从合并前后精确一致的 `~1e-6` 级别变成误用后的显著非零)。
- **02 号(量化+LoRA)**:手写 NF4 量化和真实 `bitsandbytes.functional.quantize_nf4` 在随机张量上**位级一致**(复验时换了不同形状的张量,65536 个元素差异依然精确为 0)。
- **03 号(Adapter核心)**:IA3 的手写实现和 peft 库实现总参数量都是 55,296,但直接读 `peft/tuners/ia3/layer.py` 源码确认两者缩放的是完全不同的张量(手写版本 3 个张量对应 K/V/FFN中间激活,peft 版本 4 个张量位置和维度都不同,含手写版本没有的两个 `c_proj` 缩放)——总量相等是 GPT-2 具体维度搭配出来的巧合,不是设计等价,复验时独立读了一遍 peft 源码的 `update_layer` 方法确认了这一点(`is_feedforward` 分支决定参数形状是 `(1,in_features)` 还是 `(out_features,1)`)。
- **04 号(Adapter进阶)**:MAD-X 的 `InvertibleAdapter` 被创建、被计入参数量,但整个前向链路里从未被调用——复验时用 `register_forward_hook` 直接确认调用次数为 0。

## 结论

4 个文件全部完成并逐条独立验证(每个文件都完整重新提取并执行了全部代码块,外加对每个文件里最关键的 1-3 处发现用不同参数/不同方法做了二次独立复现),合计 24 个知识点,2,397 行(含 roadmap)。这是 46 模块 runbook 验证任务完成之后启动的 4 条新系列(long-context/kernel-gpu/alignment-algorithms/peft)里规模最大、耗时最长、也是唯一一条撰写过程中真实用到本机 GPU 的一条。

加上已有的 10 条系列(numpy/python-advanced/torch/huggingface/tensorflow/python-idioms/long-context/kernel-gpu/alignment-algorithms,约 518 点)+ 1 条独立技能系列(rhcsa-bash,100 点,以及并发会话同期完成的 dsa-deep-dive,140 点+2 篇专题文章),仓库现在有 11 条深挖/精读系列 + 2 条独立技能系列,合计约 542 个知识点(不含 dsa-deep-dive)。至此,用户要求的"一口气做完"的 4 条新系列(long-context-deep-dive/kernel-gpu-deep-dive/alignment-algorithms-deep-dive/peft-deep-dive)全部完成并提交。
