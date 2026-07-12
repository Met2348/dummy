# Q&A —— kernel-gpu 深挖系列的建立与验证(2026-07-12)

## Q:long-context-deep-dive 完成后,推进第二条新系列 kernel-gpu-deep-dive

延续同一套方法论,这次的特殊之处是:源模块 `learning/gpu-architecture/` + `learning/kernel-engineering/` 是我(Claude)当天早些时候在 46 模块 runbook 验证任务里亲自逐脚本验证过的(M8 系统与Infra组),对每个函数的真实输出、已知的坑都有第一手认知,不需要额外派研究子代理调研,直接凭已有知识规划知识点分类。

## 做了什么

建立了第 8 条深挖系列:**[kernel-gpu-deep-dive/](../kernel-gpu-deep-dive/00-roadmap.md)** —— 19 个知识点,4 个分类,覆盖 GPU 硬件与存储层次(规格数据表→存储层次tier推荐→Tensor Core吞吐→SM Occupancy→NVLink拓扑带宽)→ Roofline 性能建模(arithmetic intensity→ridge point→反直觉发现→capstone zoo)→ Kernel 设计语言(Triton autotune SMEM约束+打分公式→CUTLASS layout代数→coalescing判定→swizzle_32b stub方法论)→ FlashAttention与Kernel Fusion(online softmax→HBM流量对比→kernel fusion核算→RMSNorm fusion→128k capstone曲线)。

## 怎么做的(这次遇到了 API 限额中断,处理方式和早前 kernel-engineering 模块验证时一致)

- **4 个内容 agent 并行派发,2 个撞上 5 小时限额**:01(GPU硬件)和 04(FlashAttention)先后遇到 429 限额耗尽,03 号中途遇到一次连接重置(ECONNRESET)。**检查磁盘实际状态发现 01、03 号文件其实已经完整写完**(只是子代理在收尾报告阶段被打断,没能返回结果),04 号文件则完全没生成(子代理还在"准备最终验证"阶段就被打断,没来得及创建目标文件)。处理方式:01、03 直接按已有内容独立复验;04 重新完整派发一次(这次成功,说明限额已经恢复)。这是继 kernel-engineering 模块之后第二次遇到同类中断,处理套路已经比较熟练——先看磁盘状态,能用的用,缺的补,不盲目全部重来。
- **独立复验强度不因为"内容早前验证过"而降低**:虽然 01/02/03/04 引用的很多数字我当天早些时候在 `learning/gpu-architecture/`+`learning/kernel-engineering/` 验证时已经跑过一遍,撰写这批新内容时依然要求子代理和我自己都重新独立跑一遍——事实证明这个纪律是对的:多处子代理**独立发现了我早前验证时没注意到的额外问题**,包括:
  - `memory_hierarchy.py::cost_to_load()` 有一处精确 1000 倍的单位换算错误(字节数除以 `1e9` 当成转 GB,却直接拿去除以以 TB/s 为单位的带宽),这个函数从未被 `_self_test()` 覆盖,今天首次被独立发现。
  - `tensor_core.py` 的注释写"FP4 相比 FP8 有 4 倍吞吐",实测精确算出来是 2 倍,断言用的 `>=` 阈值又恰好没暴露这个出入。
  - `triton_style.py::autotune()` 在 4096³ 大 GEMM 场景下,128×256 和 256×128 两个候选打分**位级完全相等**,最终选中谁纯粹取决于候选列表的书写顺序,不是算法"判定"谁更优——用交换 `configs` 参数顺序的方式现场验证了这一点。
  - `rmsnorm_kernel.py` 的 fused/unfused 两条路径有 `8.88e-16` 的数值差异,追根溯源发现是 Python 3.12+ 内置 `sum()` 用了 Neumaier 补偿求和算法(比手写 `acc+=` 循环精度更高),不是 fusion 本身引入的误差——把手写循环换成 `sum()` 后差异归零。
  - 所有这些发现都由我亲自重新独立验证过,不是照抄子代理报告。
- **今天已知的两个"诚实标注 stub/发现"话题直接复用,不重新调研**:`memory_hierarchy.py` 的死代码分支(`if False else` 恒真三元表达式)和 `cutlass_layout.py::swizzle_32b()` 的诚实 stub 标注,都是当天 runbook 验证时已经修复/记录过的真实历史,这次撰写时直接引用 `git show <commit sha>` 定位原始代码状态,把"怎么发现+怎么修"的完整过程讲给学生看,而不是只呈现修复后的干净版本。

## 结论

4 个文件全部完成并逐条独立验证(含对全部 4 个文件里子代理独立发现的额外 bug/现象的二次复验),合计 19 个知识点,1671 行(含 roadmap)。这条系列印证了一个和 long-context 系列相同、但换了个场景的教训:**"我今天早些时候已经验证过这份代码"不能成为降低撰写新内容时验证强度的理由**——同一份代码从"验证文档命令能不能跑通"的角度看和从"能不能扛住面试反复追问、有没有更深层的坑"的角度看,是两次独立的、都不能省略的审视,这次审视揪出的 4 处新发现(单位换算错误、注释倍数不符、打分平局靠列表顺序、数值差异根源是求和算法而非 fusion),没有一处是早前 runbook 验证时发现过的。

加上已有的 8 条系列(numpy/python-advanced/torch/huggingface/tensorflow/python-idioms/long-context,约 484 点)+ 1 条独立技能系列(rhcsa-bash,100 点),仓库现在有 9 条深挖/精读系列 + 1 条独立技能系列,合计约 503 个知识点。规划中的另外两条新系列(alignment-algorithms-deep-dive / peft-deep-dive)延续同一套方法论继续推进。
